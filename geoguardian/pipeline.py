"""End-to-end orchestration: stack GeoTIFF -> every deliverable in ``outputs/``.

    project = GeoGuardian(cfg)
    project.run_all()        # idempotent; re-run after you fill the validation labels
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import viz
from .change import (CHANGE_MAP_COLORS, CHANGE_MAP_LABELS, change_class_map, detect_change,
                     hotspot_table)
from .classify import majority_filter, predict_year, rule_map, train_and_select
from .config import BUILT, CLASS_COLORS, CLASS_NAMES, CLASS_SHORT, OTHER, VEG, ProjectConfig
from .error_analysis import run_error_analysis
from .export import conversion_geojson, hotspot_cells_geojson, points_geojson, write_tables
from .indices import load_scene
from .io import write_class_raster, write_json, write_stack
from .sampling import generate_training_samples, generate_validation_points, training_table, \
    validation_strata
from .validation import assess_accuracy, build_validation_sheet, chip_sheet, load_labels


@dataclass
class Paths:
    root: Path

    def __post_init__(self):
        for d in ("maps", "charts", "geotiff", "vectors", "tables", "validation", "web",
                  "report", "models"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, item):
        return self.root / item


class GeoGuardian:
    def __init__(self, cfg: ProjectConfig):
        cfg.validate()
        self.cfg = cfg
        self.p = Paths(cfg.out)
        self.scene = None
        self.samples = None
        self.train = None
        self.model = None
        self.maps: Dict[int, np.ndarray] = {}
        self.maps_raw: Dict[int, np.ndarray] = {}
        self.conf: Dict[int, np.ndarray] = {}
        self.rule_maps: Dict[int, np.ndarray] = {}
        self.change = None
        self.strata = None
        self.val_sheet = None
        self.validation = None
        self.rule_validation = None
        self.errors = None
        self.figures: Dict[str, Path] = {}

    # ------------------------------------------------------------ steps
    def load(self):
        self.scene = load_scene(self.cfg.stack_path, self.cfg.stack_meta_path, self.cfg.years)
        g = self.scene.grid
        for y, yd in self.scene.years.items():
            print(f"[load] {y}: {100 * yd.valid.mean():.1f}% of grid has clear data "
                  f"(median {np.median(yd.obs[yd.valid]) if yd.valid.any() else 0:.0f} obs/pixel)")
        print(f"[load] grid {g.width}x{g.height} @ {g.res} m, AOI "
              f"{self.scene.aoi_mask.sum() * g.pixel_area_km2:.2f} km²")
        return self

    def build_training(self):
        self.samples = generate_training_samples(self.scene, self.cfg)
        review = self.p.validation / "training_samples_review.csv"
        if review.exists():  # analyst may have set keep=False for bad samples
            old = pd.read_csv(review)
            if {"lon", "lat", "keep"} <= set(old.columns):
                bad = set(zip(old.loc[~old["keep"].astype(bool), "lon"].round(6),
                              old.loc[~old["keep"].astype(bool), "lat"].round(6)))
                if bad:
                    m = [(a, b) in bad for a, b in zip(self.samples.lon.round(6),
                                                       self.samples.lat.round(6))]
                    self.samples.loc[m, "keep"] = False
                    print(f"[samples] {sum(m)} samples excluded after visual review")
        else:
            self.samples.to_csv(review, index=False)
        self.train = training_table(self.scene, self.samples)
        self.samples.to_csv(self.p.tables / "training_samples.csv", index=False)
        return self

    def classify(self):
        self.model = train_and_select(self.train, self.cfg)
        for y, yd in self.scene.years.items():
            raw, conf = predict_year(self.model, yd)
            self.maps_raw[y] = raw
            self.maps[y] = majority_filter(raw) if self.cfg.majority_filter else raw
            self.conf[y] = conf
            self.rule_maps[y] = rule_map(yd, self.model.rule)
        import joblib
        joblib.dump(self.model.model, self.p.models / f"{self.model.name}.joblib")
        write_json(self.p.models / "model_metrics.json",
                   {"selected_model": self.model.name, "cv": self.model.all_cv,
                    "rule_thresholds": self.model.rule,
                    "feature_importance": self.model.importances.round(4).to_dict(),
                    "n_training_rows": len(self.train),
                    "training_rows_per_class": self.train.groupby("class_id").size()
                    .rename(index=CLASS_SHORT).to_dict()})
        return self

    def detect_change(self):
        self.change = detect_change(self.scene, self.maps, self.conf, self.cfg)
        return self

    def _fingerprint(self) -> str:
        keys = ("aoi_key", "aoi_mode", "center_lat", "center_lon", "width_km", "height_km",
                "district_name", "baseline_year", "recent_year", "season_start", "season_end",
                "max_cloud_pct", "random_seed", "samples_per_class_per_year",
                "require_spectral_change", "min_patch_pixels", "majority_filter")
        s = json.dumps({k: getattr(self.cfg, k) for k in keys}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def validation_sheet(self, force_new: bool = False):
        r = self.cfg.recent_year
        self.strata = validation_strata(self.maps[r], self.change.conversion)
        csv = self.p.validation / "validation_points.csv"
        fp_file = self.p.validation / "sampling_fingerprint.json"
        if csv.exists() and not force_new:
            pts = pd.read_csv(csv)[["point_id", "stratum", "stratum_name", "row", "col", "lon", "lat"]]
            old_fp = json.loads(fp_file.read_text())["fingerprint"] if fp_file.exists() else None
            if old_fp and old_fp != self._fingerprint():
                print("[validation] WARNING: settings changed since the points were drawn. "
                      "Labels are kept, but for a clean design delete validation_points.csv "
                      "and label the new points.")
        else:
            pts = generate_validation_points(self.scene, self.cfg, self.strata, self.samples)
            fp_file.write_text(json.dumps({"fingerprint": self._fingerprint()}))
        self.val_sheet = build_validation_sheet(self.scene, pts, self.maps, self.conf,
                                                self.change.conversion, self.cfg, csv)
        chips_pdf = self.p.validation / "validation_chips.pdf"
        if force_new or not chips_pdf.exists():
            chip_sheet(self.scene, self.val_sheet, self.cfg, chips_pdf,
                       "Validation points - label each point WITHOUT looking at the prediction",
                       chip_dir=self.p.validation / "chips")
            rev = self.samples.groupby("class_id", group_keys=False).apply(
                lambda d: d.sample(min(len(d), 24), random_state=self.cfg.random_seed))
            chip_sheet(self.scene, rev, self.cfg, self.p.validation / "training_samples_review.pdf",
                       "Training-sample spot check (24 per class)", id_col="sample_id",
                       label_col="class_name")
        return self

    def assess(self):
        labels = load_labels(self.p.validation / "validation_points.csv", self.cfg)
        self.validation = assess_accuracy(labels, self.strata, self.scene, self.cfg)
        # Option A (rule-based map) on the same points, for comparison
        r = self.cfg.recent_year
        rl = labels.copy()
        rl["predicted_class"] = [CLASS_SHORT.get(int(v), "NoData")
                                 for v in self.rule_maps[r][rl.row.values, rl.col.values]]
        rule_conv = (self.rule_maps[self.cfg.baseline_year] == VEG) & (self.rule_maps[r] == BUILT)
        rl["predicted_change"] = np.where(rule_conv[rl.row.values, rl.col.values], "yes", "no")
        self.rule_validation = assess_accuracy(rl, self.strata, self.scene, self.cfg)
        self.errors = run_error_analysis(self.validation, self.val_sheet, self.cfg)
        return self

    # ----------------------------------------------------------- outputs
    def _src(self) -> str:
        return ("Source: SYNTHETIC test scene (no real satellite data). " if self.cfg.synthetic else
                "Source: Copernicus Sentinel-2 L2A (ESA), processed in Google Earth Engine. ") + \
            f"Author: {self.cfg.author_name}. Estimated change - learning prototype."

    def preprocessing_maps(self):
        """Week-1 maps: AOI, RGB, NDVI and NDBI for both years (no classification needed)."""
        cfg, sc, g = self.cfg, self.scene, self.scene.grid
        b, r = cfg.baseline_year, cfg.recent_year
        syn, src, M, f = cfg.synthetic, self._src(), self.p.maps, self.figures
        lims = viz.shared_stretch(sc, cfg.years)
        self.rgb = rgb = {y: viz.rgb_image(sc.years[y], lims) for y in cfg.years}
        f["aoi"] = viz.location_map(g, sc.meta.get("context", {}), sc.meta["aoi_geojson"],
                                    rgb[r], f"Area of Interest: {cfg.aoi_name}", M / "01_aoi_location.png", syn)
        f["rgb_b"] = viz.map_rgb(rgb[b], g, f"Sentinel-2 true colour (RGB) - {b}", M / f"02_rgb_{b}.png", syn, src)
        f["rgb_r"] = viz.map_rgb(rgb[r], g, f"Sentinel-2 true colour (RGB) - {r}", M / f"03_rgb_{r}.png", syn, src)
        f["ndvi_b"] = viz.map_continuous(sc.years[b].idx["NDVI"], g, f"NDVI - {b}", M / f"04_ndvi_{b}.png",
                                         viz.NDVI_CMAP, viz.NDVI_RANGE, "NDVI = (NIR - Red) / (NIR + Red)", syn, src)
        f["ndvi_r"] = viz.map_continuous(sc.years[r].idx["NDVI"], g, f"NDVI - {r}", M / f"05_ndvi_{r}.png",
                                         viz.NDVI_CMAP, viz.NDVI_RANGE, "NDVI = (NIR - Red) / (NIR + Red)", syn, src)
        f["ndbi_b"] = viz.map_continuous(sc.years[b].idx["NDBI"], g, f"NDBI - {b}", M / f"06_ndbi_{b}.png",
                                         viz.NDBI_CMAP, viz.NDBI_RANGE, "NDBI = (SWIR - NIR) / (SWIR + NIR)", syn, src)
        f["ndbi_r"] = viz.map_continuous(sc.years[r].idx["NDBI"], g, f"NDBI - {r}", M / f"07_ndbi_{r}.png",
                                         viz.NDBI_CMAP, viz.NDBI_RANGE, "NDBI = (SWIR - NIR) / (SWIR + NIR)", syn, src)
        f["rgb_pair"] = viz.side_by_side([rgb[b], rgb[r]], [f"{b}", f"{r}"], g, M / "16_rgb_before_after.png",
                                         "Before / after - Sentinel-2 true colour", syn)
        return self

    def make_maps(self):
        if not hasattr(self, "rgb"):
            self.preprocessing_maps()
        cfg, sc, g = self.cfg, self.scene, self.scene.grid
        b, r = cfg.baseline_year, cfg.recent_year
        syn, src, rgb = cfg.synthetic, self._src(), self.rgb
        M, C = self.p.maps, self.p.charts
        f = self.figures
        px = g.pixel_area_km2
        area = {y: {k: float(np.count_nonzero(self.maps[y] == k) * px) for k in (VEG, BUILT, OTHER)}
                for y in cfg.years}
        names = {k: CLASS_NAMES[k] for k in (VEG, BUILT, OTHER)}
        f["class_b"] = viz.map_classes(self.maps[b], g, f"Land-cover classification - {b}",
                                       M / f"08_landcover_{b}.png", CLASS_COLORS, names, syn, src, area[b])
        f["class_r"] = viz.map_classes(self.maps[r], g, f"Land-cover classification - {r}",
                                       M / f"09_landcover_{r}.png", CLASS_COLORS, names, syn, src, area[r])
        hot = hotspot_table(self.change, g, 10)
        f["change"] = viz.map_change_overlay(rgb[r], self.change.conversion, g,
                                             f"Detected agricultural-to-urban conversion {b}-{r}",
                                             M / f"10_change_detection_{b}_{r}.png", syn, src, hot)
        cmap_ = change_class_map(self.maps, self.change.conversion, cfg)
        self.change_map = cmap_
        f["change_classes"] = viz.map_classes(cmap_, g, f"Change classes {b}-{r}",
                                              M / f"11_change_classes_{b}_{r}.png", CHANGE_MAP_COLORS,
                                              CHANGE_MAP_LABELS, syn, src)
        f["hotspots"] = viz.map_coarse(np.where(self.change.hotspot_grid > 0, self.change.hotspot_grid, np.nan),
                                       g, f"Conversion hotspots - % of each {cfg.hotspot_cell_km:g} km cell",
                                       M / "12_hotspot_density.png", "YlOrRd", "% of cell converted",
                                       0, max(5, float(np.nanmax(self.change.hotspot_grid))), syn, src)
        f["dndvi"] = viz.map_continuous(self.change.d_ndvi, g, f"NDVI change ({r} minus {b})",
                                        M / "13_ndvi_change.png", "BrBG", (-0.5, 0.5), "ΔNDVI", syn, src)
        f["dndbi"] = viz.map_continuous(self.change.d_ndbi, g, f"NDBI change ({r} minus {b})",
                                        M / "14_ndbi_change.png", "PuOr_r", (-0.3, 0.3), "ΔNDBI", syn, src)
        f["heat"] = viz.map_coarse(self.change.heat_grid, g, "Urban heat-risk proxy (vegetation loss + built-up gain)",
                                   M / "15_heat_risk_proxy.png", "inferno_r", "Relative risk score (0-1)", 0, 1, syn, src)
        f["class_pair"] = viz.side_by_side([viz.class_rgba(self.maps[b], CLASS_COLORS),
                                            viz.class_rgba(self.maps[r], CLASS_COLORS)],
                                           [f"{b}", f"{r}"], g, M / "17_landcover_before_after.png",
                                           "Before / after - land cover", syn,
                                           legend=(CLASS_COLORS, CLASS_SHORT))
        f["rule_r"] = viz.map_classes(self.rule_maps[r], g, f"Rule-based (Option A) land cover - {r}",
                                      M / f"18_rule_based_landcover_{r}.png", CLASS_COLORS, names, syn, src)
        for y in cfg.extra_years:
            f[f"class_{y}"] = viz.map_classes(self.maps[y], g, f"Land-cover classification - {y}",
                                              M / f"19_landcover_{y}.png", CLASS_COLORS, names, syn, src, area[y])
            f[f"rgb_{y}"] = viz.map_rgb(rgb[y], g, f"Sentinel-2 true colour (RGB) - {y}",
                                        M / f"22_rgb_{y}.png", syn, src)
            f[f"ndvi_{y}"] = viz.map_continuous(sc.years[y].idx["NDVI"], g, f"NDVI - {y}", M / f"23_ndvi_{y}.png",
                                                viz.NDVI_CMAP, viz.NDVI_RANGE, "NDVI", syn, src)
            f[f"ndbi_{y}"] = viz.map_continuous(sc.years[y].idx["NDBI"], g, f"NDBI - {y}", M / f"24_ndbi_{y}.png",
                                                viz.NDBI_CMAP, viz.NDBI_RANGE, "NDBI", syn, src)
        # year-keyed aliases used by the dashboard's year selector
        for y, key in ((b, "b"), (r, "r")):
            for layer in ("rgb", "ndvi", "ndbi", "class"):
                f[f"{layer}_{y}"] = f[f"{layer}_{key}"]
        # charts
        f["areas"] = viz.chart_area_bars(self.change.timeseries, C / "area_by_year.png", syn)
        if len(cfg.years) > 2:
            f["trend"] = viz.chart_timeseries(self.change.timeseries, C / "landcover_trend.png")
        f["transition"] = viz.chart_matrix(self.change.transition, C / "transition_matrix.png",
                                           f"Land-cover transitions {b} -> {r} (km²)", "{:.2f}", "Oranges",
                                           f"Class in {r}", f"Class in {b}")
        f["importance"] = viz.chart_importance(self.model.importances, C / "feature_importance.png",
                                               self.model.name)
        f["models"] = viz.chart_model_comparison(self.model.all_cv, C / "model_comparison.png")
        cv_cm = pd.DataFrame(self.model.cv["confusion_matrix"], index=[CLASS_SHORT[k] for k in (0, 1, 2)],
                             columns=[CLASS_SHORT[k] for k in (0, 1, 2)])
        f["cv_cm"] = viz.chart_matrix(cv_cm, C / "cv_confusion_matrix.png",
                                      f"Spatial-CV confusion matrix ({self.model.name})", "{:.0f}",
                                      "Blues", "Predicted", "Reference sample")
        f["samples"] = viz.map_points(rgb[r], g, self.samples[self.samples.keep], M / "20_training_samples.png",
                                      "Training sample locations", "class_name",
                                      {CLASS_SHORT[k]: CLASS_COLORS[k] for k in (0, 1, 2)}, syn, label_ids=False)
        if self.validation:
            cm = self.validation["confusion_matrix"]
            f["val_cm"] = viz.chart_matrix(cm, C / "validation_confusion_matrix.png",
                                           f"Validation confusion matrix ({r})", "{:.0f}", "Greens",
                                           "Predicted", "Actual")
            t = self.validation["table"].rename(columns={"Point ID": "point_id", "Latitude": "lat",
                                                         "Longitude": "lon"})
            f["val_pts"] = viz.map_points(rgb[r], g, t, M / "21_validation_points.png",
                                          f"Validation points ({len(t)}) - correct vs incorrect",
                                          "Correct / Incorrect", {"Correct": "#1a9850", "Incorrect": "#d73027"}, syn)
        print(f"[maps] {len([k for k in f])} figures written to {self.p.maps} and {self.p.charts}")
        return self

    def make_web_map(self):
        from .webmap import build_web_map
        cfg, sc = self.cfg, self.scene
        layers = {}
        for y in cfg.years:
            layers[f"rgb_{y}"] = self.rgb[y]
            layers[f"valid_{y}"] = sc.years[y].valid
            layers[f"class_{y}"] = self.maps[y]
            layers[f"ndvi_{y}"] = sc.years[y].idx["NDVI"]
            layers[f"ndbi_{y}"] = sc.years[y].idx["NDBI"]
        layers["change"] = self.change.conversion
        pts = None
        if self.validation:
            t = self.validation["table"]
            pts = pd.DataFrame({"point_id": t["Point ID"], "lat": t["Latitude"], "lon": t["Longitude"],
                                "predicted_class": t["Predicted Class"], "actual": t["Actual Class"],
                                "status": t["Correct / Incorrect"]})
        self.web_map = build_web_map(self.p.web / "interactive_map.html", sc.grid, cfg, layers,
                                     sc.meta["aoi_geojson"], sc.meta.get("context"),
                                     hotspot_table(self.change, sc.grid, 10), pts)
        return self

    def export(self):
        cfg, sc, g = self.cfg, self.scene, self.scene.grid
        b, r = cfg.baseline_year, cfg.recent_year
        G, V = self.p.geotiff, self.p.vectors
        for y in cfg.years:
            write_class_raster(G / f"landcover_{y}.tif", self.maps[y], g, CLASS_COLORS, f"landcover_{y}")
            write_stack(G / f"indices_{y}.tif", np.stack([sc.years[y].idx["NDVI"], sc.years[y].idx["NDBI"],
                                                           self.conf[y]]).astype(np.float32),
                        g, ["NDVI", "NDBI", "model_confidence"], nodata=np.nan)
        write_class_raster(G / f"change_{b}_{r}.tif", self.change_map, g, CHANGE_MAP_COLORS, f"change_{b}_{r}")
        n_poly = conversion_geojson(self.change.conversion, g, V / f"conversion_polygons_{b}_{r}.geojson", (b, r))
        hotspot_cells_geojson(self.change.hotspot_cells, g, cfg.hotspot_cell_km, V / "hotspot_cells.geojson")
        points_geojson(self.samples, V / "training_samples.geojson")
        (V / "aoi.geojson").write_text(json.dumps(sc.meta["aoi_geojson"]))
        if self.val_sheet is not None:
            cols = [c for c in self.val_sheet.columns if c not in ("row", "col")]
            points_geojson(self.val_sheet[cols], V / "validation_points.geojson")
        tables = {
            "area_statistics": self.change.stats,
            "transition_matrix_km2": self.change.transition,
            "landcover_area_by_year": self.change.timeseries,
            "hotspot_cells": hotspot_table(self.change, g, len(self.change.hotspot_cells)),
            "conversion_patches": self.change.patches,
            "model_comparison_cv": pd.DataFrame({k: {m: v for m, v in d.items() if m != "confusion_matrix"}
                                                 for k, d in self.model.all_cv.items()}).T,
            "heat_risk_summary": self.change.heat_cells,
        }
        if self.validation:
            tables["validation_table"] = self.validation["table"]
            tables["validation_per_class"] = self.validation["per_class"]
            tables["validation_confusion_matrix"] = self.validation["confusion_matrix"]
            tables["change_validation"] = pd.DataFrame([self.validation["change"]])
        if self.errors:
            tables["error_analysis"] = self.errors["table"]
        write_tables(tables, self.p.tables)
        write_json(self.p.tables / "area_statistics.json", self.change.stats_dict)
        print(f"[export] GeoTIFFs, {n_poly} conversion polygons, CSV tables written")
        return self

    def week1_summary(self):
        from .report import week1_markdown
        (self.p.report / "WEEK1_PROGRESS_SUMMARY.md").write_text(week1_markdown(self), encoding="utf-8")
        return self

    def report(self):
        from .report import build_report
        from .summary import linkedin_markdown, observations
        (self.p.report / "LINKEDIN_PORTFOLIO.md").write_text(linkedin_markdown(self), encoding="utf-8")
        pdf = build_report(self)
        summary = {
            "aoi": self.cfg.aoi_name, "years": [self.cfg.baseline_year, self.cfg.recent_year],
            "author": self.cfg.author_name, "synthetic": self.cfg.synthetic,
            "image_info": self.scene.meta.get("image_info"), "stats": self.change.stats_dict,
            "model": self.model.name, "cv": {k: {m: v for m, v in d.items() if m != "confusion_matrix"}
                                            for k, d in self.model.all_cv.items()},
            "validation": None if not self.validation else {
                k: v for k, v in self.validation.items()
                if k in ("n_points", "n_manual", "provisional", "correct", "incorrect",
                         "overall_accuracy", "area_weighted_overall_accuracy", "kappa", "change")},
            "observations": observations(self),
            "figures": {k: str(v.relative_to(self.cfg.out)) for k, v in self.figures.items()},
            "report_pdf": str(pdf.relative_to(self.cfg.out)),
        }
        write_json(self.cfg.out / "run_summary.json", summary)
        return self

    # ------------------------------------------------------------- all
    def run_all(self, force_new_validation: bool = False):
        (self.load().build_training().classify().detect_change()
         .validation_sheet(force_new_validation).assess().make_maps().week1_summary()
         .make_web_map().export().report())
        print(f"\nDone. Outputs in: {self.cfg.out.resolve()}")
        return self

    def zip_outputs(self, dest: Optional[Path] = None) -> Path:
        dest = Path(dest or self.cfg.out.parent / f"punjab_geoguardian_outputs_{self.cfg.aoi_key}")
        return Path(shutil.make_archive(str(dest), "zip", root_dir=self.cfg.out))
