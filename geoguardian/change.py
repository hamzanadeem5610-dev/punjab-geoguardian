"""Change detection, area statistics and hotspots (PDF sections 15 and 3).

Agricultural-to-Urban Conversion = Vegetation/Agriculture in the baseline year AND
Built-up/Urban in the recent year (post-classification comparison). Two filters
make this more reliable:

* Spectral confirmation (optional, on by default): the pixel must also show a real
  NDVI drop or NDBI rise. This removes changes that come only from small
  classification differences between the two maps.
* Minimum mapping unit: isolated patches smaller than ``min_patch_pixels`` are
  removed (8-connected components).

All areas come from pixel counts on the 10 m UTM grid (1 pixel = 100 m^2).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import ndimage

from .aoi import Grid
from .config import BUILT, CLASS_SHORT, OTHER, VEG, ProjectConfig
from .indices import Scene

NODATA = 255
EIGHT = np.ones((3, 3), bool)


@dataclass
class ChangeResult:
    conversion: np.ndarray             # bool, final agricultural-to-urban conversion
    candidate: np.ndarray              # bool, class transition before filters
    transition: pd.DataFrame           # km^2, rows = baseline class, cols = recent class
    stats: pd.DataFrame                # Metric / Value table (PDF example table)
    stats_dict: Dict
    timeseries: pd.DataFrame           # class areas per year
    hotspot_cells: pd.DataFrame
    hotspot_grid: np.ndarray           # % of each cell converted (coarse grid)
    patches: pd.DataFrame
    patch_labels: np.ndarray
    heat_grid: np.ndarray              # 0..1 urban heat-risk proxy (coarse grid)
    heat_cells: pd.DataFrame
    d_ndvi: np.ndarray
    d_ndbi: np.ndarray


def remove_small_patches(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not mask.any():
        return mask.copy()
    lab, n = ndimage.label(mask, structure=EIGHT)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


def _block_reduce(arr: np.ndarray, f: int, func=np.nanmean) -> np.ndarray:
    h, w = arr.shape
    H, W = int(np.ceil(h / f)), int(np.ceil(w / f))
    pad = np.full((H * f, W * f), np.nan, np.float32)
    pad[:h, :w] = arr
    with warnings.catch_warnings():  # all-NaN cells (outside AOI) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(pad.reshape(H, f, W, f), axis=(1, 3))


def detect_change(scene: Scene, maps: Dict[int, np.ndarray], conf: Dict[int, np.ndarray],
                  cfg: ProjectConfig) -> ChangeResult:
    g: Grid = scene.grid
    b, r = cfg.baseline_year, cfg.recent_year
    cb, cr = maps[b], maps[r]
    valid = scene.common_valid & (cb != NODATA) & (cr != NODATA)
    yb, yr = scene.years[b], scene.years[r]
    d_ndvi = yr.idx["NDVI"] - yb.idx["NDVI"]
    d_ndbi = yr.idx["NDBI"] - yb.idx["NDBI"]

    candidate = valid & (cb == VEG) & (cr == BUILT)
    conv = candidate.copy()
    if cfg.require_spectral_change:
        conv &= (d_ndvi <= -cfg.min_ndvi_drop) | (d_ndbi >= cfg.min_ndbi_rise)
    conv = remove_small_patches(conv, cfg.min_patch_pixels)

    px = g.pixel_area_km2
    aoi_px = int(scene.aoi_mask.sum())
    valid_px = int(valid.sum())

    # transition matrix
    names = [CLASS_SHORT[k] for k in (VEG, BUILT, OTHER)]
    tm = np.zeros((3, 3))
    for i, ki in enumerate((VEG, BUILT, OTHER)):
        for j, kj in enumerate((VEG, BUILT, OTHER)):
            tm[i, j] = np.count_nonzero(valid & (cb == ki) & (cr == kj)) * px
    transition = pd.DataFrame(tm.round(3), index=[f"{n} {b}" for n in names],
                              columns=[f"{n} {r}" for n in names])

    def area(mask):
        return float(np.count_nonzero(mask) * px)

    low_conf = conv & (np.fmin(conf[b], conf[r]) < 0.6)
    veg_b = area(valid & (cb == VEG))
    built_b = area(valid & (cb == BUILT))
    built_r = area(valid & (cr == BUILT))
    veg_r = area(valid & (cr == VEG))
    conv_km2 = area(conv)
    total = aoi_px * px
    sd = {
        "total_aoi_area_km2": round(total, 3),
        "analysed_area_km2": round(valid_px * px, 3),
        "nodata_area_km2": round(total - valid_px * px, 3),
        f"vegetation_area_{b}_km2": round(veg_b, 3),
        f"vegetation_area_{r}_km2": round(veg_r, 3),
        f"builtup_area_{b}_km2": round(built_b, 3),
        f"builtup_area_{r}_km2": round(built_r, 3),
        "net_builtup_growth_km2": round(built_r - built_b, 3),
        "net_vegetation_change_km2": round(veg_r - veg_b, 3),
        "conversion_candidate_km2": round(area(candidate), 3),
        "agri_to_urban_conversion_km2": round(conv_km2, 3),
        "agri_to_urban_conversion_ha": round(conv_km2 * 100, 1),
        "low_confidence_conversion_km2": round(area(low_conf), 3),
        "percent_aoi_converted": round(100 * conv_km2 / total, 2) if total else 0.0,
        f"percent_of_{b}_vegetation_converted": round(100 * conv_km2 / veg_b, 2) if veg_b else 0.0,
        "builtup_growth_percent": round(100 * (built_r - built_b) / built_b, 1) if built_b else None,
    }
    stats = pd.DataFrame([
        ("Total AOI Area", f"{sd['total_aoi_area_km2']:.2f} sq. km"),
        ("Area analysed (cloud-free both years)", f"{sd['analysed_area_km2']:.2f} sq. km"),
        (f"Vegetation / Agriculture Area in {b}", f"{veg_b:.2f} sq. km"),
        (f"Vegetation / Agriculture Area in {r}", f"{veg_r:.2f} sq. km"),
        (f"Built-up Area in {b}", f"{built_b:.2f} sq. km"),
        (f"Built-up Area in {r}", f"{built_r:.2f} sq. km"),
        ("Net Built-up Growth", f"{built_r - built_b:+.2f} sq. km"),
        ("Agriculture-to-Urban Conversion (estimated)", f"{conv_km2:.2f} sq. km "
                                                        f"({conv_km2 * 100:.0f} ha)"),
        ("Percentage of AOI Converted", f"{sd['percent_aoi_converted']:.2f} %"),
        (f"Share of {b} Vegetation Converted", f"{sd[f'percent_of_{b}_vegetation_converted']:.2f} %"),
        ("Low-confidence part of conversion", f"{sd['low_confidence_conversion_km2']:.2f} sq. km"),
    ], columns=["Metric", "Value"])

    # time series over all years
    ts_rows = []
    for y in sorted(maps):
        m = maps[y]
        vy = scene.common_valid & (m != NODATA)
        row = {"year": y}
        for k in (VEG, BUILT, OTHER):
            row[f"{CLASS_SHORT[k]}_km2"] = round(area(vy & (m == k)), 3)
        row["Built-up_percent"] = round(100 * row["Built-up_km2"] / max(area(vy), 1e-9), 2)
        ts_rows.append(row)
    timeseries = pd.DataFrame(ts_rows)

    # hotspots: coarse grid cells
    f = max(1, int(round(cfg.hotspot_cell_km * 1000 / g.res)))
    conv_f = np.where(valid, conv.astype(np.float32), np.nan)
    frac = _block_reduce(conv_f, f)  # fraction of valid pixels converted
    cnt = _block_reduce(np.where(valid, conv, 0).astype(np.float32), f, np.nansum)
    cells = []
    for (i, j), v in np.ndenumerate(cnt):
        if v > 0:
            rr, cc = (i + 0.5) * f, (j + 0.5) * f
            lon, lat = g.rowcol_to_lonlat(min(rr, g.height - 1), min(cc, g.width - 1))
            cells.append({"cell_row": i, "cell_col": j, "converted_km2": round(v * px, 4),
                          "converted_pct_of_cell": round(100 * float(frac[i, j]), 2),
                          "center_lon": round(float(lon), 5), "center_lat": round(float(lat), 5)})
    hot = pd.DataFrame(cells, columns=["cell_row", "cell_col", "converted_km2",
                                       "converted_pct_of_cell", "center_lon", "center_lat"])
    hot = hot.sort_values("converted_km2", ascending=False).reset_index(drop=True)
    hot.insert(0, "rank", np.arange(1, len(hot) + 1))

    # contiguous patches
    lab, n = ndimage.label(conv, structure=EIGHT)
    patches = pd.DataFrame(columns=["patch_id", "area_ha", "centroid_lon", "centroid_lat",
                                    "mean_confidence"])
    if n:
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum(np.ones_like(lab), lab, idx)
        com = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        cmean = ndimage.mean(np.nan_to_num(np.fmin(conf[b], conf[r])), lab, idx)
        rr = np.array([c[0] for c in com])
        cc = np.array([c[1] for c in com])
        lon, lat = g.rowcol_to_lonlat(rr, cc)
        patches = pd.DataFrame({"patch_id": idx, "area_ha": np.round(sizes * g.pixel_area_ha, 2),
                                "centroid_lon": np.round(lon, 5), "centroid_lat": np.round(lat, 5),
                                "mean_confidence": np.round(cmean, 3)})
        patches = patches.sort_values("area_ha", ascending=False).reset_index(drop=True)

    # urban heat-risk proxy: vegetation loss + built-up increase (not temperature!)
    fh = max(1, int(round(cfg.heat_cell_km * 1000 / g.res)))
    dv = _block_reduce(np.where(valid, d_ndvi, np.nan), fh)
    db = _block_reduce(np.where(valid, d_ndbi, np.nan), fh)
    built_share = _block_reduce(np.where(valid, (cr == BUILT).astype(np.float32), np.nan), fh)

    # fixed physical scaling (not per-scene percentiles) so scores are comparable across AOIs:
    # an NDVI loss of 0.4 or an NDBI gain of 0.25 averaged over a cell counts as maximal.
    veg_loss = np.clip(-dv / 0.4, 0, 1)
    built_gain = np.clip(db / 0.25, 0, 1)
    heat = (0.4 * veg_loss + 0.3 * built_gain
            + 0.3 * np.nan_to_num(built_share, nan=0)).astype(np.float32)
    heat[np.isnan(dv)] = np.nan
    heat_cells = pd.DataFrame({
        "level": pd.cut(heat[~np.isnan(heat)], [-0.01, 0.33, 0.66, 1.01],
                        labels=["Low", "Moderate", "High"]).astype(str)})
    heat_cells = heat_cells.value_counts().rename("cells").reset_index()
    heat_cells["area_km2"] = heat_cells["cells"] * (fh * g.res / 1000) ** 2

    print(f"[change] conversion {conv_km2:.2f} km^2 ({sd['percent_aoi_converted']:.2f}% of AOI); "
          f"candidates before filters {sd['conversion_candidate_km2']:.2f} km^2")
    return ChangeResult(conversion=conv, candidate=candidate, transition=transition, stats=stats,
                        stats_dict=sd, timeseries=timeseries, hotspot_cells=hot,
                        hotspot_grid=(frac * 100).astype(np.float32), patches=patches,
                        patch_labels=lab, heat_grid=heat, heat_cells=heat_cells,
                        d_ndvi=d_ndvi, d_ndbi=d_ndbi)


def change_class_map(maps: Dict[int, np.ndarray], conv: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    """Categorical change map: 0 stable veg, 1 stable built, 2 other/other changes,
    3 agri-to-urban conversion, 255 no data."""
    cb, cr = maps[cfg.baseline_year], maps[cfg.recent_year]
    out = np.full(cb.shape, 2, np.uint8)
    out[(cb == VEG) & (cr == VEG)] = 0
    out[(cb == BUILT) & (cr == BUILT)] = 1
    out[conv] = 3
    out[(cb == NODATA) | (cr == NODATA)] = NODATA
    return out


CHANGE_MAP_LABELS = {0: "Stable vegetation", 1: "Stable built-up", 2: "Other / other change",
                     3: "Agri-to-urban conversion"}
CHANGE_MAP_COLORS = {0: "#B7DDB0", 1: "#9E9E9E", 2: "#F3E9D2", 3: "#FF0000"}


def describe_location(lon: float, lat: float, grid: Grid) -> str:
    """Plain-language position inside the AOI (e.g. 'north-east part, 3.2 km from centre')."""
    clon, clat = grid.center_lonlat()
    x, y = grid.from_lonlat(lon, lat)
    cx, cy = grid.from_lonlat(clon, clat)
    dx, dy = float(x - cx), float(y - cy)
    dist = (dx ** 2 + dy ** 2) ** 0.5 / 1000
    if dist < 1.5:
        return f"central part of the AOI ({dist:.1f} km from centre)"
    ang = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dirs = ["northern", "north-eastern", "eastern", "south-eastern", "southern", "south-western",
            "western", "north-western"]
    d = dirs[int(((ang + 22.5) % 360) // 45)]
    return f"{d} part of the AOI ({dist:.1f} km from centre)"


def hotspot_table(change: ChangeResult, grid: Grid, top: int = 10) -> pd.DataFrame:
    h = change.hotspot_cells.head(top).copy()
    if h.empty:
        return h
    h["location"] = [describe_location(a, b, grid) for a, b in zip(h.center_lon, h.center_lat)]
    h["google_maps"] = [f"https://www.google.com/maps/@{b:.5f},{a:.5f},1500m/data=!3m1!1e3"
                        for a, b in zip(h.center_lon, h.center_lat)]
    return h


def summarize_patches(change: ChangeResult) -> List[str]:
    p = change.patches
    if p.empty:
        return ["No conversion patches above the minimum mapping unit were detected."]
    big = p[p.area_ha >= 1]
    return [f"{len(p)} separate conversion patches were detected; {len(big)} are 1 ha or larger.",
            f"The largest contiguous patch covers about {p.area_ha.iloc[0]:.1f} ha.",
            f"The 10 largest patches together cover {p.area_ha.head(10).sum():.1f} ha "
            f"({100 * p.area_ha.head(10).sum() / max(p.area_ha.sum(), 1e-9):.0f}% of all "
            f"detected conversion)."]
