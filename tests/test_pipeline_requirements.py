"""Acceptance test: run the whole pipeline on a synthetic scene and check every
numeric minimum from the internship PDF (section 6 and 21)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoguardian.config import BUILT, CLASS_SHORT, VEG, ProjectConfig  # noqa: E402
from geoguardian.pipeline import GeoGuardian  # noqa: E402
from geoguardian.synthetic import make_synthetic_stack  # noqa: E402


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = ProjectConfig(aoi_key="test", aoi_name="Synthetic test scene", synthetic=True,
                        data_dir=str(tmp / "data"), output_dir=str(tmp / "outputs"))
    _, truth = make_synthetic_stack(cfg, size=400)
    p = GeoGuardian(cfg)
    p.load().build_training().classify().detect_change().validation_sheet()
    # simulate the manual visual check using the synthetic ground truth
    csv = p.p.validation / "validation_points.csv"
    df = pd.read_csv(csv)
    b, r = cfg.baseline_year, cfg.recent_year
    df["actual_class"] = [CLASS_SHORT[int(truth[r][a, c])] for a, c in zip(df.row, df.col)]
    df["actual_change"] = ["yes" if truth[b][a, c] == VEG and truth[r][a, c] == BUILT else "no"
                           for a, c in zip(df.row, df.col)]
    df["notes"] = "simulated check"
    df.to_csv(csv, index=False)
    p.run_all()  # second full run picks up the labels
    return p


def test_training_samples_min_50_per_class(project):
    counts = project.samples[project.samples.keep].groupby("class_id").size()
    assert len(counts) == 3 and (counts >= 50).all()
    assert project.samples.block.nunique() >= 10  # spread over the AOI, not one location


def test_training_and_validation_separate(project):
    tr = set(zip(project.samples.row, project.samples.col))
    va = set(zip(project.val_sheet.row, project.val_sheet.col))
    assert not tr & va
    d = np.hypot(project.val_sheet.row.values[:, None] - project.samples.row.values[None],
                 project.val_sheet.col.values[:, None] - project.samples.col.values[None])
    assert d.min() * project.scene.grid.res >= project.cfg.validation_min_distance_to_training_m - 10


def test_validation_and_error_analysis_minimums(project):
    v = project.validation
    assert v["n_points"] >= 50 and not v["provisional"]
    assert set(v["table"].columns) >= {"Point ID", "Actual Class", "Predicted Class",
                                       "Correct / Incorrect", "Notes"}
    assert 0 <= v["overall_accuracy"] <= 1
    assert len(project.errors["table"]) >= 15
    assert len(project.errors["improvements"]) >= 3


def test_land_classes_and_change(project):
    for m in project.maps.values():
        assert set(np.unique(m)) <= {0, 1, 2, 255}
    sd = project.change.stats_dict
    for k in ("total_aoi_area_km2", "vegetation_area_2019_km2", "builtup_area_2025_km2",
              "agri_to_urban_conversion_km2", "percent_aoi_converted"):
        assert k in sd
    assert sd["agri_to_urban_conversion_km2"] > 0
    assert sd["total_aoi_area_km2"] == pytest.approx(16.0)  # 400 x 400 px x 100 m²


def test_output_files(project):
    out = project.cfg.out
    maps = list((out / "maps").glob("*.png"))
    assert len(maps) >= 8
    for name in ("02_rgb_2019", "03_rgb_2025", "04_ndvi_2019", "05_ndvi_2025", "06_ndbi_2019",
                 "07_ndbi_2025", "08_landcover_2019", "09_landcover_2025",
                 "10_change_detection_2019_2025"):
        assert (out / "maps" / f"{name}.png").exists(), name
    for f in ("web/interactive_map.html", "report/LINKEDIN_PORTFOLIO.md",
              "report/WEEK1_PROGRESS_SUMMARY.md", "tables/area_statistics.csv",
              "tables/validation_table.csv", "tables/error_analysis.csv",
              "vectors/conversion_polygons_2019_2025.geojson", "geotiff/change_2019_2025.tif",
              "validation/validation_chips.pdf", "run_summary.json"):
        assert (out / f).exists(), f
    pdf = next((out / "report").glob("*.pdf"))
    assert "Hamza_Nadeem" in pdf.name and pdf.stat().st_size > 100_000
    html = (out / "web" / "interactive_map.html").read_text()
    for layer in ("Sentinel-2 RGB 2019", "Sentinel-2 RGB 2025", "Land cover 2019",
                  "Land cover 2025", "Detected change"):
        assert layer in html
