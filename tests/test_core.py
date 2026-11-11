"""Unit tests for the analysis building blocks (no Earth Engine needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoguardian.aoi import grid_from_center, utm_epsg  # noqa: E402
from geoguardian.change import remove_small_patches  # noqa: E402
from geoguardian.classify import RuleThresholds, majority_filter, rule_classify  # noqa: E402
from geoguardian.config import BUILT, OTHER, VEG, ProjectConfig  # noqa: E402
from geoguardian.indices import ndbi, ndvi  # noqa: E402
from geoguardian.validation import _olofsson, parse_change, parse_class  # noqa: E402


def test_ndvi_ndbi_formulas():
    nir, red, swir = np.array([0.4]), np.array([0.1]), np.array([0.2])
    assert ndvi(nir, red)[0] == pytest.approx((0.4 - 0.1) / (0.4 + 0.1))
    assert ndbi(swir, nir)[0] == pytest.approx((0.2 - 0.4) / (0.2 + 0.4))
    assert np.isnan(ndvi(np.array([0.0]), np.array([0.0]))[0])  # 0/0 -> NaN, not inf


def test_grid_is_exact_and_utm():
    g = grid_from_center(31.40, 74.25, 15, 15)
    assert utm_epsg(74.25, 31.40) == 32643 and g.crs == "EPSG:32643"
    assert (g.width, g.height) == (1500, 1500)
    assert g.width * g.height * g.pixel_area_km2 == pytest.approx(225.0)
    lon, lat = g.center_lonlat()
    assert lon == pytest.approx(74.25, abs=1e-3) and lat == pytest.approx(31.40, abs=1e-3)
    r, c = g.lonlat_to_rowcol(*g.rowcol_to_lonlat(123, 456))
    assert (int(r), int(c)) == (123, 456)


def test_rule_classifier():
    th = RuleThresholds(0.35, -0.05, 0.1)
    out = rule_classify(np.array([0.7, 0.1, 0.05, -0.2]), np.array([-0.3, 0.1, -0.2, -0.5]),
                        np.array([-0.5, -0.2, -0.3, 0.4]), th)
    assert out.tolist() == [VEG, BUILT, OTHER, OTHER]


def test_majority_filter_removes_single_pixels_and_keeps_nodata():
    a = np.zeros((5, 5), np.uint8)
    a[2, 2] = BUILT
    a[0, 0] = 255
    f = majority_filter(a)
    assert f[2, 2] == VEG and f[0, 0] == 255


def test_small_patch_removal():
    m = np.zeros((10, 10), bool)
    m[1, 1] = True                # 1-pixel patch -> removed
    m[5:8, 5:8] = True            # 9-pixel patch -> kept
    out = remove_small_patches(m, 5)
    assert not out[1, 1] and out[5:8, 5:8].all()


def test_label_parsing():
    assert parse_class("Built-up") == BUILT and parse_class("veg") == VEG
    assert parse_class("Other") == OTHER and parse_class("") is None
    assert parse_change("Yes") is True and parse_change("no") is False


def test_olofsson_estimator_matches_hand_calculation():
    # two strata: 900 px "no change" map, 100 px "change" map; 10 points each
    strata_px = {0: 900, 3: 100}
    df = pd.DataFrame({"stratum": [0] * 10 + [3] * 10,
                       "actual_change_bool": [True] + [False] * 9 + [True] * 8 + [False] * 2})
    out = _olofsson(strata_px, df, px_km2=0.0001)
    p = 0.9 * 0.1 + 0.1 * 0.8
    assert out["error_adjusted_conversion_km2"] == pytest.approx(p * 1000 * 0.0001, abs=1e-3)
    se = np.sqrt(0.9 ** 2 * (0.1 * 0.9 * 10 / 9) / 10 + 0.1 ** 2 * (0.8 * 0.2 * 10 / 9) / 10)
    assert out["ci95_km2"] == pytest.approx(1.96 * se * 1000 * 0.0001, abs=1e-3)


def test_config_enforces_pdf_minimums():
    with pytest.raises(AssertionError):
        ProjectConfig(min_samples_per_class=20).validate()
    with pytest.raises(AssertionError):
        ProjectConfig(validation_points_per_stratum=10).validate()
    ProjectConfig().validate()
