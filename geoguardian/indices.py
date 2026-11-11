"""Remote-sensing indices and the in-memory scene.

NDVI  = (NIR - Red)   / (NIR + Red)    -> vegetation   (Sentinel-2: B8, B4)
NDBI  = (SWIR - NIR)  / (SWIR + NIR)   -> built-up     (Sentinel-2: B11, B8)
MNDWI = (Green - SWIR)/ (Green + SWIR) -> open water   (Sentinel-2: B3, B11)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .aoi import Grid
from .config import FEATURES, S2_BANDS
from .io import read_json, read_stack

REFLECTANCE_SCALE = 10000.0


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = np.where(np.abs(den) > 1e-6, (a - b) / den, np.nan)
    return np.clip(nd, -1.0, 1.0).astype(np.float32)


def ndvi(nir, red):
    return normalized_difference(nir, red)


def ndbi(swir, nir):
    return normalized_difference(swir, nir)


def mndwi(green, swir):
    return normalized_difference(green, swir)


@dataclass
class YearData:
    year: int
    bands: Dict[str, np.ndarray]          # surface reflectance 0..1 (float32)
    valid: np.ndarray                     # bool
    obs: np.ndarray                       # clear observations per pixel
    dw: Optional[np.ndarray] = None       # Dynamic World mode label mapped to 0/1/2 (255 = none)
    idx: Dict[str, np.ndarray] = field(default_factory=dict)

    def compute_indices(self) -> None:
        b = self.bands
        self.idx["NDVI"] = ndvi(b["B8"], b["B4"])
        self.idx["NDBI"] = ndbi(b["B11"], b["B8"])
        self.idx["MNDWI"] = mndwi(b["B3"], b["B11"])
        for k in self.idx:
            self.idx[k][~self.valid] = np.nan

    def layer(self, name: str) -> np.ndarray:
        return self.bands[name] if name in self.bands else self.idx[name]

    def feature_stack(self, rows=None, cols=None, features: List[str] = FEATURES) -> np.ndarray:
        if rows is None:
            return np.stack([self.layer(f)[self.valid] for f in features], axis=1)
        return np.stack([self.layer(f)[rows, cols] for f in features], axis=1)


@dataclass
class Scene:
    grid: Grid
    years: Dict[int, YearData]
    aoi_mask: np.ndarray
    wc: Optional[np.ndarray]
    meta: Dict

    @property
    def common_valid(self) -> np.ndarray:
        v = self.aoi_mask.copy()
        for yd in self.years.values():
            v &= yd.valid
        return v


def load_scene(stack_path, meta_path, years: List[int]) -> Scene:
    layers, grid = read_stack(stack_path)
    meta = read_json(meta_path)
    aoi_mask = layers.get("aoi_mask", np.ones(grid.shape, np.int16)).astype(bool)
    ydata = {}
    for y in years:
        missing = [b for b in S2_BANDS if f"y{y}_{b}" not in layers]
        if missing:
            raise KeyError(f"Year {y} missing bands {missing} in {stack_path}. "
                           "Re-run data collection with this year in the config.")
        raw = {b: layers[f"y{y}_{b}"] for b in S2_BANDS}
        obs = layers.get(f"y{y}_obs", np.ones(grid.shape, np.int16))
        valid = aoi_mask & (obs > 0)
        for a in raw.values():
            valid &= a > 0
        bands = {b: (a.astype(np.float32) / REFLECTANCE_SCALE) for b, a in raw.items()}
        for a in bands.values():
            a[~valid] = np.nan
        dw = layers.get(f"y{y}_dw")
        yd = YearData(year=y, bands=bands, valid=valid, obs=obs.astype(np.int16),
                      dw=None if dw is None else dw.astype(np.uint8))
        yd.compute_indices()
        ydata[y] = yd
    wc = layers.get("wc2021")
    return Scene(grid=grid, years=ydata, aoi_mask=aoi_mask,
                 wc=None if wc is None else wc.astype(np.uint8), meta=meta)
