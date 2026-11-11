"""Synthetic Sentinel-2-like scene for offline testing.

It writes a stack in exactly the same format as ``gee_data.collect_data``, so the
whole pipeline can be tested without Earth Engine. The scene has crop fields,
fallow fields (bare soil, which is easily confused with built-up), an old town,
housing schemes that grow between years, roads, a canal, ponds, cloud gaps and
mixed pixels. ``truth`` maps are returned so that simulated "manual" labels can be
produced in tests.

Synthetic outputs are watermarked everywhere and must never be reported as real.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from .aoi import aoi_geojson, grid_from_center
from .config import BUILT, OTHER, S2_BANDS, VEG, ProjectConfig
from .io import write_json, write_stack

SIG = {  # surface reflectance B2 B3 B4 B8 B11 B12
    "crop": [0.030, 0.065, 0.040, 0.360, 0.190, 0.095],
    "tree": [0.025, 0.050, 0.030, 0.300, 0.150, 0.070],
    "soil": [0.085, 0.120, 0.160, 0.235, 0.300, 0.255],
    "built": [0.115, 0.125, 0.140, 0.185, 0.225, 0.205],
    "road": [0.100, 0.110, 0.120, 0.150, 0.190, 0.180],
    "water": [0.060, 0.075, 0.050, 0.030, 0.015, 0.010],
}


def _rect(m, r0, c0, h, w, v):
    m[max(r0, 0):r0 + h, max(c0, 0):c0 + w] = v


def _truth(size: int, rng) -> Dict[int, np.ndarray]:
    """Cover-type maps per year: 0 crop, 1 tree, 2 soil, 3 built, 4 road, 5 water."""
    base = np.zeros((size, size), np.uint8)
    # field mosaic: some fallow fields (soil)
    fs = 30
    for r in range(0, size, fs):
        for c in range(0, size, fs):
            if rng.random() < 0.12:
                _rect(base, r, c, fs, fs, 2)
    # tree belts
    for _ in range(25):
        r, c = rng.integers(0, size, 2)
        _rect(base, r, c, rng.integers(4, 12), rng.integers(4, 12), 1)
    # old town (north-east) with green housing (trees inside built area)
    yy, xx = np.mgrid[:size, :size]
    town = ((yy - size * 0.18) ** 2 + (xx - size * 0.8) ** 2) < (size * 0.17) ** 2
    base[town] = 3
    base[town & (rng.random((size, size)) < 0.08)] = 1
    # canal and ponds
    base[:, int(size * 0.3):int(size * 0.3) + 3] = 5
    for _ in range(4):
        r, c = rng.integers(20, size - 20, 2)
        base[((yy - r) ** 2 + (xx - c) ** 2) < rng.integers(16, 64)] = 5
    # main road (diagonal) and grid roads in town
    for i in range(size):
        j = int(size * 0.15 + i * 0.6)
        if 0 <= j < size - 2:
            base[i, j:j + 2] = 4
    y19 = base.copy()
    y22 = base.copy()
    y25 = base.copy()
    # housing schemes: appear 2022, expand 2025
    schemes = [(int(size * 0.55), int(size * 0.55), 70, 90), (int(size * 0.62), int(size * 0.1), 60, 60),
               (int(size * 0.2), int(size * 0.45), 50, 70), (int(size * 0.8), int(size * 0.7), 60, 80)]
    for i, (r, c, h, w) in enumerate(schemes):
        if i < 2:
            _rect(y22, r, c, h // 2, w, 3)
        _rect(y25, r, c, h, w, 3)
        for rr in range(r, r + h, 15):  # internal roads
            _rect(y25, rr, c, 1, w, 4)
    # 2025: one scheme has cleared land (construction, still soil) -> hard case
    _rect(y25, int(size * 0.35), int(size * 0.12), 40, 40, 2)
    # town edge growth
    grow = ((yy - size * 0.18) ** 2 + (xx - size * 0.8) ** 2) < (size * 0.22) ** 2
    y25[grow & (y25 == 0)] = 3
    grow22 = ((yy - size * 0.18) ** 2 + (xx - size * 0.8) ** 2) < (size * 0.195) ** 2
    y22[grow22 & (y22 == 0)] = 3
    # seasonal: different fallow fields in 2025 (not real change)
    for r in range(0, size, fs):
        for c in range(0, size, fs):
            blk = y25[r:r + fs, c:c + fs]
            if rng.random() < 0.05:
                blk[blk == 0] = 2
            elif rng.random() < 0.3:
                blk[blk == 2] = 0
    return {2019: y19, 2022: y22, 2025: y25}


def to_class(t: np.ndarray) -> np.ndarray:
    lut = np.array([VEG, VEG, OTHER, BUILT, BUILT, OTHER], np.uint8)
    return lut[t]


def make_synthetic_stack(cfg: ProjectConfig, size: int = 600, seed: int = 7) -> Tuple[Dict, Dict]:
    rng = np.random.default_rng(seed)
    grid = grid_from_center(cfg.center_lat, cfg.center_lon, size * cfg.pixel_size_m / 1000,
                            size * cfg.pixel_size_m / 1000, cfg.pixel_size_m)
    truth = _truth(size, rng)
    names, arrs = [], []
    keys = ["crop", "tree", "soil", "built", "road", "water"]
    infos = {}
    for y in cfg.years:
        t = truth[y]
        onehot = np.stack([(t == i).astype(np.float32) for i in range(6)])
        mixed = np.stack([ndimage.gaussian_filter(o, 0.7) for o in onehot])  # mixed pixels
        mixed /= mixed.sum(0, keepdims=True)
        haze = 1 + 3.0 * ndimage.gaussian_filter(rng.normal(size=t.shape), 40)  # ~2 % smooth haze
        obs = rng.integers(3, 9, t.shape).astype(np.int16)
        if y == 2022:  # a cloud gap with no clear observation
            field = ndimage.gaussian_filter(rng.random(t.shape), 25)
            cl = field > np.percentile(field, 97)  # ~3 % of the scene under persistent cloud
            obs[cl] = 0
        for bi, b in enumerate(S2_BANDS):
            val = sum(mixed[k] * SIG[keys[k]][bi] for k in range(6))
            val = val * (1 + rng.normal(0, 0.06, t.shape)) * haze
            # crop vigour varies by field and year
            val += (t == 0) * rng.normal(0, 0.02, t.shape) * (1 if b == "B8" else 0.3)
            dn = np.clip(np.round(val * 10000), 1, 20000).astype(np.int16)
            dn[obs == 0] = -1
            arrs.append(dn)
            names.append(f"y{y}_{b}")
        arrs.append(obs)
        names.append(f"y{y}_obs")
        cls = to_class(t)
        dw = cls.copy()
        flip = rng.random(t.shape) < 0.10
        dw[flip] = rng.integers(0, 3, flip.sum())
        arrs.append(dw.astype(np.int16))
        names.append(f"y{y}_dw")
        start, end = cfg.season_range(y)
        infos[str(y)] = {"year": y, "start": start, "end": end, "n_scenes": 9, "n_dates": 6,
                         "dates": [], "cloud_pct": [], "tiles": ["SYNTHETIC"],
                         "window_extended_days": 0}
    wc = to_class(truth[2022])
    flip = rng.random(wc.shape) < 0.08
    wc[flip] = rng.integers(0, 3, flip.sum())
    arrs += [wc.astype(np.int16), np.ones(wc.shape, np.int16)]
    names += ["wc2021", "aoi_mask"]
    write_stack(cfg.stack_path, np.stack(arrs).astype(np.int16), grid, names)
    meta = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "config": cfg.to_dict(), "grid": grid.to_dict(), "band_names": names,
            "image_info": infos, "aoi_geojson": aoi_geojson(grid, cfg.aoi_name), "context": {},
            "data_sources": ["SYNTHETIC TEST SCENE - generated by geoguardian.synthetic "
                             "(no real satellite data)"], "synthetic": True}
    write_json(cfg.stack_meta_path, meta)
    return meta, {y: to_class(t) for y, t in truth.items()}
