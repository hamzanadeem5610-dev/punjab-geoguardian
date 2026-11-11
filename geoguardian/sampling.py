"""Training samples (PDF section 14) and validation points (PDF section 18).

Training samples - reference-assisted, then checked by eye
-----------------------------------------------------------
Hand-clicking 150+ points is slow and tends to cluster. Instead, candidate pixels
are chosen where two independent global land-cover products agree with each
other *in every year* (ESA WorldCover 2021 and Dynamic World for each season).
That agreement means the land is stable and clearly of one class. Spectral
purity filters remove doubtful pixels: a vegetation sample must be green in every
year, and a built-up sample must not be.
Samples are then spread across a k x k block grid (round-robin), at least
``sample_spacing_m`` apart, so they are *not all from one small location*.
``validation.make_sample_review_sheet`` creates image chips so every sample can
be checked visually before training, as the PDF asks.

Validation points - independent, stratified, blind
---------------------------------------------------
60 points, stratified by the mapped result (15 vegetation, 15 built-up, 15 other,
15 detected conversions), kept >= 100 m from every training sample (so training
and validation are never the same) and >= 200 m apart from each other.
Stratification guarantees that the rare "conversion" class is actually checked.
It also enables an unbiased, error-adjusted area estimate (Olofsson et al. 2014).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import BUILT, CLASS_SHORT, FEATURES, OTHER, VEG, ProjectConfig
from .indices import Scene

STRATA = {0: "Vegetation (stable)", 1: "Built-up (not new)", 2: "Other",
          3: "Detected conversion"}


# ----------------------------------------------------------------- helpers
def _block_ids(shape, k: int) -> np.ndarray:
    h, w = shape
    rb = np.minimum((np.arange(h) * k) // h, k - 1)
    cb = np.minimum((np.arange(w) * k) // w, k - 1)
    return (rb[:, None] * k + cb[None, :]).astype(np.int32)


def _spread_sample(cand: np.ndarray, blocks: np.ndarray, n: int, spacing_px: int,
                   rng: np.random.Generator, occupied: Optional[set] = None) -> np.ndarray:
    """Pick up to n pixels from boolean mask ``cand``: round-robin over spatial blocks,
    at most one pixel per spacing cell. Returns flat indices."""
    occupied = set() if occupied is None else occupied
    h, w = cand.shape
    flat = np.flatnonzero(cand)
    if flat.size == 0:
        return np.array([], dtype=np.int64)
    if flat.size > 100 * n:  # keep the Python loop small; still a uniform random subset
        flat = rng.choice(flat, size=100 * n, replace=False)
    blk = blocks.ravel()[flat]
    per_block: Dict[int, List[int]] = {}
    order = rng.permutation(flat.size)
    for i in order:
        per_block.setdefault(int(blk[i]), []).append(int(flat[i]))
    queues = [per_block[b] for b in rng.permutation(list(per_block))]
    ptr = [0] * len(queues)
    chosen: List[int] = []
    s = max(1, spacing_px)
    while len(chosen) < n:
        progressed = False
        for qi, q in enumerate(queues):
            while ptr[qi] < len(q):
                p = q[ptr[qi]]
                ptr[qi] += 1
                cell = ((p // w) // s, (p % w) // s)
                if cell in occupied:
                    continue
                occupied.add(cell)
                chosen.append(p)
                progressed = True
                break
            if len(chosen) >= n:
                break
        if not progressed:
            break
    return np.array(chosen, dtype=np.int64)


def _purity(scene: Scene, cls: int, cfg: ProjectConfig) -> np.ndarray:
    ok = scene.common_valid.copy()
    for yd in scene.years.values():
        nd = yd.idx["NDVI"]
        if cls == VEG:
            ok &= nd >= cfg.veg_ndvi_min
        elif cls == BUILT:
            ok &= nd <= cfg.built_ndvi_max
        else:
            ok &= nd < cfg.veg_ndvi_min
    return ok


def reference_agreement(scene: Scene) -> np.ndarray:
    """Pixel label where WorldCover and Dynamic World (every year) agree, else 255."""
    ref = scene.wc.copy() if scene.wc is not None else None
    for yd in scene.years.values():
        if yd.dw is None:
            continue
        if ref is None:
            ref = yd.dw.copy()
        else:
            ref = np.where(ref == yd.dw, ref, 255).astype(np.uint8)
    if ref is None:
        return np.full(scene.grid.shape, 255, np.uint8)
    ref[~scene.common_valid] = 255
    return ref


def _rule_label(scene: Scene, cfg: ProjectConfig) -> np.ndarray:
    """Spectral last-resort labels, consistent in all years (tier 3)."""
    lab = np.full(scene.grid.shape, 255, np.uint8)
    per_year = []
    for yd in scene.years.values():
        v, b, w = yd.idx["NDVI"], yd.idx["NDBI"], yd.idx["MNDWI"]
        l = np.full(v.shape, 255, np.uint8)
        l[(w > cfg.rule_mndwi_water) & (v < 0.2)] = OTHER
        l[(v >= cfg.veg_ndvi_min + 0.1)] = VEG
        l[(b > cfg.rule_ndbi_built + 0.05) & (v < 0.2) & (w < 0)] = BUILT
        per_year.append(l)
    same = np.all([p == per_year[0] for p in per_year], axis=0)
    lab[same] = per_year[0][same]
    lab[~scene.common_valid] = 255
    return lab


# ----------------------------------------------------------- training samples
def generate_training_samples(scene: Scene, cfg: ProjectConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    grid = scene.grid
    blocks = _block_ids(grid.shape, cfg.spatial_blocks)
    spacing_px = int(round(cfg.sample_spacing_m / grid.res))
    ref = reference_agreement(scene)
    recent_dw = scene.years[cfg.recent_year].dw
    rule = None
    occupied: set = set()
    rows = []
    for cls in (VEG, BUILT, OTHER):
        pure = _purity(scene, cls, cfg)
        n_target = cfg.samples_per_class_per_year
        picked = _spread_sample((ref == cls) & pure, blocks, n_target, spacing_px, rng, occupied)
        tiers = ["T1: WorldCover = Dynamic World (all years) + spectral purity"] * len(picked)
        if len(picked) < n_target and recent_dw is not None:
            extra = _spread_sample((recent_dw == cls) & pure & (ref == 255), blocks,
                                   n_target - len(picked), spacing_px, rng, occupied)
            picked = np.concatenate([picked, extra])
            tiers += ["T2: Dynamic World recent year + spectral purity"] * len(extra)
        if len(picked) < cfg.min_samples_per_class:
            rule = _rule_label(scene, cfg) if rule is None else rule
            extra = _spread_sample((rule == cls) & pure, blocks,
                                   cfg.min_samples_per_class - len(picked), spacing_px, rng,
                                   occupied)
            picked = np.concatenate([picked, extra])
            tiers += ["T3: consistent spectral rule (all years)"] * len(extra)
        if len(picked) < cfg.min_samples_per_class:
            print(f"[samples] WARNING: only {len(picked)} samples for {CLASS_SHORT[cls]} "
                  f"(minimum {cfg.min_samples_per_class}). Add manual samples.")
        r, c = np.unravel_index(picked, grid.shape)
        x, y = grid.pixel_center_xy(r, c)
        lon, lat = grid.to_lonlat(x, y)
        for i in range(len(picked)):
            rows.append({"class_id": cls, "class_name": CLASS_SHORT[cls], "row": int(r[i]),
                         "col": int(c[i]), "x": float(x[i]), "y": float(y[i]),
                         "lon": round(float(lon[i]), 6), "lat": round(float(lat[i]), 6),
                         "block": int(blocks[r[i], c[i]]), "source": tiers[i]})
    df = pd.DataFrame(rows)
    df.insert(0, "sample_id", np.arange(1, len(df) + 1))
    df["keep"] = True  # set False after visual review to drop a bad sample
    counts = df.groupby("class_name").size().to_dict()
    print(f"[samples] training locations per class: {counts} "
          f"(x {len(scene.years)} years = {len(df) * len(scene.years)} training rows)")
    return df


def training_table(scene: Scene, samples: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (sample location, year) with all features."""
    s = samples[samples["keep"]].reset_index(drop=True)
    parts = []
    for y, yd in scene.years.items():
        feats = yd.feature_stack(s["row"].values, s["col"].values)
        part = pd.DataFrame(feats, columns=FEATURES)
        part.insert(0, "year", y)
        part.insert(0, "block", s["block"].values)
        part.insert(0, "class_id", s["class_id"].values)
        part.insert(0, "sample_id", s["sample_id"].values)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna().reset_index(drop=True)


# --------------------------------------------------------- validation points
def validation_strata(class_recent: np.ndarray, conversion: np.ndarray) -> np.ndarray:
    st = np.full(class_recent.shape, 255, np.uint8)
    st[class_recent == VEG] = 0
    st[class_recent == BUILT] = 1
    st[class_recent == OTHER] = 2
    st[conversion] = 3
    return st


def generate_validation_points(scene: Scene, cfg: ProjectConfig, strata: np.ndarray,
                               training: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed + 1)
    grid = scene.grid
    # exclusion zone around every training sample
    train_mask = np.zeros(grid.shape, bool)
    train_mask[training["row"].values, training["col"].values] = True
    rad = int(np.ceil(cfg.validation_min_distance_to_training_m / grid.res))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy ** 2 + xx ** 2) <= rad ** 2
    near_train = ndimage.binary_dilation(train_mask, structure=disk)
    allowed = scene.common_valid & ~near_train
    blocks = _block_ids(grid.shape, 4)
    spacing_px = int(round(cfg.validation_min_spacing_m / grid.res))
    occupied: set = set()
    n = cfg.validation_points_per_stratum
    picks: Dict[int, np.ndarray] = {}
    for s in (3, 0, 1, 2):  # rarest stratum first
        picks[s] = _spread_sample(allowed & (strata == s), blocks, n, spacing_px, rng, occupied)
    total = sum(len(v) for v in picks.values())
    need = 4 * n - total
    if need > 0:  # e.g. very few conversions -> top up the larger strata
        print(f"[validation] only {len(picks[3])} conversion pixels available; "
              f"topping up {need} points from other strata")
        for s in (0, 1, 2):
            extra = _spread_sample(allowed & (strata == s), blocks, need, spacing_px, rng, occupied)
            picks[s] = np.concatenate([picks[s], extra])
            need -= len(extra)
            if need <= 0:
                break
    rows = []
    for s, flat in picks.items():
        r, c = np.unravel_index(flat, grid.shape)
        x, y = grid.pixel_center_xy(r, c)
        lon, lat = grid.to_lonlat(x, y)
        for i in range(len(flat)):
            rows.append({"stratum": s, "stratum_name": STRATA[s], "row": int(r[i]),
                         "col": int(c[i]), "lon": round(float(lon[i]), 6),
                         "lat": round(float(lat[i]), 6)})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=cfg.random_seed).reset_index(drop=True)
    df.insert(0, "point_id", np.arange(1, len(df) + 1))
    print(f"[validation] {len(df)} points: "
          f"{df.groupby('stratum_name').size().to_dict()}")
    return df
