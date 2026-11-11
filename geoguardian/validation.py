"""Validation with manually checked points (PDF section 18).

Workflow
--------
1. ``build_validation_sheet``: writes ``outputs/validation/validation_points.csv``
   (60 points, with empty ``actual_*`` columns) and ``validation_chips.pdf``
   (2 Sentinel-2 chips per point plus a Google Maps satellite link).
2. You check each point by eye: open the link (or the geemap map), look at the
   chips, and fill ``actual_class`` (Vegetation / Built-up / Other),
   ``actual_change`` (yes / no) and ``notes``. Use the Colab widget
   (``labeler.py``) or any spreadsheet. The chips do **not** show the model's
   prediction, so the labelling stays blind and unbiased.
3. ``assess_accuracy`` builds the required table (Point ID, Actual, Predicted,
   Correct/Incorrect, Notes), the confusion matrix, overall / area-weighted accuracy,
   kappa and producer's / user's accuracy. It also gives the error-adjusted
   conversion area with a 95 % confidence interval (Olofsson et al., 2014,
   *Remote Sensing of Environment* 148: 42-57).

If any point is still unlabelled, those points fall back to the Dynamic World
label, and every output is clearly marked PROVISIONAL. That fallback lets you see
results early. It is **not** a substitute for the manual check the PDF requires.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import BUILT, CLASS_SHORT, OTHER, VEG, ProjectConfig
from .indices import Scene
from .sampling import STRATA

CLASS_ALIASES = {
    "0": VEG, "veg": VEG, "vegetation": VEG, "agriculture": VEG, "vegetation / agriculture": VEG,
    "green": VEG, "crop": VEG, "trees": VEG,
    "1": BUILT, "built": BUILT, "built-up": BUILT, "builtup": BUILT, "urban": BUILT,
    "built-up / urban": BUILT, "built up": BUILT,
    "2": OTHER, "other": OTHER, "bare": OTHER, "water": OTHER, "barren": OTHER,
}
YES = {"yes", "y", "1", "true", "t", "changed", "change"}
NO = {"no", "n", "0", "false", "f", "stable", "no change"}
LABEL_COLUMNS = ["actual_class", "actual_change", "notes"]


def parse_class(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    return CLASS_ALIASES.get(s)


def parse_change(v) -> Optional[bool]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s.endswith(".0"):
        s = s[:-2]
    if s in YES:
        return True
    if s in NO:
        return False
    return None


# ---------------------------------------------------------------- sheet
def build_validation_sheet(scene: Scene, points: pd.DataFrame, maps: Dict[int, np.ndarray],
                           conf: Dict[int, np.ndarray], conversion: np.ndarray,
                           cfg: ProjectConfig, path: Path) -> pd.DataFrame:
    b, r = cfg.baseline_year, cfg.recent_year
    rows, cols = points["row"].values, points["col"].values
    df = points.copy()
    df[f"pred_class_{b}"] = [CLASS_SHORT.get(int(v), "NoData") for v in maps[b][rows, cols]]
    df["predicted_class"] = [CLASS_SHORT.get(int(v), "NoData") for v in maps[r][rows, cols]]
    df["predicted_change"] = np.where(conversion[rows, cols], "yes", "no")
    df["confidence"] = np.round(np.fmin(conf[b][rows, cols], conf[r][rows, cols]), 3)
    for y in (b, r):
        yd = scene.years[y]
        df[f"NDVI_{y}"] = np.round(yd.idx["NDVI"][rows, cols], 3)
        df[f"NDBI_{y}"] = np.round(yd.idx["NDBI"][rows, cols], 3)
        df[f"MNDWI_{y}"] = np.round(yd.idx["MNDWI"][rows, cols], 3)
        df[f"obs_{y}"] = yd.obs[rows, cols]
        df[f"hint_dw_{y}"] = ([CLASS_SHORT.get(int(v), "") for v in yd.dw[rows, cols]]
                              if yd.dw is not None else "")
    df["hint_worldcover_2021"] = ([CLASS_SHORT.get(int(v), "") for v in scene.wc[rows, cols]]
                                  if scene.wc is not None else "")
    df["google_maps"] = [f"https://www.google.com/maps/@{la:.6f},{lo:.6f},250m/data=!3m1!1e3"
                         for lo, la in zip(df.lon, df.lat)]
    for c in LABEL_COLUMNS:
        df[c] = ""
    # keep labels that were already entered for the same location
    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        key_old = old["lon"].astype(float).round(6).astype(str) + "_" + \
            old["lat"].astype(float).round(6).astype(str)
        lut = old.set_index(key_old)
        key = df["lon"].round(6).astype(str) + "_" + df["lat"].round(6).astype(str)
        for c in LABEL_COLUMNS:
            if c in lut.columns:
                df[c] = [lut[c].get(k, "") for k in key]
        n_kept = int((df["actual_class"] != "").sum())
        if n_kept:
            print(f"[validation] kept {n_kept} existing manual labels from {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _chip(rgb: np.ndarray, r: int, c: int, half: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.ones((2 * half, 2 * half, 3), np.float32)
    r0, c0 = r - half, c - half
    rs, cs = max(0, r0), max(0, c0)
    re, ce = min(h, r0 + 2 * half), min(w, c0 + 2 * half)
    out[rs - r0:re - r0, cs - c0:ce - c0] = rgb[rs:re, cs:ce]
    return out


def chip_sheet(scene: Scene, df: pd.DataFrame, cfg: ProjectConfig, path: Path,
               title: str, id_col: str = "point_id", half: int = 32, per_page: int = 6,
               label_col: Optional[str] = None, chip_dir: Optional[Path] = None) -> None:
    """Multi-page PDF: for each point, baseline RGB | recent RGB | recent false colour,
    centred on the point (64 x 64 px = 640 m). Optionally also saves one PNG per point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .viz import false_color, rgb_image, shared_stretch

    b, r = cfg.baseline_year, cfg.recent_year
    lims = shared_stretch(scene, [b, r])
    rgbs = {y: rgb_image(scene.years[y], lims) for y in (b, r)}
    fc = false_color(scene.years[r])
    path.parent.mkdir(parents=True, exist_ok=True)
    if chip_dir:
        chip_dir.mkdir(parents=True, exist_ok=True)
    panels = [(rgbs[b], f"{b} true colour"), (rgbs[r], f"{r} true colour"),
              (fc, f"{r} false colour (NIR-R-G)")]

    def draw(ax_row, row):
        for ax, (img, ttl) in zip(ax_row, panels):
            ax.imshow(_chip(img, int(row["row"]), int(row["col"]), half), interpolation="nearest")
            ax.plot([half - 2.5, half + 1.5, half + 1.5, half - 2.5, half - 2.5],
                    [half - 2.5, half - 2.5, half + 1.5, half + 1.5, half - 2.5],
                    color="cyan", lw=1.2)
            ax.axhline(half - 0.5, color="cyan", lw=0.4, alpha=0.5)
            ax.axvline(half - 0.5, color="cyan", lw=0.4, alpha=0.5)
            ax.set_title(ttl, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        lab = f" | {row[label_col]}" if label_col else ""
        ax_row[0].set_ylabel(f"#{int(row[id_col])}{lab}\n{row['lat']:.5f}, {row['lon']:.5f}",
                             fontsize=7)

    with PdfPages(path) as pdf:
        for start in range(0, len(df), per_page):
            sub = df.iloc[start:start + per_page]
            fig, axes = plt.subplots(per_page, 3, figsize=(8.27, 11.69))
            fig.suptitle(f"{title} - page {start // per_page + 1} "
                         f"(chip = 640 m, box = 40 m around the point)", fontsize=9)
            for i in range(per_page):
                if i < len(sub):
                    draw(axes[i], sub.iloc[i])
                else:
                    for ax in axes[i]:
                        ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    if chip_dir:
        for _, row in df.iterrows():
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
            draw(axes, row)
            fig.tight_layout()
            fig.savefig(chip_dir / f"point_{int(row[id_col]):03d}.png", dpi=90)
            plt.close(fig)


# ------------------------------------------------------------ accuracy
def load_labels(path: Path, cfg: ProjectConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in LABEL_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    r = cfg.recent_year
    actual = df["actual_class"].map(parse_class)
    change = df["actual_change"].map(parse_change)
    manual = actual.notna()
    src = np.where(manual, "Manual visual check", "Dynamic World (PROVISIONAL)")
    hint_r = df.get(f"hint_dw_{r}", pd.Series([""] * len(df))).map(parse_class)
    hint_b = df.get(f"hint_dw_{cfg.baseline_year}", pd.Series([""] * len(df))).map(parse_class)
    actual = actual.where(manual, hint_r)
    hint_change = (hint_b == VEG) & (hint_r == BUILT)
    # missing change label: manual non-built point -> cannot be a conversion; manual built-up
    # point -> unknown (left out of change metrics); unlabelled point -> provisional DW hint
    fill = pd.Series(np.where(manual, np.where(actual != BUILT, False, None), hint_change),
                     index=df.index, dtype=object)
    change = change.astype(object).where(change.notna(), fill)
    df["actual_class_id"] = actual
    df["actual_change_bool"] = change.astype(object)
    df["label_source"] = src
    df["notes"] = df["notes"].fillna("").astype(str)
    return df


def _olofsson(strata_px: Dict[int, int], df: pd.DataFrame, px_km2: float) -> Dict:
    """Stratified estimator of conversion area and 95 % CI (simple-random within strata)."""
    total_px = sum(strata_px.values())
    est, var = 0.0, 0.0
    for h, Nh in strata_px.items():
        sub = df[df["stratum"] == h]
        nh = len(sub)
        if nh == 0 or Nh == 0:
            continue
        Wh = Nh / total_px
        yv = sub["actual_change_bool"].astype(float).values
        p = yv.mean()
        s2 = p * (1 - p) * nh / (nh - 1) if nh > 1 else 0.0
        est += Wh * p
        var += Wh ** 2 * s2 / nh
    area = est * total_px * px_km2
    ci = 1.96 * np.sqrt(var) * total_px * px_km2
    return {"error_adjusted_conversion_km2": round(area, 3), "ci95_km2": round(ci, 3),
            "ci95_low_km2": round(max(area - ci, 0), 3), "ci95_high_km2": round(area + ci, 3)}


def assess_accuracy(df: pd.DataFrame, strata_map: np.ndarray, scene: Scene,
                    cfg: ProjectConfig) -> Dict:
    lab = [VEG, BUILT, OTHER]
    d = df[df["actual_class_id"].notna() & df["predicted_class"].isin(
        [CLASS_SHORT[k] for k in lab])].copy()
    inv = {v: k for k, v in CLASS_SHORT.items()}
    d["pred_id"] = d["predicted_class"].map(inv).astype(int)
    d["actual_class_id"] = d["actual_class_id"].astype(int)
    d["correct"] = d["pred_id"] == d["actual_class_id"]

    cm = pd.crosstab(pd.Categorical(d["actual_class_id"].map(CLASS_SHORT),
                                    categories=[CLASS_SHORT[k] for k in lab]),
                     pd.Categorical(d["predicted_class"], categories=[CLASS_SHORT[k] for k in lab]),
                     rownames=["Actual"], colnames=["Predicted"], dropna=False)
    n = len(d)
    oa = d["correct"].mean() if n else float("nan")
    m = cm.values.astype(float)
    pe = (m.sum(0) * m.sum(1)).sum() / max(n, 1) ** 2
    kappa = (oa - pe) / (1 - pe) if n and pe < 1 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        pa = np.diag(m) / m.sum(1)   # producer's (recall) per actual class
        ua = np.diag(m) / m.sum(0)   # user's (precision) per mapped class
    per_class = pd.DataFrame({"class": [CLASS_SHORT[k] for k in lab],
                              "reference_points": m.sum(1).astype(int),
                              "mapped_points": m.sum(0).astype(int),
                              "producers_accuracy": np.round(pa, 3),
                              "users_accuracy": np.round(ua, 3)})

    # area-weighted overall accuracy (corrects for equal-allocation stratified sampling)
    valid = scene.common_valid & (strata_map != 255)
    strata_px = {h: int(np.count_nonzero(valid & (strata_map == h))) for h in STRATA}
    tot = sum(strata_px.values())
    oa_w = sum(strata_px[h] / tot * d[d["stratum"] == h]["correct"].mean()
               for h in STRATA if strata_px[h] and (d["stratum"] == h).any())

    # change accuracy
    c = df[df["actual_change_bool"].notna()].copy()
    c["actual_change_bool"] = c["actual_change_bool"].astype(bool)
    pred = c["predicted_change"].eq("yes")
    act = c["actual_change_bool"]
    tp, fp = int((pred & act).sum()), int((pred & ~act).sum())
    fn, tn = int((~pred & act).sum()), int((~pred & ~act).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    change_metrics = {"true_positive": tp, "false_positive (false conversion)": fp,
                      "false_negative (missed change)": fn, "true_negative": tn,
                      "conversion_precision_users_acc": round(prec, 3),
                      "conversion_recall_producers_acc": round(rec, 3),
                      "change_overall_accuracy": round((tp + tn) / max(len(c), 1), 3)}
    change_metrics.update(_olofsson(strata_px, c, scene.grid.pixel_area_km2))

    table = pd.DataFrame({
        "Point ID": d["point_id"].values,
        "Latitude": d["lat"].values, "Longitude": d["lon"].values,
        "Actual Class": d["actual_class_id"].map(CLASS_SHORT).values,
        "Predicted Class": d["predicted_class"].values,
        "Correct / Incorrect": np.where(d["correct"], "Correct", "Incorrect"),
        "Actual Change": d["actual_change_bool"].map({True: "yes", False: "no"}).fillna("").values,
        "Predicted Change": d["predicted_change"].values,
        "Notes": d["notes"].values,
        "Label Source": d["label_source"].values,
    }).sort_values("Point ID").reset_index(drop=True)

    n_manual = int((df["label_source"] == "Manual visual check").sum())
    out = {
        "n_points": int(n), "n_manual": n_manual, "provisional": n_manual < len(df),
        "correct": int(d["correct"].sum()), "incorrect": int((~d["correct"]).sum()),
        "overall_accuracy": round(float(oa), 4),
        "area_weighted_overall_accuracy": round(float(oa_w), 4),
        "kappa": round(float(kappa), 4), "per_class": per_class, "confusion_matrix": cm,
        "change": change_metrics, "table": table, "strata_pixels": strata_px,
        "merged": d,
    }
    tag = " (PROVISIONAL - not all points manually checked)" if out["provisional"] else ""
    print(f"[validation] {n} points, {out['correct']} correct -> accuracy "
          f"{100 * oa:.1f}% (area-weighted {100 * oa_w:.1f}%), kappa {kappa:.2f}{tag}")
    return out
