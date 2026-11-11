"""Static maps and charts (PDF section 16).

Every map uses the same layout: lat/lon tick labels, a scale bar, a north arrow and
a legend or colour bar. Both years share one RGB stretch and the same NDVI/NDBI
colour limits, so the two years can be compared directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from .aoi import Grid  # noqa: E402
from .config import BUILT, CLASS_COLORS, OTHER, VEG  # noqa: E402
from .indices import Scene, YearData  # noqa: E402

NDVI_RANGE = (-0.1, 0.8)
NDBI_RANGE = (-0.45, 0.25)
NDVI_CMAP = "RdYlGn"
NDBI_CMAP = "RdYlBu_r"
DPI = 150
WATERMARK = "SYNTHETIC TEST DATA - NOT REAL RESULTS"


# ----------------------------------------------------------------- images
def shared_stretch(scene: Scene, years: Sequence[int], bands=("B4", "B3", "B2"),
                   pct=(2, 98)) -> Dict[str, Tuple[float, float]]:
    lims = {}
    for b in set(bands) | {"B8"}:
        vals = np.concatenate([scene.years[y].bands[b][scene.years[y].valid][::7] for y in years])
        lims[b] = (float(np.nanpercentile(vals, pct[0])), float(np.nanpercentile(vals, pct[1])))
    return lims


def _stretch(a, lo, hi):
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def rgb_image(yd: YearData, lims: Dict[str, Tuple[float, float]], gamma: float = 0.9) -> np.ndarray:
    rgb = np.dstack([_stretch(yd.bands[b], *lims[b]) for b in ("B4", "B3", "B2")]) ** gamma
    rgb = np.nan_to_num(rgb, nan=1.0)
    rgb[~yd.valid] = 1.0
    return rgb.astype(np.float32)


def false_color(yd: YearData) -> np.ndarray:
    out = []
    for b in ("B8", "B4", "B3"):
        a = yd.bands[b]
        lo, hi = np.nanpercentile(a[yd.valid][::7], [2, 98])
        out.append(_stretch(a, lo, hi))
    fc = np.nan_to_num(np.dstack(out), nan=1.0)
    fc[~yd.valid] = 1.0
    return fc.astype(np.float32)


def class_rgba(cls: np.ndarray, colors: Dict[int, str]) -> np.ndarray:
    rgba = np.ones(cls.shape + (4,), np.float32)
    rgba[..., 3] = 0
    for k, c in colors.items():
        m = cls == k
        rgba[m, :3] = matplotlib.colors.to_rgb(c)
        rgba[m, 3] = 1
    return rgba


# ----------------------------------------------------------------- frame
def _extent(grid: Grid):
    xmin, ymin, xmax, ymax = grid.bounds
    return (xmin, xmax, ymin, ymax)


def decorate(ax, grid: Grid, title: str, synthetic: bool = False) -> None:
    xmin, ymin, xmax, ymax = grid.bounds
    xt = np.linspace(xmin, xmax, 5)[1:-1]
    yt = np.linspace(ymin, ymax, 5)[1:-1]
    lon, _ = grid.to_lonlat(xt, np.full_like(xt, ymin))
    _, lat = grid.to_lonlat(np.full_like(yt, xmin), yt)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:.3f}°E" for v in lon], fontsize=7)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.3f}°N" for v in lat], fontsize=7, rotation=90, va="center")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    # scale bar (bottom-left)
    width_km = (xmax - xmin) / 1000
    bar_km = next(v for v in (0.5, 1, 2, 5, 10, 20, 50, 100) if v >= width_km / 6)
    x0, y0 = xmin + 0.04 * (xmax - xmin), ymin + 0.05 * (ymax - ymin)
    ax.add_patch(Rectangle((x0, y0), bar_km * 1000, 0.012 * (ymax - ymin), color="black"))
    ax.add_patch(Rectangle((x0 + bar_km * 500, y0), bar_km * 500, 0.012 * (ymax - ymin),
                           color="white", ec="black", lw=0.5))
    ax.text(x0 + bar_km * 500, y0 + 0.022 * (ymax - ymin), f"{bar_km:g} km", ha="center",
            fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    # north arrow (top-right)
    xa, ya = xmax - 0.06 * (xmax - xmin), ymax - 0.16 * (ymax - ymin)
    ax.annotate("N", xy=(xa, ya + 0.1 * (ymax - ymin)), xytext=(xa, ya),
                arrowprops=dict(facecolor="black", width=3, headwidth=9), ha="center",
                fontsize=9, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    if synthetic:
        ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="red", alpha=0.35, rotation=30, fontweight="bold")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _legend(ax, colors: Dict[int, str], labels: Dict[int, str], loc="lower right"):
    ax.legend(handles=[Patch(fc=colors[k], ec="black", lw=0.4, label=labels[k]) for k in colors],
              loc=loc, fontsize=7, framealpha=0.9)


def _footer(fig, text: str):
    fig.text(0.01, 0.005, text, fontsize=6, color="#555555", ha="left", va="bottom")


# ------------------------------------------------------------------ maps
def map_rgb(img, grid, title, path, synthetic=False, footer=""):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, extent=_extent(grid), interpolation="nearest")
    decorate(ax, grid, title, synthetic)
    _footer(fig, footer)
    return _save(fig, path)


def map_continuous(arr, grid, title, path, cmap, vrange, label, synthetic=False, footer=""):
    fig, ax = plt.subplots(figsize=(7.6, 7))
    cm = plt.get_cmap(cmap).with_extremes(bad="white")
    im = ax.imshow(np.ma.masked_invalid(arr), extent=_extent(grid), cmap=cm, vmin=vrange[0],
                   vmax=vrange[1], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    decorate(ax, grid, title, synthetic)
    _footer(fig, footer)
    return _save(fig, path)


def map_classes(cls, grid, title, path, colors, labels, synthetic=False, footer="", areas=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(class_rgba(cls, colors), extent=_extent(grid), interpolation="nearest")
    lab = dict(labels)
    if areas:
        lab = {k: f"{v} ({areas.get(k, 0):.1f} km²)" for k, v in labels.items()}
    _legend(ax, colors, lab)
    decorate(ax, grid, title, synthetic)
    _footer(fig, footer)
    return _save(fig, path)


def map_change_overlay(rgb_recent, conv, grid, title, path, synthetic=False, footer="",
                       hot: Optional[pd.DataFrame] = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    gray = rgb_recent.mean(axis=2, keepdims=True)
    base = 0.55 + 0.45 * np.repeat(gray, 3, axis=2)  # faded recent image as context
    ax.imshow(base, extent=_extent(grid), interpolation="nearest")
    ov = np.zeros(conv.shape + (4,), np.float32)
    ov[conv] = (1, 0, 0, 1)
    ax.imshow(ov, extent=_extent(grid), interpolation="nearest")
    if hot is not None and len(hot):
        x, y = grid.from_lonlat(hot.center_lon.values, hot.center_lat.values)
        for i, (xx, yy) in enumerate(zip(x, y)):
            ax.annotate(str(i + 1), (xx, yy), fontsize=8, fontweight="bold", color="navy",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.2", fc="yellow", ec="navy", lw=0.8))
    ax.legend(handles=[Patch(fc="#FF0000", label="Possible agri-to-urban conversion"),
                       Patch(fc="yellow", ec="navy", label="Top hotspot (rank)")],
              loc="lower right", fontsize=7, framealpha=0.9)
    decorate(ax, grid, title, synthetic)
    _footer(fig, footer)
    return _save(fig, path)


def map_coarse(arr, grid, title, path, cmap, label, vmin=None, vmax=None, synthetic=False,
               footer=""):
    fig, ax = plt.subplots(figsize=(7.6, 7))
    cm = plt.get_cmap(cmap).with_extremes(bad="white")
    xmin, ymin, xmax, ymax = grid.bounds
    H, W = arr.shape
    # each coarse cell is f x f pixels; the last row/col may extend past the AOI edge
    f_r, f_c = int(np.ceil(grid.height / H)), int(np.ceil(grid.width / W))
    ext = (xmin, xmin + W * f_c * grid.res, ymax - H * f_r * grid.res, ymax)
    im = ax.imshow(np.ma.masked_invalid(arr), extent=ext, cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    decorate(ax, grid, title, synthetic)
    _footer(fig, footer)
    return _save(fig, path)


def side_by_side(imgs: List[np.ndarray], titles: List[str], grid, path, suptitle, synthetic=False,
                 legend: Optional[Tuple[Dict, Dict]] = None):
    fig, axes = plt.subplots(1, len(imgs), figsize=(6.2 * len(imgs), 6.4))
    for ax, img, t in zip(np.atleast_1d(axes), imgs, titles):
        ax.imshow(img, extent=_extent(grid), interpolation="nearest")
        decorate(ax, grid, t, synthetic)
        if legend:
            _legend(ax, *legend)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def location_map(grid: Grid, context: Dict, aoi_gj: Dict, rgb, title, path, synthetic=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    drawn = False
    for key, style in (("province", dict(color="#444444", lw=1.0)),
                       ("district", dict(color="#1f77b4", lw=1.2))):
        fc = context.get(key)
        if not fc:
            continue
        for feat in fc["features"]:
            for ring in _rings(feat["geometry"]):
                ring = np.asarray(ring)
                ax.plot(ring[:, 0], ring[:, 1], **style,
                        label=f"{key.title()}: {feat['properties'].get('name', '')}")
                drawn = True
    ring = np.asarray(aoi_gj["features"][0]["geometry"]["coordinates"][0])
    ax.fill(ring[:, 0], ring[:, 1], color="red", alpha=0.5, label="AOI")
    ax.plot(ring[:, 0], ring[:, 1], color="red", lw=1.5)
    if not drawn:
        lon0, lat0, lon1, lat1 = grid.bounds_lonlat()
        ax.set_xlim(lon0 - 0.5, lon1 + 0.5)
        ax.set_ylim(lat0 - 0.5, lat1 + 0.5)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="lower left")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ring[:, 1]))))
    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("Location in Punjab", fontsize=10, fontweight="bold")
    ax2 = axes[1]
    ax2.imshow(rgb, extent=_extent(grid), interpolation="nearest")
    xs, ys = grid.from_lonlat(ring[:, 0], ring[:, 1])
    ax2.plot(xs, ys, color="yellow", lw=2)
    decorate(ax2, grid, "AOI boundary (recent-year Sentinel-2)", synthetic)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def _rings(geom: Dict):
    t = geom.get("type")
    if t == "Polygon":
        return [geom["coordinates"][0]]
    if t == "MultiPolygon":
        return [p[0] for p in geom["coordinates"]]
    if t == "GeometryCollection":
        return [r for g in geom["geometries"] for r in _rings(g)]
    return []


# ------------------------------------------------------------------ charts
def chart_area_bars(ts: pd.DataFrame, path: Path, synthetic=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = ["Vegetation_km2", "Built-up_km2", "Other_km2"]
    colors = [CLASS_COLORS[VEG], CLASS_COLORS[BUILT], CLASS_COLORS[OTHER]]
    x = np.arange(len(ts))
    w = 0.26
    for i, (c, col) in enumerate(zip(cols, colors)):
        bars = ax.bar(x + (i - 1) * w, ts[c], w, color=col, label=c.replace("_km2", ""))
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ts["year"].astype(str))
    ax.set_ylabel("Area (km²)")
    ax.set_title("Land-cover area by year", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if synthetic:
        ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes, ha="center", color="red", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def chart_matrix(df: pd.DataFrame, path: Path, title: str, fmt: str = "{:.2f}",
                 cmap="Blues", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    m = df.values.astype(float)
    im = ax.imshow(m, cmap=cmap)
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels(df.columns, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels(df.index, fontsize=8)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                color="white" if v > m.max() * 0.6 else "black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return _save(fig, path)


def chart_importance(imp: pd.Series, path: Path, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    imp.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title(f"Feature importance ({model_name})", fontweight="bold")
    ax.set_xlabel("Relative importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def chart_model_comparison(all_cv: Dict, path: Path):
    names = list(all_cv)
    oa = [all_cv[n]["overall_accuracy"] for n in names]
    f1 = [all_cv[n]["macro_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.18, oa, 0.36, label="Overall accuracy", color="#4C72B0")
    b2 = ax.bar(x + 0.18, f1, 0.36, label="Macro F1", color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=7)
    ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in names], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Classifier comparison (spatial block cross-validation)", fontweight="bold",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def map_points(rgb, grid, df: pd.DataFrame, path, title, color_col: str, palette: Dict[str, str],
               synthetic=False, label_ids=True):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(0.35 + 0.65 * rgb, extent=_extent(grid), interpolation="nearest")
    x, y = grid.from_lonlat(df["lon"].values, df["lat"].values)
    for key, col in palette.items():
        m = (df[color_col] == key).values
        ax.scatter(x[m], y[m], s=26, c=col, edgecolors="black", linewidths=0.5, label=key,
                   zorder=3)
    if label_ids and "point_id" in df:
        for xx, yy, pid in zip(x, y, df["point_id"]):
            ax.text(xx + 120, yy + 120, str(pid), fontsize=5.5, zorder=4, clip_on=True)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    decorate(ax, grid, title, synthetic)
    return _save(fig, path)


def chart_timeseries(ts: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for c, col in (("Vegetation_km2", CLASS_COLORS[VEG]), ("Built-up_km2", CLASS_COLORS[BUILT]),
                   ("Other_km2", CLASS_COLORS[OTHER])):
        ax.plot(ts["year"], ts[c], marker="o", color=col, lw=2, label=c.replace("_km2", ""))
        for x, y in zip(ts["year"], ts[c]):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    ax.set_xticks(ts["year"])
    ax.set_ylabel("Area (km²)")
    ax.set_title("Land-cover trend", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)
