"""GeoTIFF read/write helpers (rasterio)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from .aoi import Grid


def write_stack(path, arr: np.ndarray, grid: Grid, band_names: List[str],
                nodata: Optional[float] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None]
    profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=arr.shape[0],
                   dtype=str(arr.dtype), crs=grid.crs, transform=grid.transform,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                   predictor=2 if np.issubdtype(arr.dtype, np.integer) else 3)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def read_stack(path) -> Tuple[Dict[str, np.ndarray], Grid]:
    with rasterio.open(path) as src:
        arr = src.read()
        names = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        t = src.transform
        grid = Grid(crs=src.crs.to_string(), x0=t.c, y0=t.f, width=src.width,
                    height=src.height, res=t.a)
    return {n: arr[i] for i, n in enumerate(names)}, grid


def write_class_raster(path, arr: np.ndarray, grid: Grid, colors: Dict[int, str],
                       description: str, nodata: int = 255) -> None:
    """Single-band uint8 GeoTIFF with an embedded colour table (opens styled in QGIS)."""
    write_stack(path, arr.astype(np.uint8), grid, [description], nodata=nodata)
    cmap = {}
    for k, hexc in colors.items():
        h = hexc.lstrip("#")
        cmap[int(k)] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    with rasterio.open(path, "r+") as dst:
        dst.write_colormap(1, cmap)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
