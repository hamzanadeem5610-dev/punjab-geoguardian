"""GeoJSON / CSV / GeoTIFF exports (optional advanced features 79-80)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from pyproj import Transformer

from .aoi import Grid


def _ring_to_ll(ring, tr):
    xs, ys = zip(*ring)
    lon, lat = tr.transform(np.array(xs), np.array(ys))
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)]


def conversion_geojson(conv: np.ndarray, grid: Grid, path: Path, years) -> int:
    """Vectorise conversion pixels into polygons (EPSG:4326) with area in hectares."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    feats = []
    for i, (geom, val) in enumerate(shapes(conv.astype(np.uint8), mask=conv,
                                           transform=grid.transform, connectivity=8)):
        area_ha = shape(geom).area / 1e4  # area computed in UTM metres
        coords = [_ring_to_ll(r, tr) for r in geom["coordinates"]]
        feats.append({"type": "Feature",
                      "properties": {"id": i + 1, "area_ha": round(area_ha, 3),
                                     "change": f"possible agri-to-urban {years[0]}-{years[1]}"},
                      "geometry": {"type": "Polygon", "coordinates": coords}})
    fc = {"type": "FeatureCollection", "features": feats}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc))
    return len(feats)


def points_geojson(df: pd.DataFrame, path: Path, lon="lon", lat="lat") -> None:
    feats = []
    for _, r in df.iterrows():
        props = {k: (None if (isinstance(v, float) and np.isnan(v)) else
                     (v.item() if hasattr(v, "item") else v))
                 for k, v in r.items() if k not in (lon, lat)}
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [float(r[lon]), float(r[lat])]}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, default=str))


def hotspot_cells_geojson(hot: pd.DataFrame, grid: Grid, cell_km: float, path: Path) -> None:
    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    f = int(round(cell_km * 1000 / grid.res))
    feats = []
    for _, h in hot.iterrows():
        x0 = grid.x0 + h.cell_col * f * grid.res
        y0 = grid.y0 - h.cell_row * f * grid.res
        x1, y1 = x0 + f * grid.res, y0 - f * grid.res
        ring = _ring_to_ll([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], tr)
        feats.append({"type": "Feature",
                      "properties": {"rank": int(h["rank"]), "converted_ha": round(h.converted_km2 * 100, 2),
                                     "converted_pct_of_cell": float(h.converted_pct_of_cell)},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def write_tables(tables: Dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=name in ("transition_matrix_km2",
                                                          "validation_confusion_matrix"))
