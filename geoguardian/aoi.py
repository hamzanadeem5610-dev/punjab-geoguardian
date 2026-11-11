"""Area-of-Interest geometry and the analysis pixel grid.

The analysis runs on a north-up 10 m grid in the local UTM zone (EPSG:326xx).
Working in UTM keeps every pixel at exactly 100 m^2 (UTM scale error in the AOI is
under 0.05 %), so area statistics are simple pixel counts and need no
latitude correction.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Grid:
    crs: str          # e.g. "EPSG:32643"
    x0: float         # left edge (m)
    y0: float         # top edge (m)
    width: int        # columns
    height: int       # rows
    res: float = 10.0

    # ------------------------------------------------------------------ basics
    @property
    def epsg(self) -> int:
        return int(self.crs.split(":")[1])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self):
        from rasterio.transform import Affine
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y0)

    @property
    def pixel_area_km2(self) -> float:
        return (self.res * self.res) / 1e6

    @property
    def pixel_area_ha(self) -> float:
        return (self.res * self.res) / 1e4

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Grid":
        return cls(**d)

    # --------------------------------------------------------- coordinates
    def _to_ll(self) -> Transformer:
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

    def _from_ll(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def pixel_center_xy(self, rows, cols):
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        x = self.x0 + (cols + 0.5) * self.res
        y = self.y0 - (rows + 0.5) * self.res
        return x, y

    def rowcol(self, x, y):
        col = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        row = np.floor((self.y0 - np.asarray(y)) / self.res).astype(int)
        return row, col

    def to_lonlat(self, x, y):
        return self._to_ll().transform(np.asarray(x), np.asarray(y))

    def from_lonlat(self, lon, lat):
        return self._from_ll().transform(np.asarray(lon), np.asarray(lat))

    def rowcol_to_lonlat(self, rows, cols):
        x, y = self.pixel_center_xy(rows, cols)
        return self.to_lonlat(x, y)

    def lonlat_to_rowcol(self, lon, lat):
        x, y = self.from_lonlat(lon, lat)
        return self.rowcol(x, y)

    def boundary_lonlat(self, densify: int = 20) -> List[Tuple[float, float]]:
        """Closed ring (lon, lat) of the grid rectangle, densified so it plots
        correctly in any projection."""
        xmin, ymin, xmax, ymax = self.bounds
        t = np.linspace(0, 1, densify, endpoint=False)
        xs = np.concatenate([xmin + t * (xmax - xmin), np.full_like(t, xmax),
                             xmax - t * (xmax - xmin), np.full_like(t, xmin)])
        ys = np.concatenate([np.full_like(t, ymax), ymax - t * (ymax - ymin),
                             np.full_like(t, ymin), ymin + t * (ymax - ymin)])
        lon, lat = self.to_lonlat(xs, ys)
        ring = list(zip(np.round(lon, 7).tolist(), np.round(lat, 7).tolist()))
        ring.append(ring[0])
        return ring

    def bounds_lonlat(self) -> Tuple[float, float, float, float]:
        ring = np.array(self.boundary_lonlat())
        return (ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max())

    def center_lonlat(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bounds
        lon, lat = self.to_lonlat((xmin + xmax) / 2, (ymin + ymax) / 2)
        return float(lon), float(lat)

    def ee_affine(self) -> Dict:
        """Grid description in the format of Earth Engine's computePixels."""
        return {
            "dimensions": {"width": self.width, "height": self.height},
            "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                "shearY": 0, "scaleY": -self.res, "translateY": self.y0},
            "crsCode": self.crs,
        }


def grid_from_center(lat: float, lon: float, width_km: float, height_km: float,
                     res: float = 10.0) -> Grid:
    """Exact width x height rectangle (in metres) centred on a lat/lon point."""
    epsg = utm_epsg(lon, lat)
    crs = f"EPSG:{epsg}"
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    w = int(round(width_km * 1000 / res))
    h = int(round(height_km * 1000 / res))
    x0 = math.floor((cx - w * res / 2) / res) * res
    y0 = math.ceil((cy + h * res / 2) / res) * res
    return Grid(crs=crs, x0=float(x0), y0=float(y0), width=w, height=h, res=res)


def grid_from_bounds(xmin: float, ymin: float, xmax: float, ymax: float, crs: str,
                     res: float = 10.0) -> Grid:
    """Snap projected bounds outward onto the res-metre lattice."""
    x0 = math.floor(xmin / res) * res
    y0 = math.ceil(ymax / res) * res
    x1 = math.ceil(xmax / res) * res
    y1 = math.floor(ymin / res) * res
    return Grid(crs=crs, x0=float(x0), y0=float(y0), width=int(round((x1 - x0) / res)),
                height=int(round((y0 - y1) / res)), res=res)


def grid_from_config(cfg) -> Grid:
    return grid_from_center(cfg.center_lat, cfg.center_lon, cfg.width_km, cfg.height_km,
                            cfg.pixel_size_m)


def aoi_geojson(grid: Grid, name: str) -> Dict:
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": name, "area_km2": round(grid.width * grid.height
                                                           * grid.pixel_area_km2, 3)},
            "geometry": {"type": "Polygon", "coordinates": [grid.boundary_lonlat()]},
        }],
    }
