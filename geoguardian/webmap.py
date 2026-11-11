"""Interactive Folium map with layer toggles (PDF section 17, Option B).

Layers: old/recent RGB, NDVI, NDBI, old/recent classification, detected change,
hotspot cells, validation points, AOI and district boundary. A high-resolution
Esri satellite basemap is included, which helps with visual validation. Rasters
are reprojected from UTM to Web-Mercator before overlay, so they line up exactly
with the basemap. The result is one self-contained HTML file.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .aoi import Grid

MAX_OVERLAY_PX = 1400


def _to_webmercator(arr: np.ndarray, grid: Grid, categorical: bool) -> Tuple[np.ndarray, Tuple]:
    """Reproject (H, W[, C]) array to EPSG:3857. Returns array and lat/lon bounds."""
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from pyproj import Transformer

    multi = arr.ndim == 3
    src = np.moveaxis(arr, 2, 0) if multi else arr[None]
    xmin, ymin, xmax, ymax = grid.bounds
    scale = max(grid.width, grid.height) / MAX_OVERLAY_PX
    dst_res = grid.res * max(1.0, scale)
    transform, w, h = calculate_default_transform(grid.crs, "EPSG:3857", grid.width, grid.height,
                                                  xmin, ymin, xmax, ymax, resolution=dst_res)
    fill = 255 if categorical else np.nan
    dst = np.full((src.shape[0], h, w), fill, dtype=src.dtype if categorical else np.float32)
    for i in range(src.shape[0]):
        reproject(src[i].astype(dst.dtype), dst[i], src_transform=grid.transform,
                  src_crs=grid.crs, dst_transform=transform, dst_crs="EPSG:3857",
                  resampling=Resampling.nearest if categorical else Resampling.bilinear,
                  src_nodata=fill, dst_nodata=fill)
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    t = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = t.transform(left, bottom)
    east, north = t.transform(right, top)
    out = np.moveaxis(dst, 0, 2) if multi else dst[0]
    return out, ((south, west), (north, east))


def _png_uri(rgba: np.ndarray) -> str:
    from PIL import Image
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _rgb_layer(rgb: np.ndarray, valid: np.ndarray, grid: Grid):
    rgba = np.dstack([rgb, valid.astype(np.float32)])
    out, bounds = _to_webmercator(rgba.astype(np.float32), grid, categorical=False)
    out = np.nan_to_num(out, nan=0.0)
    return _png_uri(out), bounds


def _cat_layer(cls: np.ndarray, colors: Dict[int, str], grid: Grid, alpha: float = 1.0):
    import matplotlib.colors as mc
    wm, bounds = _to_webmercator(cls.astype(np.uint8), grid, categorical=True)
    rgba = np.zeros(wm.shape + (4,), np.float32)
    for k, c in colors.items():
        m = wm == k
        rgba[m, :3] = mc.to_rgb(c)
        rgba[m, 3] = alpha
    return _png_uri(rgba), bounds


def _cont_layer(arr: np.ndarray, grid: Grid, cmap: str, vmin: float, vmax: float):
    import matplotlib.pyplot as plt
    wm, bounds = _to_webmercator(arr.astype(np.float32), grid, categorical=False)
    norm = np.clip((wm - vmin) / (vmax - vmin), 0, 1)
    rgba = plt.get_cmap(cmap)(np.nan_to_num(norm, nan=0)).astype(np.float32)
    rgba[np.isnan(wm), 3] = 0
    return _png_uri(rgba), bounds


def _legend_html(title: str, items: List[Tuple[str, str]], ramps: List[Tuple[str, str]]) -> str:
    rows = "".join(f"<div><span style='display:inline-block;width:12px;height:12px;"
                   f"background:{c};border:1px solid #333;margin-right:6px'></span>{l}</div>"
                   for c, l in items)
    rr = "".join(f"<div style='margin-top:4px'>{l}<div style='height:8px;width:150px;"
                 f"background:linear-gradient(to right,{g});border:1px solid #333'></div></div>"
                 for l, g in ramps)
    return f"""
    <div style="position:fixed;bottom:28px;left:10px;z-index:9999;background:white;
         padding:8px 10px;border:1px solid #999;border-radius:6px;font-size:12px;
         box-shadow:0 1px 4px rgba(0,0,0,.3)">
      <b>{title}</b>{rows}{rr}
      <div style="margin-top:4px;color:#666;font-size:10px">Estimated change - learning prototype</div>
    </div>"""


def build_web_map(path: Path, grid: Grid, cfg, layers: Dict, aoi_gj: Dict,
                  context: Optional[Dict] = None, hotspots: Optional[pd.DataFrame] = None,
                  points: Optional[pd.DataFrame] = None) -> Path:
    """``layers`` keys: rgb_{year}, valid_{year}, class_{year}, ndvi_{year}, ndbi_{year}, change."""
    import folium
    from branca.element import Element
    from folium.plugins import Fullscreen, MousePosition

    from .config import CHANGE_COLOR, CLASS_COLORS, CLASS_SHORT

    lon, lat = grid.center_lonlat()
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="High-res satellite (Esri)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin, OpenStreetMap contributors", name="Light basemap").add_to(m)

    b, r = cfg.baseline_year, cfg.recent_year

    def add(uri_bounds, name, show=False, opacity=1.0):
        uri, bounds = uri_bounds
        folium.raster_layers.ImageOverlay(image=uri, bounds=[list(bounds[0]), list(bounds[1])],
                                          name=name, opacity=opacity, show=show,
                                          interactive=False, cross_origin=False,
                                          zindex=1).add_to(m)

    for y in sorted({b, r} | {int(k.split("_")[1]) for k in layers if k.startswith("rgb_")}):
        if f"rgb_{y}" in layers:
            add(_rgb_layer(layers[f"rgb_{y}"], layers[f"valid_{y}"], grid),
                f"Sentinel-2 RGB {y}", show=(y == r))
    for y in (b, r):
        add(_cont_layer(layers[f"ndvi_{y}"], grid, "RdYlGn", -0.1, 0.8), f"NDVI {y}")
        add(_cont_layer(layers[f"ndbi_{y}"], grid, "RdYlBu_r", -0.45, 0.25), f"NDBI {y}")
    for y in (b, r):
        add(_cat_layer(layers[f"class_{y}"], CLASS_COLORS, grid, 0.85), f"Land cover {y}")
    add(_cat_layer(layers["change"].astype(np.uint8), {1: CHANGE_COLOR}, grid),
        f"Detected change: agri-to-urban {b}-{r}", show=True)

    folium.GeoJson(aoi_gj, name="AOI boundary",
                   style_function=lambda f: {"color": "#FFD700", "weight": 3, "fillOpacity": 0}
                   ).add_to(m)
    if context and context.get("district"):
        folium.GeoJson(context["district"], name="District boundary", show=False,
                       style_function=lambda f: {"color": "#1f77b4", "weight": 2,
                                                 "fillOpacity": 0, "dashArray": "6"}).add_to(m)
    if hotspots is not None and len(hotspots):
        fg = folium.FeatureGroup(name="Top change hotspots (1 km cells)", show=True)
        for _, h in hotspots.iterrows():
            folium.Marker(
                [h.center_lat, h.center_lon],
                tooltip=f"Hotspot #{int(h['rank'])}: {h.converted_km2 * 100:.1f} ha",
                popup=folium.Popup(f"<b>Hotspot #{int(h['rank'])}</b><br>"
                                   f"Estimated conversion: {h.converted_km2 * 100:.1f} ha "
                                   f"({h.converted_pct_of_cell:.1f}% of the 1 km² cell)<br>"
                                   f"{h.center_lat:.5f}, {h.center_lon:.5f}", max_width=260),
                icon=folium.DivIcon(html=f"<div style='background:#FFD700;border:2px solid #000080;"
                                         f"border-radius:50%;width:22px;height:22px;text-align:center;"
                                         f"font-weight:bold;line-height:19px'>{int(h['rank'])}</div>"),
            ).add_to(fg)
        fg.add_to(m)
    if points is not None and len(points):
        fg = folium.FeatureGroup(name="Validation points", show=False)
        for _, p in points.iterrows():
            status = p.get("status", "")
            col = {"Correct": "#1a9850", "Incorrect": "#d73027"}.get(status, "#3388ff")
            folium.CircleMarker([p.lat, p.lon], radius=5, color="black", weight=1, fill=True,
                                fill_color=col, fill_opacity=0.9,
                                tooltip=f"Point {int(p.point_id)}: predicted {p.predicted_class}"
                                        f"{' | actual ' + str(p.actual) if p.get('actual') else ''}"
                                ).add_to(fg)
        fg.add_to(m)

    items = [(CLASS_COLORS[k], CLASS_SHORT[k]) for k in CLASS_COLORS] + \
            [(CHANGE_COLOR, "Agri-to-urban conversion")]
    m.get_root().html.add_child(Element(_legend_html(
        "Punjab GeoGuardian", items,
        [("NDVI -0.1 to 0.8", "#a50026,#ffffbf,#006837"),
         ("NDBI -0.45 to 0.25", "#313695,#ffffbf,#a50026")])))
    title = (f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:white;"
             f"padding:6px 12px;border-radius:6px;border:1px solid #999;font-size:14px'>"
             f"<b>{cfg.aoi_name}</b> - possible agricultural-to-urban conversion {b} to {r}</div>")
    m.get_root().html.add_child(Element(title))
    Fullscreen().add_to(m)
    MousePosition(position="bottomright", separator=" | ", num_digits=5).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    xmin, ymin, xmax, ymax = grid.bounds_lonlat()
    m.fit_bounds([[ymin, xmin], [ymax, xmax]])
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    print(f"[webmap] saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path
