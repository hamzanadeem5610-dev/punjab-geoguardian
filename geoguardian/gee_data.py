"""Satellite data collection and preprocessing on Google Earth Engine.

PDF section 12 preprocessing steps, in order:
    1. Select AOI                      -> ``build_region``
    2. Load Sentinel-2 collection      -> ``s2_collection`` (COPERNICUS/S2_SR_HARMONIZED)
    3. Filter by date                  -> same season (Feb-Mar) every year
    4. Filter by cloud percentage      -> CLOUDY_PIXEL_PERCENTAGE < 20 %
    5. Median composite                -> ``year_composite``
    6. Clip to AOI                     -> download on the AOI grid (+ AOI mask band)
    7-9. RGB / NDVI / NDBI             -> computed locally in ``indices.py`` and drawn in ``viz.py``

Extra quality steps on top of the PDF minimum:
    * Pixel-level cloud and shadow masking with Google Cloud Score+ and the SCL band,
      so the median is built only from clear observations.
    * The *harmonized* collection removes the +1000 DN offset that ESA introduced in
      processing baseline 04.00 (Jan 2022). Without it, 2025 reflectance would not
      be comparable with 2019.
    * If a season has too few clear scenes, the date window is widened symmetrically.

Only this module needs Earth Engine. Everything downstream runs on the
downloaded GeoTIFF, so the analysis is reproducible and can be tested offline.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from .aoi import Grid, aoi_geojson, grid_from_bounds, grid_from_config, utm_epsg
from .config import S2_BANDS, ProjectConfig

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
WORLDCOVER_2021 = "ESA/WorldCover/v200"
GAUL_L1 = "FAO/GAUL/2015/level1"
GAUL_L2 = "FAO/GAUL/2015/level2"

# Dynamic World labels -> project classes (0 veg, 1 built, 2 other)
DW_FROM = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DW_TO = [2, 0, 0, 0, 0, 0, 1, 2, 2]   # water, trees, grass, flooded veg, crops, shrub, built, bare, snow
# ESA WorldCover classes -> project classes
WC_FROM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_TO = [0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 2]

NODATA_REFLECTANCE = -1
NODATA_LABEL = 255

DATA_SOURCES = [
    "Copernicus Sentinel-2 MSI Level-2A surface reflectance, harmonized (ESA / Copernicus, "
    "via Google Earth Engine: COPERNICUS/S2_SR_HARMONIZED)",
    "Google Cloud Score+ for Sentinel-2 (GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED) - cloud masking",
    "Google Dynamic World V1 (GOOGLE/DYNAMICWORLD/V1) - reference hints for sample selection",
    "ESA WorldCover 10 m 2021 v200 (ESA/WorldCover/v200) - reference hints for sample selection",
    "FAO GAUL 2015 administrative boundaries (FAO/GAUL/2015) - province/district outlines",
]


def _ee():
    import ee  # imported lazily so the rest of the package works without it
    return ee


def init_earth_engine(project: Optional[str] = None) -> None:
    """Authenticate (first time only) and initialise Earth Engine."""
    ee = _ee()
    try:
        ee.Initialize(project=project)
    except Exception:  # not yet authenticated in this runtime
        ee.Authenticate()
        ee.Initialize(project=project)
    print("[gee] Earth Engine initialised", f"(project={project})" if project else "")


# --------------------------------------------------------------------------- AOI
def district_geometry(cfg: ProjectConfig):
    ee = _ee()
    fc = (ee.FeatureCollection(GAUL_L2)
          .filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
          .filter(ee.Filter.eq("ADM1_NAME", cfg.province_name)))
    sel = fc.filter(ee.Filter.eq("ADM2_NAME", cfg.district_name))
    if sel.size().getInfo() == 0:
        names = sorted(set(fc.aggregate_array("ADM2_NAME").getInfo()))
        raise ValueError(f"District '{cfg.district_name}' not found in GAUL. Available: {names}")
    return sel.geometry()


def build_region(cfg: ProjectConfig) -> Tuple[object, Grid, Dict]:
    """Return (ee.Geometry, analysis Grid, AOI GeoJSON)."""
    ee = _ee()
    if cfg.aoi_mode == "rectangle":
        grid = grid_from_config(cfg)
        region = ee.Geometry.Rectangle(list(grid.bounds), proj=grid.crs, geodesic=False)
        return region, grid, aoi_geojson(grid, cfg.aoi_name)

    geom = district_geometry(cfg)
    gj = geom.simplify(maxError=20).getInfo()
    ring = np.array(geom.bounds(maxError=1).getInfo()["coordinates"][0])
    lon_c, lat_c = ring[:, 0].mean(), ring[:, 1].mean()
    crs = f"EPSG:{utm_epsg(lon_c, lat_c)}"
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    # densify the lat/lon bounding box before projecting so the UTM box covers it fully
    lons = np.linspace(ring[:, 0].min(), ring[:, 0].max(), 50)
    lats = np.linspace(ring[:, 1].min(), ring[:, 1].max(), 50)
    LON, LAT = np.meshgrid(lons, lats)
    xs, ys = tr.transform(LON.ravel(), LAT.ravel())
    grid = grid_from_bounds(xs.min(), ys.min(), xs.max(), ys.max(), crs, cfg.pixel_size_m)
    feature = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": cfg.aoi_name}, "geometry": gj}]}
    return geom, grid, feature


def context_boundaries(cfg: ProjectConfig) -> Dict[str, Dict]:
    """Simplified province and district outlines for location maps / overlays."""
    ee = _ee()
    out = {}
    try:
        prov = (ee.FeatureCollection(GAUL_L1).filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
                .filter(ee.Filter.eq("ADM1_NAME", cfg.province_name)))
        out["province"] = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {"name": cfg.province_name},
            "geometry": prov.geometry().simplify(maxError=1500).getInfo()}]}
        dist = (ee.FeatureCollection(GAUL_L2).filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
                .filter(ee.Filter.eq("ADM1_NAME", cfg.province_name))
                .filter(ee.Filter.eq("ADM2_NAME", cfg.district_name)))
        if dist.size().getInfo() > 0:
            out["district"] = {"type": "FeatureCollection", "features": [{
                "type": "Feature", "properties": {"name": cfg.district_name},
                "geometry": dist.geometry().simplify(maxError=100).getInfo()}]}
    except Exception as exc:  # boundaries are optional context only
        print(f"[gee] boundary download skipped: {exc}")
    return out


# ------------------------------------------------------------------- Sentinel-2
def _mask_clouds(cfg: ProjectConfig):
    ee = _ee()

    def fn(img):
        clear = img.select("cs_cdf").gte(cfg.cloud_score_threshold)
        scl = img.select("SCL")
        # SCL: 1 saturated/defective, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow
        scl_ok = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9))
                  .And(scl.neq(10)).And(scl.neq(11)))
        return img.select(S2_BANDS).updateMask(clear.And(scl_ok))
    return fn


def s2_collection(region, start: str, end: str, cfg: ProjectConfig):
    ee = _ee()
    col = (ee.ImageCollection(S2_COLLECTION)
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.max_cloud_pct)))
    return col.linkCollection(ee.ImageCollection(CS_PLUS_COLLECTION), ["cs_cdf"])


def collection_info(col) -> Dict:
    ee = _ee()
    info = ee.Dictionary({
        "dates": col.aggregate_array("system:time_start").map(
            lambda t: ee.Date(t).format("YYYY-MM-dd")),
        "cloud_pct": col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE"),
        "tiles": col.aggregate_array("MGRS_TILE"),
    }).getInfo()
    info["n_scenes"] = len(info["dates"])
    info["n_dates"] = len(set(info["dates"]))
    return info


def year_composite(region, year: int, cfg: ProjectConfig) -> Dict:
    """Cloud-masked median composite for one year, widening the window if needed."""
    ee = _ee()
    start, end = cfg.season_range(year)
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    ext = 0
    while True:
        s = (d0 - dt.timedelta(days=ext)).isoformat()
        e = (d1 + dt.timedelta(days=ext + 1)).isoformat()  # filterDate end is exclusive
        col = s2_collection(region, s, e, cfg)
        n_dates = len(set(col.aggregate_array("system:time_start").map(
            lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()))
        if n_dates >= cfg.min_images or ext >= cfg.max_window_extension_days:
            break
        ext += 15
    info = collection_info(col)
    info.update({"year": year, "start": s, "end": (dt.date.fromisoformat(e)
                                                   - dt.timedelta(days=1)).isoformat(),
                 "window_extended_days": ext})
    if info["n_scenes"] == 0:
        raise RuntimeError(f"No Sentinel-2 scenes for {year} in {s}..{e} under "
                           f"{cfg.max_cloud_pct}% cloud. Widen the season or cloud filter.")
    masked = col.map(_mask_clouds(cfg))
    composite = masked.median()
    obs = masked.select("B4").count().rename("obs")
    dw_col = ee.ImageCollection(DW_COLLECTION).filterBounds(region).filterDate(s, e).select("label")
    # an empty collection would yield a band-less image and break the download -> fall back to 255
    dw = ee.Image(ee.Algorithms.If(dw_col.size().gt(0),
                                   dw_col.mode().remap(DW_FROM, DW_TO),
                                   ee.Image.constant(NODATA_LABEL)))
    print(f"[gee] {year}: {info['n_dates']} dates / {info['n_scenes']} scenes "
          f"({info['start']} -> {info['end']}), window +/-{ext} d")
    return {"composite": composite, "obs": obs, "dw": dw, "info": info}


def build_stack_image(region, cfg: ProjectConfig):
    """One multi-band int16 image holding everything the local analysis needs."""
    ee = _ee()
    bands, names, infos = [], [], {}
    for y in cfg.years:
        c = year_composite(region, y, cfg)
        infos[str(y)] = c["info"]
        refl = c["composite"].select(S2_BANDS).round().unmask(NODATA_REFLECTANCE)
        bands.append(refl.rename([f"y{y}_{b}" for b in S2_BANDS]))
        names += [f"y{y}_{b}" for b in S2_BANDS]
        bands.append(c["obs"].unmask(0).rename(f"y{y}_obs"))
        names.append(f"y{y}_obs")
        bands.append(c["dw"].unmask(NODATA_LABEL).rename(f"y{y}_dw"))
        names.append(f"y{y}_dw")
    wc = (ee.ImageCollection(WORLDCOVER_2021).first().select("Map")
          .remap(WC_FROM, WC_TO).unmask(NODATA_LABEL).rename("wc2021"))
    aoi_mask = ee.Image.constant(1).clip(region).unmask(0).rename("aoi_mask")
    bands += [wc, aoi_mask]
    names += ["wc2021", "aoi_mask"]
    image = ee.Image.cat(bands).toInt16()
    return image, names, infos


# --------------------------------------------------------------------- download
def _tiles(grid: Grid, tile: int):
    for r0 in range(0, grid.height, tile):
        for c0 in range(0, grid.width, tile):
            yield r0, c0, min(tile, grid.height - r0), min(tile, grid.width - c0)


def download_image(image, grid: Grid, band_names: List[str], tile: int = 512,
                   workers: int = 4, retries: int = 5) -> np.ndarray:
    """Download an ee.Image on ``grid`` as (bands, rows, cols) int16 using
    ``ee.data.computePixels`` in tiles (each request stays far below the 48 MB limit)."""
    ee = _ee()
    out = np.full((len(band_names), grid.height, grid.width), NODATA_REFLECTANCE, np.int16)
    tiles = list(_tiles(grid, tile))

    def fetch(t):
        r0, c0, h, w = t
        sub = Grid(grid.crs, grid.x0 + c0 * grid.res, grid.y0 - r0 * grid.res, w, h, grid.res)
        req = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "bandIds": band_names,
               "grid": sub.ee_affine()}
        for attempt in range(retries):
            try:
                return t, ee.data.computePixels(req)
            except Exception as exc:  # transient quota / timeout errors
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt * 3
                print(f"[gee] tile {r0},{c0} failed ({exc}); retry in {wait}s")
                time.sleep(wait)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(fetch, t) for t in tiles]):
            (r0, c0, h, w), arr = fut.result()
            for i, b in enumerate(band_names):
                out[i, r0:r0 + h, c0:c0 + w] = arr[b]
            done += 1
            if done % max(1, len(tiles) // 10) == 0 or done == len(tiles):
                print(f"[gee] downloaded {done}/{len(tiles)} tiles")
    return out


def collect_data(cfg: ProjectConfig, tile: int = 512, workers: int = 4) -> Dict:
    """Full data-collection phase: build composites on GEE and save a local GeoTIFF stack."""
    from .io import write_stack
    region, grid, aoi_gj = build_region(cfg)
    print(f"[gee] AOI '{cfg.aoi_name}': {grid.width} x {grid.height} px @ {grid.res} m "
          f"({grid.crs})")
    image, names, infos = build_stack_image(region, cfg)
    arr = download_image(image, grid, names, tile=tile, workers=workers)
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg.to_dict(), "grid": grid.to_dict(), "band_names": names,
        "image_info": infos, "aoi_geojson": aoi_gj, "context": context_boundaries(cfg),
        "data_sources": DATA_SOURCES, "synthetic": False,
    }
    write_stack(cfg.stack_path, arr, grid, names)
    cfg.stack_meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.stack_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[gee] saved {cfg.stack_path} ({arr.nbytes / 1e6:.0f} MB in memory)")
    return meta


# ---------------------------------------------------------------- live preview
def preview_map(cfg: ProjectConfig, height: str = "600px"):
    """Interactive geemap map with live Earth Engine layers (layer toggles, Google
    satellite basemap). Use it to screenshot the AOI boundary (Week 1 deliverable)
    and to inspect validation points at high resolution."""
    import geemap
    ee = _ee()
    region, grid, _ = build_region(cfg)
    lon, lat = grid.center_lonlat()
    m = geemap.Map(center=(lat, lon), zoom=12, height=height)
    m.add_basemap("SATELLITE")
    for y in [cfg.baseline_year, cfg.recent_year]:
        s, e = cfg.season_range(y)
        comp = s2_collection(region, s, e, cfg).map(_mask_clouds(cfg)).median().clip(region)
        m.addLayer(comp, {"bands": ["B4", "B3", "B2"], "min": 200, "max": 2500, "gamma": 1.1},
                   f"Sentinel-2 RGB {y}", y == cfg.recent_year)
        ndvi = comp.normalizedDifference(["B8", "B4"])
        m.addLayer(ndvi, {"min": -0.2, "max": 0.8, "palette": ["#a50026", "#fdae61", "#ffffbf",
                                                               "#a6d96a", "#006837"]},
                   f"NDVI {y}", False)
    m.addLayer(ee.FeatureCollection([ee.Feature(region)]).style(color="yellow", fillColor="00000000",
                                                                width=3), {}, "AOI boundary")
    return m
