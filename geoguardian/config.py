"""Project configuration for Punjab GeoGuardian.

Everything that controls the analysis lives here, so a run is fully described by
one JSON file (see ``config/lahore_raiwind.json``). Change the AOI, years or
season there - no code edits are needed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Land classes (PDF section 7) - IDs are fixed by the assignment.
# ---------------------------------------------------------------------------
VEG, BUILT, OTHER = 0, 1, 2
CLASS_NAMES: Dict[int, str] = {
    VEG: "Vegetation / Agriculture",
    BUILT: "Built-up / Urban",
    OTHER: "Other",
}
CLASS_SHORT: Dict[int, str] = {VEG: "Vegetation", BUILT: "Built-up", OTHER: "Other"}
# PDF section 16 colour guidance: green / gray / yellow-brown / bright red.
CLASS_COLORS: Dict[int, str] = {VEG: "#2E9E3A", BUILT: "#6B6B6B", OTHER: "#D8B365"}
CHANGE_COLOR = "#FF0000"
NODATA_CLASS = 255

# Sentinel-2 bands used (10 m: B2 B3 B4 B8, 20 m: B11 B12 resampled to 10 m).
S2_BANDS: List[str] = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_LABELS = {
    "B2": "Blue", "B3": "Green", "B4": "Red", "B8": "NIR",
    "B11": "SWIR1", "B12": "SWIR2",
}
# Model features: the PDF's suggested set (R, G, B, NIR, SWIR, NDVI, NDBI)
# plus SWIR2 and MNDWI, which help separate water / bare soil ("Other").
FEATURES: List[str] = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "MNDWI"]

# Approximate centres of the PDF's suggested AOIs (section 5). Used with
# ``aoi_mode = "rectangle"``; switch to ``aoi_mode = "district"`` to use the
# official FAO GAUL district polygon instead.
AOI_PRESETS: Dict[str, Dict] = {
    "lahore_raiwind": {
        "aoi_name": "Lahore peri-urban expansion zone near Raiwind Road",
        "center_lat": 31.400, "center_lon": 74.250, "width_km": 15.0, "height_km": 15.0,
        "district_name": "Lahore",
    },
    "sheikhupura": {"aoi_name": "Sheikhupura - Lahore corridor", "center_lat": 31.700,
                    "center_lon": 74.050, "width_km": 15.0, "height_km": 15.0,
                    "district_name": "Sheikhupura"},
    "kasur": {"aoi_name": "Kasur north (Lahore - Kasur road belt)", "center_lat": 31.180,
              "center_lon": 74.400, "width_km": 15.0, "height_km": 15.0, "district_name": "Kasur"},
    "faisalabad": {"aoi_name": "Faisalabad peri-urban belt", "center_lat": 31.420,
                   "center_lon": 73.080, "width_km": 15.0, "height_km": 15.0,
                   "district_name": "Faisalabad"},
    "multan": {"aoi_name": "Multan peri-urban belt", "center_lat": 30.190,
               "center_lon": 71.470, "width_km": 15.0, "height_km": 15.0, "district_name": "Multan"},
    "rawalpindi": {"aoi_name": "Rawalpindi south-east expansion zone", "center_lat": 33.560,
                   "center_lon": 73.120, "width_km": 15.0, "height_km": 15.0,
                   "district_name": "Rawalpindi"},
    "gujranwala": {"aoi_name": "Gujranwala peri-urban belt", "center_lat": 32.160,
                   "center_lon": 74.190, "width_km": 15.0, "height_km": 15.0,
                   "district_name": "Gujranwala"},
    "sialkot": {"aoi_name": "Sialkot peri-urban belt", "center_lat": 32.490,
                "center_lon": 74.530, "width_km": 15.0, "height_km": 15.0, "district_name": "Sialkot"},
    "bahawalpur": {"aoi_name": "Bahawalpur peri-urban belt", "center_lat": 29.390,
                   "center_lon": 71.680, "width_km": 15.0, "height_km": 15.0,
                   "district_name": "Bahawalpur"},
    "sahiwal": {"aoi_name": "Sahiwal peri-urban belt", "center_lat": 30.670,
                "center_lon": 73.110, "width_km": 15.0, "height_km": 15.0, "district_name": "Sahiwal"},
}


@dataclass
class ProjectConfig:
    # --- People / report ----------------------------------------------------
    project_title: str = "Punjab GeoGuardian - AI-Based Monitoring of Urban Expansion and Farmland Loss"
    author_name: str = "Hamza Nadeem"
    submitted_to: str = "Omar Farooq - Associate AI/ML Engineer"
    organisation: str = "AIRI Team PITB - AI Internship Task 3"

    # --- Area of Interest (PDF section 5) ------------------------------------
    aoi_key: str = "lahore_raiwind"
    aoi_name: str = "Lahore peri-urban expansion zone near Raiwind Road"
    aoi_description: str = (
        "South-western peri-urban belt of Lahore along the Raiwind Road corridor, south of "
        "Thokar Niaz Baig. The area mixes established housing, new housing schemes, roads, "
        "brick kilns and wheat/rice farmland, which makes it a typical example of Lahore's "
        "outward growth onto agricultural land."
    )
    aoi_mode: str = "rectangle"          # "rectangle" or "district"
    center_lat: float = 31.400
    center_lon: float = 74.250
    width_km: float = 15.0
    height_km: float = 15.0
    district_name: str = "Lahore"         # used when aoi_mode == "district"
    province_name: str = "Punjab"
    pixel_size_m: float = 10.0

    # --- Time (PDF section 11) -----------------------------------------------
    baseline_year: int = 2019
    recent_year: int = 2025
    extra_years: List[int] = field(default_factory=lambda: [2022])  # optional time series
    # Same season every year: Feb-Mar = rabi wheat is green, low cloud, low smog.
    season_start: str = "02-01"
    season_end: str = "03-31"
    max_cloud_pct: float = 20.0           # scene-level CLOUDY_PIXEL_PERCENTAGE filter
    cloud_score_threshold: float = 0.60   # Cloud Score+ cs_cdf; pixels below are masked
    min_images: int = 4                   # widen the window if fewer clear scenes exist
    max_window_extension_days: int = 45

    # --- Training samples (PDF section 14) -----------------------------------
    samples_per_class_per_year: int = 200  # PDF minimum is 50 per class
    min_samples_per_class: int = 50
    sample_spacing_m: float = 100.0
    spatial_blocks: int = 5                # 5x5 blocks -> samples spread over the AOI
    veg_ndvi_min: float = 0.35             # purity filters for reference-assisted samples
    built_ndvi_max: float = 0.30

    # --- Classification (PDF section 13) -------------------------------------
    models: List[str] = field(default_factory=lambda: ["random_forest", "xgboost"])
    cv_folds: int = 5
    majority_filter: bool = True           # 3x3 majority filter removes salt-and-pepper noise
    # Rule-based (Option A) starting thresholds; tuned on training samples.
    rule_ndvi_veg: float = 0.35
    rule_ndbi_built: float = -0.05
    rule_mndwi_water: float = 0.10

    # --- Change detection (PDF section 15) -----------------------------------
    require_spectral_change: bool = True   # conversion must also show NDVI drop or NDBI rise
    min_ndvi_drop: float = 0.10
    min_ndbi_rise: float = 0.05
    min_patch_pixels: int = 5              # minimum mapping unit (5 px = 0.05 ha)
    hotspot_cell_km: float = 1.0
    heat_cell_km: float = 0.5

    # --- Validation (PDF section 18) -----------------------------------------
    validation_points_per_stratum: int = 15  # 4 strata -> 60 points (PDF minimum 50)
    validation_min_distance_to_training_m: float = 100.0
    validation_min_spacing_m: float = 200.0
    min_error_cases: int = 15

    # --- Misc ---------------------------------------------------------------
    random_seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs"
    synthetic: bool = False                 # True only for the offline test scene

    # ------------------------------------------------------------------
    @property
    def years(self) -> List[int]:
        ys = {self.baseline_year, self.recent_year, *self.extra_years}
        return sorted(ys)

    def season_range(self, year: int):
        return f"{year}-{self.season_start}", f"{year}-{self.season_end}"

    @property
    def out(self) -> Path:
        return Path(self.output_dir)

    @property
    def stack_path(self) -> Path:
        return Path(self.data_dir) / f"stack_{self.aoi_key}.tif"

    @property
    def stack_meta_path(self) -> Path:
        return Path(self.data_dir) / f"stack_{self.aoi_key}_meta.json"

    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, key: str, **overrides) -> "ProjectConfig":
        if key not in AOI_PRESETS:
            raise KeyError(f"Unknown AOI preset '{key}'. Options: {sorted(AOI_PRESETS)}")
        params = dict(AOI_PRESETS[key])
        params.update(overrides)
        return cls(aoi_key=key, **params)

    @classmethod
    def from_json(cls, path) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            print(f"[config] ignoring unknown keys: {sorted(unknown)}")
        return cls(**known)

    def to_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict:
        return asdict(self)

    def validate(self) -> None:
        assert self.recent_year > self.baseline_year, "recent_year must be after baseline_year"
        assert self.aoi_mode in ("rectangle", "district")
        assert self.samples_per_class_per_year >= self.min_samples_per_class >= 50, \
            "PDF requires at least 50 training samples per class"
        assert self.validation_points_per_stratum * 4 >= 50, \
            "PDF requires at least 50 validation points"
        assert self.min_error_cases >= 15, "PDF requires at least 15 error-analysis cases"


def default_config(**overrides) -> ProjectConfig:
    cfg = ProjectConfig(**overrides)
    cfg.validate()
    return cfg
