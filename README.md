# 🛰️ Punjab GeoGuardian
### AI-Based Monitoring of Urban Expansion and Farmland Loss using Sentinel-2 and Machine Learning

**AIRI Team PITB - AI Internship Task 3** · **Author:** Hamza Nadeem · **Supervisor:** Omar Farooq (Associate AI/ML Engineer)

Punjab GeoGuardian compares Sentinel-2 satellite imagery from **2019 and 2025** for a peri-urban
belt of Lahore (Raiwind Road corridor). For every 10 m pixel it computes **NDVI** and **NDBI**,
classifies **land cover** (Vegetation/Agriculture, Built-up/Urban, Other) with **Random Forest /
XGBoost**, and detects **possible agricultural-to-urban conversion**. It also measures the
**area**, ranks **hotspots**, validates the result against **60 manually checked points**, and
presents everything in maps, an **interactive layer-toggle map**, a **Streamlit dashboard**, and an
**automated PDF report**.

> ⚠️ Learning prototype - not an official government monitoring system. All results are
> *possible / estimated* changes, and no claim is made about any specific housing society or development.

---

## Workflow
```
Select AOI → Load Sentinel-2 (GEE) → Date + cloud filter → Cloud-masked median composite → Clip
   → NDVI, NDBI (+MNDWI) → Training samples (≥50/class) → Rule-based vs Random Forest vs XGBoost
   → Classify 2019, 2022, 2025 → Change detection (veg → built-up + spectral check) → Area & hotspots
   → Maps, Folium map, Streamlit dashboard → Validation (60 points) → Error analysis → PDF report
```

| | |
|---|---|
| **AOI** | Lahore peri-urban expansion zone near Raiwind Road - 15 km × 15 km (225 km²), centre 31.40°N 74.25°E |
| **Years** | 2019 vs 2025 (plus 2022 for the trend), same Feb-Mar season every year |
| **Satellite** | Sentinel-2 L2A harmonized, 10 m (`COPERNICUS/S2_SR_HARMONIZED`) |
| **Cloud handling** | Scene cloud < 20 % + Cloud Score+ and SCL pixel masking + median composite |
| **Indices** | NDVI = (NIR−Red)/(NIR+Red) · NDBI = (SWIR−NIR)/(SWIR+NIR) · MNDWI = (Green−SWIR)/(Green+SWIR) |
| **Classes** | 0 Vegetation/Agriculture · 1 Built-up/Urban · 2 Other |
| **Models** | Option A rule-based (tuned thresholds) · Option B Random Forest & XGBoost (spatial block CV) |
| **Validation** | 60 stratified, blind, manually checked points; accuracy, kappa, error-adjusted area ± 95 % CI |

## Results
After a run, the numbers are filled in automatically in
`outputs/report/LINKEDIN_PORTFOLIO.md` and in the final report PDF
`outputs/report/Punjab_GeoGuardian_Final_Report_Hamza_Nadeem.pdf`:
total AOI area, vegetation area 2019, built-up area 2025, estimated conversion (km², % of AOI),
hotspots and validation accuracy. Key maps:

| Change map | Land cover before / after |
|---|---|
| `outputs/maps/10_change_detection_2019_2025.png` | `outputs/maps/17_landcover_before_after.png` |

---

## Quick start (Google Colab - recommended)
1. **Earth Engine access (one-time):** sign up at <https://earthengine.google.com>, create or choose a
   Google Cloud project, and enable the *Earth Engine API* for it (non-commercial / research use is free).
2. Upload this folder to Google Drive as `MyDrive/punjab-geoguardian` (or push it to GitHub and set `REPO_URL`).
3. Open `notebooks/Punjab_GeoGuardian.ipynb` in Colab, set `EE_PROJECT = "your-project-id"` and
   choose **Runtime → Run all**. The first run takes about 15-25 min; most of that is the Earth Engine download.
4. In **part 8**, label the 60 validation points with the widget (about 30-40 min), then run the remaining cells.
5. Collect the deliverables from `outputs/` (also saved on your Drive).

### Command line (local)
```bash
pip install -r requirements.txt
earthengine authenticate
python scripts/run_pipeline.py --config config/lahore_raiwind.json --download --ee-project YOUR_PROJECT
# after labelling outputs/validation/validation_points.csv:
python scripts/run_pipeline.py --config config/lahore_raiwind.json --zip
streamlit run app/streamlit_app.py           # dashboard
python -m pytest -q tests                     # 14 tests, run offline on a synthetic scene
python scripts/run_pipeline.py --synthetic    # offline demo (watermarked, not real results)
```

### Change the area (district-level monitoring)
Edit `config/lahore_raiwind.json`:
* another rectangle: `center_lat`, `center_lon`, `width_km`, `height_km` (presets for Sheikhupura, Kasur,
  Faisalabad, Multan, Rawalpindi, Gujranwala, Sialkot, Bahawalpur and Sahiwal are in `geoguardian/config.py`);
* a whole district: `"aoi_mode": "district", "district_name": "Kasur"` (official FAO GAUL boundary).
  Downloads are tiled, so large districts work without code changes.

---

## Project structure
```
punjab-geoguardian/
├── notebooks/Punjab_GeoGuardian.ipynb   # end-to-end Colab notebook
├── geoguardian/                         # Python package
│   ├── config.py          # AOI, years, season, thresholds (one place for all settings)
│   ├── aoi.py             # AOI + exact 10 m UTM analysis grid
│   ├── gee_data.py        # Earth Engine: Sentinel-2 composites, cloud masks, tiled download
│   ├── indices.py         # NDVI, NDBI, MNDWI
│   ├── sampling.py        # training samples (≥50/class) + stratified validation points
│   ├── classify.py        # rule-based, Random Forest, XGBoost, spatial CV, majority filter
│   ├── change.py          # conversion, area table, transition matrix, hotspots, heat-risk proxy
│   ├── validation.py      # chip sheets, accuracy, kappa, Olofsson error-adjusted area
│   ├── labeler.py         # Colab widget for manual point labelling
│   ├── error_analysis.py  # ≥15 cases with suggested error types/reasons + improvements
│   ├── viz.py / webmap.py # static maps / interactive Folium map
│   ├── export.py          # GeoJSON, CSV, GeoTIFF
│   ├── summary.py / report.py   # findings text, LinkedIn section, PDF report
│   ├── pipeline.py        # orchestration (GeoGuardian.run_all)
│   └── synthetic.py       # synthetic scene for offline tests
├── app/streamlit_app.py   # dashboard
├── scripts/run_pipeline.py
├── config/lahore_raiwind.json
├── tests/                 # unit + acceptance tests (every PDF minimum is asserted)
└── docs/                  # requirement checklist, validation guide
```

## Outputs (`outputs/`)
| Folder | Contents |
|---|---|
| `maps/` | 01 AOI · 02-03 RGB 2019/2025 · 04-05 NDVI · 06-07 NDBI · 08-09 land cover · 10 change map · 11 change classes · 12 hotspots · 13-14 ΔNDVI/ΔNDBI · 15 heat-risk proxy · 16-17 before/after · 18 rule-based map · 19+ extra year · 20 training samples · 21 validation points |
| `charts/` | area by year, trend, transition matrix, model comparison, feature importance, confusion matrices |
| `tables/` | area statistics, transition matrix, hotspots, patches, validation table, error analysis, model CV (CSV) |
| `vectors/` | conversion polygons, hotspot cells, validation and training points, AOI (GeoJSON) |
| `geotiff/` | land-cover maps, change map, NDVI/NDBI/confidence rasters (with colour tables) |
| `validation/` | `validation_points.csv` (labels), `validation_chips.pdf`, per-point chips, training-sample review sheet |
| `web/interactive_map.html` | Folium map: toggle RGB, NDVI, NDBI, land cover, change, hotspots, validation points |
| `report/` | **Final report PDF**, `LINKEDIN_PORTFOLIO.md`, `WEEK1_PROGRESS_SUMMARY.md` |

## Method notes (why the numbers can be trusted)
* **Comparable years:** the harmonized collection removes ESA's 2022 reflectance offset; the same season and
  the same colour stretch are used for both years.
* **Honest model evaluation:** spatial block cross-validation (whole 3 km blocks held out) instead of random
  splits, which would overstate accuracy because neighbouring pixels are nearly identical.
* **Fewer false changes:** a conversion must also show an NDVI drop ≥ 0.10 or an NDBI rise ≥ 0.05, and patches
  smaller than 5 pixels are removed.
* **Unbiased validation:** stratified points (15 per class + 15 detected conversions) at least 100 m from training
  samples, labelled blind. Accuracy is also reported area-weighted, and the conversion area is error-adjusted with
  a 95 % confidence interval (Olofsson et al., 2014).
* **Exact areas:** a 10 m UTM grid, so 1 pixel = 100 m² (scale error < 0.05 %).

## Limitations
10 m pixels miss small buildings and mix pixels at urban edges. Harvested or fallow fields can look like
construction. Tree-covered housing can look like vegetation. The result covers only one season per year.
Validation relies on visual interpretation. See section 11 of the report.

## Data sources
Copernicus Sentinel-2 (ESA) · Google Cloud Score+ · Google Dynamic World V1 · ESA WorldCover 2021 · FAO GAUL 2015 -
all accessed through Google Earth Engine.

## Portfolio / LinkedIn
The complete LinkedIn-ready write-up (overview, tools, data, methodology, results, validation, error analysis,
learnings, future work and the post text) is generated with real numbers at
`outputs/report/LINKEDIN_PORTFOLIO.md`.

*Hamza Nadeem - AIRI Team PITB AI Internship*
