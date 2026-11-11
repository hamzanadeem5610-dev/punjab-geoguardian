# Requirement checklist - AIRI Team PITB AI Internship Task 3

Every requirement in the task PDF is listed below with where it is met. ✅ = built and tested;
👤 = needs your action when you run it (Earth Engine run, manual labelling, publishing).

## Section 4 - Mandatory project rules
| # | Rule | Where | |
|---|---|---|---|
| 5 | Uses satellite imagery | `gee_data.py` (Sentinel-2 L2A) | ✅ |
| 6 | Compares ≥ 2 years | 2019 vs 2025 (+2022) - `config.py` | ✅ |
| 7 | ≥ 1 Area of Interest | Lahore, Raiwind Road corridor, 225 km² | ✅ |
| 8 | Sentinel-2 where possible | `COPERNICUS/S2_SR_HARMONIZED` | ✅ |
| 9 | NDVI and NDBI | `indices.py` (+ MNDWI) | ✅ |
| 10 | Basic land classification | `classify.py` - rules + RF + XGBoost | ✅ |
| 11 | Change detection | `change.py` | ✅ |
| 12 | Approximate changed area | `change.py` stats table + error-adjusted area | ✅ |
| 13 | Visual maps | `viz.py` → `outputs/maps` (21+ maps) | ✅ |
| 14 | Validation with manually checked points | `validation.py`, `labeler.py` - 60 points | 👤 label the points |
| 15 | Error analysis | `error_analysis.py` (≥ 15 cases) | ✅ (review the notes 👤) |
| 16 | Interactive map / dashboard | `web/interactive_map.html` + `app/streamlit_app.py` | ✅ |
| 17 | Final report | `report.py` → `outputs/report/*.pdf` | ✅ |
| 18 | GitHub / Drive submission | project + outputs on Drive; README has GitHub steps | 👤 push / share |
| 19 | LinkedIn-ready portfolio section | `outputs/report/LINKEDIN_PORTFOLIO.md` + report §12 | ✅ (post it 👤) |

## Section 6 - Minimum requirements
| Requirement | Minimum | This project |
|---|---|---|
| AOI | 1 | 1 (any preset / district via config) |
| Years | 2 (2019 & 2025 recommended) | 2019, 2025 + 2022 |
| Satellite | Sentinel-2 | Sentinel-2 L2A harmonized |
| Indices | NDVI, NDBI | NDVI, NDBI, MNDWI |
| Land classes | ≥ 3 | 3 (IDs 0/1/2 exactly as the PDF) |
| Validation points | ≥ 50 | 60 (asserted in `tests/`) |
| Error-analysis cases | ≥ 15 | ≥ 15 (asserted) |
| Output maps | ≥ 8 | 21+ maps + 7 charts (9 required ones asserted) |
| Area calculation | required | ✅ |
| Interactive map / dashboard | required | Folium + Streamlit |
| Final report / GitHub-Drive / LinkedIn | required | ✅ / 👤 / ✅ |

## Section 11-12 - Data collection and preprocessing
| Step | Implementation |
|---|---|
| Low cloud, similar season, same area, clear visibility, minimal haze | Feb-Mar both years; `CLOUDY_PIXEL_PERCENTAGE < 20`; Cloud Score+ ≥ 0.6 + SCL mask; identical AOI grid |
| 20 Select AOI | `build_region()` (rectangle or GAUL district) |
| 21 Load Sentinel-2 collection | `s2_collection()` |
| 22 Filter by date | `filterDate(season)`; window widened automatically if < 4 clear dates |
| 23 Filter by cloud % | `ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)` |
| 24 Median composite | `masked.median()` |
| 25 Clip to AOI | download on the AOI grid + `aoi_mask` band (`.clip(region)`) |
| 26 Visualise RGB | `02_rgb_2019.png`, `03_rgb_2025.png` (shared stretch) |
| 27-28 NDVI / NDBI | `04`-`07` maps |

## Section 13-14 - Classification and training data
* Option A (rule-based, thresholds tuned) ✅ · Option B (Random Forest + XGBoost, features R, G, B, NIR, SWIR1, SWIR2, NDVI, NDBI, MNDWI) ✅ · Option C (U-Net) - optional, listed as future work.
* ≥ 50 samples per class (200 used) ✅; spread over a 5×5 block grid, ≥ 100 m apart ✅; built-up includes buildings, roads and housing; vegetation includes fields and trees; other includes bare land and water ✅; training ≠ validation (≥ 100 m exclusion, tested) ✅; visual review sheet `training_samples_review.pdf` + `keep` column 👤.

## Section 15 - Change detection (all five numbers are in `tables/area_statistics.csv`)
29 Total AOI area ✅ · 30 Vegetation area old year ✅ · 31 Built-up area recent year ✅ · 32 Converted area ✅ · 33 % of AOI ✅ - labelled "Agriculture-to-Urban Conversion".

## Section 16 - Required maps (`outputs/maps/`)
34 `02_rgb_2019` · 35 `03_rgb_2025` · 36 `04_ndvi_2019` · 37 `05_ndvi_2025` · 38 `06_ndbi_2019` · 39 `07_ndbi_2025` · 40 `08_landcover_2019` · 41 `09_landcover_2025` · 42 `10_change_detection_2019_2025` - colours: vegetation green, built-up gray, other yellow-brown, change bright red ✅.

## Section 17 - Interactive map / dashboard
* **Streamlit:** AOI name ✅, year selection (slider) ✅, RGB ✅, NDVI ✅, NDBI ✅, change map ✅, area statistics table ✅, short observations ✅.
* **Folium:** toggle old/recent RGB ✅, old/recent classification ✅, detected change ✅ (+ NDVI/NDBI, hotspots, validation points, AOI, district boundary, hi-res basemap).

## Section 18-19 - Validation and error analysis
* Table columns Point ID / Actual Class / Predicted Class / Correct-Incorrect / Notes ✅; approximate accuracy reported ✅ (+ area-weighted, kappa, producer's/user's accuracy, change accuracy, error-adjusted area).
* ≥ 15 cases with expected result, model result, error type and possible reason ✅; ≥ 3 improvements ✅.

## Section 20 - Week 1 review deliverables (43-52)
`outputs/report/WEEK1_PROGRESS_SUMMARY.md` lists the AOI and boundary screenshot (`01_aoi_location.png` + geemap map in notebook part 2 👤 screenshot), years and date ranges, the 6 RGB/NDVI/NDBI maps, the initial classified maps and a progress summary ✅.

## Section 21 - Final deliverables (53-71)
| # | Deliverable | Location |
|---|---|---|
| 53 | Colab notebook with visible outputs | `notebooks/Punjab_GeoGuardian.ipynb` - 👤 run in Colab and save with outputs |
| 54 | GitHub / Drive link | 👤 README "Quick start" |
| 55 | Python scripts | `geoguardian/`, `scripts/run_pipeline.py` |
| 56 | requirements.txt | ✅ |
| 57 | README | ✅ |
| 58-59 | AOI description, years compared | README, report §3 |
| 60-64 | RGB, NDVI, NDBI, classified maps, change map | `outputs/maps/` |
| 65 | Area statistics table | `outputs/tables/area_statistics.csv`, report §6.4 |
| 66 | Validation table ≥ 50 points | `outputs/tables/validation_table.csv`, report §7 |
| 67 | Error-analysis table ≥ 15 cases | `outputs/tables/error_analysis.csv`, report §8 |
| 68 | Screenshots of final outputs | map PNGs; 👤 screenshot the interactive map and dashboard |
| 69 | Interactive map / dashboard | `outputs/web/interactive_map.html`, `app/streamlit_app.py` |
| 70 | Final report PDF | `outputs/report/Punjab_GeoGuardian_Final_Report_Hamza_Nadeem.pdf` |
| 71 | LinkedIn-ready section | `outputs/report/LINKEDIN_PORTFOLIO.md` |

## Section 23 - Optional advanced features
| # | Feature | Status |
|---|---|---|
| 72 | Streamlit dashboard | ✅ |
| 73 | Folium/Geemap layer toggle | ✅ both |
| 74 | District boundary overlay | ✅ (GAUL, on location map + Folium) |
| 75 | Housing-society boundary overlay | ✗ no open official dataset; deliberately not used (section 24 caution) |
| 76 | Time slider | ✅ dashboard year slider |
| 77 | 2016/2019/2022/2025 comparison | 2019/2022/2025 ✅ - 2016 is not in the harmonized L2A archive (starts 2017), so it is not included |
| 78 | U-Net | ✗ listed as future work |
| 79 | GeoJSON export | ✅ |
| 80 | CSV statistics | ✅ |
| 81 | Automated PDF report | ✅ |
| 82 | Generated summary of findings | ✅ rule-based text generated from the numbers (`summary.py`) |
| 83 | Compare multiple districts | ✅ supported by config (run per district); not run by default |
| 84 | Urban heat-risk analysis | ✅ proxy from NDVI loss + NDBI gain (not temperature) |
| 85 | Policy recommendations | ✅ report §10 |

## Section 24 - Important notes
Prototype disclaimer on every map footer, the report, the dashboard and the LinkedIn text ✅ · wording "possible conversion / detected change / estimated change" ✅ · no named developments (hotspots are described by position and coordinates only) ✅ · data sources and assumptions listed (report §3, §11) ✅ · uncertainty quantified (low-confidence area, 95 % CI) ✅.

## Section 25 - LinkedIn template (10 parts)
All 10 parts are generated with real numbers: Overview, Tools, Area & Data, Methodology, Results, Validation, Error Analysis, What I Learned, Future Improvements and the LinkedIn post ✅. Personalise the wording before posting 👤.
