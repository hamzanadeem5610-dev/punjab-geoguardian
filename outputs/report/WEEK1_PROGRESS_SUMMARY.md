# Week 1 Review - Punjab GeoGuardian
**Intern:** Hamza Nadeem | **Submitted to:** Omar Farooq - Associate AI/ML Engineer

1. **AOI:** Lahore peri-urban expansion zone near Raiwind Road - boundary screenshot: `maps/01_aoi_location.png` (also `web/interactive_map.html`)
2. **Years and date ranges:**

| Year | Date range | Clear image dates |
|---|---|---|
| 2019 | 2019-02-01 to 2019-03-31 | 11 |
| 2022 | 2022-02-01 to 2022-03-31 | 18 |
| 2025 | 2025-02-01 to 2025-03-31 | 13 |

3. RGB map 2019: `maps/02_rgb_2019.png`
4. RGB map 2025: `maps/03_rgb_2025.png`
5. NDVI map 2019: `maps/04_ndvi_2019.png`
6. NDVI map 2025: `maps/05_ndvi_2025.png`
7. NDBI map 2019: `maps/06_ndbi_2019.png`
8. NDBI map 2025: `maps/07_ndbi_2025.png`
9. Initial classified land-cover maps: `maps/08_landcover_2019.png`, `maps/09_landcover_2025.png`

## Short progress summary
- Sentinel-2 L2A composites built for 2019, 2022, 2025 (same Feb-Mar season,
  scenes < 20% cloud, Cloud Score+ pixel masking, median composite).
- NDVI and NDBI computed for every year. Vegetation shows high NDVI; built-up shows high NDBI and low NDVI.
- 600 training locations generated (>= 50 per class; review sheet in validation/); a random forest
  model selected by spatial cross-validation (macro-F1 0.917).
- Initial land cover: built-up 112.8 km² in 2019 and 118.4 km² in 2025;
  vegetation 83.4 km² in 2019 and 73.1 km² in 2025.
- Next (Week 2): change detection, area statistics, validation with 60 points, error analysis, dashboard, report.
