# Punjab GeoGuardian - Detecting Urban Expansion and Farmland Loss Using Satellite Imagery

**Hamza Nadeem** | AIRI Team PITB - AI Internship Task 3

## 1. Project Overview
I built a GeoAI prototype that uses Sentinel-2 satellite imagery to detect urban expansion and
possible farmland loss in the Lahore peri-urban expansion zone near Raiwind Road, Punjab. The system compares 2019 with 2025, computes
vegetation (NDVI) and built-up (NDBI) indices, classifies every 10 m pixel as vegetation/agriculture,
built-up or other with a Random Forest model, and highlights where green or agricultural land may have
turned into built-up land. The outputs are maps, area statistics, ranked change hotspots, an
interactive layer-toggle map, a Streamlit dashboard and an automated PDF report.

## 2. Tools and Technologies Used
Python, Google Colab, Google Earth Engine, Sentinel-2 (harmonized L2A), Cloud Score+, Geemap, Folium,
NumPy, Pandas, Matplotlib, Scikit-learn (Random Forest), XGBoost, Rasterio, SciPy, Streamlit, ReportLab.

## 3. Area and Data Selection
- **AOI:** Lahore peri-urban expansion zone near Raiwind Road (225 km²) - South-western peri-urban belt of Lahore along the Raiwind Road corridor, south of Thokar Niaz Baig. The area mixes established housing, new housing schemes, roads, brick kilns and wheat/rice farmland, which makes it a typical example of Lahore's outward growth onto agricultural land.
- **Years compared:** 2019 and 2025 (plus 2022 for the trend)
- **Satellite source:** Copernicus Sentinel-2 MSI Level-2A (10 m)
- **Date ranges:** 2019: 2019-02-01 to 2019-03-31 (11 image dates); 2025: 2025-02-01 to 2025-03-31 (13 image dates) - same Feb-Mar season every year so crop stage matches
- **Cloud filter:** scenes < 20% cloud, plus pixel-level Cloud Score+ masking

## 4. Methodology
AOI selection -> Sentinel-2 loading -> date and cloud filtering -> cloud-masked median composites ->
RGB, NDVI, NDBI (+MNDWI) -> training locations (Built-up: 200, Other: 200, Vegetation: 200; minimum 50 per class), spread over the AOI and
kept separate from validation points -> Rule-based baseline vs Random Forest vs XGBoost (spatial
cross-validation) -> classification of both years -> post-classification change detection with
spectral confirmation -> area calculation -> hotspot ranking -> maps, dashboard and report.

## 5. Results
- Total AOI area: **225.00 km²**
- Vegetation / agriculture in 2019: **83.39 km²**
- Built-up area in 2025: **118.42 km²** (was 112.82 km² in 2019)
- Estimated agriculture-to-urban conversion: **12.31 km²** (1231 ha)
- Percentage of AOI converted: **5.47%**
- Main hotspots: Hotspot #1: about 46.1 ha of possible conversion (46.0% of the 1 km² cell) in the north-western part of the AOI (9.2 km from centre) [31.4627°N, 74.1864°E]. Hotspot #2: about 35.8 ha of possible conversion (35.8% of the 1 km² cell) in the north-western part of the AOI (8.5 km from centre) [31.4537°N, 74.1865°E]. Hotspot #3: about 34.3 ha of possible conversion (34.3% of the 1 km² cell) in the north-western part of the AOI (8.6 km from centre) [31.4628°N, 74.1969°E]. The top 3 one-square-kilometre cells hold 9% of all detected conversion, so the change is spread across many locations.

## 6. Validation
60 points checked, 29 correct and 31 incorrect (approximate accuracy 48.3%, area-weighted 56.6%, kappa 0.11). Common confusion areas: New road / construction site confused with barren land (11), Bare / fallow soil confused with built-up (9), Tree-covered housing area classified as vegetation (9).

## 7. Error Analysis
I analysed 36 weak or incorrect cases. The main reasons were New road / construction site confused with barren land (11), Bare / fallow soil confused with built-up (9), Tree-covered housing area classified as vegetation (9).
Dry or harvested fields can look like built-up land in the SWIR bands; tree-covered housing can look like
vegetation; and small buildings are mixed with their surroundings at 10 m resolution. Improvements:
- Add a separate 'Construction / bare urban' class and a road-network layer (OSM).
- Add more 'Other' training samples on harvested/fallow fields and use a multi-date (e.g. Feb + Apr) composite so fields that green up at any time are recognised.
- Add texture features (e.g. GLCM on NIR) or higher-resolution imagery to detect rooftops under tree canopy.

## 8. What I Learned
- How to use satellite imagery in an AI project, and how to pick an Area of Interest.
- How NDVI identifies vegetation and NDBI identifies built-up surfaces.
- How to build a land-use classifier and evaluate it honestly with spatial cross-validation.
- How to run change detection between two years and turn pixels into area estimates.
- How to validate a map with manually checked points and report its uncertainty.
- How to communicate map-based AI results for public-sector decision support.

## 9. Future Improvements
- Add more years for a full time series (annual 2019-2025).
- Run at district level with official boundaries, and compare districts.
- Use higher-resolution imagery and train a U-Net segmentation model.
- Add more land classes (water, barren, industrial/brick kilns, roads).
- Validate with official land-use data and approved scheme boundaries.
- Add automated policy-summary generation for planners.

## 10. LinkedIn Post
Today, I completed AIRI Team PITB - AI Internship Task 3, where I built an end-to-end GeoAI project
using satellite imagery and machine learning.

For this task, I developed Punjab GeoGuardian, a prototype that detects possible urban expansion and
farmland loss by comparing Sentinel-2 imagery of the Lahore peri-urban expansion zone near Raiwind Road from 2019 and 2025.
It estimates about 12.3 km² of possible agricultural-to-urban
conversion (5.5% of the study area) and ranks where it is concentrated.

The project covered the complete GeoAI workflow:
- Area of Interest selection
- Satellite imagery collection (Google Earth Engine)
- NDVI and NDBI calculation
- Land-use classification (Random Forest / XGBoost)
- Change detection
- Area estimation
- Map visualization and an interactive dashboard
- Validation and error analysis
- Final documentation

This project helped me understand how AI can be used for real-world public-sector challenges such as
urban planning, agricultural land monitoring, and environmental decision support.

Tools used: Python, Google Colab, Google Earth Engine, Sentinel-2, Geemap/Folium, Scikit-learn,
XGBoost and Streamlit.

Key learning: AI becomes more powerful when it is connected with real-world geography, public data,
and decision-making problems.

#GeoAI #RemoteSensing #SatelliteImagery #MachineLearning #UrbanPlanning #Agriculture #Python
#GoogleEarthEngine #AIRI #PITB #AIInternship

---
*Learning prototype - not an official government monitoring system. All figures are estimates of
possible / detected change, and no claim is made about any specific development.*
