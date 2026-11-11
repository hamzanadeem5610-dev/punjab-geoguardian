"""Automated, plain-language findings (optional advanced feature 82).

All text is generated from the computed numbers using rule templates. No external
AI service is called, so every sentence can be traced to a number. The wording
follows PDF section 24: "possible conversion", "estimated change",
"detected change", and no claims about any specific housing society or
development.
"""
from __future__ import annotations

from typing import Dict, List

from .change import hotspot_table, summarize_patches


def key_findings(R) -> List[str]:
    cfg, sd = R.cfg, R.change.stats_dict
    b, r = cfg.baseline_year, cfg.recent_year
    f = [
        f"Within the {sd['total_aoi_area_km2']:.1f} km² Area of Interest, an estimated "
        f"{sd['agri_to_urban_conversion_km2']:.2f} km² ({sd['agri_to_urban_conversion_ha']:.0f} ha) "
        f"was classified as vegetation/agriculture in {b} and as built-up in {r}. This is a "
        f"possible agricultural-to-urban conversion equal to {sd['percent_aoi_converted']:.2f}% "
        f"of the AOI and {sd[f'percent_of_{b}_vegetation_converted']:.2f}% of the {b} "
        f"vegetation area.",
        f"Mapped built-up area changed from {sd[f'builtup_area_{b}_km2']:.2f} km² in {b} to "
        f"{sd[f'builtup_area_{r}_km2']:.2f} km² in {r} (net {sd['net_builtup_growth_km2']:+.2f} km²"
        + (f", {sd['builtup_growth_percent']:+.1f}%" if sd.get("builtup_growth_percent") is not None
           else "") + ").",
        f"Mapped vegetation/agriculture changed from {sd[f'vegetation_area_{b}_km2']:.2f} km² to "
        f"{sd[f'vegetation_area_{r}_km2']:.2f} km² (net {sd['net_vegetation_change_km2']:+.2f} km²).",
    ]
    if R.validation:
        v, c = R.validation, R.validation["change"]
        prov = " (provisional: not all points have been checked manually yet)" if v["provisional"] else ""
        f.append(f"Validation on {v['n_points']} checked points: {v['correct']} correct, "
                 f"{v['incorrect']} incorrect, approximate accuracy {100 * v['overall_accuracy']:.1f}% "
                 f"(area-weighted {100 * v['area_weighted_overall_accuracy']:.1f}%){prov}.")
        f.append(f"After correcting for the observed map errors, the conversion area is estimated at "
                 f"{c['error_adjusted_conversion_km2']:.2f} km² (95% CI "
                 f"{c['ci95_low_km2']:.2f}-{c['ci95_high_km2']:.2f} km²).")
    f += summarize_patches(R.change)
    return f


def hotspot_sentences(R, top: int = 5) -> List[str]:
    h = hotspot_table(R.change, R.scene.grid, top)
    if h.empty:
        return ["No conversion hotspots were detected."]
    s = [f"Hotspot #{int(x['rank'])}: about {x.converted_km2 * 100:.1f} ha of possible conversion "
         f"({x.converted_pct_of_cell:.1f}% of the 1 km² cell) in the {x.location} "
         f"[{x.center_lat:.4f}°N, {x.center_lon:.4f}°E]." for _, x in h.iterrows()]
    share = 100 * R.change.hotspot_cells.converted_km2.head(top).sum() / max(
        R.change.hotspot_cells.converted_km2.sum(), 1e-9)
    s.append(f"The top {top} one-square-kilometre cells hold {share:.0f}% of all detected conversion, "
             f"so the change is {'concentrated in a few clusters' if share > 40 else 'spread across many locations'}.")
    return s


def answers_to_questions(R) -> Dict[str, str]:
    cfg, sd = R.cfg, R.change.stats_dict
    b, r = cfg.baseline_year, cfg.recent_year
    hs = hotspot_sentences(R, 3)
    return {
        "1. How much land has changed from vegetation/agriculture to built-up area?":
            f"An estimated {sd['agri_to_urban_conversion_km2']:.2f} km² "
            f"({sd['agri_to_urban_conversion_ha']:.0f} ha), {sd['percent_aoi_converted']:.2f}% of the "
            f"AOI, between {b} and {r}."
            + (f" With validation errors taken into account, the estimate is "
               f"{R.validation['change']['error_adjusted_conversion_km2']:.2f} ± "
               f"{R.validation['change']['ci95_km2']:.2f} km²." if R.validation else ""),
        "2. Where are the major change hotspots?": " ".join(hs),
        "3. Can the results be visualized on a map?":
            "Yes: static maps (RGB, NDVI, NDBI, land cover, change, hotspots), an interactive "
            "Folium map with layer toggles, and a Streamlit dashboard. Conversion polygons "
            "are exported as GeoJSON for QGIS / Google Earth.",
        "4. Can the project be extended for district-level monitoring?":
            "Yes. The AOI is one setting in the config: set aoi_mode='district' and "
            "district_name (e.g. 'Lahore', 'Kasur') and the same pipeline runs on the official "
            "FAO GAUL district polygon. Downloads are tiled, so larger areas work without "
            "code changes. Running it for several districts in a loop allows cross-district "
            "comparison.",
    }


def observations(R) -> List[str]:
    """Short observations for the dashboard."""
    sd = R.change.stats_dict
    b, r = R.cfg.baseline_year, R.cfg.recent_year
    obs = [f"Possible agri-to-urban conversion: {sd['agri_to_urban_conversion_km2']:.2f} km² "
           f"({sd['percent_aoi_converted']:.2f}% of AOI).",
           f"Built-up area {b}: {sd[f'builtup_area_{b}_km2']:.1f} km² -> {r}: "
           f"{sd[f'builtup_area_{r}_km2']:.1f} km².",
           f"{sd['low_confidence_conversion_km2']:.2f} km² of the detected conversion has low model "
           f"confidence and should be treated with caution."]
    obs += hotspot_sentences(R, 3)[:3]
    if R.validation:
        obs.append(f"Validation accuracy ≈ {100 * R.validation['overall_accuracy']:.0f}% on "
                   f"{R.validation['n_points']} points"
                   + (" (provisional)." if R.validation["provisional"] else "."))
    return obs


def policy_recommendations(R) -> List[str]:
    return [
        "Use the hotspot list as a screening tool: send field teams or high-resolution imagery "
        "reviews to the top-ranked cells first, instead of surveying the whole area.",
        "Repeat the analysis every year in the same season to build an early-warning record of "
        "farmland loss; one model run for a district takes minutes once set up.",
        "Overlay the detected change with approved master-plan / land-use zoning and approved "
        "scheme boundaries. Only that comparison, not this model, can show whether a change "
        "was planned or unplanned.",
        "Combine vegetation loss with built-up gain (the urban heat-risk proxy) to prioritise "
        "green-space and tree-planting interventions in fast-growing cells.",
        "Treat all numbers as estimates with the stated uncertainty. They support, but do not "
        "replace, official surveys and legal processes.",
    ]


def linkedin_markdown(R) -> str:
    cfg, sd = R.cfg, R.change.stats_dict
    b, r = cfg.baseline_year, cfg.recent_year
    v = R.validation
    acc = (f"{v['n_points']} points checked, {v['correct']} correct and {v['incorrect']} incorrect "
           f"(approximate accuracy {100 * v['overall_accuracy']:.1f}%, area-weighted "
           f"{100 * v['area_weighted_overall_accuracy']:.1f}%, kappa {v['kappa']:.2f})"
           + (" - PROVISIONAL until all points are manually checked" if v["provisional"] else "")
           ) if v else "pending"
    common = ", ".join(f"{k} ({n})" for k, n in sorted(
        R.errors["reason_counts"].items(), key=lambda kv: -kv[1])[:3]) if R.errors else "n/a"
    hs = hotspot_sentences(R, 3)
    model = R.model.name.replace("_", " ").title()
    imgs = R.scene.meta.get("image_info", {})
    n_per_class = ", ".join(f"{k}: {v}" for k, v in
                            R.samples[R.samples.keep].groupby("class_name").size().items())

    def win(y):
        i = imgs.get(str(y), {})
        return f"{i.get('start', '?')} to {i.get('end', '?')} ({i.get('n_dates', '?')} image dates)"

    return f"""# Punjab GeoGuardian - Detecting Urban Expansion and Farmland Loss Using Satellite Imagery

**{cfg.author_name}** | {cfg.organisation}

## 1. Project Overview
I built a GeoAI prototype that uses Sentinel-2 satellite imagery to detect urban expansion and
possible farmland loss in the {cfg.aoi_name}, Punjab. The system compares {b} with {r}, computes
vegetation (NDVI) and built-up (NDBI) indices, classifies every 10 m pixel as vegetation/agriculture,
built-up or other with a {model} model, and highlights where green or agricultural land may have
turned into built-up land. The outputs are maps, area statistics, ranked change hotspots, an
interactive layer-toggle map, a Streamlit dashboard and an automated PDF report.

## 2. Tools and Technologies Used
Python, Google Colab, Google Earth Engine, Sentinel-2 (harmonized L2A), Cloud Score+, Geemap, Folium,
NumPy, Pandas, Matplotlib, Scikit-learn (Random Forest), XGBoost, Rasterio, SciPy, Streamlit, ReportLab.

## 3. Area and Data Selection
- **AOI:** {cfg.aoi_name} ({sd['total_aoi_area_km2']:.0f} km²) - {cfg.aoi_description}
- **Years compared:** {b} and {r}{(' (plus ' + ', '.join(map(str, cfg.extra_years)) + ' for the trend)') if cfg.extra_years else ''}
- **Satellite source:** Copernicus Sentinel-2 MSI Level-2A (10 m)
- **Date ranges:** {b}: {win(b)}; {r}: {win(r)} - same Feb-Mar season every year so crop stage matches
- **Cloud filter:** scenes < {cfg.max_cloud_pct:.0f}% cloud, plus pixel-level Cloud Score+ masking

## 4. Methodology
AOI selection -> Sentinel-2 loading -> date and cloud filtering -> cloud-masked median composites ->
RGB, NDVI, NDBI (+MNDWI) -> training locations ({n_per_class}; minimum 50 per class), spread over the AOI and
kept separate from validation points -> Rule-based baseline vs Random Forest vs XGBoost (spatial
cross-validation) -> classification of both years -> post-classification change detection with
spectral confirmation -> area calculation -> hotspot ranking -> maps, dashboard and report.

## 5. Results
- Total AOI area: **{sd['total_aoi_area_km2']:.2f} km²**
- Vegetation / agriculture in {b}: **{sd[f'vegetation_area_{b}_km2']:.2f} km²**
- Built-up area in {r}: **{sd[f'builtup_area_{r}_km2']:.2f} km²** (was {sd[f'builtup_area_{b}_km2']:.2f} km² in {b})
- Estimated agriculture-to-urban conversion: **{sd['agri_to_urban_conversion_km2']:.2f} km²** ({sd['agri_to_urban_conversion_ha']:.0f} ha)
- Percentage of AOI converted: **{sd['percent_aoi_converted']:.2f}%**
- Main hotspots: {' '.join(hs)}

## 6. Validation
{acc}. Common confusion areas: {common}.

## 7. Error Analysis
I analysed {len(R.errors['table']) if R.errors else 0} weak or incorrect cases. The main reasons were {common}.
Dry or harvested fields can look like built-up land in the SWIR bands; tree-covered housing can look like
vegetation; and small buildings are mixed with their surroundings at 10 m resolution. Improvements:
{chr(10).join('- ' + i for i in R.errors['improvements'][:3]) if R.errors else ''}

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
farmland loss by comparing Sentinel-2 imagery of the {cfg.aoi_name} from {b} and {r}.
It estimates about {sd['agri_to_urban_conversion_km2']:.1f} km² of possible agricultural-to-urban
conversion ({sd['percent_aoi_converted']:.1f}% of the study area) and ranks where it is concentrated.

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
"""
