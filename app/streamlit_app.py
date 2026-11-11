"""Punjab GeoGuardian - Streamlit dashboard (PDF section 17, Option A).

Run:  streamlit run app/streamlit_app.py
      (optional) GEOGUARDIAN_OUTPUTS=/path/to/outputs streamlit run app/streamlit_app.py

The dashboard reads only the files in ``outputs/``. It needs no Earth Engine
access, so it can be deployed as-is to Streamlit Community Cloud.
"""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GEOGUARDIAN_OUTPUTS", ROOT / "outputs"))

st.set_page_config(page_title="Punjab GeoGuardian", page_icon="🛰️", layout="wide")


@st.cache_data
def load_summary(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


summary_file = OUT / "run_summary.json"
if not summary_file.exists():
    st.error(f"No results found in `{OUT}`. Run the notebook or "
             "`python scripts/run_pipeline.py` first.")
    st.stop()

S = load_summary(str(summary_file))
F = {k: OUT / v for k, v in S["figures"].items()}
sd = S["stats"]
b, r = S["years"]
years = sorted({int(k.split("_")[1]) for k in F if k.startswith("rgb_") and k.split("_")[1].isdigit()})


def table(name):
    p = OUT / "tables" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def show(key, caption=None):
    if key in F and F[key].exists():
        st.image(str(F[key]), caption=caption, width="stretch")
    else:
        st.info(f"Figure '{key}' not available")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛰️ Punjab GeoGuardian")
st.sidebar.markdown(f"**AOI:** {S['aoi']}")
st.sidebar.markdown(f"**Years compared:** {b} vs {r}")
st.sidebar.markdown(f"**Prepared by:** {S['author']}")
st.sidebar.markdown(f"**Model:** {S['model'].replace('_', ' ').title()}")
view_year = st.sidebar.select_slider("Year to display", options=years, value=r)
compare = st.sidebar.checkbox("Compare with another year side by side", value=True)
other_year = st.sidebar.selectbox("Comparison year", [y for y in years if y != view_year], index=0) \
    if compare and len(years) > 1 else None
layer = st.sidebar.radio("Layer", ["RGB", "NDVI", "NDBI", "Land cover"], index=0)
st.sidebar.divider()
st.sidebar.caption("Learning prototype - not an official government monitoring system. "
                   "Results show *possible / estimated* change only.")

if S.get("synthetic"):
    st.warning("⚠️ These outputs come from the SYNTHETIC test scene, not real satellite data.")

# ------------------------------------------------------------------- header
st.title("Punjab GeoGuardian")
st.subheader(f"AI-based monitoring of urban expansion and farmland loss - {S['aoi']}")

c = st.columns(5)
c[0].metric("Total AOI area", f"{sd['total_aoi_area_km2']:.1f} km²")
c[1].metric(f"Vegetation {b}", f"{sd[f'vegetation_area_{b}_km2']:.1f} km²",
            f"{sd['net_vegetation_change_km2']:+.1f} km² by {r}", delta_color="normal")
c[2].metric(f"Built-up {r}", f"{sd[f'builtup_area_{r}_km2']:.1f} km²",
            f"{sd['net_builtup_growth_km2']:+.1f} km² since {b}", delta_color="inverse")
c[3].metric("Agri-to-urban conversion", f"{sd['agri_to_urban_conversion_km2']:.2f} km²",
            f"{sd['agri_to_urban_conversion_ha']:.0f} ha", delta_color="off")
c[4].metric("% of AOI converted", f"{sd['percent_aoi_converted']:.2f} %")

tabs = st.tabs(["🗺️ Maps", "🔴 Change & hotspots", "📊 Statistics", "✅ Validation",
                "🌐 Interactive map", "📥 Downloads"])

key = {"RGB": "rgb", "NDVI": "ndvi", "NDBI": "ndbi", "Land cover": "class"}[layer]
with tabs[0]:
    if other_year:
        a1, a2 = st.columns(2)
        first, second = sorted([view_year, other_year])
        with a1:
            show(f"{key}_{first}", f"{layer} {first}")
        with a2:
            show(f"{key}_{second}", f"{layer} {second}")
    else:
        show(f"{key}_{view_year}", f"{layer} {view_year}")
    with st.expander("What do NDVI and NDBI mean?"):
        st.markdown("- **NDVI = (NIR − Red) / (NIR + Red)**: high values mean healthy vegetation or crops.\n"
                    "- **NDBI = (SWIR − NIR) / (SWIR + NIR)**: high values often mean built-up surfaces "
                    "(but dry bare soil is high too).\n"
                    "- **Land cover**: green = vegetation/agriculture, gray = built-up/urban, "
                    "tan = other (bare land, water, mixed).")

with tabs[1]:
    a1, a2 = st.columns([3, 2])
    with a1:
        show("change", f"Possible agricultural-to-urban conversion {b}-{r} (bright red)")
    with a2:
        st.markdown("#### Short observations")
        for o in S["observations"]:
            st.markdown(f"- {o}")
        hs = table("hotspot_cells")
        if hs is not None and len(hs):
            st.markdown("#### Top hotspots (1 km cells)")
            hs = hs.head(10).copy()
            hs["converted_ha"] = (hs["converted_km2"] * 100).round(1)
            st.dataframe(hs[["rank", "converted_ha", "converted_pct_of_cell", "location", "google_maps"]],
                         hide_index=True, width="stretch",
                         column_config={"google_maps": st.column_config.LinkColumn("satellite view")})
    a1, a2 = st.columns(2)
    with a1:
        show("hotspots", "Hotspot density")
    with a2:
        show("heat", "Urban heat-risk proxy (relative)")

with tabs[2]:
    a1, a2 = st.columns([2, 3])
    with a1:
        st.markdown("#### Area statistics")
        st.dataframe(table("area_statistics"), hide_index=True, width="stretch")
        st.markdown(f"#### Transition matrix {b} → {r} (km²)")
        st.dataframe(table("transition_matrix_km2"), hide_index=True, width="stretch")
    with a2:
        show("areas")
        if "trend" in F:
            show("trend")
    st.markdown("#### Classifier comparison (spatial block cross-validation)")
    st.dataframe(table("model_comparison_cv"), width="stretch")

with tabs[3]:
    v = S.get("validation")
    if v:
        if v["provisional"]:
            st.warning(f"Provisional: only {v['n_manual']} points have been checked manually so far.")
        a = st.columns(4)
        a[0].metric("Points checked", v["n_points"])
        a[1].metric("Correct / incorrect", f"{v['correct']} / {v['incorrect']}")
        a[2].metric("Approx. accuracy", f"{100 * v['overall_accuracy']:.1f} %")
        a[3].metric("Error-adjusted conversion",
                    f"{v['change']['error_adjusted_conversion_km2']:.2f} ± {v['change']['ci95_km2']:.2f} km²")
        a1, a2 = st.columns(2)
        with a1:
            show("val_cm")
        with a2:
            show("val_pts")
        st.markdown("#### Validation table")
        st.dataframe(table("validation_table"), hide_index=True, width="stretch")
        st.markdown("#### Error analysis")
        st.dataframe(table("error_analysis"), hide_index=True, width="stretch")

with tabs[4]:
    html = OUT / "web" / "interactive_map.html"
    if html.exists():
        st.caption("Use the layer control (top right) to switch RGB, land cover, NDVI/NDBI and the change layer.")
        if hasattr(st, "iframe"):  # Streamlit >= 1.52
            st.iframe(html, height=680)
        else:
            components.html(html.read_text(encoding="utf-8"), height=680, scrolling=False)

with tabs[5]:
    for label, rel in [("Final report (PDF)", S.get("report_pdf")),
                       ("Area statistics (CSV)", "tables/area_statistics.csv"),
                       ("Conversion polygons (GeoJSON)", f"vectors/conversion_polygons_{b}_{r}.geojson"),
                       ("Hotspot cells (GeoJSON)", "vectors/hotspot_cells.geojson"),
                       ("Validation table (CSV)", "tables/validation_table.csv"),
                       ("Error analysis (CSV)", "tables/error_analysis.csv"),
                       ("LinkedIn write-up (Markdown)", "report/LINKEDIN_PORTFOLIO.md")]:
        if rel and (OUT / rel).exists():
            st.download_button(label, (OUT / rel).read_bytes(), file_name=Path(rel).name)
