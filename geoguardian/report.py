"""Automated final report PDF (PDF section 21 deliverable + optional feature 81)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from .config import CLASS_NAMES
from .summary import (answers_to_questions, hotspot_sentences, key_findings,
                      policy_recommendations)

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 3.6 * cm
NAVY = colors.HexColor("#14365D")


def _register_fonts():
    """DejaVu (bundled with matplotlib) renders km², →, ≥, ±, Δ which Helvetica lacks."""
    import os

    import matplotlib
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    try:
        for name, f in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                        ("DejaVu-Italic", "DejaVuSans-Oblique.ttf"),
                        ("DejaVu-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                        ("DejaVuMono-Bold", "DejaVuSansMono-Bold.ttf")):
            pdfmetrics.registerFont(TTFont(name, os.path.join(d, f)))
        addMapping("DejaVu", 0, 0, "DejaVu")
        addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        addMapping("DejaVu", 0, 1, "DejaVu-Italic")
        addMapping("DejaVu", 1, 1, "DejaVu-BoldItalic")
        return "DejaVu", "DejaVu-Bold", "DejaVuMono-Bold"
    except Exception:  # fall back to built-in fonts
        return "Helvetica", "Helvetica-Bold", "Courier-Bold"


FONT, FONT_B, FONT_MONO = _register_fonts()
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName=FONT_B, textColor=NAVY, fontSize=16,
                    spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=FONT_B, textColor=NAVY, fontSize=12.5,
                    spaceBefore=8, spaceAfter=5)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontName=FONT, fontSize=9.4, leading=13.2,
                      spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.6, leading=9.4, spaceAfter=0)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8.2, textColor=colors.HexColor("#444444"),
                     alignment=TA_CENTER, spaceAfter=10)
FORMULA = ParagraphStyle("F", parent=BODY, fontName=FONT_MONO, fontSize=9,
                         alignment=TA_CENTER, backColor=colors.HexColor("#F1F4F8"),
                         borderPadding=5, spaceBefore=3, spaceAfter=8)
NOTE = ParagraphStyle("N", parent=BODY, backColor=colors.HexColor("#FFF6E0"),
                      borderColor=colors.HexColor("#E0A800"), borderWidth=0.6, borderPadding=6,
                      spaceBefore=4, spaceAfter=10)


def _p(t, st=BODY):
    return Paragraph(t, st)


def _bullets(items: List[str], st=BODY):
    return ListFlowable([ListItem(_p(i, st), leftIndent=10) for i in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontSize=8)


def _img(path: Path, width=CONTENT_W, caption: str = "", max_h=17 * cm):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    ww = width
    hh = ww * h / w
    if hh > max_h:
        hh = max_h
        ww = hh * w / h
    out = [Image(str(path), width=ww, height=hh)]
    if caption:
        out.append(_p(caption, CAP))
    return KeepTogether(out)


def _table(df: pd.DataFrame, col_widths=None, font=7.6, header_bg=NAVY, zebra=True,
           highlight_col=None):
    st = ParagraphStyle("t", parent=SMALL, fontSize=font, leading=font + 1.8)
    sth = ParagraphStyle("th", parent=st, textColor=colors.white, fontName=FONT_B)
    data = [[_p(str(c), sth) for c in df.columns]]
    for _, row in df.iterrows():
        data.append([_p("" if pd.isna(v) else str(v), st) for v in row.values])
    if col_widths is None:
        col_widths = [CONTENT_W / len(df.columns)] * len(df.columns)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), header_bg),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B0B7C3")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F3F6FA")))
    if highlight_col is not None:
        ci = list(df.columns).index(highlight_col)
        for i, v in enumerate(df[highlight_col].values, start=1):
            if str(v) == "Incorrect":
                style.append(("BACKGROUND", (ci, i), (ci, i), colors.HexColor("#F8D0CC")))
    t.setStyle(TableStyle(style))
    return t


def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _footer_fn(R):
    def fn(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"{R.cfg.organisation} | Punjab GeoGuardian | "
                                              f"{R.cfg.author_name}")
        canvas.drawRightString(PAGE_W - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        if R.cfg.synthetic:
            canvas.setFont(FONT_B, 30)
            canvas.setFillColor(colors.Color(1, 0, 0, alpha=0.18))
            canvas.translate(PAGE_W / 2, PAGE_H / 2)
            canvas.rotate(35)
            canvas.drawCentredString(0, 0, "SYNTHETIC TEST DATA")
        canvas.restoreState()
    return fn


def build_report(R) -> Path:
    cfg, sd, F = R.cfg, R.change.stats_dict, R.figures
    b, r = cfg.baseline_year, cfg.recent_year
    info = R.scene.meta.get("image_info", {})
    name = cfg.author_name.replace(" ", "_")
    path = R.p.report / f"Punjab_GeoGuardian_Final_Report_{name}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title="Punjab GeoGuardian - Final Report", author=cfg.author_name)
    s: list = []

    # ------------------------------------------------------------ title
    s += [Spacer(1, 1.6 * cm),
          _p(cfg.organisation, ParagraphStyle("o", parent=BODY, fontSize=11, alignment=TA_CENTER,
                                              textColor=colors.HexColor("#555555"))),
          Spacer(1, 0.4 * cm),
          _p("Punjab GeoGuardian", ParagraphStyle("t", parent=H1, fontSize=30, alignment=TA_CENTER,
                                                  leading=36)),
          _p("AI-Based Monitoring of Urban Expansion and Farmland Loss",
             ParagraphStyle("t2", parent=H2, fontSize=15, alignment=TA_CENTER)),
          Spacer(1, 0.3 * cm),
          _p(f"Final Report - {cfg.aoi_name} - {b} vs {r}",
             ParagraphStyle("t3", parent=BODY, fontSize=11.5, alignment=TA_CENTER)),
          Spacer(1, 0.8 * cm)]
    if "change" in F:
        s.append(_img(F["change"], width=10 * cm, max_h=10 * cm))
    meta_tbl = pd.DataFrame([
        ("Prepared by", cfg.author_name), ("Submitted to", cfg.submitted_to),
        ("Task", "AIRI Team PITB - AI Internship Task 3 (Individual)"),
        ("Area of Interest", cfg.aoi_name), ("Years compared", f"{b} and {r}"),
        ("Satellite data", "Copernicus Sentinel-2 Level-2A (10 m)"),
        ("Report date", dt.date.today().strftime("%d %B %Y"))], columns=["Item", "Detail"])
    s += [Spacer(1, 0.3 * cm), _table(meta_tbl, [4.5 * cm, CONTENT_W - 4.5 * cm], font=8.5,
                                      zebra=False)]
    if cfg.synthetic:
        s.append(_p("<b>TEST RUN ON SYNTHETIC DATA.</b> These numbers come from a simulated scene and "
                    "only show that the pipeline works. Run the notebook with Earth Engine to get "
                    "real results.", NOTE))
    s.append(PageBreak())

    # --------------------------------------------------------- 1 summary
    s += [_p("1. Executive Summary", H1), _bullets(key_findings(R)),
          _p("<b>Important:</b> This is a learning prototype, not an official government "
             "monitoring system. All results are <i>estimated</i> or <i>possible</i> changes "
             "detected from 10 m satellite data. They are not a finding about any specific "
             "housing society, development or landowner (see Section 11).", NOTE)]

    # ------------------------------------------------ 2 problem & objectives
    s += [_p("2. Problem Statement and Objectives", H1),
          _p("Punjab is urbanising quickly. Around large cities such as Lahore, agricultural land, "
             "green zones and peri-urban land are being converted into housing, roads and commercial "
             "areas. Monitoring this by hand is slow and inconsistent. Punjab GeoGuardian is a GeoAI "
             "prototype that compares Sentinel-2 imagery from two years, classifies land cover, "
             "detects where vegetation/agriculture became built-up, measures the area and "
             "presents the results on maps and a dashboard."),
          _p("The project answers four questions:"),
          *[_p(q, ParagraphStyle("q", parent=BODY, leftIndent=12, spaceAfter=2))
            for q in answers_to_questions(R)],
          _p("GeoAI workflow followed:", BODY),
          _p("Select AOI → Load Sentinel-2 → Cloud-free median composite → NDVI and NDBI → Classify "
             "land cover → Compare old vs recent year → Detect changed pixels → Calculate area → "
             "Visualise on maps → Validate and analyse errors → Report and recommendations",
             FORMULA)]

    # ----------------------------------------------------- 3 AOI & data
    s += [_p("3. Area of Interest and Data", H1), _p(f"<b>{cfg.aoi_name}.</b> {cfg.aoi_description}")]
    if cfg.aoi_mode == "rectangle":
        s.append(_p(f"The AOI is a {cfg.width_km:g} km × {cfg.height_km:g} km rectangle centred at "
                    f"{cfg.center_lat:.4f}°N, {cfg.center_lon:.4f}°E, analysed on a {cfg.pixel_size_m:g} m "
                    f"grid in {R.scene.grid.crs} (UTM), total area {sd['total_aoi_area_km2']:.2f} km². "
                    "Reasons for choosing it: it is a well-known peri-urban growth corridor (the PDF's "
                    "own example), it mixes farmland with new development, and its size is manageable "
                    "for a two-week project while being large enough for meaningful statistics."))
    else:
        s.append(_p(f"District mode: official FAO GAUL boundary of {cfg.district_name} district, "
                    f"total area {sd['total_aoi_area_km2']:.2f} km²."))
    if "aoi" in F:
        s.append(_img(F["aoi"], caption="Figure 1 - AOI location and boundary."))
    rows = []
    for y in cfg.years:
        i = info.get(str(y), {})
        rows.append((str(y), f"{i.get('start', '-')} to {i.get('end', '-')}",
                     str(i.get("n_dates", "-")), str(i.get("n_scenes", "-")),
                     ", ".join(sorted(set(i.get("tiles", [])))) or "-",
                     f"{i.get('window_extended_days', 0)} d"))
    s += [_p("Image selection", H2),
          _table(pd.DataFrame(rows, columns=["Year", "Date range", "Image dates", "Scenes",
                                             "MGRS tiles", "Window widened"]),
                 [1.4 * cm, 4.6 * cm, 2.1 * cm, 1.8 * cm, 3.6 * cm, CONTENT_W - 13.5 * cm], font=8),
          Spacer(1, 4),
          _bullets([
              "Satellite: Copernicus Sentinel-2 MSI Level-2A surface reflectance "
              "(COPERNICUS/S2_SR_HARMONIZED). The <i>harmonized</i> version removes the +1000 offset "
              "ESA added in processing baseline 04.00 (2022), so the 2019 and 2025 values are "
              "comparable.",
              f"Same season each year ({cfg.season_start} to {cfg.season_end}, MM-DD): rabi wheat is "
              "green, cloud cover is low and winter smog has mostly cleared. Comparing the same "
              "crop stage reduces false change.",
              f"Cloud filter: scenes with CLOUDY_PIXEL_PERCENTAGE &lt; {cfg.max_cloud_pct:g}%, then "
              f"pixel-level masking with Google Cloud Score+ (cs_cdf ≥ {cfg.cloud_score_threshold}) "
              "and the Scene Classification Layer (clouds, shadows, cirrus, saturated pixels).",
              "Median composite of all clear observations, which removes leftover haze and noise; "
              "clipped to the AOI.",
              "Bands: Blue (B2), Green (B3), Red (B4), NIR (B8) at 10 m; SWIR1 (B11) and SWIR2 (B12) at "
              "20 m, resampled to 10 m.",
          ])]

    # --------------------------------------------------- 4 concepts
    s += [PageBreak(), _p("4. Remote-Sensing Concepts", H1),
          _p("NDVI - Normalized Difference Vegetation Index", H2),
          _p("NDVI = (NIR − Red) / (NIR + Red)   [Sentinel-2: (B8 − B4) / (B8 + B4)]", FORMULA),
          _p("Healthy plants reflect strongly in near-infrared and absorb red light for "
             "photosynthesis, so vegetation has a high NDVI (about 0.4-0.9). Bare soil and built "
             "surfaces are near 0.1-0.2, and water is negative. <b>High NDVI usually indicates "
             "vegetation or agricultural land.</b>"),
          _p("NDBI - Normalized Difference Built-up Index", H2),
          _p("NDBI = (SWIR − NIR) / (SWIR + NIR)   [Sentinel-2: (B11 − B8) / (B11 + B8)]", FORMULA),
          _p("Concrete, asphalt and roofs reflect more short-wave infrared than near-infrared, so "
             "<b>high NDBI often indicates built-up or urban surfaces</b>. Dry bare soil also has a "
             "high SWIR response, which is why NDBI on its own confuses fallow fields with "
             "built-up land. That is one reason this project combines NDBI with NDVI, all spectral "
             "bands and a machine-learning model."),
          _p("MNDWI (supporting index)", H2),
          _p("MNDWI = (Green − SWIR) / (Green + SWIR)   [(B3 − B11) / (B3 + B11)]", FORMULA),
          _p("Used to separate open water and wet soil (part of the 'Other' class)."),
          _p("Change detection", H2),
          _p("Change detection compares data for the same place at two times. This project uses "
             "<b>post-classification comparison</b>: both years are classified with the same model, "
             f"and a pixel that is Vegetation/Agriculture in {b} and Built-up in {r} is marked as "
             "<b>Agricultural-to-Urban Conversion</b>. To avoid counting small classification "
             f"differences as change, the pixel must also show a real spectral change (NDVI drop ≥ "
             f"{cfg.min_ndvi_drop} or NDBI rise ≥ {cfg.min_ndbi_rise}). Isolated patches smaller than "
             f"{cfg.min_patch_pixels} pixels ({cfg.min_patch_pixels * 100} m²) are removed."),
          _p(f"Example:  {b}: Vegetation  →  {r}: Built-up  ⇒  Agricultural / green land converted "
             f"into urban land", FORMULA)]

    # ---------------------------------------------------- 5 methodology
    m = R.model
    s += [_p("5. Methodology", H1),
          _p("5.1 Land classes", H2),
          _table(pd.DataFrame([(k, CLASS_NAMES[k], d) for k, d in
                               [(0, "Crops, fields, green land, trees"),
                                (1, "Buildings, roads, housing societies, developed land"),
                                (2, "Bare land, water, mixed land, unclear land")]],
                              columns=["Class ID", "Class Name", "Description"]),
                 [2 * cm, 5 * cm, CONTENT_W - 7 * cm], font=8.5),
          _p("5.2 Training samples", H2),
          _p(f"{len(R.samples[R.samples.keep])} sample locations were used "
             f"({', '.join(f'{k}: {v}' for k, v in R.samples[R.samples.keep].groupby('class_name').size().items())}), "
             f"each extracted in every year, giving {len(R.train)} training rows. The PDF minimum "
             "is 50 per class. Candidates were taken only where two independent global land-cover "
             "products (ESA WorldCover 2021 and Google Dynamic World for each season) agree on the "
             "class in <i>every</i> year, so the land is stable and unambiguous. Spectral purity "
             f"filters were then applied (vegetation NDVI ≥ {cfg.veg_ndvi_min} in all years; built-up "
             f"NDVI ≤ {cfg.built_ndvi_max}). Samples were spread round-robin over a "
             f"{cfg.spatial_blocks}×{cfg.spatial_blocks} block grid at least {cfg.sample_spacing_m:g} m "
             "apart, so they cover the whole AOI and are not all from one small location. Built-up "
             "samples include buildings, roads and housing zones; vegetation samples include fields, "
             "trees and green patches; 'Other' covers bare land, water and unclear areas. A chip "
             "sheet (training_samples_review.pdf) is generated for checking the samples visually; "
             "any rejected sample is switched off with keep = False and the model is retrained. Validation points are at least "
             f"{cfg.validation_min_distance_to_training_m:g} m from every training sample."),
          _img(F["samples"], width=CONTENT_W * 0.62, caption="Figure - training sample locations."),
          _p("5.3 Classification models", H2),
          _p("<b>Option A (rule-based):</b> NDVI/NDBI/MNDWI thresholds, with starting values "
             f"tuned by grid search on the training samples (tuned: NDVI ≥ {m.rule['tuned']['ndvi_veg']} → "
             f"Vegetation; NDBI ≥ {m.rule['tuned']['ndbi_built']} → Built-up; MNDWI ≥ "
             f"{m.rule['tuned']['mndwi_water']} → Other/water). <b>Option B (machine learning):</b> "
             "Random Forest (400 trees) and XGBoost (400 boosted trees) on 9 features: Blue, Green, "
             "Red, NIR, SWIR1, SWIR2, NDVI, NDBI, MNDWI. One model is trained on the pooled "
             "samples from all years, so the class definitions are identical in each year. Models "
             f"were compared with {cfg.cv_folds}-fold <b>spatial block cross-validation</b>: whole "
             "blocks are held out, which avoids the optimistic accuracy of random splits on "
             "neighbouring pixels. The best model by macro-F1 was used, followed by a 3×3 majority "
             "filter to remove salt-and-pepper noise.")]
    cvdf = pd.DataFrame([(k.replace("_", " ").title(), _fmt(v["overall_accuracy"]),
                          _fmt(v["macro_f1"]), _fmt(v["kappa"]),
                          v.get("note", "spatial block CV"))
                         for k, v in m.all_cv.items()],
                        columns=["Model", "Overall accuracy", "Macro F1", "Kappa", "Evaluation"])
    s += [_table(cvdf, [3.6 * cm, 2.6 * cm, 2.2 * cm, 2 * cm, CONTENT_W - 10.4 * cm], font=8.5),
          _p(f"Selected model: <b>{m.name.replace('_', ' ').title()}</b>.", BODY),
          _img(F["models"], width=CONTENT_W * 0.6),
          _img(F["importance"], width=CONTENT_W * 0.55,
               caption="Figure - feature importance of the selected model."),
          _p("5.4 Change detection and area calculation", H2),
          _p("Change was computed as described in Section 4. All areas are pixel counts × 100 m² "
             "on the 10 m UTM grid (UTM scale error inside the AOI &lt; 0.05%). Hotspots are "
             f"{cfg.hotspot_cell_km:g} km × {cfg.hotspot_cell_km:g} km cells ranked by converted "
             "area. Contiguous conversion patches are also listed. As an extra, an urban heat-risk "
             "proxy combines NDVI loss, NDBI gain and built-up share per 500 m cell. It is a "
             "relative indicator only, not a temperature measurement.")]

    # --------------------------------------------------------- 6 results
    s += [PageBreak(), _p("6. Results", H1), _p("6.1 Pre-processing outputs", H2)]
    k = 2
    for key, cap in (("rgb_b", f"RGB map - {b}"), ("rgb_r", f"RGB map - {r}"),
                     ("ndvi_b", f"NDVI map - {b}"), ("ndvi_r", f"NDVI map - {r}"),
                     ("ndbi_b", f"NDBI map - {b}"), ("ndbi_r", f"NDBI map - {r}")):
        s.append(_img(F[key], width=CONTENT_W * 0.78, caption=f"Figure {k} - {cap}."))
        k += 1
    s += [_p("6.2 Land-cover classification", H2)]
    for key, cap in (("class_b", f"Classified land-cover map - {b}"),
                     ("class_r", f"Classified land-cover map - {r}")):
        s.append(_img(F[key], width=CONTENT_W * 0.78, caption=f"Figure {k} - {cap}."))
        k += 1
    for y in cfg.extra_years:
        if f"class_{y}" in F:
            s.append(_img(F[f"class_{y}"], width=CONTENT_W * 0.6,
                          caption=f"Figure {k} - Land cover {y} (extra year for the trend)."))
            k += 1
    s.append(_img(F["class_pair"], caption=f"Figure {k} - Land cover before / after."))
    k += 1
    s += [KeepTogether([_p("6.3 Change detection", H2),
                        _img(F["change"], width=CONTENT_W * 0.85,
                             caption=f"Figure {k} - Change-detection map: possible agricultural-"
                                     f"to-urban conversion {b}-{r} (bright red), top hotspots "
                                     f"numbered.")]),
          _img(F["change_classes"], width=CONTENT_W * 0.7, caption=f"Figure {k + 1} - Change classes.")]
    k += 2
    s += [_p("6.4 Area statistics", H2), _table(R.change.stats, [9 * cm, CONTENT_W - 9 * cm], font=9)]
    tr = R.change.transition.reset_index().rename(columns={"index": f"From {b} \\ To {r}"})
    s += [Spacer(1, 6), _p(f"Transition matrix {b} → {r} (km²):", BODY),
          _table(tr.round(3), font=8.5), Spacer(1, 6),
          _img(F["areas"], width=CONTENT_W * 0.75, caption=f"Figure {k} - Area by class and year.")]
    k += 1
    if "trend" in F:
        s.append(_img(F["trend"], width=CONTENT_W * 0.65, caption=f"Figure {k} - Land-cover trend."))
        k += 1
    s += [_p("6.5 Change hotspots", H2), _bullets(hotspot_sentences(R, 5)),
          _img(F["hotspots"], width=CONTENT_W * 0.75, caption=f"Figure {k} - Hotspot density.")]
    k += 1
    ht = R.change.hotspot_cells.head(10)[["rank", "converted_km2", "converted_pct_of_cell",
                                          "center_lat", "center_lon"]].copy()
    ht["rank"] = ht["rank"].astype(int)
    ht["converted_km2"] = (ht["converted_km2"] * 100).map(lambda v: f"{v:.1f}")
    ht["converted_pct_of_cell"] = ht["converted_pct_of_cell"].map(lambda v: f"{v:.1f}")
    ht["center_lat"] = ht["center_lat"].map(lambda v: f"{v:.5f}")
    ht["center_lon"] = ht["center_lon"].map(lambda v: f"{v:.5f}")
    ht.columns = ["Rank", "Converted (ha)", "% of 1 km² cell", "Lat", "Lon"]
    s += [_table(ht, font=8.5), Spacer(1, 6),
          _img(F["dndvi"], width=CONTENT_W * 0.6, caption=f"Figure {k} - NDVI change."),
          _img(F["dndbi"], width=CONTENT_W * 0.6, caption=f"Figure {k + 1} - NDBI change."),
          _img(F["heat"], width=CONTENT_W * 0.6,
               caption=f"Figure {k + 2} - Urban heat-risk proxy (relative, not temperature).")]
    k += 3

    # ------------------------------------------------------ 7 validation
    v = R.validation
    s += [PageBreak(), _p("7. Validation", H1)]
    if v["provisional"]:
        s.append(_p(f"<b>PROVISIONAL:</b> only {v['n_manual']} of {len(R.val_sheet)} points have "
                    "been checked manually so far. The rest use the Dynamic World label as a "
                    "temporary reference. Finish labelling validation_points.csv and re-run to "
                    "replace these numbers.", NOTE))
    s += [_p(f"{v['n_points']} validation points (PDF minimum: 50) were drawn by stratified random "
             f"sampling from the {r} map: {cfg.validation_points_per_stratum} per stratum "
             "(vegetation, built-up, other, detected conversion). Points are at least "
             f"{cfg.validation_min_spacing_m:g} m apart and at least "
             f"{cfg.validation_min_distance_to_training_m:g} m from any training sample. Each point "
             f"was checked visually without seeing the model prediction, using {b}/{r} Sentinel-2 "
             "chips (true and false colour) and the Google Maps high-resolution satellite basemap. "
             "The class visible at the point was recorded, along with whether vegetation had "
             "become built-up."),
          _table(pd.DataFrame([
              ("Points checked", v["n_points"]), ("Correct predictions", v["correct"]),
              ("Incorrect predictions", v["incorrect"]),
              ("Approximate validation accuracy", f"{100 * v['overall_accuracy']:.1f} %"),
              ("Area-weighted overall accuracy", f"{100 * v['area_weighted_overall_accuracy']:.1f} %"),
              ("Cohen's kappa", f"{v['kappa']:.3f}"),
              ("Conversion user's accuracy (precision)",
               _fmt(v["change"]["conversion_precision_users_acc"])),
              ("Conversion producer's accuracy (recall)",
               _fmt(v["change"]["conversion_recall_producers_acc"])),
              ("Error-adjusted conversion area (95% CI)",
               f"{v['change']['error_adjusted_conversion_km2']:.2f} km² "
               f"({v['change']['ci95_low_km2']:.2f}-{v['change']['ci95_high_km2']:.2f})"),
              ("Rule-based (Option A) accuracy on the same points",
               f"{100 * R.rule_validation['overall_accuracy']:.1f} %"),
          ], columns=["Metric", "Value"]), [9 * cm, CONTENT_W - 9 * cm], font=9),
          Spacer(1, 4),
          _p("The area-weighted accuracy corrects for the equal number of points per stratum. The "
             "error-adjusted area uses the stratified estimator of Olofsson et al. (2014), the "
             "standard good-practice method for estimating area from a map plus reference sample.",
             SMALL), Spacer(1, 6)]
    pc = v["per_class"].copy()
    pc.columns = ["Class", "Reference pts", "Mapped pts", "Producer's acc.", "User's acc."]
    s += [_table(pc, font=8.5), Spacer(1, 6),
          _img(F["val_cm"], width=CONTENT_W * 0.55, caption="Figure - validation confusion matrix."),
          _img(F["val_pts"], width=CONTENT_W * 0.7, caption="Figure - validation points (green = "
                                                             "correct, red = incorrect).")]
    vt = v["table"][["Point ID", "Actual Class", "Predicted Class", "Correct / Incorrect",
                     "Actual Change", "Predicted Change", "Notes"]]
    s += [_p("Validation table", H2),
          _table(vt, [1.4 * cm, 2.2 * cm, 2.3 * cm, 2.3 * cm, 1.7 * cm, 1.8 * cm,
                      CONTENT_W - 11.7 * cm], font=7.2, highlight_col="Correct / Incorrect")]

    # ---------------------------------------------------- 8 error analysis
    e = R.errors
    s += [PageBreak(), _p("8. Error Analysis", H1),
          _p(f"{len(e['table'])} weak or incorrect cases were analysed (PDF minimum: 15): "
             f"{e['n_incorrect']} incorrect results plus the lowest-confidence correct results. "
             "Error types and possible reasons are suggested automatically from the pixel evidence "
             "(NDVI/NDBI in both years, clear observations, model confidence). They are "
             "hypotheses to be confirmed against each point's image chip; analyst comments "
             "appear in the Notes column of the validation table."),
          _table(e["table"][["Case", "Location / Point", "Expected Result", "Model Result",
                             "Error Type", "Possible Reason (suggested)", "Evidence"]],
                 [1.0 * cm, 1.5 * cm, 2.6 * cm, 2.6 * cm, 2.3 * cm, 3.6 * cm,
                  CONTENT_W - 13.6 * cm], font=6.8),
          _p("Most common error sources", H2),
          _bullets([f"{k}: {n} case(s)" for k, n in sorted(e["reason_counts"].items(),
                                                           key=lambda kv: -kv[1])]),
          _p("Known limitations that affect all results: cloud cover and haze, seasonal crop "
             "changes (harvested or fallow fields), bare soil confused with built-up, new roads "
             "confused with barren land, small buildings missed at 10 m, tree-covered housing "
             "classified as vegetation, different image dates within the season, possibly "
             "imperfect training samples or thresholds, and mixed pixels.", BODY),
          _p("Suggested improvements", H2), _bullets(e["improvements"])]

    # ------------------------------------------------------ 9 answers
    s += [_p("9. Answers to the Key Questions", H1)]
    for q, a in answers_to_questions(R).items():
        s += [_p(f"<b>{q}</b>"), _p(a)]
    s += [_p("10. Policy Recommendations (decision support)", H1), _bullets(policy_recommendations(R))]

    # ------------------------------------------- 11 limitations / sources
    s += [_p("11. Uncertainty, Assumptions and Data Sources", H1),
          _bullets([
              "Learning prototype - not an official government monitoring or enforcement system.",
              "Terms used: 'possible conversion', 'detected change', 'estimated change'. No claim is "
              "made about the legality or approval status of any development, housing society or "
              "landowner.",
              "Sentinel-2 resolution is 10-20 m. Small buildings, narrow roads and mixed pixels at "
              "the urban edge cannot be resolved reliably.",
              "A single season per year is compared. A field that is bare in one season can look like "
              "construction (false conversion), and active construction sites may still be classed "
              "as 'Other' (missed change).",
              f"Accuracy depends on image quality, date selection, cloud cover, thresholds and "
              f"validation quality. {sd['low_confidence_conversion_km2']:.2f} km² of the detected "
              f"conversion has low model confidence.",
              "Validation is based on visual interpretation of satellite imagery, which is itself "
              "uncertain for mixed or ambiguous points.",
              "Training-sample candidates came from global land-cover products (WorldCover, Dynamic "
              "World) and were filtered and checked visually. Errors in those products may still "
              "affect a few samples.",
          ]),
          _p("Data sources", H2), _bullets(R.scene.meta.get("data_sources", []))]

    # ------------------------------------------------------- 12 linkedin
    s += [PageBreak(), _p("12. LinkedIn / Portfolio Section", H1),
          _p("The full LinkedIn-ready write-up (all ten template sections, including the post text) "
             "is in <b>report/LINKEDIN_PORTFOLIO.md</b> and the repository README. Summary:"),
          _bullets([
              f"<b>Project:</b> Punjab GeoGuardian - Detecting Urban Expansion Using Satellite Imagery "
              f"({cfg.aoi_name}, {b} vs {r}).",
              "<b>Tools:</b> Python, Google Colab, Google Earth Engine, Sentinel-2, Geemap, Folium, "
              "Scikit-learn (Random Forest), XGBoost, Rasterio, Streamlit, ReportLab.",
              f"<b>Result:</b> about {sd['agri_to_urban_conversion_km2']:.2f} km² of possible "
              f"agricultural-to-urban conversion ({sd['percent_aoi_converted']:.2f}% of the AOI).",
              f"<b>Validation:</b> {v['n_points']} points, approximate accuracy "
              f"{100 * v['overall_accuracy']:.1f}%" + (" (provisional)" if v["provisional"] else "") + ".",
              "<b>Deliverables:</b> Colab notebook, Python package, maps, interactive map, Streamlit "
              "dashboard, GeoJSON/CSV exports and this report.",
          ]),
          _p("Reproducibility", H2),
          _p(f"Configuration: random seed {cfg.random_seed}; season {cfg.season_start} to "
             f"{cfg.season_end}; cloud &lt; {cfg.max_cloud_pct:g}%; {cfg.samples_per_class_per_year} "
             "samples per class. Re-running notebooks/Punjab_GeoGuardian.ipynb reproduces every "
             "number in this report.", BODY)]

    doc.build(s, onFirstPage=_footer_fn(R), onLaterPages=_footer_fn(R))
    print(f"[report] saved {path}")
    return path


def week1_markdown(R) -> str:
    cfg, info, F = R.cfg, R.scene.meta.get("image_info", {}), R.figures
    b, r = cfg.baseline_year, cfg.recent_year
    rel = lambda k: str(F[k].relative_to(cfg.out)) if k in F else "-"  # noqa: E731
    ts = R.change.timeseries.set_index("year")
    rows = "\n".join(f"| {y} | {info.get(str(y), {}).get('start', '-')} to "
                     f"{info.get(str(y), {}).get('end', '-')} | {info.get(str(y), {}).get('n_dates', '-')} |"
                     for y in cfg.years)
    return f"""# Week 1 Review - Punjab GeoGuardian
**Intern:** {cfg.author_name} | **Submitted to:** {cfg.submitted_to}

1. **AOI:** {cfg.aoi_name} - boundary screenshot: `{rel('aoi')}` (also `web/interactive_map.html`)
2. **Years and date ranges:**

| Year | Date range | Clear image dates |
|---|---|---|
{rows}

3. RGB map {b}: `{rel('rgb_b')}`
4. RGB map {r}: `{rel('rgb_r')}`
5. NDVI map {b}: `{rel('ndvi_b')}`
6. NDVI map {r}: `{rel('ndvi_r')}`
7. NDBI map {b}: `{rel('ndbi_b')}`
8. NDBI map {r}: `{rel('ndbi_r')}`
9. Initial classified land-cover maps: `{rel('class_b')}`, `{rel('class_r')}`

## Short progress summary
- Sentinel-2 L2A composites built for {', '.join(map(str, cfg.years))} (same Feb-Mar season,
  scenes < {cfg.max_cloud_pct:g}% cloud, Cloud Score+ pixel masking, median composite).
- NDVI and NDBI computed for every year. Vegetation shows high NDVI; built-up shows high NDBI and low NDVI.
- {len(R.samples)} training locations generated (>= 50 per class; review sheet in validation/); a {R.model.name.replace('_', ' ')}
  model selected by spatial cross-validation (macro-F1 {R.model.cv['macro_f1']:.3f}).
- Initial land cover: built-up {ts.loc[b, 'Built-up_km2']:.1f} km² in {b} and {ts.loc[r, 'Built-up_km2']:.1f} km² in {r};
  vegetation {ts.loc[b, 'Vegetation_km2']:.1f} km² in {b} and {ts.loc[r, 'Vegetation_km2']:.1f} km² in {r}.
- Next (Week 2): change detection, area statistics, validation with 60 points, error analysis, dashboard, report.
"""
