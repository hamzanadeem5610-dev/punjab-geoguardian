"""Error analysis (PDF section 19): at least 15 weak or incorrect cases.

Cases are taken from the validation points in this order:
    1. incorrect land-cover predictions,
    2. false conversions and missed changes,
    3. then the weakest correct points (lowest model confidence) until >= 15 cases.

For each case, a *suggested* error type and possible reason are generated from
the evidence at that pixel: NDVI/NDBI/MNDWI in both years, the number of clear
observations and the model confidence. These are hypotheses for you to confirm
or edit after looking at the chips. They are not ground truth.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ProjectConfig

IMPROVEMENT_LIBRARY = {
    "Bare / fallow soil confused with built-up":
        "Add more 'Other' training samples on harvested/fallow fields and use a multi-date "
        "(e.g. Feb + Apr) composite so fields that green up at any time are recognised.",
    "Seasonal crop change":
        "Compare images from the same crop stage, or use several dates per year (seasonal "
        "NDVI maximum) instead of a single seasonal median.",
    "Tree-covered housing area classified as vegetation":
        "Add texture features (e.g. GLCM on NIR) or higher-resolution imagery to detect "
        "rooftops under tree canopy.",
    "Small buildings / mixed pixels at 10 m":
        "Use higher-resolution imagery (PlanetScope / aerial) or a U-Net segmentation model "
        "trained on building footprints.",
    "New road / construction site confused with barren land":
        "Add a separate 'Construction / bare urban' class and a road-network layer (OSM).",
    "Cloud / haze or few clear observations":
        "Use stricter cloud filtering (higher Cloud Score+ threshold) and a longer window "
        "to get more clear observations.",
    "Low classifier confidence (borderline pixel)":
        "Collect more training samples near class boundaries and calibrate the decision "
        "threshold.",
    "Water / wet soil confusion":
        "Add a dedicated Water class (MNDWI) and samples along canals and ponds.",
    "Baseline-year map error (old year mapped as built-up)":
        "Check the baseline year separately: add baseline-only training samples in green "
        "peri-urban plots and compare against historical high-resolution imagery.",
    "Change already present in baseline map":
        "Improve the baseline-year map with samples drawn from baseline imagery only, and "
        "validate the baseline year separately.",
}
GENERAL_IMPROVEMENTS = [
    "Use more, and more varied, training samples and review every sample visually.",
    "Validate against official boundaries or known approved housing schemes "
    "(e.g. LDA records) rather than only visual interpretation.",
    "Use multiple years (2019-2025 annually) to separate permanent conversion from "
    "temporary or seasonal change.",
    "Add more land classes (Water, Barren, Industrial / brick kilns, Roads).",
    "Train a U-Net semantic segmentation model for sharper built-up boundaries.",
]


def _reason(row: pd.Series, cfg: ProjectConfig) -> Tuple[str, str, str]:
    b, r = cfg.baseline_year, cfg.recent_year
    exp, got = row["Actual Class"], row["Predicted Class"]
    a_ch, p_ch = row.get("Actual Change", ""), row.get("Predicted Change", "")
    nv_r, nb_r, nw_r = row.get(f"NDVI_{r}", np.nan), row.get(f"NDBI_{r}", np.nan), \
        row.get(f"MNDWI_{r}", np.nan)
    nv_b = row.get(f"NDVI_{b}", np.nan)
    obs = min(row.get(f"obs_{b}", 99), row.get(f"obs_{r}", 99))
    conf = row.get("confidence", 1.0)

    if p_ch == "yes" and a_ch == "no":
        etype = "False conversion"
        if exp == "Built-up":
            reason = "Change already present in baseline map"
        elif nv_r < 0.25 and nb_r > -0.1:
            reason = "Bare / fallow soil confused with built-up"
        else:
            reason = "Seasonal crop change"
    elif p_ch == "no" and a_ch == "yes":
        etype = "Missed change"
        if got == "Built-up" and row.get(f"pred_class_{b}", "") == "Built-up":
            reason = "Baseline-year map error (old year mapped as built-up)"
        elif got == "Other":
            reason = "New road / construction site confused with barren land"
        elif got == "Vegetation" and nv_r > 0.35:
            reason = "Tree-covered housing area classified as vegetation"
        else:
            reason = "Small buildings / mixed pixels at 10 m"
    elif exp != got:
        if exp == "Built-up" and got == "Vegetation":
            etype, reason = "Missed urban area", "Tree-covered housing area classified as vegetation"
        elif exp == "Built-up" and got == "Other":
            etype = "Missed urban area"
            reason = ("New road / construction site confused with barren land" if nb_r > -0.05
                      else "Small buildings / mixed pixels at 10 m")
        elif exp == "Vegetation" and got == "Built-up":
            etype, reason = "False built-up", "Bare / fallow soil confused with built-up"
        elif exp == "Other" and got == "Built-up":
            etype, reason = "False built-up", "Bare / fallow soil confused with built-up"
        elif exp == "Other" and got == "Vegetation":
            etype, reason = "Wrong class", "Seasonal crop change"
        elif exp == "Vegetation" and got == "Other":
            etype, reason = "Wrong class", ("Seasonal crop change" if nv_r < 0.3 else
                                            "Low classifier confidence (borderline pixel)")
        else:
            etype, reason = "Wrong class", "Low classifier confidence (borderline pixel)"
        if exp == "Other" and nw_r > 0:
            reason = "Water / wet soil confusion"
    else:
        etype, reason = "Weak (low-confidence) correct result", \
            "Low classifier confidence (borderline pixel)"
    if obs <= 2:
        reason = "Cloud / haze or few clear observations"
    detail = (f"NDVI {b}={nv_b:.2f}, {r}={nv_r:.2f}; NDBI {r}={nb_r:.2f}; "
              f"clear obs={int(obs)}; confidence={conf:.2f}")
    return etype, reason, detail


def run_error_analysis(val: Dict, sheet: pd.DataFrame, cfg: ProjectConfig) -> Dict:
    t = val["table"].merge(
        sheet.drop(columns=[c for c in ("notes", "lat", "lon") if c in sheet.columns]),
        left_on="Point ID", right_on="point_id", how="left")
    wrong_lc = t["Correct / Incorrect"].eq("Incorrect")
    wrong_ch = (t["Actual Change"].isin(["yes", "no"])) & \
        (t["Actual Change"] != t["Predicted Change"])
    t["_prio"] = np.select([wrong_lc & wrong_ch, wrong_ch, wrong_lc], [0, 1, 2], 3)
    t = t.sort_values(["_prio", "confidence"]).reset_index(drop=True)
    n_wrong = int((t["_prio"] < 3).sum())
    n_cases = max(cfg.min_error_cases, n_wrong)
    cases = t.head(n_cases).copy()
    out = []
    for i, row in cases.iterrows():
        etype, reason, detail = _reason(row, cfg)
        exp = row["Actual Class"]
        got = row["Predicted Class"]
        if row["Actual Change"] in ("yes", "no"):
            exp_txt = "Vegetation-to-built-up" if row["Actual Change"] == "yes" else f"{exp} (no change)"
            got_txt = ("Conversion detected" if row["Predicted Change"] == "yes"
                       else f"{got} (no change)")
        else:
            exp_txt, got_txt = exp, got
        out.append({"Case": i + 1, "Location / Point": f"Point {int(row['Point ID'])}",
                    "Latitude": row["Latitude"], "Longitude": row["Longitude"],
                    "Expected Result": exp_txt, "Model Result": got_txt,
                    "Error Type": etype, "Possible Reason (suggested)": reason,
                    "Evidence": detail, "Analyst Notes": row.get("Notes", "")})
    table = pd.DataFrame(out)
    freq = Counter(table.loc[~table["Error Type"].str.startswith("Weak"),
                             "Possible Reason (suggested)"])
    if not freq:
        freq = Counter(table["Possible Reason (suggested)"])
    improvements: List[str] = [IMPROVEMENT_LIBRARY[k] for k, _ in freq.most_common()
                               if k in IMPROVEMENT_LIBRARY]
    for g in GENERAL_IMPROVEMENTS:
        if len(improvements) >= 6:
            break
        if g not in improvements:
            improvements.append(g)
    print(f"[errors] {len(table)} cases analysed ({n_wrong} incorrect, "
          f"{len(table) - n_wrong} weak-correct)")
    return {"table": table, "n_incorrect": n_wrong, "reason_counts": dict(freq),
            "improvements": improvements}
