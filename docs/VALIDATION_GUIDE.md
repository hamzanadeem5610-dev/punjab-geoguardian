# How to label the 60 validation points (about 30-40 minutes)

The PDF requires **at least 50 manually checked points**. The pipeline draws 60 and gives you
everything you need to check them by eye.

## Tools
* **Colab widget** (notebook part 8): one point at a time, with chips and buttons, saved automatically.
* or **`outputs/validation/validation_points.csv`** in Google Sheets / Excel, with
  `validation_chips.pdf` open next to it.

## For each point
1. Look at the chips: 2019 true colour | 2025 true colour | 2025 false colour (vegetation looks **red** in false colour).
   The cyan box is about 40 m around the point.
2. Open the `google_maps` link (high-resolution satellite) and look at what is **at the centre**.
3. Fill in:
   * `actual_class`: **Vegetation** (crop field, trees, green land), **Built-up** (buildings, roads, paved
     housing plots, factories) or **Other** (bare/fallow soil, cleared plot, water, canal, unclear).
     Judge the **2025** situation.
   * `actual_change`: **yes** if the place was vegetation/agriculture in 2019 and is built-up in 2025;
     otherwise **no**. Compare the 2019 and 2025 chips, and optionally use Google Earth Pro's
     *historical imagery* slider.
   * `notes`: a few words, e.g. "new housing plots", "wheat field", "brick kiln", "canal", "fallow field".
4. If the point falls on a boundary between two classes, choose the class that covers most of the 10 m pixel
   and write "mixed pixel" in `notes`.

## Rules for an honest validation
* **Don't look at the prediction.** The widget and the chips hide it on purpose. Do not open the
  `predicted_*` or `hint_*` columns while labelling.
* Label what you **see**, not what you expect.
* Label every point. Unlabelled points make the report show **PROVISIONAL**.

Then run the rest of the notebook (or `python scripts/run_pipeline.py`). Accuracy, the confusion matrix,
the error analysis and the report update automatically. Your labels are never overwritten.
