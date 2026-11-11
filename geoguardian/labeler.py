"""Point-by-point validation labelling widget for Colab / Jupyter (ipywidgets).

    from geoguardian.labeler import ValidationLabeler
    ValidationLabeler("outputs/validation/validation_points.csv",
                      "outputs/validation/chips").show()

For each point it shows the 2019 / 2025 Sentinel-2 chips and a Google Maps
satellite link. You choose the actual class and whether vegetation became
built-up, and add a note. Every answer is saved to the CSV immediately. The
model prediction is hidden, so the labelling stays blind.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASSES = ["Vegetation", "Built-up", "Other"]


class ValidationLabeler:
    def __init__(self, csv_path, chip_dir, start_at_first_unlabelled: bool = True):
        self.csv = Path(csv_path)
        self.chips = Path(chip_dir)
        self.df = pd.read_csv(self.csv, dtype=str).fillna("")
        self.i = 0
        if start_at_first_unlabelled:
            empty = self.df.index[self.df["actual_class"] == ""]
            self.i = int(empty[0]) if len(empty) else 0

    def _save(self):
        self.df.to_csv(self.csv, index=False)

    def show(self):
        import ipywidgets as w
        from IPython.display import display

        self.img = w.Image(format="png", layout=w.Layout(width="100%"))
        self.info = w.HTML()
        self.cls = w.ToggleButtons(options=CLASSES, value=None, description="Actual class:",
                                   style={"description_width": "initial"})
        self.chg = w.ToggleButtons(options=["no", "yes"], value=None,
                                   description="Vegetation → built-up?",
                                   style={"description_width": "initial"})
        self.notes = w.Text(placeholder="e.g. housing area / crop field / bare soil / canal",
                            description="Notes:", layout=w.Layout(width="70%"))
        prev_b = w.Button(description="◀ Previous")
        next_b = w.Button(description="Save & next ▶", button_style="success")
        self.progress = w.HTML()
        prev_b.on_click(lambda _: self._move(-1))
        next_b.on_click(lambda _: self._move(+1))
        display(w.VBox([self.progress, self.info, self.img, self.cls, self.chg, self.notes,
                        w.HBox([prev_b, next_b])]))
        self._render()

    def _render(self):
        row = self.df.iloc[self.i]
        pid = int(float(row["point_id"]))
        chip = self.chips / f"point_{pid:03d}.png"
        self.img.value = chip.read_bytes() if chip.exists() else b""
        self.info.value = (f"<b>Point {pid}</b> - {float(row['lat']):.6f}, {float(row['lon']):.6f} "
                           f"- <a href='{row['google_maps']}' target='_blank'>open high-res satellite "
                           f"in Google Maps</a> (the cyan box is about 40 m)")
        self.cls.value = row["actual_class"] if row["actual_class"] in CLASSES else None
        self.chg.value = row["actual_change"] if row["actual_change"] in ("yes", "no") else None
        self.notes.value = row["notes"]
        done = int((self.df["actual_class"] != "").sum())
        self.progress.value = f"<b>{done} / {len(self.df)} labelled</b> - saved to {self.csv.name}"

    def _move(self, step: int):
        if self.cls.value:
            self.df.at[self.i, "actual_class"] = self.cls.value
        if self.chg.value:
            self.df.at[self.i, "actual_change"] = self.chg.value
        self.df.at[self.i, "notes"] = self.notes.value
        self._save()
        self.i = max(0, min(len(self.df) - 1, self.i + step))
        self._render()
