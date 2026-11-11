#!/usr/bin/env python
"""Command-line entry point.

Examples
--------
# 1) download Sentinel-2 data from Earth Engine and run everything
python scripts/run_pipeline.py --config config/lahore_raiwind.json --download --ee-project my-gcp-project

# 2) re-run the analysis on the downloaded stack (e.g. after labelling validation points)
python scripts/run_pipeline.py --config config/lahore_raiwind.json

# 3) offline demo on a synthetic scene (no Earth Engine; outputs are watermarked)
python scripts/run_pipeline.py --synthetic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoguardian.config import ProjectConfig  # noqa: E402
from geoguardian.pipeline import GeoGuardian  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Punjab GeoGuardian pipeline")
    ap.add_argument("--config", default="config/lahore_raiwind.json")
    ap.add_argument("--download", action="store_true", help="collect data from Earth Engine first")
    ap.add_argument("--ee-project", default=None, help="Google Cloud project for Earth Engine")
    ap.add_argument("--synthetic", action="store_true", help="offline test on a synthetic scene")
    ap.add_argument("--new-validation", action="store_true", help="draw a fresh set of validation points")
    ap.add_argument("--zip", action="store_true", help="zip the outputs folder at the end")
    a = ap.parse_args()

    if a.synthetic:
        from geoguardian.synthetic import make_synthetic_stack
        cfg = ProjectConfig(aoi_key="synthetic_demo", aoi_name="Synthetic test scene",
                            data_dir="data_synthetic", output_dir="outputs_synthetic", synthetic=True)
        make_synthetic_stack(cfg)
    else:
        cfg = ProjectConfig.from_json(a.config)
        if a.download or not cfg.stack_path.exists():
            from geoguardian.gee_data import collect_data, init_earth_engine
            init_earth_engine(a.ee_project)
            collect_data(cfg)
    project = GeoGuardian(cfg).run_all(force_new_validation=a.new_validation)
    if a.zip:
        print("zipped:", project.zip_outputs())


if __name__ == "__main__":
    main()
