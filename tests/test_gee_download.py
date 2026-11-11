"""Tiled Earth Engine download is assembled correctly (computePixels mocked)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoguardian import gee_data  # noqa: E402
from geoguardian.aoi import Grid  # noqa: E402


def test_tiled_download_reassembles_grid(monkeypatch):
    grid = Grid("EPSG:32643", 400000.0, 3480000.0, width=1100, height=700, res=10.0)
    bands = ["a", "b"]
    calls = []

    def compute_pixels(req):
        g = req["grid"]
        assert req["fileFormat"] == "NUMPY_NDARRAY" and req["bandIds"] == bands
        w, h = g["dimensions"]["width"], g["dimensions"]["height"]
        x0, y0 = g["affineTransform"]["translateX"], g["affineTransform"]["translateY"]
        c0 = int((x0 - grid.x0) / grid.res)
        r0 = int((grid.y0 - y0) / grid.res)
        calls.append((r0, c0, h, w))
        rr, cc = np.mgrid[r0:r0 + h, c0:c0 + w]
        out = np.zeros((h, w), dtype=[("a", np.int16), ("b", np.int16)])
        out["a"], out["b"] = rr, cc
        return out

    monkeypatch.setattr(gee_data, "_ee", lambda: SimpleNamespace(
        data=SimpleNamespace(computePixels=compute_pixels)))
    arr = gee_data.download_image(object(), grid, bands, tile=512, workers=2)
    rr, cc = np.mgrid[0:700, 0:1100]
    assert (arr[0] == rr).all() and (arr[1] == cc).all()
    assert len(calls) == 6  # ceil(700/512) * ceil(1100/512) = 2 * 3
    # every request stays far below Earth Engine's 48 MB computePixels limit
    assert max(h * w for _, _, h, w in calls) * 26 * 2 < 48 * 1024 ** 2
