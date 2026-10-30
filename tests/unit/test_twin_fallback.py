"""Approximate twin for a steady camera the workers could not calibrate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipeline.mapping.camera_model import Camera, default_camera, visible_ground
from pipeline.output.twin_data import FALLBACK_FAR_WH, TwinRecorder


def _worker(x: float, y: float, relative=None):  # noqa: ANN001, ANN202
    return SimpleNamespace(relative_position=relative, anchor_reference=SimpleNamespace(x=x, y=y))


def test_default_camera_sees_ground_from_the_bottom_edge_outwards() -> None:
    camera = Camera(default_camera(1920, 1080))
    near, far = camera.ground(np.array([[960.0, 1080.0], [960.0, 300.0]]))
    assert 0 < near[1] < far[1]  # lower in the image is nearer the camera
    assert abs(near[0]) < 1e-6  # the centre column stays on the camera's axis
    ring = visible_ground(camera, 1920, 1080, far_wh=FALLBACK_FAR_WH)
    assert len(ring) == 4 and max(y for _, y in ring) <= FALLBACK_FAR_WH


def test_fallback_places_uncalibrated_workers_but_never_overrides_calibration() -> None:
    camera = Camera(default_camera(1920, 1080))
    twin = TwinRecorder(fps=25.0, fallback=camera.ground)
    placed = twin._position(_worker(960.0, 1000.0))
    assert placed is not None and placed[1] > 0
    calibrated = twin._position(_worker(960.0, 1000.0, SimpleNamespace(x=1.5, y=2.5)))
    assert calibrated == (1.5, 2.5)
    assert twin._position(_worker(960.0, 0.0)) is None  # near the horizon: too far to trust
    assert TwinRecorder(fps=25.0)._position(_worker(960.0, 1000.0)) is None  # no fallback
