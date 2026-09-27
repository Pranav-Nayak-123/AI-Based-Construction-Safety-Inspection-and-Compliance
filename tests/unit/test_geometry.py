"""Ground geometry for R2/R4/R5: bands, distances, footprints, operating state, the join."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rule_frames import rules_config
from test_camera_model import TRUE, fit, synthetic_pairs

from pipeline.cache.pass1_store import FrameRecord
from pipeline.mapping.camera_model import Camera, GroundMapper
from pipeline.mapping.frame_context import FrameContextBuilder
from pipeline.mapping.geometry import (
    OperatingStateTracker,
    assign_band,
    distance_to_polygon,
    distance_to_polyline,
    footprint_from_contact,
    summarise,
    summarise_many,
)
from shared.coordinates import Box2, DistanceInterval, Point2
from shared.enums import CoordinateSpace, DistanceBand, OperatingState
from shared.identity import canonical_track_id
from shared.schemas.scene import SceneCache
from shared.schemas.tracks import Pass1TrackObservation


def interval(lower: float, upper: float) -> DistanceInterval:
    return DistanceInterval(lower_wh=lower, median_wh=(lower + upper) / 2, upper_wh=upper)


@pytest.mark.parametrize(
    ("lower", "upper", "band"),
    [
        (0.2, 1.5, DistanceBand.NEAR),  # exact upper 1.5 is near (v7 §47.1)
        (1.5, 2.0, DistanceBand.INDETERMINATE),  # exact lower 1.5 with a larger upper
        (1.6, 2.9, DistanceBand.CAUTION),
        (1.2, 1.9, DistanceBand.INDETERMINATE),  # straddles near/caution
        (3.1, 5.0, DistanceBand.CLEAR),
        (2.5, 3.5, DistanceBand.INDETERMINATE),
    ],
)
def test_band_needs_the_whole_interval(lower: float, upper: float, band: DistanceBand) -> None:
    assert assign_band(interval(lower, upper), 1.5, 3.0) is band


def test_summaries_abstain_when_too_many_replicates_fail() -> None:
    samples = np.array([1.0] * 60 + [np.nan] * 40)
    assert summarise(samples, 100) is None
    [good, bad] = summarise_many(np.column_stack([np.linspace(1, 2, 100), samples]), 100)
    assert bad is None
    assert good is not None and good.lower_wh == pytest.approx(1.025, abs=0.01)


def test_polygon_and_polyline_distances() -> None:
    square = np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]], float)
    points = np.array([[[1, 1], [3, 1], [5, 6]]], float)
    assert distance_to_polygon(points, square)[0].tolist() == pytest.approx([0.0, 1.0, 5.0])
    line = np.array([[[0, 0], [4, 0]]], float)
    assert distance_to_polyline(points, line)[0].tolist() == pytest.approx(
        [1.0, 1.0, math.sqrt(37)]
    )


def test_footprint_extends_away_from_the_camera() -> None:
    contact = np.array([[[-1.0, 10.0], [1.0, 10.0]]])
    quad = footprint_from_contact(contact)[0]
    assert quad[2][1] > 10.0 and quad[3][1] > 10.0  # extruded to larger depth
    assert np.linalg.norm(quad[3] - quad[0]) == pytest.approx(1.2)


def _run_tracker(positions, energies, fps=15.0):
    tracker = OperatingStateTracker()
    state = OperatingState.UNKNOWN
    for index, (position, energy) in enumerate(zip(positions, energies, strict=True)):
        state = tracker.update("m1", index / fps, position, energy)
    return state


def test_operating_state_from_base_motion_and_articulation() -> None:
    still = [np.array([0.0, 10.0])] * 40
    moving = [np.array([0.02 * i, 10.0]) for i in range(40)]  # 0.3 WH/s
    assert _run_tracker(moving, [0.0] * 40) is OperatingState.ACTIVE
    assert _run_tracker(still, [0.01] * 40) is OperatingState.STATIONARY
    assert _run_tracker(still, [0.08] * 40) is OperatingState.ACTIVE  # boom working in place
    assert _run_tracker(still, [0.03] * 40) is OperatingState.UNKNOWN  # ambiguous
    assert _run_tracker([None] * 40, [0.0] * 40) is OperatingState.UNKNOWN  # no evidence


def _observation(local_id: int, cls: str, foot: np.ndarray, height_px: float, width_px: float):
    x, y = float(foot[0]), float(foot[1])
    return Pass1TrackObservation(
        shot_id="s0",
        frame_index=0,
        video_time_s=0.0,
        track_id=local_id,
        canonical_track_id=canonical_track_id("s0", local_id),
        object_class=cls,
        box_reference=Box2(
            x1=x - width_px / 2,
            y1=y - height_px,
            x2=x + width_px / 2,
            y2=y,
            space=CoordinateSpace.REFERENCE,
        ),
        anchor_reference=Point2(x=x, y=y, space=CoordinateSpace.REFERENCE),
        confidence=0.9,
    )


def test_join_produces_near_band_for_a_worker_beside_a_machine() -> None:
    truth = Camera(TRUE)
    mapper = GroundMapper(fit(synthetic_pairs(truth), samples=80))
    worker_foot = truth.project(np.array([[0.0, 12.0, 0.0]]))[0]
    machine_left = truth.project(np.array([[0.6, 12.0, 0.0]]))[0]
    machine_right = truth.project(np.array([[3.0, 12.0, 0.0]]))[0]
    machine_foot = (machine_left + machine_right) / 2
    machine_width = machine_right[0] - machine_left[0]
    builder = FrameContextBuilder(
        run_id="run-test",
        scene=SceneCache(cache_key="none", video_sha256="v", shot_proposals={}),
        geometry_supported=True,
        ppe_available=True,
        machinery_classes=("machinery",),
        mappers={"s0": mapper},
        rules=rules_config(),
    )
    frame = FrameRecord(
        frame_index=0,
        source_index=0,
        video_time_s=0.0,
        shot_id="s0",
        transform_valid=True,
        detection_count=2,
    )
    context = builder.build(
        frame,
        [
            _observation(1, "person", worker_foot, 120.0, 45.0),
            _observation(2, "machinery", machine_foot, 150.0, machine_width / 0.8),
        ],
    )
    assert context.capabilities.relative_plane
    [pair] = context.machinery_proximities
    assert pair.distance is not None
    assert pair.distance.upper_wh <= 1.5
    assert pair.band is DistanceBand.NEAR
    worker = next(t for t in context.tracks if t.object_class == "person")
    assert worker.relative_position is not None
    assert worker.relative_position.x == pytest.approx(0.0, abs=0.3)
    assert math.isfinite(worker.relative_position.y)
