from __future__ import annotations

import json

import pytest

from jobs.pipeline_job import _fake_tracks, run_rules_on_synthetic_tracks
from shared.config import load_config
from shared.coordinates import Box2, DistanceInterval, Point2
from shared.enums import (
    CoordinateSpace,
    HelmetState,
    RuleId,
    RuleStatus,
    VestState,
)
from shared.schemas.incidents import IncidentRecord
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation, VestRecord


def test_pass1_track_roundtrip() -> None:
    obs = Pass1TrackObservation(
        shot_id="shot-0",
        frame_index=0,
        video_time_s=0.0,
        track_id=1,
        canonical_track_id="shot-0/person-1",
        object_class="person",
        box_reference=Box2(x1=1.0, y1=2.0, x2=3.0, y2=4.0, space=CoordinateSpace.REFERENCE),
        anchor_reference=Point2(x=2.0, y=4.0, space=CoordinateSpace.REFERENCE),
        helmet=HelmetRecord(state=HelmetState.NO_HELMET, confidence=0.9, visible=True),
        vest=VestRecord(state=VestState.VEST, confidence=0.8, visible=True),
    )
    restored = Pass1TrackObservation.model_validate_json(obs.model_dump_json())
    assert restored.canonical_track_id == "shot-0/person-1"
    assert restored.helmet is not None
    assert restored.helmet.state is HelmetState.NO_HELMET


def test_inverted_box_rejected() -> None:
    with pytest.raises(ValueError, match="inverted"):
        Box2(x1=10.0, y1=10.0, x2=5.0, y2=20.0, space=CoordinateSpace.RAW)


def test_metric_distance_fields_absent() -> None:
    interval = DistanceInterval(lower_wh=0.5, median_wh=1.0, upper_wh=1.5)
    payload = interval.model_dump()
    assert "distance_m" not in payload
    assert "metres" not in payload
    assert "lower_wh" in payload


def test_synthetic_tracks_run_through_the_real_engine() -> None:
    result = run_rules_on_synthetic_tracks(
        _fake_tracks("shot-0", 3.0), run_id="run-x", config=load_config()
    )
    coverage = {entry.rule_id: entry for entry in result.coverage}
    assert list(coverage) == list(RuleId)
    assert len(result.incidents) == 1
    assert result.incidents[0].rule_id is RuleId.R1
    assert result.incidents[0].status is RuleStatus.EVALUATED_ALERT
    assert coverage[RuleId.R2].status is RuleStatus.EVALUATED_CLEAR
    for rule_id in (RuleId.R3, RuleId.R4, RuleId.R5):
        assert coverage[rule_id].status is RuleStatus.UNSUPPORTED
    IncidentRecord.model_validate(result.incidents[0].model_dump())


def test_incident_json_serializable() -> None:
    result = run_rules_on_synthetic_tracks(
        _fake_tracks("shot-0", 3.0), run_id="run-y", config=load_config()
    )
    raw = json.dumps(result.incidents[0].model_dump(mode="json"))
    assert "R1" in raw
    assert "distance_m" not in raw
