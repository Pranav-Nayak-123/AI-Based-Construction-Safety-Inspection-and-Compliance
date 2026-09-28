from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evaluation.ppe_eval import Observed, score
from pipeline.cache.pass1_store import FrameRecord, Pass1Store
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, CoordinateSpace
from shared.identity import canonical_track_id
from shared.schemas.tracks import Pass1TrackObservation
from tools.label_ppe import MAX_PER_TRACK, height_bin, raw_box, sample_observations


def _store(path: Path) -> Pass1Store:
    """40 workers of three sizes, each visible for 90 frames, spread over 60 s."""
    store = Pass1Store(path)
    for frame in range(900):
        observations = []
        for track in range(40):
            if not track * 20 <= frame < track * 20 + 90:
                continue
            height = (50.0, 90.0, 160.0)[track % 3]
            box = Box2(
                x1=100.0, y1=200.0, x2=140.0, y2=200.0 + height, space=CoordinateSpace.REFERENCE
            )
            observations.append(
                Pass1TrackObservation(
                    shot_id="s0",
                    frame_index=frame,
                    video_time_s=frame / 15,
                    track_id=track,
                    canonical_track_id=canonical_track_id("s0", track),
                    object_class=PERSON_CLASS,
                    box_reference=box,
                    anchor_reference=Point2(x=120.0, y=box.y2, space=CoordinateSpace.REFERENCE),
                    confidence=0.8,
                )
            )
        store.add_frame(
            FrameRecord(
                frame_index=frame,
                source_index=frame,
                video_time_s=frame / 15,
                shot_id="s0",
                transform_valid=True,
                detection_count=len(observations),
            ),
            observations,
        )
    store.commit()
    return store


def test_sample_is_stratified_capped_and_reproducible(tmp_path: Path) -> None:
    store = _store(tmp_path / "pass1.sqlite")
    first = sample_observations(store, 60)
    assert len(first) == 60
    assert max(Counter(s.track_id for s in first).values()) <= MAX_PER_TRACK
    assert set(Counter(height_bin(s.height) for s in first)) == {"far", "mid", "near"}
    times = sorted(s.time_s for s in first)
    assert times[0] < 15 and times[-1] > 45  # early and late parts of the clip both sampled
    assert [s.sample_id for s in sample_observations(store, 60)] == [s.sample_id for s in first]


def test_reference_boxes_map_back_to_source_pixels() -> None:
    # 2560x1440 letterboxed into 1920x1080 is a pure 0.75 scale.
    assert raw_box((750, 375, 900, 600), (2560, 1440), (1920, 1080)) == (1000, 500, 1200, 800)
    # 4:3 source: pillarboxed, so x carries an offset.
    assert raw_box((240, 0, 1680, 1080), (1440, 1080), (1920, 1080)) == (0, 0, 1440, 1080)


def _observed(helmet: str, confidence: float, vest: str = "vest") -> Observed:
    return Observed(
        box=(100, 100, 140, 200),
        helmet=helmet,
        helmet_confidence=confidence,
        vest=vest,
        vest_confidence=0.9,
    )


def test_scoring_matches_by_frame_and_applies_the_confidence_floor() -> None:
    box = [100, 100, 140, 200]
    samples = [{"sample_id": f"s{i}", "frame_index": i, "box": box} for i in range(6)]
    labels = {
        "s0": {"sample_id": "s0", "helmet": "no_helmet", "vest": "vest"},  # true alert
        "s1": {"sample_id": "s1", "helmet": "helmet", "vest": "vest"},  # false alert
        "s2": {"sample_id": "s2", "helmet": "no_helmet", "vest": "vest"},  # low confidence
        "s3": {"sample_id": "s3", "helmet": "unclear", "vest": "vest"},  # excluded for helmet
        "s4": {"sample_id": "s4", "helmet": "not_a_person", "vest": "not_a_person"},
        "s5": {"sample_id": "s5", "helmet": "helmet", "vest": "vest"},  # no detection there
    }
    observed = {
        0: [_observed("no_helmet", 0.9)],
        1: [_observed("no_helmet", 0.9)],
        2: [_observed("no_helmet", 0.5)],
        3: [_observed("helmet", 0.9)],
        4: [_observed("helmet", 0.9)],
        5: [],
    }
    report = score(samples, labels, observed, confidence_floor=0.7)
    assert report["labelled"] == 6 and report["not_a_person"] == 1 and report["unmatched"] == 1
    state, effective = report["helmet"]["all/state"], report["helmet"]["all/effective"]
    assert state["n"] == 3 and state["hazard_labelled"] == 2
    assert state["hazard_precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert state["hazard_recall"] == 1.0
    # Below the floor the rules treat "no helmet" as inconclusive: one fewer alert.
    assert effective["hazard_precision"] == 0.5 and effective["hazard_recall"] == 0.5
    assert effective["abstained"] == pytest.approx(1 / 3, abs=1e-3)
    assert report["helmet"]["confusion"]["unclear->helmet"] == 1
    assert report["detector_person_precision"] == pytest.approx(5 / 6, abs=1e-3)


# ------------------------------------------------------------------ incident review

from tools.review_incidents import precision, same_event  # noqa: E402


def _fp(rule="R1", start=10.0, end=14.0, frame=160, box=(100, 100, 140, 200)):
    return {
        "rule_id": rule,
        "start_s": start,
        "end_s": end,
        "evidence_frame": frame,
        "box": list(box) if box else None,
    }


def test_verdicts_carry_over_only_to_the_same_event() -> None:
    reviewed = _fp()
    assert same_event(reviewed, _fp(start=11.0, end=15.0, frame=165, box=(102, 101, 141, 202)))
    assert not same_event(reviewed, _fp(rule="R2"))  # other rule
    assert not same_event(reviewed, _fp(start=20.0, end=22.0))  # no time overlap
    assert not same_event(reviewed, _fp(box=(400, 100, 440, 200)))  # other worker
    assert not same_event(reviewed, _fp(frame=400))  # far-away evidence


def test_precision_counts_real_against_false_alarms_per_rule() -> None:
    incidents = [_fp(), _fp(start=30, end=33, frame=460), _fp(rule="R3"), _fp(rule="R4")]
    verdicts = [
        {"fingerprint": _fp(), "verdict": "real"},
        {
            "fingerprint": _fp(start=30, end=33, frame=460),
            "verdict": "false_alarm",
            "reason": "ppe",
        },
        {"fingerprint": _fp(rule="R3"), "verdict": "unsure"},
    ]
    summary = precision(incidents, verdicts)
    assert summary["R1"]["precision"] == 0.5 and summary["R1"]["false_alarm/ppe"] == 1
    assert summary["R3"]["precision"] is None and summary["R3"]["unsure"] == 1
    assert summary["R4"]["unreviewed"] == 1
