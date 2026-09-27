from __future__ import annotations

from pathlib import Path

from pipeline.cache.pass1_store import FrameRecord, Pass1Store
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, CoordinateSpace
from shared.identity import canonical_track_id
from shared.schemas.tracks import Pass1TrackObservation


def _frame(index: int, detections: int = 0) -> FrameRecord:
    return FrameRecord(
        frame_index=index,
        source_index=index * 2,
        video_time_s=index / 15,
        shot_id="s0",
        transform_valid=True,
        detection_count=detections,
    )


def _observation(frame_index: int, track_id: int) -> Pass1TrackObservation:
    box = Box2(x1=10.0, y1=20.0, x2=40.0, y2=120.0, space=CoordinateSpace.REFERENCE)
    return Pass1TrackObservation(
        shot_id="s0",
        frame_index=frame_index,
        video_time_s=frame_index / 15,
        track_id=track_id,
        canonical_track_id=canonical_track_id("s0", track_id),
        object_class=PERSON_CLASS,
        box_reference=box,
        anchor_reference=Point2(x=25.0, y=120.0, space=CoordinateSpace.REFERENCE),
        confidence=0.8,
    )


def test_store_round_trips_frames_and_observations(tmp_path: Path) -> None:
    path = tmp_path / "pass1.sqlite"
    with Pass1Store(path, commit_every=2) as store:
        store.add_frame(_frame(0, 2), [_observation(0, 1), _observation(0, 2)])
        store.add_frame(_frame(1), [])
        store.add_frame(_frame(2, 1), [_observation(2, 1)])
        store.set_meta("detector_sha256", "abc")
        assert not store.complete
        store.mark_complete()

    with Pass1Store(path) as store:
        assert store.complete
        assert store.meta("detector_sha256") == "abc"
        assert store.frame_count() == 3
        pairs = list(store.frames_with_observations())
        assert [(frame.frame_index, len(obs)) for frame, obs in pairs] == [(0, 2), (1, 0), (2, 1)]
        assert pairs[0][1][1].track_id == 2
        history = store.observations_for(canonical_track_id("s0", 1))
        assert [obs.frame_index for obs in history] == [0, 2]
