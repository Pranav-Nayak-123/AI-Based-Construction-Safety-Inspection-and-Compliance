from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from rule_frames import zone

from pipeline.intake.probe import probe_video
from pipeline.render.encoder import VideoEncoder, transcode_proxy
from pipeline.render.source_overlay import ALERT, PENDING, render_source_frame
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, AlertState, CoordinateSpace, HelmetState, RuleId
from shared.errors import PipelineError
from shared.identity import canonical_track_id
from shared.schemas.frame import ActiveAlert, FrameRuleOutput
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def _gradient(width: int, height: int, shade: int) -> np.ndarray:
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:] = shade
    frame[:, : width // 2, 1] = 200
    return frame


@needs_ffmpeg
def test_encoder_writes_complete_video_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "out.mp4"
    with VideoEncoder(destination, width=320, height=180, fps=15.0, preset="ultrafast") as encoder:
        for index in range(30):
            encoder.write(_gradient(320, 180, index * 5))
        assert not destination.exists()  # only the .partial exists while encoding
    metadata = probe_video(destination)
    assert (metadata.width, metadata.height) == (320, 180)
    assert metadata.duration_s == pytest.approx(2.0, abs=0.1)
    assert not list(tmp_path.glob("*.partial.mp4"))
    assert not list(tmp_path.glob("*.ffmpeg.log"))

    proxy = transcode_proxy(destination, tmp_path / "proxy.mp4", height=90)
    assert probe_video(proxy).height == 90


@needs_ffmpeg
def test_encoder_rejects_bad_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even"):
        VideoEncoder(tmp_path / "odd.mp4", width=321, height=180, fps=15.0)
    encoder = VideoEncoder(tmp_path / "out.mp4", width=320, height=180, fps=15.0)
    with pytest.raises(ValueError, match="must be 180x320x3"):
        encoder.write(np.zeros((10, 10, 3), np.uint8))
    encoder.abort()
    assert not (tmp_path / "out.mp4").exists()


@needs_ffmpeg
def test_ffmpeg_failure_is_reported_and_leaves_no_output(tmp_path: Path) -> None:
    encoder = VideoEncoder(
        tmp_path / "out.mp4", width=320, height=180, fps=15.0, preset="no-such-preset"
    )
    with pytest.raises(PipelineError) as raised:
        for _ in range(200):
            encoder.write(_gradient(320, 180, 10))
        encoder.close()
    assert raised.value.code == "FFMPEG_FAILED"
    assert not (tmp_path / "out.mp4").exists()


def _worker(helmet: HelmetState) -> Pass1TrackObservation:
    box = Box2(x1=100.0, y1=100.0, x2=160.0, y2=260.0, space=CoordinateSpace.REFERENCE)
    return Pass1TrackObservation(
        shot_id="s0",
        frame_index=0,
        video_time_s=0.0,
        track_id=5,
        canonical_track_id=canonical_track_id("s0", 5),
        object_class=PERSON_CLASS,
        box_reference=box,
        anchor_reference=Point2(x=130.0, y=260.0, space=CoordinateSpace.REFERENCE),
        helmet=HelmetRecord(state=helmet, confidence=0.9, visible=True),
    )


def _output(state: AlertState | None) -> FrameRuleOutput:
    alerts = ()
    if state is not None:
        alerts = (
            ActiveAlert(
                rule_id=RuleId.R1,
                track_id=canonical_track_id("s0", 5),
                state=state,
                incident_id=None,
            ),
        )
    return FrameRuleOutput(frame_index=0, video_time_s=0.0, results=(), active_alerts=alerts)


def _count(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int(np.all(image == np.array(color, np.uint8), axis=-1).sum())


def test_box_colour_follows_the_alert_state() -> None:
    canvas = np.full((360, 640, 3), 90, np.uint8)
    calm = render_source_frame(
        canvas,
        observations=[_worker(HelmetState.HELMET)],
        output=_output(None),
        proposals=(),
        time_s=1.0,
    )
    pending = render_source_frame(
        canvas,
        observations=[_worker(HelmetState.NO_HELMET)],
        output=_output(AlertState.PENDING_ALERT),
        proposals=(),
        time_s=1.0,
    )
    alerting = render_source_frame(
        canvas,
        observations=[_worker(HelmetState.NO_HELMET)],
        output=_output(AlertState.ALERT),
        proposals=(),
        time_s=1.0,
    )
    assert _count(calm, ALERT) == 0 and _count(calm, PENDING) == 0
    assert _count(pending, PENDING) > 0 and _count(pending, ALERT) == 0
    assert _count(alerting, ALERT) > 0
    assert canvas.max() == 90  # the input frame is never modified


def test_render_resizes_and_scales_annotations() -> None:
    canvas = np.full((1080, 1920, 3), 90, np.uint8)
    image = render_source_frame(
        canvas,
        observations=[_worker(HelmetState.HELMET)],
        output=None,
        proposals=[zone("zone-1")],
        time_s=65.5,
        size=(940, 529),
    )
    assert image.shape == (529, 940, 3)
