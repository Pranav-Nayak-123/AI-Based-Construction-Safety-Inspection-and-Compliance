from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.intake.frames import FrameSource, Letterbox, sample_raw_frames
from pipeline.intake.geometry_gate import (
    CameraMode,
    ShotCutDetector,
    classify_drift,
    gate_geometry,
)
from pipeline.intake.probe import probe_video
from pipeline.intake.stabilization import (
    AffineFit,
    DriftProbe,
    ReferenceMatcher,
    apply_to_points,
    build_ledger,
    identity_ledger,
    probe_drift,
)
from shared.config import load_config
from shared.errors import PipelineError

INTAKE = load_config().intake


def textured_scene(width: int = 1280, height: int = 720, seed: int = 7) -> np.ndarray:
    """A static scene with plenty of corners for ORB to lock onto."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 90, np.uint8)
    for _ in range(500):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(8, 70)), int(rng.integers(8, 70))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return image


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV cannot write mp4v on this machine"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def counter_frames(count: int, size: tuple[int, int] = (320, 180)) -> list[np.ndarray]:
    """Frames whose top-left pixel block encodes their source index."""
    width, height = size
    frames = []
    for index in range(count):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:20, :20] = (index * 3) % 256
        frames.append(frame)
    return frames


# ------------------------------------------------------------------ letterbox


def test_letterbox_widescreen_scales_without_padding() -> None:
    box = Letterbox.fit((1280, 720), (1920, 1080))
    assert (box.resized_width, box.resized_height, box.offset_x, box.offset_y) == (
        1920,
        1080,
        0,
        0,
    )


def test_letterbox_portrait_is_padded_not_cropped() -> None:
    box = Letterbox.fit((1080, 1920), (1920, 1080))
    assert box.resized_height == 1080
    assert box.resized_width == 608
    assert box.offset_x == (1920 - 608) // 2
    canvas = box.apply(np.full((1920, 1080, 3), 255, np.uint8))
    assert canvas.shape == (1080, 1920, 3)
    assert canvas[:, : box.offset_x].max() == 0  # padding stays black
    assert canvas[540, 960].min() == 255


def test_letterbox_round_trip_is_exact() -> None:
    box = Letterbox.fit((2560, 1440), (1920, 1080))
    points = np.array([[0.0, 0.0], [2560.0, 1440.0], [1234.5, 777.25]])
    homogeneous = np.c_[points, np.ones(len(points))]
    back = (box.inverse() @ (box.forward() @ homogeneous.T)).T[:, :2]
    assert np.abs(back - points).max() < 1e-9
    record = box.record()
    assert record.source_space.value == "raw"
    assert record.destination_space.value == "oriented_canvas"


# ------------------------------------------------------------------ decoding


@pytest.mark.parametrize(("source_fps", "expected"), [(30.0, 30), (25.0, 30), (10.0, 20)])
def test_frame_source_selects_target_rate(tmp_path: Path, source_fps: float, expected: int) -> None:
    count = int(source_fps * 2)
    video = write_video(tmp_path / "clip.mp4", counter_frames(count), source_fps)
    source = FrameSource(video, target_fps=15.0, canvas_size=(320, 180))
    frames = list(source)
    assert len(frames) == expected
    assert [frame.frame_index for frame in frames] == list(range(expected))
    times = [frame.time_s for frame in frames]
    assert times == sorted(times)
    assert source.decoded_count == count
    assert frames[0].image.shape == (180, 320, 3)


def test_frame_source_is_deterministic(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", counter_frames(40), 30.0)
    source = FrameSource(video, target_fps=15.0, canvas_size=(320, 180))
    first = [(f.source_index, f.time_s) for f in source]
    second = [(f.source_index, f.time_s) for f in source]
    assert first == second
    assert [index for index, _ in first] == list(range(0, 40, 2))


def test_frame_source_letterboxes_onto_canvas(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", counter_frames(10, size=(180, 320)), 15.0)
    source = FrameSource(video, target_fps=15.0, canvas_size=(320, 180))
    frame = next(iter(source))
    assert frame.image.shape == (180, 320, 3)
    assert source.letterbox is not None and source.letterbox.offset_x > 0


def test_unreadable_input_fails_clearly(tmp_path: Path) -> None:
    garbage = tmp_path / "broken.mp4"
    garbage.write_bytes(b"not a video")
    with pytest.raises(PipelineError) as raised:
        list(FrameSource(garbage, target_fps=15.0))
    assert raised.value.code in {"DECODE_OPEN_FAILED", "DECODE_NO_FRAMES"}


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_probe_reads_geometry_and_timing(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", counter_frames(30), 15.0)
    metadata = probe_video(video)
    assert (metadata.width, metadata.height) == (320, 180)
    assert metadata.fps == pytest.approx(15.0)
    assert metadata.duration_s == pytest.approx(2.0, abs=0.1)
    assert metadata.display_size == (320, 180)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_probe_rejects_non_video(tmp_path: Path) -> None:
    garbage = tmp_path / "broken.mp4"
    garbage.write_bytes(b"not a video")
    with pytest.raises(PipelineError):
        probe_video(garbage)


# ------------------------------------------------------------------ stabilisation


def test_matcher_recovers_camera_translation() -> None:
    scene = textured_scene()
    fit = ReferenceMatcher(scene).fit(shifted(scene, 12.0, -7.0))
    assert fit.valid, fit.reason_code
    assert fit.matrix is not None
    # The fit maps the moved frame back onto the reference.
    assert fit.matrix[0, 2] == pytest.approx(-12.0, abs=0.75)
    assert fit.matrix[1, 2] == pytest.approx(7.0, abs=0.75)
    assert fit.residual_rms_px is not None and fit.residual_rms_px < 1.5


def test_matcher_reports_failure_instead_of_guessing() -> None:
    scene = textured_scene()
    fit = ReferenceMatcher(scene).fit(np.full_like(scene, 128))
    assert not fit.valid
    assert fit.reason_code == "too_few_features"


def test_drift_probe_measures_peak_corner_displacement() -> None:
    scene = textured_scene()
    probe = probe_drift([scene, shifted(scene, 3, 0), shifted(scene, 6, 0)])
    assert probe.frames_measured == 2
    assert probe.peak_displacement_px == pytest.approx(6.0, abs=0.75)
    assert probe.peak_fraction == pytest.approx(6.0 / np.hypot(1280, 720), rel=0.15)


def _fit(dx: float) -> AffineFit:
    return AffineFit(np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]]), 100, 80, 0.5, "ok")


_FAILED = AffineFit(None, 3, 0, None, "too_few_matches")


def test_short_bounded_gap_is_interpolated() -> None:
    ledger = build_ledger(
        [_fit(0.0), _FAILED, _FAILED, _fit(3.0)], reference_frame_index=0, max_gap=3
    )
    assert [entry.valid for entry in ledger.entries] == [True, True, True, True]
    assert [entry.interpolated for entry in ledger.entries] == [False, True, True, False]
    assert ledger.entries[1].forward_2x3 is not None
    assert ledger.entries[1].forward_2x3[0][2] == pytest.approx(1.0)
    assert ledger.entries[2].forward_2x3 is not None
    assert ledger.entries[2].forward_2x3[0][2] == pytest.approx(2.0)


def test_long_gap_is_never_filled_with_a_stale_transform() -> None:
    fits = [_fit(0.0)] + [_FAILED] * 4 + [_fit(4.0)]
    ledger = build_ledger(fits, reference_frame_index=0, max_gap=3)
    gap = ledger.entries[1:5]
    assert all(not entry.valid and entry.forward_2x3 is None for entry in gap)
    assert {entry.reason_code for entry in gap} == {"transform_gap_too_long"}
    assert ledger.valid_fraction() == pytest.approx(2 / 6)


def test_unbounded_gaps_stay_invalid() -> None:
    ledger = build_ledger([_FAILED, _fit(0.0), _FAILED], reference_frame_index=1, max_gap=3)
    assert [entry.valid for entry in ledger.entries] == [False, True, False]
    assert ledger.entries[0].reason_code == "gap_not_bounded"


def test_ledger_records_carry_exact_inverses() -> None:
    ledger = build_ledger([_fit(2.0), _fit(5.0)], reference_frame_index=0, max_gap=3)
    record = ledger.records()[1]
    assert record.forward_3x3 is not None and record.inverse_3x3 is not None
    product = np.array(record.forward_3x3) @ np.array(record.inverse_3x3)
    assert np.allclose(product, np.eye(3))
    points = apply_to_points(ledger.entries[1].forward_2x3 or [], np.array([[10.0, 20.0]]))
    assert points.tolist() == [[15.0, 20.0]]


def test_identity_ledger_is_valid_everywhere() -> None:
    ledger = identity_ledger(5)
    assert ledger.mode == "identity"
    assert ledger.valid_fraction() == 1.0
    assert ledger.entry(3).frame_index == 3


# ------------------------------------------------------------------ gates


def _probe(peak: float | None) -> DriftProbe:
    return DriftProbe(
        peak_displacement_px=peak, diagonal_px=2203.0, frames_sampled=24, frames_measured=23
    )


def test_drift_classification_thresholds() -> None:
    assert classify_drift(_probe(1.5), INTAKE).mode is CameraMode.FIXED  # 0.07 % of diagonal
    assert classify_drift(_probe(40.0), INTAKE).mode is CameraMode.STABILIZE
    assert classify_drift(_probe(None), INTAKE).mode is CameraMode.UNMEASURABLE


def test_geometry_gate_normalises_residual_by_person_height() -> None:
    fixed = classify_drift(_probe(1.5), INTAKE)
    supported = gate_geometry(fixed, 1.5, 120.0, 2203.0, INTAKE)
    assert supported.supported
    assert supported.residual_drift_wh == pytest.approx(1.5 / 120.0)

    tiny_people = gate_geometry(fixed, 1.5, 20.0, 2203.0, INTAKE)
    assert not tiny_people.supported
    assert tiny_people.reason_code == "residual_drift_exceeds_wh_limit"

    no_people = gate_geometry(fixed, 1.5, None, 2203.0, INTAKE)
    assert no_people.reason_code == "no_reliable_person_scale"


def test_geometry_gate_rejects_poor_stabilisation() -> None:
    drifting = classify_drift(_probe(40.0), INTAKE)
    gate = gate_geometry(drifting, 9.0, 120.0, 2203.0, INTAKE)  # > 0.2 % of diagonal
    assert not gate.supported
    assert gate.reason_code == "stabilization_residual_too_high"


def test_shot_cut_detector_ignores_lighting_drift_but_flags_cuts() -> None:
    detector = ShotCutDetector()
    scene_a = textured_scene(seed=1)
    scene_b = textured_scene(seed=2)[:, ::-1] // 3
    brighter = cv2.convertScaleAbs(scene_a, alpha=1.0, beta=12)
    cuts = [detector.is_cut(frame) for frame in (scene_a, scene_a, brighter, scene_b, scene_b)]
    assert cuts == [False, False, False, True, False]


def test_sample_raw_frames_spans_the_clip(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", counter_frames(50), 25.0)
    frames = sample_raw_frames(video, 5)
    assert len(frames) == 5
    assert frames[0].shape == (180, 320, 3)
