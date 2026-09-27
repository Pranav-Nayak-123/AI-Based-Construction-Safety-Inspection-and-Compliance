"""Pass one: decode → detect → track → store (v7 §35).

The expensive pass. It runs once per clip; rules and rendering re-read its store.

* Detection is batched for GPU throughput; tracking consumes frames strictly in order.
* A hard cut starts a new shot: the tracker is reset so IDs never continue across shots.
* For a drifting camera each frame is fitted to the shot's first frame; a frame whose fit
  fails is stored with `transform_valid=False` and its geometry is never trusted.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from pipeline.cache.pass1_store import FrameRecord, Pass1Store
from pipeline.intake.frames import DecodedFrame, FrameSource
from pipeline.intake.geometry_gate import CameraMode, ShotCutDetector
from pipeline.intake.prefetch import Prefetch
from pipeline.intake.stabilization import AffineFit, ReferenceMatcher, apply_to_points
from pipeline.vision.detector import Detection, Stage1Detector
from pipeline.vision.tracker import ByteTrackAdapter, TrackedObject
from shared.config import AppConfig
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, CoordinateSpace
from shared.identity import canonical_track_id
from shared.schemas.tracks import Pass1TrackObservation

ProgressCallback = Callable[[int, float], None]

# People used to set the geometry gate's scale: confident and not cut off by the frame.
SCALE_MIN_CONFIDENCE = 0.5
BORDER_MARGIN_PX = 4.0


@dataclass
class Pass1Summary:
    frames: int = 0
    shots: list[str] = field(default_factory=list)
    detections: int = 0
    invalid_transform_frames: int = 0
    person_heights_px: list[float] = field(default_factory=list)
    stabilization_residuals_px: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    detect_s: float = 0.0
    decode_wait_s: float = 0.0
    track_s: float = 0.0
    store_s: float = 0.0

    @property
    def median_person_height_px(self) -> float | None:
        return statistics.median(self.person_heights_px) if self.person_heights_px else None

    @property
    def stabilization_residual_p95_px(self) -> float | None:
        if not self.stabilization_residuals_px:
            return None
        return float(np.percentile(self.stabilization_residuals_px, 95))


class VisionPass:
    def __init__(
        self,
        detector: Stage1Detector,
        tracker: ByteTrackAdapter,
        config: AppConfig,
        *,
        camera_mode: CameraMode,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.camera_mode = camera_mode

    def run(
        self,
        source: FrameSource,
        store: Pass1Store,
        *,
        progress: ProgressCallback | None = None,
    ) -> Pass1Summary:
        summary = Pass1Summary()
        started = time.perf_counter()
        cuts = ShotCutDetector()
        shot_index = -1
        matcher: ReferenceMatcher | None = None

        for frame, detections in self._detected(source, summary):
            cut = cuts.is_cut(frame.image)  # always called, so it sees every frame
            if shot_index < 0 or cut:
                shot_index += 1
                summary.shots.append(f"s{shot_index}")
                self.tracker.reset()
                matcher = (
                    ReferenceMatcher(frame.image)
                    if self.camera_mode is CameraMode.STABILIZE
                    else None
                )
            shot_id = summary.shots[-1]
            tick = time.perf_counter()
            fit = matcher.fit(frame.image) if matcher is not None else None
            transform_valid = self._transform_valid(fit)
            if fit is not None and fit.residual_rms_px is not None:
                summary.stabilization_residuals_px.append(fit.residual_rms_px)
            tracked = self.tracker.update(detections)
            observations = [
                self._observation(frame, shot_id, item, fit if transform_valid else None)
                for item in tracked
            ]
            stored = time.perf_counter()
            summary.track_s += stored - tick
            store.add_frame(
                FrameRecord(
                    frame_index=frame.frame_index,
                    source_index=frame.source_index,
                    video_time_s=frame.time_s,
                    shot_id=shot_id,
                    transform_valid=transform_valid,
                    detection_count=len(detections),
                ),
                observations,
            )
            summary.store_s += time.perf_counter() - stored
            summary.frames += 1
            summary.detections += len(detections)
            summary.invalid_transform_frames += not transform_valid
            summary.person_heights_px.extend(
                d.height for d in detections if self._reliable_person(d, frame.image.shape)
            )
            if progress is not None:
                progress(summary.frames, frame.time_s)

        store.commit()
        summary.elapsed_s = time.perf_counter() - started
        return summary

    def _detected(
        self, source: FrameSource, summary: Pass1Summary
    ) -> Iterator[tuple[DecodedFrame, list[Detection]]]:
        batch: list[DecodedFrame] = []
        batch_size = max(1, self.config.stage1.batch_size)

        def flush() -> Iterator[tuple[DecodedFrame, list[Detection]]]:
            tick = time.perf_counter()
            results = self.detector.predict([frame.image for frame in batch])
            summary.detect_s += time.perf_counter() - tick
            yield from zip(batch, results, strict=True)
            batch.clear()

        frames = Prefetch(source)
        for frame in frames:
            batch.append(frame)
            if len(batch) == batch_size:
                yield from flush()
        if batch:
            yield from flush()
        summary.decode_wait_s = frames.wait_s

    def _transform_valid(self, fit: AffineFit | None) -> bool:
        if self.camera_mode is CameraMode.UNMEASURABLE:
            return False
        if self.camera_mode is CameraMode.FIXED:
            return True
        return fit is not None and fit.valid

    @staticmethod
    def _reliable_person(detection: Detection, shape: tuple[int, ...]) -> bool:
        height, width = shape[:2]
        return (
            detection.class_name == PERSON_CLASS
            and detection.confidence >= SCALE_MIN_CONFIDENCE
            and detection.y1 > BORDER_MARGIN_PX
            and detection.y2 < height - BORDER_MARGIN_PX
            and detection.x1 > BORDER_MARGIN_PX
            and detection.x2 < width - BORDER_MARGIN_PX
        )

    @staticmethod
    def _observation(
        frame: DecodedFrame, shot_id: str, item: TrackedObject, fit: AffineFit | None
    ) -> Pass1TrackObservation:
        d = item.detection
        oriented = Box2(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2, space=CoordinateSpace.ORIENTED)
        corners = np.array([[d.x1, d.y1], [d.x2, d.y1], [d.x2, d.y2], [d.x1, d.y2]])
        anchor = np.array([[(d.x1 + d.x2) / 2, d.y2]])  # bottom-centre ground contact
        if fit is not None and fit.matrix is not None:
            matrix = fit.matrix.tolist()
            corners = apply_to_points(matrix, corners)
            anchor = apply_to_points(matrix, anchor)
        (x1, y1), (x2, y2) = corners.min(axis=0), corners.max(axis=0)
        return Pass1TrackObservation(
            shot_id=shot_id,
            frame_index=frame.frame_index,
            video_time_s=frame.time_s,
            track_id=item.track_id,
            canonical_track_id=canonical_track_id(shot_id, item.track_id),
            object_class=d.class_name,
            box_reference=Box2(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                space=CoordinateSpace.REFERENCE,
            ),
            anchor_reference=Point2(
                x=float(anchor[0, 0]), y=float(anchor[0, 1]), space=CoordinateSpace.REFERENCE
            ),
            confidence=d.confidence,
            box_oriented=oriented,
        )
