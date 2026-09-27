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

import cv2
import numpy as np

from pipeline.cache.pass1_store import FrameRecord, Pass1Store
from pipeline.data.ppe_crops import Box as CropBox
from pipeline.intake.frames import DecodedFrame, FrameSource
from pipeline.intake.geometry_gate import CameraMode, ShotCutDetector
from pipeline.intake.prefetch import Prefetch
from pipeline.intake.stabilization import AffineFit, ReferenceMatcher, apply_to_points
from pipeline.vision.detector import Detection, Stage1Detector
from pipeline.vision.ppe_model import PPEClassifier, PPEPrediction
from pipeline.vision.ppe_temporal import TemporalPPEFilter
from pipeline.vision.tracker import ByteTrackAdapter, TrackedObject
from shared.config import AppConfig
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, CoordinateSpace
from shared.identity import canonical_track_id
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation, VestRecord

ProgressCallback = Callable[[int, float], None]

# People used to set the geometry gate's scale: confident and not cut off by the frame.
SCALE_MIN_CONFIDENCE = 0.5
BORDER_MARGIN_PX = 4.0


MOTION_SCALE = 0.25  # motion energy is measured on a quarter-size grey frame
# A detected hard hat on the head is strong positive evidence; not seeing one is no evidence.
HARD_HAT_HEAD = np.array([0.93, 0.05, 0.02])
HEAD_REGION = 0.35  # top fraction of the person box where a worn hard hat sits


def _has_hard_hat(person: Detection, evidence: list[Detection], minimum: float) -> bool:
    top = person.y1 + HEAD_REGION * person.height
    margin = 0.15 * (person.x2 - person.x1)
    for hat in evidence:
        cx, cy = (hat.x1 + hat.x2) / 2, (hat.y1 + hat.y2) / 2
        if (
            hat.confidence >= minimum
            and person.x1 - margin <= cx <= person.x2 + margin
            and person.y1 - 0.1 * person.height <= cy <= top
        ):
            return True
    return False


@dataclass
class _Staged:
    frame: DecodedFrame
    shot_id: str
    detections: list[Detection]
    tracked: list[TrackedObject]
    fit: AffineFit | None
    transform_valid: bool
    energy: dict[int, float]
    evidence: list[Detection]


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
    ppe_s: float = 0.0
    helmet_evidence_overrides: int = 0
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
        ppe: PPEClassifier | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.camera_mode = camera_mode
        self.ppe = ppe
        self.ppe_filter = TemporalPPEFilter(
            alpha=config.ppe.temporal_alpha,
            minimum_observations=config.ppe.minimum_observations,
            confidence_floor=config.ppe.confidence_floor,
        )

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

        staged: list[_Staged] = []
        previous_grey: np.ndarray | None = None
        for frame, detections in self._detected(source, summary):
            cut = cuts.is_cut(frame.image)  # always called, so it sees every frame
            if shot_index < 0 or cut:
                self._flush(staged, store, summary, progress)
                shot_index += 1
                summary.shots.append(f"s{shot_index}")
                self.tracker.reset()
                self.ppe_filter.reset()
                previous_grey = None
                matcher = (
                    ReferenceMatcher(frame.image)
                    if self.camera_mode is CameraMode.STABILIZE
                    else None
                )
            tick = time.perf_counter()
            fit = matcher.fit(frame.image) if matcher is not None else None
            transform_valid = self._transform_valid(fit)
            if fit is not None and fit.residual_rms_px is not None:
                summary.stabilization_residuals_px.append(fit.residual_rms_px)
            evidence_classes = set(self.config.stage1.evidence_classes)
            evidence = [d for d in detections if d.class_name in evidence_classes]
            trackable = [d for d in detections if d.class_name not in evidence_classes]
            tracked = self.tracker.update(trackable)
            grey = cv2.resize(
                cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY),
                None,
                fx=MOTION_SCALE,
                fy=MOTION_SCALE,
                interpolation=cv2.INTER_AREA,
            )
            energy = self._motion_energy(tracked, grey, previous_grey)
            previous_grey = grey
            summary.track_s += time.perf_counter() - tick
            staged.append(
                _Staged(
                    frame,
                    summary.shots[-1],
                    trackable,
                    tracked,
                    fit,
                    transform_valid,
                    energy,
                    evidence,
                )
            )
            if len(staged) >= max(1, self.config.stage1.batch_size):
                self._flush(staged, store, summary, progress)
        self._flush(staged, store, summary, progress)

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

    def _flush(
        self,
        staged: list[_Staged],
        store: Pass1Store,
        summary: Pass1Summary,
        progress: ProgressCallback | None,
    ) -> None:
        """Classify PPE for all staged frames in one batch, then store them in order."""
        if not staged:
            return
        tick = time.perf_counter()
        crops, owners = [], []
        for position, item in enumerate(staged):
            if item.frame.frame_index % self.config.ppe.classify_every_frames:
                continue  # the temporal filter bridges skipped frames
            height, width = item.frame.image.shape[:2]
            for tracked in item.tracked:
                d = tracked.detection
                if self.ppe is None or d.class_name != PERSON_CLASS:
                    continue
                if not self._assessable(d):
                    continue
                box = CropBox(d.x1, d.y1, d.x2, d.y2).padded(width, height)
                crop = item.frame.image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
                if crop.size:
                    crops.append(crop)
                    owners.append((position, tracked.track_id))
        predictions = {}
        if crops and self.ppe is not None:
            predictions = dict(zip(owners, self.ppe.predict(crops), strict=True))
        summary.ppe_s += time.perf_counter() - tick

        for position, item in enumerate(staged):
            observations = []
            for tracked in item.tracked:
                ppe: tuple[HelmetRecord, VestRecord] | None = None
                classified = item.frame.frame_index % self.config.ppe.classify_every_frames == 0
                if self.ppe is not None and tracked.detection.class_name == PERSON_CLASS:
                    prediction = predictions.get((position, tracked.track_id))
                    if prediction is not None and _has_hard_hat(
                        tracked.detection, item.evidence, self.config.ppe.helmet_evidence_confidence
                    ):
                        summary.helmet_evidence_overrides += int(
                            int(np.argmax(prediction.head)) != 0
                        )
                        prediction = PPEPrediction(HARD_HAT_HEAD, prediction.torso)
                    key = canonical_track_id(item.shot_id, tracked.track_id)
                    if prediction is not None:
                        ppe = self.ppe_filter.update(key, prediction)
                    elif not classified:
                        ppe = self.ppe_filter.current(key)
                    else:
                        ppe = self.ppe_filter.unobserved()
                observations.append(
                    self._observation(
                        item.frame,
                        item.shot_id,
                        tracked,
                        item.fit if item.transform_valid else None,
                        ppe,
                        item.energy.get(tracked.track_id),
                    )
                )
            stored = time.perf_counter()
            store.add_frame(
                FrameRecord(
                    frame_index=item.frame.frame_index,
                    source_index=item.frame.source_index,
                    video_time_s=item.frame.time_s,
                    shot_id=item.shot_id,
                    transform_valid=item.transform_valid,
                    detection_count=len(item.detections),
                ),
                observations,
            )
            summary.store_s += time.perf_counter() - stored
            summary.frames += 1
            summary.detections += len(item.detections)
            summary.invalid_transform_frames += not item.transform_valid
            summary.person_heights_px.extend(
                d.height
                for d in item.detections
                if self._reliable_person(d, item.frame.image.shape)
            )
            if progress is not None:
                progress(summary.frames, item.frame.time_s)
        staged.clear()

    def _assessable(self, detection: Detection) -> bool:
        """PPE bands assume an upright worker of useful size (v7 §8.2 `unknown` cases)."""
        width = max(detection.x2 - detection.x1, 1e-6)
        return (
            detection.height >= self.config.ppe.minimum_person_height_px
            and detection.height / width >= self.config.ppe.minimum_upright_aspect
        )

    def _motion_energy(
        self, tracked: list[TrackedObject], grey: np.ndarray, previous: np.ndarray | None
    ) -> dict[int, float]:
        """Mean absolute intensity change inside each machine box (articulation cue)."""
        if previous is None:
            return {}
        energy: dict[int, float] = {}
        people = [t.detection for t in tracked if t.detection.class_name == PERSON_CLASS]
        for item in tracked:
            d = item.detection
            if d.class_name not in self.config.stage1.machinery_classes:
                continue
            x1, y1 = int(d.x1 * MOTION_SCALE), int(d.y1 * MOTION_SCALE)
            x2, y2 = int(np.ceil(d.x2 * MOTION_SCALE)), int(np.ceil(d.y2 * MOTION_SCALE))
            now, before = grey[y1:y2, x1:x2], previous[y1:y2, x1:x2]
            if not now.size or now.shape != before.shape:
                continue
            # Workers walking in front of a machine are not the machine moving.
            keep = np.ones(now.shape, bool)
            for person in people:
                px1 = int(person.x1 * MOTION_SCALE) - x1
                py1 = int(person.y1 * MOTION_SCALE) - y1
                px2 = int(np.ceil(person.x2 * MOTION_SCALE)) - x1
                py2 = int(np.ceil(person.y2 * MOTION_SCALE)) - y1
                keep[max(py1, 0) : max(py2, 0), max(px1, 0) : max(px2, 0)] = False
            if keep.mean() < 0.3:
                continue  # mostly hidden behind people: no reliable articulation evidence
            energy[item.track_id] = float(cv2.absdiff(now, before)[keep].mean() / 255.0)
        return energy

    @staticmethod
    def _observation(
        frame: DecodedFrame,
        shot_id: str,
        item: TrackedObject,
        fit: AffineFit | None,
        ppe: tuple[HelmetRecord, VestRecord] | None = None,
        motion_energy: float | None = None,
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
            helmet=ppe[0] if ppe else None,
            vest=ppe[1] if ppe else None,
            confidence=d.confidence,
            box_oriented=oriented,
            motion_energy=motion_energy,
        )
