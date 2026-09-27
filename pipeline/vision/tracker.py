"""ByteTrack over stage-1 detections (v7 §8.1, §33.2).

Wraps Ultralytics' `BYTETracker` so detection and tracking stay separate stages: the
detector runs batched, the tracker consumes one frame at a time. Output boxes are the
matched *detections* (not Kalman-smoothed predictions), so anchors and evidence crops
come from what the detector actually saw. Ultralytics >= 8.4.163 reports each track's
detection index in full detection-set space, which is what links the two.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from pipeline.vision.detector import Detection
from shared.config import TrackerConfig


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    detection: Detection


class _Detections:
    """The minimal `Results.boxes`-like view BYTETracker reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2
        xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2
        return xywh

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: np.ndarray) -> _Detections:
        return _Detections(self.xyxy[index], self.conf[index], self.cls[index])


class ByteTrackAdapter:
    def __init__(self, config: TrackerConfig, fps: float, class_names: Sequence[str]) -> None:
        self.config = config
        self.fps = fps
        self.class_names = list(class_names)
        self._class_index = {name: index for index, name in enumerate(self.class_names)}
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> BYTETracker:
        args = SimpleNamespace(
            track_high_thresh=self.config.track_high_threshold,
            track_low_thresh=self.config.track_low_threshold,
            new_track_thresh=self.config.new_track_threshold,
            match_thresh=self.config.match_threshold,
            track_buffer=round(self.config.buffer_seconds * self.fps),  # in processed frames
            fuse_score=True,
        )
        return BYTETracker(args)

    def reset(self) -> None:
        """Forget all tracks (shot boundary): IDs are never continued across shots."""
        self._tracker = self._new_tracker()

    def update(self, detections: Sequence[Detection]) -> list[TrackedObject]:
        if detections:
            xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], np.float32)
            conf = np.array([d.confidence for d in detections], np.float32)
            cls = np.array([self._class_index[d.class_name] for d in detections], np.float32)
        else:
            xyxy = np.zeros((0, 4), np.float32)
            conf = np.zeros((0,), np.float32)
            cls = np.zeros((0,), np.float32)
        tracks = self._tracker.update(_Detections(xyxy, conf, cls))
        if len(tracks) == 0 or len(detections) == 0:
            return []

        tracked: list[TrackedObject] = []
        for row in tracks:  # x1 y1 x2 y2 track_id score cls detection_index
            track_id, index = int(row[4]), int(row[7])
            if not 0 <= index < len(detections):
                continue
            tracked.append(TrackedObject(track_id=track_id, detection=detections[index]))
        tracked.sort(key=lambda item: item.track_id)
        return tracked
