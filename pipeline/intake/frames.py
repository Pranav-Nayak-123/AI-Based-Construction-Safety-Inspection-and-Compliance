"""Decode a clip at the processing rate onto a fixed letterboxed canvas (v7 §7.3, §35).

Frames are decoded and letterboxed in memory instead of transcoding a working copy first:
that saves a full H.264 encode per clip, which matters for FR-1's five-minute budget. The
raw → oriented-canvas transform is exact and recorded, so every detection can be mapped
back to source pixels.

Frame selection is deterministic: the source timeline is cut into 1/target_fps buckets
and the first frame in each bucket is kept, with its source timestamp. Nothing is dropped
adaptively.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from shared.enums import CoordinateSpace
from shared.errors import PipelineError
from shared.schemas.geometry import TransformRecord

IMPLEMENTATION_VERSION = "frames-v1"
MAX_CONSECUTIVE_DECODE_FAILURES = 5
TRUNCATION_TOLERANCE = 0.95


@dataclass(frozen=True)
class Letterbox:
    """Uniform scale plus centred padding from source pixels onto the canvas."""

    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int

    @classmethod
    def fit(cls, source: tuple[int, int], canvas: tuple[int, int]) -> Letterbox:
        source_w, source_h = source
        canvas_w, canvas_h = canvas
        if min(source_w, source_h, canvas_w, canvas_h) <= 0:
            raise ValueError("letterbox sizes must be positive")
        scale = min(canvas_w / source_w, canvas_h / source_h)
        resized_w = min(canvas_w, round(source_w * scale))
        resized_h = min(canvas_h, round(source_h * scale))
        return cls(
            source_width=source_w,
            source_height=source_h,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            resized_width=resized_w,
            resized_height=resized_h,
            offset_x=(canvas_w - resized_w) // 2,
            offset_y=(canvas_h - resized_h) // 2,
        )

    @property
    def is_identity(self) -> bool:
        return (self.source_width, self.source_height) == (self.canvas_width, self.canvas_height)

    def forward(self) -> np.ndarray:
        sx = self.resized_width / self.source_width
        sy = self.resized_height / self.source_height
        return np.array(
            [[sx, 0.0, self.offset_x], [0.0, sy, self.offset_y], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.forward())

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self.is_identity:
            return image
        height, width = image.shape[:2]
        if (width, height) != (self.source_width, self.source_height):
            raise PipelineError(
                "FRAME_SIZE_CHANGED",
                f"frame is {width}x{height}, expected {self.source_width}x{self.source_height}",
            )
        # Area averaging only pays for itself on strong downscales; linear is ~3x faster.
        strong_downscale = self.resized_width < width / 2
        interpolation = cv2.INTER_AREA if strong_downscale else cv2.INTER_LINEAR
        resized = cv2.resize(
            image, (self.resized_width, self.resized_height), interpolation=interpolation
        )
        if (self.resized_width, self.resized_height) == (self.canvas_width, self.canvas_height):
            return resized
        return cv2.copyMakeBorder(
            resized,
            self.offset_y,
            self.canvas_height - self.resized_height - self.offset_y,
            self.offset_x,
            self.canvas_width - self.resized_width - self.offset_x,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    def record(self) -> TransformRecord:
        return TransformRecord(
            transform_id="raw-to-oriented",
            frame_index=None,
            source_space=CoordinateSpace.RAW,
            destination_space=CoordinateSpace.ORIENTED,
            kind="letterbox",
            forward_3x3=self.forward().tolist(),
            inverse_3x3=self.inverse().tolist(),
            valid=True,
            implementation_version=IMPLEMENTATION_VERSION,
        )


@dataclass(frozen=True)
class DecodedFrame:
    frame_index: int  # position among processed frames
    source_index: int  # position in the source stream
    time_s: float  # source timestamp
    image: np.ndarray  # BGR on the oriented canvas


@dataclass
class FrameSource:
    """Iterate a clip's processed frames. Re-iterating decodes again, identically."""

    path: Path
    target_fps: float
    canvas_size: tuple[int, int] = (1920, 1080)
    expected_frames: int | None = None
    letterbox: Letterbox | None = field(default=None, init=False)
    decoded_count: int = field(default=0, init=False)
    warnings: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

    def __iter__(self) -> Iterator[DecodedFrame]:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise PipelineError("DECODE_OPEN_FAILED", str(self.path))
        self.decoded_count = 0
        self.warnings = []
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        last_bucket = -1
        last_time = -math.inf
        emitted = 0
        source_index = -1
        failures = 0
        try:
            while True:
                # grab() demuxes and decodes; retrieve() (colour conversion, ~5 ms at
                # 1440p) is only paid for frames that are kept.
                ok = capture.grab()
                if not ok:
                    failures += 1
                    remaining = self.expected_frames is None or (
                        source_index + 1 < self.expected_frames
                    )
                    if failures >= MAX_CONSECUTIVE_DECODE_FAILURES or not remaining:
                        break
                    continue
                failures = 0
                source_index += 1
                self.decoded_count += 1
                time_s = self._timestamp(capture, source_index, source_fps, last_time)
                last_time = time_s
                bucket = math.floor(time_s * self.target_fps + 1e-6)
                if bucket <= last_bucket:
                    continue
                ok, image = capture.retrieve()
                if not ok:
                    self.warnings.append(f"frame {source_index} could not be converted")
                    continue
                last_bucket = bucket
                if self.letterbox is None:
                    height, width = image.shape[:2]
                    self.letterbox = Letterbox.fit((width, height), self.canvas_size)
                yield DecodedFrame(
                    frame_index=emitted,
                    source_index=source_index,
                    time_s=time_s,
                    image=self.letterbox.apply(image),
                )
                emitted += 1
        finally:
            capture.release()
        if self.decoded_count == 0:
            raise PipelineError("DECODE_NO_FRAMES", str(self.path))
        if (
            self.expected_frames
            and self.decoded_count < TRUNCATION_TOLERANCE * self.expected_frames
        ):
            self.warnings.append(
                f"decode_truncated: {self.decoded_count} of {self.expected_frames} frames"
            )

    @staticmethod
    def _timestamp(
        capture: cv2.VideoCapture, source_index: int, source_fps: float, last_time: float
    ) -> float:
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        time_s = position_ms / 1000.0 if math.isfinite(position_ms) else -1.0
        if time_s < 0 or (source_index > 0 and time_s <= last_time):
            # Container without usable timestamps: fall back to the nominal rate.
            time_s = source_index / source_fps if source_fps > 0 else float(source_index)
        return time_s


def read_frame_at(path: Path, source_index: int) -> np.ndarray:
    """Read one source frame (raw pixels) for evidence extraction."""
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise PipelineError("DECODE_SEEK_FAILED", f"{path} frame {source_index}")
    return image


def sample_raw_frames(path: Path, count: int) -> list[np.ndarray]:
    """Evenly spaced raw frames across the clip, for the drift probe."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError("DECODE_OPEN_FAILED", str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise PipelineError("DECODE_NO_FRAMES", str(path))
        frames: list[np.ndarray] = []
        for index in np.linspace(0, total - 1, min(count, total)).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if ok:
                frames.append(image)
    finally:
        capture.release()
    return frames
