"""Camera-drift measurement and the per-frame stabilisation ledger (v7 §7.2, §36.3).

Matching follows the approach proven in `tools/cctv.py`: ORB features, Lowe ratio test,
and a RANSAC 4-DoF partial affine, which holds up better than a full homography on
multi-depth construction scenes. Workers and machines fall out as RANSAC outliers.

Corrections over the original tool (v7 §19): transform validity is recorded per frame, a
failed estimate is never replaced by a stale earlier transform, and only gaps of at most
`max_gap` frames bounded by valid transforms on both sides are interpolated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from shared.coordinates import StrictModel
from shared.enums import CoordinateSpace
from shared.schemas.geometry import TransformRecord

IMPLEMENTATION_VERSION = "stabilization-v1"
MATCH_WIDTH = 960
RATIO_TEST = 0.75


@dataclass(frozen=True)
class AffineFit:
    """Frame → reference partial affine in canvas pixels, or why there is none."""

    matrix: np.ndarray | None
    match_count: int
    inlier_count: int
    residual_rms_px: float | None
    reason_code: str

    @property
    def valid(self) -> bool:
        return self.matrix is not None


def corner_displacement_px(matrix: np.ndarray, width: int, height: int) -> float:
    """Largest distance any image corner moves under `matrix` (pan, rotation and zoom)."""
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    moved = cv2.transform(corners, matrix).reshape(-1, 2)
    return float(np.linalg.norm(moved - corners.reshape(-1, 2), axis=1).max())


class ReferenceMatcher:
    """Fits partial affines from frames onto one reference frame."""

    def __init__(
        self,
        reference_bgr: np.ndarray,
        *,
        features: int = 3000,
        min_matches: int = 25,
        min_inliers: int = 20,
        ransac_threshold_px: float = 2.0,
    ) -> None:
        self.height, self.width = reference_bgr.shape[:2]
        self.scale = self.width / MATCH_WIDTH
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_threshold_px = ransac_threshold_px
        self._orb = cv2.ORB_create(nfeatures=features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._reference = self._features(reference_bgr)

    def _features(self, image_bgr: np.ndarray) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
        grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, (MATCH_WIDTH, round(MATCH_WIDTH * self.height / self.width)))
        return self._orb.detectAndCompute(small, None)

    def fit(self, image_bgr: np.ndarray) -> AffineFit:
        reference_points, reference_descriptors = self._reference
        points, descriptors = self._features(image_bgr)
        if reference_descriptors is None or descriptors is None or len(points) < 40:
            return AffineFit(None, 0, 0, None, "too_few_features")
        pairs = self._matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < RATIO_TEST * second.distance
        ]
        if len(good) < self.min_matches:
            return AffineFit(None, len(good), 0, None, "too_few_matches")
        source = np.float32([points[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([reference_points[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px / self.scale,
        )
        if matrix is None or inliers is None:
            return AffineFit(None, len(good), 0, None, "ransac_failed")
        mask = inliers.ravel().astype(bool)
        inlier_count = int(mask.sum())
        if inlier_count < self.min_inliers:
            return AffineFit(None, len(good), inlier_count, None, "too_few_inliers")
        projected = cv2.transform(source[mask], matrix).reshape(-1, 2)
        errors = np.linalg.norm(projected - target[mask].reshape(-1, 2), axis=1)
        residual = float(np.sqrt(np.mean(errors**2))) * self.scale
        native = matrix.astype(np.float64)
        native[:, 2] *= self.scale  # translation back to full-resolution pixels
        return AffineFit(native, len(good), inlier_count, residual, "ok")


class DriftProbe(StrictModel):
    peak_displacement_px: float | None
    diagonal_px: float
    frames_sampled: int
    frames_measured: int

    @property
    def peak_fraction(self) -> float | None:
        if self.peak_displacement_px is None:
            return None
        return self.peak_displacement_px / self.diagonal_px


def probe_drift(frames: Sequence[np.ndarray]) -> DriftProbe:
    """Peak corner displacement of sampled frames relative to the first usable one."""
    if not frames:
        raise ValueError("probe_drift needs at least one frame")
    height, width = frames[0].shape[:2]
    diagonal = float(np.hypot(width, height))
    matcher = ReferenceMatcher(frames[0])
    displacements = [
        corner_displacement_px(fit.matrix, width, height)
        for fit in (matcher.fit(frame) for frame in frames[1:])
        if fit.matrix is not None
    ]
    return DriftProbe(
        peak_displacement_px=max(displacements) if len(displacements) >= 2 else None,
        diagonal_px=diagonal,
        frames_sampled=len(frames),
        frames_measured=len(displacements),
    )


class LedgerEntry(StrictModel):
    frame_index: int
    forward_2x3: list[list[float]] | None
    valid: bool
    interpolated: bool = False
    match_count: int = 0
    inlier_count: int = 0
    residual_rms_px: float | None = None
    reason_code: str = "ok"


class StabilizationLedger(StrictModel):
    """Oriented-canvas → reference transform for every processed frame."""

    mode: Literal["identity", "stabilized"]
    reference_frame_index: int | None
    entries: tuple[LedgerEntry, ...]

    def entry(self, frame_index: int) -> LedgerEntry:
        entry = self.entries[frame_index]
        if entry.frame_index != frame_index:
            raise ValueError("ledger entries must be indexed by processed frame")
        return entry

    def valid_fraction(self) -> float:
        if not self.entries:
            return 0.0
        return sum(entry.valid for entry in self.entries) / len(self.entries)

    def records(self) -> list[TransformRecord]:
        records: list[TransformRecord] = []
        for entry in self.entries:
            forward = _as_3x3(entry.forward_2x3) if entry.forward_2x3 is not None else None
            records.append(
                TransformRecord(
                    transform_id=f"oriented-to-reference/{entry.frame_index}",
                    frame_index=entry.frame_index,
                    source_space=CoordinateSpace.ORIENTED,
                    destination_space=CoordinateSpace.REFERENCE,
                    kind="partial_affine" if self.mode == "stabilized" else "identity",
                    forward_3x3=forward.tolist() if forward is not None else None,
                    inverse_3x3=np.linalg.inv(forward).tolist() if forward is not None else None,
                    valid=entry.valid,
                    interpolated=entry.interpolated,
                    residual_rms_px=entry.residual_rms_px,
                    inlier_count=entry.inlier_count or None,
                    reason_code=entry.reason_code,
                    implementation_version=IMPLEMENTATION_VERSION,
                )
            )
        return records


_IDENTITY_2X3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def identity_ledger(frame_count: int) -> StabilizationLedger:
    return StabilizationLedger(
        mode="identity",
        reference_frame_index=None,
        entries=tuple(
            LedgerEntry(frame_index=index, forward_2x3=_IDENTITY_2X3, valid=True)
            for index in range(frame_count)
        ),
    )


def build_ledger(
    fits: Sequence[AffineFit], *, reference_frame_index: int, max_gap: int
) -> StabilizationLedger:
    """Turn per-frame fits into a ledger, interpolating only short bounded gaps."""
    entries: list[LedgerEntry] = [
        LedgerEntry(
            frame_index=index,
            forward_2x3=fit.matrix.tolist() if fit.matrix is not None else None,
            valid=fit.valid,
            match_count=fit.match_count,
            inlier_count=fit.inlier_count,
            residual_rms_px=fit.residual_rms_px,
            reason_code=fit.reason_code,
        )
        for index, fit in enumerate(fits)
    ]
    index = 0
    while index < len(entries):
        if entries[index].valid:
            index += 1
            continue
        start = index
        while index < len(entries) and not entries[index].valid:
            index += 1
        end = index  # first valid entry after the gap, or len(entries)
        bounded = start > 0 and end < len(entries)
        if not bounded or end - start > max_gap:
            reason = "transform_gap_too_long" if bounded else "gap_not_bounded"
            for gap_index in range(start, end):
                entries[gap_index] = entries[gap_index].model_copy(update={"reason_code": reason})
            continue
        before = np.array(entries[start - 1].forward_2x3)
        after = np.array(entries[end].forward_2x3)
        for gap_index in range(start, end):
            weight = (gap_index - start + 1) / (end - start + 1)
            matrix = (1 - weight) * before + weight * after
            entries[gap_index] = entries[gap_index].model_copy(
                update={
                    "forward_2x3": matrix.tolist(),
                    "valid": True,
                    "interpolated": True,
                    "reason_code": "interpolated",
                }
            )
    return StabilizationLedger(
        mode="stabilized",
        reference_frame_index=reference_frame_index,
        entries=tuple(entries),
    )


def _as_3x3(matrix_2x3: list[list[float]]) -> np.ndarray:
    return np.vstack([np.asarray(matrix_2x3, dtype=np.float64), [0.0, 0.0, 1.0]])


def apply_to_points(matrix_2x3: list[list[float]], points: np.ndarray) -> np.ndarray:
    """Map an (N, 2) array of points through a 2x3 affine."""
    matrix = np.asarray(matrix_2x3, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]
