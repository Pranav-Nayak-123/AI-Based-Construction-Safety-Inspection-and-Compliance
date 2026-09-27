"""Decide how far a feed's geometry can be trusted (v7 §7.2, §36.3).

Two gates:

1. Before detection, the sampled drift probe classifies the camera as fixed (use frames
   as-is), drifting (estimate a per-frame stabilisation ledger) or unmeasurable.
2. After detection, the residual drift is normalised by the median reliable person height:
   R3–R5 need `residual_drift_wh <= geometry_residual_limit_wh` (0.05 WH by default).
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

from pipeline.intake.stabilization import DriftProbe, StabilizationLedger
from shared.config import IntakeConfig
from shared.coordinates import StrictModel


class CameraMode(StrEnum):
    FIXED = "fixed"
    STABILIZE = "stabilize"
    UNMEASURABLE = "unmeasurable"


class DriftDecision(StrictModel):
    mode: CameraMode
    peak_fraction: float | None
    reason_code: str


def classify_drift(probe: DriftProbe, config: IntakeConfig) -> DriftDecision:
    fraction = probe.peak_fraction
    if fraction is None:
        return DriftDecision(
            mode=CameraMode.UNMEASURABLE, peak_fraction=None, reason_code="drift_unmeasurable"
        )
    if fraction <= config.native_drift_limit_diagonal:
        return DriftDecision(
            mode=CameraMode.FIXED, peak_fraction=fraction, reason_code="native_fixed_camera"
        )
    return DriftDecision(
        mode=CameraMode.STABILIZE, peak_fraction=fraction, reason_code="coherent_drift"
    )


def residual_drift_px(
    decision: DriftDecision, probe: DriftProbe, ledger: StabilizationLedger
) -> float | None:
    """Worst remaining camera motion in canvas pixels after intake."""
    if decision.mode is CameraMode.FIXED:
        return probe.peak_displacement_px
    residuals = [
        entry.residual_rms_px
        for entry in ledger.entries
        if entry.valid and not entry.interpolated and entry.residual_rms_px is not None
    ]
    if not residuals:
        return None
    return float(np.percentile(residuals, 95))


class GeometryGate(StrictModel):
    supported: bool
    residual_drift_px: float | None
    residual_drift_wh: float | None
    median_person_height_px: float | None
    reason_code: str


def gate_geometry(
    decision: DriftDecision,
    residual_px: float | None,
    median_person_height_px: float | None,
    diagonal_px: float,
    config: IntakeConfig,
) -> GeometryGate:
    def gate(supported: bool, reason: str, residual_wh: float | None = None) -> GeometryGate:
        return GeometryGate(
            supported=supported,
            residual_drift_px=residual_px,
            residual_drift_wh=residual_wh,
            median_person_height_px=median_person_height_px,
            reason_code=reason,
        )

    if decision.mode is CameraMode.UNMEASURABLE or residual_px is None:
        return gate(False, "drift_unmeasurable")
    if (
        decision.mode is CameraMode.STABILIZE
        and residual_px > config.provisional_residual_limit_diagonal * diagonal_px
    ):
        return gate(False, "stabilization_residual_too_high")
    if not median_person_height_px or median_person_height_px <= 0:
        return gate(False, "no_reliable_person_scale")
    residual_wh = residual_px / median_person_height_px
    if residual_wh > config.geometry_residual_limit_wh:
        return gate(False, "residual_drift_exceeds_wh_limit", residual_wh)
    return gate(True, "geometry_supported", residual_wh)


class ShotCutDetector:
    """Flags hard cuts from a jump in the grey-level histogram of a thumbnail.

    A cut needs both a histogram decorrelation and a large mean pixel change, so slow
    lighting changes on a fixed camera are not mistaken for cuts.
    """

    def __init__(
        self,
        *,
        correlation_threshold: float = 0.5,
        mean_change_threshold: float = 25.0,
        thumbnail: tuple[int, int] = (64, 36),
    ) -> None:
        self.correlation_threshold = correlation_threshold
        self.mean_change_threshold = mean_change_threshold
        self.thumbnail = thumbnail
        self._previous: tuple[np.ndarray, np.ndarray] | None = None

    def is_cut(self, image_bgr: np.ndarray) -> bool:
        grey = cv2.cvtColor(
            cv2.resize(image_bgr, self.thumbnail, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        histogram = cv2.calcHist([grey], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)
        previous = self._previous
        self._previous = (grey, histogram)
        if previous is None:
            return False
        correlation = cv2.compareHist(previous[1], histogram, cv2.HISTCMP_CORREL)
        mean_change = float(np.mean(cv2.absdiff(previous[0], grey)))
        return correlation < self.correlation_threshold and mean_change > self.mean_change_threshold
