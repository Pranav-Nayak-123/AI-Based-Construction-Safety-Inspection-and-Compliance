"""Per-track temporal smoothing of PPE predictions (v7 §8.3, §33.2).

* An exponential moving average per track and region over (present, absent).
* A prediction whose most likely class is `unknown` is not accepted: it neither confirms
  nor contradicts anything, so unknown never votes negative.
* A state is only reported after `minimum_observations` accepted observations.
* When the current observation is not accepted, the record says `visible=False`, so the
  rules treat that frame as not assessable instead of repeating a stale verdict.

Smoothing removes flicker; it cannot fix a classifier that is consistently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.vision.ppe_model import PPEPrediction
from shared.enums import HelmetState, VestState
from shared.schemas.tracks import HelmetRecord, VestRecord

UNKNOWN_INDEX = 2


@dataclass
class _RegionState:
    ema: np.ndarray | None = None  # (present, absent), sums to 1
    accepted: int = 0
    last_visible: bool = False


@dataclass
class _TrackState:
    head: _RegionState = field(default_factory=_RegionState)
    torso: _RegionState = field(default_factory=_RegionState)


class TemporalPPEFilter:
    def __init__(self, *, alpha: float, minimum_observations: int, confidence_floor: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.minimum_observations = minimum_observations
        self.confidence_floor = confidence_floor
        self._tracks: dict[str, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, track_id: str, prediction: PPEPrediction) -> tuple[HelmetRecord, VestRecord]:
        state = self._tracks.setdefault(track_id, _TrackState())
        head_visible, head_index, head_confidence = self._update(state.head, prediction.head)
        torso_visible, torso_index, torso_confidence = self._update(state.torso, prediction.torso)
        helmet = HelmetRecord(
            state=(HelmetState.HELMET, HelmetState.NO_HELMET, HelmetState.UNKNOWN)[head_index],
            confidence=head_confidence,
            visible=head_visible,
        )
        vest = VestRecord(
            state=(VestState.VEST, VestState.NO_VEST, VestState.UNKNOWN)[torso_index],
            confidence=torso_confidence,
            visible=torso_visible,
        )
        return helmet, vest

    def current(self, track_id: str) -> tuple[HelmetRecord, VestRecord]:
        """The smoothed state without a new observation (for frames not classified)."""
        state = self._tracks.get(track_id)
        if state is None:
            return self.unobserved()
        records = []
        for region, labels in (
            (state.head, (HelmetState.HELMET, HelmetState.NO_HELMET, HelmetState.UNKNOWN)),
            (state.torso, (VestState.VEST, VestState.NO_VEST, VestState.UNKNOWN)),
        ):
            if region.ema is None or region.accepted < self.minimum_observations:
                records.append((labels[2], 0.0, region.last_visible))
                continue
            index = int(np.argmax(region.ema))
            confidence = float(region.ema[index])
            if confidence < self.confidence_floor:
                index = 2
            records.append((labels[index], confidence, region.last_visible))
        (
            (head_state, head_confidence, head_visible),
            (torso_state, torso_confidence, torso_visible),
        ) = records
        return (
            HelmetRecord(state=head_state, confidence=head_confidence, visible=head_visible),
            VestRecord(state=torso_state, confidence=torso_confidence, visible=torso_visible),
        )

    def unobserved(self) -> tuple[HelmetRecord, VestRecord]:
        """Record for a person too small or truncated to classify at all."""
        return (
            HelmetRecord(state=HelmetState.UNKNOWN, confidence=0.0, visible=False),
            VestRecord(state=VestState.UNKNOWN, confidence=0.0, visible=False),
        )

    def _update(self, region: _RegionState, probabilities: np.ndarray) -> tuple[bool, int, float]:
        accepted = int(np.argmax(probabilities)) != UNKNOWN_INDEX
        region.last_visible = accepted
        if accepted:
            known = probabilities[:2] / max(float(probabilities[:2].sum()), 1e-9)
            region.ema = (
                known if region.ema is None else (1 - self.alpha) * region.ema + self.alpha * known
            )
            region.accepted += 1
        if region.ema is None or region.accepted < self.minimum_observations:
            return accepted, UNKNOWN_INDEX, float(probabilities[UNKNOWN_INDEX])
        index = int(np.argmax(region.ema))
        confidence = float(region.ema[index])
        if confidence < self.confidence_floor:
            return accepted, UNKNOWN_INDEX, confidence
        return accepted, index, confidence
