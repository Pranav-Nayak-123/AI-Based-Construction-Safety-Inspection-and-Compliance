"""Shared evaluator contract and helpers for R1–R5.

An evaluator turns one `FrameContext` into raw per-frame `RuleResult`s:

* one result per worker when the rule can be assessed per worker, or
* exactly one frame-level result (`track_id=None`) when it cannot — e.g. the feed is
  unsupported or there is nobody to assess.

`evaluated_alert` here means "the alert condition holds on this frame". Debounce,
hysteresis and incident creation are the engine's job (see `temporal.py`).

Design rule shared by every evaluator: return `inconclusive` only when the unknown
quantity could change the outcome. A worker whose whole distance interval is outside the
alert band is clear even if the machine's operating state is unknown.
"""

from __future__ import annotations

from typing import Protocol

from shared.coordinates import DistanceInterval
from shared.enums import DistanceBand, HelmetState, RuleId, RuleStatus, VestState
from shared.schemas.frame import FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.tracks import HelmetRecord, VestRecord

# When several per-target outcomes exist for one worker (several zones, machines or
# edges), the worker's result is the most safety-relevant one.
_STATUS_PRIORITY = {
    RuleStatus.EVALUATED_ALERT: 0,
    RuleStatus.INCONCLUSIVE: 1,
    RuleStatus.UNSUPPORTED: 2,
    RuleStatus.EVALUATED_CLEAR: 3,
    RuleStatus.NOT_APPLICABLE: 4,
}


class RuleEvaluator(Protocol):
    rule_id: RuleId

    def evaluate(self, context: FrameContext) -> list[RuleResult]: ...


class ResultFactory:
    """Builds results for one rule and frame so evaluators stay declarative."""

    def __init__(self, rule_id: RuleId, basis: str, context: FrameContext) -> None:
        self.rule_id = rule_id
        self.basis = basis
        self.time_s = context.video_time_s

    def make(
        self,
        status: RuleStatus,
        reason_code: str,
        *,
        track_id: str | None = None,
        object_id: str | None = None,
        proposal_ids: tuple[str, ...] = (),
        confidence: float | None = None,
        distance: DistanceInterval | None = None,
        relative_band: DistanceBand | None = None,
    ) -> RuleResult:
        return RuleResult(
            rule_id=self.rule_id,
            status=status,
            reason_code=reason_code,
            basis=self.basis,
            track_id=track_id,
            object_id=object_id,
            proposal_ids=proposal_ids,
            observed_from_s=self.time_s,
            observed_to_s=self.time_s,
            confidence=confidence,
            distance=distance,
            relative_band=relative_band,
        )

    def frame(self, status: RuleStatus, reason_code: str) -> list[RuleResult]:
        return [self.make(status, reason_code)]


def most_relevant(results: list[RuleResult]) -> RuleResult:
    """Pick the worker-level result from per-target results (alert > inconclusive > clear).

    Among alerts the closest target wins, so the incident card names the nearest hazard.
    """
    if not results:
        raise ValueError("most_relevant needs at least one result")

    def key(result: RuleResult) -> tuple[int, float]:
        closeness = result.distance.median_wh if result.distance is not None else 0.0
        return (_STATUS_PRIORITY[result.status], closeness)

    return min(results, key=key)


def helmet_assessable(record: HelmetRecord | None) -> bool:
    return record is not None and record.visible and record.state is not HelmetState.UNKNOWN


def vest_assessable(record: VestRecord | None) -> bool:
    return record is not None and record.visible and record.state is not VestState.UNKNOWN
