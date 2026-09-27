"""R1 — apparent missing helmet (v7 §11 R1, §40.2)."""

from __future__ import annotations

from pipeline.rules.base import ResultFactory, helmet_assessable
from shared.enums import HelmetState, RuleId, RuleStatus
from shared.schemas.frame import FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.tracks import MappedTrackRecord

ALERT_REASON = "persistent_no_helmet"


class R1MissingHelmet:
    rule_id = RuleId.R1

    def __init__(self, *, basis: str, confidence_floor: float) -> None:
        self.basis = basis
        self.confidence_floor = confidence_floor

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        capabilities = context.capabilities
        if not capabilities.person_detection:
            return make.frame(RuleStatus.UNSUPPORTED, "person_detection_unavailable")
        if not capabilities.ppe_classification:
            return make.frame(RuleStatus.UNSUPPORTED, "helmet_classifier_unavailable")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_person_tracks")
        return [self._worker(make, worker) for worker in workers]

    def _worker(self, make: ResultFactory, worker: MappedTrackRecord) -> RuleResult:
        track_id = worker.canonical_track_id
        helmet = worker.helmet
        if helmet is None or not helmet_assessable(helmet):
            return make.make(
                RuleStatus.INCONCLUSIVE, "helmet_visibility_insufficient", track_id=track_id
            )
        if helmet.state is HelmetState.NO_HELMET:
            if helmet.confidence >= self.confidence_floor:
                return make.make(
                    RuleStatus.EVALUATED_ALERT,
                    ALERT_REASON,
                    track_id=track_id,
                    confidence=helmet.confidence,
                )
            # Weak negative evidence never becomes a clear verdict.
            return make.make(
                RuleStatus.INCONCLUSIVE,
                "helmet_confidence_below_floor",
                track_id=track_id,
                confidence=helmet.confidence,
            )
        return make.make(
            RuleStatus.EVALUATED_CLEAR,
            "helmet_not_observed_missing",
            track_id=track_id,
            confidence=helmet.confidence,
        )
