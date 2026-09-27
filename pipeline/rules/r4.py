"""R4 — approximate machinery proximity (v7 §11 R4, §40.5).

Alert only when an *active* machine's full distance interval lies inside the near band.
Unknown operating state or a band-straddling interval makes the pair inconclusive — but
only if that uncertainty could produce an alert (a stationary machine or a worker whose
whole interval is beyond the near band is clear).
"""

from __future__ import annotations

from pipeline.rules.base import ResultFactory, most_relevant
from shared.enums import DistanceBand, OperatingState, RuleId, RuleStatus
from shared.schemas.frame import FrameContext, MachineryProximity
from shared.schemas.incidents import RuleResult

ALERT_REASON = "active_machine_near_band"


class R4MachineryProximity:
    rule_id = RuleId.R4

    def __init__(self, *, basis: str) -> None:
        self.basis = basis

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        if not context.geometry_supported:
            return make.frame(RuleStatus.UNSUPPORTED, "geometry_unstable_for_feed")
        if not context.capabilities.person_detection:
            return make.frame(RuleStatus.UNSUPPORTED, "person_detection_unavailable")
        if not context.machinery:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_machinery_tracks")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_worker_machinery_overlap")
        if not context.capabilities.relative_plane:
            return make.frame(RuleStatus.INCONCLUSIVE, "relative_plane_unavailable")

        results: list[RuleResult] = []
        for worker in workers:
            track_id = worker.canonical_track_id
            pairs = sorted(
                (p for p in context.machinery_proximities if p.worker_track_id == track_id),
                key=lambda p: p.machinery_id,
            )
            if not pairs:
                results.append(
                    make.make(RuleStatus.INCONCLUSIVE, "proximity_unavailable", track_id=track_id)
                )
                continue
            results.append(most_relevant([self._pair(make, track_id, pair) for pair in pairs]))
        return results

    def _pair(self, make: ResultFactory, track_id: str, pair: MachineryProximity) -> RuleResult:
        status, reason = _classify(pair)
        return make.make(
            status,
            reason,
            track_id=track_id,
            object_id=pair.machinery_id,
            distance=pair.distance,
            relative_band=pair.band,
        )


def _classify(pair: MachineryProximity) -> tuple[RuleStatus, str]:
    if not pair.compatible_plane:
        return RuleStatus.INCONCLUSIVE, "target_plane_incompatible"
    if pair.band in (DistanceBand.CLEAR, DistanceBand.CAUTION):
        return RuleStatus.EVALUATED_CLEAR, "outside_active_near_band"
    if pair.operating_state is OperatingState.STATIONARY:
        return RuleStatus.EVALUATED_CLEAR, "outside_active_near_band"
    if pair.band is DistanceBand.INDETERMINATE:
        return RuleStatus.INCONCLUSIVE, "inconclusive_distance"
    if pair.operating_state is OperatingState.UNKNOWN:
        return RuleStatus.INCONCLUSIVE, "operating_state_unknown"
    return RuleStatus.EVALUATED_ALERT, ALERT_REASON
