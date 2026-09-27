"""R3 — restricted-zone intrusion (v7 §11 R3, §40.4).

Membership is a ground-anchor-in-polygon test in reference image space, so R3 needs a
stable camera and a scene state but not the relative plane.
"""

from __future__ import annotations

from pipeline.rules.base import ResultFactory, most_relevant
from shared.enums import RuleId, RuleStatus
from shared.schemas.frame import FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.scene import ProposalType

ALERT_REASON = "restricted_zone_entry"


class R3RestrictedZone:
    rule_id = RuleId.R3

    def __init__(self, *, basis: str) -> None:
        self.basis = basis

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        if not context.geometry_supported:
            return make.frame(RuleStatus.UNSUPPORTED, "geometry_unstable_for_feed")
        if not context.capabilities.person_detection:
            return make.frame(RuleStatus.UNSUPPORTED, "person_detection_unavailable")
        if not context.capabilities.scene_state:
            return make.frame(RuleStatus.INCONCLUSIVE, "restricted_zone_scene_state_unavailable")
        zone_ids = sorted(
            proposal.proposal_id
            for proposal in context.scene_proposals
            if proposal.proposal_type is ProposalType.RESTRICTED_ZONE
        )
        if not zone_ids:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_supported_restricted_zone")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_person_tracks")

        results: list[RuleResult] = []
        for worker in workers:
            track_id = worker.canonical_track_id
            memberships = {
                membership.proposal_id: membership
                for membership in context.zone_memberships
                if membership.worker_track_id == track_id
            }
            per_zone: list[RuleResult] = []
            for zone_id in zone_ids:
                membership = memberships.get(zone_id)
                if membership is None or membership.inside is None:
                    status, reason = RuleStatus.INCONCLUSIVE, "worker_anchor_unavailable"
                elif membership.boundary_uncertain:
                    status, reason = RuleStatus.INCONCLUSIVE, "zone_boundary_uncertain"
                elif membership.inside:
                    status, reason = RuleStatus.EVALUATED_ALERT, ALERT_REASON
                else:
                    status, reason = RuleStatus.EVALUATED_CLEAR, "worker_outside_restricted_zone"
                per_zone.append(
                    make.make(status, reason, track_id=track_id, proposal_ids=(zone_id,))
                )
            results.append(most_relevant(per_zone))
        return results
