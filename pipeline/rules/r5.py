"""R5 — possible missing fall protection near an elevated/open edge (v7 §11 R5, §40.6).

The edge's semantics come from the scene proposal; they are never inferred from the
absence of a guardrail detection. A worker whose whole distance interval is beyond the
edge band is clear whatever the edge semantics are.
"""

from __future__ import annotations

from pipeline.rules.base import ResultFactory, most_relevant
from shared.enums import RuleId, RuleStatus
from shared.schemas.frame import EdgeProximity, FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.scene import ProposalType, ProtectionState, ProtectionVisibility, SceneProposal

ALERT_REASON = "visible_unguarded_edge"
CLEAR_REASON = "guarded_or_outside_edge_band"


class R5FallEdge:
    rule_id = RuleId.R5

    def __init__(self, *, basis: str, edge_max_wh: float) -> None:
        self.basis = basis
        self.edge_max_wh = edge_max_wh

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        if not context.geometry_supported:
            return make.frame(RuleStatus.UNSUPPORTED, "geometry_unstable_for_feed")
        if not context.capabilities.person_detection:
            return make.frame(RuleStatus.UNSUPPORTED, "person_detection_unavailable")
        if not context.capabilities.scene_state:
            return make.frame(RuleStatus.INCONCLUSIVE, "edge_scene_state_unavailable")
        edges = {
            proposal.proposal_id: proposal
            for proposal in context.scene_proposals
            if proposal.proposal_type is ProposalType.OPEN_EDGE
        }
        if not edges:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_visible_open_edge_candidate")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_person_tracks")

        results: list[RuleResult] = []
        for worker in workers:
            track_id = worker.canonical_track_id
            pairs = {
                pair.proposal_id: pair
                for pair in context.edge_proximities
                if pair.worker_track_id == track_id and pair.proposal_id in edges
            }
            per_edge = [
                self._edge(make, context, track_id, edges[edge_id], pairs.get(edge_id))
                for edge_id in sorted(edges)
            ]
            results.append(most_relevant(per_edge))
        return results

    def _edge(
        self,
        make: ResultFactory,
        context: FrameContext,
        track_id: str,
        edge: SceneProposal,
        pair: EdgeProximity | None,
    ) -> RuleResult:
        status, reason = self._classify(context, edge, pair)
        distance = pair.distance if pair is not None else None
        return make.make(
            status,
            reason,
            track_id=track_id,
            proposal_ids=(edge.proposal_id,),
            distance=distance,
        )

    def _classify(
        self, context: FrameContext, edge: SceneProposal, pair: EdgeProximity | None
    ) -> tuple[RuleStatus, str]:
        # Semantics that rule the edge out need no distance at all.
        if edge.protection_state is ProtectionState.GUARDED:
            return RuleStatus.EVALUATED_CLEAR, CLEAR_REASON
        if edge.visible_drop_or_open_side is False:
            return RuleStatus.EVALUATED_CLEAR, "edge_not_a_drop"

        if not context.capabilities.relative_plane:
            return RuleStatus.INCONCLUSIVE, "relative_plane_unavailable"
        if pair is None or pair.distance is None:
            return RuleStatus.INCONCLUSIVE, "edge_distance_unavailable"
        if not pair.compatible_plane:
            return RuleStatus.INCONCLUSIVE, "target_plane_incompatible"
        if pair.distance.lower_wh > self.edge_max_wh:
            return RuleStatus.EVALUATED_CLEAR, CLEAR_REASON

        # The worker may be within the edge band: now the semantics must be positive.
        if edge.visible_drop_or_open_side is not True:
            return RuleStatus.INCONCLUSIVE, "edge_semantics_unknown"
        if edge.protection_visibility is not ProtectionVisibility.SUFFICIENT:
            return RuleStatus.INCONCLUSIVE, "protection_visibility_insufficient"
        if edge.protection_state is not ProtectionState.UNGUARDED_OR_INADEQUATE:
            return RuleStatus.INCONCLUSIVE, "protection_state_unknown"
        if pair.distance.upper_wh > self.edge_max_wh:
            return RuleStatus.INCONCLUSIVE, "inconclusive_distance"
        return RuleStatus.EVALUATED_ALERT, ALERT_REASON
