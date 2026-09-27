"""R2 — apparent missing hi-vis in a movement area (v7 §11 R2, §40.3).

A movement area is a traffic zone, a machinery operating region, or a machine whose full
distance interval lies in the near or caution band. The joint predicate — no vest *and*
inside a movement area — is what gets debounced.
"""

from __future__ import annotations

from enum import StrEnum

from pipeline.rules.base import ResultFactory, vest_assessable
from shared.enums import DistanceBand, RuleId, RuleStatus, VestState
from shared.schemas.frame import FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.scene import ProposalType
from shared.schemas.tracks import MappedTrackRecord

ALERT_REASON = "persistent_no_vest_in_movement_area"

MOVEMENT_AREA_TYPES = frozenset(
    {ProposalType.TRAFFIC_ZONE, ProposalType.MACHINERY_OPERATING_REGION}
)
MOVEMENT_BANDS = frozenset({DistanceBand.NEAR, DistanceBand.CAUTION})


class _Context(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    UNCERTAIN = "uncertain"


class R2MissingHiVis:
    rule_id = RuleId.R2

    def __init__(self, *, basis: str, confidence_floor: float) -> None:
        self.basis = basis
        self.confidence_floor = confidence_floor

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        make = ResultFactory(self.rule_id, self.basis, context)
        capabilities = context.capabilities
        if not capabilities.person_detection:
            return make.frame(RuleStatus.UNSUPPORTED, "person_detection_unavailable")
        if not capabilities.ppe_classification:
            return make.frame(RuleStatus.UNSUPPORTED, "vest_detection_unavailable")
        workers = context.workers()
        if not workers:
            return make.frame(RuleStatus.NOT_APPLICABLE, "no_person_tracks")
        return [self._worker(make, context, worker) for worker in workers]

    def _worker(
        self, make: ResultFactory, context: FrameContext, worker: MappedTrackRecord
    ) -> RuleResult:
        track_id = worker.canonical_track_id
        area, proposal_ids, machinery_id = _movement_context(context, track_id)
        if area is _Context.OUTSIDE:
            return make.make(RuleStatus.NOT_APPLICABLE, "movement_area_absent", track_id=track_id)

        vest = worker.vest
        if vest is None or not vest_assessable(vest):
            return make.make(
                RuleStatus.INCONCLUSIVE, "vest_visibility_insufficient", track_id=track_id
            )
        if vest.state is not VestState.NO_VEST:
            return make.make(
                RuleStatus.EVALUATED_CLEAR,
                "hi_vis_condition_not_observed",
                track_id=track_id,
                confidence=vest.confidence,
            )
        if vest.confidence < self.confidence_floor:
            return make.make(
                RuleStatus.INCONCLUSIVE,
                "vest_confidence_below_floor",
                track_id=track_id,
                confidence=vest.confidence,
            )
        if area is _Context.UNCERTAIN:
            return make.make(
                RuleStatus.INCONCLUSIVE,
                "movement_area_uncertain",
                track_id=track_id,
                confidence=vest.confidence,
            )
        return make.make(
            RuleStatus.EVALUATED_ALERT,
            ALERT_REASON,
            track_id=track_id,
            object_id=machinery_id,
            proposal_ids=proposal_ids,
            confidence=vest.confidence,
        )


def _movement_context(
    context: FrameContext, track_id: str
) -> tuple[_Context, tuple[str, ...], str | None]:
    """Is this worker inside a movement area? Positive evidence beats uncertainty."""
    uncertain = False

    inside_zones: list[str] = []
    movement_zone_ids = {
        proposal.proposal_id
        for proposal in context.scene_proposals
        if proposal.proposal_type in MOVEMENT_AREA_TYPES
    }
    if not context.capabilities.scene_state:
        uncertain = True  # we cannot know whether movement zones exist at all
    elif movement_zone_ids:
        assessed: set[str] = set()
        if context.geometry_supported:
            for membership in context.zone_memberships:
                if (
                    membership.worker_track_id != track_id
                    or membership.proposal_id not in movement_zone_ids
                ):
                    continue
                assessed.add(membership.proposal_id)
                if membership.inside is True and not membership.boundary_uncertain:
                    inside_zones.append(membership.proposal_id)
                elif membership.inside is None or membership.boundary_uncertain:
                    uncertain = True
        if assessed != movement_zone_ids:
            uncertain = True

    near_machine: str | None = None
    closest = float("inf")
    if context.machinery:
        assessed_machines: set[str] = set()
        for proximity in context.machinery_proximities:
            if proximity.worker_track_id != track_id:
                continue
            assessed_machines.add(proximity.machinery_id)
            if proximity.compatible_plane and proximity.band in MOVEMENT_BANDS:
                median = proximity.distance.median_wh if proximity.distance else 0.0
                if median < closest:
                    closest, near_machine = median, proximity.machinery_id
            elif proximity.band is DistanceBand.INDETERMINATE or not proximity.compatible_plane:
                uncertain = True
        if assessed_machines != {machine.machinery_id for machine in context.machinery}:
            uncertain = True

    if inside_zones or near_machine is not None:
        return _Context.INSIDE, tuple(sorted(inside_zones)), near_machine
    if uncertain:
        return _Context.UNCERTAIN, (), None
    return _Context.OUTSIDE, (), None
