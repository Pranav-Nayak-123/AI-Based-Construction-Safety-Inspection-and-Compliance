"""Per-frame inputs to and outputs from the rule engine (v7 §31.8).

Pass two builds one `FrameContext` per processed frame by joining mapped tracks with the
scene proposals and relative-plane geometry. Rules only ever see this joined view; they
never touch pixels, models, or the plane fit directly.
"""

from __future__ import annotations

from pydantic import Field

from shared.coordinates import DistanceInterval, StrictModel
from shared.enums import PERSON_CLASS, AlertState, DistanceBand, OperatingState, RuleId
from shared.schemas.geometry import MachineryFootprintRecord
from shared.schemas.incidents import RuleResult
from shared.schemas.scene import SceneProposal
from shared.schemas.tracks import MappedTrackRecord


class FeedCapabilities(StrictModel):
    """Which upstream evidence exists for the current shot.

    A missing capability makes the dependent rules `unsupported_for_feed` or
    `inconclusive` instead of silently clear.
    """

    person_detection: bool = True
    ppe_classification: bool = True
    scene_state: bool = True
    relative_plane: bool = True


class ZoneMembership(StrictModel):
    worker_track_id: str
    proposal_id: str
    coordinate_mode: str
    inside: bool | None
    boundary_uncertain: bool
    reason_code: str


class MachineryProximity(StrictModel):
    worker_track_id: str
    machinery_id: str
    compatible_plane: bool
    operating_state: OperatingState
    distance: DistanceInterval | None
    band: DistanceBand
    reason_code: str


class EdgeProximity(StrictModel):
    worker_track_id: str
    proposal_id: str
    compatible_plane: bool
    distance: DistanceInterval | None
    reason_code: str


class FrameContext(StrictModel):
    run_id: str
    shot_id: str
    frame_index: int = Field(ge=0)
    video_time_s: float = Field(ge=0.0)
    geometry_supported: bool
    capabilities: FeedCapabilities = FeedCapabilities()
    tracks: tuple[MappedTrackRecord, ...] = ()
    machinery: tuple[MachineryFootprintRecord, ...] = ()
    scene_proposals: tuple[SceneProposal, ...] = ()
    zone_memberships: tuple[ZoneMembership, ...] = ()
    machinery_proximities: tuple[MachineryProximity, ...] = ()
    edge_proximities: tuple[EdgeProximity, ...] = ()

    def workers(self) -> tuple[MappedTrackRecord, ...]:
        return tuple(track for track in self.tracks if track.object_class == PERSON_CLASS)

    def proposal(self, proposal_id: str) -> SceneProposal | None:
        for candidate in self.scene_proposals:
            if candidate.proposal_id == proposal_id:
                return candidate
        return None


class ActiveAlert(StrictModel):
    """A (rule, subject) pair that is pending or alerting on this frame, for overlays."""

    rule_id: RuleId
    track_id: str
    state: AlertState
    incident_id: str | None


class FrameRuleOutput(StrictModel):
    frame_index: int = Field(ge=0)
    video_time_s: float = Field(ge=0.0)
    results: tuple[RuleResult, ...]
    active_alerts: tuple[ActiveAlert, ...] = ()
    opened_incident_ids: tuple[str, ...] = ()
    resolved_incident_ids: tuple[str, ...] = ()
