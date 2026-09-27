from __future__ import annotations

from pydantic import Field

from shared.coordinates import DistanceInterval, StrictModel
from shared.enums import DistanceBand, RuleId, RuleStatus
from shared.schemas.tracks import HelmetRecord, VestRecord


class RuleResult(StrictModel):
    rule_id: RuleId
    status: RuleStatus
    reason_code: str
    basis: str
    track_id: str | None = None
    object_id: str | None = None
    proposal_ids: tuple[str, ...] = ()
    observed_from_s: float | None = None
    observed_to_s: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    distance: DistanceInterval | None = None
    relative_band: DistanceBand | None = None
    references: tuple[str, ...] = ()


class IncidentRecord(StrictModel):
    schema_version: int = 1
    incident_id: str
    run_id: str
    shot_id: str
    catalogue_sha256: str
    rule_id: RuleId
    status: RuleStatus
    reason_code: str
    basis: str
    severity: str
    canonical_track_id: str | None = None
    object_id: str | None = None
    proposal_ids: tuple[str, ...] = ()
    first_seen_s: float = Field(ge=0.0)
    confirmed_at_s: float = Field(ge=0.0)
    resolved_at_s: float | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    distance: DistanceInterval | None = None
    relative_band: DistanceBand | None = None
    title: str = ""
    observation_text: str
    action_code: str
    action_text: str
    references: tuple[str, ...] = ()
    evidence_path: str
    source_frame_index: int = Field(ge=0)
    # Original-resolution person crop at the evidence frame (L1 vision adjudication input).
    evidence_crop_path: str | None = None
    # Smoothed PPE state at the evidence frame, shown beside the rule verdict.
    helmet: HelmetRecord | None = None
    vest: VestRecord | None = None
    # True when a PPE head was unknown or near the confidence floor at the evidence
    # frame, so the cloud service should queue the crop for L1 adjudication.
    adjudication_candidate: bool = False
    # Why the episode ended: condition_cleared | evidence_lost | shot_boundary | end_of_clip.
    resolution_reason: str | None = None
