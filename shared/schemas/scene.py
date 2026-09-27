from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, model_validator

from shared.coordinates import StrictModel

ReferencePoint = tuple[float, float]


class ProposalType(StrEnum):
    RESTRICTED_ZONE = "restricted_zone"
    OPEN_EDGE = "open_edge"
    VISIBLE_GUARDRAIL = "visible_guardrail"
    BARRIER = "barrier"
    WARNING_SIGN = "warning_sign"
    TRAFFIC_ZONE = "traffic_zone"
    MACHINERY_OPERATING_REGION = "machinery_operating_region"


class GeometryType(StrEnum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    BOX = "box"


class ProposalSource(StrEnum):
    MANUAL_SITE_CONFIG = "manual_site_config"
    GEMINI_CONSENSUS = "gemini_consensus"
    LOCAL_HEURISTIC = "local_heuristic"
    BUNDLED_CACHE = "bundled_cache"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class ProtectionVisibility(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class ProtectionState(StrEnum):
    GUARDED = "guarded"
    UNGUARDED_OR_INADEQUATE = "unguarded_or_inadequate"
    UNKNOWN = "unknown"


_MINIMUM_POINTS = {
    GeometryType.POLYGON: 3,
    GeometryType.POLYLINE: 2,
    GeometryType.BOX: 2,
}


class SceneProposal(StrictModel):
    """One static scene element in reference-frame pixel coordinates.

    Semantic fields are typed rather than a free-form dict so that R2/R5 can never
    read a misspelled attribute as "absent". `None` means the source did not assess it.
    """

    proposal_id: str
    shot_id: str
    proposal_type: ProposalType
    geometry_type: GeometryType
    label: str = ""
    reference_points: tuple[ReferencePoint, ...]
    relative_points: tuple[ReferencePoint, ...] | None = None
    plane_id: str | None = None
    visible_drop_or_open_side: bool | None = None
    protection_visibility: ProtectionVisibility | None = None
    protection_state: ProtectionState | None = None
    road_work: bool | None = None
    support_count: int = Field(default=0, ge=0)
    eligible_count: int = Field(default=0, ge=0)
    source: ProposalSource
    heuristic: bool = True

    @model_validator(mode="after")
    def _valid_geometry(self) -> SceneProposal:
        minimum = _MINIMUM_POINTS[self.geometry_type]
        if len(self.reference_points) < minimum:
            raise ValueError(
                f"{self.geometry_type.value} needs at least {minimum} points, "
                f"got {len(self.reference_points)}"
            )
        if self.geometry_type is GeometryType.BOX and len(self.reference_points) != 2:
            raise ValueError("box geometry is exactly two corner points")
        for x, y in self.reference_points:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("scene coordinates must be finite")
        if self.support_count > self.eligible_count:
            raise ValueError("support_count cannot exceed eligible_count")
        if (
            self.proposal_type is ProposalType.OPEN_EDGE
            and self.geometry_type is not GeometryType.POLYLINE
        ):
            raise ValueError("open_edge proposals must be polylines")
        return self


class SceneCache(StrictModel):
    schema_version: int = 1
    cache_key: str
    video_sha256: str
    # None when no scene state exists for the run (no site config, cache or bootstrap).
    source: ProposalSource | None = None
    shot_proposals: dict[str, tuple[SceneProposal, ...]]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> SceneCache:
        seen: set[str] = set()
        for shot_id, proposals in self.shot_proposals.items():
            for proposal in proposals:
                if proposal.shot_id != shot_id:
                    raise ValueError(
                        f"proposal {proposal.proposal_id} is filed under shot {shot_id} "
                        f"but declares shot {proposal.shot_id}"
                    )
                if proposal.proposal_id in seen:
                    raise ValueError(f"duplicate proposal id {proposal.proposal_id}")
                seen.add(proposal.proposal_id)
        return self

    def proposals_for(self, shot_id: str) -> tuple[SceneProposal, ...]:
        return self.shot_proposals.get(shot_id, ())
