"""Builders for synthetic rule-engine inputs used across the rule tests."""

from __future__ import annotations

from typing import Any

from shared.config import RulesConfig, load_config
from shared.coordinates import DistanceInterval, Point2
from shared.enums import (
    PERSON_CLASS,
    CoordinateSpace,
    DistanceBand,
    HelmetState,
    OperatingState,
    VestState,
)
from shared.identity import canonical_track_id
from shared.schemas.frame import (
    EdgeProximity,
    FeedCapabilities,
    FrameContext,
    MachineryProximity,
    ZoneMembership,
)
from shared.schemas.geometry import MachineryFootprintRecord
from shared.schemas.scene import (
    GeometryType,
    ProposalSource,
    ProposalType,
    ProtectionState,
    ProtectionVisibility,
    SceneProposal,
)
from shared.schemas.tracks import HelmetRecord, MappedTrackRecord, VestRecord

SHOT = "s0"
FPS = 15.0


def tid(local_id: int, shot: str = SHOT) -> str:
    return canonical_track_id(shot, local_id)


def rules_config(**overrides: Any) -> RulesConfig:
    return load_config().rules.model_copy(update=overrides)


def helmet(state: HelmetState, confidence: float = 0.9, visible: bool = True) -> HelmetRecord:
    return HelmetRecord(state=state, confidence=confidence, visible=visible)


def vest(state: VestState, confidence: float = 0.9, visible: bool = True) -> VestRecord:
    return VestRecord(state=state, confidence=confidence, visible=visible)


def worker(
    local_id: int,
    *,
    helmet_record: HelmetRecord | None = None,
    vest_record: VestRecord | None = None,
    shot: str = SHOT,
) -> MappedTrackRecord:
    return MappedTrackRecord(
        shot_id=shot,
        frame_index=0,
        video_time_s=0.0,
        canonical_track_id=tid(local_id, shot),
        object_class=PERSON_CLASS,
        reference_observation_key=f"{shot}:{local_id}",
        anchor_reference=Point2(x=100.0, y=200.0, space=CoordinateSpace.REFERENCE),
        anchor_covariance_2x2=[[4.0, 0.0], [0.0, 4.0]],
        helmet=helmet_record,
        vest=vest_record,
        mapping_mode="relative_plane",
    )


def machine(
    machine_id: str,
    state: OperatingState = OperatingState.ACTIVE,
    class_name: str = "excavator",
) -> MachineryFootprintRecord:
    return MachineryFootprintRecord(
        machinery_id=machine_id,
        shot_id=SHOT,
        frame_index=0,
        class_name=class_name,
        polygon_plane_xy=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        compatible_plane_id="plane-0",
        operating_state=state,
        construction_method="bottom_contact_envelope",
        inflation_wh=0.35,
        covariance=[[0.01, 0.0], [0.0, 0.01]],
        valid=True,
        reason_code="ok",
    )


def zone(
    proposal_id: str,
    proposal_type: ProposalType = ProposalType.RESTRICTED_ZONE,
    *,
    label: str = "",
    road_work: bool | None = None,
) -> SceneProposal:
    return SceneProposal(
        proposal_id=proposal_id,
        shot_id=SHOT,
        proposal_type=proposal_type,
        geometry_type=GeometryType.POLYGON,
        label=label,
        reference_points=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0)),
        road_work=road_work,
        source=ProposalSource.SYNTHETIC_FIXTURE,
    )


def edge(
    proposal_id: str,
    *,
    drop: bool | None = True,
    visibility: ProtectionVisibility | None = ProtectionVisibility.SUFFICIENT,
    protection: ProtectionState | None = ProtectionState.UNGUARDED_OR_INADEQUATE,
    label: str = "",
) -> SceneProposal:
    return SceneProposal(
        proposal_id=proposal_id,
        shot_id=SHOT,
        proposal_type=ProposalType.OPEN_EDGE,
        geometry_type=GeometryType.POLYLINE,
        label=label,
        reference_points=((0.0, 50.0), (200.0, 50.0)),
        visible_drop_or_open_side=drop,
        protection_visibility=visibility,
        protection_state=protection,
        source=ProposalSource.SYNTHETIC_FIXTURE,
    )


def membership(
    local_id: int,
    proposal_id: str,
    inside: bool | None,
    *,
    uncertain: bool = False,
) -> ZoneMembership:
    return ZoneMembership(
        worker_track_id=tid(local_id),
        proposal_id=proposal_id,
        coordinate_mode="reference",
        inside=inside,
        boundary_uncertain=uncertain,
        reason_code="ok",
    )


def interval(lower: float, upper: float) -> DistanceInterval:
    return DistanceInterval(
        lower_wh=lower,
        median_wh=(lower + upper) / 2,
        upper_wh=upper,
        successful_bootstraps=500,
        total_bootstraps=500,
    )


def proximity(
    local_id: int,
    machine_id: str,
    band: DistanceBand,
    state: OperatingState = OperatingState.ACTIVE,
    *,
    distance: DistanceInterval | None = None,
    compatible: bool = True,
) -> MachineryProximity:
    return MachineryProximity(
        worker_track_id=tid(local_id),
        machinery_id=machine_id,
        compatible_plane=compatible,
        operating_state=state,
        distance=distance,
        band=band,
        reason_code="ok",
    )


def edge_proximity(
    local_id: int,
    proposal_id: str,
    distance: DistanceInterval | None,
    *,
    compatible: bool = True,
) -> EdgeProximity:
    return EdgeProximity(
        worker_track_id=tid(local_id),
        proposal_id=proposal_id,
        compatible_plane=compatible,
        distance=distance,
        reason_code="ok",
    )


def frame(
    time_s: float = 0.0,
    *,
    frame_index: int | None = None,
    shot: str = SHOT,
    geometry: bool = True,
    capabilities: FeedCapabilities | None = None,
    tracks: tuple[MappedTrackRecord, ...] = (),
    machinery: tuple[MachineryFootprintRecord, ...] = (),
    proposals: tuple[SceneProposal, ...] = (),
    memberships: tuple[ZoneMembership, ...] = (),
    proximities: tuple[MachineryProximity, ...] = (),
    edges: tuple[EdgeProximity, ...] = (),
    run_id: str = "run-test",
) -> FrameContext:
    return FrameContext(
        run_id=run_id,
        shot_id=shot,
        frame_index=round(time_s * FPS) if frame_index is None else frame_index,
        video_time_s=time_s,
        geometry_supported=geometry,
        capabilities=capabilities or FeedCapabilities(),
        tracks=tracks,
        machinery=machinery,
        scene_proposals=proposals,
        zone_memberships=memberships,
        machinery_proximities=proximities,
        edge_proximities=edges,
    )


def times(start_s: float, end_s: float, fps: float = FPS) -> list[float]:
    """Frame timestamps in [start, end) at a fixed rate, computed without drift."""
    count = round((end_s - start_s) * fps)
    return [start_s + index / fps for index in range(count)]
