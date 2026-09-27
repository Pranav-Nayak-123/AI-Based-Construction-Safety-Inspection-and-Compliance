from __future__ import annotations

from pydantic import Field

from shared.coordinates import Box2, Point2, StrictModel
from shared.enums import HelmetState, OperatingState, VestState


class HelmetRecord(StrictModel):
    state: HelmetState
    confidence: float = Field(ge=0.0, le=1.0)
    visible: bool


class VestRecord(StrictModel):
    state: VestState
    confidence: float = Field(ge=0.0, le=1.0)
    visible: bool


class Pass1TrackObservation(StrictModel):
    schema_version: int = 1
    shot_id: str
    frame_index: int = Field(ge=0)
    video_time_s: float = Field(ge=0.0)
    track_id: int
    canonical_track_id: str
    object_class: str
    box_reference: Box2
    anchor_reference: Point2
    helmet: HelmetRecord | None = None
    vest: VestRecord | None = None
    # Detector confidence of the matched box, and the box on the oriented canvas for
    # drawing overlays on source frames (equal to box_reference for a fixed camera).
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    box_oriented: Box2 | None = None
    # Machinery only: mean frame-to-frame intensity change inside the box (0-1), the
    # articulation cue for operating state (v7 §38).
    motion_energy: float | None = Field(default=None, ge=0.0, le=1.0)


class MappedTrackRecord(StrictModel):
    schema_version: int = 1
    shot_id: str
    frame_index: int = Field(ge=0)
    video_time_s: float = Field(ge=0.0)
    canonical_track_id: str
    object_class: str
    reference_observation_key: str
    anchor_reference: Point2
    anchor_covariance_2x2: list[list[float]]
    relative_position: Point2 | None = None
    plane_id: str | None = None
    helmet: HelmetRecord | None = None
    vest: VestRecord | None = None
    operating_state: OperatingState | None = None
    mapping_mode: str
