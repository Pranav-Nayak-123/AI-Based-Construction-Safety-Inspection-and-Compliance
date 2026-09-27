from __future__ import annotations

from pydantic import Field

from shared.coordinates import Point2, StrictModel
from shared.enums import CoordinateSpace, OperatingState


class VerticalObservation(StrictModel):
    """Paired head/foot points of one upright person in one frame (v7 §37.1).

    Head and foot are never aggregated independently: the pair is the unit of evidence
    for the relative-plane fit.
    """

    observation_id: str
    shot_id: str
    frame_index: int = Field(ge=0)
    track_id: int
    head_reference: Point2
    foot_reference: Point2
    head_confidence: float = Field(ge=0.0, le=1.0)
    foot_confidence: float = Field(ge=0.0, le=1.0)
    standing_probability: float = Field(ge=0.0, le=1.0)
    occlusion_fraction: float = Field(ge=0.0, le=1.0)
    covariance_4x4: list[list[float]]


class MachineryContactCandidate(StrictModel):
    """Pass-one ground-contact evidence for one machine in reference space."""

    shot_id: str
    frame_index: int = Field(ge=0)
    machinery_id: str
    class_name: str
    contact_points_reference: list[Point2]
    construction_method: str
    covariance: list[list[float]]
    valid: bool


class MachineryMotionObservation(StrictModel):
    """Pass-one motion evidence; the operating-state verdict is made in pass two."""

    shot_id: str
    frame_index: int = Field(ge=0)
    machinery_id: str
    base_translation_px: tuple[float, float] | None
    articulation_score: float | None
    valid_flow_points: int = Field(ge=0)
    reason_code: str


class MachineryFootprintRecord(StrictModel):
    """Pass-two machine footprint on the relative plane, with its operating state."""

    machinery_id: str
    shot_id: str
    frame_index: int = Field(ge=0)
    class_name: str
    polygon_plane_xy: list[tuple[float, float]] | None
    compatible_plane_id: str | None
    operating_state: OperatingState
    construction_method: str
    inflation_wh: float = Field(ge=0.0)
    covariance: list[list[float]]
    valid: bool
    reason_code: str


class TransformRecord(StrictModel):
    """One coordinate-space transform in the ledger (v7 §31.3).

    Homogeneous column vectors, ``p' ~ H p``; matrices serialised row-major.
    """

    transform_id: str
    frame_index: int | None
    source_space: CoordinateSpace
    destination_space: CoordinateSpace
    kind: str
    forward_3x3: list[list[float]] | None
    inverse_3x3: list[list[float]] | None
    valid: bool
    interpolated: bool = False
    residual_rms_px: float | None = None
    inlier_count: int | None = None
    inlier_ratio: float | None = None
    reason_code: str = "ok"
    implementation_version: str
