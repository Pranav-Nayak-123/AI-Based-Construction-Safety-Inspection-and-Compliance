"""Operator-drawn scene state for one fixed camera (manual alternative to v7 §9 bootstrap).

A site config lives in `config/sites/<camera_id>.json` next to the reference frame it was
drawn on (`<camera_id>.png`). Zones and edges are in that frame's pixel coordinates on the
1920x1080 oriented canvas. They only apply to a shot whose view still matches the
reference frame; a moved camera gets no zones rather than misplaced ones.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import Field, model_validator

from pipeline.intake.stabilization import ReferenceMatcher, corner_displacement_px
from shared.coordinates import StrictModel
from shared.schemas.scene import (
    GeometryType,
    ProposalSource,
    ProposalType,
    ProtectionState,
    ProtectionVisibility,
    SceneCache,
    SceneProposal,
)

SITE_CONFIG_VERSION = 1
# A view that moved more than this fraction of the diagonal no longer matches the zones.
MAX_VIEW_SHIFT_DIAGONAL = 0.004


class SiteZone(StrictModel):
    zone_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    proposal_type: ProposalType
    label: str = ""
    points: tuple[tuple[float, float], ...]
    road_work: bool | None = None
    visible_drop_or_open_side: bool | None = None
    protection_visibility: ProtectionVisibility | None = None
    protection_state: ProtectionState | None = None

    @property
    def geometry_type(self) -> GeometryType:
        if self.proposal_type is ProposalType.OPEN_EDGE:
            return GeometryType.POLYLINE
        return GeometryType.POLYGON


class SiteConfig(StrictModel):
    schema_version: int = SITE_CONFIG_VERSION
    camera_id: str
    canvas_size: tuple[int, int] = (1920, 1080)
    reference_image: str
    notes: str = ""
    zones: tuple[SiteZone, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> SiteConfig:
        ids = [zone.zone_id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone ids must be unique within a site config")
        width, height = self.canvas_size
        for zone in self.zones:
            for x, y in zone.points:
                if not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError(f"zone {zone.zone_id} has a point outside the canvas")
        return self


def load_site_config(path: Path) -> SiteConfig:
    return SiteConfig.model_validate_json(path.read_text(encoding="utf-8"))


def load_reference_image(config: SiteConfig, config_path: Path) -> np.ndarray:
    image_path = config_path.parent / config.reference_image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"site reference image missing: {image_path}")
    return image


def view_matches(reference: np.ndarray, frame: np.ndarray) -> tuple[bool, float | None]:
    """Does `frame` show the same fixed view the zones were drawn on?"""
    fit = ReferenceMatcher(reference).fit(frame)
    if fit.matrix is None:
        return False, None
    height, width = reference.shape[:2]
    shift = corner_displacement_px(fit.matrix, width, height)
    return shift <= MAX_VIEW_SHIFT_DIAGONAL * float(np.hypot(width, height)), shift


def scene_from_site_config(
    config: SiteConfig,
    *,
    shot_ids: list[str],
    video_sha256: str,
    warnings: tuple[str, ...] = (),
) -> SceneCache:
    """Scene proposals for the shots whose view matched the site reference."""
    multiple = len(shot_ids) > 1
    shot_proposals: dict[str, tuple[SceneProposal, ...]] = {}
    for shot_id in shot_ids:
        shot_proposals[shot_id] = tuple(
            SceneProposal(
                proposal_id=f"{shot_id}/{zone.zone_id}" if multiple else zone.zone_id,
                shot_id=shot_id,
                proposal_type=zone.proposal_type,
                geometry_type=zone.geometry_type,
                label=zone.label,
                reference_points=zone.points,
                visible_drop_or_open_side=zone.visible_drop_or_open_side,
                protection_visibility=zone.protection_visibility,
                protection_state=zone.protection_state,
                road_work=zone.road_work,
                source=ProposalSource.MANUAL_SITE_CONFIG,
                heuristic=False,
            )
            for zone in config.zones
        )
    return SceneCache(
        cache_key=f"site:{config.camera_id}",
        video_sha256=video_sha256,
        source=ProposalSource.MANUAL_SITE_CONFIG,
        shot_proposals=shot_proposals,
        warnings=warnings,
    )
