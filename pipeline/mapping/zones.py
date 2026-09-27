"""Worker-in-zone tests in reference image space (feeds R2 and R3)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from shapely.geometry import Point, Polygon
from shapely.prepared import PreparedGeometry, prep

from shared.enums import PERSON_CLASS
from shared.schemas.frame import ZoneMembership
from shared.schemas.scene import GeometryType, SceneProposal
from shared.schemas.tracks import MappedTrackRecord

COORDINATE_MODE = "reference_image"
# An anchor within this many standard deviations of the boundary is "uncertain".
BOUNDARY_SIGMAS = 2.0


class ZoneIndex:
    """Prepared polygons for the area-type proposals of one shot."""

    def __init__(self, proposals: Sequence[SceneProposal]) -> None:
        self._zones: list[tuple[str, Polygon, PreparedGeometry]] = []
        for proposal in proposals:
            if proposal.geometry_type is GeometryType.POLYLINE:
                continue
            points = proposal.reference_points
            if proposal.geometry_type is GeometryType.BOX:
                (x1, y1), (x2, y2) = points
                points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # repair self-touching hand-drawn outlines
            self._zones.append((proposal.proposal_id, polygon, prep(polygon)))

    def __bool__(self) -> bool:
        return bool(self._zones)

    def memberships(self, tracks: Sequence[MappedTrackRecord]) -> tuple[ZoneMembership, ...]:
        results: list[ZoneMembership] = []
        for track in tracks:
            if track.object_class != PERSON_CLASS:
                continue
            anchor = Point(track.anchor_reference.x, track.anchor_reference.y)
            sigma = anchor_sigma_px(track.anchor_covariance_2x2)
            for proposal_id, polygon, prepared in self._zones:
                inside = prepared.contains(anchor)
                boundary_distance = polygon.exterior.distance(anchor)
                uncertain = boundary_distance <= BOUNDARY_SIGMAS * sigma
                results.append(
                    ZoneMembership(
                        worker_track_id=track.canonical_track_id,
                        proposal_id=proposal_id,
                        coordinate_mode=COORDINATE_MODE,
                        inside=inside,
                        boundary_uncertain=uncertain,
                        reason_code="near_boundary" if uncertain else "ok",
                    )
                )
        return tuple(results)


def anchor_sigma_px(covariance: list[list[float]]) -> float:
    """Isotropic standard deviation from a 2x2 anchor covariance (largest axis)."""
    (a, b), (c, d) = covariance
    trace, determinant = a + d, a * d - b * c
    largest = trace / 2 + math.sqrt(max(trace * trace / 4 - determinant, 0.0))
    return math.sqrt(max(largest, 0.0))
