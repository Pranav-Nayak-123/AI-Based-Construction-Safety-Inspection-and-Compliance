from __future__ import annotations

from pathlib import Path

import cv2
import pytest
from pydantic import ValidationError
from rule_frames import edge, zone
from video_fixtures import shifted, textured_scene

from pipeline.mapping.anchors import image_aligned_record
from pipeline.mapping.zones import ZoneIndex, anchor_sigma_px
from pipeline.scene.site_config import (
    SiteConfig,
    SiteZone,
    load_reference_image,
    load_site_config,
    scene_from_site_config,
    view_matches,
)
from shared.coordinates import Box2, Point2
from shared.enums import PERSON_CLASS, CoordinateSpace
from shared.identity import canonical_track_id
from shared.schemas.scene import ProposalSource, ProposalType, ProtectionState
from shared.schemas.tracks import Pass1TrackObservation


def _person_at(x: float, y: float, height: float = 100.0) -> Pass1TrackObservation:
    return Pass1TrackObservation(
        shot_id="s0",
        frame_index=0,
        video_time_s=0.0,
        track_id=1,
        canonical_track_id=canonical_track_id("s0", 1),
        object_class=PERSON_CLASS,
        box_reference=Box2(
            x1=x - 20, y1=y - height, x2=x + 20, y2=y, space=CoordinateSpace.REFERENCE
        ),
        anchor_reference=Point2(x=x, y=y, space=CoordinateSpace.REFERENCE),
    )


def _membership(x: float, y: float) -> tuple[bool | None, bool]:
    square = zone("zone-1")  # (0,0) (100,0) (100,100)
    record = image_aligned_record(_person_at(x, y))
    [membership] = ZoneIndex([square]).memberships([record])
    return membership.inside, membership.boundary_uncertain


def test_anchor_inside_outside_and_on_boundary() -> None:
    assert _membership(80.0, 20.0) == (True, False)  # well inside the triangle
    assert _membership(20.0, 80.0) == (False, False)  # well outside
    assert _membership(98.0, 50.0)[1] is True  # within 2 sigma of an edge


def test_anchor_uncertainty_grows_with_person_height() -> None:
    small = image_aligned_record(_person_at(0, 0, height=40))
    large = image_aligned_record(_person_at(0, 0, height=300))
    assert anchor_sigma_px(small.anchor_covariance_2x2) == pytest.approx(2.0)
    assert anchor_sigma_px(large.anchor_covariance_2x2) == pytest.approx(9.0)


def test_edges_are_not_treated_as_areas() -> None:
    assert not ZoneIndex([edge("edge-1")])


def _site(tmp_path: Path, zones: tuple[SiteZone, ...]) -> Path:
    cv2.imwrite(str(tmp_path / "cam-1.png"), textured_scene(640, 360))
    config = SiteConfig(
        camera_id="cam-1", canvas_size=(640, 360), reference_image="cam-1.png", zones=zones
    )
    path = tmp_path / "cam-1.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


FENCE = SiteZone(
    zone_id="crane-swing",
    proposal_type=ProposalType.RESTRICTED_ZONE,
    label="Crane swing area",
    points=((10.0, 10.0), (200.0, 10.0), (200.0, 150.0)),
)
DECK_EDGE = SiteZone(
    zone_id="deck-edge",
    proposal_type=ProposalType.OPEN_EDGE,
    points=((10.0, 300.0), (600.0, 300.0)),
    visible_drop_or_open_side=True,
    protection_state=ProtectionState.UNGUARDED_OR_INADEQUATE,
)


def test_site_config_becomes_manual_scene_proposals(tmp_path: Path) -> None:
    config = load_site_config(_site(tmp_path, (FENCE, DECK_EDGE)))
    scene = scene_from_site_config(config, shot_ids=["s0"], video_sha256="v")
    zone_proposal, edge_proposal = scene.proposals_for("s0")
    assert zone_proposal.proposal_id == "crane-swing"
    assert zone_proposal.source is ProposalSource.MANUAL_SITE_CONFIG
    assert zone_proposal.heuristic is False
    assert edge_proposal.geometry_type.value == "polyline"
    two_shots = scene_from_site_config(config, shot_ids=["s0", "s1"], video_sha256="v")
    assert two_shots.proposals_for("s1")[0].proposal_id == "s1/crane-swing"


def test_site_config_rejects_bad_zones() -> None:
    with pytest.raises(ValidationError, match="unique"):
        SiteConfig(camera_id="c", reference_image="c.png", zones=(FENCE, FENCE))
    outside = FENCE.model_copy(update={"points": ((0.0, 0.0), (5000.0, 0.0), (0.0, 10.0))})
    with pytest.raises(ValidationError, match="outside the canvas"):
        SiteConfig(camera_id="c", reference_image="c.png", zones=(outside,))
    with pytest.raises(ValidationError):
        SiteZone(zone_id="Bad Id", proposal_type=ProposalType.BARRIER, points=())


def test_zones_apply_only_when_the_view_still_matches(tmp_path: Path) -> None:
    path = _site(tmp_path, (FENCE,))
    reference = load_reference_image(load_site_config(path), path)
    assert view_matches(reference, reference)[0]
    assert view_matches(reference, shifted(reference, 1.0, 0.0))[0]
    moved, shift = view_matches(reference, shifted(reference, 30.0, 12.0))
    assert not moved and shift is not None and shift > 20
    assert view_matches(reference, textured_scene(640, 360, seed=99))[0] is False
