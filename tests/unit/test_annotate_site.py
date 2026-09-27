from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.scene.site_config import load_site_config
from shared.schemas.scene import ProposalType, ProtectionState
from tools.annotate_site import Shape, build_site_config, save_site, slug


def test_slug_makes_valid_zone_ids() -> None:
    assert slug("Crane swing area!", "zone-1") == "crane-swing-area"
    assert slug("", "zone-1") == "zone-1"


def test_drawn_shapes_round_trip_through_a_site_config(tmp_path: Path) -> None:
    shapes = [
        Shape(ProposalType.RESTRICTED_ZONE, [(10, 10), (200, 10), (200, 150)], label="Pit"),
        Shape(ProposalType.RESTRICTED_ZONE, [(300, 10), (400, 10), (400, 90)], label="Pit"),
        Shape(
            ProposalType.OPEN_EDGE,
            [(10, 300), (600, 300)],
            visible_drop=True,
            protection_state=ProtectionState.UNGUARDED_OR_INADEQUATE,
        ),
    ]
    config = build_site_config("yard-east", shapes, "yard-east.png", (1920, 1080))
    assert [zone.zone_id for zone in config.zones] == ["pit", "pit-2", "open-edge-3"]

    path = save_site(config, np.zeros((1080, 1920, 3), np.uint8), tmp_path)
    loaded = load_site_config(path)
    assert loaded == config
    assert (tmp_path / "yard-east.png").is_file()
    assert loaded.zones[2].visible_drop_or_open_side is True
