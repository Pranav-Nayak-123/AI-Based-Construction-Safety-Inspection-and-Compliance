from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pipeline.rules.catalogue import default_catalogue_path, load_catalogue
from shared.enums import RuleId
from shared.schemas.scene import (
    GeometryType,
    ProposalSource,
    ProposalType,
    SceneCache,
    SceneProposal,
)


def _write_catalogue(tmp_path: Path, mutate: object) -> Path:
    payload = yaml.safe_load(default_catalogue_path().read_text(encoding="utf-8"))
    mutate(payload)  # type: ignore[operator]
    path = tmp_path / "catalogue.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_catalogue_covers_every_rule_with_a_stable_hash() -> None:
    first = load_catalogue()
    second = load_catalogue()
    assert [rule.rule_id for rule in first.catalogue.rules] == list(RuleId)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_catalogue_hash_changes_with_content(tmp_path: Path) -> None:
    def retitle(payload: dict) -> None:
        payload["rules"][0]["title"] = "Something else"

    assert load_catalogue(_write_catalogue(tmp_path, retitle)).sha256 != load_catalogue().sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["rules"].pop(), "missing rules"),
        (lambda p: p["rules"].append(dict(p["rules"][0])), "duplicate"),
        (
            lambda p: p["rules"][0].update(observation_template="Worker {name}"),
            "unsupported variables",
        ),
        (lambda p: p["rules"][0].update(action_code="verify helmet"), "action_code"),
        (lambda p: p["rules"][0].update(severity="extreme"), "severity"),
        (lambda p: p["rules"][0].update(surprise=True), "surprise"),
    ],
)
def test_invalid_catalogue_is_rejected(tmp_path: Path, mutation: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        load_catalogue(_write_catalogue(tmp_path, mutation))


def _proposal(**overrides: object) -> SceneProposal:
    fields: dict[str, object] = {
        "proposal_id": "zone-1",
        "shot_id": "s0",
        "proposal_type": ProposalType.RESTRICTED_ZONE,
        "geometry_type": GeometryType.POLYGON,
        "reference_points": ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),
        "source": ProposalSource.MANUAL_SITE_CONFIG,
    }
    fields.update(overrides)
    return SceneProposal(**fields)  # type: ignore[arg-type]


def test_polygon_needs_three_points() -> None:
    with pytest.raises(ValidationError, match="at least 3 points"):
        _proposal(reference_points=((0.0, 0.0), (1.0, 1.0)))


def test_open_edge_must_be_a_polyline() -> None:
    with pytest.raises(ValidationError, match="polylines"):
        _proposal(proposal_type=ProposalType.OPEN_EDGE)
    _proposal(
        proposal_type=ProposalType.OPEN_EDGE,
        geometry_type=GeometryType.POLYLINE,
        reference_points=((0.0, 0.0), (5.0, 0.0)),
    )


def test_non_finite_scene_points_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        _proposal(reference_points=((0.0, 0.0), (float("nan"), 0.0), (1.0, 1.0)))


def test_support_cannot_exceed_eligible_frames() -> None:
    with pytest.raises(ValidationError, match="support_count"):
        _proposal(support_count=6, eligible_count=5)


def test_scene_cache_rejects_misfiled_and_duplicate_proposals() -> None:
    with pytest.raises(ValidationError, match="filed under shot"):
        SceneCache(
            cache_key="k",
            video_sha256="v",
            source=ProposalSource.MANUAL_SITE_CONFIG,
            shot_proposals={"s1": (_proposal(),)},
        )
    with pytest.raises(ValidationError, match="duplicate proposal"):
        SceneCache(
            cache_key="k",
            video_sha256="v",
            source=ProposalSource.MANUAL_SITE_CONFIG,
            shot_proposals={"s0": (_proposal(), _proposal())},
        )
    cache = SceneCache(
        cache_key="k",
        video_sha256="v",
        source=ProposalSource.MANUAL_SITE_CONFIG,
        shot_proposals={"s0": (_proposal(),)},
    )
    assert cache.proposals_for("s0")[0].proposal_id == "zone-1"
    assert cache.proposals_for("s9") == ()
