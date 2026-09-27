"""Pass-one store → pass two → incidents, with real rules and a manual site config."""

from __future__ import annotations

from pathlib import Path

from jobs.pass2 import run_pass2
from pipeline.cache.pass1_store import FrameRecord, Pass1Store
from pipeline.mapping.frame_context import FrameContextBuilder
from pipeline.output.summaries import TrackSummaryBuilder
from pipeline.rules.engine import RuleEngine
from pipeline.scene.site_config import SiteConfig, SiteZone, scene_from_site_config
from shared.config import load_config
from shared.coordinates import Box2, Point2
from shared.enums import CoordinateSpace, RuleId, RuleStatus
from shared.identity import canonical_track_id
from shared.schemas.scene import ProposalType
from shared.schemas.tracks import Pass1TrackObservation

FPS = 15.0
CONFIG = load_config()


def _observation(frame: int, local_id: int, cls: str, x: float, y: float) -> Pass1TrackObservation:
    height = 120.0 if cls == "person" else 200.0
    box = Box2(x1=x - 25, y1=y - height, x2=x + 25, y2=y, space=CoordinateSpace.REFERENCE)
    return Pass1TrackObservation(
        shot_id="s0",
        frame_index=frame,
        video_time_s=frame / FPS,
        track_id=local_id,
        canonical_track_id=canonical_track_id("s0", local_id),
        object_class=cls,
        box_reference=box,
        anchor_reference=Point2(x=x, y=y, space=CoordinateSpace.REFERENCE),
        confidence=0.8,
    )


def _store(tmp_path: Path) -> Pass1Store:
    """Worker 3 walks right across a fenced zone (x 800–1200) over 8 s; an excavator idles."""
    store = Pass1Store(tmp_path / "pass1.sqlite")
    for frame in range(int(8 * FPS)):
        x = 400.0 + frame * 8.0  # enters the zone at ~3.3 s, leaves at ~6.7 s
        observations = [
            _observation(frame, 3, "person", x, 700.0),
            _observation(frame, 9, "machinery", 1500.0, 650.0),
        ]
        store.add_frame(
            FrameRecord(
                frame_index=frame,
                source_index=frame * 2,
                video_time_s=frame / FPS,
                shot_id="s0",
                transform_valid=True,
                detection_count=len(observations),
            ),
            observations,
        )
    store.mark_complete()
    return store


def _scene():
    site = SiteConfig(
        camera_id="yard",
        reference_image="yard.png",
        zones=(
            SiteZone(
                zone_id="pit",
                proposal_type=ProposalType.RESTRICTED_ZONE,
                label="Excavation pit",
                points=((800.0, 600.0), (1200.0, 600.0), (1200.0, 800.0), (800.0, 800.0)),
            ),
        ),
    )
    return scene_from_site_config(site, shot_ids=["s0"], video_sha256="v")


def test_worker_crossing_a_fenced_zone_yields_one_r3_incident(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        builder = FrameContextBuilder(
            run_id="run-it",
            scene=_scene(),
            geometry_supported=True,
            ppe_available=False,
            machinery_classes=CONFIG.stage1.machinery_classes,
        )
        engine = RuleEngine(run_id="run-it", config=CONFIG.rules)
        result = run_pass2(store, builder, engine, TrackSummaryBuilder("run-it"))

    [incident] = result.rules.incidents
    assert incident.rule_id is RuleId.R3
    assert incident.canonical_track_id == canonical_track_id("s0", 3)
    assert 'restricted zone "Excavation pit"' in incident.observation_text
    assert 3.2 < incident.first_seen_s < 3.5
    assert incident.resolved_at_s is not None and 6.5 < incident.resolved_at_s < 6.9

    coverage = {entry.rule_id: entry for entry in result.rules.coverage}
    assert coverage[RuleId.R1].status is RuleStatus.UNSUPPORTED  # no PPE model yet
    assert coverage[RuleId.R3].status is RuleStatus.EVALUATED_ALERT
    assert coverage[RuleId.R4].status is RuleStatus.INCONCLUSIVE  # no relative plane yet
    assert coverage[RuleId.R4].reason_code == "relative_plane_unavailable"
    assert coverage[RuleId.R5].status is RuleStatus.NOT_APPLICABLE  # no edges configured

    worker = next(t for t in result.tracks_summary.tracks if t.display_label == "3")
    assert [z.proposal_id for z in worker.zones] == ["pit"]
    assert worker.incident_ids == (incident.incident_id,)
    assert len(result.frame_outputs) == int(8 * FPS)


def test_unstable_geometry_turns_zone_rules_unsupported(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        builder = FrameContextBuilder(
            run_id="run-it",
            scene=_scene(),
            geometry_supported=False,
            ppe_available=False,
            machinery_classes=CONFIG.stage1.machinery_classes,
        )
        engine = RuleEngine(run_id="run-it", config=CONFIG.rules)
        result = run_pass2(store, builder, engine, TrackSummaryBuilder("run-it"))
    assert result.rules.incidents == ()
    coverage = {entry.rule_id: entry for entry in result.rules.coverage}
    for rule_id in (RuleId.R3, RuleId.R4, RuleId.R5):
        assert coverage[rule_id].status is RuleStatus.UNSUPPORTED
