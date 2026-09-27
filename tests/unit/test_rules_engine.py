from __future__ import annotations

import pytest
from rule_frames import (
    frame,
    helmet,
    interval,
    machine,
    membership,
    proximity,
    rules_config,
    tid,
    times,
    vest,
    worker,
    zone,
)

from pipeline.rules import r1, r2, r3, r4, r5
from pipeline.rules.catalogue import load_catalogue
from pipeline.rules.engine import EngineResult, RuleEngine
from shared.enums import (
    AlertState,
    DistanceBand,
    HelmetState,
    OperatingState,
    RuleId,
    RuleStatus,
    VestState,
)
from shared.schemas.frame import FeedCapabilities, FrameContext
from shared.schemas.incidents import IncidentRecord, RuleResult
from shared.schemas.scene import ProposalType

NO_HELMET = helmet(HelmetState.NO_HELMET, 0.9)
WITH_HELMET = helmet(HelmetState.HELMET, 0.95)
WITH_VEST = vest(VestState.VEST, 0.95)


def run(contexts: list[FrameContext], run_id: str = "run-test") -> EngineResult:
    engine = RuleEngine(run_id=run_id, config=rules_config())
    for context in contexts:
        engine.evaluate_frame(context)
    return engine.finalize()


def helmet_clip(run_id: str = "run-test") -> list[FrameContext]:
    """Worker 7 without a helmet from 1 s to 4 s, wearing one from 4 s to 8 s."""
    contexts = []
    for t in times(0.0, 8.0):
        record = NO_HELMET if 1.0 <= t < 4.0 else WITH_HELMET
        contexts.append(
            frame(
                t, run_id=run_id, tracks=(worker(7, helmet_record=record, vest_record=WITH_VEST),)
            )
        )
    return contexts


def by_rule(result: EngineResult, rule_id: RuleId) -> list[IncidentRecord]:
    return [incident for incident in result.incidents if incident.rule_id is rule_id]


def test_r1_clip_produces_one_catalogue_driven_incident() -> None:
    result = run(helmet_clip())
    [incident] = result.incidents
    catalogue = load_catalogue()

    assert incident.rule_id is RuleId.R1
    assert incident.status is RuleStatus.EVALUATED_ALERT
    assert incident.reason_code == "persistent_no_helmet"
    assert incident.severity == "high"
    assert incident.basis == "visual"
    assert incident.references == ("BOCW Rule 54", "BOCW Rule 46(1)")
    assert incident.action_code == "VERIFY_HELMET"
    assert incident.catalogue_sha256 == catalogue.sha256
    assert incident.canonical_track_id == tid(7)

    assert incident.first_seen_s == pytest.approx(1.0)
    assert incident.confirmed_at_s == pytest.approx(1.0 + 23 / 15)
    assert incident.resolved_at_s == pytest.approx(4.0)
    assert incident.resolution_reason == "condition_cleared"
    assert incident.confidence == pytest.approx(0.9)
    assert incident.observation_text == (
        "Worker 7 was repeatedly observed without a visible helmet for 3.0s."
    )

    assert incident.evidence_path == f"evidence/{incident.incident_id}.jpg"
    assert incident.evidence_crop_path == f"evidence/{incident.incident_id}_crop.jpg"
    assert incident.helmet == NO_HELMET
    assert incident.adjudication_candidate is False


def test_incident_ids_are_deterministic_per_run() -> None:
    first = run(helmet_clip("run-a"), "run-a").incidents[0].incident_id
    again = run(helmet_clip("run-a"), "run-a").incidents[0].incident_id
    other = run(helmet_clip("run-b"), "run-b").incidents[0].incident_id
    assert first == again
    assert first != other
    assert first.startswith("r1-")


def test_coverage_always_lists_all_rules() -> None:
    result = run(helmet_clip())
    assert [entry.rule_id for entry in result.coverage] == list(RuleId)
    coverage = {entry.rule_id: entry for entry in result.coverage}

    assert coverage[RuleId.R1].status is RuleStatus.EVALUATED_ALERT
    assert coverage[RuleId.R1].incident_count == 1
    assert sum(coverage[RuleId.R1].status_counts.values()) == len(times(0.0, 8.0))
    # No scene proposals and no machines in this clip.
    assert coverage[RuleId.R2].status is RuleStatus.NOT_APPLICABLE
    assert coverage[RuleId.R3].status is RuleStatus.NOT_APPLICABLE
    assert coverage[RuleId.R3].reason_code == "no_supported_restricted_zone"
    assert coverage[RuleId.R4].reason_code == "no_machinery_tracks"
    assert coverage[RuleId.R5].reason_code == "no_visible_open_edge_candidate"


def test_unsupported_feed_is_reported_not_hidden() -> None:
    contexts = [
        frame(t, geometry=False, tracks=(worker(1, helmet_record=WITH_HELMET),))
        for t in times(0.0, 2.0)
    ]
    coverage = {entry.rule_id: entry for entry in run(contexts).coverage}
    for rule_id in (RuleId.R3, RuleId.R4, RuleId.R5):
        assert coverage[rule_id].status is RuleStatus.UNSUPPORTED
        assert coverage[rule_id].reason_code == "geometry_unstable_for_feed"
    assert coverage[RuleId.R1].status is RuleStatus.EVALUATED_CLEAR


def test_transient_condition_is_reported_as_not_persistent() -> None:
    contexts = [
        frame(t, tracks=(worker(1, helmet_record=NO_HELMET if t < 0.5 else None),))
        for t in times(0.0, 3.0)
    ]
    coverage = {entry.rule_id: entry for entry in run(contexts).coverage}
    assert coverage[RuleId.R1].status is RuleStatus.EVALUATED_CLEAR
    assert coverage[RuleId.R1].reason_code == "alert_condition_not_persistent"


def test_frame_output_exposes_pending_then_active_alerts() -> None:
    engine = RuleEngine(run_id="run-test", config=rules_config())
    states: list[AlertState | None] = []
    opened: list[str] = []
    for context in helmet_clip():
        output = engine.evaluate_frame(context)
        states.append(output.active_alerts[0].state if output.active_alerts else None)
        opened.extend(output.opened_incident_ids)
    assert states[0] is None
    assert states[15] is AlertState.PENDING_ALERT  # 1.0 s: condition starts
    assert states[45] is AlertState.ALERT
    assert states[65] is AlertState.PENDING_RESOLUTION  # 4.33 s: clearing
    assert states[-1] is None
    [incident] = engine.finalize().incidents
    assert opened == [incident.incident_id]


def test_r4_incident_uses_closest_approach_as_evidence() -> None:
    contexts = []
    for index, t in enumerate(times(0.0, 3.0)):
        upper = 1.4 - 0.02 * min(index, 30)  # approaches until frame 30, then holds
        pair = proximity(
            3,
            "shot:s0/track:9",
            DistanceBand.NEAR,
            OperatingState.ACTIVE,
            distance=interval(upper - 0.4, upper),
        )
        contexts.append(
            frame(
                t,
                tracks=(worker(3, helmet_record=WITH_HELMET, vest_record=WITH_VEST),),
                machinery=(machine("shot:s0/track:9", class_name="excavator"),),
                proximities=(pair,),
            )
        )
    [incident] = by_rule(run(contexts), RuleId.R4)
    assert incident.object_id == "shot:s0/track:9"
    assert incident.relative_band is DistanceBand.NEAR
    assert incident.source_frame_index == 30
    assert incident.distance is not None
    assert incident.distance.upper_wh == pytest.approx(0.8)
    assert "near-machine band of excavator 9" in incident.observation_text
    assert incident.references == ("BOCW Rule 125(h)", "BOCW Rule 130")
    assert incident.severity == "high"


def test_r2_road_work_context_adds_contextual_references() -> None:
    road = zone("road-1", ProposalType.TRAFFIC_ZONE, road_work=True)
    contexts = [
        frame(
            t,
            tracks=(worker(2, helmet_record=WITH_HELMET, vest_record=vest(VestState.NO_VEST)),),
            proposals=(road,),
            memberships=(membership(2, "road-1", True),),
        )
        for t in times(0.0, 3.0)
    ]
    [incident] = by_rule(run(contexts), RuleId.R2)
    assert incident.references == ("BOCW Rule 48(1)", "BOCW Rule 92(c)")
    assert incident.proposal_ids == ("road-1",)


def test_r3_observation_names_the_zone() -> None:
    fence = zone("zone-1", label="Crane swing area")
    contexts = [
        frame(
            t,
            tracks=(worker(4, helmet_record=WITH_HELMET, vest_record=WITH_VEST),),
            proposals=(fence,),
            memberships=(membership(4, "zone-1", True),),
        )
        for t in times(0.0, 2.0)
    ]
    [incident] = by_rule(run(contexts), RuleId.R3)
    assert 'restricted zone "Crane swing area"' in incident.observation_text
    assert incident.references == ()
    assert incident.resolution_reason == "end_of_clip"


def test_uncertain_ppe_marks_incident_for_adjudication() -> None:
    fence = zone("zone-1")
    contexts = [
        frame(
            t,
            tracks=(
                worker(5, helmet_record=WITH_HELMET, vest_record=vest(VestState.UNKNOWN, 0.4)),
            ),
            proposals=(fence,),
            memberships=(membership(5, "zone-1", True),),
        )
        for t in times(0.0, 2.0)
    ]
    [incident] = by_rule(run(contexts), RuleId.R3)
    assert incident.adjudication_candidate is True
    assert incident.vest is not None and incident.vest.state is VestState.UNKNOWN


def test_shot_boundary_closes_open_incidents() -> None:
    first_shot = [frame(t, tracks=(worker(1, helmet_record=NO_HELMET),)) for t in times(0.0, 3.0)]
    second_shot = [
        frame(t, shot="s1", tracks=(worker(1, helmet_record=NO_HELMET, shot="s1"),))
        for t in times(3.0, 6.0)
    ]
    result = run(first_shot + second_shot)
    first, second = result.incidents
    assert first.shot_id == "s0" and first.resolution_reason == "shot_boundary"
    assert second.shot_id == "s1" and second.canonical_track_id == tid(1, "s1")
    assert first.incident_id != second.incident_id


def test_frames_must_arrive_in_time_order() -> None:
    engine = RuleEngine(run_id="run-test", config=rules_config())
    engine.evaluate_frame(frame(1.0))
    with pytest.raises(ValueError, match="time order"):
        engine.evaluate_frame(frame(0.5))


def test_frames_from_another_run_are_rejected() -> None:
    engine = RuleEngine(run_id="run-a", config=rules_config())
    with pytest.raises(ValueError, match="run-b"):
        engine.evaluate_frame(frame(0.0, run_id="run-b"))


def test_engine_cannot_be_reused_after_finalize() -> None:
    engine = RuleEngine(run_id="run-test", config=rules_config())
    engine.evaluate_frame(frame(0.0))
    engine.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        engine.evaluate_frame(frame(1.0))


class _DuplicatingEvaluator:
    rule_id = RuleId.R1

    def evaluate(self, context: FrameContext) -> list[RuleResult]:
        result = RuleResult(
            rule_id=RuleId.R1,
            status=RuleStatus.EVALUATED_CLEAR,
            reason_code="x",
            basis="visual",
            track_id=tid(1),
        )
        return [result, result]


def test_evaluator_contract_is_enforced() -> None:
    engine = RuleEngine(
        run_id="run-test", config=rules_config(), evaluators=[_DuplicatingEvaluator()]
    )
    with pytest.raises(ValueError, match="two results"):
        engine.evaluate_frame(frame(0.0))


def test_catalogue_alert_reasons_match_evaluators() -> None:
    catalogue = load_catalogue()
    reasons = {
        RuleId.R1: r1.ALERT_REASON,
        RuleId.R2: r2.ALERT_REASON,
        RuleId.R3: r3.ALERT_REASON,
        RuleId.R4: r4.ALERT_REASON,
        RuleId.R5: r5.ALERT_REASON,
    }
    for rule_id, reason in reasons.items():
        assert catalogue.rule(rule_id).alert_reason_code == reason


def test_missing_ppe_capability_keeps_r1_r2_visible() -> None:
    contexts = [
        frame(t, capabilities=FeedCapabilities(ppe_classification=False), tracks=(worker(1),))
        for t in times(0.0, 1.0)
    ]
    coverage = {entry.rule_id: entry for entry in run(contexts).coverage}
    assert coverage[RuleId.R1].status is RuleStatus.UNSUPPORTED
    assert coverage[RuleId.R2].reason_code == "vest_detection_unavailable"
