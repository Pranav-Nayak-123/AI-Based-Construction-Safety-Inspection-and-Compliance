"""Every status path of every rule evaluator (v7 §40.2–40.6, §47.1)."""

from __future__ import annotations

from rule_frames import (
    edge,
    edge_proximity,
    frame,
    helmet,
    interval,
    machine,
    membership,
    proximity,
    tid,
    vest,
    worker,
    zone,
)

from pipeline.rules.r1 import R1MissingHelmet
from pipeline.rules.r2 import R2MissingHiVis
from pipeline.rules.r3 import R3RestrictedZone
from pipeline.rules.r4 import R4MachineryProximity
from pipeline.rules.r5 import R5FallEdge
from shared.enums import DistanceBand, HelmetState, OperatingState, RuleStatus, VestState
from shared.schemas.frame import FeedCapabilities, FrameContext
from shared.schemas.incidents import RuleResult
from shared.schemas.scene import ProposalType, ProtectionState, ProtectionVisibility

ALERT = RuleStatus.EVALUATED_ALERT
CLEAR = RuleStatus.EVALUATED_CLEAR
INCONCLUSIVE = RuleStatus.INCONCLUSIVE
NOT_APPLICABLE = RuleStatus.NOT_APPLICABLE
UNSUPPORTED = RuleStatus.UNSUPPORTED

R1 = R1MissingHelmet(basis="visual", confidence_floor=0.70)
R2 = R2MissingHiVis(basis="heuristic", confidence_floor=0.70)
R3 = R3RestrictedZone(basis="heuristic")
R4 = R4MachineryProximity(basis="heuristic")
R5 = R5FallEdge(basis="heuristic", edge_max_wh=0.8)

NO_PPE = FeedCapabilities(ppe_classification=False)
NO_PEOPLE = FeedCapabilities(person_detection=False)
NO_SCENE = FeedCapabilities(scene_state=False)
NO_PLANE = FeedCapabilities(relative_plane=False)


def only(results: list[RuleResult]) -> RuleResult:
    assert len(results) == 1
    return results[0]


def outcome(results: list[RuleResult]) -> tuple[RuleStatus, str]:
    result = only(results)
    return result.status, result.reason_code


# ---------------------------------------------------------------------------- R1


class TestR1:
    def test_unsupported_without_person_detection(self) -> None:
        assert outcome(R1.evaluate(frame(capabilities=NO_PEOPLE))) == (
            UNSUPPORTED,
            "person_detection_unavailable",
        )

    def test_unsupported_without_ppe_classifier(self) -> None:
        context = frame(capabilities=NO_PPE, tracks=(worker(1),))
        assert outcome(R1.evaluate(context)) == (UNSUPPORTED, "helmet_classifier_unavailable")

    def test_not_applicable_without_workers(self) -> None:
        assert outcome(R1.evaluate(frame())) == (NOT_APPLICABLE, "no_person_tracks")

    def test_frame_level_results_have_no_track(self) -> None:
        assert only(R1.evaluate(frame())).track_id is None

    def test_missing_unknown_or_hidden_helmet_is_inconclusive(self) -> None:
        workers = (
            worker(1),
            worker(2, helmet_record=helmet(HelmetState.UNKNOWN, 0.5)),
            worker(3, helmet_record=helmet(HelmetState.NO_HELMET, 0.95, visible=False)),
        )
        results = R1.evaluate(frame(tracks=workers))
        assert [(r.status, r.reason_code) for r in results] == [
            (INCONCLUSIVE, "helmet_visibility_insufficient")
        ] * 3

    def test_weak_no_helmet_is_inconclusive_not_clear(self) -> None:
        context = frame(tracks=(worker(1, helmet_record=helmet(HelmetState.NO_HELMET, 0.69)),))
        assert outcome(R1.evaluate(context)) == (INCONCLUSIVE, "helmet_confidence_below_floor")

    def test_confident_no_helmet_holds_the_alert_condition(self) -> None:
        context = frame(tracks=(worker(1, helmet_record=helmet(HelmetState.NO_HELMET, 0.70)),))
        result = only(R1.evaluate(context))
        assert (result.status, result.reason_code) == (ALERT, "persistent_no_helmet")
        assert result.track_id == tid(1)
        assert result.confidence == 0.70
        assert result.basis == "visual"

    def test_helmet_present_is_clear(self) -> None:
        context = frame(tracks=(worker(1, helmet_record=helmet(HelmetState.HELMET)),))
        assert outcome(R1.evaluate(context)) == (CLEAR, "helmet_not_observed_missing")

    def test_one_result_per_worker(self) -> None:
        workers = tuple(worker(i, helmet_record=helmet(HelmetState.HELMET)) for i in range(4))
        results = R1.evaluate(frame(tracks=workers))
        assert [r.track_id for r in results] == [tid(i) for i in range(4)]


# ---------------------------------------------------------------------------- R2

NO_VEST = vest(VestState.NO_VEST, 0.9)


def r2_frame(**kwargs: object) -> FrameContext:
    kwargs.setdefault("tracks", (worker(1, vest_record=NO_VEST),))
    return frame(**kwargs)  # type: ignore[arg-type]


class TestR2:
    traffic = zone("traffic-1", ProposalType.TRAFFIC_ZONE)

    def test_unsupported_paths(self) -> None:
        assert outcome(R2.evaluate(frame(capabilities=NO_PEOPLE))) == (
            UNSUPPORTED,
            "person_detection_unavailable",
        )
        assert outcome(R2.evaluate(frame(capabilities=NO_PPE))) == (
            UNSUPPORTED,
            "vest_detection_unavailable",
        )

    def test_not_applicable_without_workers(self) -> None:
        assert outcome(R2.evaluate(frame())) == (NOT_APPLICABLE, "no_person_tracks")

    def test_no_movement_area_is_not_applicable(self) -> None:
        assert outcome(R2.evaluate(r2_frame())) == (NOT_APPLICABLE, "movement_area_absent")

    def test_outside_traffic_zone_is_not_applicable(self) -> None:
        context = r2_frame(
            proposals=(self.traffic,), memberships=(membership(1, "traffic-1", False),)
        )
        assert outcome(R2.evaluate(context)) == (NOT_APPLICABLE, "movement_area_absent")

    def test_no_vest_in_traffic_zone_holds(self) -> None:
        context = r2_frame(
            proposals=(self.traffic,), memberships=(membership(1, "traffic-1", True),)
        )
        result = only(R2.evaluate(context))
        assert (result.status, result.reason_code) == (
            ALERT,
            "persistent_no_vest_in_movement_area",
        )
        assert result.proposal_ids == ("traffic-1",)

    def test_no_vest_in_machinery_caution_band_holds(self) -> None:
        context = r2_frame(
            machinery=(machine("m1"),),
            proximities=(
                proximity(
                    1,
                    "m1",
                    DistanceBand.CAUTION,
                    OperatingState.STATIONARY,
                    distance=interval(1.8, 2.6),
                ),
            ),
        )
        result = only(R2.evaluate(context))
        assert result.status is ALERT
        assert result.object_id == "m1"

    def test_vest_present_in_area_is_clear(self) -> None:
        context = r2_frame(
            tracks=(worker(1, vest_record=vest(VestState.VEST)),),
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", True),),
        )
        assert outcome(R2.evaluate(context)) == (CLEAR, "hi_vis_condition_not_observed")

    def test_unassessable_vest_in_area_is_inconclusive(self) -> None:
        context = r2_frame(
            tracks=(worker(1, vest_record=vest(VestState.UNKNOWN, 0.4)),),
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", True),),
        )
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "vest_visibility_insufficient")

    def test_weak_no_vest_is_inconclusive(self) -> None:
        context = r2_frame(
            tracks=(worker(1, vest_record=vest(VestState.NO_VEST, 0.6)),),
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", True),),
        )
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "vest_confidence_below_floor")

    def test_uncertain_area_with_no_vest_is_inconclusive(self) -> None:
        context = r2_frame(
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", True, uncertain=True),),
        )
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "movement_area_uncertain")

    def test_uncertain_area_with_vest_is_still_clear(self) -> None:
        context = r2_frame(
            tracks=(worker(1, vest_record=vest(VestState.VEST)),),
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", None),),
        )
        assert outcome(R2.evaluate(context)) == (CLEAR, "hi_vis_condition_not_observed")

    def test_unknown_scene_state_makes_area_uncertain(self) -> None:
        context = r2_frame(capabilities=NO_SCENE)
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "movement_area_uncertain")

    def test_unstable_geometry_with_traffic_zones_is_uncertain(self) -> None:
        context = r2_frame(geometry=False, proposals=(self.traffic,))
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "movement_area_uncertain")

    def test_machine_without_proximity_estimate_is_uncertain(self) -> None:
        context = r2_frame(machinery=(machine("m1"),))
        assert outcome(R2.evaluate(context)) == (INCONCLUSIVE, "movement_area_uncertain")

    def test_positive_area_beats_uncertainty(self) -> None:
        context = r2_frame(
            capabilities=NO_PLANE,
            machinery=(machine("m1"),),
            proposals=(self.traffic,),
            memberships=(membership(1, "traffic-1", True),),
        )
        assert only(R2.evaluate(context)).status is ALERT


# ---------------------------------------------------------------------------- R3


class TestR3:
    fence = zone("zone-1", label="Crane swing area")
    pit = zone("zone-2")

    def r3_frame(self, *memberships_: object, **kwargs: object) -> FrameContext:
        kwargs.setdefault("tracks", (worker(1),))
        kwargs.setdefault("proposals", (self.fence,))
        return frame(memberships=memberships_, **kwargs)  # type: ignore[arg-type]

    def test_unstable_geometry_is_unsupported(self) -> None:
        context = self.r3_frame(geometry=False)
        assert outcome(R3.evaluate(context)) == (UNSUPPORTED, "geometry_unstable_for_feed")

    def test_missing_scene_state_is_inconclusive(self) -> None:
        context = self.r3_frame(capabilities=NO_SCENE)
        assert outcome(R3.evaluate(context)) == (
            INCONCLUSIVE,
            "restricted_zone_scene_state_unavailable",
        )

    def test_no_restricted_zone_is_not_applicable(self) -> None:
        context = self.r3_frame(proposals=(zone("t", ProposalType.TRAFFIC_ZONE),))
        assert outcome(R3.evaluate(context)) == (NOT_APPLICABLE, "no_supported_restricted_zone")

    def test_no_workers_is_not_applicable(self) -> None:
        assert outcome(R3.evaluate(self.r3_frame(tracks=()))) == (
            NOT_APPLICABLE,
            "no_person_tracks",
        )

    def test_inside_zone_holds(self) -> None:
        result = only(R3.evaluate(self.r3_frame(membership(1, "zone-1", True))))
        assert (result.status, result.reason_code) == (ALERT, "restricted_zone_entry")
        assert result.proposal_ids == ("zone-1",)

    def test_outside_zone_is_clear(self) -> None:
        context = self.r3_frame(membership(1, "zone-1", False))
        assert outcome(R3.evaluate(context)) == (CLEAR, "worker_outside_restricted_zone")

    def test_boundary_uncertain_is_inconclusive(self) -> None:
        context = self.r3_frame(membership(1, "zone-1", True, uncertain=True))
        assert outcome(R3.evaluate(context)) == (INCONCLUSIVE, "zone_boundary_uncertain")

    def test_missing_anchor_is_inconclusive(self) -> None:
        assert outcome(R3.evaluate(self.r3_frame())) == (
            INCONCLUSIVE,
            "worker_anchor_unavailable",
        )

    def test_any_zone_entry_wins_over_other_zones(self) -> None:
        context = self.r3_frame(
            membership(1, "zone-1", False),
            membership(1, "zone-2", True),
            proposals=(self.fence, self.pit),
        )
        result = only(R3.evaluate(context))
        assert result.status is ALERT
        assert result.proposal_ids == ("zone-2",)


# ---------------------------------------------------------------------------- R4


class TestR4:
    def r4_frame(self, *pairs: object, **kwargs: object) -> FrameContext:
        kwargs.setdefault("tracks", (worker(1),))
        kwargs.setdefault("machinery", (machine("m1"),))
        return frame(proximities=pairs, **kwargs)  # type: ignore[arg-type]

    def test_frame_level_paths(self) -> None:
        assert outcome(R4.evaluate(self.r4_frame(geometry=False))) == (
            UNSUPPORTED,
            "geometry_unstable_for_feed",
        )
        assert outcome(R4.evaluate(self.r4_frame(machinery=()))) == (
            NOT_APPLICABLE,
            "no_machinery_tracks",
        )
        assert outcome(R4.evaluate(self.r4_frame(tracks=()))) == (
            NOT_APPLICABLE,
            "no_worker_machinery_overlap",
        )
        assert outcome(R4.evaluate(self.r4_frame(capabilities=NO_PLANE))) == (
            INCONCLUSIVE,
            "relative_plane_unavailable",
        )

    def test_active_machine_in_near_band_holds(self) -> None:
        pair = proximity(1, "m1", DistanceBand.NEAR, distance=interval(0.6, 1.2))
        result = only(R4.evaluate(self.r4_frame(pair)))
        assert (result.status, result.reason_code) == (ALERT, "active_machine_near_band")
        assert result.object_id == "m1"
        assert result.relative_band is DistanceBand.NEAR
        assert result.distance is not None and result.distance.upper_wh == 1.2

    def test_caution_band_is_clear_even_with_unknown_state(self) -> None:
        pair = proximity(1, "m1", DistanceBand.CAUTION, OperatingState.UNKNOWN)
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (CLEAR, "outside_active_near_band")

    def test_stationary_machine_never_alerts(self) -> None:
        pair = proximity(1, "m1", DistanceBand.NEAR, OperatingState.STATIONARY)
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (CLEAR, "outside_active_near_band")
        pair = proximity(1, "m1", DistanceBand.INDETERMINATE, OperatingState.STATIONARY)
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (CLEAR, "outside_active_near_band")

    def test_near_band_with_unknown_state_is_inconclusive(self) -> None:
        pair = proximity(1, "m1", DistanceBand.NEAR, OperatingState.UNKNOWN)
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (
            INCONCLUSIVE,
            "operating_state_unknown",
        )

    def test_straddling_interval_is_inconclusive(self) -> None:
        pair = proximity(1, "m1", DistanceBand.INDETERMINATE, distance=interval(1.2, 1.9))
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (INCONCLUSIVE, "inconclusive_distance")

    def test_incompatible_plane_is_inconclusive(self) -> None:
        pair = proximity(1, "m1", DistanceBand.NEAR, compatible=False)
        assert outcome(R4.evaluate(self.r4_frame(pair))) == (
            INCONCLUSIVE,
            "target_plane_incompatible",
        )

    def test_missing_pair_is_inconclusive(self) -> None:
        assert outcome(R4.evaluate(self.r4_frame())) == (INCONCLUSIVE, "proximity_unavailable")

    def test_nearest_alerting_machine_is_reported(self) -> None:
        context = self.r4_frame(
            proximity(1, "m1", DistanceBand.NEAR, distance=interval(1.0, 1.4)),
            proximity(1, "m2", DistanceBand.NEAR, distance=interval(0.2, 0.6)),
            machinery=(machine("m1"), machine("m2")),
        )
        assert only(R4.evaluate(context)).object_id == "m2"

    def test_alert_on_one_machine_beats_clear_on_another(self) -> None:
        context = self.r4_frame(
            proximity(1, "m1", DistanceBand.CLEAR, distance=interval(4.0, 5.0)),
            proximity(1, "m2", DistanceBand.NEAR, distance=interval(0.5, 1.0)),
            machinery=(machine("m1"), machine("m2")),
        )
        assert only(R4.evaluate(context)).status is ALERT


# ---------------------------------------------------------------------------- R5


class TestR5:
    def r5_frame(self, *pairs: object, **kwargs: object) -> FrameContext:
        kwargs.setdefault("tracks", (worker(1),))
        kwargs.setdefault("proposals", (edge("edge-1"),))
        return frame(edges=pairs, **kwargs)  # type: ignore[arg-type]

    def test_frame_level_paths(self) -> None:
        assert outcome(R5.evaluate(self.r5_frame(geometry=False))) == (
            UNSUPPORTED,
            "geometry_unstable_for_feed",
        )
        assert outcome(R5.evaluate(self.r5_frame(proposals=()))) == (
            NOT_APPLICABLE,
            "no_visible_open_edge_candidate",
        )
        assert outcome(R5.evaluate(self.r5_frame(tracks=()))) == (
            NOT_APPLICABLE,
            "no_person_tracks",
        )
        assert outcome(R5.evaluate(self.r5_frame(capabilities=NO_SCENE))) == (
            INCONCLUSIVE,
            "edge_scene_state_unavailable",
        )

    def test_unguarded_edge_within_band_holds(self) -> None:
        pair = edge_proximity(1, "edge-1", interval(0.3, 0.7))
        result = only(R5.evaluate(self.r5_frame(pair)))
        assert (result.status, result.reason_code) == (ALERT, "visible_unguarded_edge")
        assert result.proposal_ids == ("edge-1",)

    def test_exact_upper_bound_satisfies_the_band(self) -> None:
        pair = edge_proximity(1, "edge-1", interval(0.4, 0.8))
        assert only(R5.evaluate(self.r5_frame(pair))).status is ALERT

    def test_interval_crossing_the_band_is_inconclusive(self) -> None:
        pair = edge_proximity(1, "edge-1", interval(0.5, 1.1))
        assert outcome(R5.evaluate(self.r5_frame(pair))) == (INCONCLUSIVE, "inconclusive_distance")

    def test_outside_band_is_clear_whatever_the_semantics(self) -> None:
        unknown_edge = edge("edge-1", drop=None, visibility=None, protection=None)
        pair = edge_proximity(1, "edge-1", interval(1.2, 2.0))
        context = self.r5_frame(pair, proposals=(unknown_edge,))
        assert outcome(R5.evaluate(context)) == (CLEAR, "guarded_or_outside_edge_band")

    def test_guarded_edge_is_clear_without_distance(self) -> None:
        guarded = edge("edge-1", protection=ProtectionState.GUARDED)
        context = self.r5_frame(proposals=(guarded,), capabilities=NO_PLANE)
        assert outcome(R5.evaluate(context)) == (CLEAR, "guarded_or_outside_edge_band")

    def test_edge_that_is_not_a_drop_is_clear(self) -> None:
        context = self.r5_frame(proposals=(edge("edge-1", drop=False),))
        assert outcome(R5.evaluate(context)) == (CLEAR, "edge_not_a_drop")

    def test_semantics_unknown_near_the_edge_is_inconclusive(self) -> None:
        pair = edge_proximity(1, "edge-1", interval(0.2, 0.5))
        cases = [
            (edge("edge-1", drop=None), "edge_semantics_unknown"),
            (
                edge("edge-1", visibility=ProtectionVisibility.INSUFFICIENT),
                "protection_visibility_insufficient",
            ),
            (edge("edge-1", protection=ProtectionState.UNKNOWN), "protection_state_unknown"),
        ]
        for proposal, reason in cases:
            context = self.r5_frame(pair, proposals=(proposal,))
            assert outcome(R5.evaluate(context)) == (INCONCLUSIVE, reason)

    def test_geometry_gaps_are_inconclusive(self) -> None:
        assert outcome(R5.evaluate(self.r5_frame(capabilities=NO_PLANE))) == (
            INCONCLUSIVE,
            "relative_plane_unavailable",
        )
        assert outcome(R5.evaluate(self.r5_frame())) == (
            INCONCLUSIVE,
            "edge_distance_unavailable",
        )
        pair = edge_proximity(1, "edge-1", interval(0.2, 0.5), compatible=False)
        assert outcome(R5.evaluate(self.r5_frame(pair))) == (
            INCONCLUSIVE,
            "target_plane_incompatible",
        )
