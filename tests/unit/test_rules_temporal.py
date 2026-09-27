from __future__ import annotations

import pytest

from pipeline.rules.temporal import (
    EvidenceSample,
    Predicate,
    SubjectTimeline,
    TemporalParams,
    Transition,
    TransitionKind,
    predicate_for,
)
from shared.enums import AlertState, RuleId, RuleStatus
from shared.schemas.incidents import RuleResult

PARAMS = TemporalParams(debounce_s=1.5, grace_s=0.25, hysteresis_s=2.0, cooldown_s=30.0)
HOLDS, CLEAR, UNKNOWN = Predicate.HOLDS, Predicate.CLEAR, Predicate.UNKNOWN


def _sample(time_s: float, confidence: float = 0.9) -> EvidenceSample:
    result = RuleResult(
        rule_id=RuleId.R1,
        status=RuleStatus.EVALUATED_ALERT,
        reason_code="persistent_no_helmet",
        basis="visual",
        track_id="shot:s0/track:1",
        confidence=confidence,
    )
    return EvidenceSample(
        frame_index=round(time_s * 15),
        video_time_s=time_s,
        score=confidence,
        result=result,
    )


def _feed(
    timeline: SubjectTimeline, events: list[tuple[float, Predicate]]
) -> list[tuple[float, Transition]]:
    transitions: list[tuple[float, Transition]] = []
    for time_s, predicate in events:
        sample = _sample(time_s) if predicate is HOLDS else None
        transition = timeline.observe(time_s, predicate, sample)
        if transition is not None:
            transitions.append((time_s, transition))
    return transitions


def _frames(first: int, last: int, predicate: Predicate, fps: float = 15.0) -> list:
    """Frames first..last inclusive at `fps`, as (timestamp, predicate) events."""
    return [(k / fps, predicate) for k in range(first, last + 1)]


def test_predicate_mapping() -> None:
    assert predicate_for(RuleStatus.EVALUATED_ALERT) is HOLDS
    assert predicate_for(RuleStatus.EVALUATED_CLEAR) is CLEAR
    assert predicate_for(RuleStatus.NOT_APPLICABLE) is CLEAR
    assert predicate_for(RuleStatus.INCONCLUSIVE) is UNKNOWN
    assert predicate_for(RuleStatus.UNSUPPORTED) is UNKNOWN


def test_negative_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="debounce_s"):
        TemporalParams(debounce_s=-1.0, grace_s=0.25, hysteresis_s=2.0, cooldown_s=30.0)


def test_alert_confirms_once_debounce_elapses() -> None:
    timeline = SubjectTimeline(PARAMS)
    transitions = _feed(timeline, _frames(0, 40, HOLDS))
    assert len(transitions) == 1
    time_s, transition = transitions[0]
    # 1.5 s at 15 fps: frame 22 is 1.467 s (too early), frame 23 is 1.533 s.
    assert time_s == pytest.approx(23 / 15)
    assert transition.kind is TransitionKind.OPENED
    assert transition.episode.first_seen_s == 0.0
    assert transition.episode.confirmed_at_s == pytest.approx(23 / 15)
    assert timeline.state is AlertState.ALERT


def test_condition_shorter_than_debounce_never_alerts() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 21, HOLDS) + _frames(22, 60, CLEAR)
    assert _feed(timeline, events) == []
    assert timeline.state is AlertState.CLEAR
    assert timeline.episodes == []


def test_debounce_uses_timestamps_under_variable_frame_rate() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = [(t, HOLDS) for t in (0.0, 0.4, 0.9, 1.3, 1.6, 2.0)]
    transitions = _feed(timeline, events)
    assert [t for t, _ in transitions] == [1.6]


def test_debounce_is_rate_independent() -> None:
    at_5fps = SubjectTimeline(PARAMS)
    at_30fps = SubjectTimeline(PARAMS)
    first_5 = _feed(at_5fps, _frames(0, 20, HOLDS, fps=5.0))[0][0]
    first_30 = _feed(at_30fps, _frames(0, 120, HOLDS, fps=30.0))[0][0]
    assert first_5 == pytest.approx(1.6)
    assert first_30 == pytest.approx(1.5)


def test_short_missing_gap_pauses_debounce() -> None:
    timeline = SubjectTimeline(PARAMS)
    # Holds for frames 0-9 (0.6 s), unknown for 10-11, holds again from 12 (0.8 s).
    events = _frames(0, 9, HOLDS) + _frames(10, 11, UNKNOWN) + _frames(12, 40, HOLDS)
    transitions = _feed(timeline, events)
    # 0.6 s banked + 0.9 s more from 0.8 s → confirmed at frame 26, not frame 23.
    assert [round(t * 15) for t, _ in transitions] == [26]
    assert transitions[0][1].episode.first_seen_s == 0.0


def test_absent_frames_count_as_missing_observations() -> None:
    timeline = SubjectTimeline(PARAMS)
    for time_s, _ in _frames(0, 9, HOLDS):
        timeline.observe(time_s, HOLDS, _sample(time_s))
    for time_s, _ in _frames(10, 11, UNKNOWN):
        assert timeline.expire(time_s) is None
    assert timeline.state is AlertState.PENDING_ALERT
    timeline.expire(13 / 15)  # 0.267 s after the last holding frame
    assert timeline.state is AlertState.CLEAR


def test_long_missing_gap_resets_debounce() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 9, HOLDS) + _frames(10, 14, UNKNOWN) + _frames(15, 60, HOLDS)
    transitions = _feed(timeline, events)
    assert len(transitions) == 1
    episode = transitions[0][1].episode
    assert episode.first_seen_s == pytest.approx(1.0)
    assert round(transitions[0][0] * 15) == 38  # 1.0 s + 1.5 s → frame 37.5 → 38


def test_single_clear_frame_resets_pending() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 15, HOLDS) + [(16 / 15, CLEAR)] + _frames(17, 60, HOLDS)
    transitions = _feed(timeline, events)
    assert transitions[0][1].episode.first_seen_s == pytest.approx(17 / 15)


def test_unknown_never_resolves_within_hysteresis() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 30, HOLDS) + _frames(31, 58, UNKNOWN) + _frames(59, 70, HOLDS)
    transitions = _feed(timeline, events)
    assert [transition.kind for _, transition in transitions] == [TransitionKind.OPENED]
    assert timeline.state is AlertState.ALERT


def test_sustained_unknown_resolves_at_last_evidence() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 30, HOLDS) + _frames(31, 70, UNKNOWN)
    transitions = _feed(timeline, events)
    assert [transition.kind for _, transition in transitions] == [
        TransitionKind.OPENED,
        TransitionKind.RESOLVED,
    ]
    episode = transitions[-1][1].episode
    assert episode.resolved_at_s == pytest.approx(30 / 15)
    assert episode.resolution_reason == "evidence_lost"


def test_alert_resolves_after_hysteresis_at_first_clear() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 30, HOLDS) + _frames(31, 70, CLEAR)
    transitions = _feed(timeline, events)
    time_s, resolution = transitions[-1]
    assert resolution.kind is TransitionKind.RESOLVED
    assert round(time_s * 15) == 61  # first clear at frame 31 + 2.0 s
    assert resolution.episode.resolved_at_s == pytest.approx(31 / 15)
    assert resolution.episode.resolution_reason == "condition_cleared"
    assert resolution.episode.duration_s() == pytest.approx(31 / 15)
    assert timeline.state is AlertState.COOLDOWN


def test_clear_blip_does_not_split_the_incident() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 30, HOLDS) + _frames(31, 45, CLEAR) + _frames(46, 80, HOLDS)
    transitions = _feed(timeline, events)
    assert [transition.kind for _, transition in transitions] == [TransitionKind.OPENED]
    assert len(timeline.episodes) == 1


def test_retrigger_within_cooldown_reopens_same_episode() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = (
        _frames(0, 30, HOLDS)  # alert
        + _frames(31, 70, CLEAR)  # resolves at 2.07 s
        + _frames(150, 200, HOLDS)  # returns at 10 s — inside the 30 s cooldown
    )
    transitions = _feed(timeline, events)
    kinds = [transition.kind for _, transition in transitions]
    assert kinds == [TransitionKind.OPENED, TransitionKind.RESOLVED, TransitionKind.REOPENED]
    assert transitions[0][1].episode is transitions[2][1].episode
    assert len(timeline.episodes) == 1
    assert timeline.episodes[0].reopen_count == 1
    assert timeline.episodes[0].resolved_at_s is None


def test_retrigger_after_cooldown_opens_new_episode() -> None:
    timeline = SubjectTimeline(PARAMS)
    events = _frames(0, 30, HOLDS) + _frames(31, 70, CLEAR) + _frames(600, 650, HOLDS)
    transitions = _feed(timeline, events)
    kinds = [transition.kind for _, transition in transitions]
    assert kinds == [TransitionKind.OPENED, TransitionKind.RESOLVED, TransitionKind.OPENED]
    assert len(timeline.episodes) == 2


def test_close_settles_an_active_alert() -> None:
    timeline = SubjectTimeline(PARAMS)
    _feed(timeline, _frames(0, 40, HOLDS))
    transition = timeline.close(5.0, "end_of_clip")
    assert transition is not None
    assert transition.kind is TransitionKind.RESOLVED
    assert transition.episode.resolved_at_s == pytest.approx(40 / 15)
    assert transition.episode.resolution_reason == "end_of_clip"
    assert timeline.state is AlertState.CLEAR


def test_close_during_pending_creates_nothing() -> None:
    timeline = SubjectTimeline(PARAMS)
    _feed(timeline, _frames(0, 10, HOLDS))
    assert timeline.close(1.0, "shot_boundary") is None
    assert timeline.episodes == []


def test_peak_evidence_is_the_highest_scoring_frame() -> None:
    timeline = SubjectTimeline(PARAMS)
    for k in range(0, 40):
        time_s = k / 15
        confidence = 0.95 if k == 30 else 0.8
        timeline.observe(time_s, HOLDS, _sample(time_s, confidence))
    assert timeline.episodes[0].peak.frame_index == 30
    assert timeline.episodes[0].mean_confidence == pytest.approx((39 * 0.8 + 0.95) / 40)


def test_holding_observation_requires_a_sample() -> None:
    with pytest.raises(ValueError, match="evidence sample"):
        SubjectTimeline(PARAMS).observe(0.0, HOLDS, None)
