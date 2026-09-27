"""Timestamp-based debounce, hysteresis and cooldown for one (rule, subject) pair.

Semantics (v7 §40.1, §11.2):

* An alert is confirmed only after the condition has held for `debounce_s` of video time.
* A missing or unknown observation *pauses* the debounce clock for at most `grace_s`;
  a longer gap resets it. Unknown never counts for or against the condition.
* A confirmed alert resolves after `hysteresis_s` of continuous clear observations;
  its resolution time is the first clear observation.
* Within `cooldown_s` after a resolution, a re-confirmed condition reopens the same
  episode instead of creating a duplicate incident.

Everything is driven by video timestamps, never frame counts, so the behaviour is the same
at 15 fps, 5 fps, or a variable frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from shared.enums import AlertState, RuleStatus
from shared.schemas.incidents import RuleResult
from shared.schemas.tracks import HelmetRecord, VestRecord

# Summed frame intervals drift by float rounding; 1 µs is far below one frame.
_TIME_EPSILON_S = 1e-6


class Predicate(StrEnum):
    HOLDS = "holds"
    CLEAR = "clear"
    UNKNOWN = "unknown"


def predicate_for(status: RuleStatus) -> Predicate:
    if status is RuleStatus.EVALUATED_ALERT:
        return Predicate.HOLDS
    if status in (RuleStatus.EVALUATED_CLEAR, RuleStatus.NOT_APPLICABLE):
        return Predicate.CLEAR
    return Predicate.UNKNOWN


@dataclass(frozen=True)
class TemporalParams:
    debounce_s: float
    grace_s: float
    hysteresis_s: float
    cooldown_s: float

    def __post_init__(self) -> None:
        for name in ("debounce_s", "grace_s", "hysteresis_s", "cooldown_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EvidenceSample:
    """One frame on which the alert condition held, with what is needed for the card."""

    frame_index: int
    video_time_s: float
    score: float
    result: RuleResult
    helmet: HelmetRecord | None = None
    vest: VestRecord | None = None
    context_tags: frozenset[str] = frozenset()
    zone_label: str | None = None
    object_label: str | None = None


@dataclass
class Episode:
    """Evidence accumulated for one incident, from first sighting to resolution."""

    first_seen_s: float
    confirmed_at_s: float
    last_holds_s: float
    peak: EvidenceSample
    confidence_sum: float = 0.0
    confidence_count: int = 0
    context_tags: set[str] = field(default_factory=set)
    resolved_at_s: float | None = None
    resolution_reason: str | None = None
    reopen_count: int = 0

    def add(self, sample: EvidenceSample) -> None:
        self.last_holds_s = max(self.last_holds_s, sample.video_time_s)
        if sample.score > self.peak.score:
            self.peak = sample
        if sample.result.confidence is not None:
            self.confidence_sum += sample.result.confidence
            self.confidence_count += 1
        self.context_tags.update(sample.context_tags)

    @property
    def mean_confidence(self) -> float | None:
        if self.confidence_count == 0:
            return None
        return self.confidence_sum / self.confidence_count

    def duration_s(self) -> float:
        end = self.resolved_at_s if self.resolved_at_s is not None else self.last_holds_s
        return max(0.0, end - self.first_seen_s)


class TransitionKind(StrEnum):
    OPENED = "opened"
    REOPENED = "reopened"
    RESOLVED = "resolved"


@dataclass(frozen=True)
class Transition:
    kind: TransitionKind
    episode: Episode


class SubjectTimeline:
    """State machine for one (rule, subject) pair."""

    def __init__(self, params: TemporalParams) -> None:
        self.params = params
        self.state = AlertState.CLEAR
        self.episode: Episode | None = None
        self.episodes: list[Episode] = []
        self._pending: list[EvidenceSample] = []
        self._pending_since_s = 0.0
        self._accumulated_s = 0.0
        self._last_holds_s = 0.0
        self._previous_holds = False
        self._clear_since_s = 0.0
        self._cooldown_until_s = float("-inf")

    # ------------------------------------------------------------------ public API

    def observe(
        self,
        time_s: float,
        predicate: Predicate,
        sample: EvidenceSample | None,
    ) -> Transition | None:
        """Feed one observation. `sample` is required when the condition holds."""
        self._leave_expired_cooldown(time_s)
        if predicate is Predicate.HOLDS:
            if sample is None:
                raise ValueError("a holding observation needs an evidence sample")
            return self._on_holds(time_s, sample)
        if predicate is Predicate.CLEAR:
            return self._on_clear(time_s)
        return self._on_missing(time_s)

    def expire(self, time_s: float) -> Transition | None:
        """The subject was not observed on this frame: treat as a missing observation."""
        self._leave_expired_cooldown(time_s)
        return self._on_missing(time_s)

    def close(self, time_s: float, reason: str) -> Transition | None:
        """End of shot or clip: settle whatever is in flight."""
        transition: Transition | None = None
        if self.state is AlertState.ALERT and self.episode is not None:
            transition = self._resolve(self.episode.last_holds_s, reason)
        elif self.state is AlertState.PENDING_RESOLUTION and self.episode is not None:
            transition = self._resolve(self._clear_since_s, "condition_cleared")
        self._reset_pending()
        self.state = AlertState.CLEAR
        self._previous_holds = False
        self._cooldown_until_s = float("-inf")
        return transition

    @property
    def active(self) -> bool:
        return self.state in (
            AlertState.PENDING_ALERT,
            AlertState.ALERT,
            AlertState.PENDING_RESOLUTION,
        )

    # ------------------------------------------------------------------ transitions

    def _on_holds(self, time_s: float, sample: EvidenceSample) -> Transition | None:
        if self.state in (AlertState.ALERT, AlertState.PENDING_RESOLUTION):
            assert self.episode is not None
            self.state = AlertState.ALERT
            self.episode.add(sample)
            self._last_holds_s = time_s
            self._previous_holds = True
            return None

        if self.state is AlertState.PENDING_ALERT:
            if self._previous_holds:
                self._accumulated_s += time_s - self._last_holds_s
            elif self._gap_exceeds_grace(time_s):
                self._start_pending(time_s)
            # else: resuming inside the grace window — the gap is paused, not counted.
        else:
            self._start_pending(time_s)

        self._pending.append(sample)
        self._last_holds_s = time_s
        self._previous_holds = True
        if self._accumulated_s >= self.params.debounce_s - _TIME_EPSILON_S:
            return self._confirm(time_s)
        return None

    def _on_clear(self, time_s: float) -> Transition | None:
        self._previous_holds = False
        if self.state is AlertState.PENDING_ALERT:
            self._abandon_pending()
            return None
        if self.state is AlertState.ALERT:
            self.state = AlertState.PENDING_RESOLUTION
            self._clear_since_s = time_s
            return self._maybe_resolve_on_clear(time_s)
        if self.state is AlertState.PENDING_RESOLUTION:
            return self._maybe_resolve_on_clear(time_s)
        return None

    def _on_missing(self, time_s: float) -> Transition | None:
        self._previous_holds = False
        if self.state is AlertState.PENDING_ALERT:
            if self._gap_exceeds_grace(time_s):
                self._abandon_pending()
            return None
        if self.state is AlertState.ALERT:
            assert self.episode is not None
            if _elapsed(self.episode.last_holds_s, time_s, self.params.hysteresis_s):
                return self._resolve(self.episode.last_holds_s, "evidence_lost")
            return None
        if self.state is AlertState.PENDING_RESOLUTION:
            if _elapsed(self._clear_since_s, time_s, self.params.hysteresis_s):
                return self._resolve(self._clear_since_s, "condition_cleared")
        return None

    # ------------------------------------------------------------------ helpers

    def _start_pending(self, time_s: float) -> None:
        self._pending = []
        self._pending_since_s = time_s
        self._accumulated_s = 0.0
        self.state = AlertState.PENDING_ALERT

    def _reset_pending(self) -> None:
        self._pending = []
        self._accumulated_s = 0.0

    def _abandon_pending(self) -> None:
        self._reset_pending()
        self.state = (
            AlertState.COOLDOWN if self._cooldown_until_s > self._last_holds_s else AlertState.CLEAR
        )

    def _confirm(self, time_s: float) -> Transition:
        pending = self._pending
        self._reset_pending()
        self.state = AlertState.ALERT
        previous = self.episodes[-1] if self.episodes else None
        if previous is not None and self._pending_since_s < self._cooldown_until_s:
            previous.resolved_at_s = None
            previous.resolution_reason = None
            previous.reopen_count += 1
            for sample in pending:
                previous.add(sample)
            self.episode = previous
            return Transition(TransitionKind.REOPENED, previous)

        episode = Episode(
            first_seen_s=self._pending_since_s,
            confirmed_at_s=time_s,
            last_holds_s=pending[0].video_time_s,
            peak=pending[0],
        )
        for sample in pending:
            episode.add(sample)
        self.episode = episode
        self.episodes.append(episode)
        return Transition(TransitionKind.OPENED, episode)

    def _maybe_resolve_on_clear(self, time_s: float) -> Transition | None:
        if _elapsed(self._clear_since_s, time_s, self.params.hysteresis_s):
            return self._resolve(self._clear_since_s, "condition_cleared")
        return None

    def _resolve(self, resolved_at_s: float, reason: str) -> Transition:
        assert self.episode is not None
        episode = self.episode
        episode.resolved_at_s = max(resolved_at_s, episode.first_seen_s)
        episode.resolution_reason = reason
        self.episode = None
        self.state = AlertState.COOLDOWN
        self._cooldown_until_s = episode.resolved_at_s + self.params.cooldown_s
        return Transition(TransitionKind.RESOLVED, episode)

    def _leave_expired_cooldown(self, time_s: float) -> None:
        if self.state is AlertState.COOLDOWN and _elapsed(self._cooldown_until_s, time_s, 0.0):
            self.state = AlertState.CLEAR

    def _gap_exceeds_grace(self, time_s: float) -> bool:
        return time_s - self._last_holds_s > self.params.grace_s + _TIME_EPSILON_S


def _elapsed(since_s: float, now_s: float, duration_s: float) -> bool:
    return now_s - since_s >= duration_s - _TIME_EPSILON_S
