"""R1–R5 rule engine: per-frame evaluation, temporal control and incident assembly.

The engine is deterministic and model-free. It consumes `FrameContext`s in time order and
produces:

* a `FrameRuleOutput` per frame (raw results + pending/active alerts, for overlays), and
* at `finalize()`, the run's `IncidentRecord`s and one `RuleCoverageEntry` per rule.

Title, severity, basis, action and references come only from the compiled catalogue.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.rules.base import RuleEvaluator
from pipeline.rules.catalogue import CompiledCatalogue, load_catalogue
from pipeline.rules.r1 import R1MissingHelmet
from pipeline.rules.r2 import R2MissingHiVis
from pipeline.rules.r3 import R3RestrictedZone
from pipeline.rules.r4 import R4MachineryProximity
from pipeline.rules.r5 import R5FallEdge
from pipeline.rules.temporal import (
    Episode,
    EvidenceSample,
    Predicate,
    SubjectTimeline,
    TemporalParams,
    Transition,
    TransitionKind,
    predicate_for,
)
from shared.config import RulesConfig
from shared.enums import HelmetState, RuleId, RuleStatus, VestState
from shared.identity import display_track_label, incident_id
from shared.schemas.frame import ActiveAlert, FrameContext, FrameRuleOutput
from shared.schemas.incidents import IncidentRecord, RuleResult
from shared.schemas.run import RuleCoverageEntry
from shared.schemas.tracks import HelmetRecord, VestRecord

# Coverage precedence when a rule never confirmed an incident.
_COVERAGE_ORDER = (
    RuleStatus.EVALUATED_CLEAR,
    RuleStatus.INCONCLUSIVE,
    RuleStatus.NOT_APPLICABLE,
    RuleStatus.UNSUPPORTED,
)


def build_evaluators(config: RulesConfig, catalogue: CompiledCatalogue) -> list[RuleEvaluator]:
    def basis(rule_id: RuleId) -> str:
        return catalogue.rule(rule_id).basis

    return [
        R1MissingHelmet(basis=basis(RuleId.R1), confidence_floor=config.ppe_confidence_floor),
        R2MissingHiVis(basis=basis(RuleId.R2), confidence_floor=config.ppe_confidence_floor),
        R3RestrictedZone(basis=basis(RuleId.R3)),
        R4MachineryProximity(basis=basis(RuleId.R4)),
        R5FallEdge(basis=basis(RuleId.R5), edge_max_wh=config.r5_edge_max_wh),
    ]


def temporal_params(config: RulesConfig) -> dict[RuleId, TemporalParams]:
    debounce = {
        RuleId.R1: config.r1_debounce_seconds,
        RuleId.R2: config.r2_debounce_seconds,
        RuleId.R3: config.r3_debounce_seconds,
        RuleId.R4: config.r4_debounce_seconds,
        RuleId.R5: config.r5_debounce_seconds,
    }
    return {
        rule_id: TemporalParams(
            debounce_s=seconds,
            grace_s=config.missing_observation_grace_seconds,
            hysteresis_s=config.resolution_hysteresis_seconds,
            cooldown_s=config.cooldown_seconds,
        )
        for rule_id, seconds in debounce.items()
    }


@dataclass
class _Subject:
    rule_id: RuleId
    track_id: str
    shot_id: str
    timeline: SubjectTimeline
    last_seen_s: float


@dataclass
class _IncidentEntry:
    incident_id: str
    rule_id: RuleId
    track_id: str
    shot_id: str
    episode: Episode


@dataclass(frozen=True)
class EngineResult:
    incidents: tuple[IncidentRecord, ...]
    coverage: tuple[RuleCoverageEntry, ...]


class RuleEngine:
    def __init__(
        self,
        *,
        run_id: str,
        config: RulesConfig,
        catalogue: CompiledCatalogue | None = None,
        evaluators: Sequence[RuleEvaluator] | None = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.catalogue = catalogue or load_catalogue()
        self.evaluators = list(evaluators or build_evaluators(config, self.catalogue))
        self._params = temporal_params(config)
        # A subject that stays clear this long is dropped; it will be recreated if seen.
        self._idle_prune_s = max(
            config.missing_observation_grace_seconds,
            config.resolution_hysteresis_seconds,
            config.cooldown_seconds,
        )
        self._subjects: dict[tuple[RuleId, str], _Subject] = {}
        self._incidents: dict[int, _IncidentEntry] = {}
        self._incident_ids: set[str] = set()
        self._observations: dict[RuleId, Counter[tuple[RuleStatus, str]]] = {
            rule_id: Counter() for rule_id in RuleId
        }
        self._shot_id: str | None = None
        self._last_time_s = 0.0
        self._finalized = False

    # ------------------------------------------------------------------ per frame

    def evaluate_frame(self, context: FrameContext) -> FrameRuleOutput:
        if self._finalized:
            raise RuntimeError("the rule engine has already been finalized")
        if context.run_id != self.run_id:
            raise ValueError(f"frame belongs to run {context.run_id}, engine is {self.run_id}")
        opened: list[str] = []
        resolved: list[str] = []
        if self._shot_id is not None and context.shot_id != self._shot_id:
            self._close_subjects(self._last_time_s, "shot_boundary", opened, resolved)
        elif self._shot_id is not None and context.video_time_s < self._last_time_s:
            raise ValueError(
                f"frames must arrive in time order: {context.video_time_s} < {self._last_time_s}"
            )
        self._shot_id = context.shot_id
        self._last_time_s = context.video_time_s
        time_s = context.video_time_s

        results = self._evaluate(context)
        observed: set[tuple[RuleId, str]] = set()
        for result in results:
            self._observations[result.rule_id][(result.status, result.reason_code)] += 1
            if result.track_id is None:
                continue
            key = (result.rule_id, result.track_id)
            observed.add(key)
            subject = self._subjects.get(key)
            if subject is None:
                subject = _Subject(
                    rule_id=result.rule_id,
                    track_id=result.track_id,
                    shot_id=context.shot_id,
                    timeline=SubjectTimeline(self._params[result.rule_id]),
                    last_seen_s=time_s,
                )
                self._subjects[key] = subject
            subject.last_seen_s = time_s
            predicate = predicate_for(result.status)
            sample = self._sample(context, result) if predicate is Predicate.HOLDS else None
            transition = subject.timeline.observe(time_s, predicate, sample)
            self._record(subject, transition, opened, resolved)

        for key, subject in list(self._subjects.items()):
            if key in observed:
                continue
            self._record(subject, subject.timeline.expire(time_s), opened, resolved)
            idle = time_s - subject.last_seen_s
            if not subject.timeline.active and subject.timeline.episode is None:
                if idle >= self._idle_prune_s:
                    del self._subjects[key]

        return FrameRuleOutput(
            frame_index=context.frame_index,
            video_time_s=time_s,
            results=tuple(results),
            active_alerts=self._active_alerts(),
            opened_incident_ids=tuple(opened),
            resolved_incident_ids=tuple(resolved),
        )

    def finalize(self, end_time_s: float | None = None) -> EngineResult:
        if self._finalized:
            raise RuntimeError("the rule engine has already been finalized")
        end = self._last_time_s if end_time_s is None else end_time_s
        self._close_subjects(end, "end_of_clip", [], [])
        self._finalized = True
        entries = sorted(
            self._incidents.values(),
            key=lambda entry: (entry.episode.confirmed_at_s, entry.rule_id, entry.track_id),
        )
        incidents = tuple(self._incident_record(entry) for entry in entries)
        return EngineResult(incidents=incidents, coverage=self._coverage(incidents))

    # ------------------------------------------------------------------ internals

    def _evaluate(self, context: FrameContext) -> list[RuleResult]:
        results: list[RuleResult] = []
        for evaluator in self.evaluators:
            produced = evaluator.evaluate(context)
            if not produced:
                raise ValueError(f"{evaluator.rule_id.value} returned no result")
            seen: set[str | None] = set()
            for result in produced:
                if result.rule_id is not evaluator.rule_id:
                    raise ValueError(
                        f"{evaluator.rule_id.value} returned a {result.rule_id.value} result"
                    )
                if result.track_id in seen:
                    raise ValueError(
                        f"{evaluator.rule_id.value} returned two results for {result.track_id}"
                    )
                seen.add(result.track_id)
            if None in seen and len(produced) > 1:
                raise ValueError(
                    f"{evaluator.rule_id.value} mixed frame-level and per-worker results"
                )
            results.extend(produced)
        return results

    def _sample(self, context: FrameContext, result: RuleResult) -> EvidenceSample:
        track = next(
            (t for t in context.tracks if t.canonical_track_id == result.track_id),
            None,
        )
        tags: set[str] = set()
        zone_label: str | None = None
        for proposal_id in result.proposal_ids:
            proposal = context.proposal(proposal_id)
            if proposal is None:
                continue
            if proposal.road_work:
                tags.add("road_work")
            if zone_label is None:
                zone_label = proposal.label or proposal.proposal_id
        object_label: str | None = None
        if result.object_id is not None:
            machine = next(
                (m for m in context.machinery if m.machinery_id == result.object_id),
                None,
            )
            local = display_track_label(result.object_id)
            object_label = f"{machine.class_name} {local}" if machine else f"machine {local}"
        if result.distance is not None:
            score = -result.distance.median_wh  # closest approach is the best evidence
        else:
            score = result.confidence if result.confidence is not None else 0.0
        return EvidenceSample(
            frame_index=context.frame_index,
            video_time_s=context.video_time_s,
            score=score,
            result=result,
            helmet=track.helmet if track else None,
            vest=track.vest if track else None,
            context_tags=frozenset(tags),
            zone_label=zone_label,
            object_label=object_label,
        )

    def _record(
        self,
        subject: _Subject,
        transition: Transition | None,
        opened: list[str],
        resolved: list[str],
    ) -> None:
        if transition is None:
            return
        key = id(transition.episode)
        if transition.kind is TransitionKind.OPENED:
            new_id = incident_id(
                self.run_id, subject.track_id, subject.rule_id, transition.episode.first_seen_s
            )
            suffix = 2
            candidate = new_id
            while candidate in self._incident_ids:
                candidate, suffix = f"{new_id}-{suffix}", suffix + 1
            self._incident_ids.add(candidate)
            self._incidents[key] = _IncidentEntry(
                incident_id=candidate,
                rule_id=subject.rule_id,
                track_id=subject.track_id,
                shot_id=subject.shot_id,
                episode=transition.episode,
            )
            opened.append(candidate)
        elif transition.kind is TransitionKind.REOPENED:
            opened.append(self._incidents[key].incident_id)
        else:
            resolved.append(self._incidents[key].incident_id)

    def _close_subjects(
        self, time_s: float, reason: str, opened: list[str], resolved: list[str]
    ) -> None:
        for subject in self._subjects.values():
            self._record(subject, subject.timeline.close(time_s, reason), opened, resolved)
        self._subjects.clear()

    def _active_alerts(self) -> tuple[ActiveAlert, ...]:
        alerts: list[ActiveAlert] = []
        for subject in self._subjects.values():
            timeline = subject.timeline
            if not timeline.active:
                continue
            entry = self._incidents.get(id(timeline.episode)) if timeline.episode else None
            alerts.append(
                ActiveAlert(
                    rule_id=subject.rule_id,
                    track_id=subject.track_id,
                    state=timeline.state,
                    incident_id=entry.incident_id if entry else None,
                )
            )
        alerts.sort(key=lambda alert: (alert.rule_id, alert.track_id))
        return tuple(alerts)

    def _incident_record(self, entry: _IncidentEntry) -> IncidentRecord:
        rule = self.catalogue.rule(entry.rule_id)
        episode = entry.episode
        peak = episode.peak
        values = {
            "track_id": display_track_label(entry.track_id),
            "duration_s": f"{episode.duration_s():.1f}",
            "zone_label": peak.zone_label or "unlabelled zone",
            "machine_label": peak.object_label or "a machine",
        }
        references = rule.references + tuple(
            reference
            for tag in sorted(episode.context_tags)
            for reference in rule.contextual_references.get(tag, ())
        )
        return IncidentRecord(
            incident_id=entry.incident_id,
            run_id=self.run_id,
            shot_id=entry.shot_id,
            catalogue_sha256=self.catalogue.sha256,
            rule_id=entry.rule_id,
            status=RuleStatus.EVALUATED_ALERT,
            reason_code=rule.alert_reason_code,
            basis=rule.basis,
            severity=rule.severity.value,
            canonical_track_id=entry.track_id,
            object_id=peak.result.object_id,
            proposal_ids=peak.result.proposal_ids,
            first_seen_s=episode.first_seen_s,
            confirmed_at_s=episode.confirmed_at_s,
            resolved_at_s=episode.resolved_at_s,
            confidence=episode.mean_confidence,
            distance=peak.result.distance,
            relative_band=peak.result.relative_band,
            title=rule.title,
            observation_text=rule.render_observation(values),
            action_code=rule.action_code,
            action_text=rule.render_action(values),
            references=references,
            evidence_path=f"evidence/{entry.incident_id}.jpg",
            source_frame_index=peak.frame_index,
            evidence_crop_path=f"evidence/{entry.incident_id}_crop.jpg",
            helmet=peak.helmet,
            vest=peak.vest,
            adjudication_candidate=self._needs_adjudication(peak.helmet, peak.vest),
            resolution_reason=episode.resolution_reason,
        )

    def _needs_adjudication(self, helmet: HelmetRecord | None, vest: VestRecord | None) -> bool:
        threshold = self.config.ppe_confidence_floor + self.config.adjudication_margin
        for record in (helmet, vest):
            if record is None or not record.visible:
                return True
            if record.state in (HelmetState.UNKNOWN, VestState.UNKNOWN):
                return True
            if record.confidence < threshold:
                return True
        return False

    def _coverage(self, incidents: tuple[IncidentRecord, ...]) -> tuple[RuleCoverageEntry, ...]:
        entries: list[RuleCoverageEntry] = []
        for rule_id in RuleId:
            observations = self._observations[rule_id]
            by_status: Counter[RuleStatus] = Counter()
            for (status, _), count in observations.items():
                by_status[status] += count
            incident_count = sum(1 for incident in incidents if incident.rule_id is rule_id)
            status, reason = self._summary(rule_id, observations, by_status, incident_count)
            entries.append(
                RuleCoverageEntry(
                    rule_id=rule_id,
                    status=status,
                    reason_code=reason,
                    status_counts={s: by_status[s] for s in RuleStatus if by_status[s]},
                    incident_count=incident_count,
                )
            )
        return tuple(entries)

    def _summary(
        self,
        rule_id: RuleId,
        observations: Counter[tuple[RuleStatus, str]],
        by_status: Counter[RuleStatus],
        incident_count: int,
    ) -> tuple[RuleStatus, str]:
        if incident_count:
            return RuleStatus.EVALUATED_ALERT, self.catalogue.rule(rule_id).alert_reason_code
        if by_status[RuleStatus.EVALUATED_ALERT] and not by_status[RuleStatus.EVALUATED_CLEAR]:
            return RuleStatus.EVALUATED_CLEAR, "alert_condition_not_persistent"
        for status in _COVERAGE_ORDER:
            if by_status[status]:
                return status, _most_common_reason(observations, status)
        return RuleStatus.UNSUPPORTED, "no_frames_evaluated"


def _most_common_reason(observations: Counter[tuple[RuleStatus, str]], status: RuleStatus) -> str:
    reasons = [(count, reason) for (s, reason), count in observations.items() if s is status]
    # Highest count first; alphabetical among ties so the summary is deterministic.
    reasons.sort(key=lambda item: (-item[0], item[1]))
    return reasons[0][1]
