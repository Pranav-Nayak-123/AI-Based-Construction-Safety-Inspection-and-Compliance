"""Condense per-frame pass-two data into per-track timelines for the run bundle.

Per-frame records are too large to publish; the cloud agent (L3) answers questions such as
"was worker 17 near machinery at any other point?" from these merged spans instead.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from shared.enums import PERSON_CLASS, DistanceBand
from shared.identity import display_track_label
from shared.schemas.bundle import (
    ProximitySpan,
    Span,
    StateSpan,
    TracksSummaryFile,
    TrackSummary,
    ZoneSpan,
)
from shared.schemas.frame import FrameContext
from shared.schemas.incidents import IncidentRecord
from shared.schemas.tracks import HelmetRecord, VestRecord

UNOBSERVED = "unobserved"
CONFIDENCE_DECIMALS = 4
REPORTED_BANDS = frozenset({DistanceBand.NEAR, DistanceBand.CAUTION})


@dataclass
class _Run:
    start_s: float
    end_s: float
    value: Hashable
    confidence_sum: float = 0.0
    confidence_count: int = 0


class SpanMerger:
    """Merge (time, value) samples into spans; a value change or a long gap splits."""

    def __init__(self, gap_tolerance_s: float) -> None:
        self.gap_tolerance_s = gap_tolerance_s
        self.runs: list[_Run] = []

    def add(self, time_s: float, value: Hashable, confidence: float | None = None) -> None:
        last = self.runs[-1] if self.runs else None
        if last is not None and last.value == value and time_s - last.end_s <= self.gap_tolerance_s:
            last.end_s = time_s
        else:
            last = _Run(start_s=time_s, end_s=time_s, value=value)
            self.runs.append(last)
        if confidence is not None:
            last.confidence_sum += confidence
            last.confidence_count += 1


@dataclass
class _TrackEntry:
    track_id: str
    object_class: str
    shot_id: str
    gap_s: float
    count: int = 0
    first_s: float = float("inf")
    last_s: float = float("-inf")
    presence: SpanMerger = field(init=False)
    helmet: SpanMerger = field(init=False)
    vest: SpanMerger = field(init=False)
    zones: dict[str, SpanMerger] = field(default_factory=dict)
    proximity: dict[str, SpanMerger] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.presence = SpanMerger(self.gap_s)
        self.helmet = SpanMerger(self.gap_s)
        self.vest = SpanMerger(self.gap_s)

    def seen(self, time_s: float) -> None:
        self.count += 1
        self.first_s = min(self.first_s, time_s)
        self.last_s = max(self.last_s, time_s)
        self.presence.add(time_s, True)


class TrackSummaryBuilder:
    def __init__(self, run_id: str, *, gap_tolerance_s: float = 0.5) -> None:
        self.run_id = run_id
        self.gap_tolerance_s = gap_tolerance_s
        self._entries: dict[str, _TrackEntry] = {}

    def add_frame(self, context: FrameContext) -> None:
        time_s = context.video_time_s
        for track in context.tracks:
            entry = self._entry(track.canonical_track_id, track.object_class, context.shot_id)
            entry.seen(time_s)
            if track.object_class == PERSON_CLASS:
                _add_ppe(entry.helmet, time_s, track.helmet)
                _add_ppe(entry.vest, time_s, track.vest)
        for machine in context.machinery:
            self._entry(machine.machinery_id, machine.class_name, context.shot_id).seen(time_s)
        for membership in context.zone_memberships:
            if membership.inside is True and not membership.boundary_uncertain:
                entry = self._entries.get(membership.worker_track_id)
                if entry is not None:
                    merger = entry.zones.setdefault(
                        membership.proposal_id, SpanMerger(self.gap_tolerance_s)
                    )
                    merger.add(time_s, membership.proposal_id)
        for pair in context.machinery_proximities:
            if pair.compatible_plane and pair.band in REPORTED_BANDS:
                entry = self._entries.get(pair.worker_track_id)
                if entry is not None:
                    merger = entry.proximity.setdefault(
                        pair.machinery_id, SpanMerger(self.gap_tolerance_s)
                    )
                    merger.add(time_s, pair.band)

    def build(self, incidents: Iterable[IncidentRecord] = ()) -> TracksSummaryFile:
        incidents_by_track: dict[str, list[str]] = {}
        for incident in incidents:
            if incident.canonical_track_id is not None:
                incidents_by_track.setdefault(incident.canonical_track_id, []).append(
                    incident.incident_id
                )
        entries = sorted(self._entries.values(), key=lambda e: (e.first_s, e.track_id))
        return TracksSummaryFile(
            run_id=self.run_id,
            span_gap_tolerance_s=self.gap_tolerance_s,
            tracks=tuple(
                self._summary(entry, incidents_by_track.get(entry.track_id, []))
                for entry in entries
            ),
        )

    def _entry(self, track_id: str, object_class: str, shot_id: str) -> _TrackEntry:
        entry = self._entries.get(track_id)
        if entry is None:
            entry = _TrackEntry(track_id, object_class, shot_id, self.gap_tolerance_s)
            self._entries[track_id] = entry
        return entry

    @staticmethod
    def _summary(entry: _TrackEntry, incident_ids: list[str]) -> TrackSummary:
        zones = sorted(
            (
                ZoneSpan(start_s=run.start_s, end_s=run.end_s, proposal_id=proposal_id)
                for proposal_id, merger in entry.zones.items()
                for run in merger.runs
            ),
            key=lambda span: (span.start_s, span.proposal_id),
        )
        proximity = sorted(
            (
                ProximitySpan(
                    start_s=run.start_s,
                    end_s=run.end_s,
                    machinery_id=machinery_id,
                    band=DistanceBand(str(run.value)),
                )
                for machinery_id, merger in entry.proximity.items()
                for run in merger.runs
            ),
            key=lambda span: (span.start_s, span.machinery_id),
        )
        return TrackSummary(
            canonical_track_id=entry.track_id,
            display_label=display_track_label(entry.track_id),
            object_class=entry.object_class,
            shot_id=entry.shot_id,
            first_seen_s=entry.first_s,
            last_seen_s=entry.last_s,
            observation_count=entry.count,
            presence=tuple(Span(start_s=r.start_s, end_s=r.end_s) for r in entry.presence.runs),
            helmet=_state_spans(entry.helmet),
            vest=_state_spans(entry.vest),
            zones=tuple(zones),
            machinery_proximity=tuple(proximity),
            incident_ids=tuple(incident_ids),
        )


def _add_ppe(merger: SpanMerger, time_s: float, record: HelmetRecord | VestRecord | None) -> None:
    if record is None:
        merger.add(time_s, UNOBSERVED)
    else:
        merger.add(time_s, record.state.value, record.confidence)


def _state_spans(merger: SpanMerger) -> tuple[StateSpan, ...]:
    return tuple(
        StateSpan(
            start_s=run.start_s,
            end_s=run.end_s,
            state=str(run.value),
            mean_confidence=(
                round(run.confidence_sum / run.confidence_count, CONFIDENCE_DECIMALS)
                if run.confidence_count
                else None
            ),
        )
        for run in merger.runs
    )
