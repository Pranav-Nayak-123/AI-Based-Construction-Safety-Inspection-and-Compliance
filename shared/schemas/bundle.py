"""Run bundle: the contract between the offline pipeline and the cloud service.

One processed clip produces one directory. `POST /api/runs/publish` ingests it, the web UI
renders it, L1 adjudicates its evidence crops and the L3 agent's tools query it.

    output/<run_id>/
      run_manifest.json      RunManifest — provenance, models, R1–R5 coverage, timings
      incidents.json         IncidentsFile — confirmed incidents, catalogue-driven text
      tracks_summary.json    TracksSummaryFile — per-worker timelines ("track a worker")
      scene.json             SceneCache — zones and edges used ("look up a zone")
      evidence/<id>.jpg      full frame at the evidence moment, subject highlighted
      evidence/<id>_crop.jpg original-resolution person crop (L1 input)
      safety_twin.mp4        1920x1080 H.264 side-by-side
      safety_twin_720p.mp4   web proxy for the 720p player
      report.html            offline human-readable report

Every time is video seconds from the start of the clip. No field carries metres.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from shared.coordinates import StrictModel
from shared.enums import DistanceBand
from shared.schemas.incidents import IncidentRecord

BUNDLE_SCHEMA_VERSION = 1

RUN_MANIFEST_FILE = "run_manifest.json"
INCIDENTS_FILE = "incidents.json"
TRACKS_SUMMARY_FILE = "tracks_summary.json"
SCENE_FILE = "scene.json"
VIDEO_FILE = "safety_twin.mp4"
VIDEO_PROXY_FILE = "safety_twin_720p.mp4"
REPORT_FILE = "report.html"
EVIDENCE_DIR = "evidence"

REQUIRED_FILES = (RUN_MANIFEST_FILE, INCIDENTS_FILE, TRACKS_SUMMARY_FILE, SCENE_FILE)


class IncidentsFile(StrictModel):
    schema_version: int = BUNDLE_SCHEMA_VERSION
    run_id: str
    catalogue_sha256: str
    incidents: tuple[IncidentRecord, ...]

    @model_validator(mode="after")
    def _consistent(self) -> IncidentsFile:
        ids = [incident.incident_id for incident in self.incidents]
        if len(ids) != len(set(ids)):
            raise ValueError("incident ids must be unique")
        for incident in self.incidents:
            if incident.run_id != self.run_id:
                raise ValueError(f"incident {incident.incident_id} belongs to another run")
            if incident.catalogue_sha256 != self.catalogue_sha256:
                raise ValueError(f"incident {incident.incident_id} used another catalogue")
        return self


class Span(StrictModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Span:
        if self.end_s < self.start_s:
            raise ValueError("span end precedes start")
        return self


class StateSpan(Span):
    """A stretch of time with one smoothed PPE state (`unobserved` when no record)."""

    state: str
    mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ZoneSpan(Span):
    proposal_id: str


class ProximitySpan(Span):
    machinery_id: str
    band: DistanceBand


class TrackSummary(StrictModel):
    canonical_track_id: str
    display_label: str
    object_class: str
    shot_id: str
    first_seen_s: float = Field(ge=0.0)
    last_seen_s: float = Field(ge=0.0)
    observation_count: int = Field(ge=1)
    presence: tuple[Span, ...]
    helmet: tuple[StateSpan, ...] = ()
    vest: tuple[StateSpan, ...] = ()
    zones: tuple[ZoneSpan, ...] = ()
    machinery_proximity: tuple[ProximitySpan, ...] = ()
    incident_ids: tuple[str, ...] = ()


class TracksSummaryFile(StrictModel):
    schema_version: int = BUNDLE_SCHEMA_VERSION
    run_id: str
    span_gap_tolerance_s: float = Field(ge=0.0)
    tracks: tuple[TrackSummary, ...]
