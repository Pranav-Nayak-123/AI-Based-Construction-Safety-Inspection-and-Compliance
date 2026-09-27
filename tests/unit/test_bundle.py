from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from rule_frames import frame, helmet, interval, machine, membership, proximity, tid, worker, zone

from jobs.pipeline_job import FakePipelineJob
from pipeline.output.bundle import read_model, validate_bundle
from pipeline.output.summaries import SpanMerger, TrackSummaryBuilder
from shared.enums import DistanceBand, HelmetState
from shared.schemas.bundle import INCIDENTS_FILE, IncidentsFile, Span, TracksSummaryFile


@pytest.fixture()
def fake_run(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.bin"
    clip.write_bytes(b"synthetic input")
    job = FakePipelineJob(
        clip, output_root=tmp_path / "output", run_id="run-bundle", skip_video=True
    )
    job.run()
    return job.run_dir


def test_fake_pipeline_emits_a_valid_bundle(fake_run: Path) -> None:
    check = validate_bundle(fake_run)
    assert check.ok, check.problems
    incidents = read_model(fake_run / INCIDENTS_FILE, IncidentsFile)
    assert len(incidents.incidents) == 1
    summary = read_model(fake_run / "tracks_summary.json", TracksSummaryFile)
    assert summary.tracks[0].incident_ids == (incidents.incidents[0].incident_id,)


def test_missing_evidence_fails_validation(fake_run: Path) -> None:
    for evidence in (fake_run / "evidence").iterdir():
        evidence.unlink()
    check = validate_bundle(fake_run)
    assert not check.ok
    assert any("missing evidence/" in problem for problem in check.problems)


def test_missing_media_only_matters_when_required(fake_run: Path) -> None:
    assert validate_bundle(fake_run).ok
    assert "missing safety_twin_720p.mp4" in validate_bundle(fake_run, require_media=True).problems


def test_incidents_file_rejects_foreign_incidents(fake_run: Path) -> None:
    incidents = read_model(fake_run / INCIDENTS_FILE, IncidentsFile)
    with pytest.raises(ValidationError, match="another run"):
        IncidentsFile(
            run_id="some-other-run",
            catalogue_sha256=incidents.catalogue_sha256,
            incidents=incidents.incidents,
        )
    with pytest.raises(ValidationError, match="unique"):
        IncidentsFile(
            run_id=incidents.run_id,
            catalogue_sha256=incidents.catalogue_sha256,
            incidents=incidents.incidents * 2,
        )


def test_span_merger_splits_on_value_change_and_long_gaps() -> None:
    merger = SpanMerger(gap_tolerance_s=0.5)
    for time_s, value in [(0.0, "a"), (0.2, "a"), (0.6, "a"), (1.5, "a"), (1.6, "b")]:
        merger.add(time_s, value)
    assert [(run.start_s, run.end_s, run.value) for run in merger.runs] == [
        (0.0, 0.6, "a"),
        (1.5, 1.5, "a"),
        (1.6, 1.6, "b"),
    ]


def test_track_summary_condenses_frames_into_spans() -> None:
    builder = TrackSummaryBuilder("run-test", gap_tolerance_s=0.5)
    fence = zone("zone-1")
    for index in range(30):
        time_s = index / 10
        near = 1.0 <= time_s < 2.0
        builder.add_frame(
            frame(
                time_s,
                tracks=(worker(17, helmet_record=helmet(HelmetState.HELMET, 0.8)),),
                machinery=(machine("shot:s0/track:4"),),
                proposals=(fence,),
                memberships=(membership(17, "zone-1", time_s < 0.5),),
                proximities=(
                    proximity(
                        17,
                        "shot:s0/track:4",
                        DistanceBand.NEAR if near else DistanceBand.CLEAR,
                        distance=interval(0.5, 1.0) if near else interval(4.0, 5.0),
                    ),
                ),
            )
        )
    summary = builder.build()
    person, excavator = summary.tracks
    assert person.canonical_track_id == tid(17)
    assert person.display_label == "17"
    assert person.observation_count == 30
    assert person.presence == (Span(start_s=0.0, end_s=2.9),)
    assert [(s.state, s.mean_confidence) for s in person.helmet] == [("helmet", 0.8)]
    assert [(z.proposal_id, z.start_s, z.end_s) for z in person.zones] == [("zone-1", 0.0, 0.4)]
    [near_span] = person.machinery_proximity
    assert (near_span.machinery_id, near_span.band) == ("shot:s0/track:4", DistanceBand.NEAR)
    assert (near_span.start_s, near_span.end_s) == (1.0, 1.9)
    assert excavator.object_class == "excavator"


def test_span_end_cannot_precede_start() -> None:
    with pytest.raises(ValidationError, match="precedes"):
        Span(start_s=2.0, end_s=1.0)
