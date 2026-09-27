from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pipeline.mapping.anchors import image_aligned_record
from pipeline.output.bundle import write_model
from pipeline.output.summaries import TrackSummaryBuilder
from pipeline.rules.catalogue import load_catalogue
from pipeline.rules.engine import EngineResult, RuleEngine
from shared.config import AppConfig, load_config
from shared.coordinates import Box2, Point2
from shared.enums import (
    CoordinateSpace,
    HelmetState,
    JobStage,
    ProcessingStatus,
    VestState,
)
from shared.identity import canonical_track_id
from shared.schemas.bundle import (
    INCIDENTS_FILE,
    RUN_MANIFEST_FILE,
    SCENE_FILE,
    TRACKS_SUMMARY_FILE,
    IncidentsFile,
)
from shared.schemas.frame import FeedCapabilities, FrameContext
from shared.schemas.jobs import JobRecord
from shared.schemas.run import InputVideoMeta, PassTiming, RunManifest
from shared.schemas.scene import SceneCache
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation, VestRecord


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> InputVideoMeta:
    """Best-effort ffprobe; falls back to path + hash only."""
    sha = _sha256_file(path)
    meta = InputVideoMeta(path=str(path.resolve()), sha256=sha)
    if shutil.which("ffprobe") is None:
        return meta
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return meta
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    width = height = fps = duration = None
    if streams:
        stream = streams[0]
        width = int(stream["width"]) if stream.get("width") is not None else None
        height = int(stream["height"]) if stream.get("height") is not None else None
        rate = stream.get("r_frame_rate")
        if isinstance(rate, str) and "/" in rate:
            num_s, den_s = rate.split("/", 1)
            num, den = float(num_s), float(den_s)
            if den:
                fps = num / den
    fmt = payload.get("format") or {}
    if fmt.get("duration") is not None:
        duration = float(fmt["duration"])
    return InputVideoMeta(
        path=str(path.resolve()),
        sha256=sha,
        width=width,
        height=height,
        duration_s=duration,
        fps=fps,
    )


def _fake_tracks(shot_id: str, duration_s: float) -> list[Pass1TrackObservation]:
    observations: list[Pass1TrackObservation] = []
    frame_count = max(1, int(duration_s * 15))
    for frame_index in range(min(frame_count, 45)):
        t = frame_index / 15.0
        observations.append(
            Pass1TrackObservation(
                shot_id=shot_id,
                frame_index=frame_index,
                video_time_s=t,
                track_id=1,
                canonical_track_id=canonical_track_id(shot_id, 1),
                object_class="person",
                box_reference=Box2(
                    x1=100.0 + frame_index,
                    y1=200.0,
                    x2=180.0 + frame_index,
                    y2=360.0,
                    space=CoordinateSpace.REFERENCE,
                ),
                anchor_reference=Point2(
                    x=140.0 + frame_index,
                    y=360.0,
                    space=CoordinateSpace.REFERENCE,
                ),
                helmet=HelmetRecord(
                    state=HelmetState.NO_HELMET,
                    confidence=0.84,
                    visible=True,
                ),
                vest=VestRecord(
                    state=VestState.VEST,
                    confidence=0.91,
                    visible=True,
                ),
            )
        )
    return observations


def synthetic_frame_contexts(
    tracks: list[Pass1TrackObservation], *, run_id: str
) -> list[FrameContext]:
    """Frame contexts for synthetic pass-one tracks.

    The fake feed has PPE states but no stabilisation, scene state or relative plane, so
    R1/R2 evaluate while R3–R5 honestly report `unsupported_for_feed`.
    """
    capabilities = FeedCapabilities(
        person_detection=True,
        ppe_classification=True,
        scene_state=False,
        relative_plane=False,
    )
    by_frame: dict[tuple[str, int], list[Pass1TrackObservation]] = {}
    for observation in tracks:
        by_frame.setdefault((observation.shot_id, observation.frame_index), []).append(observation)
    return [
        FrameContext(
            run_id=run_id,
            shot_id=shot_id,
            frame_index=frame_index,
            video_time_s=observations[0].video_time_s,
            geometry_supported=False,
            capabilities=capabilities,
            tracks=tuple(image_aligned_record(obs) for obs in observations),
        )
        for (shot_id, frame_index), observations in sorted(
            by_frame.items(), key=lambda item: item[1][0].video_time_s
        )
    ]


def run_rules_on_synthetic_tracks(
    tracks: list[Pass1TrackObservation],
    *,
    run_id: str,
    config: AppConfig,
    summary: TrackSummaryBuilder | None = None,
) -> EngineResult:
    """Feed synthetic pass-one tracks through the real R1–R5 engine."""
    engine = RuleEngine(run_id=run_id, config=config.rules)
    for context in synthetic_frame_contexts(tracks, run_id=run_id):
        engine.evaluate_frame(context)
        if summary is not None:
            summary.add_frame(context)
    return engine.finalize()


def render_placeholder_mp4(
    *,
    output_path: Path,
    duration_s: float,
    width: int,
    height: int,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render the placeholder side-by-side video")
    pane_w = width // 2
    pane_h = height
    duration_s = max(1.0, min(duration_s, 10.0))
    filter_complex = (
        f"color=c=0x1a1a1a:s={pane_w}x{pane_h}:d={duration_s:.3f}[left];"
        f"color=c=0x2f3e2f:s={pane_w}x{pane_h}:d={duration_s:.3f}[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-r",
        "15",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        f"{duration_s:.3f}",
        str(output_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip() or completed.stdout}")


class FakePipelineJob:
    """Week-1 placeholder job: schemas + fake tracks + side-by-side stub video."""

    def __init__(
        self,
        input_path: Path,
        *,
        output_root: Path | None = None,
        config: AppConfig | None = None,
        run_id: str | None = None,
        skip_video: bool = False,
    ) -> None:
        self.input_path = input_path.resolve()
        self.config = config or load_config()
        self.output_root = (output_root or Path("output")).resolve()
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.skip_video = skip_video
        self.shot_id = "shot-0"
        self.run_dir = self.output_root / self.run_id

    def run(self) -> RunManifest:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"input video not found: {self.input_path}")

        started = _utc_now()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        evidence_dir = self.run_dir / "evidence"
        evidence_dir.mkdir()

        job = JobRecord(
            job_id=f"job-{self.run_id}",
            run_id=self.run_id,
            status=ProcessingStatus.RUNNING,
            stage=JobStage.INTAKE,
            progress=0.05,
            message="Probing input",
        )
        self._write_json(self.run_dir / "job.json", job)

        input_meta = probe_video(self.input_path)
        duration_s = input_meta.duration_s or 2.0
        out_w = self.config.render.output_width
        out_h = self.config.render.output_height

        job = job.model_copy(
            update={
                "stage": JobStage.DETECTION,
                "progress": 0.35,
                "message": "Writing fake tracks",
            }
        )
        self._write_json(self.run_dir / "job.json", job)

        tracks = _fake_tracks(self.shot_id, duration_s)
        tracks_path = self.run_dir / "tracks.jsonl"
        with tracks_path.open("w", encoding="utf-8") as handle:
            for obs in tracks:
                handle.write(obs.model_dump_json())
                handle.write("\n")

        job = job.model_copy(
            update={
                "stage": JobStage.MAPPING_RULES,
                "progress": 0.55,
                "message": "Evaluating placeholder R1–R5",
            }
        )
        self._write_json(self.run_dir / "job.json", job)

        summary = TrackSummaryBuilder(self.run_id)
        rules = run_rules_on_synthetic_tracks(
            tracks, run_id=self.run_id, config=self.config, summary=summary
        )
        coverage, incidents = rules.coverage, rules.incidents
        write_model(
            self.run_dir / INCIDENTS_FILE,
            IncidentsFile(
                run_id=self.run_id,
                catalogue_sha256=load_catalogue().sha256,
                incidents=incidents,
            ),
        )
        write_model(self.run_dir / TRACKS_SUMMARY_FILE, summary.build(incidents))
        write_model(
            self.run_dir / SCENE_FILE,
            SceneCache(
                cache_key="none",
                video_sha256=input_meta.sha256,
                shot_proposals={},
                warnings=("Week-1 fake pipeline: no scene state; R3/R5 are unsupported.",),
            ),
        )
        for incident in incidents:
            for relative in (incident.evidence_path, incident.evidence_crop_path):
                if relative is None:
                    continue
                evidence_file = self.run_dir / relative
                evidence_file.parent.mkdir(parents=True, exist_ok=True)
                evidence_file.write_text(
                    f"placeholder evidence for synthetic {incident.rule_id.value} alert\n",
                    encoding="utf-8",
                )

        job = job.model_copy(
            update={
                "stage": JobStage.RENDER,
                "progress": 0.8,
                "message": "Rendering placeholder side-by-side video",
            }
        )
        self._write_json(self.run_dir / "job.json", job)

        video_path = self.run_dir / "safety_twin.mp4"
        artifacts: dict[str, str] = {
            "incidents": INCIDENTS_FILE,
            "tracks_summary": TRACKS_SUMMARY_FILE,
            "scene": SCENE_FILE,
            "tracks": "tracks.jsonl",
            "run_manifest": RUN_MANIFEST_FILE,
        }
        if self.skip_video:
            video_path.write_bytes(b"")
            artifacts["safety_twin"] = "safety_twin.mp4"
        else:
            render_placeholder_mp4(
                output_path=video_path,
                duration_s=duration_s,
                width=out_w,
                height=out_h,
            )
            artifacts["safety_twin"] = "safety_twin.mp4"
            artifacts["safety_twin_sha256"] = _sha256_file(video_path)

        finished = _utc_now()
        timing = PassTiming(
            stage=JobStage.PUBLISH,
            started_at=_iso(started),
            finished_at=_iso(finished),
            duration_s=(finished - started).total_seconds(),
        )
        manifest = RunManifest(
            run_id=self.run_id,
            status=ProcessingStatus.COMPLETED,
            created_at=_iso(started),
            completed_at=_iso(finished),
            input_video=input_meta,
            model_ids={"stage1": "fake", "ppe": "fake"},
            dependency_versions={"pipeline": "fake-0.1.0"},
            gemini_status="unused",
            scene_cache_status="none",
            rule_coverage=coverage,
            warnings=(
                "Week-1 fake pipeline: detections and twin geometry are synthetic placeholders.",
            ),
            pass_timings=(timing,),
            output_artifacts=artifacts,
        )
        self._write_json(self.run_dir / "run_manifest.json", manifest)

        report_path = self.run_dir / "report.html"
        report_path.write_text(self._report_html(manifest, incidents), encoding="utf-8")
        artifacts = {**artifacts, "report": "report.html"}
        manifest = manifest.model_copy(update={"output_artifacts": artifacts})
        self._write_json(self.run_dir / "run_manifest.json", manifest)

        job = JobRecord(
            job_id=f"job-{self.run_id}",
            run_id=self.run_id,
            status=ProcessingStatus.COMPLETED,
            stage=JobStage.PUBLISH,
            progress=1.0,
            message="Completed",
            cancellable=False,
        )
        self._write_json(self.run_dir / "job.json", job)
        return manifest

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        if hasattr(payload, "model_dump"):
            data = payload.model_dump(mode="json")  # type: ignore[union-attr]
        else:
            data = payload
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _report_html(manifest: RunManifest, incidents: tuple) -> str:
        cards = "".join(
            f"<li><strong>{i.rule_id.value}</strong> — {i.observation_text}</li>" for i in incidents
        )
        coverage = "".join(
            f"<li>{e.rule_id.value}: {e.status.value} ({e.reason_code})</li>"
            for e in manifest.rule_coverage
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Run {manifest.run_id}</title></head>
<body>
  <h1>Construction Safety Twin — run report</h1>
  <p><em>{manifest.disclaimer}</em></p>
  <p>Status: {manifest.status.value} · mode: {manifest.mode}</p>
  <h2>Incidents</h2>
  <ul>{cards}</ul>
  <h2>Rule coverage</h2>
  <ul>{coverage}</ul>
</body>
</html>
"""
