"""End-to-end offline processing of one clip into a run bundle (FR-1).

    probe → drift gate → pass 1 (detect, track) → geometry gate → scene
          → pass 2 (rules) → pass 3 (render, evidence) → bundle → validate

Each stage is timed into the manifest, and every capability the run lacks (fine-tuned
detector, PPE model, site config, relative plane) is written as a warning so a reviewer
can see why a rule was unsupported or inconclusive.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from jobs.pass2 import Pass2Result, run_pass2
from jobs.render_stage import RenderResult, render_run
from pipeline.cache.pass1_store import Pass1Store
from pipeline.intake.frames import FrameSource, sample_raw_frames
from pipeline.intake.geometry_gate import CameraMode, GeometryGate, classify_drift, gate_geometry
from pipeline.intake.probe import VideoMetadata, probe_video
from pipeline.intake.stabilization import DriftProbe, probe_drift
from pipeline.mapping.frame_context import FrameContextBuilder
from pipeline.output.bundle import validate_bundle, write_model
from pipeline.output.summaries import TrackSummaryBuilder
from pipeline.report.html import write_report
from pipeline.rules.catalogue import load_catalogue
from pipeline.rules.engine import RuleEngine
from pipeline.scene.site_config import (
    load_reference_image,
    load_site_config,
    scene_from_site_config,
    view_matches,
)
from pipeline.vision.detector import Stage1Detector, resolve_weights
from pipeline.vision.pass1 import Pass1Summary, VisionPass
from pipeline.vision.tracker import ByteTrackAdapter
from shared.config import AppConfig, load_config
from shared.enums import JobStage, ProcessingStatus
from shared.errors import PipelineError
from shared.paths import run_work_dir
from shared.schemas.bundle import (
    INCIDENTS_FILE,
    REPORT_FILE,
    RUN_MANIFEST_FILE,
    SCENE_FILE,
    TRACKS_SUMMARY_FILE,
    VIDEO_FILE,
    VIDEO_PROXY_FILE,
    IncidentsFile,
)
from shared.schemas.run import InputVideoMeta, PassTiming, RunManifest
from shared.schemas.scene import SceneCache

DRIFT_SAMPLES = 24
Reporter = Callable[[str], None]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "ultralytics", "opencv-python", "numpy", "shapely", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@dataclass
class _Timer:
    timings: list[PassTiming]

    def stage(self, stage: JobStage) -> _StageClock:
        return _StageClock(self, stage)


class _StageClock:
    def __init__(self, timer: _Timer, stage: JobStage) -> None:
        self.timer = timer
        self.stage = stage

    def __enter__(self) -> _StageClock:
        self.started_at = _now()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.timer.timings.append(
            PassTiming(
                stage=self.stage,
                started_at=self.started_at,
                finished_at=_now(),
                duration_s=time.perf_counter() - self.started,
            )
        )


class ProcessJob:
    def __init__(
        self,
        video: Path,
        *,
        output_root: Path = Path("output"),
        site_config: Path | None = None,
        run_id: str | None = None,
        config: AppConfig | None = None,
        project_root: Path | None = None,
        report: Reporter | None = None,
    ) -> None:
        self.video = video.resolve()
        self.config = config or load_config()
        self.root = project_root or Path(__file__).resolve().parents[1]
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.run_dir = output_root.resolve() / self.run_id
        self.work_dir = run_work_dir(self.run_id)
        self.site_config = site_config
        self.report = report or (lambda message: None)
        self.warnings: list[str] = []

    def run(self) -> RunManifest:
        if self.run_dir.exists():
            raise PipelineError("RUN_EXISTS", f"{self.run_dir} already exists")
        self.run_dir.mkdir(parents=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        created_at = _now()
        timer = _Timer([])

        with timer.stage(JobStage.INTAKE):
            metadata = probe_video(self.video)
            video_sha = _sha256(self.video)
            drift = probe_drift(sample_raw_frames(self.video, DRIFT_SAMPLES))
            decision = classify_drift(drift, self.config.intake)
            self.report(
                f"intake: {metadata.width}x{metadata.height} {metadata.fps:.2f} fps "
                f"{metadata.duration_s:.1f} s, camera {decision.mode.value}"
            )
            if decision.mode is CameraMode.UNMEASURABLE:
                self.warnings.append("camera drift unmeasurable: geometry rules unsupported")

        source = FrameSource(
            self.video,
            target_fps=self.config.intake.target_fps,
            canvas_size=(self.config.intake.target_width, self.config.intake.target_height),
            expected_frames=metadata.frame_count,
        )

        with timer.stage(JobStage.DETECTION):
            weights, fine_tuned = resolve_weights(self.config.stage1, self.root)
            if not fine_tuned:
                self.warnings.append(
                    f"stage-1 detector is the pretrained fallback ({weights.name}), "
                    "not a construction fine-tune"
                )
            detector = Stage1Detector(self.config.stage1, weights)
            tracker = ByteTrackAdapter(
                self.config.tracker,
                self.config.intake.target_fps,
                sorted(set(detector.classes.values())),
            )
            store = Pass1Store(self.work_dir / "pass1.sqlite")
            expected = max(1, round(metadata.duration_s * self.config.intake.target_fps))
            pass1 = VisionPass(detector, tracker, self.config, camera_mode=decision.mode).run(
                source, store, progress=self._progress("detect", expected)
            )
            store.set_meta("detector_sha256", detector.model_sha256())
            store.mark_complete()
            self.warnings += source.warnings
            self.report(
                f"pass 1: {pass1.frames} frames, {pass1.detections} detections, "
                f"{len(pass1.shots)} shot(s) in {pass1.elapsed_s:.1f} s "
                f"(detector {pass1.detect_s:.1f} s)"
            )

        with timer.stage(JobStage.MAPPING_RULES):
            gate = self._geometry_gate(decision.mode, drift, pass1)
            scene = self._scene(source, store, pass1, video_sha)
            pass2 = self._pass2(store, scene, gate)
            self.report(
                f"pass 2: {len(pass2.rules.incidents)} incident(s); geometry "
                f"{'supported' if gate.supported else 'unsupported'} ({gate.reason_code})"
            )

        with timer.stage(JobStage.RENDER):
            rendered = render_run(
                source=source,
                store=store,
                pass2=pass2,
                scene=scene,
                config=self.config,
                run_dir=self.run_dir,
                title=f"Construction Safety Twin  |  {self.video.name}",
                progress=self._progress("render", pass1.frames),
            )
        store.close()

        with timer.stage(JobStage.PUBLISH):
            manifest = self._publish(
                created_at,
                metadata,
                video_sha,
                detector,
                weights,
                decision.mode,
                drift,
                gate,
                pass1,
                pass2,
                scene,
                rendered,
                timer,
            )
        return manifest

    # ------------------------------------------------------------------ stages

    def _progress(self, label: str, expected: int) -> Callable[..., None]:
        last = [0.0]

        def report(frames: int, *_: object) -> None:
            now = time.perf_counter()
            if now - last[0] >= 5.0 or frames == expected:
                last[0] = now
                self.report(f"{label}: {frames}/{expected} frames")

        return report

    def _geometry_gate(
        self, mode: CameraMode, drift: DriftProbe, pass1: Pass1Summary
    ) -> GeometryGate:
        """Residual camera motion, normalised by person height, decides R3-R5 support."""
        canvas_diagonal = float(
            np.hypot(self.config.intake.target_width, self.config.intake.target_height)
        )
        residual: float | None = None
        if mode is CameraMode.FIXED and drift.peak_displacement_px is not None:
            # The probe measures raw pixels; the gate works on the processing canvas.
            residual = drift.peak_displacement_px * canvas_diagonal / drift.diagonal_px
        elif mode is CameraMode.STABILIZE:
            residual = pass1.stabilization_residual_p95_px
        gate = gate_geometry(
            classify_drift(drift, self.config.intake),
            residual,
            pass1.median_person_height_px,
            canvas_diagonal,
            self.config.intake,
        )
        if not gate.supported:
            self.warnings.append(f"geometry unsupported ({gate.reason_code}): R3-R5 unsupported")
        return gate

    def _scene(
        self, source: FrameSource, store: Pass1Store, pass1: Pass1Summary, video_sha: str
    ) -> SceneCache:
        if self.site_config is None:
            self.warnings.append("no site config: R3/R5 have no zones or edges to evaluate")
            return SceneCache(cache_key="none", video_sha256=video_sha, shot_proposals={})
        config = load_site_config(self.site_config)
        reference = load_reference_image(config, self.site_config)
        first_frames = {}
        for record in store.frames():
            first_frames.setdefault(record.shot_id, record.frame_index)
        wanted = {index: shot for shot, index in first_frames.items()}
        matched: list[str] = []
        for decoded in source:
            shot = wanted.pop(decoded.frame_index, None)
            if shot is not None:
                ok, shift = view_matches(reference, decoded.image)
                if ok:
                    matched.append(shot)
                else:
                    self.warnings.append(
                        f"site config {config.camera_id} does not match shot {shot} "
                        f"(view shift {shift if shift is None else round(shift, 1)} px): "
                        "its zones are not applied"
                    )
            if not wanted:
                break
        return scene_from_site_config(
            config, shot_ids=matched, video_sha256=video_sha, warnings=tuple(self.warnings[-1:])
        )

    def _pass2(self, store: Pass1Store, scene: SceneCache, gate: GeometryGate) -> Pass2Result:
        self.warnings.append("PPE classifier not trained yet: R1/R2 unsupported")
        self.warnings.append("relative plane not fitted yet: R4/R5 distances inconclusive")
        builder = FrameContextBuilder(
            run_id=self.run_id,
            scene=scene,
            geometry_supported=gate.supported,
            ppe_available=False,
            machinery_classes=self.config.stage1.machinery_classes,
        )
        engine = RuleEngine(run_id=self.run_id, config=self.config.rules)
        return run_pass2(store, builder, engine, TrackSummaryBuilder(self.run_id))

    def _publish(
        self,
        created_at: str,
        metadata: VideoMetadata,
        video_sha: str,
        detector: Stage1Detector,
        weights: Path,
        mode: CameraMode,
        drift: DriftProbe,
        gate: GeometryGate,
        pass1: Pass1Summary,
        pass2: Pass2Result,
        scene: SceneCache,
        rendered: RenderResult,
        timer: _Timer,
    ) -> RunManifest:
        catalogue = load_catalogue()
        write_model(
            self.run_dir / INCIDENTS_FILE,
            IncidentsFile(
                run_id=self.run_id,
                catalogue_sha256=catalogue.sha256,
                incidents=pass2.rules.incidents,
            ),
        )
        write_model(self.run_dir / TRACKS_SUMMARY_FILE, pass2.tracks_summary)
        write_model(self.run_dir / SCENE_FILE, scene)
        manifest = RunManifest(
            run_id=self.run_id,
            status=ProcessingStatus.COMPLETED,
            created_at=created_at,
            completed_at=_now(),
            mode="pipeline",
            input_video=InputVideoMeta(
                path=str(self.video),
                sha256=video_sha,
                width=metadata.width,
                height=metadata.height,
                duration_s=metadata.duration_s,
                fps=metadata.fps,
            ),
            model_ids={"stage1": f"{weights.name}:{detector.model_sha256()[:16]}"},
            dependency_versions=_versions(),
            scene_cache_status=scene.cache_key,
            rule_coverage=pass2.rules.coverage,
            warnings=tuple(dict.fromkeys(self.warnings)),
            pass_timings=tuple(timer.timings),
            output_artifacts={
                "run_manifest": RUN_MANIFEST_FILE,
                "incidents": INCIDENTS_FILE,
                "tracks_summary": TRACKS_SUMMARY_FILE,
                "scene": SCENE_FILE,
                "video": VIDEO_FILE,
                "video_proxy": VIDEO_PROXY_FILE,
                "report": REPORT_FILE,
            },
            diagnostics={
                "camera_mode": mode.value,
                "drift_peak_px": drift.peak_displacement_px,
                "drift_peak_fraction": drift.peak_fraction,
                "geometry_gate": gate.model_dump(mode="json"),
                "frames_processed": pass1.frames,
                "shots": pass1.shots,
                "detections": pass1.detections,
                "invalid_transform_frames": pass1.invalid_transform_frames,
                "work_dir": str(self.work_dir),
                "pass1_seconds": {
                    "total": round(pass1.elapsed_s, 1),
                    "detector": round(pass1.detect_s, 1),
                    "waiting_for_decode": round(pass1.decode_wait_s, 1),
                    "tracking": round(pass1.track_s, 1),
                    "store": round(pass1.store_s, 1),
                },
                "rendered_frames": rendered.frames,
            },
        )
        write_model(self.run_dir / RUN_MANIFEST_FILE, manifest)
        write_report(self.run_dir / REPORT_FILE, manifest, pass2.rules.incidents)
        check = validate_bundle(self.run_dir, require_media=True)
        if not check.ok:
            raise PipelineError("BUNDLE_INVALID", "; ".join(check.problems))
        return manifest
