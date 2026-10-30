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
import json
import math
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jobs.pass2 import Pass2Result, run_pass2
from jobs.render_stage import RenderResult, render_run
from pipeline.cache.pass1_store import Pass1Store
from pipeline.intake.frames import FrameSource, Letterbox, sample_keyframes
from pipeline.intake.geometry_gate import (
    CameraMode,
    DriftDecision,
    GeometryGate,
    classify_drift,
    gate_geometry,
)
from pipeline.intake.probe import VideoMetadata, probe_video
from pipeline.intake.stabilization import DriftProbe, probe_drift
from pipeline.mapping.camera_model import (
    Camera,
    GroundMapper,
    PlaneFit,
    default_camera,
    fit_camera,
    sample_vertical_pairs,
    visible_ground,
)
from pipeline.mapping.frame_context import FrameContextBuilder
from pipeline.mapping.stitching import StitchedStore, appearance_from_video, stitch
from pipeline.output.bundle import read_model, validate_bundle, write_model
from pipeline.output.summaries import TrackSummaryBuilder
from pipeline.output.twin_data import FALLBACK_FAR_WH, TwinRecorder
from pipeline.report.html import write_report
from pipeline.rules.catalogue import load_catalogue
from pipeline.rules.engine import RuleEngine
from pipeline.scene.site_config import (
    load_reference_image,
    load_site_config,
    scene_from_site_config,
    view_matches,
)
from pipeline.scene.zone_health import REVIEW_THRESHOLD, median_background, zone_change
from pipeline.vision.detector import Stage1Detector, file_sha256, resolve_weights
from pipeline.vision.pass1 import Pass1Summary, VisionPass, summary_from_store
from pipeline.vision.posture import PoseVerifier, resolve_pose_weights
from pipeline.vision.ppe_model import PPEClassifier
from pipeline.vision.tracker import ByteTrackAdapter
from shared.config import AppConfig, load_config
from shared.enums import JobStage, ProcessingStatus
from shared.errors import PipelineError
from shared.paths import run_work_dir
from shared.schemas.bundle import (
    INCIDENTS_FILE,
    INSIGHTS_FILE,
    REPORT_FILE,
    RUN_MANIFEST_FILE,
    SCENE_FILE,
    TRACKS_SUMMARY_FILE,
    TWIN_FILE,
    VIDEO_FILE,
    VIDEO_PROXY_FILE,
    IncidentsFile,
)
from shared.schemas.run import InputVideoMeta, PassTiming, RunManifest
from shared.schemas.scene import SceneCache

DRIFT_SAMPLES = 24
Reporter = Callable[[str], None]


def _on_battery() -> bool:
    """Laptops cap clocks on battery: the GTX 1660 Ti runs detection about 3x slower."""
    try:
        import psutil  # installed with the vision stack
    except ImportError:
        return False
    battery = psutil.sensors_battery()
    return battery is not None and not battery.power_plugged


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


@dataclass
class Analysis:
    """What `ProcessJob.analyse()` found, before rendering and publishing."""

    created_at: str
    timer: _Timer
    metadata: VideoMetadata
    video_sha: str
    decision: DriftDecision
    drift: DriftProbe
    source: FrameSource
    store: Pass1Store
    pass1: Pass1Summary
    stage1_id: str
    gate: GeometryGate
    scene: SceneCache
    tracks: Pass1Store | StitchedStore
    pass2: Pass2Result


# Models loaded once per process and shared by every job in it: a benchmark of hundreds of
# clips would otherwise reload the detector each time. They hold no per-video state.
_MODELS: dict[tuple[str, ...], object] = {}


def _model(key: tuple[str, ...], load: Callable[[], object]) -> Any:
    if key not in _MODELS:
        _MODELS[key] = load()
    return _MODELS[key]


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
        reuse_pass1: Path | None = None,
    ) -> None:
        self.video = video.resolve()
        # A finished run whose detections, tracks and PPE are reused: only geometry, rules
        # and rendering run again (new zones or rule settings in minutes, no GPU needed).
        self.reuse_run = reuse_pass1.resolve() if reuse_pass1 is not None else None
        self.config = config or load_config()
        self.root = project_root or Path(__file__).resolve().parents[1]
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.run_dir = output_root.resolve() / self.run_id
        self.work_dir = run_work_dir(self.run_id)
        self.site_config = site_config
        self.report = report or (lambda message: None)
        self.warnings: list[str] = []
        self.model_ids: dict[str, str] = {}
        self.plane_fits: dict[str, PlaneFit] = {}
        self.on_battery = False
        self.reused_from: str | None = None
        self.keyframes: list[np.ndarray] = []
        self.zone_health: dict[str, float] = {}
        self.stitching: dict[str, int] = {}
        self.posture_available = False
        self.detector_classes: list[str] = []

    def run(self) -> RunManifest:
        if self.run_dir.exists():
            raise PipelineError("RUN_EXISTS", f"{self.run_dir} already exists")
        self.run_dir.mkdir(parents=True)
        found = self.analyse()
        timer, pass1 = found.timer, found.pass1

        with timer.stage(JobStage.RENDER):
            rendered = render_run(
                source=found.source,
                store=found.tracks,
                pass2=found.pass2,
                scene=found.scene,
                config=self.config,
                run_dir=self.run_dir,
                title=f"Construction Safety Twin  |  {self.video.name}",
                mappers={
                    shot: GroundMapper(fit) for shot, fit in self.plane_fits.items() if fit.accepted
                },
                progress=self._progress("render", pass1.frames),
            )
        found.store.close()

        with timer.stage(JobStage.PUBLISH):
            manifest = self._publish(
                found.created_at,
                found.metadata,
                found.video_sha,
                found.stage1_id,
                found.decision.mode,
                found.drift,
                found.gate,
                pass1,
                found.pass2,
                found.scene,
                rendered,
                timer,
            )
        return manifest

    def analyse(self) -> Analysis:
        """Intake, detection, geometry and rules: everything but rendering and the bundle.

        Writes only the pass-1 store in the work directory, so benchmarks can score rules on
        many clips quickly (evaluation/synthsite_eval.py); `run()` renders and publishes.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        created_at = _now()
        timer = _Timer([])
        self.on_battery = _on_battery()
        if self.on_battery:
            self.warnings.append(
                "processed on battery power: GPU/CPU clocks are capped, so stage timings are "
                "not representative of FR-1 performance"
            )

        with timer.stage(JobStage.INTAKE):
            metadata = probe_video(self.video)
            video_sha = _sha256(self.video)
            self.keyframes = sample_keyframes(
                self.video, DRIFT_SAMPLES, (metadata.width, metadata.height)
            )
            drift = probe_drift(self.keyframes)
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
            if self.reuse_run is not None:
                store, pass1, stage1_id, ppe_available = self._reuse_pass1()
            else:
                store, pass1, stage1_id, ppe_available = self._run_pass1(source, metadata, decision)

        with timer.stage(JobStage.MAPPING_RULES):
            gate = self._geometry_gate(decision.mode, drift, pass1)
            self.plane_fits = self._fit_planes(store, pass1, gate)
            scene = self._scene(source, store, pass1, video_sha)
            tracks = self._stitch(store)
            pass2 = self._pass2(tracks, scene, gate, ppe_available=ppe_available)
            self.report(
                f"pass 2: {len(pass2.rules.incidents)} incident(s); geometry "
                f"{'supported' if gate.supported else 'unsupported'} ({gate.reason_code})"
            )
        return Analysis(
            created_at=created_at,
            timer=timer,
            metadata=metadata,
            video_sha=video_sha,
            decision=decision,
            drift=drift,
            source=source,
            store=store,
            pass1=pass1,
            stage1_id=stage1_id,
            gate=gate,
            scene=scene,
            tracks=tracks,
            pass2=pass2,
        )

    # ------------------------------------------------------------------ stages

    def _run_pass1(
        self, source: FrameSource, metadata: VideoMetadata, decision: DriftDecision
    ) -> tuple[Pass1Store, Pass1Summary, str, bool]:
        weights, fine_tuned = resolve_weights(self.config.stage1, self.root)
        if not fine_tuned:
            self.warnings.append(
                f"stage-1 detector is the pretrained fallback ({weights.name}), "
                "not a construction fine-tune"
            )
        detector = _model(
            ("stage1", str(weights), self.config.stage1.model_dump_json()),
            lambda: Stage1Detector(self.config.stage1, weights),
        )
        tracker = ByteTrackAdapter(
            self.config.tracker,
            # The tracker only sees detection frames: its memory is counted in those.
            self.config.intake.target_fps / self.config.stage1.detect_every_frames,
            sorted(set(detector.classes.values()) - set(self.config.stage1.evidence_classes)),
        )
        ppe = self._ppe_classifier(detector.device)
        pose_weights = resolve_pose_weights(self.config.pose.model_path, self.root)
        pose = (
            _model(("pose", str(pose_weights)), lambda: PoseVerifier(pose_weights, detector.device))
            if pose_weights
            else None
        )
        if pose is None:
            self.warnings.append("pose model missing: R6 judges posture from box shape only")
        store = Pass1Store(self.work_dir / "pass1.sqlite")
        expected = max(1, round(metadata.duration_s * self.config.intake.target_fps))
        pass1 = VisionPass(
            detector, tracker, self.config, camera_mode=decision.mode, ppe=ppe, pose=pose
        ).run(source, store, progress=self._progress("detect", expected))
        store.set_meta("posture", True)
        store.set_meta("hazard_classes", sorted(set(detector.classes.values())))
        stage1_id = f"{weights.name}:{detector.model_sha256()[:16]}"
        self.posture_available = True
        self.detector_classes = sorted(set(detector.classes.values()))
        store.set_meta("detector_sha256", detector.model_sha256())
        store.set_meta("stage1_id", stage1_id)
        store.set_meta("ppe_id", self.model_ids.get("ppe"))
        store.set_meta("pass1_summary", asdict(pass1))
        store.set_meta("source_warnings", list(source.warnings))
        store.mark_complete()
        self.warnings += source.warnings
        self.report(
            f"pass 1: {pass1.frames} frames, {pass1.detections} detections, "
            f"{len(pass1.shots)} shot(s) in {pass1.elapsed_s:.1f} s "
            f"(detector {pass1.detect_s:.1f} s)"
        )
        return store, pass1, stage1_id, ppe is not None

    def _reuse_pass1(self) -> tuple[Pass1Store, Pass1Summary, str, bool]:
        """Copy a finished run's pass-1 store and carry over what it recorded."""
        assert self.reuse_run is not None
        previous = read_model(self.reuse_run / RUN_MANIFEST_FILE, RunManifest)
        if Path(previous.input_video.path).resolve() != self.video:
            raise PipelineError(
                "REUSE_MISMATCH", f"{self.reuse_run} processed {previous.input_video.path}"
            )
        source_store = Path(previous.diagnostics["work_dir"]) / "pass1.sqlite"
        if not source_store.is_file():
            raise PipelineError("REUSE_MISSING", f"pass-1 store not found: {source_store}")
        shutil.copyfile(source_store, self.work_dir / "pass1.sqlite")
        store = Pass1Store(self.work_dir / "pass1.sqlite")
        if not store.complete:
            raise PipelineError("REUSE_INCOMPLETE", f"{source_store} is not a finished pass 1")
        self.posture_available = bool(store.meta("posture", False))
        self.detector_classes = store.meta("hazard_classes", [])
        recorded = store.meta("pass1_summary")
        pass1 = Pass1Summary(**recorded) if recorded else summary_from_store(store)
        stage1_id = store.meta("stage1_id") or previous.model_ids.get("stage1", "unknown")
        ppe_id = store.meta("ppe_id") or previous.model_ids.get("ppe")
        if ppe_id:
            self.model_ids["ppe"] = ppe_id
        else:
            self.warnings.append("PPE classifier not trained yet: R1/R2 unsupported")
        if not stage1_id.startswith(Path(self.config.stage1.model_path).name):
            self.warnings.append(
                f"stage-1 detector is the pretrained fallback ({stage1_id.split(':')[0]}), "
                "not a construction fine-tune"
            )
        self.warnings += store.meta("source_warnings", [])
        self.warnings.append(f"detections reused from run {previous.run_id} (reprocess)")
        self.reused_from = previous.run_id
        self.report(f"pass 1 reused from {previous.run_id}: {pass1.frames} frames")
        return store, pass1, stage1_id, bool(ppe_id)

    def _stitch(self, store: Pass1Store) -> Pass1Store | StitchedStore:
        """One identity per worker across occlusions; the store itself stays as recorded."""
        if not self.config.tracker.stitch:
            return store
        appearance = appearance_from_video(
            self.video,
            store.frames_with_observations(),
            (self.config.intake.target_width, self.config.intake.target_height),
        )
        result = stitch(store.frames_with_observations(), appearance=appearance)
        self.stitching = {
            "tracks_before": result.tracks_before,
            "identities_after": result.tracks_after,
            "links": len(result.links),
        }
        self.report(
            f"stitching: {result.tracks_before} person tracks -> {result.tracks_after} "
            f"identities ({len(result.links)} links)"
        )
        return StitchedStore(store, result.mapping)

    def _twin_export(self, pass2: Pass2Result, scene: SceneCache) -> dict | None:
        """The calibrated shot's ground-plane twin for the report's 3D viewer."""
        accepted = [(shot, fit) for shot, fit in self.plane_fits.items() if fit.accepted]
        if pass2.twin is None:
            return None
        if not accepted:
            if pass2.twin.fallback is None or not pass2.twin.positions_seen():
                return None
            camera_model = self._fallback_camera()
            size = (self.config.intake.target_width, self.config.intake.target_height)
            shot = next(iter(self.plane_fits), next(iter(scene.shot_proposals), "s0"))
            return pass2.twin.export(
                proposals=scene.proposals_for(shot),
                ground=camera_model.ground,
                camera={
                    "height_wh": camera_model.p.height_wh,
                    "tilt_deg": round(math.degrees(camera_model.p.tilt_rad), 2),
                    "roll_deg": 0.0,
                    "support": visible_ground(camera_model, *size, far_wh=FALLBACK_FAR_WH),
                    "approximate": True,
                },
            )
        shot, fit = accepted[0]
        mapper = GroundMapper(fit)
        parameters = fit.parameters
        camera = {
            "height_wh": round(parameters.height_wh, 3),
            "tilt_deg": round(math.degrees(parameters.tilt_rad), 2),
            "roll_deg": round(math.degrees(parameters.roll_rad), 2),
            "support": [[round(x, 3), round(y, 3)] for x, y in (fit.support_polygon or [])],
        }
        return pass2.twin.export(
            proposals=scene.proposals_for(shot), ground=mapper.ground, camera=camera
        )

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
        scene = scene_from_site_config(
            config, shot_ids=matched, video_sha256=video_sha, warnings=tuple(self.warnings[-1:])
        )
        if len(first_frames) == 1 and matched and self.keyframes:
            # One fixed view: the clip's median background against the drawing reference.
            canvas = (self.config.intake.target_width, self.config.intake.target_height)
            height, width = self.keyframes[0].shape[:2]
            letterbox = Letterbox.fit((width, height), canvas)
            background = median_background([letterbox.apply(f) for f in self.keyframes])
            reference_canvas = cv2.resize(reference, canvas, interpolation=cv2.INTER_AREA)
            self.zone_health = zone_change(
                reference_canvas, background, scene.proposals_for(matched[0])
            )
            labels = {
                p.proposal_id: p.label or p.proposal_id for p in scene.proposals_for(matched[0])
            }
            for proposal_id, change in self.zone_health.items():
                if change >= REVIEW_THRESHOLD:
                    self.warnings.append(
                        f"zone health: '{labels[proposal_id]}' looks different from when it was "
                        f"drawn (change {change:.2f}); review the site config"
                    )
        return scene

    def _fit_planes(
        self, store: Pass1Store, pass1: Pass1Summary, gate: GeometryGate
    ) -> dict[str, PlaneFit]:
        """Calibrate the camera from upright workers, once per shot (v7 §37)."""
        if not gate.supported:
            return {}
        plane, uncertainty = self.config.relative_plane, self.config.uncertainty
        canvas = (self.config.intake.target_width, self.config.intake.target_height)
        fits: dict[str, PlaneFit] = {}
        for shot in pass1.shots:
            frames = (
                (record, observations)
                for record, observations in store.frames_with_observations()
                if record.shot_id == shot and record.transform_valid
            )
            pairs = sample_vertical_pairs(
                frames,
                minimum_height_px=plane.minimum_pixel_height,
                minimum_confidence=0.5,
                minimum_aspect=self.config.ppe.minimum_upright_aspect + 0.4,
                spacing_s=self.config.pose.sample_spacing_seconds,
                maximum_per_track=self.config.pose.maximum_samples_per_track,
                canvas=canvas,
            )
            fit = fit_camera(
                pairs,
                canvas,
                minimum_pairs=plane.minimum_pairs,
                minimum_tracks=plane.minimum_distinct_tracks,
                minimum_inlier_ratio=plane.minimum_vertical_inlier_ratio,
                bootstrap_samples=uncertainty.bootstrap_samples,
                minimum_successful=uncertainty.minimum_successful_samples,
                seed=uncertainty.seed,
                focal_prior=plane.focal_prior_widths,
                maximum_relative_rms=plane.maximum_relative_rms,
            )
            fits[shot] = fit
            if fit.accepted and fit.parameters is not None:
                p = fit.parameters
                self.report(
                    f"camera {shot}: tilt {math.degrees(p.tilt_rad):.1f} deg, "
                    f"roll {math.degrees(p.roll_rad):.1f} deg, height {p.height_wh:.2f} WH, "
                    f"focal {p.focal_px:.0f} px from {fit.observations} observations"
                )
            else:
                self.warnings.append(
                    f"camera calibration rejected for {shot} ({fit.reason_code}): "
                    "R4/R5 distances inconclusive"
                )
        return fits

    def _ppe_classifier(self, device: str) -> PPEClassifier | None:
        path = self.root / self.config.ppe.model_path
        if not path.is_file():
            self.warnings.append("PPE classifier not trained yet: R1/R2 unsupported")
            return None
        self.model_ids["ppe"] = f"{path.name}:{file_sha256(path)[:16]}"
        return _model(
            ("ppe", str(path), str(self.config.ppe.batch_size)),
            lambda: PPEClassifier(path, device, batch_size=self.config.ppe.batch_size),
        )

    def _pass2(
        self,
        store: Pass1Store | StitchedStore,
        scene: SceneCache,
        gate: GeometryGate,
        *,
        ppe_available: bool,
    ) -> Pass2Result:
        builder = FrameContextBuilder(
            run_id=self.run_id,
            scene=scene,
            geometry_supported=gate.supported,
            ppe_available=ppe_available,
            machinery_classes=self.config.stage1.machinery_classes,
            mappers={
                shot: GroundMapper(fit, max_replicates=self.config.uncertainty.frame_replicates)
                for shot, fit in self.plane_fits.items()
                if fit.accepted
            },
            rules=self.config.rules,
            seed=self.config.uncertainty.seed,
            hazard_classes=[
                c for c in self.config.stage1.hazard_classes if c in self.detector_classes
            ],
            posture_available=self.posture_available,
        )
        engine = RuleEngine(run_id=self.run_id, config=self.config.rules)
        return run_pass2(
            store,
            builder,
            engine,
            TrackSummaryBuilder(self.run_id),
            TwinRecorder(fps=self.config.intake.target_fps, fallback=self._fallback_ground(gate)),
        )

    def _fallback_camera(self) -> Camera:
        size = (self.config.intake.target_width, self.config.intake.target_height)
        return Camera(default_camera(*size))

    def _fallback_ground(self, gate: GeometryGate):  # noqa: ANN202
        """A steady camera that the workers could not calibrate still gets an approximate twin."""
        if not gate.supported or any(fit.accepted for fit in self.plane_fits.values()):
            return None
        return self._fallback_camera().ground

    def _publish(
        self,
        created_at: str,
        metadata: VideoMetadata,
        video_sha: str,
        stage1_id: str,
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
        incidents = tuple(
            incident.model_copy(
                update={"evidence_clip_path": rendered.clips.get(incident.incident_id)}
            )
            for incident in pass2.rules.incidents
        )
        write_model(
            self.run_dir / INCIDENTS_FILE,
            IncidentsFile(
                run_id=self.run_id,
                catalogue_sha256=catalogue.sha256,
                incidents=incidents,
            ),
        )
        write_model(self.run_dir / TRACKS_SUMMARY_FILE, pass2.tracks_summary)
        write_model(self.run_dir / SCENE_FILE, scene)
        artifacts = {}
        insights = None
        if pass2.insights is not None:
            insights = pass2.insights.model_copy(
                update={"heatmap_path": rendered.heatmap, "zone_health": self.zone_health}
            )
            write_model(self.run_dir / INSIGHTS_FILE, insights)
            artifacts["insights"] = INSIGHTS_FILE
        if rendered.heatmap:
            artifacts["heatmap"] = rendered.heatmap
        if rendered.site_view:
            artifacts["site_view"] = rendered.site_view
        twin = self._twin_export(pass2, scene)
        if twin is not None:
            (self.run_dir / TWIN_FILE).write_text(
                json.dumps(twin, separators=(",", ":")), encoding="utf-8"
            )
            artifacts["twin"] = TWIN_FILE
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
            model_ids={
                "stage1": stage1_id,
                **self.model_ids,
            },
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
                **artifacts,
            },
            diagnostics={
                "camera_mode": mode.value,
                "on_battery": self.on_battery,
                "reused_from": self.reused_from,
                "privacy": self.config.render.privacy,
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
                    "ppe": round(pass1.ppe_s, 1),
                    "hard_hat_overrides": pass1.helmet_evidence_overrides,
                    "store": round(pass1.store_s, 1),
                },
                "rendered_frames": rendered.frames,
                "stitching": self.stitching,
                "camera_fits": {
                    shot: fit.model_dump(mode="json", exclude={"bootstrap", "support_polygon"})
                    | {"bootstrap_replicates": len(fit.bootstrap)}
                    for shot, fit in self.plane_fits.items()
                },
            },
        )
        write_model(self.run_dir / RUN_MANIFEST_FILE, manifest)
        write_report(
            self.run_dir / REPORT_FILE,
            manifest,
            incidents,
            tracks=pass2.tracks_summary,
            insights=insights,
            zone_labels={
                p.proposal_id: p.label or p.proposal_id
                for proposals in scene.shot_proposals.values()
                for p in proposals
            },
            near_wh=self.config.rules.r4_near_max_wh,
            caution_wh=self.config.rules.r4_caution_max_wh,
            private=self.config.render.privacy == "faces",
            twin=twin,
        )
        check = validate_bundle(self.run_dir, require_media=True)
        if not check.ok:
            raise PipelineError("BUNDLE_INVALID", "; ".join(check.problems))
        return manifest
