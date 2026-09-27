"""Pass three: render the side-by-side video and cut evidence images (v7 §6, §15.4).

Frames are decoded again (identically, see `FrameSource`) and joined with the pass-one
store and pass-two outputs by processed frame index.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from jobs.pass2 import Pass2Result
from pipeline.cache.pass1_store import Pass1Store
from pipeline.intake.frames import FrameSource, Letterbox, read_frame_at
from pipeline.intake.prefetch import Prefetch
from pipeline.render.compositor import Compositor
from pipeline.render.encoder import VideoEncoder
from pipeline.render.plan_view import PlanView
from pipeline.render.source_overlay import ALERT, render_source_frame
from shared.config import AppConfig
from shared.enums import PERSON_CLASS
from shared.errors import PipelineError
from shared.schemas.bundle import VIDEO_FILE, VIDEO_PROXY_FILE
from shared.schemas.incidents import IncidentRecord
from shared.schemas.scene import SceneCache
from shared.schemas.tracks import Pass1TrackObservation

CROP_PADDING = 0.15
EVIDENCE_JPEG_QUALITY = 92


@dataclass
class RenderResult:
    video: Path
    proxy: Path
    frames: int
    evidence: list[Path] = field(default_factory=list)


def _crop(image: np.ndarray, observation: Pass1TrackObservation, scale: float = 1.0) -> np.ndarray:
    box = observation.box_oriented or observation.box_reference
    pad_x, pad_y = box.width() * CROP_PADDING, box.height() * CROP_PADDING
    x1 = int(max(0, (box.x1 - pad_x) * scale))
    y1 = int(max(0, (box.y1 - pad_y) * scale))
    x2 = int(min(image.shape[1], (box.x2 + pad_x) * scale))
    y2 = int(min(image.shape[0], (box.y2 + pad_y) * scale))
    return image[y1:y2, x1:x2].copy()


def render_run(
    *,
    source: FrameSource,
    store: Pass1Store,
    pass2: Pass2Result,
    scene: SceneCache,
    config: AppConfig,
    run_dir: Path,
    title: str,
    progress: Callable[[int], None] | None = None,
) -> RenderResult:
    render = config.render
    incidents = pass2.rules.incidents
    compositor = Compositor(
        render,
        title=title,
        duration_s=pass2.end_time_s,
        incidents=incidents,
        coverage=pass2.rules.coverage,
    )
    plan = PlanView(
        (render.pane_width, render.pane_height),
        (config.intake.target_width, config.intake.target_height),
        config.intake.target_fps,
    )
    evidence_by_frame: dict[int, list[IncidentRecord]] = {}
    for incident in incidents:
        evidence_by_frame.setdefault(incident.source_frame_index, []).append(incident)

    video_path = run_dir / VIDEO_FILE
    evidence: list[Path] = []
    frames = 0
    stored = store.frames_with_observations()
    previous_shot: str | None = None
    with VideoEncoder(
        video_path,
        width=render.output_width,
        height=render.output_height,
        fps=render.output_fps,
        crf=render.crf,
        preset=render.preset,
        codec=render.video_codec,
        proxy=run_dir / VIDEO_PROXY_FILE,
        proxy_height=render.proxy_height,
    ) as encoder:
        for decoded in Prefetch(source):
            record, observations = next(stored, (None, []))
            if record is None or record.frame_index != decoded.frame_index:
                raise PipelineError(
                    "RENDER_FRAME_MISMATCH",
                    f"decoded frame {decoded.frame_index} has no matching pass-one record",
                )
            if record.shot_id != previous_shot:
                plan.reset()
                previous_shot = record.shot_id
            output = pass2.frame_outputs.get(record.frame_index)
            proposals = scene.proposals_for(record.shot_id)
            source_pane = render_source_frame(
                decoded.image,
                observations=observations,
                output=output,
                proposals=proposals,
                time_s=record.video_time_s,
                size=(render.pane_width, render.pane_height),
                hud=False,
            )
            plan_pane = plan.render(observations=observations, output=output, proposals=proposals)
            by_track = {o.canonical_track_id: o for o in observations}
            crops = {
                incident.incident_id: _crop(decoded.image, by_track[incident.canonical_track_id])
                for incident in compositor.visible_incidents(record.video_time_s)
                if incident.canonical_track_id in by_track
            }
            workers = sum(1 for o in observations if o.object_class == PERSON_CLASS)
            status = (
                f"Tracking {workers} worker(s) and {len(observations) - workers} machine(s) "
                "in this frame"
            )
            encoder.write(
                compositor.compose(
                    source_pane, plan_pane, time_s=record.video_time_s, crops=crops, status=status
                )
            )
            for incident in evidence_by_frame.get(record.frame_index, []):
                evidence += _write_evidence(
                    run_dir, incident, decoded.image, source, record.source_index, by_track
                )
            frames += 1
            if progress is not None:
                progress(frames)
    return RenderResult(
        video=video_path, proxy=run_dir / VIDEO_PROXY_FILE, frames=frames, evidence=evidence
    )


def _write_evidence(
    run_dir: Path,
    incident: IncidentRecord,
    canvas: np.ndarray,
    source: FrameSource,
    source_index: int,
    by_track: dict[str, Pass1TrackObservation],
) -> list[Path]:
    """Full frame with the subject highlighted, plus an original-resolution crop."""
    subject = by_track.get(incident.canonical_track_id or "")
    frame = canvas.copy()
    if subject is not None:
        box = subject.box_oriented or subject.box_reference
        cv2.rectangle(
            frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), ALERT, 3, cv2.LINE_AA
        )
    paths = [run_dir / incident.evidence_path]
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paths[0]), frame, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])
    if subject is not None and incident.evidence_crop_path is not None:
        crop = _original_resolution_crop(source, source_index, subject)
        if crop is None:
            crop = _crop(canvas, subject)
        crop_path = run_dir / incident.evidence_crop_path
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, EVIDENCE_JPEG_QUALITY])
        paths.append(crop_path)
    return paths


def _original_resolution_crop(
    source: FrameSource, source_index: int, subject: Pass1TrackObservation
) -> np.ndarray | None:
    """Crop from the raw source frame so L1 sees full detail, not the 1080p canvas."""
    letterbox: Letterbox | None = source.letterbox
    if letterbox is None:
        return None
    try:
        raw = read_frame_at(source.path, source_index)
    except PipelineError:
        return None
    box = subject.box_oriented or subject.box_reference
    corners = np.array([[box.x1, box.y1, 1.0], [box.x2, box.y2, 1.0]]).T
    (x1, x2), (y1, y2) = (letterbox.inverse() @ corners)[:2]
    pad_x, pad_y = (x2 - x1) * CROP_PADDING, (y2 - y1) * CROP_PADDING
    crop = raw[
        int(max(0, y1 - pad_y)) : int(min(raw.shape[0], y2 + pad_y)),
        int(max(0, x1 - pad_x)) : int(min(raw.shape[1], x2 + pad_x)),
    ]
    return crop.copy() if crop.size else None
