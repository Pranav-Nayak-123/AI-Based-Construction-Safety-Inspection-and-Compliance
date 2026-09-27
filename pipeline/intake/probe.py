"""Probe a video with ffprobe (v7 §33.1)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field

from shared.coordinates import StrictModel
from shared.errors import PipelineError

PROBE_TIMEOUT_S = 30.0


class VideoMetadata(StrictModel):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    codec: str
    frame_count: int | None = None
    rotation_deg: int = 0
    bit_rate_kbps: float | None = None

    @property
    def display_size(self) -> tuple[int, int]:
        """Width and height after applying the container's rotation."""
        if self.rotation_deg % 180:
            return self.height, self.width
        return self.width, self.height


def _rate(value: str | None) -> float | None:
    if not value or "/" not in value:
        return None
    numerator, denominator = (float(part) for part in value.split("/", 1))
    return numerator / denominator if denominator else None


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"]))) % 360
    rotate = (stream.get("tags") or {}).get("rotate")
    return int(rotate) % 360 if rotate is not None else 0


def probe_video(path: Path) -> VideoMetadata:
    if shutil.which("ffprobe") is None:
        raise PipelineError("FFPROBE_MISSING", "ffprobe is not on PATH")
    if not path.is_file():
        raise PipelineError("INPUT_NOT_FOUND", str(path))
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,codec_name,"
        "side_data_list:stream_tags=rotate",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise PipelineError("FFPROBE_TIMEOUT", str(path)) from error
    if completed.returncode != 0:
        raise PipelineError("FFPROBE_FAILED", completed.stderr.strip() or str(path))

    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise PipelineError("NO_VIDEO_STREAM", str(path))
    stream = streams[0]
    fmt = payload.get("format") or {}

    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    duration = float(fmt["duration"]) if fmt.get("duration") else None
    if fps is None or duration is None:
        raise PipelineError("UNREADABLE_TIMING", f"{path}: no frame rate or duration")
    frames = stream.get("nb_frames")
    bit_rate = fmt.get("bit_rate")
    return VideoMetadata(
        path=str(path.resolve()),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration_s=duration,
        codec=str(stream.get("codec_name", "unknown")),
        frame_count=int(frames) if frames and str(frames).isdigit() else None,
        rotation_deg=_rotation(stream),
        bit_rate_kbps=float(bit_rate) / 1000 if bit_rate else None,
    )
