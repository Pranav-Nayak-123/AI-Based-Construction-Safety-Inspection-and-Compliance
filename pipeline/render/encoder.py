"""Stream BGR frames into ffmpeg as H.264 (v7 §15.4, §44)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np

from shared.errors import PipelineError

FINALIZE_TIMEOUT_S = 300.0


class VideoEncoder:
    """Pipe raw frames to ffmpeg; the output only appears once encoding succeeded.

    Frames are written to `<name>.partial.mp4` and renamed on success, so a crash never
    leaves a truncated video that looks complete.
    """

    def __init__(
        self,
        destination: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise PipelineError("FFMPEG_MISSING", "ffmpeg is not on PATH")
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p needs even frame dimensions")
        self.destination = destination
        self.partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
        self.width = width
        self.height = height
        self.frames_written = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_path = destination.with_name(destination.stem + ".ffmpeg.log")
        self._stderr: IO[bytes] = self._stderr_path.open("wb")
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.partial),
            ],
            stdin=subprocess.PIPE,
            stderr=self._stderr,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"frame must be {self.height}x{self.width}x3 uint8, got {frame.shape} {frame.dtype}"
            )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError as error:  # BrokenPipeError on POSIX, EINVAL on Windows
            self._fail("ffmpeg exited while receiving frames", error)
        self.frames_written += 1

    def close(self) -> Path:
        assert self._process.stdin is not None
        self._process.stdin.close()
        try:
            code = self._process.wait(timeout=FINALIZE_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            self._process.kill()
            self._fail("ffmpeg did not finish in time", error)
        self._stderr.close()
        if code != 0:
            self._fail(f"ffmpeg exited with status {code}")
        self.partial.replace(self.destination)
        self._stderr_path.unlink(missing_ok=True)
        return self.destination

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._stderr.close()
        self.partial.unlink(missing_ok=True)

    def _fail(self, message: str, cause: BaseException | None = None) -> None:
        self.abort()
        details = self._stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        error = PipelineError("FFMPEG_FAILED", f"{message}: {details or 'no stderr'}")
        raise error from cause

    def __enter__(self) -> VideoEncoder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def transcode_proxy(source: Path, destination: Path, *, height: int = 720, crf: int = 23) -> Path:
    """Downscale the final video into the web player's proxy."""
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale=-2:{height}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(partial),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=FINALIZE_TIMEOUT_S)
    if completed.returncode != 0:
        partial.unlink(missing_ok=True)
        raise PipelineError("FFMPEG_FAILED", completed.stderr.strip() or "proxy transcode failed")
    partial.replace(destination)
    return destination
