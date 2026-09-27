"""Stream BGR frames into ffmpeg as H.264 (v7 §15.4, §44)."""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path
from types import TracebackType
from typing import IO

import numpy as np

from shared.errors import PipelineError

FINALIZE_TIMEOUT_S = 300.0
PROBE_TIMEOUT_S = 20.0
PROXY_QUALITY_OFFSET = 5  # proxy is smaller and viewed smaller, so it tolerates more loss


@functools.lru_cache(maxsize=4)
def hardware_encoder_works(codec: str) -> bool:
    """Whether this ffmpeg can actually encode with `codec` here (listed is not enough:
    NVENC also needs a recent enough driver, Quick Sync an enabled iGPU)."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=256x256:d=0.1",
                "-c:v",
                codec,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


HARDWARE_PREFERENCE = ("h264_nvenc", "h264_qsv")


def resolve_codec(requested: str) -> str:
    """`auto` picks the first working hardware encoder, else libx264."""
    if requested != "auto":
        return requested
    return next((c for c in HARDWARE_PREFERENCE if hardware_encoder_works(c)), "libx264")


def _codec_arguments(codec: str, *, quality: int, preset: str) -> list[str]:
    if codec == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    if codec == "h264_qsv":
        return [
            "-c:v",
            "h264_qsv",
            "-preset",
            "veryfast",
            "-global_quality",
            str(quality + 3),
            "-pix_fmt",
            "nv12",
        ]
    if codec == "libx264":
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(quality), "-pix_fmt", "yuv420p"]
    raise ValueError(f"unsupported video codec {codec!r}")


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
        codec: str = "libx264",
        proxy: Path | None = None,
        proxy_height: int = 720,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise PipelineError("FFMPEG_MISSING", "ffmpeg is not on PATH")
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p needs even frame dimensions")
        self.destination = destination
        self.partial = _partial(destination)
        self.proxy = proxy
        self.codec = resolve_codec(codec)
        self.width = width
        self.height = height
        self.frames_written = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_path = destination.with_name(destination.stem + ".ffmpeg.log")
        self._stderr: IO[bytes] = self._stderr_path.open("wb")
        command = [
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
        ]
        main_args = _codec_arguments(self.codec, quality=crf, preset=preset)
        if proxy is None:
            command += [*main_args, "-movflags", "+faststart", str(self.partial)]
        else:
            # One decode of the stream, two encodes: full resolution and the web proxy.
            command += [
                "-filter_complex",
                f"[0:v]split=2[full][small];[small]scale=-2:{proxy_height}[proxy]",
                "-map",
                "[full]",
                *main_args,
                "-movflags",
                "+faststart",
                str(self.partial),
                "-map",
                "[proxy]",
                *_codec_arguments(
                    self.codec, quality=crf + PROXY_QUALITY_OFFSET, preset="veryfast"
                ),
                "-movflags",
                "+faststart",
                str(_partial(proxy)),
            ]
        self._process = subprocess.Popen(
            command,
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
        if self.proxy is not None:
            _partial(self.proxy).replace(self.proxy)
        self._stderr_path.unlink(missing_ok=True)
        return self.destination

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._stderr.close()
        self.partial.unlink(missing_ok=True)
        if self.proxy is not None:
            _partial(self.proxy).unlink(missing_ok=True)

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


def _partial(path: Path) -> Path:
    return path.with_name(path.stem + ".partial" + path.suffix)


def transcode_proxy(source: Path, destination: Path, *, height: int = 720, crf: int = 23) -> Path:
    """Downscale the final video into the web player's proxy."""
    partial = _partial(destination)
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
