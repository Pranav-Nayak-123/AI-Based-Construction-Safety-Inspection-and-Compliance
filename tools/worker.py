"""GPU worker: process videos uploaded on the hosted site, on this workstation.

    uv run --no-sync python tools/worker.py --site https://safety-twin.onrender.com \\
        --inbox <user>/safety-twin-inbox --runs <user>/safety-twin-runs

Needs a Hugging Face token with write access (`hf auth login`, or HF_TOKEN) and the
site's worker key in SAFETY_WORKER_KEY. Each loop it claims the oldest queued upload,
runs the full pipeline (`ProcessJob`), publishes the run through the same privacy gate as
`tools/publish_run.py`, marks the job done and deletes the raw upload. Progress and a
heartbeat go to the site so the uploader sees which stage the job is in. Heavy GPU work
waits while the laptop runs on battery.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inbox import MAX_VIDEO_SECONDS, Inbox, parse_progress  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Site:
    """Heartbeat and progress to the hosted app; failures are never fatal to a job."""

    def __init__(self, url: str | None, key: str | None) -> None:
        self.url = url.rstrip("/") if url else None
        self.key = key
        self.last = 0.0

    def beat(self, state: str, job: dict[str, Any] | None = None, message: str = "") -> None:
        if not self.url or not self.key:
            return
        body = {"state": state, "job_id": job["job_id"] if job else None, **parse_progress(message)}
        request = urllib.request.Request(
            f"{self.url}/api/worker/beat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Worker-Key": self.key},
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=20).close()
            self.last = time.time()
        except OSError as error:
            print(f"  (site unreachable: {error})", flush=True)


def on_battery() -> bool:
    try:
        import psutil

        battery = psutil.sensors_battery()
    except Exception:  # noqa: BLE001 - no sensor means mains
        return False
    return battery is not None and not battery.power_plugged


def video_seconds(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            raise ValueError("this file could not be read as a video")
        return frames / fps
    finally:
        capture.release()


def process(job: dict[str, Any], inbox: Inbox, site: Site, runs_repo: str, output: Path) -> None:
    job_id = job["job_id"]
    run_id = f"upload-{job_id}"
    inbox.update(job, status="running", message="Processing on the GPU worker.")
    site.beat("running", job, "Downloading the upload")
    scratch = Path(tempfile.mkdtemp(prefix=f"upload-{job_id}-"))
    try:
        video = inbox.download(job, scratch)
        seconds = video_seconds(video)
        if seconds > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"the video is {seconds:.0f} s long; uploads are limited to "
                f"{MAX_VIDEO_SECONDS:.0f} s"
            )
        # The models load only for a video that passed the checks above.
        from jobs.process_job import ProcessJob
        from tools.publish_run import check, upload

        run_dir = output / run_id
        if run_dir.exists():  # a previous attempt died part-way
            shutil.rmtree(run_dir)

        def report(message: str) -> None:
            print(f"  {message}", flush=True)
            if time.time() - site.last >= 4:
                site.beat("running", job, message)

        ProcessJob(
            video, output_root=output, run_id=run_id, project_root=PROJECT_ROOT, report=report
        ).run()
        problems = check(run_dir)
        if problems:
            raise RuntimeError("not published: " + "; ".join(problems))
        site.beat("running", job, "Publishing the run")
        upload(run_dir, runs_repo, private=True)
        inbox.update(job, status="done", run_id=run_id, message="Finished.")
        site.beat("idle", job, "Finished")
        print(f"[{job_id}] done -> {run_id}", flush=True)
    except Exception as error:  # noqa: BLE001 - every failure is reported to the uploader
        traceback.print_exc()
        reason = (
            str(error).strip().splitlines()[0][:240] if str(error).strip() else type(error).__name__
        )
        inbox.update(job, status="failed", message=f"Processing failed: {reason}")
        site.beat("idle", job, "Failed")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            inbox.discard_video(job)
        except Exception as error:  # noqa: BLE001
            print(f"[{job_id}] could not delete the raw upload: {error}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", default=os.environ.get("SAFETY_SITE_URL"))
    parser.add_argument("--inbox", default=os.environ.get("SAFETY_INBOX_REPO"), required=False)
    parser.add_argument("--runs", default=os.environ.get("SAFETY_RUNS_REPO"), required=False)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--poll", type=float, default=15.0, help="seconds between inbox checks")
    parser.add_argument("--once", action="store_true", help="process what is queued, then exit")
    args = parser.parse_args(argv)
    if not args.inbox or not args.runs:
        parser.error("--inbox and --runs (or SAFETY_INBOX_REPO / SAFETY_RUNS_REPO) are required")

    inbox = Inbox(args.inbox, os.environ.get("HF_TOKEN") or None)
    inbox.ensure()
    site = Site(args.site, os.environ.get("SAFETY_WORKER_KEY"))
    if args.site and not site.key:
        print("SAFETY_WORKER_KEY is not set: the site will not see this worker.", flush=True)
    print(f"Worker watching {args.inbox}; runs publish to {args.runs}. Ctrl+C stops.", flush=True)
    while True:
        if on_battery():
            site.beat("paused", message="Paused: the laptop is on battery")
            print("On battery: waiting for mains power.", flush=True)
            time.sleep(60)
            continue
        try:
            jobs = inbox.pending()
        except Exception as error:  # noqa: BLE001 - the network comes and goes
            print(f"Inbox unreachable: {error}", flush=True)
            jobs = []
        if jobs:
            job = jobs[0]
            print(f"[{job['job_id']}] {job['video']} ({job['size'] / 1e6:.1f} MB)", flush=True)
            process(job, inbox, site, args.runs, args.output)
            continue
        if args.once:
            return 0
        site.beat("idle")
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Worker stopped.")
