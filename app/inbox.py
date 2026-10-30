"""Upload inbox: the hosted site queues videos, the workstation's GPU worker runs them.

The inbox is a private Hugging Face dataset (`SAFETY_INBOX_REPO`), separate from the
published runs, so raw footage never mixes with privacy-filtered bundles:

    inbox/<job_id>/<video>      the upload, deleted once the job finishes or fails
    inbox/<job_id>/job.json     status: queued -> running -> done | failed

A job whose folder still holds a video is pending; the worker (`tools/worker.py`) claims
the oldest, runs the full pipeline, publishes the run with `tools/publish_run.py` and
marks the job done. Live progress goes from the worker straight to the site
(`/api/worker/beat`); only status changes are committed, three small commits per job.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

# The chunked Xet transfer stalled on a home uplink where plain HTTP finished; read at import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

JOB_ID = re.compile(r"^[a-f0-9]{12}$")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 300.0
STATUSES = ("queued", "running", "done", "failed")


def safe_name(filename: str) -> str:
    """A filename that is safe in a repo path and in a report title."""
    path = PurePath(filename.replace("\\", "/"))
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:60] or "upload"
    suffix = path.suffix.lower()
    return stem + (suffix if suffix in VIDEO_SUFFIXES else ".mp4")


def new_job(filename: str, size: int) -> dict[str, Any]:
    now = time.time()
    return {
        "job_id": secrets.token_hex(6),
        "status": "queued",
        "video": safe_name(filename),
        "size": size,
        "created_at": now,
        "updated_at": now,
        "message": "Waiting for the processing worker.",
        "run_id": None,
    }


@dataclass
class Inbox:
    repo: str
    token: str | None = None

    def __post_init__(self) -> None:
        if self.token is None:  # on the workstation: the `hf auth login` token
            try:
                from huggingface_hub import get_token

                self.token = get_token()
            except ImportError:
                pass

    def _api(self):  # noqa: ANN202 - huggingface_hub is optional outside serve/worker
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def ensure(self) -> None:
        self._api().create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)

    # ------------------------------------------------------------------ site

    def submit(self, video: Path, filename: str) -> dict[str, Any]:
        """Queue an upload: the video and its job.json in one commit."""
        from huggingface_hub import CommitOperationAdd

        job = new_job(filename, video.stat().st_size)
        folder = f"inbox/{job['job_id']}"
        if not getattr(self, "_ready", False):  # the first upload creates the private inbox
            self.ensure()
            self._ready = True
        self._api().create_commit(
            repo_id=self.repo,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(f"{folder}/{job['video']}", str(video)),
                CommitOperationAdd(f"{folder}/job.json", _encode(job)),
            ],
            commit_message=f"Queue upload {job['job_id']}",
        )
        return job

    def job(self, job_id: str) -> dict[str, Any] | None:
        """The job's current state, read past any cache."""
        if not JOB_ID.match(job_id):
            return None
        url = f"https://huggingface.co/datasets/{self.repo}/resolve/main/inbox/{job_id}/job.json"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                return None
            raise

    # ------------------------------------------------------------------ worker

    def pending(self) -> list[dict[str, Any]]:
        """Jobs whose upload is still in the inbox, oldest first."""
        files = self._api().list_repo_files(self.repo, repo_type="dataset")
        folders: dict[str, set[str]] = {}
        for name in files:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "inbox" and JOB_ID.match(parts[1]):
                folders.setdefault(parts[1], set()).add(parts[2])
        jobs = []
        for job_id, names in folders.items():
            if "job.json" in names and any(PurePath(n).suffix in VIDEO_SUFFIXES for n in names):
                job = self.job(job_id)
                if job and job.get("status") in ("queued", "running"):
                    jobs.append(job)
        return sorted(jobs, key=lambda j: j.get("created_at", 0))

    def update(self, job: dict[str, Any], **changes: Any) -> dict[str, Any]:
        job.update(changes, updated_at=time.time())
        self._api().upload_file(
            path_or_fileobj=_encode(job),
            path_in_repo=f"inbox/{job['job_id']}/job.json",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"Job {job['job_id']}: {job['status']}",
        )
        return job

    def download(self, job: dict[str, Any], target: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo,
                f"inbox/{job['job_id']}/{job['video']}",
                repo_type="dataset",
                local_dir=target,
                token=self.token,
            )
        )

    def discard_video(self, job: dict[str, Any]) -> None:
        """Raw footage leaves the inbox as soon as the job ends, whatever the outcome."""
        self._api().delete_file(
            f"inbox/{job['job_id']}/{job['video']}",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"Discard upload {job['job_id']}",
        )


def _encode(job: dict[str, Any]) -> bytes:
    return (json.dumps(job, indent=2) + "\n").encode("utf-8")


PROGRESS = re.compile(r"^(?P<stage>[a-z]+): (?P<done>\d+)/(?P<total>\d+) frames$")


def parse_progress(message: str) -> dict[str, Any]:
    """ProcessJob reports "detect: 120/4325 frames"; the site shows a stage and a fraction."""
    m = PROGRESS.match(message.strip())
    if not m:
        return {"message": message[:200]}
    done, total = int(m["done"]), int(m["total"])
    return {"stage": m["stage"], "done": done, "total": total, "message": message[:200]}
