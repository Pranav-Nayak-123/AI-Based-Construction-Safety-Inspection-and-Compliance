"""Video uploads: the inbox contract, the site's endpoints and the GPU worker's failure path."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.inbox import new_job, parse_progress, safe_name  # noqa: E402
from jobs.pipeline_job import FakePipelineJob  # noqa: E402

CODE, KEY = "site-code-123", "worker-key-456"


class FakeInbox:
    """The inbox in memory: what the site and the worker see, with no network."""

    jobs: dict[str, dict[str, Any]] = {}
    videos: dict[str, bytes] = {}
    discarded: list[str] = []

    def __init__(self, repo: str, token: str | None = None) -> None:
        self.repo = repo

    def ensure(self) -> None:
        pass

    def submit(self, video: Path, filename: str) -> dict[str, Any]:
        job = new_job(filename, video.stat().st_size)
        self.jobs[job["job_id"]] = dict(job)
        self.videos[job["job_id"]] = video.read_bytes()
        return job

    def job(self, job_id: str) -> dict[str, Any] | None:
        found = self.jobs.get(job_id)
        return dict(found) if found else None

    def update(self, job: dict[str, Any], **changes: Any) -> dict[str, Any]:
        job.update(changes)
        self.jobs[job["job_id"]] = dict(job)
        return job

    def download(self, job: dict[str, Any], target: Path) -> Path:
        path = target / job["video"]
        path.write_bytes(self.videos[job["job_id"]])
        return path

    def discard_video(self, job: dict[str, Any]) -> None:
        self.discarded.append(job["job_id"])
        self.videos.pop(job["job_id"], None)


@pytest.fixture(scope="module")
def cloud(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    root = tmp_path_factory.mktemp("uploads")
    clip = root / "clip.bin"
    clip.write_bytes(b"synthetic input")
    FakePipelineJob(clip, output_root=root / "runs", run_id="upload-run", skip_video=True).run()
    patch = pytest.MonkeyPatch()
    patch.setenv("SAFETY_RUNS_DIR", str(root / "runs"))
    patch.setenv("SAFETY_INBOX_REPO", "someone/inbox")
    patch.setenv("SAFETY_UPLOAD_CODE", CODE)
    patch.setenv("SAFETY_WORKER_KEY", KEY)
    patch.setenv("SAFETY_REFRESH_REPORTS", "0")
    for name in ("SAFETY_LLM_PROVIDER", "SAFETY_RUNS_REPO", "SUPABASE_URL"):
        patch.delenv(name, raising=False)
    import app.cloud

    module = importlib.reload(app.cloud)
    patch.setattr(module, "Inbox", FakeInbox)
    patch.setattr(module, "upload_limiter", module.Limiter(100))
    yield module
    patch.undo()


@pytest.fixture(scope="module")
def client(cloud: Any) -> Iterator[TestClient]:
    with TestClient(cloud.app) as test_client:
        yield test_client


def _upload(
    client: TestClient, *, code: str = CODE, name: str = "site cam.MP4", body: bytes = b"\0" * 64
):
    return client.post(f"/api/jobs?filename={name}", content=body, headers={"X-Upload-Code": code})


# ---------------------------------------------------------------------- inbox contract


def test_upload_names_are_safe_in_paths_and_titles() -> None:
    assert safe_name("../../etc/passwd") == "passwd.mp4"
    assert safe_name("C:\\Users\\me\\Site Cam #2.MOV") == "Site_Cam_2.mov"
    assert safe_name("<script>.webm") == "script.webm"
    assert safe_name("....") == "upload.mp4"


def test_worker_progress_is_parsed_for_the_status_page() -> None:
    assert parse_progress("detect: 120/4325 frames") == {
        "stage": "detect",
        "done": 120,
        "total": 4325,
        "message": "detect: 120/4325 frames",
    }
    assert parse_progress("Publishing the run") == {"message": "Publishing the run"}


# ---------------------------------------------------------------------- site


def test_uploads_need_the_access_code_and_a_video(client: TestClient) -> None:
    assert _upload(client, code="wrong").status_code == 403
    assert _upload(client, name="notes.pdf").status_code == 415
    assert _upload(client, body=b"").status_code == 400


def test_an_upload_is_queued_and_its_status_shown(client: TestClient) -> None:
    reply = _upload(client)
    assert reply.status_code == 200, reply.text
    job_id = reply.json()["job_id"]
    assert reply.json()["status_url"] == f"/jobs/{job_id}"
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "queued" and status["video"] == "site_cam.mp4"
    assert status["worker"]["online"] is False
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get("/api/jobs/zzzzzzzzzzzz").status_code == 404
    assert client.get("/jobs/../etc").status_code == 404


def test_only_the_worker_can_report_progress(cloud: Any, client: TestClient) -> None:
    job_id = _upload(client).json()["job_id"]
    beat = {"state": "running", "job_id": job_id, "stage": "detect", "done": 50, "total": 200}
    assert client.post("/api/worker/beat", json=beat).status_code == 403
    assert (
        client.post("/api/worker/beat", json=beat, headers={"X-Worker-Key": "no"}).status_code
        == 403
    )
    assert (
        client.post("/api/worker/beat", json=beat, headers={"X-Worker-Key": KEY}).status_code == 200
    )
    FakeInbox.jobs[job_id]["status"] = "running"
    cloud._jobs_cache.clear()
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["worker"]["online"] is True
    assert status["progress"]["done"] == 50 and status["progress"]["total"] == 200


def test_a_finished_job_links_its_report(cloud: Any, client: TestClient) -> None:
    job_id = _upload(client).json()["job_id"]
    FakeInbox.jobs[job_id].update(status="done", run_id="upload-run")
    cloud._jobs_cache.clear()
    assert client.get(f"/api/jobs/{job_id}").json()["report"] == "/runs/upload-run/"


def test_upload_page_and_disabled_uploads(
    cloud: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "Analyse your own video" in client.get("/upload").text
    monkeypatch.delenv("SAFETY_UPLOAD_CODE")
    assert _upload(client).status_code == 503
    assert "not enabled" in client.get("/upload").text


def test_uploads_are_rate_limited_per_client(
    cloud: Any, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cloud, "upload_limiter", cloud.Limiter(1))
    assert _upload(client).status_code == 200
    assert _upload(client).status_code == 429


# ---------------------------------------------------------------------- worker


def test_a_failed_job_is_reported_and_its_upload_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import worker

    inbox = FakeInbox("someone/inbox")
    video = tmp_path / "long.mp4"
    video.write_bytes(b"\0" * 32)
    job = inbox.submit(video, "long.mp4")
    monkeypatch.setattr(worker, "video_seconds", lambda path: 9999.0)
    worker.process(dict(job), inbox, worker.Site(None, None), "someone/runs", tmp_path / "out")
    final = inbox.job(job["job_id"])
    assert final["status"] == "failed" and "limited to 300 s" in final["message"]
    assert job["job_id"] in FakeInbox.discarded
