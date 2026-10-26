"""Hosted demo (Section 11): published runs, their reports, the risk register and the chatbot.

    GET  /                          the run gallery
    GET  /runs/{run_id}/            that run's report, with the live chat switched on
    GET  /runs/{run_id}/{path}      bundle files (video with Range requests, evidence, assets)
    GET  /api/runs                  published runs: clip, duration, incidents, top risk
    GET  /api/runs/{run_id}/risk    the risk register as JSON
    POST /api/runs/{run_id}/chat    {question, history} -> a grounded, checked answer
    POST /api/runs/{run_id}/reviews a reviewer's verdict (stored when a database is configured)
    GET  /api/runs/{run_id}/reviews team precision per rule and reviewer agreement (kappa)
    GET  /api/stats                 chatbot monitoring: answers, model share, latency, failures
    GET  /healthz                   liveness and configuration (never secrets)
    GET  /docs                      OpenAPI

Processing stays on a GPU workstation; finished bundles are published (tools/publish_run.py)
to a Hugging Face dataset that this service downloads at startup, so the public URL works
whether or not the workstation is on.

Configuration, all from the environment (cloud secrets; never committed):

    SAFETY_RUNS_DIR          where bundles live (default output/)
    SAFETY_RUNS_REPO         Hugging Face dataset with published bundles (optional)
    HF_TOKEN                 only if that dataset is private
    SAFETY_LLM_*             the chatbot's model (llm/assistant/providers.py)
    SUPABASE_URL, SUPABASE_SERVICE_KEY   optional: chat logs and reviews in Postgres
    SAFETY_CHAT_PER_MINUTE   per-client limit on chat requests (default 8)
    SAFETY_REVIEWERS         optional: only these reviewers' verdicts count (comma-separated)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.request
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from evaluation.reviews import allowed_reviewers, fetch_supabase, summarize
from llm.assistant.chat import RiskAssistant
from llm.assistant.providers import ProviderError, from_env
from pipeline.output.run_bundle import RunBundle
from pipeline.report import brand
from shared.schemas.bundle import REPORT_FILE, RUN_MANIFEST_FILE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Client addresses are logged only as salted hashes (stable while SAFETY_LOG_SALT is set).
LOG_SALT = os.environ.get("SAFETY_LOG_SALT") or secrets.token_hex(16)
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
log = logging.getLogger("safety.cloud")


# ---------------------------------------------------------------------- runs


class RunStore:
    """Published bundles on disk, each loaded once on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._assistants: dict[str, RiskAssistant] = {}
        self._lock = threading.Lock()

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / RUN_MANIFEST_FILE).is_file())

    def path(self, run_id: str) -> Path:
        if not RUN_ID.match(run_id) or run_id not in self.ids():
            raise HTTPException(404, "no such run")
        return self.root / run_id

    def assistant(self, run_id: str) -> RiskAssistant:
        with self._lock:
            if run_id not in self._assistants:
                self._assistants[run_id] = RiskAssistant(
                    RunBundle.load(self.path(run_id)), provider
                )
            return self._assistants[run_id]


def fetch_published(root: Path) -> None:
    """Download published bundles from the Hugging Face dataset, if one is configured."""
    repo = os.environ.get("SAFETY_RUNS_REPO")
    if not repo:
        return
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=root,
        token=os.environ.get("HF_TOKEN") or None,
    )
    log.info(
        json.dumps(
            {
                "event": "runs_fetched",
                "repo": repo,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


# ---------------------------------------------------------------------- monitoring


class Stats:
    """In-memory counters for /api/stats; the durable log is stdout (and Supabase if set)."""

    def __init__(self) -> None:
        self.answers: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.latency: deque[float] = deque(maxlen=500)
        self.tokens = 0
        self.started = time.time()

    def record(self, source: str, failures: list[str], latency_s: float, tokens: int) -> None:
        self.answers[source] += 1
        for failure in failures:
            self.failures[failure.split(":", 1)[0]] += 1
        if source == "model":
            self.latency.append(latency_s)
        self.tokens += tokens

    def summary(self) -> dict[str, Any]:
        total = sum(self.answers.values())
        ordered = sorted(self.latency)
        return {
            "answers": total,
            "model_share": round(self.answers["model"] / total, 3) if total else None,
            "by_source": dict(self.answers),
            "check_failures": dict(self.failures),
            "model_latency_s": {
                "median": ordered[len(ordered) // 2] if ordered else None,
                "p90": ordered[int(len(ordered) * 0.9)] if ordered else None,
            },
            "tokens": self.tokens,
            "uptime_s": round(time.time() - self.started),
        }


class Limiter:
    """A sliding one-minute window per client, so a public URL cannot drain the API quota."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.seen: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.time()
        with self._lock:
            window = self.seen[client]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


def supabase_insert(table: str, row: dict[str, Any]) -> None:
    """Fire-and-forget insert through Supabase's REST API; a failure is only logged."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return

    # New secret keys (sb_secret_...) go only in `apikey`; legacy service_role keys are JWTs
    # and also go in Authorization.
    headers = {"apikey": key, "Content-Type": "application/json", "Prefer": "return=minimal"}
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"

    def send() -> None:
        request = urllib.request.Request(
            f"{url.rstrip('/')}/rest/v1/{table}",
            data=json.dumps(row).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except OSError as error:
            log.warning(
                json.dumps({"event": "supabase_failed", "table": table, "error": str(error)})
            )

    threading.Thread(target=send, daemon=True).start()


# ---------------------------------------------------------------------- app

logging.basicConfig(level=logging.INFO, format="%(message)s")
store = RunStore(Path(os.environ.get("SAFETY_RUNS_DIR") or PROJECT_ROOT / "output"))
try:
    provider = from_env()
except ProviderError as error:  # misconfigured: run, but answer from the register
    log.error(json.dumps({"event": "llm_config_error", "error": str(error)}))
    provider = None
stats = Stats()
limiter = Limiter(int(os.environ.get("SAFETY_CHAT_PER_MINUTE") or 8))
review_limiter = Limiter(60)
_review_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def refresh_reports() -> None:
    """Re-render each published report with this deploy's report code.

    Published bundles carry the report.html of the day they were processed; rebuilding
    from the bundle's own JSON keeps every run on the current design. A run that cannot
    be rebuilt keeps the report it was published with.
    """
    if os.environ.get("SAFETY_REFRESH_REPORTS", "1") == "0":
        return
    from pipeline.report.html import rebuild_report

    for run_id in store.ids():
        try:
            rebuild_report(store.path(run_id))
        except Exception as error:  # noqa: BLE001 - keep the published report
            log.warning(
                json.dumps({"event": "report_rebuild_failed", "run": run_id, "error": str(error)})
            )


def _startup() -> None:
    store.root.mkdir(parents=True, exist_ok=True)
    try:
        fetch_published(store.root)
    except Exception as error:  # noqa: BLE001 - serve what is on disk rather than fail
        log.error(json.dumps({"event": "runs_fetch_failed", "error": str(error)}))
    refresh_reports()
    log.info(
        json.dumps(
            {"event": "started", "runs": store.ids(), "llm": provider.name if provider else None}
        )
    )


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    _startup()
    yield


app = FastAPI(
    title="Ground Truth API",
    description="Published site-safety runs, their risk registers and a grounded chatbot.",
    version="0.3.0",
    lifespan=_lifespan,
)


@app.middleware("http")
async def _headers(request: Request, call_next):  # noqa: ANN001, ANN202
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=800)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class Review(BaseModel):
    incident_id: str = Field(max_length=120)
    verdict: str = Field(pattern="^(real|false_alarm|unsure)$")
    reason: str | None = Field(default=None, max_length=60)
    # Self-declared; SAFETY_REVIEWERS limits whose verdicts count (evaluation/reviews.py).
    reviewer: str = Field(min_length=1, max_length=40, pattern=r"^[\w .'-]+$")


@app.get("/healthz")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runs": len(store.ids()),
        "llm": {"provider": provider.name, "model": provider.model} if provider else None,
        "database": bool(os.environ.get("SUPABASE_URL")),
    }


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    out = []
    for run_id in store.ids():
        bundle = store.assistant(run_id).bundle
        top = bundle.risk.rows[0] if bundle.risk.rows else None
        out.append(
            {
                "run_id": run_id,
                "clip": PureWindowsPath(bundle.manifest.input_video.path).name,  # / or \
                "duration_s": round(bundle.duration_s, 1),
                "incidents": len(bundle.incidents),
                "workers": bundle.workers_seen,
                "completed_at": bundle.manifest.completed_at,
                "top_risk": {"rule": top.rule_id.value, "band": top.band, "rating": top.rating}
                if top
                else None,
                "report": f"/runs/{run_id}/",
                "thumbnail": _thumbnail(run_id),
            }
        )
    return out


@app.get("/api/runs/{run_id}/risk")
def risk(run_id: str) -> dict[str, Any]:
    assessment = store.assistant(run_id).bundle.risk
    return {
        "run_id": run_id,
        "workers_seen": assessment.workers_seen,
        "rows": [
            {
                "rule": r.rule_id.value,
                "title": r.title,
                "hazard": r.hazard,
                "likelihood": r.likelihood_score,
                "severity": r.severity_score,
                "rating": r.rating,
                "band": r.band,
                "why": r.why,
                "incidents": list(r.incident_ids),
                "workers": list(r.top_workers),
                "areas": list(r.areas),
                "controls": [{"level": level, "control": text} for level, text in r.controls],
            }
            for r in assessment.rows
        ],
        "not_assessed": [
            {"rule": rule.value, "reason": reason, "why": why}
            for rule, reason, why in assessment.not_assessed
        ],
        "not_observed": [
            {"rule": rule.value, "reason": reason} for rule, reason in assessment.not_observed
        ],
        "notes": list(assessment.notes),
    }


@app.post("/api/runs/{run_id}/chat")
def chat(run_id: str, body: ChatRequest, request: Request) -> JSONResponse:
    client = _client(request)
    if not limiter.allow(client):
        raise HTTPException(429, "too many questions; try again in a minute")
    assistant = store.assistant(run_id)
    started = time.perf_counter()
    reply = assistant.ask(body.question, [t.model_dump() for t in body.history])
    tokens = (reply.prompt_tokens or 0) + (reply.completion_tokens or 0)
    stats.record(reply.source, reply.failures, reply.latency_s, tokens)
    entry = {
        "event": "chat",
        "run_id": run_id,
        "source": reply.source,
        "model": reply.model,
        "latency_s": reply.latency_s,
        "total_s": round(time.perf_counter() - started, 2),
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "failures": reply.failures,
        "citations": [f.id for f in reply.citations],
        "question_sha": hashlib.sha256(body.question.encode()).hexdigest()[:12],
        "question_chars": len(body.question),
        "client": hashlib.sha256((LOG_SALT + client).encode()).hexdigest()[:12],
    }
    log.info(json.dumps(entry))
    supabase_insert("chat_logs", entry | {"question": body.question})
    return JSONResponse(
        {
            "text": reply.text,
            "source": reply.source,
            "model": reply.model,
            "latency_s": reply.latency_s,
            "failures": reply.failures,
            "citations": [{"id": f.id, "kind": f.kind, "text": f.text} for f in reply.citations],
        }
    )


@app.post("/api/runs/{run_id}/reviews")
def review(run_id: str, body: Review, request: Request) -> dict[str, str]:
    store.path(run_id)
    if not os.environ.get("SUPABASE_URL"):
        raise HTTPException(503, "reviews are kept in the browser; no database is configured")
    if not review_limiter.allow(_client(request)):
        raise HTTPException(429, "too many reviews; try again in a minute")
    row = body.model_dump() | {"reviewer": body.reviewer.strip()}
    supabase_insert("incident_reviews", {"run_id": run_id, **row})
    _review_cache.pop(run_id, None)
    return {"status": "stored"}


@app.get("/api/runs/{run_id}/reviews")
def review_summary(run_id: str) -> dict[str, Any]:
    """Team precision per rule and reviewer agreement (evaluation/reviews.py), 60 s cache."""
    incidents = store.assistant(run_id).bundle.incidents
    if not os.environ.get("SUPABASE_URL"):
        raise HTTPException(503, "no database is configured")
    cached = _review_cache.get(run_id)
    if cached and time.time() - cached[0] < 60:
        return cached[1]
    try:
        rows = fetch_supabase(run_id)
    except (OSError, RuntimeError, ValueError) as error:
        log.warning(json.dumps({"event": "reviews_fetch_failed", "error": str(error)}))
        raise HTTPException(502, "reviews are unavailable right now") from error
    summary = summarize(
        rows, {i.incident_id: i.rule_id.value for i in incidents}, allowed=allowed_reviewers()
    )
    _review_cache[run_id] = (time.time(), summary)
    return summary


@app.get("/api/stats")
def monitoring() -> dict[str, Any]:
    return stats.summary() | {"llm": provider.model if provider else None}


@app.get("/", response_class=HTMLResponse)
def gallery() -> str:
    return gallery_page()


@app.get("/runs/{run_id}")
def run_root(run_id: str) -> RedirectResponse:
    store.path(run_id)
    return RedirectResponse(f"/runs/{run_id}/", status_code=308)


@app.get("/runs/{run_id}/", response_class=HTMLResponse)
def report(run_id: str) -> str:
    html = (store.path(run_id) / REPORT_FILE).read_text(encoding="utf-8")
    meta = f'<meta name="safety-api" content="/api/runs/{escape(run_id)}">'
    return html.replace("<head>", "<head>" + meta, 1)


@app.get("/runs/{run_id}/{path:path}")
def run_file(run_id: str, path: str) -> FileResponse:
    root = store.path(run_id).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.name == REPORT_FILE:
        raise HTTPException(404, "not found")
    return FileResponse(target, headers={"Cache-Control": "public, max-age=3600"})


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


def _thumbnail(run_id: str) -> str | None:
    """The camera view with zones, else the exposure map, else the first evidence frame."""
    root = store.path(run_id)
    for name in ("site_view.jpg", "exposure_heatmap.jpg"):
        if (root / name).is_file():
            return f"/runs/{run_id}/{name}"
    frames = sorted(p for p in (root / "evidence").glob("*.jpg") if not p.stem.endswith("_crop"))
    if frames:
        return f"/runs/{run_id}/evidence/{frames[0].name}"
    video = root / "safety_twin_720p.mp4"
    return f"/runs/{run_id}/{video.name}#t=8" if video.is_file() else None


def _card(run: dict[str, Any]) -> str:
    top = run["top_risk"]
    band = top["band"] if top else "low"
    risk = (
        f'<span class="pill {escape(band.replace(" ", "-"))}">{escape(top["rule"])} · '
        f"{escape(band)} {top['rating']}/25</span>"
        if top
        else '<span class="pill low">no hazard rated</span>'
    )
    src = run["thumbnail"]
    if not src:
        thumb = '<div class="ph">no preview</div>'
    elif ".mp4" in src:  # the side-by-side render: show its camera half
        thumb = f'<video class="composite" src="{escape(src)}" preload="metadata" muted playsinline></video>'
    else:
        thumb = f'<img src="{escape(src)}" alt="" loading="lazy">'
    stem = Path(PureWindowsPath(run["clip"]).name).stem
    title = f"Clip {stem}" if stem.isdigit() else stem.replace("_", " ").capitalize()
    n = run["incidents"]
    return f"""<a class="run" href="{escape(run["report"])}" style="--band:var(--{BAND_SLOT.get(band, "good")})">
  <div class="thumb">{thumb}<span class="dur">{_mmss(run["duration_s"])}</span></div>
  <div class="body">
    <div class="row1"><span class="clip">{escape(title)}</span>{risk}</div>
    <div class="facts"><span><b>{n}</b> incident{"s" if n != 1 else ""}</span>
      <span><b>{run["workers"]}</b> workers seen</span></div>
    <div class="foot"><span class="id">{escape(run["run_id"])}</span><span class="open">Open report →</span></div>
  </div>
</a>"""


BAND_SLOT = {"very high": "critical", "high": "serious", "medium": "warning", "low": "good"}


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


GITHUB = "https://github.com/Pranav-Nayak-123/construction-safety-twin"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap"
)

GALLERY = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ground Truth · construction safety from CCTV</title>
<link rel="icon" href="{favicon}">
<meta name="description" content="Construction CCTV turned into hazard incidents, a 3D digital twin, a risk register and a cited safety assistant.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{fonts}">
<style>
:root { color-scheme:light; --bg:#f6f7f9; --surface:#fff; --surface-2:#f1f3f6; --line:#e2e5ea; --line-2:#d3d8e0;
  --text:#0e1116; --text-2:#465061; --text-3:#7a8394; --accent:#2563eb; --accent-soft:#e8efff; --hazard:#f5b301;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b; --grid:rgba(37,99,235,.07);
  --display:"Space Grotesk", Inter, "Segoe UI", system-ui, sans-serif; --mono:"JetBrains Mono", ui-monospace, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root { color-scheme:dark; --bg:#0c0e12; --surface:#13161b; --surface-2:#1a1e25;
  --line:#242a33; --line-2:#2f3642; --text:#eef1f5; --text-2:#b3bbc7; --text-3:#7d8796; --accent:#5b9dff;
  --accent-soft:rgba(91,157,255,.13); --grid:rgba(91,157,255,.06); } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 Inter, "Segoe UI", system-ui, sans-serif; -webkit-font-smoothing:antialiased; }
a { color:var(--accent); text-decoration:none; }
.wrap { max-width:1160px; margin:0 auto; padding:0 24px; }
h1, h2, h3 { font-family:var(--display); letter-spacing:-.02em; margin:0; }
/* nav */
.nav { position:sticky; top:0; z-index:10; background:color-mix(in srgb, var(--bg) 82%, transparent);
  backdrop-filter:saturate(1.4) blur(12px); border-bottom:1px solid var(--line); }
.nav .wrap { display:flex; align-items:center; gap:22px; height:62px; }
.logo { display:flex; align-items:center; gap:10px; color:var(--text); font-family:var(--display); font-weight:700; font-size:1.05rem; }
.logo-mark { display:block; }
.wordmark { font-size:1rem; font-weight:700; letter-spacing:.05em; }
.nav .links { margin-left:auto; display:flex; gap:20px; align-items:center; font-size:.9rem; }
.nav .links a { color:var(--text-2); } .nav .links a:hover { color:var(--text); }
.btn { display:inline-flex; align-items:center; gap:8px; border-radius:10px; padding:10px 16px; font-weight:600; font-size:.92rem;
  border:1px solid var(--line-2); color:var(--text); background:var(--surface); transition:transform .15s, background .15s; }
.btn:hover { transform:translateY(-1px); }
.btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.nav .btn { padding:7px 12px; font-size:.85rem; }
/* hero */
.hero { position:relative; overflow:hidden; border-bottom:1px solid var(--line);
  background-image:linear-gradient(var(--grid) 1px, transparent 1px), linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size:32px 32px; }
.hero::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(900px 400px at 85% 10%, color-mix(in srgb, var(--accent) 16%, transparent), transparent 70%),
             linear-gradient(transparent 70%, var(--bg)); }
.hero .wrap { position:relative; z-index:1; display:grid; grid-template-columns:minmax(0,1.05fr) minmax(0,1fr); gap:48px; align-items:center; padding-top:72px; padding-bottom:72px; }
.eyebrow { display:inline-flex; align-items:center; gap:8px; font-family:var(--mono); font-size:.76rem; color:var(--text-2);
  border:1px solid var(--line-2); border-radius:999px; padding:4px 12px; background:var(--surface); }
.eyebrow i { width:7px; height:7px; border-radius:50%; background:var(--good); box-shadow:0 0 0 3px color-mix(in srgb, var(--good) 25%, transparent); }
.hero h1 { font-size:clamp(2.1rem, 4.4vw, 3.4rem); line-height:1.06; margin:18px 0 16px; font-weight:700; }
.hero h1 span { color:var(--accent); }
.hero p.lead { font-size:1.08rem; color:var(--text-2); max-width:540px; margin:0 0 26px; }
.hero .cta { display:flex; gap:10px; flex-wrap:wrap; }
.shot { position:relative; border-radius:16px; border:1px solid var(--line-2); background:var(--surface); padding:8px;
  box-shadow:0 30px 80px rgba(0,0,0,.28); transform:perspective(1400px) rotateY(-6deg) rotateX(2deg); }
.shot img { display:block; width:100%; border-radius:10px; aspect-ratio:16/9; object-fit:cover; }
.shot .tag { position:absolute; display:flex; align-items:center; gap:8px; background:color-mix(in srgb, var(--surface) 92%, transparent);
  backdrop-filter:blur(8px); border:1px solid var(--line-2); border-radius:12px; padding:8px 12px; font-size:.82rem; box-shadow:0 10px 30px rgba(0,0,0,.18); }
.shot .tag b { font-family:var(--display); font-size:1.1rem; }
.shot .t1 { left:-22px; bottom:28px; } .shot .t2 { right:-18px; top:24px; }
.dot { width:9px; height:9px; border-radius:50%; flex:none; }
/* stats */
.stats { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); border:1px solid var(--line); border-radius:16px; background:var(--surface);
  margin:-36px auto 0; position:relative; z-index:2; overflow:hidden; }
.stats div { padding:18px 22px; } .stats div + div { border-left:1px solid var(--line); }
.stats b { display:block; font-family:var(--display); font-size:1.9rem; letter-spacing:-.02em; line-height:1.15; font-variant-numeric:tabular-nums; }
.stats span { color:var(--text-3); font-size:.84rem; }
section { padding:64px 0 8px; }
.sec-head { display:flex; align-items:flex-end; gap:16px; margin-bottom:22px; }
.sec-head h2 { font-size:1.6rem; } .sec-head p { margin:6px 0 0; color:var(--text-2); }
.kicker { font-family:var(--mono); font-size:.74rem; color:var(--accent); text-transform:uppercase; letter-spacing:.1em; }
/* runs */
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:18px; }
.run { display:block; background:var(--surface); border:1px solid var(--line); border-radius:16px; overflow:hidden; color:inherit;
  transition:border-color .15s, transform .15s, box-shadow .15s; position:relative; }
.run::before { content:""; position:absolute; left:0; right:0; top:0; height:3px; background:var(--band); z-index:1; }
.run:hover { border-color:var(--line-2); transform:translateY(-3px); box-shadow:0 18px 40px rgba(0,0,0,.18); }
.thumb { position:relative; aspect-ratio:16/9; background:#000; overflow:hidden; }
.thumb img, .thumb video { width:100%; height:100%; object-fit:cover; display:block; transition:transform .4s; }
.thumb video.composite { transform:scale(2.06); transform-origin:0 13%; }
.run:hover .thumb img { transform:scale(1.04); }
.thumb .dur { position:absolute; right:10px; bottom:10px; font-family:var(--mono); font-size:.74rem; color:#fff; background:rgba(0,0,0,.6);
  border-radius:6px; padding:1px 7px; }
.thumb .ph { height:100%; display:grid; place-items:center; color:var(--text-3); font-size:.8rem; background:var(--surface-2); }
.run .body { padding:16px 18px 16px; }
.row1 { display:flex; align-items:center; gap:10px; justify-content:space-between; }
.clip { font-family:var(--display); font-weight:700; font-size:1.15rem; }
.facts { display:flex; gap:16px; color:var(--text-2); font-size:.88rem; margin:10px 0 12px; }
.facts b { color:var(--text); font-variant-numeric:tabular-nums; }
.run .foot { display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--line); padding-top:12px; font-size:.82rem; }
.id { font-family:var(--mono); color:var(--text-3); font-size:.76rem; } .open { color:var(--accent); font-weight:600; }
.pill { display:inline-flex; align-items:center; gap:6px; border-radius:999px; padding:2px 10px; font-size:.75rem; font-weight:600;
  border:1px solid var(--line-2); white-space:nowrap; }
.pill::before { content:""; width:7px; height:7px; border-radius:50%; }
.pill.very-high::before { background:var(--critical); } .pill.high::before { background:var(--serious); }
.pill.medium::before { background:var(--warning); } .pill.low::before { background:var(--good); }
/* how it works */
.steps { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; counter-reset:step; }
.step { position:relative; background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:20px 18px 18px; }
.step::before { counter-increment:step; content:"0" counter(step); font-family:var(--mono); font-size:.74rem; color:var(--text-3); }
.step .ic { width:38px; height:38px; border-radius:10px; display:grid; place-items:center; background:var(--accent-soft); color:var(--accent); margin:10px 0 14px; }
.step h3 { font-size:1.05rem; margin-bottom:6px; }
.step p { margin:0; color:var(--text-2); font-size:.89rem; }
.step .tech { margin-top:12px; display:flex; flex-wrap:wrap; gap:6px; }
.step .tech span { font-family:var(--mono); font-size:.7rem; color:var(--text-2); background:var(--surface-2); border:1px solid var(--line); border-radius:6px; padding:1px 7px; }
.trust { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:14px; margin-top:14px; }
.trust div { border:1px dashed var(--line-2); border-radius:14px; padding:14px 16px; color:var(--text-2); font-size:.88rem; }
.trust b { color:var(--text); display:block; margin-bottom:2px; }
footer { margin-top:64px; border-top:1px solid var(--line); padding:26px 0 40px; color:var(--text-3); font-size:.84rem; }
footer .wrap { display:flex; gap:10px 24px; flex-wrap:wrap; align-items:center; }
footer .links { margin-left:auto; display:flex; gap:18px; } footer a { color:var(--text-2); }
.empty { color:var(--text-3); }
@media (max-width:900px) {
  .hero .wrap { grid-template-columns:1fr; padding-top:48px; padding-bottom:64px; }
  .shot { transform:none; } .shot .t1 { left:10px; } .shot .t2 { right:10px; }
  .stats { grid-template-columns:repeat(2, minmax(0,1fr)); } .stats div:nth-child(3) { border-left:none; }
  .stats div:nth-child(n+3) { border-top:1px solid var(--line); }
  .steps, .trust { grid-template-columns:1fr 1fr; }
  .nav .links a.hide-sm { display:none; }
}
@media (max-width:560px) { .steps, .trust { grid-template-columns:1fr; } .wrap { padding:0 16px; } }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } .shot { transform:none; } }
</style></head><body>
<nav class="nav"><div class="wrap">
  <a class="logo" href="/" aria-label="Ground Truth home">{wordmark}</a>
  <div class="links"><a class="hide-sm" href="#runs">Runs</a><a class="hide-sm" href="#how">How it works</a>
    <a class="hide-sm" href="/docs">API</a><a class="btn" href="{github}">GitHub</a></div>
</div></nav>
<header class="hero"><div class="wrap">
  <div>
    <span class="eyebrow"><i></i>Live demo · {runs_n} processed site videos</span>
    <h1>Every hazard on site, <span>and why it fired.</span></h1>
    <p class="lead">Fixed construction-camera footage turned into confirmed incidents with evidence clips,
    a 3D digital twin, a 5×5 risk register and an assistant that cites the run for every claim.</p>
    <div class="cta"><a class="btn primary" href="{featured_url}">Open the {featured_name} report →</a>
      <a class="btn" href="#how">How it works</a></div>
  </div>
  {featured_shot}
</div></header>
<div class="wrap"><div class="stats">
  <div><b>{runs_n}</b><span>site videos processed</span></div>
  <div><b>{minutes}</b><span>minutes of footage analysed</span></div>
  <div><b>{incidents}</b><span>incidents confirmed</span></div>
  <div><b>8</b><span>hazard rules, R1 to R8</span></div>
</div></div>
<main class="wrap">
<section id="runs">
  <div class="sec-head"><div><div class="kicker">Reports</div><h2>Processed runs</h2>
  <p>Each opens a full report: replay, digital twin, incident inbox, risk register and the assistant.</p></div></div>
  <div class="grid">{cards}</div>
</section>
<section id="how">
  <div class="sec-head"><div><div class="kicker">Pipeline</div><h2>How it works</h2>
  <p>From raw CCTV to a checked answer, on one fixed camera, without face recognition.</p></div></div>
  <div class="steps">
    <div class="step"><div class="ic"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 9h4v4H7z"/></svg></div>
      <h3>Detect</h3><p>Workers, helmets, vests, machinery, crane loads, fire and smoke in every frame; faces pixelated.</p>
      <div class="tech"><span>YOLOE-26</span><span>PPE classifier</span></div></div>
    <div class="step"><div class="ic"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z"/><path d="M4 7.5 12 12l8-4.5M12 12v9"/></svg></div>
      <h3>Track and map</h3><p>Each worker followed through occlusion and placed on a ground plane calibrated from the workers themselves.</p>
      <div class="tech"><span>ByteTrack</span><span>3D twin</span></div></div>
    <div class="step"><div class="ic"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 4.5 6v6c0 4.4 3.2 8 7.5 9 4.3-1 7.5-4.6 7.5-9V6z"/><path d="m9 12 2 2 4-4"/></svg></div>
      <h3>Assess</h3><p>Eight hazard rules with debounce and evidence grades, leading indicators, and a likelihood × severity register.</p>
      <div class="tech"><span>R1–R8</span><span>5×5 matrix</span></div></div>
    <div class="step"><div class="ic"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/><path d="M8.5 11h7M8.5 14h4"/></svg></div>
      <h3>Explain</h3><p>A "why it fired" trace per incident, and an assistant whose every number is checked against the run.</p>
      <div class="tech"><span>{llm}</span><span>cited facts</span></div></div>
  </div>
  <div class="trust">
    <div><b>Checked, not generated</b>Counts, times and rankings are computed from the run; the model only words them, and a failed check falls back to the run data.</div>
    <div><b>Privacy by default</b>Faces are pixelated everywhere; workers are per-clip track numbers, never identities.</div>
    <div><b>Honest coverage</b>Every rule reports alert, clear, inconclusive or not applicable, so "not checked" never reads as "safe".</div>
  </div>
</section>
</main>
<footer><div class="wrap"><span><b>Ground Truth</b>: what is actually true on the ground, checked against the footage. CS3235 Team 5. Heuristic triage from video, not legal advice or certified measurement.</span>
<div class="links"><a href="/docs">API</a><a href="/api/stats">Assistant monitoring</a><a href="/healthz">Status</a><a href="{github}">GitHub</a></div></div></footer>
</body></html>"""


def gallery_page() -> str:
    listed = runs()
    cards = "".join(_card(r) for r in listed) or '<p class="empty">No runs published yet.</p>'
    featured = max(listed, key=lambda r: (r["incidents"], r["duration_s"]), default=None)
    shot = ""
    if featured and featured["thumbnail"] and ".mp4" not in featured["thumbnail"]:
        top = featured["top_risk"]
        risk_tag = (
            f'<div class="tag t2"><span class="dot" style="background:var(--'
            f'{BAND_SLOT.get(top["band"], "good")})"></span>{escape(top["rule"])} · '
            f"{escape(top['band'])} <b>{top['rating']}/25</b></div>"
            if top
            else ""
        )
        shot = (
            f'<div class="shot"><img src="{escape(featured["thumbnail"])}" alt="Camera view with '
            f'the configured zones">{risk_tag}<div class="tag t1"><b>{featured["incidents"]}</b> '
            "incidents, each with an evidence clip</div></div>"
        )
    values = {
        "fonts": FONTS.replace("&", "&amp;"),
        "favicon": brand.FAVICON,
        "wordmark": brand.wordmark(30),
        "github": GITHUB,
        "runs_n": str(len(listed)),
        "minutes": f"{sum(r['duration_s'] for r in listed) / 60:.1f}",
        "incidents": str(sum(r["incidents"] for r in listed)),
        "featured_url": escape(featured["report"]) if featured else "#runs",
        "featured_name": escape(
            f"clip {Path(PureWindowsPath(featured['clip']).name).stem}" if featured else "latest"
        ),
        "featured_shot": shot,
        "llm": escape(provider.model if provider else "run data"),
        "cards": cards,
    }
    page = GALLERY
    for key, value in values.items():
        page = page.replace("{" + key + "}", value)
    return page
