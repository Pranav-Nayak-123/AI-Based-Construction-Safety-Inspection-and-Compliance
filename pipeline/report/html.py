"""Offline, self-contained HTML run report (v7 §4.4). All text is HTML-escaped."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path

from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord
from shared.schemas.run import RunManifest

_STYLE = """
:root { color-scheme: light dark; --fg:#1d1b19; --muted:#6b645c; --bg:#faf8f5; --card:#fff;
  --line:#e4ded6; --critical:#8e1b1b; --high:#c0392b; --medium:#b9770e; --low:#7d8c2a; }
@media (prefers-color-scheme: dark) { :root { --fg:#ece8e3; --muted:#a39c93; --bg:#1b1917;
  --card:#25221f; --line:#3a3530; } }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 system-ui, sans-serif; color:var(--fg); background:var(--bg); }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.6rem; margin:0 0 4px; } h2 { margin-top:32px; font-size:1.15rem; }
.muted { color:var(--muted); }
.disclaimer { border-left:4px solid var(--medium); padding:8px 12px; background:var(--card); }
table { width:100%; border-collapse:collapse; background:var(--card); }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
  vertical-align:top; }
.card { display:grid; grid-template-columns:160px 1fr; gap:16px; background:var(--card);
  border:1px solid var(--line); border-left:6px solid var(--muted); padding:14px; margin:12px 0; }
.card img { width:100%; border-radius:4px; }
.critical { border-left-color:var(--critical); } .high { border-left-color:var(--high); }
.medium { border-left-color:var(--medium); } .low { border-left-color:var(--low); }
.tag { display:inline-block; font-size:.8rem; padding:1px 8px; border:1px solid var(--line);
  border-radius:10px; margin-right:6px; }
video { width:100%; background:#000; }
@media (max-width:640px) { .card { grid-template-columns:1fr; } }
"""


def _incident_card(incident: IncidentRecord) -> str:
    duration = ""
    end = incident.resolved_at_s
    if end is not None:
        duration = f"{end - incident.first_seen_s:.1f} s"
    crop = incident.evidence_crop_path or incident.evidence_path
    references = ", ".join(escape(ref) for ref in incident.references) or "none (heuristic)"
    ppe = []
    if incident.helmet is not None:
        ppe.append(f"helmet: {incident.helmet.state.value} ({incident.helmet.confidence:.2f})")
    if incident.vest is not None:
        ppe.append(f"vest: {incident.vest.state.value} ({incident.vest.confidence:.2f})")
    return f"""
<article class="card {escape(incident.severity)}">
  <a href="{escape(incident.evidence_path)}"><img src="{escape(crop)}"
     alt="Evidence for {escape(incident.rule_id.value)}, worker
     {escape(display_track_label(incident.canonical_track_id or ""))}"></a>
  <div>
    <div><span class="tag">{escape(incident.rule_id.value)}</span>
      <span class="tag">{escape(incident.severity)}</span>
      <span class="tag">{escape(incident.basis)}</span>
      {'<span class="tag">L1 adjudication</span>' if incident.adjudication_candidate else ""}</div>
    <p><strong>{escape(incident.observation_text)}</strong></p>
    <p>Action: {escape(incident.action_text)}</p>
    <p class="muted">First seen {incident.first_seen_s:.1f} s, confirmed
      {incident.confirmed_at_s:.1f} s{", lasted " + duration if duration else ""}.
      References: {references}. {escape("; ".join(ppe))}</p>
  </div>
</article>"""


def render_report(manifest: RunManifest, incidents: Sequence[IncidentRecord]) -> str:
    coverage_rows = "".join(
        f"<tr><td>{escape(entry.rule_id.value)}</td><td>{escape(entry.status.value)}</td>"
        f"<td>{escape(entry.reason_code)}</td><td>{entry.incident_count}</td>"
        f"<td class='muted'>"
        + escape(", ".join(f"{s.value} {n}" for s, n in entry.status_counts.items()))
        + "</td></tr>"
        for entry in manifest.rule_coverage
    )
    timing_rows = "".join(
        f"<tr><td>{escape(t.stage.value)}</td><td>{t.duration_s:.1f} s</td></tr>"
        for t in manifest.pass_timings
    )
    warnings = "".join(f"<li>{escape(warning)}</li>" for warning in manifest.warnings)
    cards = "".join(_incident_card(incident) for incident in incidents) or (
        "<p class='muted'>No confirmed incidents.</p>"
    )
    video = manifest.output_artifacts.get("video_proxy") or manifest.output_artifacts.get("video")
    player = f'<video controls preload="metadata" src="{escape(video)}"></video>' if video else ""
    source = manifest.input_video
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safety run report</title><style>{_STYLE}</style></head>
<body><main>
<h1>Construction safety run report</h1>
<p class="muted">Run {escape(manifest.run_id)} - {escape(Path(source.path).name)}
 ({source.width}x{source.height}, {source.duration_s or 0:.1f} s) - completed
 {escape(manifest.completed_at or "")}</p>
<p class="disclaimer">{escape(manifest.disclaimer)}</p>
{player}
<h2>Incidents ({len(incidents)})</h2>{cards}
<h2>Rule coverage</h2>
<table><thead><tr><th>Rule</th><th>Status</th><th>Reason</th><th>Incidents</th>
<th>Observations</th></tr></thead><tbody>{coverage_rows}</tbody></table>
<h2>Warnings</h2><ul>{warnings or "<li>None</li>"}</ul>
<h2>Processing time</h2><table><tbody>{timing_rows}</tbody></table>
</main></body></html>
"""


def write_report(path: Path, manifest: RunManifest, incidents: Sequence[IncidentRecord]) -> Path:
    path.write_text(render_report(manifest, incidents), encoding="utf-8")
    return path
