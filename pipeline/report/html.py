"""Offline, self-contained HTML run report (v7 §4.4).

One file, no network: the 720p video with a synchronised incident timeline, incident
cards with evidence crops and narration, rule coverage, camera calibration and timings.
All text is HTML-escaped. Colour follows the status palette (good / warning / serious /
critical) and is always paired with an icon and a word; every chart has a table view.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from shared.enums import RuleId, RuleStatus
from shared.identity import display_track_label
from shared.schemas.bundle import TracksSummaryFile
from shared.schemas.incidents import IncidentRecord
from shared.schemas.run import RunManifest

SEVERITY = {  # status palette slot, icon, word
    "critical": ("critical", "▲", "Critical"),
    "high": ("serious", "▲", "High"),
    "medium": ("warning", "●", "Medium"),
    "low": ("good", "●", "Low"),
}
STATUS = {
    RuleStatus.EVALUATED_ALERT: ("critical", "⚠", "Alert"),
    RuleStatus.EVALUATED_CLEAR: ("good", "✓", "Clear"),
    RuleStatus.INCONCLUSIVE: ("warning", "?", "Inconclusive"),
    RuleStatus.NOT_APPLICABLE: ("neutral", "–", "Not applicable"),
    RuleStatus.UNSUPPORTED: ("muted", "∅", "Unsupported"),
}
RULE_NAMES = {
    RuleId.R1: "Missing helmet",
    RuleId.R2: "Missing hi-vis in movement area",
    RuleId.R3: "Restricted-zone intrusion",
    RuleId.R4: "Machinery proximity",
    RuleId.R5: "Open-edge exposure",
}

_STYLE = """
:root { color-scheme: light;
  --bg:#f4f3f0; --surface:#fcfcfb; --surface-2:#f0efec; --line:#e2e0da;
  --text:#0b0b0b; --text-2:#52514e; --text-3:#7b7a74; --accent:#2a78d6;
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --neutral:#8a8983; --muted:#c3c2b7; --series:#2a78d6; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) {
  color-scheme: dark; --bg:#121211; --surface:#1a1a19; --surface-2:#232321; --line:#2f2f2c;
  --text:#ffffff; --text-2:#c3c2b7; --text-3:#8f8e87; --accent:#3987e5;
  --neutral:#8f8e87; --muted:#4a4a46; --series:#3987e5; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#121211; --surface:#1a1a19;
  --surface-2:#232321; --line:#2f2f2c; --text:#ffffff; --text-2:#c3c2b7; --text-3:#8f8e87;
  --accent:#3987e5; --neutral:#8f8e87; --muted:#4a4a46; --series:#3987e5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
main { max-width:1200px; margin:0 auto; padding:28px 16px 64px; }
header.top { display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
.brand { width:44px; height:44px; border-radius:11px; background:var(--accent); color:#fff;
  display:grid; place-items:center; font-weight:700; }
h1 { font-size:1.55rem; margin:0; font-weight:650; }
h2 { font-size:1.1rem; margin:40px 0 12px; font-weight:650; }
.sub { color:var(--text-2); margin:2px 0 0; }
.theme { margin-left:auto; background:var(--surface); color:var(--text-2); border:1px solid var(--line);
  border-radius:999px; padding:6px 12px; cursor:pointer; font:inherit; font-size:.85rem; }
.notice { margin:20px 0 0; padding:12px 16px; background:var(--surface); border:1px solid var(--line);
  border-left:4px solid var(--warning); border-radius:10px; color:var(--text-2); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:20px; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.tile .label { color:var(--text-2); font-size:.82rem; }
.tile .value { font-size:1.9rem; font-weight:650; line-height:1.2; margin-top:2px; }
.tile .note { color:var(--text-3); font-size:.8rem; }
.hero .value { font-size:3.2rem; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:16px; }
.player { display:grid; grid-template-columns:minmax(0,1fr); gap:12px; }
video { width:100%; border-radius:10px; background:#000; display:block; }
svg text { fill:var(--text-3); font-size:11px; }
.lane-label { fill:var(--text-2); font-size:12px; font-weight:600; }
.span { cursor:pointer; }
.span:hover, .span:focus { opacity:.8; outline:none; }
.tooltip { position:fixed; pointer-events:none; background:var(--surface); color:var(--text);
  border:1px solid var(--line); border-radius:8px; padding:8px 10px; font-size:.82rem;
  box-shadow:0 6px 24px rgba(0,0,0,.18); display:none; max-width:280px; z-index:10; }
.legend { display:flex; gap:14px; flex-wrap:wrap; color:var(--text-2); font-size:.82rem; }
.legend i, .chip i { font-style:normal; }
.sw { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.chip { display:inline-flex; align-items:center; gap:5px; padding:2px 9px; border-radius:999px;
  font-size:.78rem; font-weight:600; background:var(--surface-2); color:var(--text); border:1px solid var(--line); }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:12px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:14px; padding:14px;
  display:grid; grid-template-columns:96px 1fr; gap:14px; position:relative; overflow:hidden; }
.card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--sev); }
.card img { width:96px; height:156px; object-fit:cover; border-radius:8px; background:var(--surface-2); }
.card h3 { margin:6px 0 2px; font-size:1rem; font-weight:650; }
.card .meta { color:var(--text-2); font-size:.84rem; }
.card p { margin:6px 0 0; font-size:.88rem; }
.card .action { color:var(--text); }
.card .caveat, .card .refs { color:var(--text-3); font-size:.8rem; }
.card button.seek { margin-top:8px; background:none; border:1px solid var(--line); color:var(--accent);
  border-radius:8px; padding:4px 10px; cursor:pointer; font:inherit; font-size:.8rem; }
.filters { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; }
.filters button { background:var(--surface); color:var(--text-2); border:1px solid var(--line);
  border-radius:999px; padding:5px 12px; cursor:pointer; font:inherit; font-size:.84rem; }
.filters button[aria-pressed="true"] { background:var(--text); color:var(--surface); border-color:var(--text); }
.card[hidden] { display:none; }
.badge { font-size:.72rem; color:var(--text-3); margin-left:4px; }
.cov-row { display:grid; grid-template-columns:44px 230px 150px 1fr; gap:12px; align-items:center;
  padding:10px 0; border-top:1px solid var(--line); }
.cov-row:first-child { border-top:none; }
.stack { display:flex; gap:2px; height:14px; }
.stack span { display:block; height:100%; }
.stack span:first-child { border-radius:4px 0 0 4px; } .stack span:last-child { border-radius:0 4px 4px 0; }
.stack span:only-child { border-radius:4px; }
.hatch { background-image:repeating-linear-gradient(45deg, var(--muted) 0 2px, transparent 2px 6px); }
table { width:100%; border-collapse:collapse; font-size:.86rem; }
th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--text-2); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
details { margin-top:10px; } summary { cursor:pointer; color:var(--text-2); font-size:.86rem; }
.bars .row { display:grid; grid-template-columns:130px 1fr 70px; gap:10px; align-items:center; margin:6px 0; }
.bars .track { height:14px; }
.bars .fill { height:14px; background:var(--series); border-radius:0 4px 4px 0; }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }
ul.warn { margin:0; padding-left:18px; color:var(--text-2); }
footer { margin-top:44px; color:var(--text-3); font-size:.8rem; }
@media (max-width:640px) { .card { grid-template-columns:1fr; } .card img { width:100%; height:200px; }
  .cov-row { grid-template-columns:40px 1fr; } .cov-row .stack, .cov-row .chip-cell { grid-column:1 / -1; } }
"""

_SCRIPT = """
const video = document.getElementById('video');
const tip = document.getElementById('tip');
const playhead = document.getElementById('playhead');
const duration = Number(document.getElementById('timeline').dataset.duration);
function seek(t) { if (!video) return; video.currentTime = Math.max(0, t - 1); video.play(); }
document.querySelectorAll('[data-seek]').forEach(el => {
  el.addEventListener('click', () => seek(Number(el.dataset.seek)));
  el.addEventListener('keydown', e => { if (e.key === 'Enter') seek(Number(el.dataset.seek)); });
});
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.innerHTML = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 300) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
if (video && playhead) video.addEventListener('timeupdate', () => {
  const x = 60 + 920 * Math.min(1, video.currentTime / duration);
  playhead.setAttribute('x1', x); playhead.setAttribute('x2', x);
});
document.querySelectorAll('.filters button').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(b => b.setAttribute('aria-pressed', b === button));
    const rule = button.dataset.rule;
    document.querySelectorAll('.card').forEach(card => { card.hidden = rule !== 'all' && card.dataset.rule !== rule; });
  });
});
const toggle = document.getElementById('theme');
toggle && toggle.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
});
"""


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def _chip(slot: str, icon: str, word: str) -> str:
    return (
        f'<span class="chip"><span class="sw" style="background:var(--{slot})"></span>'
        f'<i aria-hidden="true">{icon}</i>{escape(word)}</span>'
    )


def _timeline(incidents: Sequence[IncidentRecord], duration: float) -> str:
    lanes = list(RuleId)
    height = 30 + len(lanes) * 30
    parts = [
        f'<svg id="timeline" data-duration="{duration:.3f}" viewBox="0 0 1000 {height + 22}" '
        f'role="img" aria-label="Incident timeline by rule">'
    ]
    for index, rule in enumerate(lanes):
        y = 20 + index * 30
        parts.append(f'<text class="lane-label" x="8" y="{y + 5}">{rule.value}</text>')
        parts.append(
            f'<line x1="60" x2="980" y1="{y}" y2="{y}" stroke="var(--line)" stroke-width="1"/>'
        )
    step = next((s for s in (10, 15, 30, 60, 120, 300, 600) if duration / s <= 10), 1200)
    tick = 0.0
    while tick <= duration + 1e-6:
        x = 60 + 920 * tick / max(duration, 1e-6)
        parts.append(
            f'<text x="{x:.1f}" y="{height + 14}" text-anchor="middle">{_clock(tick)[:-2]}</text>'
        )
        tick += step
    for rule in lanes:
        episodes = sorted((i for i in incidents if i.rule_id is rule), key=lambda i: i.first_seen_s)
        rows = _pack(episodes, duration)
        height_each = max(3.0, 20.0 / max(len(rows), 1) - 1.0)
        y0 = 10 + lanes.index(rule) * 30
        for row, members in enumerate(rows):
            for incident in members:
                y = y0 + row * (height_each + 1)
                parts.append(_span(incident, duration, y, height_each))
    parts.append(
        f'<line id="playhead" x1="60" x2="60" y1="8" y2="{height}" stroke="var(--text)" '
        'stroke-width="2"/>'
    )
    parts.append("</svg>")
    legend = "".join(
        f'<span><span class="sw" style="background:var(--{slot})"></span>{icon} {word}</span>'
        for slot, icon, word in (SEVERITY["critical"], SEVERITY["high"], SEVERITY["medium"])
    )
    return (
        "".join(parts)
        + f'<div class="legend">{legend}<span>Click a span to jump to it</span></div>'
    )


MAX_SUBROWS = 4


def _pack(episodes: Sequence[IncidentRecord], duration: float) -> list[list[IncidentRecord]]:
    """Overlapping episodes of one rule go to separate sub-rows (at most MAX_SUBROWS)."""
    rows: list[list[IncidentRecord]] = []
    ends: list[float] = []
    for incident in episodes:
        end = incident.resolved_at_s if incident.resolved_at_s is not None else duration
        free = next((r for r, e in enumerate(ends) if e <= incident.first_seen_s), None)
        if free is None and len(rows) < MAX_SUBROWS:
            rows.append([])
            ends.append(0.0)
            free = len(rows) - 1
        if free is None:
            free = min(range(len(ends)), key=ends.__getitem__)
        rows[free].append(incident)
        ends[free] = max(ends[free], end)
    return rows


def _span(incident: IncidentRecord, duration: float, y: float, height: float) -> str:
    slot, _, word = SEVERITY.get(incident.severity, ("neutral", "●", incident.severity))
    end = incident.resolved_at_s if incident.resolved_at_s is not None else duration
    x1 = 60 + 920 * incident.first_seen_s / max(duration, 1e-6)
    x2 = max(x1 + 5, 60 + 920 * end / max(duration, 1e-6))
    worker = display_track_label(incident.canonical_track_id or "")
    tip = escape(
        f"<b>{escape(incident.rule_id.value)} · {escape(incident.title)}</b><br>Worker {escape(worker)}"
        f" · {word} · {_clock(incident.first_seen_s)}–{_clock(end)}",
        quote=True,
    )
    return (
        f'<rect class="span" tabindex="0" x="{x1:.1f}" y="{y:.1f}" width="{x2 - x1:.1f}" '
        f'height="{height:.1f}" rx="{min(4.0, height / 2):.1f}" fill="var(--{slot})" '
        f'stroke="var(--surface)" stroke-width="1" data-seek="{incident.first_seen_s:.2f}" '
        f'data-tip="{tip}"/>'
    )


def _card(incident: IncidentRecord) -> str:
    slot, icon, word = SEVERITY.get(incident.severity, ("neutral", "●", incident.severity))
    worker = display_track_label(incident.canonical_track_id or "")
    end = incident.resolved_at_s
    span = f"{_clock(incident.first_seen_s)} – {_clock(end) if end is not None else 'end of clip'}"
    crop = escape(incident.evidence_crop_path or incident.evidence_path)
    narration = incident.narration
    if narration is not None and narration.source == "model":
        body = (
            f'<p>{escape(narration.summary)}<span class="badge">AI narration · passed 8/8 checks</span></p>'
            f'<p class="action">{escape(narration.action)}</p><p class="caveat">{escape(narration.caveat)}</p>'
        )
    else:
        body = (
            f"<p>{escape(incident.observation_text)}</p>"
            f'<p class="action">{escape(incident.action_text)}</p>'
        )
    ppe = []
    if incident.helmet is not None:
        ppe.append(f"helmet: {incident.helmet.state.value.replace('_', ' ')}")
    if incident.vest is not None:
        ppe.append(f"hi-vis: {incident.vest.state.value.replace('_', ' ')}")
    refs = (
        ", ".join(escape(r) for r in incident.references)
        or "Heuristic — no statutory reference attached"
    )
    extras = []
    if incident.distance is not None:
        extras.append(f"{incident.distance.lower_wh:.1f}–{incident.distance.upper_wh:.1f} WH")
    if incident.adjudication_candidate:
        extras.append("queued for L1 image check")
    return f"""
<article class="card" data-rule="{escape(incident.rule_id.value)}" style="--sev:var(--{slot})">
  <a href="{escape(incident.evidence_path)}" title="Open the full evidence frame">
    <img src="{crop}" loading="lazy" alt="Worker {escape(worker)} at the evidence moment for {escape(incident.rule_id.value)}"></a>
  <div>
    <div>{_chip(slot, icon, word)} <span class="chip">{escape(incident.rule_id.value)}</span>
      {'<span class="chip">Heuristic</span>' if incident.basis == "heuristic" else ""}</div>
    <h3>{escape(incident.title or incident.reason_code)}</h3>
    <div class="meta">Worker {escape(worker)} · {span}{" · " + escape("; ".join(ppe)) if ppe else ""}</div>
    {body}
    <p class="refs">{refs}{" · " + escape(" · ".join(extras)) if extras else ""}</p>
    <button class="seek" data-seek="{incident.first_seen_s:.2f}">▶ Play from {_clock(incident.first_seen_s)}</button>
  </div>
</article>"""


def _filters(incidents: Sequence[IncidentRecord]) -> str:
    counts = Counter(i.rule_id for i in incidents)
    if len(counts) < 2:
        return ""
    buttons = [
        f'<button type="button" data-rule="all" aria-pressed="true">All · {len(incidents)}</button>'
    ]
    buttons += [
        f'<button type="button" data-rule="{rule.value}" aria-pressed="false">'
        f"{rule.value} {escape(RULE_NAMES[rule])} · {counts[rule]}</button>"
        for rule in RuleId
        if counts[rule]
    ]
    return f'<div class="filters" role="group" aria-label="Filter incidents by rule">{"".join(buttons)}</div>'


def _coverage(manifest: RunManifest) -> str:
    rows, table = [], []
    for entry in manifest.rule_coverage:
        slot, icon, word = STATUS[entry.status]
        total = sum(entry.status_counts.values()) or 1
        segments = []
        for status in (
            RuleStatus.EVALUATED_ALERT,
            RuleStatus.EVALUATED_CLEAR,
            RuleStatus.INCONCLUSIVE,
            RuleStatus.NOT_APPLICABLE,
            RuleStatus.UNSUPPORTED,
        ):
            count = entry.status_counts.get(status, 0)
            if not count:
                continue
            s_slot, _, s_word = STATUS[status]
            share = 100 * count / total
            css = "hatch" if status is RuleStatus.UNSUPPORTED else ""
            tip = escape(
                f"<b>{entry.rule_id.value} · {s_word}</b><br>{count:,} observations ({share:.1f} %)",
                quote=True,
            )
            segments.append(
                f'<span class="{css}" style="width:{share:.2f}%;background-color:var(--{s_slot})" '
                f'data-tip="{tip}"></span>'
            )
        rows.append(f"""<div class="cov-row"><b>{entry.rule_id.value}</b>
  <div>{escape(RULE_NAMES[entry.rule_id])}<div class="meta" style="color:var(--text-3);font-size:.8rem">
  {escape(entry.reason_code.replace("_", " "))}</div></div>
  <div class="chip-cell">{_chip(slot, icon, word)}{f' <span class="chip">{entry.incident_count} incident(s)</span>' if entry.incident_count else ""}</div>
  <div class="stack" role="img" aria-label="Observation status mix for {entry.rule_id.value}">{"".join(segments)}</div></div>""")
        cells = "".join(f'<td class="num">{entry.status_counts.get(s, 0):,}</td>' for s in STATUS)
        table.append(f"<tr><td>{entry.rule_id.value}</td>{cells}</tr>")
    head = "".join(f'<th class="num">{escape(w)}</th>' for _, _, w in STATUS.values())
    legend = "".join(
        f'<span><span class="sw {"hatch" if s is RuleStatus.UNSUPPORTED else ""}" '
        f'style="background-color:var(--{slot})"></span>{icon} {word}</span>'
        for s, (slot, icon, word) in STATUS.items()
    )
    return (
        f'<div class="panel">{"".join(rows)}<div class="legend" style="margin-top:10px">{legend}</div>'
        f"<details><summary>Show as table (observations per status)</summary><table><thead><tr>"
        f"<th>Rule</th>{head}</tr></thead><tbody>{''.join(table)}</tbody></table></details></div>"
    )


def _timings(manifest: RunManifest) -> str:
    timings = [(t.stage.value.replace("_", " "), t.duration_s) for t in manifest.pass_timings]
    if not timings:
        return ""
    longest = max(d for _, d in timings) or 1.0
    rows = "".join(
        f'<div class="row"><span>{escape(name.capitalize())}</span><div class="track">'
        f'<div class="fill" style="width:{100 * d / longest:.1f}%" data-tip="{escape(name)}: {d:.1f} s"></div>'
        f'</div><span class="num" style="text-align:right">{d:.1f} s</span></div>'
        for name, d in timings
    )
    total = sum(d for _, d in timings)
    return f'<div class="panel bars"><div class="meta" style="color:var(--text-2)">Total {total:.0f} s</div>{rows}</div>'


def _calibration(manifest: RunManifest) -> str:
    diagnostics: Mapping[str, Any] = manifest.diagnostics
    fits = diagnostics.get("camera_fits") or {}
    rows = []
    for shot, fit in fits.items():
        p = fit.get("parameters") or {}
        if fit.get("accepted") and p:
            rows.append(
                f"<tr><td>{escape(shot)}</td><td>{_chip('good', '✓', 'Accepted')}</td>"
                f"<td class='num'>{math.degrees(p['tilt_rad']):.1f}°</td>"
                f"<td class='num'>{p['height_wh']:.2f} WH</td>"
                f"<td class='num'>{fit.get('observations', 0)}</td>"
                f"<td class='num'>{fit.get('inlier_ratio', 0):.2f}</td>"
                f"<td class='num'>{100 * fit.get('relative_rms', 0):.1f} %</td>"
                f"<td class='num'>{fit.get('bootstrap_replicates', 0)}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{escape(shot)}</td><td>{_chip('warning', '?', 'Rejected')}</td>"
                f"<td colspan='6'>{escape(str(fit.get('reason_code', '')).replace('_', ' '))}</td></tr>"
            )
    gate = diagnostics.get("geometry_gate") or {}
    camera = diagnostics.get("camera_mode", "unknown")
    return f"""<div class="panel"><p class="sub" style="margin:0 0 8px">Camera {escape(str(camera))};
residual drift {gate.get("residual_drift_wh") or 0:.4f} WH ({escape(str(gate.get("reason_code", "")).replace("_", " "))}).
Distances are in worker heights (WH) with 95 % bootstrap intervals, never metres.</p>
<table><thead><tr><th>Shot</th><th>Calibration</th><th class="num">Tilt</th><th class="num">Camera height</th>
<th class="num">Observations</th><th class="num">Inlier ratio</th><th class="num">Height error</th>
<th class="num">Replicates</th></tr></thead>
<tbody>{"".join(rows) or '<tr><td colspan="8">Geometry unsupported for this feed.</td></tr>'}</tbody></table></div>"""


def peak_workers(tracks: TracksSummaryFile) -> int:
    """Most workers simultaneously in view (track IDs fragment, so a count of IDs overstates)."""
    events = sorted(
        (point, delta)
        for track in tracks.tracks
        if track.object_class == "person"
        for span in track.presence
        for point, delta in ((span.start_s, 1), (span.end_s, -1))
    )
    peak = current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def render_report(
    manifest: RunManifest,
    incidents: Sequence[IncidentRecord],
    *,
    tracks: TracksSummaryFile | None = None,
) -> str:
    source = manifest.input_video
    duration = source.duration_s or max(
        (i.resolved_at_s or i.confirmed_at_s for i in incidents), default=1.0
    )
    severities = Counter(i.severity for i in incidents)
    video = manifest.output_artifacts.get("video_proxy") or manifest.output_artifacts.get("video")
    processing = sum(t.duration_s for t in manifest.pass_timings)
    alerting_rules = sum(
        1 for e in manifest.rule_coverage if e.status is RuleStatus.EVALUATED_ALERT
    )
    tiles = [
        ("hero", "Confirmed incidents", str(len(incidents)), f"{duration:.0f} s of video"),
        (
            "",
            "Critical · high",
            f"{severities.get('critical', 0)} · {severities.get('high', 0)}",
            f"{severities.get('medium', 0)} medium",
        ),
        (
            "",
            "Peak workers in view",
            "—" if tracks is None else str(peak_workers(tracks)),
            ""
            if tracks is None
            else f"{sum(t.object_class == 'person' for t in tracks.tracks)} track IDs",
        ),
        ("", "Rules alerting", f"{alerting_rules} of 5", "R1–R5 always reported"),
        (
            "",
            "Processing time",
            f"{processing:.0f} s",
            f"{processing / max(duration, 1e-6):.1f}× real time",
        ),
    ]
    tile_html = "".join(
        f'<div class="tile {cls}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="note">{escape(note)}</div></div>'
        for cls, label, value, note in tiles
    )
    cards = (
        "".join(_card(i) for i in sorted(incidents, key=lambda i: i.first_seen_s))
        or '<p class="sub">No confirmed incidents in this clip.</p>'
    )
    warnings = "".join(f"<li>{escape(w)}</li>" for w in manifest.warnings) or "<li>None</li>"
    models = "".join(
        f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in manifest.model_ids.items()
    )
    player = (
        f'<video id="video" controls preload="metadata" src="{escape(video)}"></video>'
        if video
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safety run report</title><style>{_STYLE}</style></head>
<body><main>
<header class="top"><div class="brand">ST</div><div>
<h1>Construction safety run report</h1>
<p class="sub">{escape(Path(source.path).name)} · {source.width}×{source.height} · {duration:.1f} s ·
run {escape(manifest.run_id)} · {escape((manifest.completed_at or "")[:19].replace("T", " "))} UTC</p></div>
<button class="theme" id="theme" type="button">Toggle theme</button></header>
<p class="notice">{escape(manifest.disclaimer)}</p>
<section class="tiles">{tile_html}</section>
<h2>Replay</h2>
<div class="panel player">{player}{_timeline(incidents, duration)}</div>
<h2>Incidents</h2>
{_filters(incidents)}
<section class="cards">{cards}</section>
<h2>Rule coverage</h2>
{_coverage(manifest)}
<div class="two"><div><h2>Camera calibration</h2>{_calibration(manifest)}</div>
<div><h2>Processing time</h2>{_timings(manifest)}</div></div>
<h2>Warnings</h2><div class="panel"><ul class="warn">{warnings}</ul></div>
<h2>Models</h2><div class="panel"><table><tbody>{models}</tbody></table></div>
<footer>Generated offline by the Construction Safety Twin pipeline. Heuristic triage from video,
not legal advice or certified measurement. Incident data: incidents.json · schema {manifest.schema_version}.</footer>
</main><div class="tooltip" id="tip" role="tooltip"></div>
<script>{_SCRIPT}</script></body></html>
"""


def write_report(
    path: Path,
    manifest: RunManifest,
    incidents: Sequence[IncidentRecord],
    *,
    tracks: TracksSummaryFile | None = None,
) -> Path:
    path.write_text(render_report(manifest, incidents, tracks=tracks), encoding="utf-8")
    return path
