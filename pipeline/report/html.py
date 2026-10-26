"""Offline, self-contained HTML run report (v7 §4.4; v3 layout).

One file, no network, no framework. A sticky top bar and a sidebar of seven sections:

* Overview — headline numbers, the 720p replay with a per-rule timeline, top priorities;
* Incidents — an inbox: the triaged list on the left, the selected incident on the right
  (evidence clip, why it fired, facts, action) with review buttons whose verdicts can be
  exported for `tools/review_incidents.py --import`;
* Risk assessment — the 5×5 risk register and matrix (the safety assistant is a drawer
  opened from the top bar on every section);
* Leading indicators — exposure, brief exposures, hotspots;
* Site & camera — zone health and calibration;
* Run details — rule coverage, timings, models, warnings, privacy and oversight.

All text is HTML-escaped. Colour follows the status palette (good / warning / serious /
critical) and is always paired with an icon and a word; every chart has a table view.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from pipeline.output.facts import build_facts
from pipeline.output.run_bundle import RunBundle
from pipeline.report import brand, risk_html, twin_viewer
from pipeline.report import insights_html as v3
from pipeline.report.assets import SCRIPT, STYLE
from pipeline.rules.catalogue import load_catalogue
from shared.enums import RuleId, RuleStatus
from shared.identity import display_track_label
from shared.schemas.bundle import TracksSummaryFile
from shared.schemas.incidents import IncidentRecord
from shared.schemas.insights import InsightsFile
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
    RuleId.R6: "Worker down",
    RuleId.R7: "Under a suspended load",
    RuleId.R8: "Fire or smoke",
}
FALSE_REASONS = (
    ("detection", "Not a person"),
    ("ppe", "PPE misread"),
    ("zone_or_distance", "Zone / distance wrong"),
)
MAX_SUBROWS = 4
# 24x24 line icons (stroke = currentColor), drawn for this report.
_ICON_PATHS = {
    "overview": '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/>'
    '<rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>',
    "twin": '<path d="M12 2.5 3.5 7v10L12 21.5 20.5 17V7z"/><path d="M3.5 7 12 11.5 20.5 7M12 11.5v10"/>',
    "incidents": '<path d="M10.3 3.9 2.4 17.6A2 2 0 0 0 4.1 20.6h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
    '<path d="M12 9v4M12 17h.01"/>',
    "indicators": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "site": '<path d="M3 7h3l2-3h8l2 3h3v12H3z"/><circle cx="12" cy="13" r="3.5"/>',
    "run": '<path d="M4 6h16M4 12h16M4 18h10"/>',
    "risk": '<path d="M12 3 4.5 6v6c0 4.4 3.2 8 7.5 9 4.3-1 7.5-4.6 7.5-9V6z"/><path d="m9 12 2 2 4-4"/>',
}


# Display, body and mono faces; offline, the system fonts in the stylesheet take over.
FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&amp;family=JetBrains+Mono:wght@400;500&amp;family=Space+Grotesk:wght@500;600;700&amp;display=swap"
)
_SVG = (
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
)
SPARK_ICON = (
    _SVG
    + '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/></svg>'
)
THEME_ICON = (
    _SVG
    + '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z" fill="currentColor"/></svg>'
)
DOWNLOAD_ICON = _SVG + '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>'


def icon(name: str) -> str:
    return (
        '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>'
    )


NAV = (
    ("overview", "Overview"),
    ("twin", "Digital twin"),
    ("incidents", "Incidents"),
    ("risk", "Risk assessment"),
    ("indicators", "Leading indicators"),
    ("site", "Site & camera"),
    ("run", "Run details"),
)


def subject(incident: IncidentRecord) -> str:
    """Who or what the incident is about: a worker, or the fire/smoke itself (R8)."""
    label = display_track_label(incident.canonical_track_id or "")
    if incident.rule_id is RuleId.R8:
        return f"Fire/smoke {label}"
    return f"Worker {label}"


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def _chip(slot: str, icon: str, word: str) -> str:
    return (
        f'<span class="chip"><span class="sw" style="background:var(--{slot})"></span>'
        f'<i aria-hidden="true">{icon}</i>{escape(word)}</span>'
    )


def _card(title: str, body: str, hint: str = "") -> str:
    hint_html = f'<span class="hint">{hint}</span>' if hint else ""
    return (
        f'<div class="card"><div class="card-head"><h2>{escape(title)}</h2>{hint_html}</div>'
        f'<div class="card-body">{body}</div></div>'
    )


SLOT_HEX = {"critical": "#d03b3b", "serious": "#ec835a", "warning": "#fab219", "good": "#0ca30c"}


def _twin_incidents(incidents: Sequence[IncidentRecord]) -> list[dict]:
    """Incident markers for the 3D twin's timeline (plain data, escaped at embed time)."""
    rows = []
    for incident in incidents:
        slot = SEVERITY.get(incident.severity, ("neutral", "", ""))[0]
        rows.append(
            {
                "id": incident.incident_id,
                "rule": incident.rule_id.value,
                "subject": incident.canonical_track_id,
                "start": round(incident.first_seen_s, 2),
                "end": round(incident.resolved_at_s, 2) if incident.resolved_at_s else None,
                "title": incident.title,
                "color": SLOT_HEX.get(slot, "#8a8983"),
            }
        )
    return rows


def _page_head(title: str, text: str, *, actions: str = "") -> str:
    actions_html = f'<div class="actions">{actions}</div>' if actions else ""
    return (
        f'<div class="page-head"><div><h1>{escape(title)}</h1><p>{escape(text)}</p></div>'
        f"{actions_html}</div>"
    )


BAND_SLOT = {"very high": "critical", "high": "serious", "medium": "warning", "low": "good"}


def _verdict(bundle: RunBundle, clip: str, duration: float, groups: Counter) -> str:
    """The answer to "is everything okay?", before any detail."""
    risk = bundle.risk
    top = risk.rows[0] if risk.rows else None
    n = len(bundle.incidents)
    eyebrow = f"{clip} · {_clock(duration)[:-2]} of fixed-camera video · eight hazard rules"
    if n and top:
        headline = (
            f"{n} incident{'s' if n != 1 else ''}. Highest risk: "
            f"<em>{escape(top.title.lower())}</em>, {escape(top.band)} ({top.rating}/25)."
        )
        slot = BAND_SLOT[top.band]
    elif top:
        headline = (
            f"No confirmed incidents. Watch <em>{escape(top.title.lower())}</em>: "
            f"{escape(top.band)} risk from brief exposures ({top.rating}/25)."
        )
        slot = BAND_SLOT[top.band]
    else:
        headline = "No confirmed incidents and no hazard condition held in this clip."
        slot = "good"
    parts = []
    if n:
        parts.append(f"{groups['act']} to act on now, {groups['verify']} to verify first.")
    if risk.not_assessed:
        rules = ", ".join(r.value for r, _, _ in risk.not_assessed)
        parts.append(f"Not assessed in this clip: {rules}, which is not the same as no risk.")
    actions = ('<a class="btn primary" href="#incidents">Review incidents</a>' if n else "") + (
        '<button class="btn" type="button" data-ask-open>Ask the assistant</button>'
        '<a class="btn ghost" href="#risk">Risk register →</a>'
    )
    return (
        f'<div class="hero" style="--sev:var(--{slot})"><div class="eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{headline}</h1><p>{escape(' '.join(parts))}</p>"
        f'<div class="cta">{actions}</div></div>'
    )


def _kpis(tiles: Sequence[tuple[str, str, str]]) -> str:
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>'
            f'<div class="note">{escape(note)}</div></div>'
            for label, value, note in tiles
        )
        + "</div>"
    )


# ---------------------------------------------------------------------- timeline


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
                parts.append(_span(incident, duration, y0 + row * (height_each + 1), height_each))
    parts.append(
        f'<line id="playhead" x1="60" x2="60" y1="8" y2="{height}" stroke="var(--text)" '
        'stroke-width="2"/></svg>'
    )
    legend = "".join(
        f'<span><span class="sw" style="background:var(--{slot})"></span>{icon} {word}</span>'
        for slot, icon, word in (SEVERITY["critical"], SEVERITY["high"], SEVERITY["medium"])
    )
    return (
        "".join(parts)
        + f'<div class="legend">{legend}<span class="muted">Click a span to jump there</span></div>'
    )


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
    tip = escape(
        f"<b>{escape(incident.rule_id.value)} · {escape(incident.title)}</b><br>{escape(subject(incident))}"
        f" · {word} · {_clock(incident.first_seen_s)}–{_clock(end)}",
        quote=True,
    )
    return (
        f'<rect class="span" tabindex="0" x="{x1:.1f}" y="{y:.1f}" width="{x2 - x1:.1f}" '
        f'height="{height:.1f}" rx="{min(4.0, height / 2):.1f}" fill="var(--{slot})" '
        f'stroke="var(--surface)" stroke-width="1" data-seek="{incident.first_seen_s:.2f}" '
        f'data-tip="{tip}"/>'
    )


# ---------------------------------------------------------------------- incidents


def _row(incident: IncidentRecord, duration: float) -> str:
    """One line under its rule's heading: who, when, for how long, how well evidenced."""
    slot, _, word = SEVERITY.get(incident.severity, ("neutral", "●", incident.severity))
    end = incident.resolved_at_s if incident.resolved_at_s is not None else duration
    return (
        f'<div class="row" role="option" tabindex="0" aria-selected="false" '
        f'data-id="{escape(incident.incident_id)}" data-rule="{escape(incident.rule_id.value)}" '
        f'data-group="{v3.group(incident)}" style="--sev:var(--{slot})" '
        f'aria-label="{escape(incident.title)}, {escape(subject(incident))}">'
        f'<div class="bar" aria-hidden="true"></div><div>'
        f'<div class="top"><span class="t">{escape(subject(incident))}</span>'
        f'<span class="verdict"></span><span class="tm">{_clock(incident.first_seen_s)}</span></div>'
        f'<div class="m"><span>{escape(word)} · {max(end - incident.first_seen_s, 0):.1f} s</span>'
        f"{v3.quality_dots(incident)}</div></div></div>"
    )


def _detail(incident: IncidentRecord, *, near_wh: float, caution_wh: float) -> str:
    slot, icon, word = SEVERITY.get(incident.severity, ("neutral", "●", incident.severity))
    end = incident.resolved_at_s
    span = f"{_clock(incident.first_seen_s)} – {_clock(end) if end is not None else 'end of clip'}"
    duration = (end if end is not None else incident.confirmed_at_s) - incident.first_seen_s
    crop = escape(incident.evidence_crop_path or incident.evidence_path)
    alt = f"{subject(incident)} at the evidence moment for {incident.rule_id.value}"
    if incident.evidence_clip_path:
        media = (
            f'<video src="{escape(incident.evidence_clip_path)}" poster="{escape(incident.evidence_path)}" '
            f'controls muted loop playsinline preload="none" aria-label="{escape(alt)}"></video>'
        )
    else:
        media = f'<img src="{escape(incident.evidence_path)}" loading="lazy" alt="{escape(alt)}">'
    narration = incident.narration
    if narration is not None and narration.source == "model":
        say = (
            f'<p class="say">{escape(narration.summary)}<span class="badge">AI narration · passed 8/8 '
            f'checks</span></p><p class="sub small">{escape(narration.caveat)}</p>'
        )
        action = narration.action
    else:
        say = f'<p class="say">{escape(incident.observation_text)}</p>'
        action = incident.action_text
    facts: list[tuple[str, str]] = []
    ppe_rule = incident.rule_id in (RuleId.R1, RuleId.R2)
    if ppe_rule and incident.helmet is not None:
        state = incident.helmet.state.value.replace("_", " ")
        facts.append(("Helmet", f"{state} (confidence {incident.helmet.confidence:.2f})"))
    if ppe_rule and incident.vest is not None:
        state = incident.vest.state.value.replace("_", " ")
        facts.append(("Hi-vis", f"{state} (confidence {incident.vest.confidence:.2f})"))
    if incident.distance is not None:
        d = incident.distance
        facts.append(
            ("Distance", f"{d.lower_wh:.1f}–{d.upper_wh:.1f} WH (median {d.median_wh:.1f})")
        )
    facts.append(
        ("Basis", ", ".join(incident.references) or "Heuristic: no statutory reference attached")
    )
    if incident.adjudication_candidate:
        facts.append(("Next check", "Queued for the L1 image check"))
    fact_rows = "".join(f"<tr><td>{escape(k)}</td><td>{escape(v)}</td></tr>" for k, v in facts)
    trace = v3.trace_svg(incident, near_wh=near_wh, caution_wh=caution_wh)
    review = "".join(
        [
            '<button class="btn" data-verdict="real">✓ Real</button>',
            *(
                f'<button class="btn" data-verdict="false_alarm" data-reason="{key}">✗ {escape(label)}</button>'
                for key, label in FALSE_REASONS
            ),
            '<button class="btn" data-verdict="unsure">? Unsure</button>',
        ]
    )
    why = f'<div class="section-title">Why it fired</div>{trace}' if trace else ""
    return f"""
<article class="card detail" data-id="{escape(incident.incident_id)}" style="--sev:var(--{slot})" hidden>
  <div class="d-head">
    <div>{_chip(slot, icon, word)} <span class="chip">{escape(incident.rule_id.value)}</span>
      {v3.quality_chip(incident)}</div>
    <h2>{escape(incident.title or incident.reason_code)}: {escape(subject(incident).lower())}</h2>
    <div class="when">{escape(span)} · {duration:.1f} s</div>
  </div>
  <div class="d-body">
    <div>
      <div class="media">{media}</div>
      {say}
      <div class="action"><b>Recommended action</b>{escape(action)}</div>
      {why}
      <table class="facts"><tbody>{fact_rows}</tbody></table>
    </div>
    <div class="crop">
      <div class="section-title" style="margin-top:0">Worker at full resolution</div>
      <a href="{escape(incident.evidence_path)}" title="Open the full evidence frame"><img src="{crop}" loading="lazy" alt="{escape(alt)}"></a>
      <div style="display:grid;gap:8px;margin-top:12px">
        <button class="btn primary" data-seek="{incident.first_seen_s:.2f}">▶ Play in replay</button>
        <a class="btn" href="{escape(incident.evidence_path)}">Open evidence frame</a>
      </div>
    </div>
  </div>
  <div class="review" data-id="{escape(incident.incident_id)}"><span class="muted small" style="margin-right:6px">Your review:</span>{review}</div>
</article>"""


def _incidents_page(
    incidents: Sequence[IncidentRecord], *, near_wh: float, caution_wh: float, duration: float
) -> str:
    if not incidents:
        return _page_head("Incidents", "Nothing persisted long enough to become an incident.") + (
            '<div class="card"><div class="card-body"><p class="empty">No confirmed incidents '
            "in this clip.</p></div></div>"
        )
    ordered = sorted(incidents, key=v3.triage_key)
    counts = Counter(i.rule_id for i in incidents)
    groups = Counter(v3.group(i) for i in incidents)
    rule_buttons = (
        '<button data-rule-filter="all" aria-pressed="true">All rules</button>'
        + "".join(
            f'<button data-rule-filter="{rule.value}" aria-pressed="false">{rule.value} · {counts[rule]}</button>'
            for rule in RuleId
            if counts[rule]
        )
    )
    group_buttons = (
        f'<button data-group-filter="all" aria-pressed="true">All · {len(incidents)}</button>'
        f'<button data-group-filter="act" aria-pressed="false">Act now · {groups["act"]}</button>'
        f'<button data-group-filter="verify" aria-pressed="false">Verify first · {groups["verify"]}</button>'
    )
    rows = []
    for name, label in (("act", "Act now"), ("verify", "Verify first: weak evidence")):
        members = [i for i in ordered if v3.group(i) == name]
        if members:
            rows.append(f'<div class="section-label" data-group="{name}">{label}</div>')
            # one heading per rule, in triage order, instead of the rule's title on every row
            for rule in dict.fromkeys(i.rule_id for i in members):
                inside = [i for i in members if i.rule_id is rule]
                rows.append(
                    f'<div class="rule-head" data-group="{name}" data-rule="{rule.value}">'
                    f"<b>{rule.value}</b>{escape(RULE_NAMES[rule])}<span>{len(inside)}</span></div>"
                )
                rows += [_row(i, duration) for i in inside]
    details = "".join(_detail(i, near_wh=near_wh, caution_wh=caution_wh) for i in ordered)
    head = _page_head(
        "Incidents",
        "Grouped by rule, most severe and best evidenced first. Mark each one real or a false "
        "alarm to measure precision.",
        actions=(
            '<span class="muted small" id="team-review" hidden></span>'
            '<span class="muted small" data-review-count></span>'
            '<button class="btn" id="export-review" type="button">Export review</button>'
        ),
    )
    return f"""{head}
<div class="toolbar"><div class="seg" role="group" aria-label="Filter by triage">{group_buttons}</div>
<div class="seg" role="group" aria-label="Filter by rule">{rule_buttons}</div></div>
<div class="inbox">
  <div class="card list" role="listbox" aria-label="Incidents">{"".join(rows)}</div>
  <div>{details}</div>
</div>"""


# ---------------------------------------------------------------------- other cards


def _rule_status(manifest: RunManifest) -> str:
    rows = []
    for entry in manifest.rule_coverage:
        slot, icon, word = STATUS[entry.status]
        count = (
            f' <span class="muted small">· {entry.incident_count} incident(s)</span>'
            if entry.incident_count
            else ""
        )
        rows.append(
            f"<div><b>{entry.rule_id.value}</b><span>{escape(RULE_NAMES[entry.rule_id])}{count}</span>"
            f"{_chip(slot, icon, word)}</div>"
        )
    return f'<div class="rules-mini">{"".join(rows)}</div>'


def _priorities(incidents: Sequence[IncidentRecord]) -> str:
    top = sorted(incidents, key=v3.triage_key)[:6]
    if not top:
        return '<p class="empty">No incidents.</p>'
    items = []
    for incident in top:
        slot, _, word = SEVERITY.get(incident.severity, ("neutral", "●", incident.severity))
        items.append(
            f'<li data-open="{escape(incident.incident_id)}" style="--sev:var(--{slot})" tabindex="0">'
            f'<span class="bar"></span><div><div class="t">{escape(incident.title)}</div>'
            f'<div class="muted small">{escape(incident.rule_id.value)} · {escape(word)} · '
            f"{escape(subject(incident))} · {_clock(incident.first_seen_s)}</div></div>"
            f'<span class="muted small">{v3.quality_dots(incident)}</span></li>'
        )
    return f'<ul class="prio">{"".join(items)}</ul>'


def _coverage(manifest: RunManifest) -> str:
    rows, table = [], []
    for entry in manifest.rule_coverage:
        slot, icon, word = STATUS[entry.status]
        total = sum(entry.status_counts.values()) or 1
        segments = []
        for status in STATUS:
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
        rows.append(
            f'<div class="cov-row"><b>{entry.rule_id.value}</b><div>{escape(RULE_NAMES[entry.rule_id])}'
            f'<div class="muted small">{escape(entry.reason_code.replace("_", " "))}</div></div>'
            f'<div class="chip-cell">{_chip(slot, icon, word)}</div>'
            f'<div class="stack" role="img" aria-label="Observation status mix for {entry.rule_id.value}">'
            f"{''.join(segments)}</div></div>"
        )
        cells = "".join(f'<td class="num">{entry.status_counts.get(s, 0):,}</td>' for s in STATUS)
        table.append(f"<tr><td>{entry.rule_id.value}</td>{cells}</tr>")
    head = "".join(f'<th class="num">{escape(w)}</th>' for _, _, w in STATUS.values())
    legend = "".join(
        f'<span><span class="sw {"hatch" if s is RuleStatus.UNSUPPORTED else ""}" '
        f'style="background-color:var(--{slot})"></span>{icon} {word}</span>'
        for s, (slot, icon, word) in STATUS.items()
    )
    return (
        f'{"".join(rows)}<div class="legend">{legend}</div>'
        f"<details><summary>Show as table (observations per status)</summary><table><thead><tr>"
        f"<th>Rule</th>{head}</tr></thead><tbody>{''.join(table)}</tbody></table></details>"
    )


def _timings(manifest: RunManifest) -> str:
    timings = [(t.stage.value.replace("_", " "), t.duration_s) for t in manifest.pass_timings]
    if not timings:
        return '<p class="empty">No timings recorded.</p>'
    longest = max(d for _, d in timings) or 1.0
    total = sum(d for _, d in timings)
    duration = manifest.input_video.duration_s or 0.0
    rows = "".join(
        f'<div class="row-b"><span>{escape(name.capitalize())}</span><div class="track">'
        f'<div class="fill" style="width:{100 * d / longest:.1f}%" data-tip="{escape(name)}: {d:.1f} s"></div>'
        f'</div><span class="val">{d:.1f} s</span></div>'
        for name, d in timings
    )
    ratio = f" · {total / duration:.1f}× real time" if duration else ""
    return f'<p class="sub small">Total {total:.0f} s{ratio}</p><div class="bars">{rows}</div>'


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
                f"<td class='num'>{100 * fit.get('relative_rms', 0):.1f} %</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{escape(shot)}</td><td>{_chip('warning', '?', 'Rejected')}</td>"
                f"<td colspan='4'>{escape(str(fit.get('reason_code', '')).replace('_', ' '))}</td></tr>"
            )
    gate = diagnostics.get("geometry_gate") or {}
    camera = diagnostics.get("camera_mode", "unknown")
    body = "".join(rows) or '<tr><td colspan="6">Geometry unsupported for this feed.</td></tr>'
    return f"""<p class="sub small" style="margin-bottom:10px">Camera {escape(str(camera))}; residual
drift {gate.get("residual_drift_wh") or 0:.4f} WH. The camera is calibrated from the workers
themselves; distances are in worker heights (WH) with 95 % bootstrap intervals, never metres.</p>
<table><thead><tr><th>Shot</th><th>Calibration</th><th class="num">Tilt</th><th class="num">Height</th>
<th class="num">Observations</th><th class="num">Height error</th></tr></thead><tbody>{body}</tbody></table>"""


CALIBRATION_FAILURES = {
    "too_few_upright_people": "too few upright, fully visible workers to fit a ground plane",
    "weak_inlier_support": "worker heights in the image did not agree on one ground plane",
    "high_reprojection_error": "the fitted ground plane did not match the workers' heights",
    "unstable_horizon": "the horizon moved too much between fits",
    "implausible_focal_length": "the fit needed an implausible lens",
    "bootstrap_unstable": "resampled fits disagreed too much",
    "optimizer_failed": "the fit did not converge",
}


def _twin_unavailable(manifest: RunManifest) -> str:
    """Why there is no twin, with the fit's own numbers, and what would help."""
    reasons = []
    for shot, fit in (manifest.diagnostics.get("camera_fits") or {}).items():
        code = str(fit.get("reason_code", ""))
        text = CALIBRATION_FAILURES.get(code, code.replace("_", " "))
        ratio, seen = fit.get("inlier_ratio"), fit.get("observations")
        numbers = (
            f" ({ratio:.0%} of {seen} observations from {fit.get('tracks', '?')} tracks agreed)"
            if ratio is not None and seen
            else ""
        )
        reasons.append(f"<li><b>{escape(shot)}</b>: {escape(text)}{escape(numbers)}.</li>")
    found = "".join(reasons) or "<li>No calibration was attempted for this feed.</li>"
    return f"""<p class="sub small" style="margin-bottom:10px">The twin places workers on a ground plane
fitted from their own height in the image. Without an accepted fit there is no trustworthy
position, so no twin is drawn rather than a misleading one. R1–R3 and R6–R8 do not need it;
R4 and R5 distances do.</p>
<ul class="clean">{found}</ul>
<p class="sub small" style="margin-top:10px">This usually means workers' feet were hidden (rebar,
formwork, the frame edge) or few people stood upright in view. A longer clip with more people
walking in the open, or a camera view with less occlusion, gives the fit what it needs.</p>"""


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


# ---------------------------------------------------------------------- page


def render_report(
    manifest: RunManifest,
    incidents: Sequence[IncidentRecord],
    *,
    tracks: TracksSummaryFile | None = None,
    insights: InsightsFile | None = None,
    zone_labels: Mapping[str, str] | None = None,
    near_wh: float = 1.5,
    caution_wh: float = 3.0,
    private: bool = True,
    twin: dict | None = None,
) -> str:
    source = manifest.input_video
    duration = source.duration_s or max(
        (i.resolved_at_s or i.confirmed_at_s for i in incidents), default=1.0
    )
    video = manifest.output_artifacts.get("video_proxy") or manifest.output_artifacts.get("video")
    processing = sum(t.duration_s for t in manifest.pass_timings)
    reused_from = manifest.diagnostics.get("reused_from")
    groups = Counter(v3.group(i) for i in incidents)
    exposure = sum(e.exposure_s for e in insights.exposure) if insights else None
    person_tracks = (
        f"{sum(t.object_class == 'person' for t in tracks.tracks)} track IDs" if tracks else ""
    )
    bundle = RunBundle(
        run_dir=Path("."),
        manifest=manifest,
        incidents=list(incidents),
        insights=insights,
        tracks=tracks,
        zone_labels=dict(zone_labels or {}),
        catalogue=load_catalogue(),
    )
    facts = build_facts(bundle)
    overview_kpis = _kpis(
        [
            (
                "Incidents",
                str(len(incidents)),
                f"{groups['act']} act now · {groups['verify']} verify first",
            ),
            (
                "Exposure",
                "—" if exposure is None else f"{exposure / 60:.1f} min",
                "worker-minutes in a hazard condition",
            ),
            (
                "Peak workers in view",
                "—" if tracks is None else str(peak_workers(tracks)),
                person_tracks,
            ),
            (
                "Reprocessed" if reused_from else "Processing",
                f"{processing:.0f} s",
                f"rules and render only; detections from {reused_from}"
                if reused_from
                else f"{processing / max(duration, 1e-6):.1f}× real time for {duration:.0f} s of video",
            ),
        ]
    )
    player = (
        f'<video id="video" controls preload="metadata" src="{escape(video)}"></video>'
        if video
        else '<p class="empty">No video in this bundle.</p>'
    )
    overview = f"""
{_verdict(bundle, Path(source.path).name, duration, groups)}
{overview_kpis}
<div class="grid cols-3-2">
  {_card("Replay", player + _timeline(incidents, duration), "click the timeline or an incident to seek")}
  <div class="stack-gap">
    {_card("Top priorities", _priorities(incidents), '<a href="#incidents">All incidents →</a>')}
    {_card("Rule status", _rule_status(manifest), '<a href="#run">Coverage →</a>')}
  </div>
</div>"""

    if insights is not None:
        indicators = f"""
{_page_head("Leading indicators", "Incidents count what persisted. These count the exposure behind them, including lapses too short to alert: the precursors safety teams track.")}
{v3.indicator_kpis(insights)}
<div class="grid cols-2">
  {_card("Exposure by rule", v3.exposure_bars(insights), "worker-seconds, confirmed or not")}
  {_card("Hotspots", v3.heatmap(insights))}
</div>
<div class="grid cols-2" style="margin-top:16px">
  {_card("Brief exposures", v3.brief_table(insights), "held, but below the alert threshold")}
  {_card("Where and who", v3.area_table(insights))}
</div>"""
    else:
        indicators = _page_head("Leading indicators", "Not recorded for this run.")

    site_view = manifest.output_artifacts.get("site_view")
    view_card = (
        _card(
            "Camera view and zones",
            f'<figure class="site"><img src="{escape(site_view)}" loading="lazy" '
            f'alt="The camera view with the configured zones and edges"></figure>',
            "as configured for this run",
        )
        if site_view
        else ""
    )
    site = f"""
{_page_head("Site & camera", "Is the scene still the one the zones were drawn on, and how well is the camera calibrated?")}
{view_card}
<div class="grid cols-2" style="margin-top:16px">
  {_card("Zone health", v3.zone_health(insights, zone_labels or {}))}
  {_card("Camera calibration", _calibration(manifest))}
</div>"""

    warnings = "".join(f"<li>{escape(w)}</li>" for w in manifest.warnings) or "<li>None</li>"
    models = "".join(
        f"<tr><td>{escape(k)}</td><td><code>{escape(v)}</code></td></tr>"
        for k, v in manifest.model_ids.items()
    )
    run = f"""
{_page_head("Run details", f"Run {manifest.run_id} · schema {manifest.schema_version}. What each rule could assess, how long each stage took, and what the run lacked.")}
{_card("Rule coverage", _coverage(manifest), "every rule is always reported, even when it could not run")}
<div class="grid cols-2" style="margin-top:16px">
  {_card("Processing time", _timings(manifest))}
  {_card("Models and provenance", f"<table><tbody>{models}</tbody></table>")}
</div>
<div class="grid cols-2" style="margin-top:16px">
  {_card("Warnings", f'<ul class="clean">{warnings}</ul>')}
  {_card("About this report", v3.oversight(private))}
</div>"""

    nav_counts = {"incidents": str(len(incidents))}
    nav = "".join(
        f'<a href="#{key}" data-page="{key}"><span class="ico" aria-hidden="true">{icon(key)}</span>{label}'
        + (f'<span class="count">{nav_counts[key]}</span>' if key in nav_counts else "")
        + "</a>"
        for key, label in NAV
    )
    downloads = "".join(
        f'<a href="{escape(path)}" download>{escape(label)}<span>{escape(note)}</span></a>'
        for label, path, note in (
            (
                "Incidents",
                manifest.output_artifacts.get("incidents"),
                "incidents.json · every confirmed event",
            ),
            (
                "Leading indicators",
                manifest.output_artifacts.get("insights"),
                "insights.json · exposure and near-misses",
            ),
            ("Replay video", video, "MP4 · camera and twin side by side"),
        )
        if path
    )
    menu = (
        '<details class="menu"><summary class="btn" aria-label="Downloads">'
        f'{DOWNLOAD_ICON}<span class="hide-sm">Download</span></summary>'
        f'<div class="menu-pop">{downloads}</div></details>'
        if downloads
        else ""
    )
    completed = escape((manifest.completed_at or "")[:16].replace("T", " "))
    meta = f"{source.width}×{source.height} · {_clock(duration)[:-2]} · {completed} UTC"
    incidents_page = _incidents_page(
        incidents, near_wh=near_wh, caution_wh=caution_wh, duration=duration
    )
    risk_page = risk_html.page(
        bundle,
        facts,
        head=_page_head(
            "Risk assessment",
            "Each hazard rated likelihood × severity, with controls in hierarchy-of-controls "
            "order.",
        ),
        card=_card,
    )
    if twin is not None:
        twin_page = _page_head(
            "Digital twin",
            "The site rebuilt in 3D from the camera, replayed in step with the video.",
        ) + twin_viewer.section(
            twin,
            video,
            manifest.output_artifacts.get("site_view"),
            incidents=_twin_incidents(incidents),
            areas={
                a.proposal_id: {"exposure": a.exposure_s, "workers": a.workers}
                for a in (insights.area_exposure if insights else ())
            },
        )
    else:
        twin_page = _page_head(
            "Digital twin", "Not available: the camera could not be calibrated for this clip."
        ) + _card("Why, and what would fix it", _twin_unavailable(manifest))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(Path(source.path).name)} · {brand.NAME}</title>
<link rel="icon" href="{brand.FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{STYLE}{twin_viewer.STYLE}{risk_html.STYLE}</style></head>
<body data-run="{escape(manifest.run_id)}">
<header class="topbar"><div class="ident"><span class="brand" title="{brand.NAME}">{brand.wordmark(30)}</span>
<span class="divider" aria-hidden="true"></span>
<div style="min-width:0"><div class="title">{escape(Path(source.path).name)}</div><div class="meta">{meta}</div></div></div>
<div class="spacer"></div>
<a class="back" href="/" data-hosted hidden>← All runs</a>
<button class="btn primary" type="button" data-ask-open>{SPARK_ICON}<span class="hide-sm">Ask the assistant</span></button>
{menu}
<button class="btn icon" id="theme" type="button" aria-label="Toggle light or dark theme" title="Theme">{THEME_ICON}</button></header>
<div class="shell">
<nav class="sidebar" aria-label="Sections"><div class="group">Report</div><div class="nav">{nav}</div></nav>
<main>
<section class="page" id="page-overview">{overview}</section>
<section class="page" id="page-twin">{twin_page}</section>
<section class="page" id="page-incidents">{incidents_page}</section>
<section class="page" id="page-risk">{risk_page}</section>
<section class="page" id="page-indicators">{indicators}</section>
<section class="page" id="page-site">{site}</section>
<section class="page" id="page-run">{run}</section>
<footer class="page-foot"><span>Heuristic triage from video: every incident is a prompt for a
person to check, not a verdict, legal advice or certified measurement.</span>
<span>Run <span class="mono">{escape(manifest.run_id)}</span></span></footer>
</main></div>
{risk_html.drawer(bundle, facts)}
<button class="fab" type="button" data-ask-open aria-label="Ask the safety assistant">{SPARK_ICON}Ask</button>
<div class="tooltip" id="tip" role="tooltip"></div>
{'<script src="' + twin_viewer.THREE_ASSET + '"></script>' if twin is not None else ""}
<script>{SCRIPT}{risk_html.SCRIPT}{twin_viewer.SCRIPT if twin is not None else ""}</script></body></html>
"""


def write_report(
    path: Path,
    manifest: RunManifest,
    incidents: Sequence[IncidentRecord],
    *,
    tracks: TracksSummaryFile | None = None,
    **options: Any,
) -> Path:
    if options.get("twin") is not None:
        twin_viewer.install_assets(path.parent)
    path.write_text(render_report(manifest, incidents, tracks=tracks, **options), encoding="utf-8")
    return path


def rebuild_report(run_dir: Path) -> Path:
    """Re-render `report.html` from a finished bundle (after narration, review, etc.)."""
    from pipeline.output.bundle import read_model
    from shared.config import load_config
    from shared.schemas.bundle import (
        INCIDENTS_FILE,
        INSIGHTS_FILE,
        REPORT_FILE,
        RUN_MANIFEST_FILE,
        SCENE_FILE,
        TRACKS_SUMMARY_FILE,
        TWIN_FILE,
        IncidentsFile,
    )
    from shared.schemas.scene import SceneCache

    config = load_config()
    manifest = read_model(run_dir / RUN_MANIFEST_FILE, RunManifest)
    incidents = read_model(run_dir / INCIDENTS_FILE, IncidentsFile).incidents
    tracks = read_model(run_dir / TRACKS_SUMMARY_FILE, TracksSummaryFile)
    insights_path = run_dir / INSIGHTS_FILE
    insights = read_model(insights_path, InsightsFile) if insights_path.is_file() else None
    scene = read_model(run_dir / SCENE_FILE, SceneCache)
    twin_path = run_dir / TWIN_FILE
    twin = json.loads(twin_path.read_text(encoding="utf-8")) if twin_path.is_file() else None
    labels = {
        p.proposal_id: p.label or p.proposal_id
        for proposals in scene.shot_proposals.values()
        for p in proposals
    }
    return write_report(
        run_dir / REPORT_FILE,
        manifest,
        incidents,
        tracks=tracks,
        insights=insights,
        zone_labels=labels,
        near_wh=config.rules.r4_near_max_wh,
        caution_wh=config.rules.r4_caution_max_wh,
        private=manifest.diagnostics.get("privacy", config.render.privacy) == "faces",
        twin=twin,
    )
