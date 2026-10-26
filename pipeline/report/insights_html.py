"""v3 report pieces: decision traces, evidence grades, triage, leading indicators, zone health.

Every function returns an HTML fragment for a card body; all text is escaped; status
colour always comes with an icon and a word.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord, IncidentTrace
from shared.schemas.insights import InsightsFile

QUALITY = {  # icon, word, explanation
    "strong": ("●●●", "Strong evidence", "visible most of the episode, confident, sustained"),
    "moderate": ("●●○", "Moderate evidence", "partly visible or brief"),
    "weak": ("●○○", "Weak evidence", "mostly occluded or low model confidence: verify first"),
}
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
QUALITY_RANK = {"strong": 0, "moderate": 1, "weak": 2, None: 1}
RULE_NAMES = {
    "R1": "Missing helmet",
    "R2": "Missing hi-vis",
    "R3": "Restricted zone",
    "R4": "Machinery proximity",
    "R5": "Open edge",
    "R6": "Worker down",
    "R7": "Suspended load",
    "R8": "Fire or smoke",
}


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def group(incident: IncidentRecord) -> str:
    """`verify` for weak evidence, else `act`."""
    return "verify" if incident.evidence_quality == "weak" else "act"


def triage_key(incident: IncidentRecord) -> tuple:
    """Act on severe, well-evidenced incidents first; weak evidence goes to verification."""
    return (
        group(incident) == "verify",
        SEVERITY_RANK.get(incident.severity, 4),
        QUALITY_RANK.get(incident.evidence_quality, 1),
        incident.first_seen_s,
    )


QUALITY_COLOUR = {"strong": "var(--good)", "moderate": "var(--warning)", "weak": "var(--critical)"}


def quality_dots(incident: IncidentRecord) -> str:
    """A compact evidence label for list rows: coloured dot plus the word."""
    if incident.evidence_quality not in QUALITY:
        return ""
    _, word, why = QUALITY[incident.evidence_quality]
    colour = QUALITY_COLOUR[incident.evidence_quality]
    return (
        f'<span class="q" style="--q:{colour}" title="{escape(why)}">{escape(word.lower())}</span>'
    )


def quality_chip(incident: IncidentRecord) -> str:
    if incident.evidence_quality not in QUALITY:
        return ""
    icon, word, why = QUALITY[incident.evidence_quality]
    return (
        f'<span class="chip" title="{escape(why)}"><i aria-hidden="true">{icon}</i>'
        f"{escape(word)}</span>"
    )


def trace_svg(incident: IncidentRecord, *, near_wh: float, caution_wh: float) -> str:
    """Why it fired: the condition over time (strip), the signal (line), the debounce."""
    trace: IncidentTrace | None = incident.trace
    if trace is None or len(trace.points) < 2:
        return ""
    points = trace.points
    t0, t1 = points[0].t, points[-1].t
    span = max(t1 - t0, 1e-6)
    width, top, strip_y, strip_h = 560.0, 6.0, 62.0, 12.0

    def x(t: float) -> float:
        return 6 + (width - 12) * (t - t0) / span

    colour = {"holds": "var(--sev)", "clear": "var(--good)", "unknown": "var(--muted)"}
    parts = [
        f'<svg viewBox="0 0 {width:.0f} 84" role="img" '
        f'aria-label="Rule condition over time for {escape(incident.rule_id.value)}">'
    ]
    for a, b in zip(points, points[1:], strict=False):
        opacity = "1" if a.state == "holds" else "0.45"
        parts.append(
            f'<rect x="{x(a.t):.1f}" y="{strip_y}" width="{max(x(b.t) - x(a.t), 0.6):.1f}" '
            f'height="{strip_h}" fill="{colour.get(a.state, "var(--muted)")}" opacity="{opacity}"/>'
        )
    if trace.metric == "distance_wh":
        values = [p for p in points if p.value is not None]
        if values:
            high = max(caution_wh * 1.2, max(p.upper or p.value for p in values))

            def y(v: float) -> float:
                return top + (strip_y - 8 - top) * (1 - min(v, high) / high)

            band = " ".join(f"{x(p.t):.1f},{y(p.upper or p.value):.1f}" for p in values)
            band += " " + " ".join(
                f"{x(p.t):.1f},{y(p.lower or p.value):.1f}" for p in reversed(values)
            )
            parts.append(f'<polygon points="{band}" fill="var(--series)" opacity="0.18"/>')
            line = " ".join(f"{x(p.t):.1f},{y(p.value):.1f}" for p in values)
            parts.append(
                f'<polyline points="{line}" fill="none" stroke="var(--series)" stroke-width="2"/>'
            )
            for edge, label in ((near_wh, "near"), (caution_wh, "caution")):
                parts.append(
                    f'<line x1="6" x2="{width - 6:.0f}" y1="{y(edge):.1f}" y2="{y(edge):.1f}" '
                    f'stroke="var(--line)" stroke-dasharray="3 3"/>'
                    f'<text x="{width - 8:.0f}" y="{y(edge) - 3:.1f}" text-anchor="end">'
                    f"{label} {edge:g} WH</text>"
                )
    for at, dash, label, anchor, dx in (
        (incident.first_seen_s, "3 2", "onset", "end", -3),
        (incident.confirmed_at_s, "", "confirmed", "start", 3),
    ):
        if t0 <= at <= t1:
            # onset labels to the left of its line, confirmed to the right: never overlap
            if anchor == "end" and x(at) < 40:
                anchor, dx = "start", 3
            y_label = top + 8 if label == "confirmed" or anchor == "end" else top + 20
            parts.append(
                f'<line x1="{x(at):.1f}" x2="{x(at):.1f}" y1="{top + 10}" y2="{strip_y + strip_h}" '
                f'stroke="var(--text)" stroke-width="1.2" stroke-dasharray="{dash}"/>'
                f'<text x="{x(at) + dx:.1f}" y="{y_label}" text-anchor="{anchor}">{label}</text>'
            )
    parts.append("</svg>")
    end = incident.resolved_at_s
    caption = (
        f"Condition began {_clock(incident.first_seen_s)}; confirmed after "
        f"{incident.confirmed_at_s - incident.first_seen_s:.1f} s (debounce {trace.debounce_s:g} s); "
        f"worker visible {100 * trace.observed_fraction:.0f} % of the episode; "
        + (f"cleared {_clock(end)}." if end is not None else "ongoing at the end of the clip.")
    )
    legend = (
        '<div class="legend small">'
        '<span><span class="sw" style="background:var(--sev)"></span>condition held</span>'
        '<span><span class="sw" style="background:var(--good);opacity:.45"></span>clear</span>'
        '<span><span class="sw" style="background:var(--muted)"></span>not visible</span></div>'
    )
    return (
        f'<div class="trace">{"".join(parts)}{legend}<div class="cap">{escape(caption)}</div></div>'
    )


# ---------------------------------------------------------------------- leading indicators


def indicator_kpis(insights: InsightsFile) -> str:
    total = sum(e.exposure_s for e in insights.exposure)
    closest = next((c for c in insights.closest_approaches if c.rule_id.value == "R4"), None)
    tiles = [
        ("Exposure", f"{total / 60:.1f} min", "worker-minutes in a hazard condition"),
        (
            "Brief exposures",
            str(len(insights.brief_exposures)),
            "held, but shorter than the alert threshold",
        ),
        (
            "Closest machine approach",
            f"{closest.distance.median_wh:.1f} WH" if closest else "—",
            (
                f"worker {display_track_label(closest.canonical_track_id)} at {_clock(closest.time_s)}"
                if closest
                else "no machinery in view"
            ),
        ),
        (
            "Workers exposed",
            str(insights.workers_exposed),
            "per-clip track IDs with any exposure",
        ),
    ]
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><div class="label">{escape(a)}</div><div class="value">{escape(b)}</div>'
            f'<div class="note">{escape(c)}</div></div>'
            for a, b, c in tiles
        )
        + "</div>"
    )


def exposure_bars(insights: InsightsFile) -> str:
    rows = [e for e in insights.exposure if e.exposure_s > 0 or e.brief_exposures]
    if not rows:
        return '<p class="empty">No hazard condition held for anyone in this clip.</p>'
    longest = max(e.exposure_s for e in rows) or 1.0
    bars = "".join(
        f'<div class="row-b"><span>{escape(e.rule_id.value)} · {escape(RULE_NAMES[e.rule_id.value])}</span>'
        f'<div class="track"><div class="fill" style="width:{100 * e.exposure_s / longest:.1f}%" '
        f'data-tip="{escape(f"<b>{e.rule_id.value}</b><br>{e.exposure_s:.0f} worker-seconds", quote=True)}"></div></div>'
        f'<span class="val">{e.exposure_s:.0f} s · {e.workers_exposed} worker(s)</span></div>'
        for e in rows
    )
    table = "".join(
        f"<tr><td>{escape(e.rule_id.value)}</td><td class='num'>{e.exposure_s:.1f}</td>"
        f"<td class='num'>{e.workers_exposed}</td><td class='num'>{e.brief_exposures}</td>"
        f"<td class='num'>{e.confirmed_incidents}</td></tr>"
        for e in insights.exposure
    )
    return (
        f'<div class="bars">{bars}</div><details><summary>Show as table</summary><table><thead><tr>'
        f'<th>Rule</th><th class="num">Exposure (s)</th><th class="num">Workers</th>'
        f'<th class="num">Brief</th><th class="num">Incidents</th></tr></thead><tbody>{table}</tbody>'
        f"</table></details>"
    )


def heatmap(insights: InsightsFile) -> str:
    if not insights.heatmap_path:
        return '<p class="empty">No exposure to map.</p>'
    return (
        f'<figure class="heat"><img src="{escape(insights.heatmap_path)}" loading="lazy" '
        f'alt="Where exposure happened, over the camera view"><figcaption><span>Where workers stood '
        f"while a hazard condition held (any rule).</span><span>Fewer</span>"
        f'<span class="ramp"></span><span>more worker-seconds</span></figcaption></figure>'
    )


def brief_table(insights: InsightsFile) -> str:
    briefs = sorted(insights.brief_exposures, key=lambda b: b.start_s - b.end_s)[:10]
    if not briefs:
        return '<p class="empty">No brief exposures.</p>'
    rows = "".join(
        f"<tr><td>{escape(b.rule_id.value)}</td><td>{escape(display_track_label(b.canonical_track_id))}</td>"
        f"<td class='num'>{_clock(b.start_s)}</td><td class='num'>{b.end_s - b.start_s:.1f} s</td>"
        f"<td>{escape(f'{b.closest.median_wh:.1f} WH' if b.closest else '—')}</td>"
        f'<td class="num"><button class="btn sm" data-seek="{b.start_s:.2f}">▶ Play</button></td></tr>'
        for b in briefs
    )
    return (
        "<table><thead><tr><th>Rule</th><th>Worker</th><th class='num'>At</th>"
        "<th class='num'>Held</th><th>Closest</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def area_table(insights: InsightsFile) -> str:
    areas = insights.area_exposure[:8]
    workers = [w for w in insights.top_workers if w.exposure_s > 0][:6]
    parts = []
    if areas:
        rows = "".join(
            f"<tr><td>{escape(a.label or a.proposal_id)}</td><td>{escape(a.rule_id.value)}</td>"
            f"<td class='num'>{a.exposure_s:.0f} s</td><td class='num'>{a.workers}</td></tr>"
            for a in areas
        )
        parts.append(
            "<table><thead><tr><th>Zone or edge</th><th>Rule</th><th class='num'>Exposure</th>"
            f"<th class='num'>Workers</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    if workers:
        rows = "".join(
            f"<tr><td>Worker {escape(display_track_label(w.canonical_track_id))}</td>"
            f"<td>{escape(', '.join(f'{r} {s:.0f} s' for r, s in w.by_rule.items()))}</td>"
            f"<td class='num'>{w.exposure_s:.0f} s</td></tr>"
            for w in workers
        )
        parts.append(
            '<div class="section-title">Most exposed (per-clip track IDs)</div>'
            "<table><thead><tr><th>Track</th><th>By rule</th><th class='num'>Total</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "".join(parts) or '<p class="empty">No exposure recorded.</p>'


def zone_health(insights: InsightsFile | None, labels: Mapping[str, str]) -> str:
    if insights is None or not insights.zone_health:
        return '<p class="empty">No site zones configured for this camera.</p>'
    rows = []
    for proposal_id, change in sorted(insights.zone_health.items(), key=lambda kv: -kv[1]):
        if change >= 0.5:
            chip = (
                '<span class="chip"><span class="sw" style="background:var(--warning)"></span>'
                '<i aria-hidden="true">!</i>Review zone</span>'
            )
        else:
            chip = (
                '<span class="chip"><span class="sw" style="background:var(--good)"></span>'
                '<i aria-hidden="true">✓</i>Matches drawing</span>'
            )
        rows.append(
            f"<tr><td>{escape(labels.get(proposal_id, proposal_id))}</td><td>{chip}</td>"
            f"<td class='num'>{change:.2f}</td></tr>"
        )
    return (
        '<p class="sub small" style="margin-bottom:10px">Each zone is compared with the frame it was '
        "drawn on, by structure only (lighting does not count): 0 means unchanged, 1 completely "
        "different. Above 0.5 the zone needs a fresh look.</p>"
        "<table><thead><tr><th>Zone or edge</th><th>Status</th><th class='num'>Change</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def oversight(private: bool) -> str:
    privacy = (
        "Workers' faces are pixelated in this report, the video, every evidence image and clip."
        if private
        else "Face pixelation was switched off for this run."
    )
    return f"""<ul class="clean">
<li>{escape(privacy)} No identity, face recognition or biometrics are used; worker numbers are
per-clip track IDs that mean nothing outside this video.</li>
<li>Weak-evidence incidents are listed separately under <i>Verify first</i>. Behaviour,
productivity, emotions and anything outside the camera's view are not monitored.</li>
<li>Rules can be re-evaluated on this footage with new zones or settings
(<code>safety-twin reprocess</code>) without re-running detection.</li></ul>"""
