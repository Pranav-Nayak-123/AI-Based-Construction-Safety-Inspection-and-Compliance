"""Incident narration (L4): facts in, a short grounded briefing out.

The rule engine decides everything that matters — rule, severity, basis, references,
action. The narrator only rewrites those facts as readable prose, in a fixed JSON shape,
and the guardrail (L5, `llm/guardrail.py`) rejects any output that adds, changes or
invents something. A rejected output is replaced by the deterministic template.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import Field

from shared.coordinates import StrictModel
from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord

OUTPUT_KEYS = ("summary", "caveat", "action")

SYSTEM_PROMPT = (
    "You write short safety briefings for a construction site supervisor from the facts "
    "you are given. Reply with one JSON object with exactly three string fields: "
    '"summary" (one sentence: who, what, when, for how long), "caveat" (one sentence on '
    'what the video evidence can and cannot show), and "action" (one sentence: the next '
    "step, keeping the meaning of the given action). Use only the facts given. Never add "
    "numbers, names, places, rule numbers, citations or distances in metres. Never say "
    "anything is illegal, a violation, a breach or non-compliant. Distances are in worker "
    "heights (WH)."
)


class NarrationFacts(StrictModel):
    """Everything the narrator may mention, and nothing else."""

    rule_id: str
    title: str
    severity: str
    basis: str
    worker: str
    start_s: float
    duration_s: float
    status: str
    zone: str | None = None
    machine: str | None = None
    band: str | None = None
    distance_wh: str | None = None
    helmet: str | None = None
    vest: str | None = None
    action: str

    def to_prompt(self) -> str:
        facts = {k: v for k, v in self.model_dump().items() if v is not None}
        return "Facts:\n" + json.dumps(facts, indent=1, ensure_ascii=False)


class Narration(StrictModel):
    summary: str = Field(min_length=1)
    caveat: str = Field(min_length=1)
    action: str = Field(min_length=1)


def facts_from_incident(
    incident: IncidentRecord, zone_labels: Mapping[str, str] | None = None
) -> NarrationFacts:
    end = incident.resolved_at_s
    duration = (end if end is not None else incident.confirmed_at_s) - incident.first_seen_s
    status = (
        "ongoing at the end of the clip"
        if incident.resolution_reason in (None, "end_of_clip")
        else "resolved"
    )
    distance = None
    if incident.distance is not None:
        distance = f"{incident.distance.lower_wh:.1f} to {incident.distance.upper_wh:.1f} WH"
    zone = None
    if incident.proposal_ids:
        zone = (zone_labels or {}).get(incident.proposal_ids[0]) or incident.proposal_ids[0]
    machine = None
    if incident.object_id is not None and incident.rule_id.value in ("R2", "R4"):
        machine = f"machine {display_track_label(incident.object_id)}"
    return NarrationFacts(
        rule_id=incident.rule_id.value,
        title=incident.title or incident.reason_code,
        severity=incident.severity,
        basis=incident.basis,
        worker=display_track_label(incident.canonical_track_id or "unknown"),
        start_s=round(incident.first_seen_s, 1),
        duration_s=round(max(duration, 0.0), 1),
        status=status,
        zone=zone,
        machine=machine,
        band=incident.relative_band.value if incident.relative_band else None,
        distance_wh=distance,
        helmet=incident.helmet.state.value if incident.helmet else None,
        vest=incident.vest.state.value if incident.vest else None,
        action=incident.action_text,
    )


def template_narration(facts: NarrationFacts) -> Narration:
    """The deterministic arm (and the fallback): always valid, never fluent."""
    where = f" in {facts.zone}" if facts.zone else ""
    near = f" near {facts.machine}" if facts.machine else ""
    summary = (
        f"{facts.title}: worker {facts.worker}{where}{near} from {facts.start_s}s "
        f"for {facts.duration_s}s ({facts.status})."
    )
    caveat = (
        "This is a visual observation from video; verify on site."
        if facts.basis == "visual"
        else "This is a heuristic observation from video; verify on site."
    )
    return Narration(summary=summary, caveat=caveat, action=facts.action)


def chat_messages(facts: NarrationFacts) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": facts.to_prompt()},
    ]
