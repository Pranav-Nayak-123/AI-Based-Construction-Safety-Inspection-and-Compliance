"""The run as short, citable facts: everything the chatbot may say, and nothing else.

Each fact has an id the answer cites in square brackets ([I-03], [RISK-R3], ...), so a
reader can check every claim, and the guardrail can check every number against the
facts the model was shown. Deterministic: the same bundle always gives the same facts,
and `register_answer` answers common questions from them with no model at all (the
chatbot's fallback and the offline report's prepared answers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pipeline.output.risk import RiskRow
from pipeline.output.run_bundle import RunBundle
from shared.enums import RuleId
from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord

RULE_WORDS = {
    RuleId.R1: "helmet hard hat head protection ppe headgear",
    RuleId.R2: "hi-vis high visibility vest conspicuity ppe vehicle traffic",
    RuleId.R3: "restricted zone exclusion area intrusion entry boundary barrier",
    RuleId.R4: "machine machinery plant excavator proximity distance struck",
    RuleId.R5: "edge fall height opening guardrail fall protection",
    RuleId.R6: "worker down lying collapse fall injured medical first aid",
    RuleId.R7: "crane hook suspended load lift beneath under",
    RuleId.R8: "fire smoke flames burning hot work evacuation",
}


@dataclass(frozen=True)
class Fact:
    id: str
    kind: str  # summary | risk | not_observed | not_assessed | incident | worker | area | limits
    text: str
    rule: RuleId | None = None
    workers: tuple[str, ...] = ()
    time_s: float | None = None
    keywords: str = ""
    links: dict[str, str] = field(default_factory=dict)
    brief: str = ""  # a shorter form for register answers; the model always sees `text`


def clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{rest:04.1f}"


def incident_ids(bundle: RunBundle) -> dict[str, str]:
    """Incident id → short citation id (I-01 …) in time order."""
    ordered = sorted(bundle.incidents, key=lambda i: (i.first_seen_s, i.incident_id))
    return {i.incident_id: f"I-{n:02d}" for n, i in enumerate(ordered, 1)}


def build_facts(bundle: RunBundle) -> list[Fact]:
    risk = bundle.risk
    short = incident_ids(bundle)
    facts = [_summary(bundle)]
    for row in risk.rows:
        facts.append(_risk(row, short))
    for rule_id, reason in risk.not_observed:
        facts.append(
            Fact(
                id=f"CLEAR-{rule_id.value}",
                kind="not_observed",
                rule=rule_id,
                text=f"{rule_id.value} {_title(bundle, rule_id)}: checked for the whole clip and "
                f"not observed ({reason.replace('_', ' ')}).",
                keywords=RULE_WORDS[rule_id],
            )
        )
    for rule_id, reason, why in risk.not_assessed:
        facts.append(
            Fact(
                id=f"NA-{rule_id.value}",
                kind="not_assessed",
                rule=rule_id,
                text=f"{rule_id.value} {_title(bundle, rule_id)}: not assessed in this clip "
                f"— {why} "
                f"({reason.replace('_', ' ')}). This is not the same as no risk.",
                keywords=RULE_WORDS[rule_id],
            )
        )
    ordered = sorted(bundle.incidents, key=lambda i: (i.first_seen_s, i.incident_id))
    facts += [_incident(bundle, incident, short[incident.incident_id]) for incident in ordered]
    facts += _tallies(bundle, short)
    facts += _timeline(bundle, short)
    facts += _workers(bundle, short)
    facts += _areas(bundle)
    facts.append(_limits(bundle))
    return facts


def _title(bundle: RunBundle, rule_id: RuleId) -> str:
    return bundle.catalogue.rule(rule_id).title


def _summary(bundle: RunBundle) -> Fact:
    m = bundle.manifest
    risk = bundle.risk
    top = ", ".join(f"{r.rule_id.value} {r.band} ({r.rating}/25)" for r in risk.rows[:3]) or "none"
    exposure = (
        f" Workers spent {sum(e.exposure_s for e in bundle.insights.exposure) / 60:.1f} "
        "worker-minutes in a hazard condition."
        if bundle.insights
        else ""
    )
    return Fact(
        id="RUN",
        kind="summary",
        text=(
            f"Run {m.run_id}: {bundle.duration_s:.0f} s of fixed-camera video, "
            f"{bundle.workers_seen} workers seen for at least 3 s, "
            f"{len(bundle.incidents)} confirmed incidents across "
            f"{len({i.rule_id for i in bundle.incidents})} rules.{exposure} "
            f"Highest risks: {top}."
        ),
        keywords="overview summary site run clip total how many biggest top risk",
    )


def _risk(row: RiskRow, short: dict[str, str]) -> Fact:
    cited = " ".join(short[i] for i in row.incident_ids[:6] if i in short)
    where = f" Where: {', '.join(row.areas)}." if row.areas else ""
    who = f" Most exposed workers: {', '.join(row.top_workers)}." if row.top_workers else ""
    controls = " ".join(f"{level}: {text}." for level, text in row.controls)
    return Fact(
        id=f"RISK-{row.rule_id.value}",
        kind="risk",
        rule=row.rule_id,
        workers=row.top_workers,
        text=(
            f"{row.rule_id.value} {row.title} — hazard: {row.hazard}. "
            f"Risk {row.band}, rating {row.rating}/25 = likelihood {row.likelihood_score} "
            f"({row.likelihood}) × severity {row.severity_score} ({row.severity}). {row.why} "
            f"Exposure {row.exposure_s:.1f} worker-seconds, {row.workers_exposed} workers."
            f"{where}{who} Incidents: {cited or 'none'}. "
            f"Recommended controls (hierarchy of controls, general good practice): {controls}"
        ),
        keywords=RULE_WORDS[row.rule_id] + " risk rating likelihood severity control fix",
        brief=(
            f"{row.rule_id.value} {row.title}: {row.band} risk ({row.rating}/25). {row.why} "
            f"{row.workers_exposed} workers exposed for {row.exposure_s:.1f} worker-seconds."
            f"{where}{'' if row.incidents else ' No confirmed incident.'}"
        ),
    )


def _incident(bundle: RunBundle, incident: IncidentRecord, short: str) -> Fact:
    worker = display_track_label(incident.canonical_track_id or "")
    end = incident.resolved_at_s
    duration = f"{end - incident.first_seen_s:.1f} s" if end is not None else "until the clip ended"
    subject = f"worker {worker}" if incident.rule_id is not RuleId.R8 else f"region {worker}"
    where = ""
    zones = [bundle.zone_labels.get(p, p) for p in incident.proposal_ids]
    if zones:
        where = f" in {', '.join(zones)}"
    if incident.object_id and incident.rule_id in (RuleId.R2, RuleId.R4):
        where += f" near machine {display_track_label(incident.object_id)}"
    if incident.object_id and incident.rule_id is RuleId.R7:
        where += f" under suspended load {display_track_label(incident.object_id)}"
    distance = (
        f" Closest distance {incident.distance.lower_wh:.1f}–{incident.distance.upper_wh:.1f} WH."
        if incident.distance
        else ""
    )
    ppe = ""
    if incident.helmet and incident.rule_id in (RuleId.R1, RuleId.R2):
        ppe = f" Helmet state {incident.helmet.state}, vest state "
        ppe += f"{incident.vest.state}." if incident.vest else "unknown."
    seen = (
        f" Visible {incident.trace.observed_fraction:.0%} of the episode." if incident.trace else ""
    )
    return Fact(
        id=short,
        kind="incident",
        rule=incident.rule_id,
        workers=(worker,) if incident.rule_id is not RuleId.R8 else (),
        time_s=incident.first_seen_s,
        text=(
            f"{incident.rule_id.value} {incident.title}: {subject}{where} from "
            f"{clock(incident.first_seen_s)} for {duration} ({incident.severity} severity, "
            f"{incident.evidence_quality or 'ungraded'} evidence, {incident.basis} rule)."
            f"{distance}{ppe}{seen} Action: {incident.action_text}"
        ),
        keywords=RULE_WORDS[incident.rule_id],
        links={
            "evidence": incident.evidence_path,
            **({"clip": incident.evidence_clip_path} if incident.evidence_clip_path else {}),
        },
    )


def _length(bundle: RunBundle, incident: IncidentRecord) -> float:
    end = incident.resolved_at_s if incident.resolved_at_s is not None else bundle.duration_s
    return end - incident.first_seen_s


def _subject(incident: IncidentRecord) -> str:
    label = display_track_label(incident.canonical_track_id or "")
    return f"region {label}" if incident.rule_id is RuleId.R8 else f"worker {label}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _tallies(bundle: RunBundle, short: dict[str, str]) -> list[Fact]:
    """Counts per rule, so the model quotes a number instead of counting incidents itself."""
    facts = []
    for rule_id in RuleId:
        found = sorted(
            (i for i in bundle.incidents if i.rule_id is rule_id),
            key=lambda i: (i.first_seen_s, i.incident_id),
        )
        if not found:
            continue
        workers = sorted(
            {display_track_label(i.canonical_track_id or "") for i in found}, key=_label_key
        )
        longest = max(found, key=lambda i: _length(bundle, i))
        noun = "regions" if rule_id is RuleId.R8 else "different workers"
        facts.append(
            Fact(
                id=f"COUNT-{rule_id.value}",
                kind="tally",
                rule=rule_id,
                workers=tuple(workers) if rule_id is not RuleId.R8 else (),
                text=(
                    f"{rule_id.value} {_title(bundle, rule_id)}: "
                    f"{_plural(len(found), 'confirmed incident')} involving {len(workers)} {noun} "
                    f"({'workers' if rule_id is not RuleId.R8 else 'regions'} "
                    f"{', '.join(workers)}). Incidents: "
                    f"{' '.join(short[i.incident_id] for i in found)}. First started at "
                    f"{clock(found[0].first_seen_s)} ({short[found[0].incident_id]}, "
                    f"{_subject(found[0])}); last started at {clock(found[-1].first_seen_s)} "
                    f"({short[found[-1].incident_id]}, {_subject(found[-1])}); longest "
                    f"{short[longest.incident_id]} ({_subject(longest)}, "
                    f"{_length(bundle, longest):.1f} s)."
                ),
                keywords=RULE_WORDS[rule_id] + " how many count number total",
            )
        )
    return facts


def _timeline(bundle: RunBundle, short: dict[str, str]) -> list[Fact]:
    """When incidents started, the longest ones and the evidence grades, across all rules."""
    found = sorted(bundle.incidents, key=lambda i: (i.first_seen_s, i.incident_id))
    if not found:
        return []
    minutes = []
    for start in range(0, int(bundle.duration_s) + 1, 60):
        end = min(start + 60, bundle.duration_s)
        inside = [i for i in found if start <= i.first_seen_s < start + 60]
        if start < bundle.duration_s:
            ids = " ".join(short[i.incident_id] for i in inside)
            minutes.append(
                f"{clock(start)}–{clock(end)}: {len(inside)}{' (' + ids + ')' if ids else ''}"
            )
    longest = sorted(found, key=lambda i: -_length(bundle, i))[:3]
    grades: dict[str, list[str]] = {}
    for incident in found:
        grades.setdefault(incident.evidence_quality or "ungraded", []).append(
            short[incident.incident_id]
        )
    return [
        Fact(
            id="TIMELINE",
            kind="timeline",
            text=(
                f"Confirmed incidents by start time (all rules, {len(found)} in total): "
                + "; ".join(minutes)
                + f". First started at {clock(found[0].first_seen_s)} ("
                + ", ".join(
                    f"{short[i.incident_id]} {i.rule_id.value} {_subject(i)}"
                    for i in found
                    if i.first_seen_s == found[0].first_seen_s
                )
                + f"); last started at {clock(found[-1].first_seen_s)} "
                f"({short[found[-1].incident_id]} {found[-1].rule_id.value} "
                f"{_subject(found[-1])}). "
                "Longest: "
                + ", ".join(
                    f"{short[i.incident_id]} {i.rule_id.value} {_subject(i)} "
                    f"{_length(bundle, i):.1f} s"
                    for i in longest
                )
                + "."
            ),
            keywords="when time first last earliest latest longest timeline minute start began",
        ),
        Fact(
            id="EVIDENCE",
            kind="timeline",
            text=(
                f"Evidence quality of the {len(found)} confirmed incidents: "
                + "; ".join(
                    f"{grade} {len(ids)} ({' '.join(ids)})" for grade, ids in sorted(grades.items())
                )
                + "."
            ),
            keywords="evidence quality weak strong moderate confidence how sure",
        ),
    ]


def _label_key(label: str) -> tuple[int, str]:
    return (int(label), "") if label.isdigit() else (1 << 30, label)


def _workers(bundle: RunBundle, short: dict[str, str]) -> list[Fact]:
    """A fact per worker with exposure or an incident; most exposed first."""
    by_worker: dict[str, list[IncidentRecord]] = {}
    for incident in sorted(bundle.incidents, key=lambda i: (i.first_seen_s, i.incident_id)):
        if incident.canonical_track_id and incident.rule_id is not RuleId.R8:
            by_worker.setdefault(incident.canonical_track_id, []).append(incident)

    def incidents_text(found: list[IncidentRecord]) -> str:
        if not found:
            return "no confirmed incident"
        listed = ", ".join(
            f"{short[i.incident_id]} ({i.rule_id.value} {i.title}, from {clock(i.first_seen_s)})"
            for i in found
        )
        rules = {i.rule_id.value for i in found}
        return (
            f"{_plural(len(found), 'confirmed incident')}: {listed}; "
            f"no confirmed incident for rules other than {', '.join(sorted(rules))}"
        )

    facts = []
    ranked: list[str] = []
    top = bundle.insights.top_workers if bundle.insights else []
    for worker in top:
        label = display_track_label(worker.canonical_track_id)
        rules = ", ".join(f"{rule} {seconds:.1f} s" for rule, seconds in worker.by_rule.items())
        found = by_worker.pop(worker.canonical_track_id, [])
        ranked.append(f"worker {label} {worker.exposure_s:.1f} s")
        facts.append(
            Fact(
                id=f"W-{label}",
                kind="worker",
                workers=(label,),
                text=(
                    f"Worker {label}: {worker.exposure_s:.1f} s in hazard conditions ({rules}); "
                    f"{incidents_text(found)}."
                ),
                keywords="worker person who most exposed",
            )
        )
    for track, found in sorted(
        by_worker.items(), key=lambda kv: _label_key(display_track_label(kv[0]))
    ):
        label = display_track_label(track)
        facts.append(
            Fact(
                id=f"W-{label}",
                kind="worker",
                workers=(label,),
                text=f"Worker {label}: {incidents_text(found)}.",
                keywords="worker person who",
            )
        )
    if ranked:
        facts.insert(
            0,
            Fact(
                id="EXPOSED",
                kind="ranking",
                text=(
                    "Workers ranked by time in hazard conditions, most first: "
                    + ", ".join(ranked)
                    + "."
                ),
                keywords="most exposed worker who ranking top longest time hazard",
            ),
        )
    return facts


# ---------------------------------------------------------------------- per-question facts

UNIT = r"(s|secs?|seconds?|mins?|minutes?)"
MOMENT = rf"(?:the\s+)?(?:(\d+):(\d{{2}}(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*-?\s*{UNIT}\b)"


def _seconds(match: re.Match[str], offset: int) -> float:
    minutes, rest, amount, unit = match.group(offset, offset + 1, offset + 2, offset + 3)
    if minutes is not None:
        return int(minutes) * 60 + float(rest)
    return float(amount) * (60 if unit.lower().startswith("m") else 1)


def time_window(question: str, duration_s: float) -> tuple[float, float] | None:
    """The span a question asks about ("first 30 seconds", "after 2:00", "between ...")."""
    q = question.lower()
    if m := re.search(rf"\bbetween\s+{MOMENT}\s+(?:and|to|-)\s+{MOMENT}", q):
        return _seconds(m, 1), _seconds(m, 5)
    if m := re.search(rf"\bfirst\s+{MOMENT}", q):
        return 0.0, _seconds(m, 1)
    if m := re.search(rf"\b(?:last|final)\s+{MOMENT}", q):
        return max(duration_s - _seconds(m, 1), 0.0), duration_s
    if m := re.search(rf"\b(?:after|since|from|past|beyond)\s+{MOMENT}", q):
        return _seconds(m, 1), duration_s
    if m := re.search(rf"\b(?:before|until|by|within)\s+{MOMENT}", q):
        return 0.0, _seconds(m, 1)
    return None


def query_facts(bundle: RunBundle, facts: list[Fact], question: str) -> list[Fact]:
    """Facts computed for this question: a time window's incidents, an unknown worker."""
    extra: list[Fact] = []
    short = incident_ids(bundle)
    window = time_window(question, bundle.duration_s)
    if window:
        start, end = window
        rules = rules_in(question)
        found = sorted(
            (
                i
                for i in bundle.incidents
                if start <= i.first_seen_s < end and (not rules or i.rule_id in rules)
            ),
            key=lambda i: (i.first_seen_s, i.incident_id),
        )
        scope = " ".join(r.value for r in rules) + " " if rules else ""
        workers = {display_track_label(i.canonical_track_id or "") for i in found}
        listed = "; ".join(
            f"{short[i.incident_id]} {i.rule_id.value} {_subject(i)} from {clock(i.first_seen_s)}"
            for i in found
        )
        extra.append(
            Fact(
                id="Q-TIME",
                kind="query",
                workers=tuple(sorted(workers, key=_label_key)),
                text=(
                    f"Confirmed {scope}incidents that started between {clock(start)} and "
                    f"{clock(end)}: {len(found)}"
                    + (
                        f", involving {_plural(len(workers), 'different worker')}: {listed}."
                        if found
                        else " (none)."
                    )
                ),
            )
        )
    known = {w for fact in facts for w in fact.workers}
    for label in dict.fromkeys(WORKER.findall(question)):
        if label not in known:
            extra.append(
                Fact(
                    id=f"Q-W{label}",
                    kind="query",
                    workers=(label,),
                    text=(
                        f"Worker {label}: no confirmed incident and no recorded hazard exposure "
                        "in this clip (no such worker among those with incidents or exposure)."
                    ),
                )
            )
    return extra


def _areas(bundle: RunBundle) -> list[Fact]:
    if not bundle.insights:
        return []
    return [
        Fact(
            id=f"AREA-{n}",
            kind="area",
            rule=area.rule_id,
            text=(
                f"{area.label or area.proposal_id} ({area.rule_id.value}): "
                f"{area.exposure_s:.1f} worker-seconds of exposure by {area.workers} workers."
            ),
            keywords="where area zone hotspot location " + RULE_WORDS[area.rule_id],
        )
        for n, area in enumerate(bundle.insights.area_exposure, 1)
        if area.exposure_s > 0
    ]


def _limits(bundle: RunBundle) -> Fact:
    warnings = [w for w in bundle.manifest.warnings if "battery" not in w]
    synthetic = any("synthetic" in label.lower() for label in bundle.zone_labels.values())
    return Fact(
        id="LIMITS",
        kind="limits",
        text=(
            "Limits: heuristic triage from one fixed camera, not a certified safety system "
            "and not legal advice. Distances are in worker heights (WH), not metres. "
            "Workers are known only by track number; faces are pixelated."
            + (" Some zones are synthetic demo zones." if synthetic else "")
            + (f" Run warnings: {'; '.join(warnings)}." if warnings else "")
        ),
        keywords="limits reliable accuracy trust caveat how sure confidence privacy",
    )


# ---------------------------------------------------------------------- register answers

WORKER = re.compile(r"\bworker\s*#?\s*(\d+)", re.I)
RULE = re.compile(r"\bR([1-8])\b", re.I)
# Phrases that put a question about one rule (a word prefix: "helmet" matches "helmets",
# but "edge" does not match "knowledge").
INTENT = {
    RuleId.R1: ("helmet", "hard hat", "hardhat", "head protection", "headgear"),
    RuleId.R2: ("hi-vis", "high-vis", "high vis", "hi vis", "vest", "conspicu"),
    RuleId.R3: ("restricted", "exclusion", "zone", "intrusion", "intruded"),
    RuleId.R4: ("machine", "excavator", "plant", "proximity", "vehicle"),
    RuleId.R5: ("edge", "fall protection", "guardrail", "opening"),
    RuleId.R6: (
        "worker down",
        "lying",
        "collapse",
        "injur",
        "unconscious",
        "man down",
        "fell",
        "fallen",
        "fall down",
        "falls down",
    ),
    RuleId.R7: ("crane", "suspended", "hook", "lift", "load"),
    RuleId.R8: ("fire", "smoke", "flame", "burning"),
}
INTENT_RE = {
    rule_id: re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + ")", re.I)
    for rule_id, phrases in INTENT.items()
}
COUNTING = re.compile(r"\b(how many|how much|number of|count|total)\b", re.I)
TIMING = re.compile(
    r"\b(when|first|last|earliest|latest|longest|shortest|start\w*|began|timeline|order)\b", re.I
)
EVIDENCE_WORDS = re.compile(r"\b(evidence|weak|strong|moderate)\b", re.I)
FIXING = re.compile(
    r"\b(fix\w*|priorit\w*|action\w*|recommend\w*|controls?|mitigat\w*|improve\w*|"
    r"(should|must|can) (i|we) do|what to do|do (first|next|now))\b",
    re.I,
)
# Questions the report offers as one-click prompts; answered offline from the register.
SUGGESTED = (
    "What are the biggest risks on this site?",
    "What should I fix first?",
    "What could the system not check?",
    "How reliable is this assessment?",
)


def rules_in(question: str) -> list[RuleId]:
    q = question.lower()
    found = [RuleId(f"R{n}") for n in RULE.findall(q)]
    for rule_id, pattern in INTENT_RE.items():
        if rule_id not in found and pattern.search(q):
            found.append(rule_id)
    return found[:2]


def register_answer(bundle: RunBundle, facts: list[Fact], question: str) -> tuple[str, list[Fact]]:
    """Answer from the facts alone: bullet lines, each ending with its citation."""
    q = question.lower()
    by_id = {fact.id: fact for fact in facts}
    risk = bundle.risk
    lines: list[str] = []
    cited: list[Fact] = []

    def say(fact: Fact | None, text: str | None = None) -> None:
        if fact is None or fact in cited:
            return
        lines.append(f"- {text or fact.brief or fact.text} [{fact.id}]")
        cited.append(fact)

    worker = WORKER.search(question)
    rules = rules_in(question)
    query = [f for f in facts if f.kind == "query"]
    for fact in query:  # computed for this question: the most direct answer
        say(fact)
    if worker:
        label = worker.group(1)
        found = [f for f in facts if label in f.workers and f.kind in ("worker", "incident")]
        for fact in found[:6]:
            say(fact)
    elif query:
        pass
    elif rules:
        for rule_id in rules:
            say(by_id.get(f"COUNT-{rule_id.value}"))
            say(
                by_id.get(f"RISK-{rule_id.value}")
                or by_id.get(f"NA-{rule_id.value}")
                or by_id.get(f"CLEAR-{rule_id.value}")
            )
            if not COUNTING.search(q):
                for fact in [f for f in facts if f.kind == "incident" and f.rule is rule_id][:4]:
                    say(fact)
    elif EVIDENCE_WORDS.search(q) and "EVIDENCE" in by_id:
        say(by_id["EVIDENCE"])
    elif TIMING.search(q) and not FIXING.search(q) and "TIMELINE" in by_id:
        say(by_id["TIMELINE"])
    elif COUNTING.search(q) and re.search(r"\b(incidents?|events?|alerts?|violations?)\b", q):
        say(by_id["RUN"])
        say(by_id.get("TIMELINE"))
    elif any(w in q for w in ("not check", "coverage", "miss", "blind", "not assess")):
        for fact in facts:
            if fact.kind == "not_assessed":
                say(fact)
        clear = [f for f in facts if f.kind == "not_observed"]
        if clear:
            lines.append(
                "- Checked and not observed: "
                + ", ".join(f"{f.rule.value} [{f.id}]" for f in clear if f.rule)
            )
            cited.extend(clear)
    elif re.search(r"\b(which|who|most)\b.*\b(worker|exposed|exposure|people|person)", q):
        say(by_id.get("EXPOSED"))
        ranked = [f for f in facts if f.kind == "worker"][:3]  # facts keep exposure order
        for fact in ranked or [by_id["RUN"]]:
            say(fact)
    elif re.search(r"\b(reliab\w*|accura\w*|trust\w*|sure|limit\w*|confiden\w*)\b", q):
        say(by_id["LIMITS"])
    elif FIXING.search(q):
        for row in risk.rows[:3]:
            now = bundle.catalogue.rule(row.rule_id).action_template.strip()
            level, text = row.controls[0]
            step = (
                f"Now: {now} Then ({level.lower()}): {text}."
                if row.incidents
                else f"No confirmed incident, only brief exposures; prevent ({level.lower()}): "
                f"{text}."
            )
            say(
                by_id[f"RISK-{row.rule_id.value}"],
                f"{row.rule_id.value} {row.title} ({row.band} risk, {row.rating}/25). {step}",
            )
    else:
        say(by_id["RUN"])
        for row in risk.rows[:3]:
            say(
                by_id[f"RISK-{row.rule_id.value}"],
                f"{row.rule_id.value} {row.title}: {row.band} risk ({row.rating}/25). {row.why}",
            )
    if not lines:
        say(by_id["RUN"])
    return "\n".join(lines), cited
