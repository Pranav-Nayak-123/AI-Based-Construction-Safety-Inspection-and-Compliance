"""L5 guardrail: eight deterministic checks on every generated narration (FR-7).

No model is involved. An output that fails any check is discarded and the deterministic
template is shown instead; the failure is logged with the check that caught it.

    1 shape          valid JSON object, exactly the three string fields, one sentence each
    2 numbers        every number appears in the facts (after rounding to one decimal)
    3 names          every worker/machine id mentioned is the one in the facts
    4 citations      no rule, section, act or clause references (the catalogue owns those)
    5 verdicts       no legal verdict language (violation, illegal, breach, ...)
    6 units          no metric distances: the system only knows worker heights
    7 action         the action keeps the catalogue action's subject and asks for a step
    8 tone           no certainty the evidence cannot carry, no downplaying the severity
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from llm.narration import OUTPUT_KEYS, Narration, NarrationFacts

MAX_FIELD_CHARS = 260
NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
ID_MENTION = re.compile(r"\b(worker|machine|vehicle|machinery|track)\s*#?\s*(\d+)", re.I)
CITATION = re.compile(
    r"\b(rule|rules|section|sections|sub-?section|clause|act|regulation|bocw|article|"
    r"schedule)\s*\(?\d|\bBOCW\b|\bAct,?\s+\d{4}|§",
    re.I,
)
VERDICT = re.compile(
    r"\b(violat\w*|illegal\w*|unlawful\w*|non-?complian\w*|breach\w*|liab\w*|penalt\w*|"
    r"prosecut\w*|offen[cs]e\w*|guilty|negligen\w*|fined)\b",
    re.I,
)
METRIC = re.compile(r"\d\s*(m|metres?|meters?|cm|mm|ft|feet|foot|km)\b", re.I)
OVERCLAIM = re.compile(
    r"\b(definitely|certainly|confirmed violation|proven|undeniabl\w*|without (a )?doubt|"
    r"no doubt|guaranteed)\b",
    re.I,
)
DOWNPLAY = re.compile(
    r"\b(minor|trivial|negligible|no (real )?risk|not (a )?(big )?concern|"
    r"harmless|ignore)\b",
    re.I,
)
IGNORE_ACTION = re.compile(r"\b(no action|nothing (needs|to do)|ignore|disregard)\b", re.I)

# What each rule's action must still be about, whatever the wording.
ACTION_SUBJECT = {
    "R1": ("helmet", "head protection", "hard hat", "headgear"),
    "R2": ("hi-vis", "high-visibility", "high visibility", "vest", "conspicu", "visib"),
    "R3": ("zone", "area", "boundary", "entry", "exclusion", "barrier"),
    "R4": ("machine", "plant", "operator", "separation", "distance", "equipment", "clear"),
    "R5": ("edge", "fall", "protection", "guard", "access", "rail"),
}


@dataclass
class GuardrailResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    narration: Narration | None = None


def _sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _fact_numbers(facts: NarrationFacts) -> set[str]:
    numbers: set[str] = set()
    for value in facts.model_dump().values():
        if value is None:
            continue
        for token in NUMBER.findall(str(value)):
            numbers.add(_normalise(token))
    return numbers


def _normalise(token: str) -> str:
    value = float(token)
    return f"{round(value, 1):g}"


def check(raw: str, facts: NarrationFacts) -> GuardrailResult:
    """Run all eight checks; collect every failure, not just the first."""
    failures: list[str] = []
    # 1 shape
    try:
        payload = json.loads(_extract_json(raw))
    except (ValueError, TypeError):
        return GuardrailResult(False, ["shape: not a JSON object"])
    if not isinstance(payload, dict) or tuple(sorted(payload)) != tuple(sorted(OUTPUT_KEYS)):
        return GuardrailResult(False, [f"shape: keys must be exactly {OUTPUT_KEYS}"])
    if not all(isinstance(payload[k], str) and payload[k].strip() for k in OUTPUT_KEYS):
        return GuardrailResult(False, ["shape: every field must be a non-empty string"])
    for key in OUTPUT_KEYS:
        text = payload[key]
        if len(text) > MAX_FIELD_CHARS or _sentences(text) > 2:
            failures.append(f"shape: {key} must be one short sentence")
    narration = Narration(**{k: payload[k].strip() for k in OUTPUT_KEYS})
    text = " ".join(payload[k] for k in OUTPUT_KEYS)

    # 2 numbers
    allowed = _fact_numbers(facts)
    for token in NUMBER.findall(text):
        if _normalise(token) not in allowed:
            failures.append(f"numbers: {token} is not in the facts")
    # 3 names
    known = {facts.worker}
    if facts.machine:
        known.update(NUMBER.findall(facts.machine))
    for _, number in ID_MENTION.findall(text):
        if number not in known:
            failures.append(f"names: id {number} is not in the facts")
    # 4 citations
    if CITATION.search(text):
        failures.append("citations: references are attached by the catalogue, not written")
    # 5 verdicts
    if VERDICT.search(text):
        failures.append("verdicts: legal or compliance verdict language")
    # 6 units
    if METRIC.search(text):
        failures.append("units: metric distance; only worker heights (WH) are known")
    # 7 action
    subjects = ACTION_SUBJECT.get(facts.rule_id, ())
    action = payload["action"].lower()
    if IGNORE_ACTION.search(action) or (subjects and not any(s in action for s in subjects)):
        failures.append("action: does not keep the catalogue action's subject")
    # 8 tone
    if OVERCLAIM.search(text):
        failures.append("tone: certainty the video evidence cannot carry")
    if facts.severity in ("high", "critical") and DOWNPLAY.search(text):
        failures.append("tone: downplays a high-severity observation")

    return GuardrailResult(not failures, failures, narration if not failures else None)


def _extract_json(raw: str) -> str:
    """Take the first {...} block, tolerating code fences around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return raw[start : end + 1]
