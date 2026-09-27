"""Narration runtime: the guardrail decides what a user sees (no model needed)."""

from __future__ import annotations

import json

from llm.narration import NarrationFacts, template_narration
from llm.runtime.narrator import narrate

FACTS = NarrationFacts(
    rule_id="R1",
    title="Apparent missing helmet",
    severity="high",
    basis="visual",
    worker="7",
    start_s=12.5,
    duration_s=4.0,
    status="resolved",
    helmet="no_helmet",
    action="Pause the task and verify that suitable head protection is being worn.",
)


def _reply(summary: str, caveat: str, action: str) -> str:
    return json.dumps({"summary": summary, "caveat": caveat, "action": action})


def test_a_reply_that_passes_every_check_is_shown() -> None:
    raw = _reply(
        "Worker 7 appeared without a helmet from 12.5 s for 4.0 s before it resolved.",
        "Video can show a missing helmet but not why it was removed.",
        "Pause the task and check that worker 7 is wearing head protection.",
    )
    [narration] = narrate([FACTS], [raw], "qwen+lora")
    assert narration.source == "model"
    assert narration.failed_checks == ()
    assert narration.summary.startswith("Worker 7")


def test_an_invented_number_falls_back_to_the_template() -> None:
    raw = _reply(
        "Worker 7 appeared without a helmet for 9 minutes near crane 3.",
        "Video evidence only.",
        "Check the helmet.",
    )
    [narration] = narrate([FACTS], [raw], "qwen+lora")
    assert narration.source == "template"
    assert "numbers" in narration.failed_checks
    assert narration.summary == template_narration(FACTS).summary


def test_verdicts_and_metres_are_rejected() -> None:
    raw = _reply(
        "Worker 7 was in violation of the helmet rule.",
        "He stood 2 m from the edge.",
        "Pause the task and verify head protection.",
    )
    [narration] = narrate([FACTS], [raw], None)
    assert narration.source == "template"
    assert {"verdicts", "units"} <= set(narration.failed_checks)


def test_garbage_is_not_json() -> None:
    [narration] = narrate([FACTS], ["Sure! Here is the briefing"], None)
    assert narration.source == "template"
    assert narration.failed_checks == ("shape",)
