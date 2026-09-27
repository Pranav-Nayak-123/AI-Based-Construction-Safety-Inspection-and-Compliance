"""L5 guardrail: each of the eight checks catches its corruption (FR-7)."""

from __future__ import annotations

import json

import pytest

from llm.guardrail import check
from llm.narration import NarrationFacts, template_narration

FACTS = NarrationFacts(
    rule_id="R4",
    title="Approximate machinery proximity",
    severity="high",
    basis="heuristic",
    worker="53",
    start_s=31.2,
    duration_s=1.7,
    status="resolved",
    machine="machine 11",
    band="near",
    distance_wh="0.4 to 0.6 WH",
    helmet="helmet",
    action="Pause movement and verify separation with the plant operator.",
)
GOOD = {
    "summary": "Worker 53 stayed 0.4 to 0.6 WH from machine 11 for 1.7 seconds from 31.2 s.",
    "caveat": "The distance is estimated from one camera, not measured.",
    "action": "Pause movement and agree a safe separation with the plant operator.",
}


def corrupt(**changes: str) -> str:
    return json.dumps({**GOOD, **changes})


def test_a_grounded_narration_passes() -> None:
    result = check(json.dumps(GOOD), FACTS)
    assert result.passed, result.failures
    assert result.narration is not None


def test_template_always_passes() -> None:
    assert check(template_narration(FACTS).model_dump_json(), FACTS).passed


@pytest.mark.parametrize(
    ("raw", "caught_by"),
    [
        ("not json at all", "shape"),
        (json.dumps({"summary": "x", "action": "y"}), "shape"),
        (corrupt(caveat="One. Two. Three sentences here."), "shape"),
        (corrupt(summary="Worker 53 stayed near machine 11 for 9.5 seconds."), "numbers"),
        (corrupt(summary="Worker 54 stayed 0.4 to 0.6 WH from machine 11."), "names"),
        (corrupt(caveat="See BOCW Rule 125(h) for the requirement."), "citations"),
        (corrupt(caveat="This is a clear violation of site rules."), "verdicts"),
        (corrupt(caveat="The worker was about 1 m from the excavator."), "units"),
        (corrupt(action="No action is needed at this time."), "action"),
        (corrupt(action="Tell the worker to take a break."), "action"),
        (corrupt(caveat="The camera definitely saw the worker too close."), "tone"),
        (corrupt(caveat="This is a minor issue."), "tone"),
    ],
)
def test_each_corruption_is_caught_by_its_check(raw: str, caught_by: str) -> None:
    result = check(raw, FACTS)
    assert not result.passed
    assert any(failure.startswith(caught_by) for failure in result.failures), result.failures


def test_json_inside_code_fences_is_accepted() -> None:
    assert check("```json\n" + json.dumps(GOOD) + "\n```", FACTS).passed


def test_numbers_from_the_facts_may_be_reformatted() -> None:
    assert check(corrupt(summary="Worker 53 was near machine 11 for 1.70 seconds."), FACTS).passed
