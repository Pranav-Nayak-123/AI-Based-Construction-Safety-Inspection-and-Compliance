"""Narration corpus: sampled incident facts with reference briefings (v7 §45.2).

No teacher LLM runs on the development GPU, so reference briefings come from a curated
paraphrase grammar over the real rule catalogue. Every reference must itself pass the
guardrail. Splits are made before any training:

* train / val / test share one grammar and label bank (in-distribution);
* the blind holdout uses a *different* grammar and unseen zone names, so scores on it
  measure generalisation, not memorised phrasing.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from llm.guardrail import check
from llm.narration import Narration, NarrationFacts
from pipeline.rules.catalogue import CompiledCatalogue, load_catalogue
from shared.enums import RuleId

ZONES_TRAIN = [
    "Crane swing area",
    "Excavation pit",
    "Rebar storage",
    "Loading bay",
    "Pour zone",
    "North stair core",
    "Scaffold base",
    "Lift shaft opening",
    "Material hoist bay",
    "Formwork deck",
    "Pile cap area",
    "Cement silo apron",
    "Batching plant",
    "Trench line",
    "East haul road",
    "Site gate",
    "Tower crane base",
    "Mixer wash area",
    "Precast yard",
    "Level 2 slab edge",
    "Roof edge",
    "Basement ramp",
    "Temporary works area",
    "Cutting station",
    "Welding bay",
    "Fuel store",
    "Laydown area",
    "Dewatering pit",
    "Plant parking",
    "Pedestrian crossing",
    "South perimeter",
    "Stair void",
    "Podium edge",
    "Retaining wall top",
    "Truck turning circle",
    "Skip area",
    "Blockwork store",
    "Survey station",
    "Waste bay",
    "Access road",
]
ZONES_HOLDOUT = [
    "West column grid",
    "Grid C3",
    "Plant room opening",
    "Canopy edge",
    "Shaft 2",
    "Service trench",
    "Hoarding line",
    "Barrier zone B",
    "Loading dock 4",
    "Ramp crest",
    "Atrium void",
    "Pump station",
    "Gantry area",
    "Mezzanine edge",
    "Car park level 1",
    "Cable pull area",
    "Kerb line",
    "Rebar cage bay",
    "Soffit area",
    "Hoist landing",
]


def _seconds(value: float) -> str:
    return f"{value:g}"


def sample_facts(
    rng: random.Random, catalogue: CompiledCatalogue, zones: list[str]
) -> NarrationFacts:
    rule_id = rng.choices(list(RuleId), weights=[3, 2, 3, 3, 2])[0]
    rule = catalogue.rule(rule_id)
    duration = round(min(120.0, max(1.1, rng.lognormvariate(2.0, 0.8))), 1)
    ppe_states = ("helmet", "no_helmet", "unknown", None)
    vest_states = ("vest", "no_vest", "unknown", None)
    helmet = rng.choice(ppe_states)
    vest = rng.choice(vest_states)
    zone = machine = band = distance = None
    if rule_id is RuleId.R1:
        helmet = "no_helmet"
    elif rule_id is RuleId.R2:
        vest = "no_vest"
        if rng.random() < 0.5:
            machine = f"machine {rng.randint(1, 60)}"
            band = rng.choice(("near", "caution"))
        else:
            zone = rng.choice(zones)
    elif rule_id is RuleId.R3:
        zone = rng.choice(zones)
    elif rule_id is RuleId.R4:
        machine = f"machine {rng.randint(1, 60)}"
        band = "near"
        low = round(rng.uniform(0.1, 1.0), 1)
        distance = f"{low:.1f} to {round(rng.uniform(low + 0.1, 1.5), 1):.1f} WH"
    else:
        zone = rng.choice(zones)
        low = round(rng.uniform(0.0, 0.5), 1)
        distance = f"{low:.1f} to {round(rng.uniform(low + 0.1, 0.8), 1):.1f} WH"
    return NarrationFacts(
        rule_id=rule_id.value,
        title=rule.title,
        severity=rule.severity.value,
        basis=rule.basis,
        worker=str(rng.randint(1, 299)),
        start_s=round(rng.uniform(0, 900), 1),
        duration_s=duration,
        status=rng.choices(("resolved", "ongoing at the end of the clip"), weights=[7, 3])[0],
        zone=zone,
        machine=machine,
        band=band,
        distance_wh=distance,
        helmet=helmet,
        vest=vest,
        action=rule.action_template,
    )


# ---------------------------------------------------------------------------- grammars

Writer = Callable[[NarrationFacts, random.Random], Narration]


def _status(facts: NarrationFacts, rng: random.Random, holdout: bool) -> str:
    if facts.status == "resolved":
        return (
            rng.choice([".", ", and then it cleared."])
            if not holdout
            else rng.choice([" before it cleared.", "."])
        )
    return (
        rng.choice(
            [
                ", and it was still going on when the clip ended.",
                ", still ongoing at the end of the clip.",
            ]
        )
        if not holdout
        else (" and had not cleared by the end of the recording.")
    )


def _ppe_caveat(facts: NarrationFacts, rng: random.Random) -> str:
    notes = {
        "R1": [
            "Helmet state is read from video alone, so a cap, hood or poor light can mislead it.",
            "The camera judged the helmet from a small image; confirm it in person.",
        ],
        "R2": [
            "Hi-vis is judged from colour in the video, which lighting and distance can distort.",
            "The vest reading comes from video only and may miss a jacket over the vest.",
        ],
        "R3": [
            "The zone boundary was drawn by the site team and the position is estimated from video.",
            "Position is estimated from the camera image, so check the boundary on site.",
        ],
        "R4": [
            "The distance is estimated from a single camera in worker heights, not measured.",
            "Separation is a video estimate with uncertainty, not a surveyed distance.",
        ],
        "R5": [
            "Edge protection and distance are judged from video and may miss hidden guarding.",
            "The camera cannot see all edge protection, so inspect the edge in person.",
        ],
    }
    return rng.choice(notes[facts.rule_id])


def _summary_train(facts: NarrationFacts, rng: random.Random) -> str:
    w, d, t = facts.worker, _seconds(facts.duration_s), _seconds(facts.start_s)
    end = _status(facts, rng, holdout=False)
    options = {
        "R1": [
            f"Worker {w} was seen without a visible helmet for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} appeared to have no helmet for {d} seconds{end}",
            f"Worker {w} spent {d} seconds without visible head protection starting at {t} s{end}",
        ],
        "R2": [
            f"Worker {w} was in a movement area without visible hi-vis for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} moved through a busy area with no visible vest for {d} seconds{end}",
        ],
        "R3": [
            f"Worker {w} entered {facts.zone} and stayed {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} was inside the restricted {facts.zone} for {d} seconds{end}",
        ],
        "R4": [
            f"Worker {w} stayed {facts.distance_wh} from active {facts.machine} for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} was within the near band of {facts.machine} ({facts.distance_wh}) for {d} seconds{end}",
        ],
        "R5": [
            f"Worker {w} was {facts.distance_wh} from the unguarded edge at {facts.zone} for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} worked close to the open edge at {facts.zone} ({facts.distance_wh}) for {d} seconds{end}",
        ],
    }
    if facts.rule_id == "R2" and facts.machine:
        options["R2"] = [
            f"Worker {w} had no visible hi-vis while in the {facts.band} band of {facts.machine} for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} was near {facts.machine} without a visible vest for {d} seconds{end}",
        ]
    elif facts.rule_id == "R2" and facts.zone:
        options["R2"] = [
            f"Worker {w} was in {facts.zone} without visible hi-vis for {d} seconds from {t} s{end}",
            f"From {t} s, worker {w} crossed {facts.zone} with no visible vest for {d} seconds{end}",
        ]
    return rng.choice(options[facts.rule_id])


def _action_train(facts: NarrationFacts, rng: random.Random) -> str:
    w = facts.worker
    options = {
        "R1": [
            f"Pause the task and check that worker {w} is wearing suitable head protection.",
            f"Stop the work briefly and confirm worker {w} has a proper helmet on.",
        ],
        "R2": [
            f"Check that worker {w} has visible hi-vis before work continues in the movement area.",
            f"Confirm worker {w} is wearing a high-visibility vest before carrying on.",
        ],
        "R3": [
            "Stop entry to the zone and confirm the exclusion boundary before work resumes.",
            f"Have worker {w} leave the area and check the barrier before resuming.",
        ],
        "R4": [
            "Pause movement and agree a safe separation with the plant operator.",
            f"Stop the machine work and have the operator confirm worker {w} is clear.",
        ],
        "R5": [
            "Stop access to the edge and inspect the edge protection before work resumes.",
            f"Keep worker {w} back from the edge until guarding is checked.",
        ],
    }
    return rng.choice(options[facts.rule_id])


def write_train(facts: NarrationFacts, rng: random.Random) -> Narration:
    return Narration(
        summary=_summary_train(facts, rng),
        caveat=_ppe_caveat(facts, rng),
        action=_action_train(facts, rng),
    )


def write_holdout(facts: NarrationFacts, rng: random.Random) -> Narration:
    """A different register: supervisor shorthand, other sentence shapes."""
    w, d, t = facts.worker, _seconds(facts.duration_s), _seconds(facts.start_s)
    end = _status(facts, rng, holdout=True)
    what = {
        "R1": f"no visible helmet on worker {w} for {d} seconds from {t} s",
        "R2": f"worker {w} without visible hi-vis in a movement area for {d} seconds from {t} s",
        "R3": f"worker {w} inside {facts.zone} for {d} seconds from {t} s",
        "R4": f"worker {w} at {facts.distance_wh} from {facts.machine} for {d} seconds from {t} s",
        "R5": f"worker {w} at {facts.distance_wh} from the edge at {facts.zone} for {d} seconds from {t} s",
    }[facts.rule_id]
    action = {
        "R1": f"Check worker {w}'s helmet before the task continues.",
        "R2": f"Make sure worker {w} is wearing hi-vis before going back into the area.",
        "R3": "Clear the zone and recheck the boundary markings.",
        "R4": "Hold the machine and confirm separation with the operator.",
        "R5": "Close off the edge and check the guarding before anyone returns.",
    }[facts.rule_id]
    return Narration(
        summary=f"Camera flagged {what}{end}".replace("..", "."),
        caveat=_ppe_caveat(facts, rng),
        action=action,
    )


def build(out: Path, *, seed: int = 42017) -> dict[str, int]:
    catalogue = load_catalogue()
    rng = random.Random(seed)
    sizes = {"train": 1600, "val": 150, "test": 300}
    counts: dict[str, int] = {}
    out.mkdir(parents=True, exist_ok=True)
    for split, size, zones, writer in (
        *((name, n, ZONES_TRAIN, write_train) for name, n in sizes.items()),
        ("holdout", 120, ZONES_HOLDOUT, write_holdout),
    ):
        rows = []
        while len(rows) < size:
            facts = sample_facts(rng, catalogue, zones)
            reference = writer(facts, rng)
            verdict = check(reference.model_dump_json(), facts)
            if not verdict.passed:
                raise ValueError(f"reference fails its own guardrail: {verdict.failures}")
            rows.append({"facts": facts.model_dump(), "reference": reference.model_dump()})
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        counts[split] = len(rows)
    return counts


def load_split(root: Path, split: str) -> list[tuple[NarrationFacts, Narration]]:
    rows = [
        json.loads(line)
        for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return [(NarrationFacts(**row["facts"]), Narration(**row["reference"])) for row in rows]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/working/narration"))
    parser.add_argument("--seed", type=int, default=42017)
    arguments = parser.parse_args()
    print(build(arguments.out, seed=arguments.seed))
