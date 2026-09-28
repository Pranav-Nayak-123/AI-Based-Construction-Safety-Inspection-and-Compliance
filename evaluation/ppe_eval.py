"""Score a run's PPE states against hand labels from `tools/label_ppe.py`.

    uv run python -m evaluation.ppe_eval output/<run_id> --labels evaluation/labels/sard-9

Each labelled sample is matched to the run's person observation in the same frame (IoU at
least 0.5), so labels drawn on one run can score any later run of the same clip. The
state scored is the temporally smoothed one the rules read; "effective" additionally
applies the rules' confidence floor, below which a negative is inconclusive, so it is
exactly what R1/R2 act on. The positive class is the hazard (no helmet, no hi-vis).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from shared.enums import PERSON_CLASS

MATCH_IOU = 0.5
HEIGHT_EDGES = (70.0, 120.0)
HEADS = {"helmet": ("helmet", "no_helmet"), "vest": ("vest", "no_vest")}  # (safe, hazard)


@dataclass(frozen=True)
class Observed:
    box: tuple[float, float, float, float]
    helmet: str
    helmet_confidence: float
    vest: str
    vest_confidence: float


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def size_bin(box: Sequence[float]) -> str:
    height = box[3] - box[1]
    if height < HEIGHT_EDGES[0]:
        return "far"
    return "mid" if height < HEIGHT_EDGES[1] else "near"


def _rates(pairs: Iterable[tuple[str, str]], *, safe: str, hazard: str) -> dict:
    """(label, model) pairs with clear labels; the model may also say unknown."""
    pairs = list(pairs)
    tp = sum(1 for label, model in pairs if label == hazard and model == hazard)
    fp = sum(1 for label, model in pairs if label == safe and model == hazard)
    fn = sum(1 for label, model in pairs if label == hazard and model != hazard)
    decided = [(label, model) for label, model in pairs if model in (safe, hazard)]
    return {
        "n": len(pairs),
        "hazard_labelled": sum(1 for label, _ in pairs if label == hazard),
        "hazard_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "hazard_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "abstained": round(1 - len(decided) / len(pairs), 3) if pairs else None,
        "accuracy_when_decided": (
            round(sum(label == model for label, model in decided) / len(decided), 3)
            if decided
            else None
        ),
    }


def score(
    samples: Sequence[Mapping],
    labels: Mapping[str, Mapping],
    observed: Mapping[int, Sequence[Observed]],
    *,
    confidence_floor: float,
) -> dict:
    report: dict = {"samples": len(samples), "labelled": 0, "unmatched": 0, "not_a_person": 0}
    pairs: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    confusion: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        label = labels.get(sample["sample_id"])
        if label is None:
            continue
        report["labelled"] += 1
        if label["helmet"] == "not_a_person":
            report["not_a_person"] += 1
            continue
        box = sample["box"]
        candidates = observed.get(sample["frame_index"], ())
        best = max(candidates, key=lambda o: iou(o.box, box), default=None)
        if best is None or iou(best.box, box) < MATCH_IOU:
            report["unmatched"] += 1
            continue
        for head, (_, hazard) in HEADS.items():
            truth = label[head]
            model = getattr(best, head)
            confidence = getattr(best, f"{head}_confidence")
            effective = "unknown" if model == hazard and confidence < confidence_floor else model
            confusion[head][f"{truth}->{model}"] += 1
            if truth == "unclear":
                continue
            for group in ("all", size_bin(box)):
                pairs[head][f"{group}/state"].append((truth, model))
                pairs[head][f"{group}/effective"].append((truth, effective))
    for head, (safe, hazard) in HEADS.items():
        report[head] = {
            key: _rates(values, safe=safe, hazard=hazard)
            for key, values in sorted(pairs[head].items())
        }
        report[head]["confusion"] = dict(sorted(confusion[head].items()))
    report["detector_person_precision"] = (
        round(1 - report["not_a_person"] / report["labelled"], 3) if report["labelled"] else None
    )
    return report


def observations_by_frame(store) -> dict[int, list[Observed]]:
    frames: dict[int, list[Observed]] = {}
    for record, observations in store.frames_with_observations():
        rows = []
        for o in observations:
            if o.object_class != PERSON_CLASS:
                continue
            b = o.box_oriented or o.box_reference
            rows.append(
                Observed(
                    box=(b.x1, b.y1, b.x2, b.y2),
                    helmet=o.helmet.state.value if o.helmet else "unknown",
                    helmet_confidence=o.helmet.confidence if o.helmet else 0.0,
                    vest=o.vest.state.value if o.vest else "unknown",
                    vest_confidence=o.vest.confidence if o.vest else 0.0,
                )
            )
        frames[record.frame_index] = rows
    return frames


def main() -> None:
    from pipeline.cache.pass1_store import Pass1Store
    from shared.config import load_config
    from shared.schemas.run import RunManifest
    from tools.label_ppe import load_labels

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--labels", type=Path, default=Path("evaluation/labels/sard-9"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = RunManifest.model_validate_json(
        (args.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    samples = [
        json.loads(line)
        for line in (args.labels / "ppe_sample.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = load_labels(args.labels / "ppe_labels.jsonl")
    store = Pass1Store(Path(manifest.diagnostics["work_dir"]) / "pass1.sqlite")
    report = score(
        samples,
        labels,
        observations_by_frame(store),
        confidence_floor=load_config().rules.ppe_confidence_floor,
    )
    report["run_id"] = manifest.run_id
    out = args.out or Path(f"evaluation/results/ppe_labels_{manifest.run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for head in HEADS:
        for key in ("all/effective", "far/effective", "mid/effective", "near/effective"):
            if key in report[head]:
                print(head, key, report[head][key])
    print("detector person precision:", report["detector_person_precision"], "->", out)


if __name__ == "__main__":
    main()
