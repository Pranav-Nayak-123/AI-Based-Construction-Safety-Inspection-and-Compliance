"""Review a run's incidents by hand: real, false alarm (and why), or unsure. Gives precision.

    uv run python tools/review_incidents.py output/<run_id> --out evaluation/labels/sard-9
    uv run python tools/review_incidents.py output/<run_id> --summary     # precision per rule

Each incident is shown at its evidence moment. The worker is in red, and the zone, edge or
machine the rule used is drawn too. Next to it are the original-resolution crop and the
incident's facts. Verdicts are saved after every answer, keyed so they carry over to later
runs of the same clip: a later incident on the same rule, overlapping in time and on the
same worker box, reuses the verdict.

Keys: y  real          n  false alarm (then 1 detection  2 PPE  3 zone/distance  4 other)
      u  unsure        b  back one             q  save and quit
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pipeline.cache.pass1_store import Pass1Store
from pipeline.output.bundle import read_model
from pipeline.render.source_overlay import draw_scene
from shared.identity import display_track_label
from shared.schemas.bundle import INCIDENTS_FILE, RUN_MANIFEST_FILE, SCENE_FILE, IncidentsFile
from shared.schemas.incidents import IncidentRecord
from shared.schemas.run import RunManifest
from shared.schemas.scene import SceneCache

WINDOW = "Incident review"
REASONS = {ord("1"): "detection", ord("2"): "ppe", ord("3"): "zone_or_distance", ord("4"): "other"}
REUSE_IOU = 0.5


def _box(observation) -> tuple[float, float, float, float]:
    b = observation.box_oriented or observation.box_reference
    return (b.x1, b.y1, b.x2, b.y2)


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fingerprint(incident: IncidentRecord, subject_box) -> dict:
    end = incident.resolved_at_s if incident.resolved_at_s is not None else incident.confirmed_at_s
    return {
        "rule_id": incident.rule_id.value,
        "start_s": round(incident.first_seen_s, 2),
        "end_s": round(end, 2),
        "evidence_frame": incident.source_frame_index,
        "box": [round(v, 1) for v in subject_box] if subject_box else None,
    }


def same_event(a: dict, b: dict) -> bool:
    """Is a verdict recorded for incident `a` (possibly another run) valid for `b`?"""
    if a["rule_id"] != b["rule_id"] or a["end_s"] < b["start_s"] or b["end_s"] < a["start_s"]:
        return False
    if a["box"] is None or b["box"] is None:
        return a["evidence_frame"] == b["evidence_frame"]
    return (
        abs(a["evidence_frame"] - b["evidence_frame"]) <= 15
        and _iou(a["box"], b["box"]) >= REUSE_IOU
    )


def load_verdicts(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verdict_for(fp: dict, verdicts: list[dict]) -> dict | None:
    match = None
    for row in verdicts:  # later rows win
        if same_event(row["fingerprint"], fp):
            match = row
    return match


def precision(incidents_fps: list[dict], verdicts: list[dict]) -> dict:
    per_rule: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for fp in incidents_fps:
        row = verdict_for(fp, verdicts)
        per_rule[fp["rule_id"]][row["verdict"] if row else "unreviewed"] += 1
        if row and row["verdict"] == "false_alarm":
            per_rule[fp["rule_id"]][f"false_alarm/{row.get('reason', 'other')}"] += 1
    summary = {}
    for rule, counts in sorted(per_rule.items()):
        decided = counts["real"] + counts["false_alarm"]
        summary[rule] = {
            **dict(counts),
            "precision": round(counts["real"] / decided, 3) if decided else None,
        }
    return summary


def _render(evidence, crop, lines: list[str], prompt: str) -> np.ndarray:
    screen = np.full((720, 1280, 3), 24, np.uint8)
    screen[20:560, 20:980] = cv2.resize(evidence, (960, 540), interpolation=cv2.INTER_AREA)
    if crop is not None and crop.size:
        scale = min(270 / crop.shape[1], 540 / crop.shape[0])
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        screen[20 : 20 + big.shape[0], 995 : 995 + big.shape[1]] = big
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, text in enumerate(lines):
        cv2.putText(screen, text, (20, 590 + 24 * row), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(screen, prompt, (20, 705), font, 0.7, (80, 220, 255), 2, cv2.LINE_AA)
    return screen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("evaluation/labels/sard-9"))
    parser.add_argument("--summary", action="store_true", help="print precision per rule and exit")
    args = parser.parse_args()

    manifest = read_model(args.run_dir / RUN_MANIFEST_FILE, RunManifest)
    incidents = read_model(args.run_dir / INCIDENTS_FILE, IncidentsFile).incidents
    scene = read_model(args.run_dir / SCENE_FILE, SceneCache)
    store = Pass1Store(Path(manifest.diagnostics["work_dir"]) / "pass1.sqlite")
    frames: dict[int, dict] = {}
    for record, observations in store.frames_with_observations():
        frames[record.frame_index] = {o.canonical_track_id: o for o in observations}

    fps = []
    for incident in incidents:
        subject = frames.get(incident.source_frame_index, {}).get(incident.canonical_track_id or "")
        fps.append(fingerprint(incident, _box(subject) if subject else None))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "incident_verdicts.jsonl"
    verdicts = load_verdicts(path)
    if args.summary:
        summary = precision(fps, verdicts)
        result = Path(f"evaluation/results/incident_precision_{manifest.run_id}.json")
        result.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for rule, row in summary.items():
            print(rule, row)
        return

    todo = [
        (i, fp) for i, fp in zip(incidents, fps, strict=True) if verdict_for(fp, verdicts) is None
    ]
    reviewed = len(incidents) - len(todo)
    print(f"{len(incidents)} incidents, {reviewed} already reviewed, {len(todo)} to go")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    position, written = 0, 0
    with path.open("a", encoding="utf-8") as sink:
        while position < len(todo):
            incident, fp = todo[position]
            evidence = cv2.imread(str(args.run_dir / incident.evidence_path))
            if evidence is None:
                position += 1
                continue
            draw_scene(
                evidence,
                [
                    p
                    for p in scene.proposals_for(incident.shot_id)
                    if p.proposal_id in incident.proposal_ids
                ],
            )
            machine = frames.get(incident.source_frame_index, {}).get(incident.object_id or "")
            if machine is not None:
                x1, y1, x2, y2 = map(int, _box(machine))
                cv2.rectangle(evidence, (x1, y1), (x2, y2), (232, 198, 56), 3, cv2.LINE_AA)
            crop = (
                cv2.imread(str(args.run_dir / incident.evidence_crop_path))
                if incident.evidence_crop_path
                else None
            )
            end = (
                incident.resolved_at_s
                if incident.resolved_at_s is not None
                else incident.confirmed_at_s
            )
            lines = [
                f"{position + 1}/{len(todo)}   {incident.rule_id.value}  {incident.title}   "
                f"worker {display_track_label(incident.canonical_track_id or '')}   "
                f"{incident.first_seen_s:.1f}-{end:.1f} s",
                incident.observation_text[:150],
                "helmet: "
                + (incident.helmet.state.value if incident.helmet else "-")
                + "   hi-vis: "
                + (incident.vest.state.value if incident.vest else "-")
                + (
                    f"   distance {incident.distance.lower_wh:.1f}"
                    f"-{incident.distance.upper_wh:.1f} WH"
                    if incident.distance
                    else ""
                ),
                "Is the hazard really there at this moment? "
                "(red = worker, drawn zone/edge, blue = machine)",
            ]
            cv2.imshow(
                WINDOW,
                _render(
                    evidence, crop, lines, "y real   n false alarm   u unsure   b back   q quit"
                ),
            )
            key = cv2.waitKey(0) & 0xFF
            row = None
            if key == ord("y"):
                row = {"verdict": "real"}
            elif key == ord("u"):
                row = {"verdict": "unsure"}
            elif key == ord("n"):
                cv2.imshow(
                    WINDOW,
                    _render(
                        evidence,
                        crop,
                        lines,
                        "why?  1 detection   2 PPE   3 zone/distance   4 other",
                    ),
                )
                reason = REASONS.get(cv2.waitKey(0) & 0xFF, "other")
                row = {"verdict": "false_alarm", "reason": reason}
            elif key == ord("b"):
                position = max(0, position - 1)
                continue
            elif key in (ord("q"), 27):
                break
            if row is None:
                continue
            sink.write(
                json.dumps(
                    {
                        "run_id": manifest.run_id,
                        "incident_id": incident.incident_id,
                        "fingerprint": fp,
                        **row,
                    }
                )
                + "\n"
            )
            sink.flush()
            written += 1
            position += 1
    cv2.destroyAllWindows()
    print(f"saved {written} verdicts to {path}")
    for rule, row in precision(fps, load_verdicts(path)).items():
        print(rule, row)


if __name__ == "__main__":
    main()
