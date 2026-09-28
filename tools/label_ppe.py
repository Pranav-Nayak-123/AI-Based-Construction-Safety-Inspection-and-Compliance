"""Label a blind, stratified sample of worker crops for helmet and hi-vis (PPE ground truth).

    uv run python tools/label_ppe.py output/sard9-v2 --out evaluation/labels/sard-9

Samples person observations from a run's pass-1 store, spread evenly over time and over
worker size, at most a few per track. Each crop is cut from the original-resolution video
and shown next to the frame it came from. The model's own PPE answer is never shown, so
the labels stay an independent test of it. Labels are saved after every answer and the
tool resumes where it stopped.

Keys: 1 / 2 / 3 answer the current question (helmet, then hi-vis)
      x  not a person (a false detection)      b  back one sample      q  save and quit

Definitions:
    helmet     a rigid hard hat. Straw hats, caps, hoods and bare heads are "no helmet".
    hi-vis     a fluorescent or reflective vest or jacket; ordinary bright clothing is not.
    can't tell the head or torso is hidden, cut off or too blurred to decide.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.cache.pass1_store import Pass1Store
from shared.config import load_config
from shared.enums import PERSON_CLASS
from shared.schemas.run import RunManifest

SEED = 42017
TIME_BINS = 20
HEIGHT_EDGES = (70.0, 120.0)  # px on the 1920x1080 reference canvas: far / mid / near
MAX_PER_TRACK = 3
CONTEXT_PAD = 0.35
WINDOW = "PPE labels"
QUESTIONS = (
    (
        "helmet",
        ("helmet", "no_helmet", "unclear"),
        "HELMET   1 hard hat   2 no hard hat   3 can't tell",
    ),
    ("vest", ("vest", "no_vest", "unclear"), "HI-VIS   1 hi-vis   2 no hi-vis   3 can't tell"),
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    frame_index: int
    source_index: int
    time_s: float
    track_id: str
    box: tuple[float, float, float, float]  # reference canvas

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def height_bin(height: float) -> str:
    if height < HEIGHT_EDGES[0]:
        return "far"
    return "mid" if height < HEIGHT_EDGES[1] else "near"


def sample_observations(store: Pass1Store, count: int, seed: int = SEED) -> list[Sample]:
    """Even over (time bin x size bin) cells, at most MAX_PER_TRACK per track."""
    candidates: list[Sample] = []
    for record, observations in store.frames_with_observations():
        for o in observations:
            if o.object_class != PERSON_CLASS:
                continue
            b = o.box_oriented or o.box_reference
            candidates.append(
                Sample(
                    sample_id=f"f{record.frame_index}-{o.canonical_track_id}",
                    frame_index=record.frame_index,
                    source_index=record.source_index,
                    time_s=round(record.video_time_s, 3),
                    track_id=o.canonical_track_id,
                    box=(round(b.x1, 1), round(b.y1, 1), round(b.x2, 1), round(b.y2, 1)),
                )
            )
    if not candidates:
        return []
    end = max(c.time_s for c in candidates) + 1e-6
    cells: dict[tuple[int, str], list[Sample]] = defaultdict(list)
    for c in candidates:
        cells[(int(TIME_BINS * c.time_s / end), height_bin(c.height))].append(c)
    rng = random.Random(seed)
    for members in cells.values():
        rng.shuffle(members)
    chosen: list[Sample] = []
    per_track: dict[str, int] = defaultdict(int)
    keys = sorted(cells)
    while len(chosen) < count and any(cells.values()):
        for key in keys:  # round-robin keeps every cell represented
            members = cells[key]
            while members:
                candidate = members.pop()
                if per_track[candidate.track_id] < MAX_PER_TRACK:
                    per_track[candidate.track_id] += 1
                    chosen.append(candidate)
                    break
            if len(chosen) >= count:
                break
    rng.shuffle(chosen)  # labelling order carries no time or size pattern
    return chosen


def raw_box(box, raw_size: tuple[int, int], canvas: tuple[int, int]) -> tuple[int, int, int, int]:
    """Reference-canvas box -> original-video pixels (the canvas is a letterbox of it)."""
    (width, height), (cw, ch) = raw_size, canvas
    scale = min(cw / width, ch / height)
    ox, oy = (cw - width * scale) / 2, (ch - height * scale) / 2
    x1, y1, x2, y2 = (
        (box[0] - ox) / scale,
        (box[1] - oy) / scale,
        (box[2] - ox) / scale,
        (box[3] - oy) / scale,
    )
    return int(x1), int(y1), int(x2), int(y2)


def extract(video: Path, samples: list[Sample], canvas: tuple[int, int]) -> dict[str, tuple]:
    """One sequential decode; returns sample_id -> (context image, crop)."""
    wanted: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        wanted[s.source_index].append(s)
    capture = cv2.VideoCapture(str(video))
    out: dict[str, tuple] = {}
    index, last = -1, max(wanted)
    while index < last:
        if not capture.grab():
            break
        index += 1
        if index not in wanted:
            continue
        ok, raw = capture.retrieve()
        if not ok:
            continue
        height, width = raw.shape[:2]
        for s in wanted[index]:
            x1, y1, x2, y2 = raw_box(s.box, (width, height), canvas)
            pad_x, pad_y = (x2 - x1) * CONTEXT_PAD, (y2 - y1) * CONTEXT_PAD
            top, left = max(0, int(y1 - pad_y)), max(0, int(x1 - pad_x))
            crop = raw[
                top : min(height, int(y2 + pad_y)), left : min(width, int(x2 + pad_x))
            ].copy()
            # Outline the worker being asked about: padded crops can hold a neighbour too.
            cv2.rectangle(crop, (x1 - left, y1 - top), (x2 - left, y2 - top), (0, 255, 255), 1)
            context = raw.copy()
            cv2.rectangle(context, (x1, y1), (x2, y2), (0, 255, 255), 6)
            out[s.sample_id] = (cv2.resize(context, (800, 450), interpolation=cv2.INTER_AREA), crop)
        print(f"\rextracting crops: frame {index}/{last}", end="", flush=True)
    capture.release()
    print()
    return out


def _screen(context, crop, progress: str, question: str, answered: str) -> np.ndarray:
    screen = np.full((720, 1280, 3), 24, np.uint8)
    screen[40:490, 20:820] = context
    scale = min(440 / crop.shape[1], 620 / crop.shape[0])
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    y0, x0 = 40 + (620 - big.shape[0]) // 2, 830 + (440 - big.shape[1]) // 2
    screen[y0 : y0 + big.shape[0], x0 : x0 + big.shape[1]] = big
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(screen, progress, (20, 28), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(screen, question, (20, 560), font, 0.9, (80, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(screen, answered, (20, 600), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(
        screen,
        "x not a person   b back   q save & quit   |   straw hat / cap = NO hard hat",
        (20, 690),
        font,
        0.6,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    return screen


def load_labels(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels[row["sample_id"]] = row  # later lines win (re-labelled after "back")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="a finished run: output/<run_id>")
    parser.add_argument("--out", type=Path, default=Path("evaluation/labels/sard-9"))
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()

    manifest = RunManifest.model_validate_json(
        (args.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    config = load_config()
    canvas = (config.intake.target_width, config.intake.target_height)
    args.out.mkdir(parents=True, exist_ok=True)
    sample_path, label_path = args.out / "ppe_sample.jsonl", args.out / "ppe_labels.jsonl"
    if sample_path.is_file():  # the sample is fixed once drawn, so labels stay comparable
        samples = [
            Sample(**{**row, "box": tuple(row["box"])})
            for row in map(json.loads, sample_path.read_text(encoding="utf-8").splitlines())
        ]
    else:
        store = Pass1Store(Path(manifest.diagnostics["work_dir"]) / "pass1.sqlite")
        samples = sample_observations(store, args.count)
        sample_path.write_text(
            "".join(json.dumps(asdict(s)) + "\n" for s in samples), encoding="utf-8"
        )
    labels = load_labels(label_path)
    todo = [s for s in samples if s.sample_id not in labels]
    print(f"{len(samples)} samples, {len(labels)} labelled, {len(todo)} to go")
    if not todo:
        return
    images = extract(Path(manifest.input_video.path), todo, canvas)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    position, done = 0, []
    with label_path.open("a", encoding="utf-8") as sink:
        while position < len(todo):
            sample = todo[position]
            if sample.sample_id not in images:
                position += 1
                continue
            context, crop = images[sample.sample_id]
            answers: dict[str, str] = {}
            step, quit_requested, back = 0, False, False
            while step < len(QUESTIONS):
                key_name, values, prompt = QUESTIONS[step]
                answered = "  ".join(f"{k}: {v}" for k, v in answers.items())
                progress = (
                    f"{len(labels) + len(done) + 1} / {len(samples)}   "
                    f"(height {sample.height:.0f} px, {height_bin(sample.height)})"
                )
                cv2.imshow(WINDOW, _screen(context, crop, progress, prompt, answered))
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("1"), ord("2"), ord("3")):
                    answers[key_name] = values[key - ord("1")]
                    step += 1
                elif key == ord("x"):
                    answers = {"helmet": "not_a_person", "vest": "not_a_person"}
                    step = len(QUESTIONS)
                elif key == ord("b"):
                    back = True
                    break
                elif key in (ord("q"), 27):
                    quit_requested = True
                    break
            if quit_requested:
                break
            if back:
                if done:
                    done.pop()
                    position -= 1
                continue
            row = {"sample_id": sample.sample_id, **answers}
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            done.append(sample.sample_id)
            position += 1
    cv2.destroyAllWindows()
    print(f"saved {len(done)} labels to {label_path}")


if __name__ == "__main__":
    main()
