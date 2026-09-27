"""Bake a construction vocabulary into the open-vocabulary YOLOE detector.

    uv sync --extra vision --extra openvocab
    uv run python tools/build_open_vocab.py

YOLOE detects whatever it is prompted with. Text prompts are encoded once here (MobileCLIP)
and saved into `yoloe-26l-construction.pt`, so the pipeline loads a plain detector with
fixed classes and never needs the text encoder. Machinery needs no training data this way:
COCO has no excavators or cranes, and a construction-machinery dataset is not yet licensed
for this project.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

from ultralytics import YOLOE

# Prompt → stage-1 taxonomy class. "hard hat" is kept for PPE verification, not tracking.
VOCABULARY: dict[str, str] = {
    "person": "person",
    "hard hat": "helmet",
    "excavator": "machinery",
    "tower crane": "machinery",
    "mobile crane": "machinery",
    "wheel loader": "machinery",
    "bulldozer": "machinery",
    "backhoe loader": "machinery",
    "forklift": "machinery",
    "road roller": "machinery",
    "concrete pump truck": "machinery",
    "concrete mixer truck": "vehicle",
    "dump truck": "vehicle",
    "truck": "vehicle",
    "car": "vehicle",
}


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", type=Path, default=Path("artifacts/models/pretrained/yoloe-26l-seg.pt")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/models/pretrained/yoloe-26l-construction.pt")
    )
    args = parser.parse_args()

    weights = args.weights.resolve()
    out = args.out.resolve()
    names = list(VOCABULARY)
    # The text encoder (mobileclip2_b.ts) is looked up in the working directory.
    with working_directory(weights.parent):
        model = YOLOE(str(weights))
        model.set_classes(names, model.get_text_pe(names))
        model.save(str(out))
    vocabulary = out.with_suffix(".json")
    vocabulary.write_text(
        json.dumps(
            {
                "source_weights": weights.name,
                "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "prompts": VOCABULARY,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {out} with {len(names)} prompts")


if __name__ == "__main__":
    main()
