"""Build the PPE person-crop dataset from a YOLO-format PPE dataset (v7 §34.5).

    uv run python tools/build_ppe_crops.py data/external/construction-ppe \
        --out data/working/ppe

Writes `crops/<split>/<id>.jpg` and `records.jsonl` (one labelled person per line). Crops
keep up to 320 px of height so training can degrade them to CCTV scale itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import yaml

from pipeline.data.ppe_crops import (
    HEAD_LABELS,
    IGNORE,
    TORSO_LABELS,
    Box,
    associate,
)

MAX_CROP_HEIGHT = 320
MIN_PERSON_HEIGHT = 48  # below this even the source annotation is not assessable
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Native class names of Ultralytics Construction-PPE (`none` = torso without a vest).
CLASS_ROLES = {
    "Person": "person",
    "helmet": "helmet",
    "no_helmet": "no_helmet",
    "vest": "vest",
    "none": "no_vest",
}


def _read_boxes(label_path: Path, names: dict[int, str], width: int, height: int) -> dict:
    boxes: dict[str, list[Box]] = {role: [] for role in set(CLASS_ROLES.values())}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        role = CLASS_ROLES.get(names[int(parts[0])])
        if role is None:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        boxes[role].append(
            Box(
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        )
    return boxes


def build(dataset: Path, out: Path) -> Counter[str]:
    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}
    stats: Counter[str] = Counter()
    records = []
    for split in ("train", "val", "test"):
        (out / "crops" / split).mkdir(parents=True, exist_ok=True)
        for label_path in sorted((dataset / "labels" / split).glob("*.txt")):
            image_path = next(
                (
                    p
                    for p in (dataset / "images" / split).glob(label_path.stem + ".*")
                    if p.suffix.lower() in IMAGE_SUFFIXES
                ),
                None,
            )
            if image_path is None:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            boxes = _read_boxes(label_path, names, width, height)
            for labels in associate(
                boxes["person"],
                boxes["helmet"],
                boxes["no_helmet"],
                boxes["vest"],
                boxes["no_vest"],
            ):
                if labels.person.height < MIN_PERSON_HEIGHT:
                    stats["skipped_small"] += 1
                    continue
                if labels.head == IGNORE and labels.torso == IGNORE:
                    stats["skipped_unlabelled"] += 1
                    continue
                crop_box = labels.person.padded(width, height)
                crop = image[
                    int(crop_box.y1) : int(crop_box.y2), int(crop_box.x1) : int(crop_box.x2)
                ]
                if crop.size == 0:
                    continue
                if crop.shape[0] > MAX_CROP_HEIGHT:
                    scale = MAX_CROP_HEIGHT / crop.shape[0]
                    crop = cv2.resize(
                        crop,
                        (max(1, round(crop.shape[1] * scale)), MAX_CROP_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                digest = hashlib.sha256(f"{image_path.name}:{crop_box}".encode()).hexdigest()[:16]
                crop_path = out / "crops" / split / f"{digest}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                head = HEAD_LABELS[labels.head] if labels.head != IGNORE else None
                torso = TORSO_LABELS[labels.torso] if labels.torso != IGNORE else None
                records.append(
                    {
                        "crop_id": digest,
                        "image_path": str(crop_path.relative_to(out)).replace("\\", "/"),
                        "split": split,
                        "source_dataset": "ultralytics-construction-ppe",
                        "source_image": image_path.name,
                        "person_height_px": round(labels.person.height, 1),
                        "head_state": head,
                        "torso_state": torso,
                        "label_source": "box_association",
                    }
                )
                stats[f"{split}:head:{head}"] += 1
                stats[f"{split}:torso:{torso}"] += 1
                stats[f"{split}:crops"] += 1
    with (out / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/working/ppe"))
    args = parser.parse_args()
    stats = build(args.dataset, args.out)
    for key in sorted(stats):
        print(f"{key:28s} {stats[key]}")


if __name__ == "__main__":
    main()
