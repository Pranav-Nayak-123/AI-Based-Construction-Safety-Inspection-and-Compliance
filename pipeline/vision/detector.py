"""Stage-1 detector: people and machinery on the oriented canvas (v7 §8.1, §33.2)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from pipeline.vision.class_map import resolve_model_classes
from shared.config import Stage1Config
from shared.errors import PipelineError


@dataclass(frozen=True)
class Detection:
    """One box in oriented-canvas pixels (edge coordinates)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weights(config: Stage1Config, root: Path) -> tuple[Path, bool]:
    """The promoted fine-tuned checkpoint if present, else the pretrained fallback."""
    promoted = root / config.model_path
    if promoted.is_file():
        return promoted, True
    fallback = root / config.pretrained_fallback
    if fallback.is_file():
        return fallback, False
    raise PipelineError(
        "STAGE1_WEIGHTS_MISSING",
        f"neither {promoted} nor {fallback} exists; run `safety-tools fetch-models`",
    )


class Stage1Detector:
    def __init__(
        self, config: Stage1Config, weights: Path, *, class_map: str | None = None
    ) -> None:
        self.config = config
        self.weights = weights
        self.device = resolve_device(config.device)
        self.model = YOLO(str(weights))  # task from the checkpoint (YOLOE is a seg model)
        self.classes = resolve_model_classes(self.model.names, class_map or config.class_map)
        self._keep = sorted(self.classes)
        self._sha256: str | None = None

    def model_sha256(self) -> str:
        if self._sha256 is None:
            self._sha256 = file_sha256(self.weights)
        return self._sha256

    def predict(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        results = self.model.predict(
            list(images),
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            max_det=self.config.maximum_detections,
            classes=self._keep,
            device=self.device,
            quantize=32,  # fp32: GTX 16xx and MPS are unreliable in fp16
            verbose=False,
        )
        batch: list[list[Detection]] = []
        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confidence = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            batch.append(
                [
                    Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(score),
                        class_name=self.classes[int(label)],
                    )
                    for box, score, label in zip(xyxy, confidence, classes, strict=True)
                ]
            )
        return batch
