"""Stage 2: two-head PPE classifier on tracked person crops (v7 §8.2, §34.7).

Each crop gets independent head (helmet / no_helmet / unknown) and torso (vest / no_vest /
unknown) distributions. The backbone is ImageNet-pretrained EfficientNet-B0 on 256x128
person-shaped input; each head reads the global feature plus a pooled band of the feature
map where its item is worn, so a helmet decision cannot lean on the boots.

Calibrated with one temperature per head, fitted on validation logits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

INPUT_HEIGHT, INPUT_WIDTH = 256, 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HEAD_BAND = (0.0, 0.40)  # fraction of feature-map rows, from the top
TORSO_BAND = (0.15, 0.75)
FEATURES = 1280


def _band(features: torch.Tensor, band: tuple[float, float]) -> torch.Tensor:
    rows = features.shape[2]
    start = int(band[0] * rows)
    stop = max(start + 1, round(band[1] * rows))
    return features[:, :, start:stop, :].mean(dim=(2, 3))


class TwoHeadPPEClassifier(nn.Module):
    def __init__(self, *, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * FEATURES, 3))
        self.torso = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * FEATURES, 3))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maps = self.features(images)
        pooled = maps.mean(dim=(2, 3))
        head = self.head(torch.cat([pooled, _band(maps, HEAD_BAND)], dim=1))
        torso = self.torso(torch.cat([pooled, _band(maps, TORSO_BAND)], dim=1))
        return head, torso


def letterbox_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Fit a person crop into 256x128 without distorting it; pad with mid-grey."""
    height, width = crop_bgr.shape[:2]
    scale = min(INPUT_HEIGHT / height, INPUT_WIDTH / width)
    resized = cv2.resize(
        crop_bgr,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, np.uint8)
    top = (INPUT_HEIGHT - resized.shape[0]) // 2
    left = (INPUT_WIDTH - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def to_tensor(batch_bgr: Sequence[np.ndarray], device: str | None = None) -> torch.Tensor:
    """Letterbox, RGB, ImageNet-normalise; normalisation happens on `device` if given."""
    stack = np.stack([letterbox_crop(crop) for crop in batch_bgr])
    tensor = torch.from_numpy(stack)
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    tensor = tensor.flip(-1).permute(0, 3, 1, 2).float().div_(255.0)  # BGR → RGB, NCHW
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


@dataclass(frozen=True)
class PPEPrediction:
    """Calibrated probabilities, ordered (present, absent, unknown) for each region."""

    head: np.ndarray
    torso: np.ndarray


class PPEClassifier:
    """Inference wrapper around a trained checkpoint."""

    def __init__(self, checkpoint: Path, device: str, *, batch_size: int = 64) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.metadata: dict[str, object] = payload["metadata"]
        self.model = TwoHeadPPEClassifier(pretrained=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.batch_size = batch_size
        temperatures = payload["metadata"].get("temperatures", {"head": 1.0, "torso": 1.0})
        self.head_temperature = float(temperatures["head"])
        self.torso_temperature = float(temperatures["torso"])

    @torch.inference_mode()
    def predict(self, crops: Sequence[np.ndarray]) -> list[PPEPrediction]:
        predictions: list[PPEPrediction] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            head, torso = self.model(to_tensor(chunk, self.device))
            head_p = torch.softmax(head / self.head_temperature, dim=1).cpu().numpy()
            torso_p = torch.softmax(torso / self.torso_temperature, dim=1).cpu().numpy()
            predictions += [PPEPrediction(h, t) for h, t in zip(head_p, torso_p, strict=True)]
        return predictions
