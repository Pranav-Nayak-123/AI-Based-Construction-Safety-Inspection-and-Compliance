"""Fine-tune the two-head PPE classifier (v7 §34.7).

    uv run python tools/train_ppe.py --data data/working/ppe \
        --out artifacts/checkpoints/ppe/effb0-v1

Training crops are close-ups; deployment crops are 40–200 px CCTV workers. Every sample is
therefore randomly degraded to CCTV scale (downscale, JPEG, blur, lighting) before being
fed back at network size. `unknown` is taught from constructed cases where the region is
provably not visible: head cropped away, torso occluded, or the person too small.

Selection uses macro-F1 on present/absent labels averaged over both heads and two scales
(original and 64 px); temperatures are then fitted per head on validation logits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pipeline.data.ppe_crops import HEAD_LABELS, IGNORE, TORSO_LABELS, HeadLabel, TorsoLabel
from pipeline.vision.ppe_model import TwoHeadPPEClassifier, to_tensor

SEED = 42017
UNKNOWN_BELOW_PX = 30  # a worker this small is not assessable for PPE at all


def degrade_to_height(image: np.ndarray, height: int, rng: random.Random) -> np.ndarray:
    """Simulate a CCTV worker `height` px tall: area downscale, then JPEG at that size."""
    scale = height / image.shape[0]
    small = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, height)),
        interpolation=cv2.INTER_AREA,
    )
    quality = rng.randint(30, 90)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else small


def _lighting(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.65, 1.35)  # contrast
    beta = rng.uniform(-35, 35)  # brightness
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gamma = rng.uniform(0.7, 1.4)
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    out = cv2.LUT(out, table)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-4, 4)) % 180  # small: vest colour matters
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.25), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _recolour_head(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Any hue in the head band: helmets come in every colour, and the label does not
    depend on it. The torso keeps its colour, because hi-vis colour is evidence."""
    out = image.copy()
    band = out[: max(1, int(0.32 * out.shape[0]))]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(0, 179)) % 180
    if rng.random() < 0.3:  # white/grey helmets: drain saturation
        hsv[..., 1] = (hsv[..., 1] * rng.uniform(0.1, 0.5)).astype(np.int16)
    out[: band.shape[0]] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        sigma = rng.uniform(0.3, 1.4)
        return cv2.GaussianBlur(image, (0, 0), sigma)
    length = rng.randint(3, 7)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    matrix = cv2.getRotationMatrix2D((length / 2, length / 2), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    return cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))


def augment(
    image: np.ndarray, head: int, torso: int, rng: random.Random
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    # Box jitter, as a tracker's box would have.
    if rng.random() < 0.5:
        dx, dy = rng.uniform(-0.06, 0.06) * width, rng.uniform(-0.04, 0.04) * height
        grow = rng.uniform(0.92, 1.08)
        matrix = np.float32([[grow, 0, dx], [0, grow, dy]])
        image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        image = image[:, ::-1]
    # Constructed `unknown`: the region is really not in the crop.
    roll = rng.random()
    if roll < 0.08 and head != IGNORE:
        image = image[int(rng.uniform(0.32, 0.45) * height) :]
        head = HeadLabel.UNKNOWN
    elif roll < 0.16 and torso != IGNORE:
        image = image.copy()
        top, bottom = int(0.22 * height), int(rng.uniform(0.7, 0.85) * height)
        patch = np.random.default_rng(rng.randint(0, 2**31)).integers(
            0, 255, (bottom - top, width, 3), dtype=np.uint8
        )
        image[top:bottom] = cv2.GaussianBlur(patch, (0, 0), 6)
        torso = TorsoLabel.UNKNOWN
    image = np.ascontiguousarray(image)
    if rng.random() < 0.6:
        image = _recolour_head(image, rng)
    if rng.random() < 0.8:
        image = _lighting(image, rng)
    if rng.random() < 0.3:
        image = _blur(image, rng)
    if rng.random() < 0.75:
        target = int(rng.uniform(18, 180))
        image = degrade_to_height(image, target, rng)
        if target < UNKNOWN_BELOW_PX:
            head = HeadLabel.UNKNOWN if head != IGNORE else IGNORE
            torso = TorsoLabel.UNKNOWN if torso != IGNORE else IGNORE
    return image, head, torso


class CropDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], *, train: bool, seed: int = SEED) -> None:
        self.root = root
        self.records = records
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image_path"]))
        head = HEAD_LABELS.index(record["head_state"]) if record["head_state"] else IGNORE
        torso = TORSO_LABELS.index(record["torso_state"]) if record["torso_state"] else IGNORE
        if self.train:
            rng = random.Random(hash((self.seed, self.epoch, index)))
            image, head, torso = augment(image, head, torso, rng)
        return to_tensor([image])[0], head, torso


def class_weights(labels: list[int], classes: int = 3, cap: float = 3.0) -> torch.Tensor:
    counts = np.bincount([label for label in labels if label != IGNORE], minlength=classes)
    counts = np.maximum(counts, 1)
    weights = np.minimum(np.sqrt(counts.max() / counts), cap)
    return torch.tensor(weights, dtype=torch.float32)


@torch.inference_mode()
def collect_logits(
    model: nn.Module, images: list[np.ndarray], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    heads, torsos = [], []
    for start in range(0, len(images), 128):
        head, torso = model(to_tensor(images[start : start + 128]).to(device))
        heads.append(head.cpu())
        torsos.append(torso.cpu())
    return torch.cat(heads), torch.cat(torsos)


def binary_report(probabilities: np.ndarray, labels: np.ndarray, names: tuple[str, ...]) -> dict:
    """Metrics over labelled present/absent samples; `unknown` predictions count as abstain."""
    mask = labels != IGNORE
    probabilities, labels = probabilities[mask], labels[mask]
    predicted = probabilities.argmax(axis=1)
    report: dict[str, float] = {"n": int(mask.sum())}
    f1s = []
    for cls in (0, 1):
        tp = int(((predicted == cls) & (labels == cls)).sum())
        fp = int(((predicted == cls) & (labels != cls)).sum())
        fn = int(((predicted != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[f"{names[cls]}_precision"] = round(precision, 4)
        report[f"{names[cls]}_recall"] = round(recall, 4)
        f1s.append(f1)
    report["macro_f1"] = round(float(np.mean(f1s)), 4)
    report["abstain_rate"] = round(float((predicted == 2).mean()), 4)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    report["ece"] = round(float(ece), 4)
    return report


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE
    logits, labels = logits[mask], labels[mask]
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)
    loss_fn = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.25, 10.0))


def load_split(root: Path, records: list[dict], split: str, height: int | None, seed: int):
    rng = random.Random(seed)
    images, heads, torsos = [], [], []
    for record in records:
        if record["split"] != split:
            continue
        image = cv2.imread(str(root / record["image_path"]))
        if height is not None and image.shape[0] > height:
            image = degrade_to_height(image, height, rng)
        images.append(image)
        heads.append(HEAD_LABELS.index(record["head_state"]) if record["head_state"] else IGNORE)
        torsos.append(
            TORSO_LABELS.index(record["torso_state"]) if record["torso_state"] else IGNORE
        )
    return images, torch.tensor(heads), torch.tensor(torsos)


def evaluate(model, split_data, device, temperatures=(1.0, 1.0)) -> dict:
    images, heads, torsos = split_data
    head_logits, torso_logits = collect_logits(model, images, device)
    head_p = torch.softmax(head_logits / temperatures[0], dim=1).numpy()
    torso_p = torch.softmax(torso_logits / temperatures[1], dim=1).numpy()
    return {
        "head": binary_report(head_p, heads.numpy(), HEAD_LABELS),
        "torso": binary_report(torso_p, torsos.numpy(), TORSO_LABELS),
    }


def unknown_behaviour(model, root: Path, records: list[dict], device, temperatures) -> dict:
    """How often constructed not-visible crops are recognised as unknown."""
    rng = random.Random(SEED + 7)
    headless, occluded = [], []
    for record in records:
        if record["split"] != "test":
            continue
        image = cv2.imread(str(root / record["image_path"]))
        height = image.shape[0]
        headless.append(image[int(0.4 * height) :])
        covered = image.copy()
        covered[int(0.22 * height) : int(0.8 * height)] = cv2.GaussianBlur(
            np.random.default_rng(rng.randint(0, 2**31)).integers(
                0, 255, covered[int(0.22 * height) : int(0.8 * height)].shape, np.uint8
            ),
            (0, 0),
            6,
        )
        occluded.append(covered)
    head_logits, _ = collect_logits(model, headless, device)
    _, torso_logits = collect_logits(model, occluded, device)
    head_unknown = torch.softmax(head_logits / temperatures[0], 1).argmax(1) == 2
    torso_unknown = torch.softmax(torso_logits / temperatures[1], 1).argmax(1) == 2
    return {
        "headless_crops_predicted_unknown": round(float(head_unknown.float().mean()), 4),
        "occluded_torsos_predicted_unknown": round(float(torso_unknown.float().mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/working/ppe"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/checkpoints/ppe/effb0-v1"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--promote", type=Path, default=Path("artifacts/models/ppe/best.pt"))
    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    records = [json.loads(line) for line in (args.data / "records.jsonl").read_text().splitlines()]
    train_records = [r for r in records if r["split"] == "train"]
    dataset = CropDataset(args.data, train_records, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    head_weights = class_weights(
        [HEAD_LABELS.index(r["head_state"]) for r in train_records if r["head_state"]]
        + [HeadLabel.UNKNOWN] * (len(train_records) // 8)
    )
    torso_weights = class_weights(
        [TORSO_LABELS.index(r["torso_state"]) for r in train_records if r["torso_state"]]
        + [TorsoLabel.UNKNOWN] * (len(train_records) // 8)
    )
    head_loss = nn.CrossEntropyLoss(
        head_weights.to(device), ignore_index=IGNORE, label_smoothing=0.05
    )
    torso_loss = nn.CrossEntropyLoss(
        torso_weights.to(device), ignore_index=IGNORE, label_smoothing=0.05
    )

    model = TwoHeadPPEClassifier(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(loader)
    warmup = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(step, steps) / steps)),
    )

    val_sets = {
        "original": load_split(args.data, records, "val", None, SEED),
        "cctv_64px": load_split(args.data, records, "val", 64, SEED),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, history = -1.0, -1, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        dataset.epoch = epoch
        total = 0.0
        for images, heads, torsos in loader:
            images, heads, torsos = images.to(device), heads.to(device), torsos.to(device)
            head_logits, torso_logits = model(images)
            loss = head_loss(head_logits, heads) + torso_loss(torso_logits, torsos)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss)
        scores = {name: evaluate(model, data, device) for name, data in val_sets.items()}
        score = float(
            np.mean([s[h]["macro_f1"] for s in scores.values() for h in ("head", "torso")])
        )
        history.append(
            {"epoch": epoch, "loss": round(total / len(loader), 4), "val_score": round(score, 4)}
        )
        print(f"epoch {epoch:2d} loss {total / len(loader):.3f} val {score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), args.out / "weights.pt")

    model.load_state_dict(torch.load(args.out / "weights.pt", map_location=device))
    val_head, val_torso = collect_logits(
        model, val_sets["cctv_64px"][0] + val_sets["original"][0], device
    )
    temperatures = (
        fit_temperature(val_head, torch.cat([val_sets["cctv_64px"][1], val_sets["original"][1]])),
        fit_temperature(val_torso, torch.cat([val_sets["cctv_64px"][2], val_sets["original"][2]])),
    )
    test = {
        f"test_{name}": evaluate(
            model, load_split(args.data, records, "test", height, SEED + 1), device, temperatures
        )
        for name, height in (
            ("original", None),
            ("cctv_96px", 96),
            ("cctv_64px", 64),
            ("cctv_48px", 48),
        )
    }
    metrics = {
        "backbone": "efficientnet_b0 (ImageNet)",
        "input": "256x128 letterboxed person crop",
        "train_crops": len(train_records),
        "best_epoch": best_epoch,
        "best_val_score": round(best_score, 4),
        "temperatures": {"head": round(temperatures[0], 4), "torso": round(temperatures[1], 4)},
        **test,
        "unknown_behaviour": unknown_behaviour(model, args.data, records, device, temperatures),
        "history": history,
        "hyperparameters": {k: str(v) for k, v in vars(args).items()},
        "train_seconds": round(time.perf_counter() - started, 1),
        "seed": SEED,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "metadata": {
            "labels": {"head": list(HEAD_LABELS), "torso": list(TORSO_LABELS)},
            "temperatures": {"head": temperatures[0], "torso": temperatures[1]},
            "backbone": "efficientnet_b0",
            "input_size": [256, 128],
            "dataset": "ultralytics-construction-ppe (box-association labels)",
        },
    }
    torch.save(checkpoint, args.out / "best.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    args.promote.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.out / "best.pt", args.promote)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
