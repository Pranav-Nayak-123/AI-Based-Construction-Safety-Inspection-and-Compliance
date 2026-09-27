"""Benchmark stage-1 detector candidates on real frames: speed, memory and yield.

Accuracy (mAP) comes from the labelled test split; this answers the other half of model
selection — does the model fit FR-1's budget (a 120 s clip processed in 5 minutes) on the
development GPU, and how many people does it find at CCTV scale?

    uv run python tools/benchmark_stage1.py data/external/sard/9.mp4 \
        --models yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt yolo26x.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pipeline.intake.frames import FrameSource
from pipeline.vision.detector import Stage1Detector
from shared.config import load_config

FR1_CLIP_SECONDS = 120.0
FR1_BUDGET_SECONDS = 300.0


def benchmark(
    video: Path, models: list[str], frames: int, image_size: int, batch: int
) -> list[dict[str, object]]:
    config = load_config()
    stage1 = config.stage1.model_copy(update={"image_size": image_size, "batch_size": batch})
    source = FrameSource(video, target_fps=config.intake.target_fps)
    images = []
    for decoded in source:
        images.append(decoded.image)
        if len(images) == frames:
            break

    results: list[dict[str, object]] = []
    for name in models:
        detector = Stage1Detector(stage1, Path("artifacts/models/pretrained") / name)
        detector.predict(images[:batch])  # warm-up: CUDA context, cudnn autotune
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timings: list[float] = []
        people: list[int] = []
        small_people: list[int] = []
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            tick = time.perf_counter()
            detections = detector.predict(chunk)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - tick) / len(chunk))
            for frame in detections:
                persons = [d for d in frame if d.class_name == "person" and d.confidence >= 0.5]
                people.append(len(persons))
                small_people.append(sum(1 for d in persons if d.height < 100))
        per_frame = statistics.median(timings)
        clip_frames = FR1_CLIP_SECONDS * config.intake.target_fps
        row: dict[str, object] = {
            "model": name,
            "image_size": image_size,
            "batch": batch,
            "ms_per_frame": round(per_frame * 1000, 1),
            "detector_s_for_120s_clip": round(per_frame * clip_frames, 1),
            "share_of_fr1_budget": round(per_frame * clip_frames / FR1_BUDGET_SECONDS, 2),
            "people_per_frame": round(statistics.mean(people), 2),
            "people_under_100px_per_frame": round(statistics.mean(small_people), 2),
            "peak_gpu_mb": (
                round(torch.cuda.max_memory_allocated() / 1e6)
                if torch.cuda.is_available()
                else None
            ),
            "device": detector.device,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--models", nargs="+", default=["yolo26s.pt", "yolo26m.pt", "yolo26l.pt"])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="Write results as JSON")
    args = parser.parse_args()
    results = benchmark(args.video, args.models, args.frames, args.image_size, args.batch)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
