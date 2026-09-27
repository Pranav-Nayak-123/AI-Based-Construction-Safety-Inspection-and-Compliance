"""Synthetic images and videos for intake and scene tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def textured_scene(width: int = 1280, height: int = 720, seed: int = 7) -> np.ndarray:
    """A static scene with plenty of corners for ORB to lock onto."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 90, np.uint8)
    for _ in range(500):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        w, h = int(rng.integers(8, 70)), int(rng.integers(8, 70))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
    return image


def shifted(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV cannot write mp4v on this machine"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def counter_frames(count: int, size: tuple[int, int] = (320, 180)) -> list[np.ndarray]:
    """Frames whose top-left pixel block encodes their source index."""
    width, height = size
    frames = []
    for index in range(count):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:20, :20] = (index * 3) % 256
        frames.append(frame)
    return frames
