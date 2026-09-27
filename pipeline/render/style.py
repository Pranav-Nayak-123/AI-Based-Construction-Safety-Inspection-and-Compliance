"""Shared visual language for the video: palette, Inter typography, panels and chips.

Text is rendered with Pillow in the Inter typeface (SIL OFL, bundled in `artifacts/fonts`)
and cached as alpha sprites, so the anti-aliased type costs almost nothing per frame.
Colours are BGR for OpenCV. Every status is carried by words as well as colour.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "fonts" / "InterVariable.ttf"

BGR = tuple[int, int, int]


def hex_bgr(value: str) -> BGR:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Palette (dark, warm-neutral surfaces; status hues chosen to stay distinct for common
# colour-vision deficiencies and always paired with a word).
BACKGROUND = hex_bgr("#0E1014")
SURFACE = hex_bgr("#161A20")
SURFACE_RAISED = hex_bgr("#1D222A")
STROKE = hex_bgr("#2A303A")
TEXT = hex_bgr("#ECEEF1")
TEXT_MUTED = hex_bgr("#98A0AB")
TEXT_FAINT = hex_bgr("#6B7380")
ACCENT = hex_bgr("#5B8DEF")
OK = hex_bgr("#3CCB8F")
WARN = hex_bgr("#F2A93B")
DANGER = hex_bgr("#F0524A")
CRITICAL = hex_bgr("#C2271D")
MACHINE = hex_bgr("#38C6E8")
EDGE = hex_bgr("#FF7A2F")
WORKER = hex_bgr("#DDE3EA")
HELMET_YELLOW = hex_bgr("#F6C343")

SEVERITY = {"critical": CRITICAL, "high": DANGER, "medium": WARN, "low": hex_bgr("#9BCB3C")}


@functools.lru_cache(maxsize=64)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    if not FONT_PATH.is_file():
        return ImageFont.load_default(size=size)
    face = ImageFont.truetype(str(FONT_PATH), size)
    try:
        face.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return face


@functools.lru_cache(maxsize=8192)
def _sprite(text: str, size: int, weight: str) -> tuple[np.ndarray, int, int]:
    """Alpha mask of `text` plus its left/top bearing offsets."""
    face = font(size, weight)
    left, top, right, bottom = face.getbbox(text)
    width, height = max(1, right - left), max(1, bottom - top)
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).text((-left, -top), text, font=face, fill=255)
    return np.asarray(image, np.float32) / 255.0, left, top


def text_size(text: str, size: int, weight: str = "Regular") -> tuple[int, int]:
    face = font(size, weight)
    left, top, right, bottom = face.getbbox(text)
    return right - left, bottom - top


def line_height(size: int) -> int:
    return round(size * 1.35)


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    size: int,
    color: BGR,
    *,
    weight: str = "Regular",
    anchor: str = "lt",
    alpha: float = 1.0,
) -> tuple[int, int]:
    """Draw `text`; anchor is h∈{l,m,r} + v∈{t,m,b}. Returns the drawn width and height."""
    if not text:
        return 0, 0
    mask, _, _ = _sprite(text, size, weight)
    height, width = mask.shape
    x, y = origin
    if anchor[0] == "m":
        x -= width // 2
    elif anchor[0] == "r":
        x -= width
    if anchor[1] == "m":
        y -= height // 2
    elif anchor[1] == "b":
        y -= height
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, image.shape[1]), min(y + height, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return width, height
    region = image[y0:y1, x0:x1].astype(np.float32)
    weight_map = mask[y0 - y : y1 - y, x0 - x : x1 - x, None] * alpha
    region += (np.array(color, np.float32) - region) * weight_map
    image[y0:y1, x0:x1] = region.astype(np.uint8)
    return width, height


def wrap(text: str, size: int, max_width: int, weight: str = "Regular") -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if text_size(candidate, size, weight)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: BGR,
    *,
    radius: int = 8,
    alpha: float = 1.0,
    border: BGR | None = None,
) -> None:
    """Filled rounded rectangle, optionally translucent and outlined (anti-aliased)."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    if x2 <= x1 or y2 <= y1:
        return
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    # Work on the rectangle's own region: blending a copy of the whole frame per chip is
    # what makes naive translucent UI slow.
    rx1, ry1 = max(x1 - 1, 0), max(y1 - 1, 0)
    rx2, ry2 = min(x2 + 2, image.shape[1]), min(y2 + 2, image.shape[0])
    if rx2 <= rx1 or ry2 <= ry1:
        return
    region = image[ry1:ry2, rx1:rx2]
    target = region if alpha >= 1.0 else region.copy()
    ox, oy = rx1, ry1
    cv2.rectangle(target, (x1 + radius - ox, y1 - oy), (x2 - radius - ox, y2 - oy), color, -1)
    cv2.rectangle(target, (x1 - ox, y1 + radius - oy), (x2 - ox, y2 - radius - oy), color, -1)
    for cx, cy in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(target, (cx - ox, cy - oy), radius, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, region, 1 - alpha, 0, dst=region)
    if border is not None:
        corners = (
            ((x1 + radius, y1 + radius), 180),
            ((x2 - radius, y1 + radius), 270),
            ((x2 - radius, y2 - radius), 0),
            ((x1 + radius, y2 - radius), 90),
        )
        for (cx, cy), start in corners:
            cv2.ellipse(
                image, (cx, cy), (radius, radius), 0, start, start + 90, border, 1, cv2.LINE_AA
            )
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), border, 1, cv2.LINE_AA)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), border, 1, cv2.LINE_AA)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), border, 1, cv2.LINE_AA)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), border, 1, cv2.LINE_AA)


def chip(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    size: int = 13,
    fill: BGR = SURFACE_RAISED,
    color: BGR = TEXT,
    weight: str = "SemiBold",
    anchor: str = "lt",
    alpha: float = 0.92,
    padding: tuple[int, int] = (8, 4),
) -> tuple[int, int]:
    """A pill with text; returns its width and height."""
    width, height = text_size(text, size, weight)
    pad_x, pad_y = padding
    box_w, box_h = width + 2 * pad_x, line_height(size) + 2 * pad_y - 4
    x, y = origin
    if anchor[0] == "m":
        x -= box_w // 2
    elif anchor[0] == "r":
        x -= box_w
    if anchor[1] == "b":
        y -= box_h
    elif anchor[1] == "m":
        y -= box_h // 2
    x = int(np.clip(x, 0, max(0, image.shape[1] - box_w)))
    y = int(np.clip(y, 0, max(0, image.shape[0] - box_h)))
    rounded_rect(image, (x, y), (x + box_w, y + box_h), fill, radius=box_h // 2, alpha=alpha)
    draw_text(
        image, text, (x + box_w // 2, y + box_h // 2), size, color, weight=weight, anchor="mm"
    )
    return box_w, box_h


def readable_on(fill: BGR) -> BGR:
    luminance = 0.114 * fill[0] + 0.587 * fill[1] + 0.299 * fill[2]
    return hex_bgr("#101318") if luminance > 150 else TEXT


def vertical_gradient(height: int, width: int, top: BGR, bottom: BGR) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    image = np.array(top, np.float32) * (1 - ramp) + np.array(bottom, np.float32) * ramp
    return np.repeat(image, width, axis=1).astype(np.uint8)
