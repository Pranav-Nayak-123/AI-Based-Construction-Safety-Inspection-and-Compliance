"""Draw restricted zones, traffic zones and open edges for one fixed camera.

    uv run python tools/annotate_site.py data/source/yard.mp4 --camera yard-east --time 12

Writes `config/sites/<camera>.json` plus the reference frame `<camera>.png` it was drawn
on. Then process clips from that camera with `safety-twin process clip.mp4 --site
config/sites/<camera>.json`.

Keys (in the image window):
    z  restricted zone      t  traffic zone      m  machinery operating region
    e  open edge (polyline) b  barrier
    left click  add a point            u  undo last point
    enter / n   finish the shape       d  delete the last finished shape
    s  save and quit                   q  quit without saving
Labels and edge semantics are asked for in the terminal when a shape is finished.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.intake.frames import FrameSource
from pipeline.render.source_overlay import draw_scene
from pipeline.scene.site_config import SiteConfig, SiteZone
from shared.config import load_config
from shared.schemas.scene import (
    GeometryType,
    ProposalSource,
    ProposalType,
    ProtectionState,
    ProtectionVisibility,
    SceneProposal,
)

SHAPE_KEYS = {
    ord("z"): ProposalType.RESTRICTED_ZONE,
    ord("t"): ProposalType.TRAFFIC_ZONE,
    ord("m"): ProposalType.MACHINERY_OPERATING_REGION,
    ord("e"): ProposalType.OPEN_EDGE,
    ord("b"): ProposalType.BARRIER,
}
DISPLAY_WIDTH = 1280


@dataclass
class Shape:
    proposal_type: ProposalType
    points: list[tuple[float, float]] = field(default_factory=list)
    label: str = ""
    road_work: bool | None = None
    visible_drop: bool | None = None
    protection_visibility: ProtectionVisibility | None = None
    protection_state: ProtectionState | None = None

    @property
    def minimum_points(self) -> int:
        return 2 if self.proposal_type is ProposalType.OPEN_EDGE else 3


def slug(text: str, fallback: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or fallback


def build_site_config(
    camera_id: str, shapes: list[Shape], reference_image: str, canvas: tuple[int, int]
) -> SiteConfig:
    zones: list[SiteZone] = []
    used: set[str] = set()
    for index, shape in enumerate(shapes, start=1):
        base = slug(shape.label, f"{shape.proposal_type.value.replace('_', '-')}-{index}")
        zone_id, suffix = base, 2
        while zone_id in used:
            zone_id, suffix = f"{base}-{suffix}", suffix + 1
        used.add(zone_id)
        zones.append(
            SiteZone(
                zone_id=zone_id,
                proposal_type=shape.proposal_type,
                label=shape.label,
                points=tuple((round(x, 1), round(y, 1)) for x, y in shape.points),
                road_work=shape.road_work,
                visible_drop_or_open_side=shape.visible_drop,
                protection_visibility=shape.protection_visibility,
                protection_state=shape.protection_state,
            )
        )
    return SiteConfig(
        camera_id=camera_id, canvas_size=canvas, reference_image=reference_image, zones=tuple(zones)
    )


def save_site(config: SiteConfig, reference: np.ndarray, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(directory / config.reference_image), reference)
    path = directory / f"{config.camera_id}.json"
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _ask(prompt: str, choices: dict[str, object]) -> object:
    options = "/".join(choices)
    while True:
        answer = input(f"  {prompt} [{options}]: ").strip().lower()
        if answer in choices:
            return choices[answer]


def _describe(shape: Shape) -> None:
    shape.label = input(f"  label for this {shape.proposal_type.value}: ").strip()
    if shape.proposal_type is ProposalType.TRAFFIC_ZONE:
        shape.road_work = _ask("is this road work (y/n/?)", {"y": True, "n": False, "?": None})
    if shape.proposal_type is ProposalType.OPEN_EDGE:
        shape.visible_drop = _ask(
            "visible drop or open side (y/n/?)", {"y": True, "n": False, "?": None}
        )
        shape.protection_visibility = _ask(
            "can protection be judged (s=sufficient, i=insufficient, ?)",
            {
                "s": ProtectionVisibility.SUFFICIENT,
                "i": ProtectionVisibility.INSUFFICIENT,
                "?": ProtectionVisibility.UNKNOWN,
            },
        )
        shape.protection_state = _ask(
            "edge protection (g=guarded, u=unguarded/inadequate, ?)",
            {
                "g": ProtectionState.GUARDED,
                "u": ProtectionState.UNGUARDED_OR_INADEQUATE,
                "?": ProtectionState.UNKNOWN,
            },
        )


def _preview(reference: np.ndarray, shapes: list[Shape], current: Shape | None) -> np.ndarray:
    image = reference.copy()
    proposals = [
        SceneProposal(
            proposal_id=f"shape-{index}",
            shot_id="s0",
            proposal_type=shape.proposal_type,
            geometry_type=(
                GeometryType.POLYLINE
                if shape.proposal_type is ProposalType.OPEN_EDGE
                else GeometryType.POLYGON
            ),
            label=shape.label,
            reference_points=tuple(shape.points),
            source=ProposalSource.MANUAL_SITE_CONFIG,
        )
        for index, shape in enumerate(shapes)
    ]
    draw_scene(image, proposals)
    if current is not None and current.points:
        points = np.array(current.points, np.int32)
        cv2.polylines(image, [points], False, (255, 255, 255), 2, cv2.LINE_AA)
        for x, y in current.points:
            cv2.circle(image, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    mode = current.proposal_type.value if current else "press z/t/m/e/b to start a shape"
    cv2.putText(image, mode, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def reference_frame(video: Path, time_s: float) -> np.ndarray:
    intake = load_config().intake
    source = FrameSource(
        video, target_fps=intake.target_fps, canvas_size=(intake.target_width, intake.target_height)
    )
    chosen = None
    for decoded in source:
        chosen = decoded.image
        if decoded.time_s >= time_s:
            break
    if chosen is None:
        raise SystemExit(f"could not read a frame from {video}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--camera", required=True, help="camera id, e.g. yard-east")
    parser.add_argument("--time", type=float, default=0.0, help="reference frame time (s)")
    parser.add_argument("--out", type=Path, default=Path("config/sites"))
    args = parser.parse_args()

    reference = reference_frame(args.video, args.time)
    scale = DISPLAY_WIDTH / reference.shape[1]
    shapes: list[Shape] = []
    current: Shape | None = None

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and current is not None:
            current.points.append((x / scale, y / scale))

    window = "annotate site - s to save, q to quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = _preview(reference, shapes, current)
        cv2.imshow(window, cv2.resize(preview, None, fx=scale, fy=scale))
        key = cv2.waitKey(30) & 0xFF
        if key in SHAPE_KEYS and current is None:
            current = Shape(SHAPE_KEYS[key])
        elif key == ord("u") and current is not None and current.points:
            current.points.pop()
        elif key in (13, ord("n")) and current is not None:
            if len(current.points) >= current.minimum_points:
                _describe(current)
                shapes.append(current)
            current = None
        elif key == ord("d") and shapes and current is None:
            shapes.pop()
        elif key == ord("s"):
            config = build_site_config(
                args.camera, shapes, f"{args.camera}.png", (reference.shape[1], reference.shape[0])
            )
            print(f"saved {save_site(config, reference, args.out)}")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
