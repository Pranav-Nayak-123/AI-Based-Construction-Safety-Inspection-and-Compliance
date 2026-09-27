"""The twin pane: a schematic map of workers, machines, zones and edges (v7 §3, §15.2).

Two modes share one drawing style:

* image-aligned (v7 §10.4 fallback, used until a relative plane is fitted): positions are
  reference-image ground anchors, drawn schematically; the pane says so;
* relative-plane: positions in worker-height units, top-down (added with the plane fit).

Workers leave short trails so movement reads at a glance. Zones are drawn as extruded
slabs to give the 2.5D cue without claiming 3D geometry.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import cv2
import numpy as np

from pipeline.render.source_overlay import ALERT, MACHINE, NEUTRAL, PENDING, ZONE_COLORS, draw_label
from shared.enums import PERSON_CLASS, AlertState
from shared.identity import display_track_label
from shared.schemas.frame import FrameRuleOutput
from shared.schemas.scene import GeometryType, SceneProposal
from shared.schemas.tracks import Pass1TrackObservation

BACKGROUND = (38, 34, 30)
GRID = (58, 54, 50)
EXTRUSION_PX = 10
TRAIL_SECONDS = 2.0
APPROXIMATE_NOTE = "Approximate 2.5D view - positions and zones inferred from video"
IMAGE_ALIGNED_NOTE = "Image-aligned map (relative plane unavailable)"


class PlanView:
    def __init__(self, size: tuple[int, int], canvas_size: tuple[int, int], fps: float) -> None:
        self.width, self.height = size
        self.scale = min(self.width / canvas_size[0], self.height / canvas_size[1])
        self.offset = (
            (self.width - canvas_size[0] * self.scale) / 2,
            (self.height - canvas_size[1] * self.scale) / 2,
        )
        self._trail_length = max(2, round(TRAIL_SECONDS * fps))
        self._trails: dict[str, deque[tuple[int, int]]] = {}
        self._background = self._grid()

    def _grid(self) -> np.ndarray:
        image = np.full((self.height, self.width, 3), BACKGROUND, np.uint8)
        step = 40
        for x in range(0, self.width, step):
            cv2.line(image, (x, 0), (x, self.height), GRID, 1)
        for y in range(0, self.height, step):
            cv2.line(image, (0, y), (self.width, y), GRID, 1)
        return image

    def _to_pane(self, points: np.ndarray) -> np.ndarray:
        return (points * self.scale + np.asarray(self.offset)).astype(np.int32)

    def reset(self) -> None:
        """Shot boundary: trails from the previous view are meaningless."""
        self._trails.clear()

    def render(
        self,
        *,
        observations: Sequence[Pass1TrackObservation],
        output: FrameRuleOutput | None,
        proposals: Sequence[SceneProposal],
    ) -> np.ndarray:
        image = self._background.copy()
        self._draw_zones(image, proposals)
        alerts: dict[str, AlertState] = {}
        if output is not None:
            for alert in output.active_alerts:
                previous = alerts.get(alert.track_id)
                if previous is None or previous is AlertState.PENDING_ALERT:
                    alerts[alert.track_id] = alert.state

        present = {observation.canonical_track_id for observation in observations}
        for track_id in list(self._trails):
            if track_id not in present:
                del self._trails[track_id]

        for observation in sorted(observations, key=lambda o: o.object_class != PERSON_CLASS):
            anchor = self._to_pane(
                np.array([[observation.anchor_reference.x, observation.anchor_reference.y]])
            )[0]
            label = display_track_label(observation.canonical_track_id)
            if observation.object_class == PERSON_CLASS:
                self._draw_worker(image, observation.canonical_track_id, anchor, label, alerts)
            else:
                self._draw_machine(image, observation, label)

        draw_label(image, IMAGE_ALIGNED_NOTE, (8, 24), (70, 70, 70), scale=0.45)
        draw_label(image, APPROXIMATE_NOTE, (8, self.height - 8), (70, 70, 70), scale=0.42)
        return image

    def _draw_zones(self, image: np.ndarray, proposals: Sequence[SceneProposal]) -> None:
        for proposal in proposals:
            color = ZONE_COLORS.get(proposal.proposal_type, (200, 200, 200))
            points = self._to_pane(np.asarray(proposal.reference_points, np.float64))
            if proposal.geometry_type is GeometryType.BOX:
                (x1, y1), (x2, y2) = points
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
            if proposal.geometry_type is GeometryType.POLYLINE:
                cv2.polylines(image, [points], False, color, 4, cv2.LINE_AA)
                for dash in range(len(points) - 1):
                    start, end = points[dash], points[dash + 1]
                    midpoint = ((start + end) // 2).tolist()
                    cv2.circle(image, tuple(midpoint), 4, color, -1, cv2.LINE_AA)
            else:
                lifted = points - np.array([0, EXTRUSION_PX])
                shade = tuple(int(c * 0.45) for c in color)
                for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
                    side = np.array([a, b, b - [0, EXTRUSION_PX], a - [0, EXTRUSION_PX]])
                    cv2.fillPoly(image, [side.astype(np.int32)], shade, cv2.LINE_AA)
                overlay = image.copy()
                cv2.fillPoly(overlay, [lifted], color, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
                cv2.polylines(image, [lifted], True, color, 2, cv2.LINE_AA)
            x, y = points.min(axis=0)
            draw_label(
                image,
                proposal.label or proposal.proposal_id,
                (x, y - EXTRUSION_PX - 4),
                color,
                scale=0.4,
            )

    def _draw_worker(
        self,
        image: np.ndarray,
        track_id: str,
        anchor: np.ndarray,
        label: str,
        alerts: dict[str, AlertState],
    ) -> None:
        trail = self._trails.setdefault(track_id, deque(maxlen=self._trail_length))
        trail.append((int(anchor[0]), int(anchor[1])))
        state = alerts.get(track_id)
        color = NEUTRAL
        suffix = ""
        if state is AlertState.PENDING_ALERT:
            color, suffix = PENDING, " pending"
        elif state is not None:
            color, suffix = ALERT, " ALERT"
        if len(trail) > 1:
            cv2.polylines(image, [np.array(trail, np.int32)], False, color, 1, cv2.LINE_AA)
        center = (int(anchor[0]), int(anchor[1]))
        cv2.circle(image, center, 7, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, 7, (0, 0, 0), 1, cv2.LINE_AA)
        if state is not None and state is not AlertState.PENDING_ALERT:
            cv2.circle(image, center, 13, ALERT, 2, cv2.LINE_AA)
        draw_label(image, f"{label}{suffix}", (center[0] + 9, center[1] - 6), color, scale=0.4)

    def _draw_machine(
        self, image: np.ndarray, observation: Pass1TrackObservation, label: str
    ) -> None:
        box = observation.box_reference
        # Footprint cue: the bottom strip of the box, where the machine meets the ground.
        footprint = np.array(
            [
                [box.x1, box.y2 - 0.15 * box.height()],
                [box.x2, box.y2 - 0.15 * box.height()],
                [box.x2, box.y2],
                [box.x1, box.y2],
            ]
        )
        points = self._to_pane(footprint)
        cv2.fillPoly(image, [points], tuple(int(c * 0.6) for c in MACHINE), cv2.LINE_AA)
        cv2.polylines(image, [points], True, MACHINE, 2, cv2.LINE_AA)
        x, y = points.min(axis=0)
        draw_label(image, f"{observation.object_class} {label}", (x, y - 4), MACHINE, scale=0.4)
