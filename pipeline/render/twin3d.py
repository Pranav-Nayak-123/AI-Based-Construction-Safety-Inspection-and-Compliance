"""The 2.5D situational twin, rendered from the calibrated camera (v7 §3, §15).

Everything is placed on the fitted ground plane in worker-height units (1 grid square =
one worker height) and drawn through a virtual camera that sits behind and above the
CCTV camera, swaying slowly so depth reads without disorienting the viewer. Nothing here
claims hidden geometry: machines are prisms on their observed ground contact, zones and
edges are the operator's lines lifted onto the plane.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from shapely.geometry import Polygon

from pipeline.mapping.camera_model import GroundMapper
from pipeline.mapping.geometry import footprint_from_contact
from pipeline.render import style
from shared.enums import PERSON_CLASS, AlertState, HelmetState, OperatingState, VestState
from shared.identity import display_track_label
from shared.schemas.frame import FrameRuleOutput
from shared.schemas.scene import GeometryType, ProposalType, SceneProposal
from shared.schemas.tracks import Pass1TrackObservation

SWAY_DEGREES = 11.0
SWAY_PERIOD_S = 36.0
ELEVATION_DEGREES = 46.0
FIELD_OF_VIEW_DEGREES = 58.0
WORKER_RADIUS = 0.17
MACHINE_HEIGHT = {"machinery": 2.4, "vehicle": 1.5}
NEAR_BAND, CAUTION_BAND = 1.5, 3.0
TRAIL_SECONDS = 2.5
SMOOTHING = 0.35
ZONE_WALL = 0.35
EDGE_DROP = 0.8
SKIN = style.hex_bgr("#C99A7B")
HI_VIS = style.hex_bgr("#C6F24A")
SHIRT = style.hex_bgr("#6C7A8C")
UNKNOWN_BODY = style.hex_bgr("#4E5663")
GROUND = style.hex_bgr("#151A20")
GRID_MINOR = style.hex_bgr("#1F252D")
GRID_MAJOR = style.hex_bgr("#2B333E")
OBSERVED = style.hex_bgr("#1A2230")


@dataclass
class _View:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    focal: float
    centre: tuple[float, float]

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) → pixels (N, 2) and depths (N,); points behind the view get NaN."""
        rel = points - self.position
        x, y, z = rel @ self.right, -(rel @ self.up), rel @ self.forward
        with np.errstate(divide="ignore", invalid="ignore"):
            z_safe = np.where(z > 0.05, z, np.nan)
            pixels = np.stack(
                [
                    self.centre[0] + self.focal * x / z_safe,
                    self.centre[1] + self.focal * y / z_safe,
                ],
                axis=1,
            )
        return pixels, z


def _poly(points: np.ndarray) -> np.ndarray | None:
    if not np.all(np.isfinite(points)):
        return None
    return np.round(points).astype(np.int32)


def _fill(image: np.ndarray, points: np.ndarray | None, color, alpha: float = 1.0) -> None:
    if points is None or len(points) < 3:
        return
    if alpha >= 1.0:
        cv2.fillPoly(image, [points], color, cv2.LINE_AA)
        return
    x, y, w, h = cv2.boundingRect(points)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w + 1, image.shape[1]), min(y + h + 1, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = image[y0:y1, x0:x1]
    layer = region.copy()
    cv2.fillPoly(layer, [points - [x0, y0]], color, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, region, 1 - alpha, 0, dst=region)


def _shade(color, factor: float):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


class TwinRenderer:
    """Stateful per clip (trails and smoothing); call `render` once per frame in order."""

    def __init__(
        self,
        size: tuple[int, int],
        mapper: GroundMapper,
        proposals: Sequence[SceneProposal],
        *,
        fps: float,
    ) -> None:
        self.width, self.height = size
        self.mapper = mapper
        self.fps = fps
        support = np.asarray(mapper.fit.support_polygon or [(-5, 5), (5, 5), (5, 20), (-5, 20)])
        self.support = support
        lo, hi = support.min(axis=0), support.max(axis=0)
        # Aim a little nearer than the middle: people close to the camera matter most.
        self.centre = np.array([(lo[0] + hi[0]) / 2, lo[1] + 0.42 * (hi[1] - lo[1])])
        self.extent = float(max(hi - lo)) + 4.0
        self.grid_lo = np.floor(lo - 3.0)
        self.grid_hi = np.ceil(hi + 3.0)
        (x0, y0), (x1, y1) = self.grid_lo, self.grid_hi
        segments, major = [], []
        for value in np.arange(x0, x1 + 1):
            segments += [[value, y0, 0.0], [value, y1, 0.0]]
            major.append(int(value) % 5 == 0)
        for value in np.arange(y0, y1 + 1):
            segments += [[x0, value, 0.0], [x1, value, 0.0]]
            major.append(int(value) % 5 == 0)
        self._grid_points = np.array(segments)
        self._grid_major = major
        self.zones = [self._ground_zone(p) for p in proposals]
        self._positions: dict[str, np.ndarray] = {}
        self._trails: dict[str, deque] = {}
        self._background = style.vertical_gradient(
            self.height, self.width, style.hex_bgr("#0B0D11"), style.hex_bgr("#12161C")
        )

    # ------------------------------------------------------------------ setup

    def _ground_zone(self, proposal: SceneProposal):
        points = np.asarray(proposal.reference_points, np.float64)
        if proposal.geometry_type is GeometryType.BOX:
            (x1, y1), (x2, y2) = points
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        ground = self.mapper.ground(points)
        return proposal, ground

    def reset(self) -> None:
        self._positions.clear()
        self._trails.clear()

    def _view(self, time_s: float) -> _View:
        azimuth = math.radians(SWAY_DEGREES) * math.sin(2 * math.pi * time_s / SWAY_PERIOD_S)
        elevation = math.radians(ELEVATION_DEGREES)
        focal = 0.5 * self.width / math.tan(math.radians(FIELD_OF_VIEW_DEGREES) / 2)
        distance = 0.46 * self.extent * focal / (0.5 * self.width) + 2.0
        target = np.array([self.centre[0], self.centre[1], 0.0])
        offset = np.array(
            [
                -math.sin(azimuth) * math.cos(elevation),
                -math.cos(azimuth) * math.cos(elevation),
                math.sin(elevation),
            ]
        )
        position = target + distance * offset
        forward = (target - position) / np.linalg.norm(target - position)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return _View(position, right, up, forward, focal, (self.width / 2, self.height * 0.5))

    # ------------------------------------------------------------------ frame

    def render(
        self,
        *,
        time_s: float,
        observations: Sequence[Pass1TrackObservation],
        output: FrameRuleOutput | None,
        machine_states: Mapping[str, OperatingState],
        machinery_classes: frozenset[str],
    ) -> np.ndarray:
        view = self._view(time_s)
        image = self._background.copy()
        self._draw_ground(image, view)
        self._draw_camera(image, view)
        alerts = self._alerts(output)
        workers, machines = self._objects(observations, machinery_classes)

        # Flat things first: zones' floors, edges, bands, trails, shadows.
        for proposal, ground in self.zones:
            self._draw_zone_floor(image, view, proposal, ground)
        for machine_id, _cls, quad in machines:
            state = machine_states.get(machine_id, OperatingState.UNKNOWN)
            if state is not OperatingState.STATIONARY:
                self._draw_bands(image, view, quad)
        for track_id, _obs, _position in workers:
            self._draw_trail(image, view, track_id)

        # Then solid things, far to near.
        drawables = []
        for proposal, ground in self.zones:
            drawables += self._zone_walls(view, proposal, ground)
        for machine_id, cls, quad in machines:
            depth = view.project(np.array([[*quad.mean(axis=0), 0.0]]))[1][0]
            drawables.append((depth, "machine", (machine_id, cls, quad)))
        for track_id, obs, position in workers:
            depth = view.project(np.array([[*position, 0.0]]))[1][0]
            drawables.append((depth, "worker", (track_id, obs, position)))
        for _depth, kind, payload in sorted(
            drawables, key=lambda d: -d[0] if np.isfinite(d[0]) else 0
        ):
            if kind == "wall":
                points, color, alpha = payload
                _fill(image, points, color, alpha)
            elif kind == "machine":
                machine_id, cls, quad = payload
                self._draw_machine(
                    image,
                    view,
                    machine_id,
                    cls,
                    quad,
                    machine_states.get(machine_id, OperatingState.UNKNOWN),
                )
            else:
                track_id, obs, position = payload
                self._draw_worker(
                    image, view, track_id, obs, position, alerts.get(track_id), time_s
                )
        self._draw_hud(image)
        return image

    def _alerts(self, output: FrameRuleOutput | None) -> dict[str, tuple[AlertState, list[str]]]:
        alerts: dict[str, tuple[AlertState, list[str]]] = {}
        if output is None:
            return alerts
        for alert in output.active_alerts:
            state, rules = alerts.get(alert.track_id, (AlertState.PENDING_ALERT, []))
            if alert.state is not AlertState.PENDING_ALERT:
                state = AlertState.ALERT
            alerts[alert.track_id] = (state, [*rules, alert.rule_id.value])
        return alerts

    def _objects(self, observations, machinery_classes):
        workers, machines = [], []
        present = set()
        for o in observations:
            if o.object_class == PERSON_CLASS:
                raw = self.mapper.ground(np.array([[o.anchor_reference.x, o.anchor_reference.y]]))[
                    0
                ]
                if not np.all(np.isfinite(raw)):
                    continue
                previous = self._positions.get(o.canonical_track_id)
                position = raw if previous is None else previous + SMOOTHING * (raw - previous)
                self._positions[o.canonical_track_id] = position
                trail = self._trails.setdefault(
                    o.canonical_track_id, deque(maxlen=round(TRAIL_SECONDS * self.fps))
                )
                trail.append(position.copy())
                present.add(o.canonical_track_id)
                workers.append((o.canonical_track_id, o, position))
            elif o.object_class in machinery_classes:
                box = o.box_reference
                inset = 0.1 * box.width()
                contact = np.array([[box.x1 + inset, box.y2], [box.x2 - inset, box.y2]])
                quad = footprint_from_contact(self.mapper.ground(contact)[None])[0]
                if np.all(np.isfinite(quad)):
                    machines.append((o.canonical_track_id, o.object_class, quad))
        for track_id in list(self._trails):
            if track_id not in present:
                self._trails.pop(track_id)
                self._positions.pop(track_id, None)
        return workers, machines

    # ------------------------------------------------------------------ drawing

    def _draw_ground(self, image: np.ndarray, view: _View) -> None:
        (x0, y0), (x1, y1) = self.grid_lo, self.grid_hi
        plane = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], np.float64)
        _fill(image, _poly(view.project(plane)[0]), GROUND)
        observed = view.project(np.column_stack([self.support, np.zeros(len(self.support))]))[0]
        _fill(image, _poly(observed), OBSERVED)
        pixels = view.project(self._grid_points)[0].reshape(-1, 2, 2)
        for (start, end), major in zip(pixels, self._grid_major, strict=True):
            if np.all(np.isfinite(start)) and np.all(np.isfinite(end)):
                cv2.line(
                    image,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    GRID_MAJOR if major else GRID_MINOR,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_camera(self, image: np.ndarray, view: _View) -> None:
        camera = self.mapper.camera
        eye = np.array([0.0, 0.0, camera.p.height_wh])
        width, height = 2 * camera.p.principal[0], 2 * camera.p.principal[1]
        far_row = min(height - 1, camera.horizon_v(width / 2) + 0.22 * height)
        corners = np.array(
            [[0, height - 1], [width - 1, height - 1], [width - 1, far_row], [0, far_row]]
        )
        ground = camera.ground(corners)
        if not np.all(np.isfinite(ground)):
            return
        points = np.vstack([np.column_stack([ground, np.zeros(4)]), eye])
        pixels = view.project(points)[0]
        if not np.all(np.isfinite(pixels)):
            return
        footprint = _poly(pixels[:4])
        _fill(image, footprint, style.ACCENT, 0.06)
        for corner in pixels[:4]:
            cv2.line(
                image,
                tuple(np.round(pixels[4]).astype(int)),
                tuple(np.round(corner).astype(int)),
                _shade(style.ACCENT, 0.55),
                1,
                cv2.LINE_AA,
            )
        cv2.circle(image, tuple(np.round(pixels[4]).astype(int)), 5, style.ACCENT, -1, cv2.LINE_AA)
        style.chip(
            image,
            "CCTV",
            (int(pixels[4][0]), int(pixels[4][1]) - 10),
            size=11,
            fill=style.ACCENT,
            color=style.readable_on(style.ACCENT),
            anchor="mb",
        )

    def _draw_zone_floor(self, image, view, proposal: SceneProposal, ground: np.ndarray) -> None:
        if not np.all(np.isfinite(ground)):
            return
        color = self._zone_color(proposal)
        pixels = view.project(np.column_stack([ground, np.zeros(len(ground))]))[0]
        if proposal.proposal_type is ProposalType.OPEN_EDGE:
            points = _poly(pixels)
            if points is not None:
                cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
            return
        _fill(image, _poly(pixels), color, 0.22)

    def _zone_walls(self, view, proposal: SceneProposal, ground: np.ndarray):
        if not np.all(np.isfinite(ground)):
            return []
        color = self._zone_color(proposal)
        walls = []
        if proposal.proposal_type is ProposalType.OPEN_EDGE:
            # An edge is drawn as a drop: a face falling away below the ground line.
            for a, b in zip(ground[:-1], ground[1:], strict=True):
                face = np.array([[*a, 0], [*b, 0], [*b, -EDGE_DROP], [*a, -EDGE_DROP]])
                pixels, depth = view.project(face)
                walls.append(
                    (float(np.nanmean(depth)), "wall", (_poly(pixels), _shade(color, 0.55), 0.55))
                )
            return walls
        closed = np.vstack([ground, ground[:1]])
        for a, b in zip(closed[:-1], closed[1:], strict=True):
            face = np.array([[*a, 0], [*b, 0], [*b, ZONE_WALL], [*a, ZONE_WALL]])
            pixels, depth = view.project(face)
            walls.append(
                (float(np.nanmean(depth)), "wall", (_poly(pixels), _shade(color, 0.8), 0.35))
            )
        return walls

    def _draw_bands(self, image, view, quad: np.ndarray) -> None:
        footprint = Polygon(quad)
        for band, color in ((CAUTION_BAND, style.WARN), (NEAR_BAND, style.DANGER)):
            ring = np.asarray(footprint.buffer(band, quad_segs=6).exterior.coords)
            pixels = view.project(np.column_stack([ring, np.zeros(len(ring))]))[0]
            points = _poly(pixels)
            if points is None:
                continue
            _fill(image, points, color, 0.06)
            for index in range(0, len(points) - 1, 2):  # dashed outline
                cv2.line(
                    image,
                    tuple(points[index]),
                    tuple(points[index + 1]),
                    _shade(color, 0.9),
                    1,
                    cv2.LINE_AA,
                )

    def _draw_machine(self, image, view, machine_id, cls, quad, state: OperatingState) -> None:
        height = MACHINE_HEIGHT.get(cls, 2.0)
        base = {
            OperatingState.ACTIVE: style.MACHINE,
            OperatingState.STATIONARY: _shade(style.MACHINE, 0.55),
        }.get(state, _shade(style.MACHINE, 0.75))
        bottom = np.column_stack([quad, np.zeros(4)])
        top = np.column_stack([quad, np.full(4, height)])
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            face = np.array([bottom[i], bottom[j], top[j], top[i]])
            pixels, depth = view.project(face)
            faces.append((float(np.nanmean(depth)), pixels, 0.55 + 0.1 * i))
        for _depth, pixels, factor in sorted(faces, key=lambda f: -f[0]):
            _fill(image, _poly(pixels), _shade(base, factor), 0.95)
        top_pixels = _poly(view.project(top)[0])
        _fill(image, top_pixels, _shade(base, 1.0), 0.95)
        if top_pixels is not None:
            cv2.polylines(image, [top_pixels], True, _shade(base, 1.25), 1, cv2.LINE_AA)
            kind = "Machine" if cls == "machinery" else "Vehicle"
            label = f"{kind} {display_track_label(machine_id)}"
            status = {OperatingState.ACTIVE: "active", OperatingState.STATIONARY: "stationary"}.get(
                state, "state unknown"
            )
            x, y = top_pixels.mean(axis=0).astype(int)
            style.chip(
                image,
                f"{label} · {status}",
                (int(x), int(top_pixels[:, 1].min()) - 6),
                size=11,
                fill=style.SURFACE_RAISED,
                color=base,
                anchor="mb",
            )

    def _draw_trail(self, image, view, track_id: str) -> None:
        trail = self._trails.get(track_id)
        if not trail or len(trail) < 2:
            return
        points = np.array(trail)
        pixels = view.project(np.column_stack([points, np.zeros(len(points))]))[0]
        if not np.all(np.isfinite(pixels)):
            return
        count = len(pixels)
        for index in range(1, count):
            fade = index / count
            cv2.line(
                image,
                tuple(np.round(pixels[index - 1]).astype(int)),
                tuple(np.round(pixels[index]).astype(int)),
                _shade(style.ACCENT, 0.35 + 0.5 * fade),
                1,
                cv2.LINE_AA,
            )

    def _draw_worker(self, image, view, track_id, obs, position, alert, time_s) -> None:
        x, y = position
        circle = np.array(
            [
                [x + WORKER_RADIUS * 1.4 * math.cos(a), y + WORKER_RADIUS * 1.4 * math.sin(a), 0]
                for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)
            ]
        )
        _fill(image, _poly(view.project(circle)[0]), (6, 7, 9), 0.55)  # contact shadow
        if alert is not None:
            state, rules = alert
            color = style.DANGER if state is AlertState.ALERT else style.WARN
            pulse = 0.55 + 0.12 * math.sin(time_s * 6.0)
            ring = np.array(
                [
                    [x + pulse * math.cos(a), y + pulse * math.sin(a), 0]
                    for a in np.linspace(0, 2 * math.pi, 28)
                ]
            )
            points = _poly(view.project(ring)[0])
            if points is not None:
                cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        foot, hip, shoulder, head = view.project(
            np.array([[x, y, 0.0], [x, y, 0.45], [x, y, 0.78], [x, y, 0.9]])
        )[0]
        if not np.all(np.isfinite([foot, head])):
            return
        scale = abs(
            view.project(np.array([[x, y, 0.0], [x + WORKER_RADIUS, y, 0.0]]))[0][1][0] - foot[0]
        )
        radius = max(2, int(round(scale)))
        vest = obs.vest.state if obs.vest else VestState.UNKNOWN
        helmet = obs.helmet.state if obs.helmet else HelmetState.UNKNOWN
        body = {VestState.VEST: HI_VIS, VestState.NO_VEST: SHIRT}.get(vest, UNKNOWN_BODY)
        cv2.line(
            image,
            tuple(np.round(foot).astype(int)),
            tuple(np.round(hip).astype(int)),
            _shade(SHIRT, 0.6),
            max(2, radius),
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            tuple(np.round(hip).astype(int)),
            tuple(np.round(shoulder).astype(int)),
            body,
            max(2, int(radius * 1.5)),
            cv2.LINE_AA,
        )
        head_radius = max(2, int(round(radius * 0.8)))
        centre = tuple(np.round(head).astype(int))
        cv2.circle(image, centre, head_radius, SKIN, -1, cv2.LINE_AA)
        if helmet is HelmetState.HELMET:
            cv2.ellipse(
                image,
                centre,
                (head_radius + 1, head_radius + 1),
                0,
                180,
                360,
                style.HELMET_YELLOW,
                -1,
                cv2.LINE_AA,
            )
        elif helmet is HelmetState.NO_HELMET:
            cv2.circle(image, centre, head_radius + 2, style.DANGER, 1, cv2.LINE_AA)
        label = display_track_label(track_id)
        if alert is not None:
            state, rules = alert
            phase = "ALERT" if state is AlertState.ALERT else "pending"
            text = f"{label} · {' '.join(sorted(set(rules)))} {phase}"
            fill = style.DANGER if state is AlertState.ALERT else style.WARN
            style.chip(
                image,
                text,
                (centre[0], centre[1] - head_radius - 4),
                size=11,
                fill=fill,
                color=style.readable_on(fill),
                anchor="mb",
            )
        else:
            style.chip(
                image,
                label,
                (centre[0], centre[1] - head_radius - 4),
                size=10,
                fill=style.SURFACE_RAISED,
                color=style.TEXT_MUTED,
                anchor="mb",
                padding=(5, 2),
                alpha=0.8,
            )

    def _draw_hud(self, image: np.ndarray) -> None:
        style.chip(
            image,
            "2.5D TWIN  ·  calibrated ground plane",
            (12, 12),
            size=12,
            fill=style.SURFACE,
            color=style.TEXT,
            alpha=0.85,
        )
        style.draw_text(
            image,
            "1 grid square = 1 worker height  ·  approximate, inferred from video",
            (12, self.height - 12),
            11,
            style.TEXT_FAINT,
            anchor="lb",
        )
        legend = [
            ("Helmet", style.HELMET_YELLOW),
            ("Hi-vis", HI_VIS),
            ("Machine", style.MACHINE),
            ("Near band", style.DANGER),
        ]
        x = self.width - 12
        for text, color in reversed(legend):
            width, _ = style.text_size(text, 11)
            style.draw_text(image, text, (x, self.height - 12), 11, style.TEXT_MUTED, anchor="rb")
            cv2.circle(image, (x - width - 9, self.height - 18), 4, color, -1, cv2.LINE_AA)
            x -= width + 26

    @staticmethod
    def _zone_color(proposal: SceneProposal):
        return {
            ProposalType.RESTRICTED_ZONE: style.DANGER,
            ProposalType.TRAFFIC_ZONE: style.WARN,
            ProposalType.MACHINERY_OPERATING_REGION: style.MACHINE,
            ProposalType.OPEN_EDGE: style.EDGE,
        }.get(proposal.proposal_type, style.TEXT_MUTED)
