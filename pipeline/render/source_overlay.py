"""Annotate a source frame: zones, edges, tracked people and machines, live rule alerts.

Every state is written as text as well as colour (requirements §8: no meaning carried by
colour alone). Workers get corner brackets rather than full boxes so the footage stays
readable; an alerting worker gets a solid frame and an alert chip.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from pipeline.render import style
from shared.enums import PERSON_CLASS, AlertState, HelmetState, VestState
from shared.identity import display_track_label
from shared.schemas.frame import FrameRuleOutput
from shared.schemas.scene import GeometryType, ProposalType, SceneProposal
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation, VestRecord

DISCLAIMER = "Heuristic triage from video. Not legal advice."
FONT = cv2.FONT_HERSHEY_SIMPLEX  # kept for callers that still draw simple labels

NEUTRAL = style.WORKER
MACHINE = style.MACHINE
PENDING = style.WARN
ALERT = style.DANGER
TEXT_DARK = style.hex_bgr("#101318")
ZONE_COLORS = {
    ProposalType.RESTRICTED_ZONE: style.DANGER,
    ProposalType.TRAFFIC_ZONE: style.WARN,
    ProposalType.MACHINERY_OPERATING_REGION: style.MACHINE,
    ProposalType.OPEN_EDGE: style.EDGE,
}
DEFAULT_ZONE_COLOR = style.TEXT_MUTED
ZONE_FILL_ALPHA = 0.14


def _helmet_text(record: HelmetRecord | None) -> str | None:
    if record is None:
        return None
    return {
        HelmetState.HELMET: "helmet",
        HelmetState.NO_HELMET: "no helmet",
        HelmetState.UNKNOWN: "helmet ?",
    }[record.state]


def _vest_text(record: VestRecord | None) -> str | None:
    if record is None:
        return None
    return {
        VestState.VEST: "hi-vis",
        VestState.NO_VEST: "no hi-vis",
        VestState.UNKNOWN: "hi-vis ?",
    }[record.state]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """A chip whose bottom-left sits at `origin` (kept for the plan view's call sites)."""
    del thickness
    style.chip(
        image,
        text,
        (origin[0], origin[1]),
        size=max(10, round(scale * 26)),
        fill=color,
        color=style.readable_on(color),
        anchor="lb",
        alpha=0.9,
        padding=(6, 3),
    )


def _dashed_polyline(image, points: np.ndarray, color, closed: bool, dash: int = 12) -> None:
    sequence = np.vstack([points, points[:1]]) if closed else points
    for start, end in zip(sequence[:-1], sequence[1:], strict=True):
        length = float(np.hypot(*(end - start)))
        steps = max(1, int(length // dash))
        for index in range(0, steps, 2):
            a = start + (end - start) * index / steps
            b = start + (end - start) * min(index + 1, steps) / steps
            cv2.line(
                image,
                tuple(np.round(a).astype(int)),
                tuple(np.round(b).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )


def draw_scene(
    image: np.ndarray, proposals: Sequence[SceneProposal], *, scale: float = 1.0
) -> None:
    if not proposals:
        return
    fill = image.copy()
    shapes = []
    for proposal in proposals:
        color = ZONE_COLORS.get(proposal.proposal_type, DEFAULT_ZONE_COLOR)
        points = (np.asarray(proposal.reference_points) * scale).astype(np.int32)
        if proposal.geometry_type is GeometryType.BOX:
            (x1, y1), (x2, y2) = points
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        if proposal.geometry_type is not GeometryType.POLYLINE:
            cv2.fillPoly(fill, [points], color, cv2.LINE_AA)
        shapes.append((proposal, points, color))
    cv2.addWeighted(fill, ZONE_FILL_ALPHA, image, 1 - ZONE_FILL_ALPHA, 0, dst=image)
    for proposal, points, color in shapes:
        closed = proposal.geometry_type is not GeometryType.POLYLINE
        if closed:
            _dashed_polyline(image, points.astype(float), color, True)
        else:
            cv2.polylines(image, [points], False, color, 3, cv2.LINE_AA)
        x, y = points.min(axis=0)
        kind = {
            ProposalType.RESTRICTED_ZONE: "Restricted",
            ProposalType.OPEN_EDGE: "Open edge",
            ProposalType.TRAFFIC_ZONE: "Traffic",
        }.get(proposal.proposal_type, "Zone")
        style.chip(
            image,
            f"{kind} · {proposal.label or proposal.proposal_id}",
            (int(x), int(y) - 4),
            size=11,
            fill=color,
            color=style.readable_on(color),
            anchor="lb",
            alpha=0.9,
            padding=(6, 3),
        )


def _alert_states(output: FrameRuleOutput | None) -> dict[str, list[tuple[str, AlertState]]]:
    states: dict[str, list[tuple[str, AlertState]]] = {}
    if output is None:
        return states
    for alert in output.active_alerts:
        states.setdefault(alert.track_id, []).append((alert.rule_id.value, alert.state))
    return states


def _alert_color_and_text(alerts: list[tuple[str, AlertState]]) -> tuple[tuple[int, int, int], str]:
    confirmed = [rule for rule, state in alerts if state is not AlertState.PENDING_ALERT]
    pending = [rule for rule, state in alerts if state is AlertState.PENDING_ALERT]
    if confirmed:
        return ALERT, " ".join(f"{rule} ALERT" for rule in sorted(set(confirmed)))
    return PENDING, " ".join(f"{rule} pending" for rule in sorted(set(pending)))


def _brackets(image, x1: int, y1: int, x2: int, y2: int, color, thickness: int = 2) -> None:
    arm = max(6, min(x2 - x1, y2 - y1) // 4)
    for (x, y), (dx, dy) in (
        ((x1, y1), (1, 1)),
        ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)),
        ((x2, y2), (-1, -1)),
    ):
        cv2.line(image, (x, y), (x + dx * arm, y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y), (x, y + dy * arm), color, thickness, cv2.LINE_AA)


def draw_tracks(
    image: np.ndarray,
    observations: Sequence[Pass1TrackObservation],
    output: FrameRuleOutput | None,
    *,
    scale: float = 1.0,
) -> None:
    alerts = _alert_states(output)
    placer = style.LabelPlacer((image.shape[1], image.shape[0]))

    def priority(observation: Pass1TrackObservation) -> tuple[int, float]:
        """Alerting workers claim label space first, then machines, then everyone else."""
        box = observation.box_oriented or observation.box_reference
        if observation.canonical_track_id in alerts:
            return (0, -box.y2)
        return (1 if observation.object_class != PERSON_CLASS else 2, -box.y2)

    for observation in sorted(observations, key=priority):
        box = observation.box_oriented or observation.box_reference
        x1, y1 = round(box.x1 * scale), round(box.y1 * scale)
        x2, y2 = round(box.x2 * scale), round(box.y2 * scale)
        label = display_track_label(observation.canonical_track_id)
        if observation.object_class != PERSON_CLASS:
            cv2.rectangle(image, (x1, y1), (x2, y2), MACHINE, 2, cv2.LINE_AA)
            kind = "Machine" if observation.object_class == "machinery" else "Vehicle"
            _label(
                image,
                placer,
                f"{kind} {label}",
                x1,
                y1,
                MACHINE,
                style.readable_on(MACHINE),
                size=11,
                force=True,
            )
            continue
        track_alerts = alerts.get(observation.canonical_track_id)
        ppe = [t for t in (_helmet_text(observation.helmet), _vest_text(observation.vest)) if t]
        if track_alerts:
            color, alert_text = _alert_color_and_text(track_alerts)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _label(
                image,
                placer,
                f"{label}  {alert_text}",
                x1,
                y1,
                color,
                style.readable_on(color),
                size=12,
                force=True,
            )
            continue
        _brackets(image, x1, y1, x2, y2, NEAR_WHITE)
        # Full PPE words when there is room; the ID alone when the scene is crowded.
        full = "  ".join([label, *ppe]) if ppe else label
        if not _label(image, placer, full, x1, y1, style.SURFACE, style.TEXT, size=11):
            _label(image, placer, label, x1, y1, style.SURFACE, style.TEXT_MUTED, size=10)


def _label(
    image: np.ndarray,
    placer: style.LabelPlacer,
    text: str,
    x: int,
    y: int,
    fill,
    color,
    *,
    size: int,
    force: bool = False,
) -> bool:
    padding = (6, 2)
    width, height = style.chip_size(text, size=size, padding=padding)
    spot = placer.place(x, y - 2, width, height, force=force)
    if spot is None:
        return False
    style.chip(image, text, spot, size=size, fill=fill, color=color, alpha=0.88, padding=padding)
    return True


NEAR_WHITE = style.hex_bgr("#E8EEF4")


def draw_hud(image: np.ndarray, time_s: float, *, heading: str = "") -> None:
    minutes, seconds = divmod(time_s, 60.0)
    stamp = f"{int(minutes):02d}:{seconds:05.2f}"
    style.chip(
        image,
        f"{heading}  {stamp}".strip(),
        (8, 8),
        size=13,
        fill=style.SURFACE,
        color=style.TEXT,
        alpha=0.85,
    )
    style.chip(
        image,
        DISCLAIMER,
        (8, image.shape[0] - 8),
        size=12,
        fill=style.SURFACE,
        color=style.TEXT_MUTED,
        anchor="lb",
        alpha=0.85,
    )


def render_source_frame(
    canvas: np.ndarray,
    *,
    observations: Sequence[Pass1TrackObservation],
    output: FrameRuleOutput | None,
    proposals: Sequence[SceneProposal],
    time_s: float,
    size: tuple[int, int] | None = None,
    hud: bool = True,
) -> np.ndarray:
    """Annotated copy of `canvas`, optionally resized to `size` (width, height) first.

    `hud=False` omits the clock and disclaimer, for layouts that draw them elsewhere.
    """
    image = canvas.copy()
    scale = 1.0
    if size is not None and (image.shape[1], image.shape[0]) != size:
        scale = size[0] / image.shape[1]
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    draw_scene(image, proposals, scale=scale)
    draw_tracks(image, observations, output, scale=scale)
    if hud:
        draw_hud(image, time_s)
    return image
