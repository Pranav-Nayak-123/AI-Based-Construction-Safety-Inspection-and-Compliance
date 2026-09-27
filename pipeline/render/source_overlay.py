"""Annotate a source frame: zones, edges, tracked boxes, PPE state and live rule alerts.

Every state is written as text as well as colour (requirements §8: no meaning carried by
colour alone), and the disclaimer is drawn on every frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from shared.enums import PERSON_CLASS, AlertState, HelmetState, VestState
from shared.identity import display_track_label
from shared.schemas.frame import FrameRuleOutput
from shared.schemas.scene import GeometryType, ProposalType, SceneProposal
from shared.schemas.tracks import HelmetRecord, Pass1TrackObservation, VestRecord

DISCLAIMER = "Heuristic triage from video. Not legal advice."
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR
NEUTRAL = (235, 235, 235)
MACHINE = (255, 175, 70)
PENDING = (0, 165, 255)
ALERT = (50, 50, 235)
TEXT_DARK = (20, 20, 20)
ZONE_COLORS = {
    ProposalType.RESTRICTED_ZONE: (50, 50, 235),
    ProposalType.TRAFFIC_ZONE: (0, 210, 255),
    ProposalType.MACHINERY_OPERATING_REGION: (255, 175, 70),
    ProposalType.OPEN_EDGE: (0, 110, 255),
}
DEFAULT_ZONE_COLOR = (200, 200, 200)
ZONE_FILL_ALPHA = 0.16


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
    return {VestState.VEST: "vest", VestState.NO_VEST: "no vest", VestState.UNKNOWN: "vest ?"}[
        record.state
    ]


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Text on a filled chip, kept inside the frame."""
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - width - 6)))
    y = int(np.clip(origin[1], height + 6, image.shape[0] - 2))
    cv2.rectangle(image, (x, y - height - 6), (x + width + 6, y + baseline - 2), color, -1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = TEXT_DARK if luminance > 140 else (255, 255, 255)
    cv2.putText(image, text, (x + 3, y - 3), FONT, scale, text_color, thickness, cv2.LINE_AA)


def draw_scene(
    image: np.ndarray, proposals: Sequence[SceneProposal], *, scale: float = 1.0
) -> None:
    if not proposals:
        return
    fill = image.copy()
    for proposal in proposals:
        color = ZONE_COLORS.get(proposal.proposal_type, DEFAULT_ZONE_COLOR)
        points = (np.asarray(proposal.reference_points) * scale).astype(np.int32)
        if proposal.geometry_type is GeometryType.BOX:
            (x1, y1), (x2, y2) = points
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        if proposal.geometry_type is GeometryType.POLYLINE:
            cv2.polylines(image, [points], False, color, max(2, round(4 * scale)), cv2.LINE_AA)
        else:
            cv2.fillPoly(fill, [points], color)
            cv2.polylines(image, [points], True, color, max(1, round(2 * scale)), cv2.LINE_AA)
    cv2.addWeighted(fill, ZONE_FILL_ALPHA, image, 1 - ZONE_FILL_ALPHA, 0, dst=image)
    for proposal in proposals:
        color = ZONE_COLORS.get(proposal.proposal_type, DEFAULT_ZONE_COLOR)
        x, y = (np.asarray(proposal.reference_points).min(axis=0) * scale).astype(int)
        name = proposal.label or proposal.proposal_id
        draw_label(image, f"{proposal.proposal_type.value}: {name}", (x, y), color, scale=0.45)


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
        return ALERT, " ".join(f"{rule} ALERT" for rule in sorted(confirmed))
    return PENDING, " ".join(f"{rule} pending" for rule in sorted(pending))


def draw_tracks(
    image: np.ndarray,
    observations: Sequence[Pass1TrackObservation],
    output: FrameRuleOutput | None,
    *,
    scale: float = 1.0,
) -> None:
    alerts = _alert_states(output)
    for observation in observations:
        box = observation.box_oriented or observation.box_reference
        x1, y1 = round(box.x1 * scale), round(box.y1 * scale)
        x2, y2 = round(box.x2 * scale), round(box.y2 * scale)
        label = display_track_label(observation.canonical_track_id)
        is_person = observation.object_class == PERSON_CLASS
        color = NEUTRAL if is_person else MACHINE
        parts = [label if is_person else f"{observation.object_class} {label}"]
        if is_person:
            parts += [
                text
                for text in (_helmet_text(observation.helmet), _vest_text(observation.vest))
                if text
            ]
        track_alerts = alerts.get(observation.canonical_track_id)
        thickness = 2
        if track_alerts:
            color, alert_text = _alert_color_and_text(track_alerts)
            parts.append(alert_text)
            thickness = 3
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        draw_label(image, " | ".join(parts), (x1, y1 - 2), color, scale=0.45)


def draw_hud(image: np.ndarray, time_s: float, *, heading: str = "") -> None:
    minutes, seconds = divmod(time_s, 60.0)
    stamp = f"{int(minutes):02d}:{seconds:05.2f}"
    draw_label(image, f"{heading}  {stamp}".strip(), (8, 26), (40, 40, 40), scale=0.6)
    draw_label(image, DISCLAIMER, (8, image.shape[0] - 8), (40, 40, 40), scale=0.5)


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
