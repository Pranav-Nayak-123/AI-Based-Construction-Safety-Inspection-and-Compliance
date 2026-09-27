"""Compose the 1920x1080 side-by-side frame (v7 §15.4).

    +------------------------------------------------------------------------+
    | brand · title · clip                 [workers] [alerts] [incidents] 00:12 |
    | [ annotated source 940x529 ]        [ 2.5D twin 940x529 ]               |
    | incident cards: severity, title, live elapsed time, action, crop       |
    | timeline: one lane per rule, incident spans by severity, playhead      |
    | coverage chips R1-R5 · disclaimer                                      |
    +------------------------------------------------------------------------+

Cards appear from an incident's *confirmation* time and say how long it has lasted *so
far*: the video never knows more than the rule engine did at that moment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from pipeline.render import style
from shared.config import RenderConfig
from shared.enums import RuleId, RuleStatus
from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord
from shared.schemas.run import RuleCoverageEntry

MARGIN = 16
GAP = 8
HEADER_H = 76
RAIL_H = 244
TIMELINE_H = 142
MAX_CARDS = 3
CARD_LINGER_S = 2.0
DISCLAIMER = "Heuristic triage from video. Not legal advice."
STATUS_TEXT = {
    RuleStatus.EVALUATED_ALERT: "alert",
    RuleStatus.EVALUATED_CLEAR: "clear",
    RuleStatus.INCONCLUSIVE: "inconclusive",
    RuleStatus.NOT_APPLICABLE: "not applicable",
    RuleStatus.UNSUPPORTED: "unsupported",
}
STATUS_COLOR = {
    RuleStatus.EVALUATED_ALERT: style.DANGER,
    RuleStatus.EVALUATED_CLEAR: style.OK,
    RuleStatus.INCONCLUSIVE: style.WARN,
    RuleStatus.NOT_APPLICABLE: style.TEXT_FAINT,
    RuleStatus.UNSUPPORTED: style.TEXT_FAINT,
}
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def card_headline(incident: IncidentRecord, time_s: float) -> tuple[str, str]:
    """Card title and progress line, using only what is known at `time_s`.

    The bundle's observation_text is written after the fact and states the final duration;
    a card shown mid-incident must say how long it has lasted *so far*.
    """
    ongoing = incident.resolved_at_s is None or time_s < incident.resolved_at_s
    end = time_s if ongoing or incident.resolved_at_s is None else incident.resolved_at_s
    elapsed = max(0.0, end - incident.first_seen_s)
    worker = display_track_label(incident.canonical_track_id or "")
    progress = f"{elapsed:.1f} s so far" if ongoing else f"resolved after {elapsed:.1f} s"
    return incident.title or incident.reason_code, f"Worker {worker} · {progress}"


def _clock(time_s: float) -> str:
    minutes, seconds = divmod(time_s, 60.0)
    return f"{int(minutes):02d}:{seconds:04.1f}"


class Compositor:
    def __init__(
        self,
        config: RenderConfig,
        *,
        title: str,
        duration_s: float,
        incidents: Sequence[IncidentRecord],
        coverage: Sequence[RuleCoverageEntry],
    ) -> None:
        self.width, self.height = config.output_width, config.output_height
        self.pane = (config.pane_width, config.pane_height)
        self.title = title
        self.duration_s = max(duration_s, 1e-6)
        self.incidents = sorted(incidents, key=lambda i: i.confirmed_at_s)
        self.left = (MARGIN, HEADER_H + 8)
        self.right = (self.width - MARGIN - self.pane[0], HEADER_H + 8)
        self.rail_top = self.left[1] + self.pane[1] + 12
        self.timeline_top = self.rail_top + RAIL_H + 10
        self.footer_top = self.timeline_top + TIMELINE_H + 6
        self.lane_x0 = MARGIN + 64
        self.lane_x1 = self.width - MARGIN - 12
        self._base = self._static_layer(coverage)

    # ------------------------------------------------------------------ static

    def _static_layer(self, coverage: Sequence[RuleCoverageEntry]) -> np.ndarray:
        image = np.full((self.height, self.width, 3), style.BACKGROUND, np.uint8)
        # Brand and title.
        style.rounded_rect(image, (MARGIN, 16), (MARGIN + 44, 60), style.ACCENT, radius=10)
        style.draw_text(
            image,
            "ST",
            (MARGIN + 22, 38),
            18,
            style.readable_on(style.ACCENT),
            weight="Bold",
            anchor="mm",
        )
        style.draw_text(
            image, "Construction Safety Twin", (MARGIN + 58, 16), 24, style.TEXT, weight="SemiBold"
        )
        style.draw_text(image, self.title, (MARGIN + 58, 46), 15, style.TEXT_MUTED)
        # Pane frames and captions.
        for (x, y), caption in (
            (self.left, "SOURCE  ·  annotated CCTV"),
            (self.right, "TWIN  ·  2.5D situational view"),
        ):
            style.rounded_rect(
                image,
                (x - 1, y - 1),
                (x + self.pane[0], y + self.pane[1]),
                style.SURFACE,
                radius=10,
                border=style.STROKE,
            )
            del caption
        # Timeline lanes.
        style.rounded_rect(
            image,
            (MARGIN, self.timeline_top),
            (self.width - MARGIN, self.timeline_top + TIMELINE_H),
            style.SURFACE,
            radius=12,
            border=style.STROKE,
        )
        style.draw_text(
            image,
            "TIMELINE",
            (MARGIN + 14, self.timeline_top + 10),
            12,
            style.TEXT_FAINT,
            weight="SemiBold",
        )
        for lane, rule in enumerate(RuleId):
            y = self._lane_y(lane)
            style.draw_text(
                image,
                rule.value,
                (MARGIN + 18, y),
                13,
                style.TEXT_MUTED,
                weight="SemiBold",
                anchor="lm",
            )
            cv2.line(image, (self.lane_x0, y), (self.lane_x1, y), style.STROKE, 1, cv2.LINE_AA)
            for incident in (i for i in self.incidents if i.rule_id is rule):
                end = (
                    incident.resolved_at_s
                    if incident.resolved_at_s is not None
                    else self.duration_s
                )
                a, b = (
                    self._x(incident.first_seen_s),
                    max(self._x(end), self._x(incident.first_seen_s) + 4),
                )
                color = style.SEVERITY.get(incident.severity, style.TEXT_MUTED)
                style.rounded_rect(image, (a, y - 6), (b, y + 6), color, radius=6, alpha=0.9)
        for tick in np.arange(0, self.duration_s + 1e-6, self._tick_step()):
            x = self._x(float(tick))
            bottom = self._lane_y(len(RuleId) - 1) + 18
            cv2.line(image, (x, bottom - 4), (x, bottom), style.TEXT_FAINT, 1)
            style.draw_text(
                image, _clock(float(tick))[:-2], (x, bottom + 3), 11, style.TEXT_FAINT, anchor="mt"
            )
        # Footer: coverage and disclaimer.
        x = MARGIN
        style.draw_text(
            image,
            "RULE COVERAGE",
            (x, self.footer_top + 12),
            12,
            style.TEXT_FAINT,
            weight="SemiBold",
            anchor="lm",
        )
        x += 120
        for entry in coverage:
            text = f"{entry.rule_id.value}  {STATUS_TEXT[entry.status]}"
            if entry.incident_count:
                text += f"  ·  {entry.incident_count}"
            color = STATUS_COLOR[entry.status]
            width, _ = style.chip(
                image,
                text,
                (x, self.footer_top + 12),
                size=12,
                fill=style.SURFACE_RAISED,
                color=color,
                anchor="lm",
            )
            x += width + 8
        style.draw_text(
            image,
            DISCLAIMER,
            (self.width - MARGIN, self.footer_top + 12),
            13,
            style.TEXT_MUTED,
            anchor="rm",
        )
        return image

    def _tick_step(self) -> float:
        for step in (10, 15, 30, 60, 120, 300, 600):
            if self.duration_s / step <= 12:
                return float(step)
        return 1200.0

    def _lane_y(self, lane: int) -> int:
        return self.timeline_top + 38 + lane * 19

    def _x(self, time_s: float) -> int:
        fraction = min(max(time_s / self.duration_s, 0.0), 1.0)
        return int(self.lane_x0 + fraction * (self.lane_x1 - self.lane_x0))

    # ------------------------------------------------------------------ per frame

    def visible_incidents(self, time_s: float) -> list[IncidentRecord]:
        visible = [
            incident
            for incident in self.incidents
            if incident.confirmed_at_s <= time_s
            and (incident.resolved_at_s is None or time_s <= incident.resolved_at_s + CARD_LINGER_S)
        ]
        visible.sort(key=lambda i: (SEVERITY_RANK.get(i.severity, 9), -i.confirmed_at_s))
        return visible[:MAX_CARDS]

    def compose(
        self,
        source_pane: np.ndarray,
        plan_pane: np.ndarray,
        *,
        time_s: float,
        crops: Mapping[str, np.ndarray],
        workers: int = 0,
        machines: int = 0,
        status: str = "",
    ) -> np.ndarray:
        image = self._base.copy()
        self._paste(image, source_pane, self.left)
        self._paste(image, plan_pane, self.right)
        self._kpis(image, time_s, workers, machines)
        cards = self.visible_incidents(time_s)
        if cards:
            width = (self.width - 2 * MARGIN - (MAX_CARDS - 1) * 12) // MAX_CARDS
            for index, incident in enumerate(cards):
                x = MARGIN + index * (width + 12)
                self._card(
                    image,
                    incident,
                    crops.get(incident.incident_id),
                    (x, self.rail_top),
                    (width, RAIL_H),
                    time_s,
                )
        else:
            self._quiet_rail(image, time_s, status)
        playhead = self._x(time_s)
        top, bottom = self.timeline_top + 28, self._lane_y(len(RuleId) - 1) + 10
        cv2.line(image, (playhead, top), (playhead, bottom), style.TEXT, 2, cv2.LINE_AA)
        cv2.circle(image, (playhead, top), 4, style.TEXT, -1, cv2.LINE_AA)
        return image

    def _kpis(self, image: np.ndarray, time_s: float, workers: int, machines: int) -> None:
        active = sum(
            1
            for i in self.incidents
            if i.confirmed_at_s <= time_s and (i.resolved_at_s is None or time_s <= i.resolved_at_s)
        )
        so_far = sum(1 for i in self.incidents if i.confirmed_at_s <= time_s)
        tiles = [
            ("WORKERS", str(workers), style.TEXT),
            ("MACHINES", str(machines), style.MACHINE if machines else style.TEXT),
            ("ACTIVE ALERTS", str(active), style.DANGER if active else style.OK),
            ("INCIDENTS", str(so_far), style.TEXT),
        ]
        x = self.width - MARGIN
        clock = _clock(time_s)
        cw, _ = style.text_size(clock, 30, "SemiBold")
        style.draw_text(image, clock, (x, 40), 30, style.TEXT, weight="SemiBold", anchor="rm")
        x -= cw + 22
        for label, value, color in reversed(tiles):
            width = (
                max(
                    style.text_size(label, 11, "SemiBold")[0], style.text_size(value, 22, "Bold")[0]
                )
                + 28
            )
            style.rounded_rect(
                image, (x - width, 14), (x, 66), style.SURFACE, radius=10, border=style.STROKE
            )
            style.draw_text(
                image, label, (x - width + 14, 20), 11, style.TEXT_FAINT, weight="SemiBold"
            )
            style.draw_text(image, value, (x - width + 14, 36), 22, color, weight="Bold")
            x -= width + 8

    def _quiet_rail(self, image: np.ndarray, time_s: float, status: str) -> None:
        style.rounded_rect(
            image,
            (MARGIN, self.rail_top),
            (self.width - MARGIN, self.rail_top + RAIL_H),
            style.SURFACE,
            radius=12,
            border=style.STROKE,
        )
        style.draw_text(
            image,
            "No active alerts",
            (MARGIN + 24, self.rail_top + 30),
            22,
            style.OK,
            weight="SemiBold",
        )
        so_far = sum(1 for i in self.incidents if i.confirmed_at_s <= time_s)
        style.draw_text(
            image,
            f"{so_far} incident(s) confirmed so far in this clip.",
            (MARGIN + 24, self.rail_top + 68),
            16,
            style.TEXT_MUTED,
        )
        if status:
            style.draw_text(image, status, (MARGIN + 24, self.rail_top + 98), 15, style.TEXT_FAINT)

    def _paste(self, image: np.ndarray, pane: np.ndarray, origin: tuple[int, int]) -> None:
        x, y = origin
        if (pane.shape[1], pane.shape[0]) != self.pane:
            pane = cv2.resize(pane, self.pane, interpolation=cv2.INTER_AREA)
        image[y : y + self.pane[1], x : x + self.pane[0]] = pane

    def _card(self, image, incident: IncidentRecord, crop, origin, size, time_s: float) -> None:
        x, y = origin
        width, height = size
        color = style.SEVERITY.get(incident.severity, style.TEXT_MUTED)
        style.rounded_rect(
            image, (x, y), (x + width, y + height), style.SURFACE, radius=12, border=style.STROKE
        )
        style.rounded_rect(image, (x, y), (x + 5, y + height), color, radius=2)
        text_x = x + 20
        if crop is not None and crop.size:
            crop_h = height - 28
            crop_w = min(int(crop_h * crop.shape[1] / max(crop.shape[0], 1)), width // 3)
            thumb = cv2.resize(crop, (max(crop_w, 1), crop_h), interpolation=cv2.INTER_CUBIC)
            image[y + 14 : y + 14 + crop_h, x + 18 : x + 18 + thumb.shape[1]] = thumb
            cv2.rectangle(
                image,
                (x + 18, y + 14),
                (x + 18 + thumb.shape[1], y + 14 + crop_h),
                color,
                2,
                cv2.LINE_AA,
            )
            text_x = x + 18 + thumb.shape[1] + 18
        text_w = x + width - 16 - text_x
        title, progress = card_headline(incident, time_s)
        chip_w, _ = style.chip(
            image,
            f"{incident.rule_id.value}  {incident.severity.upper()}",
            (text_x, y + 16),
            size=12,
            fill=color,
            color=style.readable_on(color),
        )
        if incident.basis == "heuristic":
            style.chip(
                image,
                "HEURISTIC",
                (text_x + chip_w + 6, y + 16),
                size=11,
                fill=style.SURFACE_RAISED,
                color=style.TEXT_MUTED,
            )
        cursor = y + 50
        for line in style.wrap(title, 19, text_w, "SemiBold")[:2]:
            style.draw_text(image, line, (text_x, cursor), 19, style.TEXT, weight="SemiBold")
            cursor += 26
        style.draw_text(image, progress, (text_x, cursor + 2), 15, color, weight="Medium")
        cursor += 30
        for line in style.wrap(incident.action_text, 14, text_w)[:3]:
            style.draw_text(image, line, (text_x, cursor), 14, style.TEXT_MUTED)
            cursor += 20
        if incident.references and cursor < y + height - 24:
            refs = ", ".join(incident.references)
            for line in style.wrap(f"Ref: {refs}", 12, text_w)[:1]:
                style.draw_text(image, line, (text_x, y + height - 26), 12, style.TEXT_FAINT)
