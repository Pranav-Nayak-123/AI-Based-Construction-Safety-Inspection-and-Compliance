"""Compose the 1920x1080 side-by-side frame (v7 §15.4).

    +------------------------------------------------------------------+
    | header: title, clock                                             |
    | [ annotated source 940x529 ]      [ 2.5D plan view 940x529 ]     |
    | incident rail: active incident cards with magnified subject crop |
    | timeline: every incident span by severity, playhead              |
    | footer: R1-R5 coverage, disclaimer                               |
    +------------------------------------------------------------------+

Cards appear from an incident's *confirmation* time, never earlier, so the video never
shows the system knowing something before its debounce did.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from pipeline.render.source_overlay import DISCLAIMER, FONT, draw_label
from shared.config import RenderConfig
from shared.enums import RuleStatus
from shared.identity import display_track_label
from shared.schemas.incidents import IncidentRecord
from shared.schemas.run import RuleCoverageEntry

BACKGROUND = (24, 22, 20)
PANEL = (44, 40, 36)
TEXT = (235, 235, 235)
MUTED = (160, 160, 160)
SEVERITY_COLORS = {
    "critical": (40, 30, 170),
    "high": (50, 50, 235),
    "medium": (0, 165, 255),
    "low": (0, 200, 200),
}
STATUS_TEXT = {
    RuleStatus.EVALUATED_ALERT: "alert",
    RuleStatus.EVALUATED_CLEAR: "clear",
    RuleStatus.INCONCLUSIVE: "inconclusive",
    RuleStatus.NOT_APPLICABLE: "n/a",
    RuleStatus.UNSUPPORTED: "unsupported",
}
MARGIN = 13
HEADER_H = 66
RAIL_GAP = 14
TIMELINE_H = 58
FOOTER_H = 44
MAX_CARDS = 3
CARD_LINGER_S = 2.0


def wrap_text(text: str, width: int, scale: float, thickness: int = 1) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cv2.getTextSize(candidate, FONT, scale, thickness)[0][0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


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
        self.left = (MARGIN, HEADER_H)
        self.right = (self.width - MARGIN - self.pane[0], HEADER_H)
        self.rail_top = HEADER_H + self.pane[1] + RAIL_GAP
        self.footer_top = self.height - FOOTER_H
        self.timeline_top = self.footer_top - TIMELINE_H
        self.rail_height = self.timeline_top - RAIL_GAP - self.rail_top
        self._base = self._static_layer(coverage)

    # ------------------------------------------------------------------ static parts

    def _static_layer(self, coverage: Sequence[RuleCoverageEntry]) -> np.ndarray:
        image = np.full((self.height, self.width, 3), BACKGROUND, np.uint8)
        cv2.putText(image, self.title, (MARGIN, 30), FONT, 0.8, TEXT, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "Annotated source",
            (self.left[0], HEADER_H - 9),
            FONT,
            0.45,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "2.5D situational twin",
            (self.right[0], HEADER_H - 9),
            FONT,
            0.45,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        # Timeline track with every incident span.
        x0, x1 = MARGIN, self.width - MARGIN
        y0 = self.timeline_top + 22
        cv2.rectangle(image, (x0, y0), (x1, y0 + 18), PANEL, -1)
        for incident in self.incidents:
            end = incident.resolved_at_s if incident.resolved_at_s is not None else self.duration_s
            a, b = self._timeline_x(incident.first_seen_s), self._timeline_x(end)
            color = SEVERITY_COLORS.get(incident.severity, MUTED)
            cv2.rectangle(image, (a, y0 + 2), (max(a + 2, b), y0 + 16), color, -1)
            cv2.putText(
                image, incident.rule_id.value, (a, y0 - 4), FONT, 0.4, color, 1, cv2.LINE_AA
            )
        cv2.putText(
            image, "Timeline", (x0, self.timeline_top + 12), FONT, 0.45, MUTED, 1, cv2.LINE_AA
        )
        # Footer: coverage chips and the disclaimer.
        chips = "   ".join(
            f"{entry.rule_id.value}: {STATUS_TEXT[entry.status]}"
            + (f" ({entry.incident_count})" if entry.incident_count else "")
            for entry in coverage
        )
        cv2.putText(
            image,
            f"Rule coverage   {chips}",
            (MARGIN, self.footer_top + 20),
            FONT,
            0.5,
            TEXT,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image, DISCLAIMER, (MARGIN, self.footer_top + 38), FONT, 0.45, MUTED, 1, cv2.LINE_AA
        )
        return image

    def _timeline_x(self, time_s: float) -> int:
        fraction = min(max(time_s / self.duration_s, 0.0), 1.0)
        return int(MARGIN + fraction * (self.width - 2 * MARGIN))

    # ------------------------------------------------------------------ per frame

    def visible_incidents(self, time_s: float) -> list[IncidentRecord]:
        visible = [
            incident
            for incident in self.incidents
            if incident.confirmed_at_s <= time_s
            and (incident.resolved_at_s is None or time_s <= incident.resolved_at_s + CARD_LINGER_S)
        ]
        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        visible.sort(key=lambda i: (severity_rank.get(i.severity, 9), -i.confirmed_at_s))
        return visible[:MAX_CARDS]

    def compose(
        self,
        source_pane: np.ndarray,
        plan_pane: np.ndarray,
        *,
        time_s: float,
        crops: Mapping[str, np.ndarray],
        status: str = "",
    ) -> np.ndarray:
        image = self._base.copy()
        self._paste(image, source_pane, self.left)
        self._paste(image, plan_pane, self.right)
        minutes, seconds = divmod(time_s, 60.0)
        clock = f"{int(minutes):02d}:{seconds:05.2f}"
        cv2.putText(image, clock, (self.width - MARGIN - 130, 30), FONT, 0.8, TEXT, 2, cv2.LINE_AA)

        cards = self.visible_incidents(time_s)
        if cards:
            card_width = (self.width - 2 * MARGIN - (MAX_CARDS - 1) * RAIL_GAP) // MAX_CARDS
            for index, incident in enumerate(cards):
                x = MARGIN + index * (card_width + RAIL_GAP)
                self._card(
                    image,
                    incident,
                    crops.get(incident.incident_id),
                    (x, self.rail_top),
                    (card_width, self.rail_height),
                    time_s,
                )
        else:
            confirmed = sum(1 for i in self.incidents if i.confirmed_at_s <= time_s)
            cv2.putText(
                image,
                f"No active alerts - {confirmed} incident(s) so far",
                (MARGIN, self.rail_top + 30),
                FONT,
                0.6,
                MUTED,
                1,
                cv2.LINE_AA,
            )
            if status:
                cv2.putText(
                    image, status, (MARGIN, self.rail_top + 60), FONT, 0.5, MUTED, 1, cv2.LINE_AA
                )

        playhead = self._timeline_x(time_s)
        cv2.line(
            image, (playhead, self.timeline_top + 16), (playhead, self.timeline_top + 46), TEXT, 2
        )
        return image

    def _paste(self, image: np.ndarray, pane: np.ndarray, origin: tuple[int, int]) -> None:
        x, y = origin
        if (pane.shape[1], pane.shape[0]) != self.pane:
            pane = cv2.resize(pane, self.pane, interpolation=cv2.INTER_AREA)
        image[y : y + self.pane[1], x : x + self.pane[0]] = pane

    def _card(
        self,
        image: np.ndarray,
        incident: IncidentRecord,
        crop: np.ndarray | None,
        origin: tuple[int, int],
        size: tuple[int, int],
        time_s: float,
    ) -> None:
        x, y = origin
        width, height = size
        color = SEVERITY_COLORS.get(incident.severity, MUTED)
        cv2.rectangle(image, (x, y), (x + width, y + height), PANEL, -1)
        cv2.rectangle(image, (x, y), (x + 6, y + height), color, -1)

        text_x = x + 18
        crop_width = 0
        if crop is not None and crop.size:
            crop_height = height - 24
            crop_width = min(int(crop_height * crop.shape[1] / crop.shape[0]), width // 3)
            resized = cv2.resize(crop, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            image[y + 12 : y + 12 + crop_height, x + 14 : x + 14 + crop_width] = resized
            cv2.rectangle(
                image, (x + 14, y + 12), (x + 14 + crop_width, y + 12 + crop_height), color, 2
            )
            text_x = x + 14 + crop_width + 14
        text_width = x + width - 12 - text_x

        status = (
            "ongoing"
            if incident.resolved_at_s is None or time_s < incident.resolved_at_s
            else "resolved"
        )
        draw_label(
            image,
            f"{incident.rule_id.value}  {incident.severity.upper()}",
            (text_x, y + 30),
            color,
            scale=0.55,
        )
        lines = [
            (incident.observation_text, TEXT, 0.5),
            (
                f"Worker {display_track_label(incident.canonical_track_id or '')} - {status}",
                MUTED,
                0.45,
            ),
            (f"Action: {incident.action_text}", TEXT, 0.45),
        ]
        if incident.references:
            lines.append(("Ref: " + ", ".join(incident.references), MUTED, 0.42))
        if incident.basis == "heuristic":
            lines.append(("Heuristic observation", MUTED, 0.42))
        cursor = y + 58
        for text, text_color, scale in lines:
            for line in wrap_text(text, text_width, scale):
                if cursor > y + height - 8:
                    return
                cv2.putText(image, line, (text_x, cursor), FONT, scale, text_color, 1, cv2.LINE_AA)
                cursor += int(28 * scale + 8)
            cursor += 4
