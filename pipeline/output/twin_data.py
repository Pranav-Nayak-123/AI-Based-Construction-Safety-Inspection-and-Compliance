"""Digital-twin data: smoothed worker and machine positions on the ground plane (v3).

Recorded during pass two from the same mapped positions the rules use, then smoothed
offline with a centred Gaussian (pass two has the whole clip, so there is no lag — a
live filter would trail every worker). Tracks are split where a worker disappears for
more than `GAP_SPLIT_S`, so nobody glides across the site while unseen.

Two consumers:

* the video's twin pane reads `positions_at(frame_index)` (full frame rate);
* the report's 3D viewer reads `export()` → `twin.json` (sampled at `SAMPLE_HZ`,
  interpolated to 60 fps in the browser).

Coordinates are the calibrated ground plane in worker heights (WH): x to the right of
the camera, y away from it, the camera at the origin.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from shared.enums import AlertState, RuleId
from shared.identity import display_track_label
from shared.schemas.frame import FrameContext, FrameRuleOutput
from shared.schemas.scene import GeometryType, ProposalType, SceneProposal

SMOOTH_SIGMA_S = 0.4
GAP_SPLIT_S = 0.6
SAMPLE_HZ = 5.0
FALLBACK_FAR_WH = 40.0  # beyond this a default camera's guess is too far off to draw
RULE_BITS = {rule: 1 << index for index, rule in enumerate(RuleId)}


@dataclass
class _Sample:
    frame: int
    t: float
    x: float
    y: float
    state: int  # 0 clear, 1 pending, 2 alert
    rules: int  # bitmask of RuleId
    helmet: str  # h helmet, n no helmet, u unknown
    supported: bool


@dataclass
class TwinRecorder:
    fps: float
    _workers: dict[str, list[_Sample]] = field(default_factory=lambda: defaultdict(list))
    _machines: dict[str, list[tuple[int, float, list, str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _positions: dict[int, dict[str, tuple[float, float]]] = field(default_factory=dict)
    end_time_s: float = 0.0
    # Uncalibrated but steady camera: foot pixels → ground through a default camera, for the
    # twin's layout only (rules never see these positions).
    fallback: Callable[[np.ndarray], np.ndarray] | None = None

    def _position(self, worker) -> tuple[float, float] | None:  # noqa: ANN001
        point = worker.relative_position
        if point is not None and np.isfinite(point.x) and np.isfinite(point.y):
            return point.x, point.y
        if point is None and self.fallback is not None:
            anchor = worker.anchor_reference
            x, y = self.fallback(np.array([[anchor.x, anchor.y]], np.float64))[0]
            if np.isfinite(x) and np.isfinite(y) and 0.0 <= y <= FALLBACK_FAR_WH:
                return float(x), float(y)
        return None

    def add(self, context: FrameContext, output: FrameRuleOutput) -> None:
        self.end_time_s = max(self.end_time_s, context.video_time_s)
        states: dict[str, tuple[int, int]] = {}
        for alert in output.active_alerts:
            state, rules = states.get(alert.track_id, (0, 0))
            level = 1 if alert.state is AlertState.PENDING_ALERT else 2
            states[alert.track_id] = (max(state, level), rules | RULE_BITS[alert.rule_id])
        for worker in context.workers():
            point = self._position(worker)
            if point is None:
                continue
            state, rules = states.get(worker.canonical_track_id, (0, 0))
            helmet = worker.helmet.state.value if worker.helmet else "unknown"
            self._workers[worker.canonical_track_id].append(
                _Sample(
                    frame=context.frame_index,
                    t=context.video_time_s,
                    x=point[0],
                    y=point[1],
                    state=state,
                    rules=rules,
                    helmet={"helmet": "h", "no_helmet": "n"}.get(helmet, "u"),
                    supported=worker.plane_id is not None,
                )
            )
        for machine in context.machinery:
            if machine.polygon_plane_xy:
                self._machines[machine.machinery_id].append(
                    (
                        context.frame_index,
                        context.video_time_s,
                        machine.polygon_plane_xy,
                        machine.class_name,
                        machine.operating_state.value,
                    )
                )

    # ------------------------------------------------------------------ smoothing

    def _segments(self, samples: list[_Sample]) -> list[list[_Sample]]:
        segments: list[list[_Sample]] = []
        for sample in samples:
            if segments and sample.t - segments[-1][-1].t <= GAP_SPLIT_S:
                segments[-1].append(sample)
            else:
                segments.append([sample])
        return segments

    def smooth(self) -> None:
        """Smooth every segment in place and index positions by frame."""
        sigma = SMOOTH_SIGMA_S * self.fps
        self._positions.clear()
        for track_id, samples in self._workers.items():
            for segment in self._segments(samples):
                if len(segment) >= 3:
                    xs = gaussian_filter1d(np.array([s.x for s in segment]), sigma, mode="nearest")
                    ys = gaussian_filter1d(np.array([s.y for s in segment]), sigma, mode="nearest")
                    for sample, x, y in zip(segment, xs, ys, strict=True):
                        sample.x, sample.y = float(x), float(y)
                for sample in segment:
                    self._positions.setdefault(sample.frame, {})[track_id] = (sample.x, sample.y)

    def positions_seen(self) -> bool:
        return bool(self._workers)

    def positions_at(self, frame_index: int) -> Mapping[str, tuple[float, float]]:
        return self._positions.get(frame_index, {})

    # ------------------------------------------------------------------ export

    def export(
        self,
        *,
        proposals: Sequence[SceneProposal],
        ground: Callable[[np.ndarray], np.ndarray],
        camera: Mapping[str, object],
    ) -> dict:
        """JSON-ready twin: camera, zones and edges, worker and machine paths."""
        step = max(1, round(self.fps / SAMPLE_HZ))
        workers = []
        for track_id, samples in sorted(self._workers.items(), key=lambda kv: kv[1][0].t):
            segments = []
            for segment in self._segments(samples):
                picked = segment[::step]
                if picked[-1] is not segment[-1]:
                    picked.append(segment[-1])
                segments.append(
                    {
                        "t": [round(s.t, 3) for s in picked],
                        "xy": [v for s in picked for v in (round(s.x, 3), round(s.y, 3))],
                        "state": "".join(str(s.state) for s in picked),
                        "rules": [s.rules for s in picked],
                        "helmet": "".join(s.helmet for s in picked),
                        "ok": "".join("1" if s.supported else "0" for s in picked),
                    }
                )
            workers.append(
                {"id": track_id, "label": display_track_label(track_id), "segments": segments}
            )
        machines = []
        for machine_id, rows in self._machines.items():
            picked = rows[::step]
            machines.append(
                {
                    "id": machine_id,
                    "label": display_track_label(machine_id),
                    "class": picked[0][3],
                    "t": [round(r[1], 3) for r in picked],
                    "polygons": [[[round(x, 3), round(y, 3)] for x, y in r[2]] for r in picked],
                    "state": [r[4] for r in picked],
                }
            )
        zones = []
        for proposal in proposals:
            points = np.asarray(proposal.reference_points, np.float64)
            if proposal.geometry_type is GeometryType.BOX:
                (x1, y1), (x2, y2) = points
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            mapped = np.asarray(ground(points))
            if not np.all(np.isfinite(mapped)):
                continue
            zones.append(
                {
                    "id": proposal.proposal_id,
                    "label": proposal.label or proposal.proposal_id,
                    "type": proposal.proposal_type.value,
                    "line": proposal.proposal_type is ProposalType.OPEN_EDGE,
                    "points": [[round(float(x), 3), round(float(y), 3)] for x, y in mapped],
                }
            )
        return {
            "version": 1,
            "units": "worker heights (WH); x right of the camera, y away from it",
            "duration": round(self.end_time_s, 3),
            "camera": dict(camera),
            "rules": {rule.value: bit for rule, bit in RULE_BITS.items()},
            "zones": zones,
            "workers": workers,
            "machines": machines,
        }
