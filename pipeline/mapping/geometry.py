"""Ground-plane geometry for R2/R4/R5: footprints, distance intervals and operating state.

Every distance is computed for all bootstrap replicates at once and summarised as a 95 %
interval; a band is only assigned when the whole interval lies inside it (v7 §37.3).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from shared.coordinates import DistanceInterval
from shared.enums import DistanceBand, OperatingState

# A machine's ground contact is its box bottom edge; its extent into the scene is unseen,
# so the footprint is extruded away from the camera by this fraction of its width.
FOOTPRINT_DEPTH_FRACTION = 0.6
MINIMUM_VALID_REPLICATES = 0.7


def assign_band(interval: DistanceInterval, near_max: float, caution_max: float) -> DistanceBand:
    if interval.upper_wh <= near_max:
        return DistanceBand.NEAR
    if interval.lower_wh > near_max and interval.upper_wh <= caution_max:
        return DistanceBand.CAUTION
    if interval.lower_wh > caution_max:
        return DistanceBand.CLEAR
    return DistanceBand.INDETERMINATE


def _column_percentiles(samples: np.ndarray, quantiles: tuple[float, ...]):
    """Per-column percentiles ignoring NaN, fully vectorised.

    `np.nanpercentile` falls back to a Python loop per column when NaNs are present, which
    dominated pass two; sorting once (NaN sorts last) and interpolating is equivalent.
    """
    ordered = np.sort(samples, axis=0)
    valid = np.isfinite(samples).sum(axis=0)
    columns = np.arange(samples.shape[1])
    results = []
    for q in quantiles:
        position = q / 100.0 * np.maximum(valid - 1, 0)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, np.maximum(valid - 1, 0))
        fraction = position - low
        values = ordered[low, columns] * (1 - fraction) + ordered[high, columns] * fraction
        results.append(values)
    return results, valid


def summarise_many(samples: np.ndarray, total: int) -> list[DistanceInterval | None]:
    """Intervals for each column of (R, N) replicate distances, in one pass."""
    if samples.size == 0:
        return []
    (lower, median, upper), valid = _column_percentiles(samples, (2.5, 50.0, 97.5))
    intervals: list[DistanceInterval | None] = []
    for index in range(samples.shape[1]):
        if valid[index] == 0 or valid[index] < MINIMUM_VALID_REPLICATES * samples.shape[0]:
            intervals.append(None)
            continue
        intervals.append(
            DistanceInterval(
                lower_wh=round(float(lower[index]), 4),
                median_wh=round(float(median[index]), 4),
                upper_wh=round(float(upper[index]), 4),
                successful_bootstraps=int(valid[index]),
                total_bootstraps=int(total),
            )
        )
    return intervals


def summarise(samples: np.ndarray, total: int) -> DistanceInterval | None:
    """95 % interval over replicate distances; None if too many replicates failed."""
    finite = samples[np.isfinite(samples)]
    if len(finite) < MINIMUM_VALID_REPLICATES * len(samples) or len(finite) == 0:
        return None
    lower, median, upper = np.percentile(finite, [2.5, 50, 97.5])
    return DistanceInterval(
        lower_wh=round(float(lower), 4),
        median_wh=round(float(median), 4),
        upper_wh=round(float(upper), 4),
        successful_bootstraps=int(len(finite)),
        total_bootstraps=int(total),
    )


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Points (R, N, 2) to segments a→b (R, K, 2) → (R, N, K)."""
    ab = (b - a)[:, None]  # (R, 1, K, 2)
    ap = points[:, :, None, :] - a[:, None]  # (R, N, K, 2)
    length2 = np.maximum((ab**2).sum(-1), 1e-12)
    t = np.clip((ap * ab).sum(-1) / length2, 0.0, 1.0)
    closest = a[:, None] + t[..., None] * ab
    return np.linalg.norm(points[:, :, None, :] - closest, axis=-1)


def _as_batch(points: np.ndarray) -> tuple[np.ndarray, bool]:
    return (points[:, None, :], True) if points.ndim == 2 else (points, False)


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """points (R, 2) or (R, N, 2), polyline (R, K, 2) → (R,) or (R, N) minimum distance."""
    batch, single = _as_batch(points)
    distances = _segment_distances(batch, polyline[:, :-1], polyline[:, 1:]).min(axis=-1)
    return distances[:, 0] if single else distances


def distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """points (R, 2) or (R, N, 2), simple polygon (R, K, 2) → (R,) or (R, N); 0 inside."""
    batch, single = _as_batch(points)
    closed_b = np.roll(polygon, -1, axis=1)
    edges = _segment_distances(batch, polygon, closed_b).min(axis=-1)
    # Even-odd ray casting, vectorised over replicates and points.
    x, y = batch[..., 0:1], batch[..., 1:2]  # (R, N, 1)
    x1, y1 = polygon[:, None, :, 0], polygon[:, None, :, 1]  # (R, 1, K)
    x2, y2 = closed_b[:, None, :, 0], closed_b[:, None, :, 1]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    inside = (crosses & (x < x_at)).sum(axis=-1) % 2 == 1
    distances = np.where(inside, 0.0, edges)
    return distances[:, 0] if single else distances


def footprint_from_contact(contact: np.ndarray, camera_xy: np.ndarray | None = None) -> np.ndarray:
    """Quad footprint (R, 4, 2) from ground contact endpoints (R, 2, 2).

    The visible bottom edge is extruded away from the camera by a fraction of its width.
    """
    left, right = contact[:, 0], contact[:, 1]
    edge = right - left
    width = np.linalg.norm(edge, axis=-1, keepdims=True)
    normal = np.stack([-edge[:, 1], edge[:, 0]], axis=-1) / np.maximum(width, 1e-9)
    origin = np.zeros(2) if camera_xy is None else camera_xy
    midpoint = (left + right) / 2
    away = np.sign(((midpoint - origin) * normal).sum(-1, keepdims=True))
    away[away == 0] = 1.0
    depth = normal * away * width * FOOTPRINT_DEPTH_FRACTION
    return np.stack([left, right, right + depth, left + depth], axis=1)


@dataclass
class _MachineHistory:
    samples: deque  # (time_s, ground_xy or None, motion_energy or None)


class OperatingStateTracker:
    """Active / stationary / unknown from the trailing window only (v7 §38).

    Active: the base moves (speed or displacement over the window), or the body keeps
    articulating (frame-difference energy inside the box) while the base stays put.
    Unknown: not enough valid observations in the window. Scene context alone never makes
    a machine active.
    """

    def __init__(
        self,
        *,
        window_s: float = 2.0,
        active_speed_wh_s: float = 0.15,
        active_displacement_wh: float = 0.25,
        stationary_speed_wh_s: float = 0.05,
        active_energy: float = 0.045,
        stationary_energy: float = 0.02,
        minimum_valid_fraction: float = 0.7,
        expected_fps: float = 15.0,
    ) -> None:
        self.window_s = window_s
        self.active_speed = active_speed_wh_s
        self.active_displacement = active_displacement_wh
        self.stationary_speed = stationary_speed_wh_s
        self.active_energy = active_energy
        self.stationary_energy = stationary_energy
        self.minimum_valid_fraction = minimum_valid_fraction
        self.expected = window_s * expected_fps
        self._machines: dict[str, _MachineHistory] = {}

    def reset(self) -> None:
        self._machines.clear()

    def update(
        self,
        machine_id: str,
        time_s: float,
        ground_xy: np.ndarray | None,
        motion_energy: float | None,
    ) -> OperatingState:
        history = self._machines.setdefault(machine_id, _MachineHistory(deque()))
        history.samples.append((time_s, ground_xy, motion_energy))
        while history.samples and history.samples[0][0] < time_s - self.window_s:
            history.samples.popleft()
        valid = [s for s in history.samples if s[1] is not None and np.all(np.isfinite(s[1]))]
        if len(valid) < self.minimum_valid_fraction * self.expected:
            return OperatingState.UNKNOWN
        span = valid[-1][0] - valid[0][0]
        displacement = float(np.linalg.norm(valid[-1][1] - valid[0][1]))
        speed = displacement / span if span > 0 else 0.0
        energies = [s[2] for s in history.samples if s[2] is not None]
        energy = float(np.median(energies)) if energies else None
        if speed >= self.active_speed or displacement >= self.active_displacement:
            return OperatingState.ACTIVE
        if energy is not None and energy >= self.active_energy:
            return OperatingState.ACTIVE
        if (
            speed <= self.stationary_speed
            and energy is not None
            and energy <= self.stationary_energy
        ):
            return OperatingState.STATIONARY
        return OperatingState.UNKNOWN
