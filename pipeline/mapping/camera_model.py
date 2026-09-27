"""Camera calibration from the workers themselves (v7 §10.3, §37).

Model: pinhole camera, square pixels, principal point at the canvas centre, looking down
at one locally planar ground at pitch `tilt` with a small `roll`, mounted `height` above
the ground. Units are worker heights (WH): the gauge is fixed by treating an upright
person as exactly 1 WH, so no distance in this module is ever in metres.

Fit: each accepted observation pairs the foot (box bottom-centre) and head (box
top-centre) of an upright person in one frame. The foot is back-projected onto the
ground, a 1 WH vertical is raised there and re-projected, and the robust (Huber) error
against the observed head is minimised over (focal, tilt, roll, height). Pairs are never
split: the pair is the unit of evidence (v7 §37.1).

Uncertainty: a block bootstrap over tracks refits the camera many times; distances are
then reported as intervals over the replicates (v7 §37.3), never as point estimates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from shared.coordinates import StrictModel

# Search starts for the focal length, in canvas widths (about 90° to 28° horizontal FOV).
FOCAL_STARTS = (0.5, 0.75, 1.1, 1.6)
HUBER_SCALE_FRACTION = 0.04  # of median person height, in pixels
INLIER_FRACTION = 0.12  # a head within 12 % of person height counts as an inlier


@dataclass(frozen=True)
class VerticalPair:
    """Foot and head pixel of one upright person in one frame (canvas coordinates)."""

    track: str
    foot: tuple[float, float]
    head: tuple[float, float]


class CameraParameters(StrictModel):
    focal_px: float
    tilt_rad: float
    roll_rad: float
    height_wh: float
    principal: tuple[float, float]


def rotation(tilt: float, roll: float) -> np.ndarray:
    """World → camera rotation. World: X right, Y forward along the ground, Z up."""
    ct, st = math.cos(tilt), math.sin(tilt)
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, -st, -ct], [0.0, ct, -st]])
    cr, sr = math.cos(roll), math.sin(roll)
    rolled = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rolled @ pitch


class Camera:
    """Vectorised projection/back-projection for one parameter set."""

    def __init__(self, parameters: CameraParameters) -> None:
        self.p = parameters
        self.R = rotation(parameters.tilt_rad, parameters.roll_rad)
        self.center = np.array([0.0, 0.0, parameters.height_wh])

    def project(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) world points → (N, 2) pixels; points behind the camera become NaN."""
        cam = (points - self.center) @ self.R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(cam[:, 2] > 1e-6, cam[:, 2], np.nan)
            u = self.p.principal[0] + self.p.focal_px * cam[:, 0] / z
            v = self.p.principal[1] + self.p.focal_px * cam[:, 1] / z
        return np.stack([u, v], axis=1)

    def ground(self, pixels: np.ndarray) -> np.ndarray:
        """(N, 2) pixels → (N, 2) ground XY in WH; above the horizon becomes NaN."""
        rays = (
            np.stack(
                [
                    (pixels[:, 0] - self.p.principal[0]) / self.p.focal_px,
                    (pixels[:, 1] - self.p.principal[1]) / self.p.focal_px,
                    np.ones(len(pixels)),
                ],
                axis=1,
            )
            @ self.R
        )  # camera → world directions (R is orthonormal)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(rays[:, 2] < -1e-9, -self.p.height_wh / rays[:, 2], np.nan)
        return self.center[:2] + rays[:, :2] * t[:, None]

    def horizon_v(self, u: float) -> float:
        """Image row of the horizon at column `u` (for drawing and gating)."""
        far = np.array([[u, 0.0]])
        # Solve for the row where the ray is parallel to the ground.
        low, high = -10 * self.p.focal_px, 10 * self.p.focal_px
        for _ in range(60):
            mid = (low + high) / 2
            far[0, 1] = mid
            if np.isnan(self.ground(far)[0, 0]):
                low = mid
            else:
                high = mid
        return high


def _residuals(
    params: np.ndarray,
    feet: np.ndarray,
    heads: np.ndarray,
    cx: float,
    cy: float,
    focal: float | None = None,
    vertical_only: bool = False,
) -> np.ndarray:
    if focal is None:
        focal, tilt, roll, height = params
    else:
        tilt, roll, height = params
    camera = Camera(
        CameraParameters(
            focal_px=focal, tilt_rad=tilt, roll_rad=roll, height_wh=height, principal=(cx, cy)
        )
    )
    ground = camera.ground(feet)
    heads_world = np.column_stack([ground, np.ones(len(ground))])
    error = camera.project(heads_world) - heads
    if vertical_only:
        error = error[:, 1:]
    return np.nan_to_num(error, nan=1e4).ravel()


class PlaneFit(StrictModel):
    """An accepted (or rejected) calibration with its diagnostics."""

    accepted: bool
    reason_code: str
    parameters: CameraParameters | None = None
    observations: int = 0
    tracks: int = 0
    inlier_ratio: float | None = None
    relative_rms: float | None = None
    horizon_shift_px: float | None = None
    support_polygon: list[tuple[float, float]] = []
    bootstrap: list[list[float]] = []  # [focal, tilt, roll, height] per replicate


@dataclass
class _Fit:
    params: np.ndarray
    cost: float
    residuals: np.ndarray = field(repr=False)


def _initial_guesses(feet: np.ndarray, heads: np.ndarray, width: float, cy: float):
    """Horizon and camera height from `pixel height ∝ (foot row − horizon row)`."""
    heights = feet[:, 1] - heads[:, 1]
    slope, intercept = np.polyfit(feet[:, 1], heights, 1)
    slope = max(slope, 1e-3)
    horizon = -intercept / slope
    height_wh = 1.0 / slope
    for fraction in FOCAL_STARTS:
        focal = fraction * width
        tilt = math.atan2(cy - horizon, focal)
        yield np.array([focal, tilt, 0.0, max(height_wh, 0.5)])


def _solve(
    feet, heads, cx, cy, width, scale, start=None, *, focal=None, vertical_only=False
) -> _Fit | None:
    """Robust fit of (focal, tilt, roll, height), or (tilt, roll, height) at a fixed focal."""
    if start is not None:
        starts = [start]
    else:
        starts = list(_initial_guesses(feet, heads, width, cy))
        if focal is not None:  # one start is enough when the focal length is given
            starts = [_initial_guesses_at(feet, heads, focal, cy)]
    if focal is None:
        bounds = ([0.3 * width, -0.2, -0.2, 0.2], [4.0 * width, 1.45, 0.2, 200.0])
    else:
        bounds = ([-0.2, -0.2, 0.2], [1.45, 0.2, 200.0])
    best: _Fit | None = None
    for guess in starts:
        guess = np.asarray(guess, np.float64)
        if focal is not None and len(guess) == 4:
            guess = guess[1:]
        guess = np.clip(guess, np.array(bounds[0]) + 1e-6, np.array(bounds[1]) - 1e-6)
        try:
            result = least_squares(
                _residuals,
                guess,
                args=(feet, heads, cx, cy, focal, vertical_only),
                bounds=bounds,
                loss="huber",
                f_scale=scale,
                max_nfev=200,
                x_scale="jac",
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or result.cost < best.cost:
            params = result.x if focal is None else np.concatenate([[focal], result.x])
            best = _Fit(params, float(result.cost), result.fun)
    return best


def _initial_guesses_at(feet: np.ndarray, heads: np.ndarray, focal: float, cy: float) -> np.ndarray:
    heights = feet[:, 1] - heads[:, 1]
    slope, intercept = np.polyfit(feet[:, 1], heights, 1)
    slope = max(slope, 1e-3)
    horizon = -intercept / slope
    return np.array([focal, math.atan2(cy - horizon, focal), 0.0, max(1.0 / slope, 0.5)])


def fit_camera(
    pairs: Sequence[VerticalPair],
    canvas: tuple[int, int],
    *,
    minimum_pairs: int,
    minimum_tracks: int,
    minimum_inlier_ratio: float,
    bootstrap_samples: int,
    minimum_successful: int,
    seed: int,
    max_pairs: int = 600,
    focal_prior: tuple[float, float] | None = None,
    maximum_relative_rms: float = 0.10,
) -> PlaneFit:
    """Calibrate from vertical pairs.

    With `focal_prior` (in canvas widths), the observations are treated as carrying no
    lean information — true of detector boxes, whose head and foot share one column — so
    only the vertical error is fitted, the focal length is taken from the prior, and each
    bootstrap replicate draws its own focal length from that range. The focal uncertainty
    then widens every distance interval instead of being hidden.
    """
    width, height = canvas
    cx, cy = width / 2, height / 2
    tracks = sorted({pair.track for pair in pairs})
    if len(pairs) < minimum_pairs or len(tracks) < minimum_tracks:
        return PlaneFit(
            accepted=False,
            reason_code="too_few_upright_people",
            observations=len(pairs),
            tracks=len(tracks),
        )
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:  # keep every track, thin long ones evenly
        keep = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(keep)]
    feet = np.array([pair.foot for pair in pairs], np.float64)
    heads = np.array([pair.head for pair in pairs], np.float64)
    median_height = float(np.median(feet[:, 1] - heads[:, 1]))
    scale = HUBER_SCALE_FRACTION * median_height

    vertical_only = focal_prior is not None
    nominal = math.sqrt(focal_prior[0] * focal_prior[1]) * width if focal_prior else None
    fit = _solve(feet, heads, cx, cy, width, scale, focal=nominal, vertical_only=vertical_only)
    if fit is None:
        return PlaneFit(accepted=False, reason_code="optimizer_failed", observations=len(pairs))
    errors = (
        np.abs(fit.residuals)
        if vertical_only
        else np.linalg.norm(fit.residuals.reshape(-1, 2), axis=1)
    )
    inliers = errors <= INLIER_FRACTION * (feet[:, 1] - heads[:, 1])
    inlier_ratio = float(inliers.mean())
    relative_rms = (
        float(np.sqrt(np.mean(errors[inliers] ** 2)) / median_height) if inliers.any() else 1.0
    )
    parameters = CameraParameters(
        focal_px=float(fit.params[0]),
        tilt_rad=float(fit.params[1]),
        roll_rad=float(fit.params[2]),
        height_wh=float(fit.params[3]),
        principal=(cx, cy),
    )
    diagnostics = {
        "parameters": parameters,
        "observations": len(pairs),
        "tracks": len(tracks),
        "inlier_ratio": round(inlier_ratio, 4),
        "relative_rms": round(relative_rms, 4),
    }

    # Temporal stability: fit each half of the observations separately.
    half = len(pairs) // 2
    first = _solve(
        feet[:half],
        heads[:half],
        cx,
        cy,
        width,
        scale,
        fit.params,
        focal=nominal,
        vertical_only=vertical_only,
    )
    second = _solve(
        feet[half:],
        heads[half:],
        cx,
        cy,
        width,
        scale,
        fit.params,
        focal=nominal,
        vertical_only=vertical_only,
    )
    shift = None
    if first is not None and second is not None:
        shift = abs(_horizon_row(first.params, cx, cy) - _horizon_row(second.params, cx, cy))
    diagnostics["horizon_shift_px"] = None if shift is None else round(shift, 2)

    reason = "accepted"
    if not vertical_only and not (0.5 * width <= parameters.focal_px <= 3.0 * width):
        reason = "implausible_focal_length"
    elif inlier_ratio < minimum_inlier_ratio:
        reason = "weak_inlier_support"
    elif relative_rms > maximum_relative_rms:
        reason = "high_reprojection_error"
    elif shift is None or shift > 0.02 * height:
        reason = "unstable_horizon"
    if reason != "accepted":
        return PlaneFit(accepted=False, reason_code=reason, **diagnostics)

    camera = Camera(parameters)
    support = _support_polygon(camera.ground(feet[inliers]))
    replicates = _bootstrap(
        pairs,
        feet,
        heads,
        cx,
        cy,
        width,
        scale,
        fit.params,
        bootstrap_samples,
        rng,
        focal_prior=focal_prior,
    )
    if len(replicates) < minimum_successful:
        return PlaneFit(accepted=False, reason_code="bootstrap_unstable", **diagnostics)
    return PlaneFit(
        accepted=True,
        reason_code="accepted",
        support_polygon=support,
        bootstrap=[list(map(float, r)) for r in replicates],
        **diagnostics,
    )


def _horizon_row(params: np.ndarray, cx: float, cy: float) -> float:
    focal, tilt, roll, height = params
    camera = Camera(
        CameraParameters(
            focal_px=focal, tilt_rad=tilt, roll_rad=roll, height_wh=height, principal=(cx, cy)
        )
    )
    return camera.horizon_v(cx)


def _support_polygon(points: np.ndarray, margin_wh: float = 0.5) -> list[tuple[float, float]]:
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return []
    hull = points[ConvexHull(points).vertices]
    centre = hull.mean(axis=0)
    directions = hull - centre
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    grown = centre + directions * (1 + margin_wh / np.maximum(lengths, 1e-9))
    return [(float(x), float(y)) for x, y in grown]


def _bootstrap(
    pairs, feet, heads, cx, cy, width, scale, start, samples, rng, focal_prior=None
) -> list[np.ndarray]:
    """Resample whole tracks with replacement and refit (block bootstrap, v7 §37.3).

    With a focal prior, every replicate also draws its focal length (log-uniform).
    """
    by_track: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        by_track.setdefault(pair.track, []).append(index)
    track_ids = list(by_track)
    replicates = []
    for _ in range(samples):
        chosen = rng.choice(len(track_ids), size=len(track_ids), replace=True)
        indices = np.concatenate([by_track[track_ids[i]] for i in chosen])
        if focal_prior is None:
            fit = _solve(feet[indices], heads[indices], cx, cy, width, scale, start)
            if fit is not None and 0.5 * width <= fit.params[0] <= 3.0 * width:
                replicates.append(fit.params)
            continue
        low, high = np.log(focal_prior[0] * width), np.log(focal_prior[1] * width)
        focal = float(np.exp(rng.uniform(low, high)))
        fit = _solve(
            feet[indices],
            heads[indices],
            cx,
            cy,
            width,
            scale,
            start,
            focal=focal,
            vertical_only=True,
        )
        if fit is not None:
            replicates.append(fit.params)
    return replicates


class GroundMapper:
    """Maps pixels to the ground for the point fit and every bootstrap replicate at once."""

    def __init__(self, fit: PlaneFit, *, max_replicates: int | None = None) -> None:
        """`max_replicates` uses a prefix of the (i.i.d.) replicates for per-frame work."""
        if not fit.accepted or fit.parameters is None:
            raise ValueError("GroundMapper needs an accepted plane fit")
        self.fit = fit
        self.camera = Camera(fit.parameters)
        replicates = np.asarray(fit.bootstrap)[:max_replicates]
        self.batch = BatchCamera(replicates, fit.parameters.principal)
        self._support = prep(Polygon(fit.support_polygon)) if fit.support_polygon else None

    def ground(self, pixels: np.ndarray) -> np.ndarray:
        return self.camera.ground(pixels)

    def ground_replicates(self, pixels: np.ndarray) -> np.ndarray:
        """(R, N, 2) ground points for every replicate; pixels may be (N, 2) or (R, N, 2)."""
        return self.batch.ground(pixels)

    def supported(self, point_xy: tuple[float, float]) -> bool:
        return self._support is not None and self._support.contains(Point(point_xy))


class BatchCamera:
    """Ground back-projection for many parameter sets at once (bootstrap replicates)."""

    def __init__(self, parameters: np.ndarray, principal: tuple[float, float]) -> None:
        # parameters: (R, 4) rows of [focal, tilt, roll, height]
        self.focal = parameters[:, 0]
        self.height = parameters[:, 3]
        self.principal = principal
        tilt, roll = parameters[:, 1], parameters[:, 2]
        ct, st = np.cos(tilt), np.sin(tilt)
        zeros, ones = np.zeros_like(tilt), np.ones_like(tilt)
        pitch = np.stack(
            [
                np.stack([ones, zeros, zeros], axis=1),
                np.stack([zeros, -st, -ct], axis=1),
                np.stack([zeros, ct, -st], axis=1),
            ],
            axis=1,
        )
        cr, sr = np.cos(roll), np.sin(roll)
        rolled = np.stack(
            [
                np.stack([cr, -sr, zeros], axis=1),
                np.stack([sr, cr, zeros], axis=1),
                np.stack([zeros, zeros, ones], axis=1),
            ],
            axis=1,
        )
        self.R = rolled @ pitch  # (R, 3, 3)

    def __len__(self) -> int:
        return len(self.focal)

    def ground(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) or (R, N, 2) → ground XY (R, N, 2); above the horizon → NaN."""
        if pixels.ndim == 2:
            pixels = np.broadcast_to(pixels, (len(self), *pixels.shape))
        focal = self.focal[:, None]
        rays = np.stack(
            [
                (pixels[..., 0] - self.principal[0]) / focal,
                (pixels[..., 1] - self.principal[1]) / focal,
                np.ones(pixels.shape[:2]),
            ],
            axis=-1,
        )
        directions = np.einsum("rnk,rkj->rnj", rays, self.R)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                directions[..., 2] < -1e-9, -self.height[:, None] / directions[..., 2], np.nan
            )
        return (directions[..., :2] * t[..., None]).astype(np.float32)


def sample_vertical_pairs(
    observations_by_frame,
    *,
    minimum_height_px: float,
    minimum_confidence: float,
    minimum_aspect: float,
    spacing_s: float,
    maximum_per_track: int,
    canvas: tuple[int, int],
    border_px: float = 4.0,
) -> list[VerticalPair]:
    """Upright, unclipped, confident people, at most every `spacing_s` per track (v7 §34.8).

    Uses the tracked box top (head) and bottom-centre (feet) in reference coordinates; a
    person wider than `minimum_aspect` allows (crouching, bending, carrying) is skipped.
    """
    width, height = canvas
    last_sample: dict[str, float] = {}
    counts: dict[str, int] = {}
    pairs: list[VerticalPair] = []
    for _, observations in observations_by_frame:
        for o in observations:
            if o.object_class != "person" or (o.confidence or 0.0) < minimum_confidence:
                continue
            box = o.box_reference
            if (
                box.height() < minimum_height_px
                or box.height() / max(box.width(), 1e-6) < minimum_aspect
            ):
                continue
            if (
                box.x1 < border_px
                or box.y1 < border_px
                or box.x2 > width - border_px
                or box.y2 > height - border_px
            ):
                continue
            track = o.canonical_track_id
            if counts.get(track, 0) >= maximum_per_track:
                continue
            if o.video_time_s - last_sample.get(track, -1e9) < spacing_s:
                continue
            last_sample[track] = o.video_time_s
            counts[track] = counts.get(track, 0) + 1
            centre = (box.x1 + box.x2) / 2
            pairs.append(VerticalPair(track, (centre, box.y2), (centre, box.y1)))
    return pairs
