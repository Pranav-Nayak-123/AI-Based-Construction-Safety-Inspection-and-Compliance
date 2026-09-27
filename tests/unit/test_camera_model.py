"""Camera calibration from people, checked against a known synthetic camera (v7 §47.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.mapping.camera_model import (
    Camera,
    CameraParameters,
    GroundMapper,
    VerticalPair,
    fit_camera,
)

CANVAS = (1920, 1080)
TRUE = CameraParameters(
    focal_px=1400.0,
    tilt_rad=math.radians(28),
    roll_rad=math.radians(1.5),
    height_wh=4.5,
    principal=(960.0, 540.0),
)


def synthetic_pairs(
    camera: Camera, *, people: int = 30, per_person: int = 12, noise_px: float = 1.0, seed: int = 3
) -> list[VerticalPair]:
    rng = np.random.default_rng(seed)
    pairs = []
    for person in range(people):
        height = rng.normal(1.0, 0.04)  # real people differ a little in height
        start = np.array([rng.uniform(-6, 6), rng.uniform(6, 22)])
        heading = rng.uniform(0, 2 * math.pi)
        for step in range(per_person):
            ground = start + 0.25 * step * np.array([math.cos(heading), math.sin(heading)])
            foot, head = camera.project(np.array([[*ground, 0.0], [*ground, height]])) + rng.normal(
                0, noise_px, (2, 2)
            )
            inside = all(0 <= p[0] < CANVAS[0] and 0 <= p[1] < CANVAS[1] for p in (foot, head))
            if inside:
                pairs.append(VerticalPair(f"t{person}", tuple(foot), tuple(head)))
    return pairs


def fit(pairs, samples: int = 60):
    return fit_camera(
        pairs,
        CANVAS,
        minimum_pairs=12,
        minimum_tracks=5,
        minimum_inlier_ratio=0.65,
        bootstrap_samples=samples,
        minimum_successful=int(0.7 * samples),
        seed=42017,
    )


def test_projection_and_ground_back_projection_invert() -> None:
    camera = Camera(TRUE)
    ground = np.array([[0.0, 10.0], [3.5, 7.0], [-4.0, 18.0]])
    pixels = camera.project(np.column_stack([ground, np.zeros(3)]))
    assert np.allclose(camera.ground(pixels), ground, atol=1e-9)


def test_points_above_the_horizon_do_not_map_to_the_ground() -> None:
    camera = Camera(TRUE)
    horizon = camera.horizon_v(960.0)
    assert np.isnan(camera.ground(np.array([[960.0, horizon - 5]]))).all()
    assert np.isfinite(camera.ground(np.array([[960.0, horizon + 5]]))).all()


def test_fit_recovers_the_camera_from_people() -> None:
    result = fit(synthetic_pairs(Camera(TRUE)))
    assert result.accepted, result.reason_code
    p = result.parameters
    assert p is not None
    assert p.tilt_rad == pytest.approx(TRUE.tilt_rad, abs=math.radians(2.0))
    assert p.roll_rad == pytest.approx(TRUE.roll_rad, abs=math.radians(1.0))
    assert p.height_wh == pytest.approx(TRUE.height_wh, rel=0.12)
    assert p.focal_px == pytest.approx(TRUE.focal_px, rel=0.15)
    assert result.inlier_ratio is not None and result.inlier_ratio > 0.9


def test_recovered_ground_distances_match_the_truth() -> None:
    truth = Camera(TRUE)
    result = fit(synthetic_pairs(truth))
    mapper = GroundMapper(result)
    a, b = np.array([[-2.0, 9.0]]), np.array([[1.0, 13.0]])  # 5 WH apart
    pixels = truth.project(np.column_stack([np.vstack([a, b]), np.zeros(2)]))
    estimated = mapper.ground(pixels)
    assert np.linalg.norm(estimated[0] - estimated[1]) == pytest.approx(5.0, rel=0.1)
    replicates = mapper.ground_replicates(pixels)
    spread = np.linalg.norm(replicates[:, 0] - replicates[:, 1], axis=1)
    low, high = np.percentile(spread, [2.5, 97.5])
    assert low <= 5.0 * 1.05 and high >= 5.0 * 0.95  # the interval covers the truth


def test_resizing_the_image_preserves_wh_distances() -> None:
    truth = Camera(TRUE)
    pairs = synthetic_pairs(truth)
    half = [
        VerticalPair(p.track, (p.foot[0] / 2, p.foot[1] / 2), (p.head[0] / 2, p.head[1] / 2))
        for p in pairs
    ]
    full_fit, half_fit = (
        fit(pairs, 20),
        fit_camera(
            half,
            (960, 540),
            minimum_pairs=12,
            minimum_tracks=5,
            minimum_inlier_ratio=0.65,
            bootstrap_samples=20,
            minimum_successful=14,
            seed=1,
        ),
    )
    assert half_fit.accepted
    points = truth.project(np.array([[-2.0, 9.0, 0.0], [1.0, 13.0, 0.0]]))
    d_full = np.linalg.norm(np.diff(GroundMapper(full_fit).ground(points), axis=0))
    d_half = np.linalg.norm(np.diff(GroundMapper(half_fit).ground(points / 2), axis=0))
    assert d_half == pytest.approx(d_full, rel=0.05)


def test_too_few_people_abstains() -> None:
    pairs = synthetic_pairs(Camera(TRUE), people=3)
    result = fit(pairs)
    assert not result.accepted
    assert result.reason_code == "too_few_upright_people"


def test_inconsistent_observations_are_rejected() -> None:
    rng = np.random.default_rng(0)
    pairs = [
        VerticalPair(
            f"t{i % 10}",
            (float(rng.uniform(0, 1920)), float(rng.uniform(300, 1080))),
            (float(rng.uniform(0, 1920)), float(rng.uniform(0, 1000))),
        )
        for i in range(120)
    ]
    assert not fit(pairs).accepted
