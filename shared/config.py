from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from shared.coordinates import StrictModel


class IntakeConfig(StrictModel):
    target_fps: float
    target_width: int
    target_height: int
    portrait_policy: str
    native_drift_limit_diagonal: float
    provisional_residual_limit_diagonal: float
    geometry_residual_limit_wh: float
    interpolation_max_frames: int


class Stage1Config(StrictModel):
    model_path: str
    pretrained_fallback: str
    class_map: str
    machinery_classes: tuple[str, ...]
    image_size: int
    confidence: float
    iou: float
    maximum_detections: int
    batch_size: int
    device: str


class TrackerConfig(StrictModel):
    type: str
    track_high_threshold: float
    track_low_threshold: float
    new_track_threshold: float
    match_threshold: float
    buffer_seconds: float


class PPEConfig(StrictModel):
    model_path: str
    image_size: int
    batch_size: int
    confidence_floor: float
    minimum_person_height_px: int
    temporal_alpha: float
    minimum_observations: int


class PoseConfig(StrictModel):
    model_path: str
    image_size: int
    confidence_floor: float
    sample_spacing_seconds: float
    maximum_samples_per_track: int


class RelativePlaneConfig(StrictModel):
    minimum_pairs: int
    minimum_distinct_tracks: int
    minimum_pixel_height: int
    minimum_keypoint_confidence: float
    minimum_standing_probability: float
    maximum_occlusion_fraction: float
    minimum_vertical_inlier_ratio: float
    maximum_reprojection_rms_px: float
    maximum_temporal_horizon_shift_px: float


class UncertaintyConfig(StrictModel):
    bootstrap_samples: int
    confidence_level: float
    minimum_successful_samples: int
    temporal_block_seconds: float
    seed: int


class SceneConfig(StrictModel):
    gemini_model: str
    prompt_version: str
    minimum_keyframes: int
    maximum_keyframes: int
    minimum_support_frames: int
    minimum_support_fraction: float
    zone_cluster_iou: float
    edge_cluster_diagonal_distance: float


class RulesConfig(StrictModel):
    ppe_confidence_floor: float
    r1_debounce_seconds: float
    r2_debounce_seconds: float
    r3_debounce_seconds: float
    r4_debounce_seconds: float
    r5_debounce_seconds: float
    resolution_hysteresis_seconds: float
    cooldown_seconds: float
    missing_observation_grace_seconds: float
    r4_near_max_wh: float
    r4_caution_max_wh: float
    r5_edge_max_wh: float
    adjudication_margin: float


class RenderConfig(StrictModel):
    output_width: int
    output_height: int
    output_fps: float
    pane_width: int
    pane_height: int
    crf: int
    preset: str
    video_codec: str
    proxy_height: int


class RuntimeConfig(StrictModel):
    processing_checkpoint_frames: int
    progress_events_per_second: int
    minimum_free_disk_gib: int
    maximum_peak_memory_gib: int


class AppConfig(StrictModel):
    schema_version: int = Field(ge=1)
    intake: IntakeConfig
    stage1: Stage1Config
    tracker: TrackerConfig
    ppe: PPEConfig
    pose: PoseConfig
    relative_plane: RelativePlaneConfig
    uncertainty: UncertaintyConfig
    scene: SceneConfig
    rules: RulesConfig
    render: RenderConfig
    runtime: RuntimeConfig


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "defaults.yaml"


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return AppConfig.model_validate(payload)
