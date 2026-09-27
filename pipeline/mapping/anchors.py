"""Worker ground anchors and image-aligned track mapping (v7 §10.1, §10.4 fallback)."""

from __future__ import annotations

from shared.schemas.tracks import MappedTrackRecord, Pass1TrackObservation

IMAGE_ALIGNED = "image_aligned"


def observation_key(observation: Pass1TrackObservation) -> str:
    return f"{observation.shot_id}:{observation.frame_index}:{observation.track_id}"


def image_aligned_record(
    observation: Pass1TrackObservation,
    *,
    anchor_sigma_px: float = 2.0,
) -> MappedTrackRecord:
    """Map a pass-one observation without a relative plane.

    The anchor stays in reference pixels and no relative position is claimed, so the
    distance rules (R4/R5) that need the plane report `inconclusive` for this track.
    """
    variance = anchor_sigma_px**2
    return MappedTrackRecord(
        shot_id=observation.shot_id,
        frame_index=observation.frame_index,
        video_time_s=observation.video_time_s,
        canonical_track_id=observation.canonical_track_id,
        object_class=observation.object_class,
        reference_observation_key=observation_key(observation),
        anchor_reference=observation.anchor_reference,
        anchor_covariance_2x2=[[variance, 0.0], [0.0, variance]],
        relative_position=None,
        plane_id=None,
        helmet=observation.helmet,
        vest=observation.vest,
        operating_state=None,
        mapping_mode=IMAGE_ALIGNED,
    )
