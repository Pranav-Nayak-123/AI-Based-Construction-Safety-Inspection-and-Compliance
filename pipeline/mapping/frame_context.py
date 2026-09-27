"""Pass two's geometry join: pass-one frame + scene state → `FrameContext` (v7 §31.8).

This is the only place that decides which evidence a frame has. Rules then read the
capabilities instead of guessing: no PPE model means R1/R2 are unsupported, no scene
state means R3/R5 are inconclusive, no relative plane means R4/R5 distances are.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.cache.pass1_store import FrameRecord
from pipeline.mapping.anchors import image_aligned_record
from pipeline.mapping.zones import ZoneIndex
from shared.enums import OperatingState
from shared.schemas.frame import FeedCapabilities, FrameContext
from shared.schemas.geometry import MachineryFootprintRecord
from shared.schemas.scene import SceneCache
from shared.schemas.tracks import MappedTrackRecord, Pass1TrackObservation


class FrameContextBuilder:
    def __init__(
        self,
        *,
        run_id: str,
        scene: SceneCache,
        geometry_supported: bool,
        ppe_available: bool,
        machinery_classes: Sequence[str],
    ) -> None:
        self.run_id = run_id
        self.scene = scene
        self.geometry_supported = geometry_supported
        self.ppe_available = ppe_available
        self.machinery_classes = frozenset(machinery_classes)
        self._zones: dict[str, ZoneIndex] = {}

    def build(
        self, frame: FrameRecord, observations: Sequence[Pass1TrackObservation]
    ) -> FrameContext:
        geometry = self.geometry_supported and frame.transform_valid
        mapped = tuple(image_aligned_record(observation) for observation in observations)
        machinery = tuple(
            _unmapped_machine(track, frame.frame_index)
            for track in mapped
            if track.object_class in self.machinery_classes
        )
        proposals = self.scene.proposals_for(frame.shot_id)
        has_scene = frame.shot_id in self.scene.shot_proposals
        memberships = self._zone_index(frame.shot_id).memberships(mapped) if geometry else ()
        return FrameContext(
            run_id=self.run_id,
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            video_time_s=frame.video_time_s,
            geometry_supported=geometry,
            capabilities=FeedCapabilities(
                person_detection=True,
                ppe_classification=self.ppe_available,
                scene_state=has_scene,
                relative_plane=False,
            ),
            tracks=mapped,
            machinery=machinery,
            scene_proposals=proposals,
            zone_memberships=memberships,
        )

    def _zone_index(self, shot_id: str) -> ZoneIndex:
        index = self._zones.get(shot_id)
        if index is None:
            index = ZoneIndex(self.scene.proposals_for(shot_id))
            self._zones[shot_id] = index
        return index


def _unmapped_machine(track: MappedTrackRecord, frame_index: int) -> MachineryFootprintRecord:
    """A machine that is present but has no plane footprint or operating state yet."""
    return MachineryFootprintRecord(
        machinery_id=track.canonical_track_id,
        shot_id=track.shot_id,
        frame_index=frame_index,
        class_name=track.object_class,
        polygon_plane_xy=None,
        compatible_plane_id=None,
        operating_state=OperatingState.UNKNOWN,
        construction_method="none",
        inflation_wh=0.0,
        covariance=[[0.0]],
        valid=False,
        reason_code="relative_plane_unavailable",
    )
