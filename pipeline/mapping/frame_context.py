"""Pass two's geometry join: pass-one frame + scene state → `FrameContext` (v7 §31.8).

This is the only place that decides which evidence a frame has. Rules then read the
capabilities instead of guessing: no PPE model means R1/R2 are unsupported, no scene
state means R3/R5 are inconclusive, no accepted camera fit means R4/R5 are inconclusive.

With an accepted camera fit, every worker/machine/edge distance is computed for all
bootstrap replicates (with anchor jitter) and reported as a 95 % interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from pipeline.cache.pass1_store import FrameRecord
from pipeline.mapping.anchors import image_aligned_record
from pipeline.mapping.camera_model import GroundMapper
from pipeline.mapping.geometry import (
    OperatingStateTracker,
    assign_band,
    distance_to_polygon,
    distance_to_polyline,
    footprint_from_contact,
    summarise_many,
)
from pipeline.mapping.zones import ZoneIndex, anchor_sigma_px
from shared.config import RulesConfig
from shared.coordinates import Point2
from shared.enums import PERSON_CLASS, CoordinateSpace, DistanceBand, OperatingState
from shared.schemas.frame import EdgeProximity, FeedCapabilities, FrameContext, MachineryProximity
from shared.schemas.geometry import MachineryFootprintRecord
from shared.schemas.scene import ProposalType, SceneCache
from shared.schemas.tracks import MappedTrackRecord, Pass1TrackObservation

PLANE_ID = "plane-0"
RELATIVE_PLANE = "relative_plane"
# Project-defined WH margins around a machine footprint (v7 §38), not physical sizes.
DEFAULT_INFLATION_WH = {"machinery": 0.5, "vehicle": 0.3}
CONTACT_INSET = 0.1  # box sides are usually wider than the ground contact


class FrameContextBuilder:
    def __init__(
        self,
        *,
        run_id: str,
        scene: SceneCache,
        geometry_supported: bool,
        ppe_available: bool,
        machinery_classes: Sequence[str],
        mappers: Mapping[str, GroundMapper] | None = None,
        rules: RulesConfig | None = None,
        inflation_wh: Mapping[str, float] | None = None,
        seed: int = 42017,
    ) -> None:
        self.run_id = run_id
        self.scene = scene
        self.geometry_supported = geometry_supported
        self.ppe_available = ppe_available
        self.machinery_classes = frozenset(machinery_classes)
        self.mappers = dict(mappers or {}) if geometry_supported else {}
        self.rules = rules
        self.inflation = dict(inflation_wh or DEFAULT_INFLATION_WH)
        self.seed = seed
        self.operating = OperatingStateTracker()
        self._zones: dict[str, ZoneIndex] = {}
        self._edges: dict[str, list[tuple[str, np.ndarray, bool]]] = {}
        if self.mappers and rules is None:
            raise ValueError("distance bands need the rules config")

    def build(
        self, frame: FrameRecord, observations: Sequence[Pass1TrackObservation]
    ) -> FrameContext:
        geometry = self.geometry_supported and frame.transform_valid
        mapper = self.mappers.get(frame.shot_id) if geometry else None
        workers = [o for o in observations if o.object_class == PERSON_CLASS]
        machines = [o for o in observations if o.object_class in self.machinery_classes]
        proposals = self.scene.proposals_for(frame.shot_id)
        has_scene = frame.shot_id in self.scene.shot_proposals

        if mapper is None:
            mapped = tuple(image_aligned_record(o) for o in observations)
            machinery = tuple(_unmapped_machine(o, frame.frame_index) for o in machines)
            memberships = self._zone_index(frame.shot_id).memberships(mapped) if geometry else ()
            return FrameContext(
                run_id=self.run_id,
                shot_id=frame.shot_id,
                frame_index=frame.frame_index,
                video_time_s=frame.video_time_s,
                geometry_supported=geometry,
                capabilities=FeedCapabilities(
                    ppe_classification=self.ppe_available,
                    scene_state=has_scene,
                    relative_plane=False,
                ),
                tracks=mapped,
                machinery=machinery,
                scene_proposals=proposals,
                zone_memberships=memberships,
            )

        rng = np.random.default_rng(self.seed + frame.frame_index)
        worker_samples, worker_supported = self._worker_samples(mapper, workers, rng)
        mapped = tuple(self._mapped(mapper, o) for o in observations)
        machine_records, machine_quads = self._machines(mapper, machines, frame)
        proximities = self._proximities(
            workers, worker_samples, worker_supported, machines, machine_records, machine_quads
        )
        edges = self._edge_proximities(
            mapper, frame.shot_id, workers, worker_samples, worker_supported
        )
        return FrameContext(
            run_id=self.run_id,
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            video_time_s=frame.video_time_s,
            geometry_supported=True,
            capabilities=FeedCapabilities(
                ppe_classification=self.ppe_available,
                scene_state=has_scene,
                relative_plane=True,
            ),
            tracks=mapped,
            machinery=tuple(machine_records),
            scene_proposals=proposals,
            zone_memberships=self._zone_index(frame.shot_id).memberships(mapped),
            machinery_proximities=tuple(proximities),
            edge_proximities=tuple(edges),
        )

    # ------------------------------------------------------------------ pieces

    def _mapped(
        self, mapper: GroundMapper, observation: Pass1TrackObservation
    ) -> MappedTrackRecord:
        record = image_aligned_record(observation)
        anchor = np.array([[observation.anchor_reference.x, observation.anchor_reference.y]])
        x, y = mapper.ground(anchor)[0]
        if not (np.isfinite(x) and np.isfinite(y)):
            return record
        supported = mapper.supported((float(x), float(y)))
        return record.model_copy(
            update={
                "relative_position": Point2(
                    x=float(x), y=float(y), space=CoordinateSpace.RELATIVE_PLANE
                ),
                "plane_id": PLANE_ID if supported else None,
                "mapping_mode": RELATIVE_PLANE,
            }
        )

    def _worker_samples(
        self, mapper: GroundMapper, workers: Sequence[Pass1TrackObservation], rng
    ) -> tuple[np.ndarray, list[bool]]:
        """(R, N, 2) ground samples per worker, with foot-point jitter per replicate."""
        if not workers:
            return np.zeros((len(mapper.batch), 0, 2)), []
        anchors = np.array([[o.anchor_reference.x, o.anchor_reference.y] for o in workers])
        sigmas = np.array(
            [anchor_sigma_px(image_aligned_record(o).anchor_covariance_2x2) for o in workers]
        )
        jitter = rng.normal(size=(len(mapper.batch), len(workers), 2)) * sigmas[None, :, None]
        samples = mapper.ground_replicates(anchors[None] + jitter)
        point = mapper.ground(anchors)
        supported = [
            bool(np.all(np.isfinite(p)) and mapper.supported((float(p[0]), float(p[1]))))
            for p in point
        ]
        return samples, supported

    def _machines(self, mapper: GroundMapper, machines, frame: FrameRecord):
        records, quads = [], []
        for observation in machines:
            box = observation.box_reference
            inset = CONTACT_INSET * box.width()
            contact = np.array([[box.x1 + inset, box.y2], [box.x2 - inset, box.y2]])
            quad = footprint_from_contact(mapper.ground_replicates(contact))
            point_quad = footprint_from_contact(mapper.ground(contact)[None])[0]
            centre = mapper.ground(np.array([[(box.x1 + box.x2) / 2, box.y2]]))[0]
            supported = bool(
                np.all(np.isfinite(point_quad))
                and all(mapper.supported((float(x), float(y))) for x, y in point_quad[:2])
            )
            state = self.operating.update(
                observation.canonical_track_id,
                frame.video_time_s,
                centre if np.all(np.isfinite(centre)) else None,
                observation.motion_energy,
            )
            inflation = self.inflation.get(observation.object_class, 0.5)
            records.append(
                MachineryFootprintRecord(
                    machinery_id=observation.canonical_track_id,
                    shot_id=observation.shot_id,
                    frame_index=frame.frame_index,
                    class_name=observation.object_class,
                    polygon_plane_xy=(
                        [(float(x), float(y)) for x, y in point_quad]
                        if np.all(np.isfinite(point_quad))
                        else None
                    ),
                    compatible_plane_id=PLANE_ID if supported else None,
                    operating_state=state,
                    construction_method="bottom_contact_envelope",
                    inflation_wh=inflation,
                    covariance=[[0.0]],
                    valid=bool(np.all(np.isfinite(point_quad))),
                    reason_code="ok" if supported else "outside_plane_support",
                )
            )
            quads.append(quad)
        return records, quads

    def _proximities(self, workers, samples, supported, machines, records, quads):
        assert self.rules is not None
        results: list[MachineryProximity] = []
        if not workers:
            return results
        per_machine = []
        for record, quad in zip(records, quads, strict=True):
            distances = distance_to_polygon(samples, quad) - record.inflation_wh  # (R, N)
            per_machine.append(summarise_many(np.maximum(distances, 0.0), len(samples)))
        for w_index, worker in enumerate(workers):
            for m_index, (machine, record) in enumerate(zip(machines, records, strict=True)):
                interval = per_machine[m_index][w_index]
                compatible = supported[w_index] and record.compatible_plane_id is not None
                band = (
                    assign_band(interval, self.rules.r4_near_max_wh, self.rules.r4_caution_max_wh)
                    if interval is not None and compatible
                    else DistanceBand.INDETERMINATE
                )
                results.append(
                    MachineryProximity(
                        worker_track_id=worker.canonical_track_id,
                        machinery_id=machine.canonical_track_id,
                        compatible_plane=compatible,
                        operating_state=record.operating_state,
                        distance=interval,
                        band=band,
                        reason_code="ok" if interval is not None else "distance_unavailable",
                    )
                )
        return results

    def _edge_proximities(self, mapper, shot_id, workers, samples, supported):
        results: list[EdgeProximity] = []
        if not workers:
            return results
        for proposal_id, polyline, edge_supported in self._edge_geometry(mapper, shot_id):
            intervals = summarise_many(distance_to_polyline(samples, polyline), len(samples))
            for w_index, worker in enumerate(workers):
                interval = intervals[w_index]
                results.append(
                    EdgeProximity(
                        worker_track_id=worker.canonical_track_id,
                        proposal_id=proposal_id,
                        compatible_plane=supported[w_index] and edge_supported,
                        distance=interval,
                        reason_code="ok" if interval is not None else "distance_unavailable",
                    )
                )
        return results

    def _edge_geometry(self, mapper: GroundMapper, shot_id: str):
        cached = self._edges.get(shot_id)
        if cached is None:
            cached = []
            for proposal in self.scene.proposals_for(shot_id):
                if proposal.proposal_type is not ProposalType.OPEN_EDGE:
                    continue
                vertices = np.asarray(proposal.reference_points, np.float64)
                point = mapper.ground(vertices)
                edge_supported = bool(
                    np.all(np.isfinite(point))
                    and all(mapper.supported((float(x), float(y))) for x, y in point)
                )
                cached.append(
                    (proposal.proposal_id, mapper.ground_replicates(vertices), edge_supported)
                )
            self._edges[shot_id] = cached
        return cached

    def _zone_index(self, shot_id: str) -> ZoneIndex:
        index = self._zones.get(shot_id)
        if index is None:
            index = ZoneIndex(self.scene.proposals_for(shot_id))
            self._zones[shot_id] = index
        return index


def _unmapped_machine(
    observation: Pass1TrackObservation, frame_index: int
) -> MachineryFootprintRecord:
    """A machine that is present but has no plane footprint or operating state."""
    return MachineryFootprintRecord(
        machinery_id=observation.canonical_track_id,
        shot_id=observation.shot_id,
        frame_index=frame_index,
        class_name=observation.object_class,
        polygon_plane_xy=None,
        compatible_plane_id=None,
        operating_state=OperatingState.UNKNOWN,
        construction_method="none",
        inflation_wh=0.0,
        covariance=[[0.0]],
        valid=False,
        reason_code="relative_plane_unavailable",
    )
