"""Write and validate run bundles (see `shared/schemas/bundle.py` for the layout)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from shared.enums import RuleId
from shared.schemas.bundle import (
    INCIDENTS_FILE,
    REQUIRED_FILES,
    RUN_MANIFEST_FILE,
    SCENE_FILE,
    TRACKS_SUMMARY_FILE,
    VIDEO_FILE,
    VIDEO_PROXY_FILE,
    IncidentsFile,
    TracksSummaryFile,
)
from shared.schemas.run import RunManifest
from shared.schemas.scene import SceneCache

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_model(path: Path, model: BaseModel) -> None:
    """Write a model as indented JSON atomically: readers never see a partial file."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class BundleCheck:
    run_dir: Path
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_bundle(run_dir: Path, *, require_media: bool = False) -> BundleCheck:
    """Check that a run directory satisfies the bundle contract before publishing."""
    check = BundleCheck(run_dir=run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if require_media:
        missing += [
            name for name in (VIDEO_FILE, VIDEO_PROXY_FILE) if not (run_dir / name).is_file()
        ]
    check.problems.extend(f"missing {name}" for name in missing)
    if missing:
        return check

    try:
        manifest = read_model(run_dir / RUN_MANIFEST_FILE, RunManifest)
        incidents = read_model(run_dir / INCIDENTS_FILE, IncidentsFile)
        tracks = read_model(run_dir / TRACKS_SUMMARY_FILE, TracksSummaryFile)
        read_model(run_dir / SCENE_FILE, SceneCache)
    except ValidationError as error:
        check.problems.append(f"schema violation: {error}")
        return check

    for name, run_id in (("incidents", incidents.run_id), ("tracks_summary", tracks.run_id)):
        if run_id != manifest.run_id:
            check.problems.append(f"{name} run_id {run_id} != manifest {manifest.run_id}")

    covered = [entry.rule_id for entry in manifest.rule_coverage]
    if covered != list(RuleId):
        check.problems.append(f"rule coverage must list R1–R5 in order, got {covered}")

    known_tracks = {track.canonical_track_id for track in tracks.tracks}
    for incident in incidents.incidents:
        for relative in (incident.evidence_path, incident.evidence_crop_path):
            if relative is not None and not (run_dir / relative).is_file():
                check.problems.append(f"{incident.incident_id}: missing {relative}")
        if (
            incident.canonical_track_id is not None
            and incident.canonical_track_id not in known_tracks
        ):
            check.problems.append(
                f"{incident.incident_id}: track {incident.canonical_track_id} not in summary"
            )
    return check
