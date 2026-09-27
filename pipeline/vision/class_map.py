"""Map a detector's native class names onto the stage-1 taxonomy (v7 §34.1)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator

from shared.coordinates import StrictModel
from shared.errors import PipelineError


class DetectorClassMap(StrictModel):
    ignore_unlisted: bool = False
    classes: dict[str, str | None]


class ClassMapFile(StrictModel):
    schema_version: int
    taxonomy: tuple[str, ...]
    maps: dict[str, DetectorClassMap]

    @model_validator(mode="after")
    def _targets_in_taxonomy(self) -> ClassMapFile:
        for name, detector_map in self.maps.items():
            unknown = {
                target
                for target in detector_map.classes.values()
                if target is not None and target not in self.taxonomy
            }
            if unknown:
                raise ValueError(f"map {name} targets classes outside the taxonomy: {unknown}")
        return self


def default_class_map_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "class_map.yaml"


def load_class_map_file(path: Path | None = None) -> ClassMapFile:
    with (path or default_class_map_path()).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return ClassMapFile.model_validate(payload)


def resolve_model_classes(
    native_names: Mapping[int, str], map_name: str, path: Path | None = None
) -> dict[int, str]:
    """Model class index → taxonomy class, for the classes this project keeps."""
    mapping_file = load_class_map_file(path)
    if map_name not in mapping_file.maps:
        raise PipelineError("CLASS_MAP_UNKNOWN", f"no class map named {map_name!r}")
    detector_map = mapping_file.maps[map_name]
    resolved: dict[int, str] = {}
    unlisted: list[str] = []
    for index, native in native_names.items():
        if native in detector_map.classes:
            target = detector_map.classes[native]
            if target is not None:
                resolved[index] = target
        elif not detector_map.ignore_unlisted:
            unlisted.append(native)
    if unlisted:
        raise PipelineError(
            "CLASS_MAP_INCOMPLETE", f"{map_name} does not map model classes {sorted(unlisted)}"
        )
    if not resolved:
        raise PipelineError("CLASS_MAP_EMPTY", f"{map_name} keeps none of the model's classes")
    return resolved
