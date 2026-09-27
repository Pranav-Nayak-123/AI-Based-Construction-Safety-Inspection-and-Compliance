"""Load, validate and hash the deterministic regulation catalogue (v7 §41)."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from string import Formatter
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from shared.coordinates import StrictModel
from shared.enums import RuleId

TEMPLATE_VARIABLES = frozenset({"track_id", "duration_s", "zone_label", "machine_label"})


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name or format_spec or conversion:
            raise ValueError(f"template placeholders must be bare names: {template!r}")
        fields.add(field_name)
    return fields


class CatalogueRule(StrictModel):
    rule_id: RuleId
    title: str = Field(min_length=1)
    basis: Literal["visual", "heuristic"]
    severity: Severity
    alert_reason_code: str = Field(min_length=1)
    observation_template: str = Field(min_length=1)
    action_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    action_template: str = Field(min_length=1)
    references: tuple[str, ...] = ()
    contextual_references: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _templates_use_known_variables(self) -> CatalogueRule:
        for name in ("observation_template", "action_template"):
            unknown = template_fields(getattr(self, name)) - TEMPLATE_VARIABLES
            if unknown:
                raise ValueError(
                    f"{self.rule_id.value} {name} uses unsupported variables {sorted(unknown)}"
                )
        return self

    def render_observation(self, values: dict[str, str]) -> str:
        return self.observation_template.format_map(values)

    def render_action(self, values: dict[str, str]) -> str:
        return self.action_template.format_map(values)


class RegulationCatalogue(StrictModel):
    catalogue_version: int = Field(ge=1)
    reference_status: Literal["unverified", "verified"]
    rules: tuple[CatalogueRule, ...]

    @model_validator(mode="after")
    def _one_entry_per_rule(self) -> RegulationCatalogue:
        ids = [rule.rule_id for rule in self.rules]
        duplicates = sorted({rule_id.value for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate catalogue entries: {duplicates}")
        missing = [rule_id.value for rule_id in RuleId if rule_id not in ids]
        if missing:
            raise ValueError(f"catalogue missing rules: {missing}")
        return self


class CompiledCatalogue:
    """A validated catalogue plus the SHA-256 of its canonical JSON form."""

    def __init__(self, catalogue: RegulationCatalogue) -> None:
        self.catalogue = catalogue
        self.canonical_json = json.dumps(
            catalogue.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.sha256 = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        self._by_id = {rule.rule_id: rule for rule in catalogue.rules}

    def rule(self, rule_id: RuleId) -> CatalogueRule:
        return self._by_id[rule_id]


def default_catalogue_path() -> Path:
    return Path(__file__).resolve().parents[2] / "regulations" / "catalogue.v1.yaml"


def load_catalogue(path: Path | None = None) -> CompiledCatalogue:
    catalogue_path = path or default_catalogue_path()
    with catalogue_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle)
    return CompiledCatalogue(RegulationCatalogue.model_validate(payload))
