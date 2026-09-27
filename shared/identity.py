"""Canonical identifiers shared by pass one, the rule engine and the run bundle.

Track identity is namespaced by shot (v7 §31.6) so IDs are never continued across a
scene cut: ``shot:<shot_id>/track:<local_id>``.
"""

from __future__ import annotations

import hashlib
import math

from shared.enums import RuleId


def canonical_track_id(shot_id: str, local_id: int | str) -> str:
    return f"shot:{shot_id}/track:{local_id}"


def display_track_label(track_id: str) -> str:
    """Short human label: ``shot:s0/track:17`` → ``17``."""
    tail = track_id.rsplit("/", 1)[-1]
    return tail.split(":", 1)[-1]


def incident_id(run_id: str, track_id: str, rule_id: RuleId, first_seen_s: float) -> str:
    """Deterministic incident ID (v7 §40.1): same run, subject, rule and window → same ID.

    The window start is quantised to whole seconds so tiny timing differences between
    devices do not change the ID. Cooldown merging keeps two episodes of one subject and
    rule from starting in the same second.
    """
    window = math.floor(first_seen_s)
    digest = hashlib.sha256(f"{run_id}|{track_id}|{rule_id.value}|{window}".encode()).hexdigest()
    return f"{rule_id.value.lower()}-{digest[:16]}"
