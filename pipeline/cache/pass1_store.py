"""SQLite store for pass-one output (v7 §32): one row per processed frame and per track.

Pass two, the renderer and the bundle writer all read from here, so rule or render
changes never re-run detection. Writes use WAL and are committed in batches; a store is
only trusted once `complete` is set in its metadata.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from pydantic import Field

from shared.coordinates import StrictModel
from shared.schemas.tracks import Pass1TrackObservation

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
  frame_index INTEGER PRIMARY KEY,
  source_index INTEGER NOT NULL,
  video_time_s REAL NOT NULL,
  shot_id TEXT NOT NULL,
  record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracks (
  shot_id TEXT NOT NULL,
  frame_index INTEGER NOT NULL,
  video_time_s REAL NOT NULL,
  track_id INTEGER NOT NULL,
  canonical_track_id TEXT NOT NULL,
  class_name TEXT NOT NULL,
  record_json TEXT NOT NULL,
  PRIMARY KEY (shot_id, frame_index, track_id)
);
CREATE INDEX IF NOT EXISTS tracks_by_identity ON tracks(canonical_track_id, frame_index);
CREATE INDEX IF NOT EXISTS tracks_by_frame ON tracks(frame_index);
"""


class FrameRecord(StrictModel):
    frame_index: int = Field(ge=0)
    source_index: int = Field(ge=0)
    video_time_s: float = Field(ge=0.0)
    shot_id: str
    transform_valid: bool
    detection_count: int = Field(ge=0)


class Pass1Store:
    def __init__(self, path: Path, *, commit_every: int = 250) -> None:
        self.path = path
        self.commit_every = commit_every
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._pending_frames = 0

    # ------------------------------------------------------------------ writing

    def set_meta(self, key: str, value: Any) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )

    def add_frame(self, frame: FrameRecord, observations: Sequence[Pass1TrackObservation]) -> None:
        self._connection.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?)",
            (
                frame.frame_index,
                frame.source_index,
                frame.video_time_s,
                frame.shot_id,
                frame.model_dump_json(),
            ),
        )
        self._connection.executemany(
            "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    observation.shot_id,
                    observation.frame_index,
                    observation.video_time_s,
                    observation.track_id,
                    observation.canonical_track_id,
                    observation.object_class,
                    observation.model_dump_json(),
                )
                for observation in observations
            ],
        )
        self._pending_frames += 1
        if self._pending_frames >= self.commit_every:
            self.commit()

    def commit(self) -> None:
        self._connection.commit()
        self._pending_frames = 0

    def mark_complete(self) -> None:
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.set_meta("complete", True)
        self.commit()

    # ------------------------------------------------------------------ reading

    def meta(self, key: str, default: Any = None) -> Any:
        row = self._connection.execute(
            "SELECT value_json FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else default

    @property
    def complete(self) -> bool:
        return bool(self.meta("complete", False))

    def frame_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])

    def frames(self) -> Iterator[FrameRecord]:
        for (record_json,) in self._connection.execute(
            "SELECT record_json FROM frames ORDER BY frame_index"
        ):
            yield FrameRecord.model_validate_json(record_json)

    def frames_with_observations(
        self,
    ) -> Iterator[tuple[FrameRecord, list[Pass1TrackObservation]]]:
        """Every frame in order with its observations, including frames with none."""
        rows = self._connection.execute(
            "SELECT frame_index, record_json FROM tracks ORDER BY frame_index, track_id"
        )
        pending = next(rows, None)
        for frame in self.frames():
            observations: list[Pass1TrackObservation] = []
            while pending is not None and pending[0] == frame.frame_index:
                observations.append(Pass1TrackObservation.model_validate_json(pending[1]))
                pending = next(rows, None)
            yield frame, observations

    def observations_for(self, canonical_track_id: str) -> list[Pass1TrackObservation]:
        return [
            Pass1TrackObservation.model_validate_json(record_json)
            for (record_json,) in self._connection.execute(
                "SELECT record_json FROM tracks WHERE canonical_track_id = ? ORDER BY frame_index",
                (canonical_track_id,),
            )
        ]

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._connection.commit()
        self._connection.close()

    def __enter__(self) -> Pass1Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
