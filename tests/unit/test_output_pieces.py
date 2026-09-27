"""Prefetching, cache paths, compositor/plan rendering and the HTML report."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from rule_frames import frame, helmet, rules_config, times, worker, zone

from pipeline.intake.prefetch import Prefetch
from pipeline.render.compositor import Compositor, card_headline
from pipeline.render.encoder import _codec_arguments, resolve_codec
from pipeline.render.plan_view import PlanView
from pipeline.report.html import render_report
from pipeline.rules.engine import RuleEngine
from shared.config import load_config
from shared.coordinates import Box2, Point2
from shared.enums import CoordinateSpace, HelmetState, ProcessingStatus
from shared.identity import canonical_track_id
from shared.paths import CACHE_ENV, cache_root, run_work_dir
from shared.schemas.run import InputVideoMeta, RunManifest
from shared.schemas.tracks import Pass1TrackObservation

# ------------------------------------------------------------------ prefetch


def test_prefetch_preserves_order() -> None:
    assert list(Prefetch(range(100), depth=4)) == list(range(100))


def test_prefetch_reraises_producer_errors() -> None:
    def broken() -> Iterator[int]:
        yield 1
        raise RuntimeError("decoder exploded")

    items = []
    with pytest.raises(RuntimeError, match="decoder exploded"):
        for item in Prefetch(broken()):
            items.append(item)
    assert items == [1]


def test_prefetch_stops_when_consumer_breaks_early() -> None:
    prefetch = Prefetch(iter(range(10_000)), depth=2)
    for item in prefetch:
        if item == 3:
            break
    assert not prefetch._thread.is_alive()


# ------------------------------------------------------------------ paths and codecs


def test_cache_root_honours_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CACHE_ENV, str(tmp_path))
    assert cache_root() == tmp_path
    assert run_work_dir("run-1") == tmp_path / "runs" / "run-1"


def test_explicit_codec_passes_through_and_unknown_fails() -> None:
    assert resolve_codec("libx264") == "libx264"
    assert "-crf" in _codec_arguments("libx264", quality=18, preset="veryfast")
    with pytest.raises(ValueError, match="unsupported"):
        _codec_arguments("mpeg1", quality=18, preset="veryfast")


# ------------------------------------------------------------------ compositor


def _incidents():
    engine = RuleEngine(run_id="run-test", config=rules_config())
    no_helmet = helmet(HelmetState.NO_HELMET, 0.9)
    for t in times(0.0, 6.0):
        record = no_helmet if 1.0 <= t < 3.0 else helmet(HelmetState.HELMET)
        engine.evaluate_frame(frame(t, tracks=(worker(7, helmet_record=record),)))
    return engine.finalize()


def test_cards_appear_at_confirmation_and_linger_briefly() -> None:
    result = _incidents()
    [incident] = result.incidents
    compositor = Compositor(
        load_config().render,
        title="test",
        duration_s=6.0,
        incidents=result.incidents,
        coverage=result.coverage,
    )
    assert compositor.visible_incidents(incident.confirmed_at_s - 0.1) == []  # not before debounce
    assert compositor.visible_incidents(incident.confirmed_at_s) == [incident]
    assert incident.resolved_at_s is not None
    assert compositor.visible_incidents(incident.resolved_at_s + 1.0) == [incident]
    assert compositor.visible_incidents(incident.resolved_at_s + 3.0) == []

    config = load_config().render
    pane = np.zeros((config.pane_height, config.pane_width, 3), np.uint8)
    crop = np.full((120, 60, 3), 200, np.uint8)
    image = compositor.compose(
        pane, pane, time_s=incident.confirmed_at_s, crops={incident.incident_id: crop}
    )
    assert image.shape == (config.output_height, config.output_width, 3)


def test_plan_view_renders_and_resets_trails() -> None:
    view = PlanView((940, 529), (1920, 1080), 15.0)

    def observation(x: float) -> Pass1TrackObservation:
        return Pass1TrackObservation(
            shot_id="s0",
            frame_index=0,
            video_time_s=0.0,
            track_id=1,
            canonical_track_id=canonical_track_id("s0", 1),
            object_class="person",
            box_reference=Box2(
                x1=x - 20, y1=500, x2=x + 20, y2=620, space=CoordinateSpace.REFERENCE
            ),
            anchor_reference=Point2(x=x, y=620, space=CoordinateSpace.REFERENCE),
        )

    for step in range(5):
        image = view.render(
            observations=[observation(500 + step * 10)], output=None, proposals=[zone("zone-1")]
        )
    assert image.shape == (529, 940, 3)
    assert len(view._trails[canonical_track_id("s0", 1)]) == 5
    view.reset()
    assert view._trails == {}


# ------------------------------------------------------------------ report


def test_report_escapes_every_text_field() -> None:
    result = _incidents()
    manifest = RunManifest(
        run_id="run-<b>",
        status=ProcessingStatus.COMPLETED,
        created_at="2026-09-27T00:00:00Z",
        input_video=InputVideoMeta(path="clip<script>.mp4", sha256="x", width=10, height=10),
        rule_coverage=result.coverage,
        warnings=("<script>alert(1)</script>",),
    )
    html = render_report(manifest, result.incidents)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "Worker 7 was repeatedly observed" in html


def test_cards_never_state_the_future() -> None:
    [incident] = _incidents().incidents  # no helmet from 1.0 s to 3.0 s
    assert incident.title == "Apparent missing helmet"
    title, progress = card_headline(incident, incident.confirmed_at_s)
    assert title == "Apparent missing helmet"
    assert progress == f"Worker 7 - {incident.confirmed_at_s - 1.0:.1f} s so far"
    assert "2.0" not in progress  # the final duration is not known yet
    assert card_headline(incident, 5.0)[1] == "Worker 7 - resolved after 2.0 s"
