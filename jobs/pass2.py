"""Pass two: pass-one store + scene → rule outputs, incidents and track summaries.

Cheap and repeatable (v7 §6): changing rule thresholds or zones re-runs this without
touching detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.cache.pass1_store import Pass1Store
from pipeline.mapping.frame_context import FrameContextBuilder
from pipeline.output.summaries import TrackSummaryBuilder
from pipeline.rules.engine import EngineResult, RuleEngine
from shared.enums import OperatingState
from shared.schemas.bundle import TracksSummaryFile
from shared.schemas.frame import FrameRuleOutput


@dataclass(frozen=True)
class Pass2Result:
    rules: EngineResult
    tracks_summary: TracksSummaryFile
    frame_outputs: dict[int, FrameRuleOutput]
    end_time_s: float
    machine_states: dict[int, dict[str, OperatingState]] = field(default_factory=dict)


def run_pass2(
    store: Pass1Store,
    builder: FrameContextBuilder,
    engine: RuleEngine,
    summary: TrackSummaryBuilder,
) -> Pass2Result:
    outputs: dict[int, FrameRuleOutput] = {}
    machine_states: dict[int, dict[str, OperatingState]] = {}
    end_time_s = 0.0
    for frame, observations in store.frames_with_observations():
        context = builder.build(frame, observations)
        outputs[frame.frame_index] = engine.evaluate_frame(context)
        if context.machinery:
            machine_states[frame.frame_index] = {
                m.machinery_id: m.operating_state for m in context.machinery
            }
        summary.add_frame(context)
        end_time_s = frame.video_time_s
    rules = engine.finalize(end_time_s)
    return Pass2Result(
        rules=rules,
        tracks_summary=summary.build(rules.incidents),
        frame_outputs=outputs,
        end_time_s=end_time_s,
        machine_states=machine_states,
    )
