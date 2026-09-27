"""Narrate a finished run's incidents (L4 behind the L5 guardrail).

    uv run safety-llm narrate output/<run_id>

Each incident's facts go to the fine-tuned student; its reply is shown only if all eight
guardrail checks pass, and otherwise the deterministic template is used and the failed
checks are recorded. The rule engine's decisions are never changed: narration is text
about an incident, not part of it. `incidents.json`, `run_manifest.json` and
`report.html` are rewritten in place, atomically.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from llm.guardrail import check
from llm.narration import NarrationFacts, chat_messages, facts_from_incident, template_narration
from pipeline.output.bundle import read_model, write_model
from pipeline.report.html import write_report
from shared.schemas.bundle import (
    INCIDENTS_FILE,
    REPORT_FILE,
    RUN_MANIFEST_FILE,
    SCENE_FILE,
    TRACKS_SUMMARY_FILE,
    IncidentsFile,
    TracksSummaryFile,
)
from shared.schemas.incidents import IncidentNarration
from shared.schemas.run import RunManifest
from shared.schemas.scene import SceneCache

DEFAULT_BASE = Path("artifacts/models/llm/Qwen2.5-1.5B-Instruct")
DEFAULT_ADAPTER = Path("artifacts/adapters/narration-qwen2.5-1.5b")
PROMPT_VERSION = "narration-v1"


@dataclass
class Narrator:
    """Qwen2.5-1.5B-Instruct plus the narration LoRA, greedy decoding."""

    base: Path = DEFAULT_BASE
    adapter: Path = DEFAULT_ADAPTER
    batch_size: int = 8
    max_new_tokens: int = 160

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from pipeline.vision.detector import resolve_device

        self._torch = torch
        self.device = resolve_device("auto")
        # fp16 where there is an accelerator: inference is memory-bound, unlike training.
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter)
        self.tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(self.base, dtype=dtype).to(self.device)
        self.model = PeftModel.from_pretrained(model, self.adapter).eval()
        self.model_id = f"{self.base.name}+{self.adapter.name}"

    def generate(self, facts: Sequence[NarrationFacts]) -> list[str]:
        torch, tokenizer = self._torch, self.tokenizer
        pad = tokenizer.pad_token_id or tokenizer.eos_token_id
        replies: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(facts), self.batch_size):
                prompts = [
                    tokenizer.apply_chat_template(
                        chat_messages(f), add_generation_prompt=True, tokenize=False
                    )
                    for f in facts[start : start + self.batch_size]
                ]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                output = self.model.generate(
                    **encoded, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=pad
                )
                replies += tokenizer.batch_decode(
                    output[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                )
        return replies


def narrate(
    facts: Sequence[NarrationFacts], raw_outputs: Sequence[str], model_id: str | None
) -> list[IncidentNarration]:
    """Guardrail every reply; the template stands in for any reply that fails a check."""
    narrations = []
    for fact, raw in zip(facts, raw_outputs, strict=True):
        result = check(raw, fact)
        if result.passed and result.narration is not None:
            text, source, failed = result.narration, "model", ()
        else:
            text = template_narration(fact)
            source = "template"
            failed = tuple(sorted({f.split(":")[0] for f in result.failures}))
        narrations.append(
            IncidentNarration(
                summary=text.summary,
                caveat=text.caveat,
                action=text.action,
                source=source,
                model=model_id,
                failed_checks=failed,
            )
        )
    return narrations


def narrate_run(run_dir: Path, narrator: Narrator | None = None) -> dict[str, object]:
    manifest = read_model(run_dir / RUN_MANIFEST_FILE, RunManifest)
    bundle = read_model(run_dir / INCIDENTS_FILE, IncidentsFile)
    scene = read_model(run_dir / SCENE_FILE, SceneCache)
    labels = {
        p.proposal_id: p.label
        for proposals in scene.shot_proposals.values()
        for p in proposals
        if p.label
    }
    facts = [facts_from_incident(incident, labels) for incident in bundle.incidents]
    started = time.perf_counter()
    if narrator is None and facts:
        narrator = Narrator()
    raw = narrator.generate(facts) if facts else []
    narrations = narrate(facts, raw, narrator.model_id if narrator else None)
    seconds = time.perf_counter() - started

    incidents = tuple(
        incident.model_copy(update={"narration": narration})
        for incident, narration in zip(bundle.incidents, narrations, strict=True)
    )
    failures = Counter(check for n in narrations for check in n.failed_checks)
    summary: dict[str, object] = {
        "incidents": len(narrations),
        "model": sum(n.source == "model" for n in narrations),
        "template_fallback": sum(n.source == "template" for n in narrations),
        "failed_checks": dict(failures.most_common()),
        "seconds": round(seconds, 1),
    }
    model_ids = dict(manifest.model_ids)
    prompts = dict(manifest.prompt_versions)
    if narrator is not None:
        model_ids["narrator"] = narrator.model_id
        prompts["narration"] = PROMPT_VERSION
    manifest = manifest.model_copy(
        update={
            "model_ids": model_ids,
            "prompt_versions": prompts,
            "diagnostics": {**manifest.diagnostics, "narration": summary},
        }
    )
    write_model(run_dir / INCIDENTS_FILE, bundle.model_copy(update={"incidents": incidents}))
    write_model(run_dir / RUN_MANIFEST_FILE, manifest)
    tracks = read_model(run_dir / TRACKS_SUMMARY_FILE, TracksSummaryFile)
    write_report(run_dir / REPORT_FILE, manifest, incidents, tracks=tracks)
    return summary
