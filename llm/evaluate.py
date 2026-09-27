"""Compare narration arms: template, base model (4 worked examples), LoRA student (FR-8).

    uv run python -m llm.evaluate --adapter artifacts/adapters/narration-qwen2.5-1.5b

Every output goes through the L5 guardrail exactly as in production. Reported per arm and
split: guardrail pass rate (what a user would actually see from the model), schema-valid
rate, unsupported-claim rate (numbers or names not in the facts), failures by check,
ROUGE-L against the reference, and generation speed. The promotion gate (v7 §45.6) asks
the student for 100 % schema validity and at most 2 % unsupported claims.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.corpus import load_split
from llm.guardrail import check
from llm.narration import NarrationFacts, chat_messages, template_narration

FEW_SHOT = 4
PROMOTION = {"schema_valid": 1.0, "unsupported_claims": 0.02}


def rouge_l(candidate: str, reference: str) -> float:
    a, b = re.findall(r"\w+", candidate.lower()), re.findall(r"\w+", reference.lower())
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def few_shot_messages(facts: NarrationFacts, examples) -> list[dict[str, str]]:
    messages = chat_messages(examples[0][0])[:1]  # system prompt
    for example_facts, reply in examples:
        messages += [
            {"role": "user", "content": example_facts.to_prompt()},
            {"role": "assistant", "content": reply.model_dump_json()},
        ]
    messages.append({"role": "user", "content": facts.to_prompt()})
    return messages


@torch.inference_mode()
def generate(model, tokenizer, conversations, batch_size: int = 12, max_new_tokens: int = 160):
    tokenizer.padding_side = "left"
    outputs, tokens, started = [], 0, time.perf_counter()
    for start in range(0, len(conversations), batch_size):
        prompts = [
            tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
            for c in conversations[start : start + batch_size]
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        replies = generated[:, encoded["input_ids"].shape[1] :]
        tokens += int((replies != (tokenizer.pad_token_id or tokenizer.eos_token_id)).sum())
        outputs += tokenizer.batch_decode(replies, skip_special_tokens=True)
    elapsed = time.perf_counter() - started
    return outputs, {"seconds": round(elapsed, 1), "tokens_per_second": round(tokens / elapsed, 1)}


def score(raw_outputs: list[str], rows) -> dict:
    failures: Counter[str] = Counter()
    passed = schema_ok = unsupported = 0
    rouge = []
    for raw, (facts, reference) in zip(raw_outputs, rows, strict=True):
        result = check(raw, facts)
        checks = {f.split(":")[0] for f in result.failures}
        failures.update(checks)
        passed += result.passed
        schema_ok += "shape" not in checks
        unsupported += bool(checks & {"numbers", "names"})
        if "shape" not in checks:
            parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            rouge.append(
                rouge_l(
                    " ".join(str(parsed[k]) for k in ("summary", "caveat", "action")),
                    " ".join(reference.model_dump().values()),
                )
            )
    n = len(rows)
    return {
        "n": n,
        "guardrail_pass": round(passed / n, 4),
        "schema_valid": round(schema_ok / n, 4),
        "unsupported_claims": round(unsupported / n, 4),
        "failures_by_check": dict(failures.most_common()),
        "rouge_l": round(sum(rouge) / len(rouge), 4) if rouge else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/models/llm/Qwen2.5-1.5B-Instruct")
    )
    parser.add_argument(
        "--adapter", type=Path, default=Path("artifacts/adapters/narration-qwen2.5-1.5b")
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/working/narration"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results/narration_eval.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16).to("cuda:0").eval()
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    examples = load_split(args.corpus, "train")[:FEW_SHOT]
    report: dict = {"arms": {}, "samples": {}, "promotion_gate": PROMOTION}

    for split in ("test", "holdout"):
        rows = load_split(args.corpus, split)
        facts = [f for f, _ in rows]
        template = [template_narration(f).model_dump_json() for f in facts]
        with model.disable_adapter():
            base_out, base_speed = generate(
                model, tokenizer, [few_shot_messages(f, examples) for f in facts]
            )
        lora_out, lora_speed = generate(model, tokenizer, [chat_messages(f) for f in facts])
        for arm, outputs, speed in (
            ("template", template, None),
            ("base_4shot", base_out, base_speed),
            ("lora", lora_out, lora_speed),
        ):
            report["arms"].setdefault(arm, {})[split] = {**score(outputs, rows), "speed": speed}
            report["samples"].setdefault(arm, {})[split] = [
                {"facts": f.model_dump(exclude_none=True), "output": o}
                for f, o in list(zip(facts, outputs, strict=True))[:4]
            ]
        print(
            json.dumps({arm: report["arms"][arm][split] for arm in report["arms"]}, indent=1),
            flush=True,
        )

    student = report["arms"]["lora"]
    report["promoted"] = all(
        student[s]["schema_valid"] >= PROMOTION["schema_valid"]
        and student[s]["unsupported_claims"] <= PROMOTION["unsupported_claims"]
        for s in ("test", "holdout")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("promoted:", report["promoted"])


if __name__ == "__main__":
    main()
