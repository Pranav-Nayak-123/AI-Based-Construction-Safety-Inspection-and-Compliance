"""LoRA fine-tune of the narration student (requirements §4, v7 §45.4).

    uv run python -m llm.train_lora --corpus data/working/narration \
        --out artifacts/adapters/narration-qwen2.5-1.5b

Adapter, as specified in the requirements document: rank 8, alpha 16, on the query and
value projections of the top 8 of Qwen2.5-1.5B-Instruct's 28 layers — 311,296 trainable
parameters (0.0202 % of the model).

Why two stages. On the development GPU (GTX 1660 Ti, no tensor cores) fp16 matrix
multiplies run ~7x slower than fp32, and the whole model in fp32 (6.2 GB) does not fit in
6 GB. Because only the top 8 layers carry adapters, the bottom 20 are a frozen feature
extractor. So:

  A. run embeddings + layers 0-19 once, in fp32, and cache their output per example;
  B. train LoRA on a model made of layers 20-27 + norm + head, fed the cached features.

This is exactly equivalent to fine-tuning the full model with the bottom frozen (Qwen has
no dropout there), and everything runs in fast fp32. The saved adapter is renamed back to
layers 20-27 so it loads onto the full model unchanged.

Stage B keeps only each layer's input and recomputes the rest in the backward pass
(gradient checkpointing): the full activations of 8 fp32 layers at batch 8 are ~6 GB and
would page to shared memory, which on Windows is 5-10x slower than the recompute. Batches
are bucketed by length so little compute goes to padding, and the Stage A cache is kept
on local disk (not OneDrive) so a rerun skips it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.corpus import load_split
from llm.narration import Narration, NarrationFacts, chat_messages

SEED = 42017
EXPECTED_TRAINABLE = 311_296
TOP_LAYERS = 8
FULL_MODEL_PARAMETERS = 1_544_025_600


def encode(tokenizer, facts: NarrationFacts, reply: Narration, max_length: int):
    prompt = tokenizer.apply_chat_template(
        chat_messages(facts), add_generation_prompt=True, tokenize=False
    )
    answer = reply.model_dump_json() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return ids, labels


@torch.inference_mode()
def cache_features(model, examples, device: str) -> list[torch.Tensor]:
    """Stage A: the top block's input for every example, computed once in fp32."""
    features = []
    for ids, _ in examples:
        hidden = model.model(input_ids=torch.tensor([ids], device=device)).last_hidden_state
        features.append(hidden[0].to(torch.float16).cpu())  # fp16 storage, fp32 compute
    return features


def top_model(full, layers: int):
    """Stage B's model: the top layers, final norm and head of `full`, fed embeddings."""
    offset = layers - TOP_LAYERS
    config = copy.deepcopy(full.config)
    config.num_hidden_layers = TOP_LAYERS
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[offset:]
    top = type(full)(config)
    state = {}
    for key, value in full.state_dict().items():
        if key.startswith("model.layers."):
            index = int(key.split(".")[2])
            if index < offset:
                continue
            key = key.replace(f"model.layers.{index}.", f"model.layers.{index - offset}.", 1)
        state[key] = value
    top.load_state_dict(state, strict=True)
    return top


def answer_loss(model, features, labels) -> torch.Tensor:
    """Cross-entropy on reply tokens only: projecting every position onto the 152k-token
    vocabulary would cost more memory than the whole top model."""
    inner = model.get_base_model()
    hidden = inner.model(inputs_embeds=features).last_hidden_state
    targets = labels[:, 1:]
    mask = targets != -100
    logits = inner.lm_head(hidden[:, :-1][mask])
    return F.cross_entropy(logits, targets[mask])


def feature_cache_path(base: Path, corpus: Path, max_length: int) -> Path:
    """Local, per-machine cache keyed by everything that changes the features."""
    digest = hashlib.sha256()
    for split in ("train", "val"):
        digest.update((corpus / f"{split}.jsonl").read_bytes())
    digest.update(f"{base.resolve()}|{max_length}|{TOP_LAYERS}".encode())
    root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "safety-twin" / "cache"
    return root / f"narration-features-{digest.hexdigest()[:16]}.pt"


def length_buckets(examples, batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffled batches of similar length: sort within windows of 32 batches, then shuffle."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    window = batch_size * 32
    chunks = []
    for start in range(0, len(order), window):
        block = sorted(order[start : start + window], key=lambda i: len(examples[i][0]))
        chunks += [block[i : i + batch_size] for i in range(0, len(block), batch_size)]
    rng.shuffle(chunks)
    return chunks


def batches(features, examples, batch_size: int, shuffle: bool, rng: random.Random):
    if shuffle:
        chunks = length_buckets(examples, batch_size, rng)
    else:
        order = list(range(len(examples)))
        chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    for chunk in chunks:
        width = max(len(examples[i][0]) for i in chunk)
        dim = features[chunk[0]].shape[1]
        inputs = torch.zeros((len(chunk), width, dim))
        labels = torch.full((len(chunk), width), -100)
        for row, index in enumerate(chunk):
            length = len(examples[index][0])
            # Right padding after the reply: padded positions carry no loss and, under
            # causal attention, cannot influence the real positions before them.
            inputs[row, :length] = features[index].float()
            labels[row, :length] = torch.tensor(examples[index][1])
        yield inputs, labels


def remap_adapter(directory: Path, offset: int) -> None:
    """Rename adapter weights from the 8-layer model's 0-7 to the full model's layers."""
    weights = directory / "adapter_model.safetensors"
    renamed = {}
    for key, value in load_file(weights).items():
        parts = key.split(".")
        index = parts.index("layers") + 1
        parts[index] = str(int(parts[index]) + offset)
        renamed[".".join(parts)] = value
    save_file(renamed, weights)
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["layers_to_transform"] = [i + offset for i in range(TOP_LAYERS)]
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", type=Path, default=Path("artifacts/models/llm/Qwen2.5-1.5B-Instruct")
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/working/narration"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/adapters/narration-qwen2.5-1.5b")
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=448)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = "cuda:0"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    full = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    layers = full.config.num_hidden_layers
    train = [encode(tokenizer, f, r, args.max_length) for f, r in load_split(args.corpus, "train")]
    val = [encode(tokenizer, f, r, args.max_length) for f, r in load_split(args.corpus, "val")]

    # Stage A: the frozen bottom, once, in fp32.
    cache = feature_cache_path(args.base, args.corpus, args.max_length)
    if cache.is_file():
        train_features, val_features = torch.load(cache)
        print(f"loaded cached features from {cache}", flush=True)
    else:
        bottom = copy.deepcopy(full)
        bottom.model.layers = bottom.model.layers[: layers - TOP_LAYERS]
        bottom.model.norm = torch.nn.Identity()
        bottom.lm_head = torch.nn.Identity()
        bottom.to(device).eval()
        train_features = cache_features(bottom, train, device)
        val_features = cache_features(bottom, val, device)
        del bottom
        torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save((train_features, val_features), cache)
    cached = time.perf_counter() - started
    print(
        f"features for {len(train_features) + len(val_features)} sequences in {cached:.0f}s",
        flush=True,
    )

    # Stage B: LoRA on the top of the model, in fp32.
    model = top_model(full, layers)
    del full
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(TOP_LAYERS)),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    share = 100 * trainable / FULL_MODEL_PARAMETERS
    print(f"trainable {trainable:,} ({share:.4f} % of the full model)", flush=True)
    if trainable != EXPECTED_TRAINABLE:
        raise SystemExit(
            f"adapter has {trainable} parameters, requirements say {EXPECTED_TRAINABLE}"
        )

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    steps_per_epoch = math.ceil(len(train) / (args.batch * args.accumulate))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, total_steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)),
    )
    history, best = [], float("inf")
    per_epoch = math.ceil(len(train) / args.batch)
    stage_b = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running, micro = 0.0, 0
        for step, (inputs, labels) in enumerate(
            batches(train_features, train, args.batch, shuffle=True, rng=rng), start=1
        ):
            loss = answer_loss(model, inputs.to(device), labels.to(device)) / args.accumulate
            loss.backward()
            running += float(loss.detach()) * args.accumulate
            micro += 1
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % 10 == 0:
                elapsed = time.perf_counter() - started
                rate = (time.perf_counter() - stage_b) / (epoch * per_epoch + step)
                left = rate * (args.epochs * per_epoch - epoch * per_epoch - step)
                print(
                    f"epoch {epoch} step {step}/{per_epoch} loss {running / micro:.4f} "
                    f"{rate:.1f}s/step {elapsed:.0f}s elapsed ~{left / 60:.0f} min left",
                    flush=True,
                )
        model.eval()
        val_loss, count = 0.0, 0
        with torch.inference_mode():
            for inputs, labels in batches(val_features, val, 8, shuffle=False, rng=rng):
                loss = answer_loss(model, inputs.to(device), labels.to(device))
                val_loss += float(loss) * len(inputs)
                count += len(inputs)
        val_loss /= count
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / micro, 4),
                "val_loss": round(val_loss, 4),
            }
        )
        print(f"epoch {epoch} val loss {val_loss:.4f}", flush=True)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(args.out)
    remap_adapter(args.out, layers - TOP_LAYERS)
    tokenizer.save_pretrained(args.out)
    summary = {
        "base": str(args.base),
        "trainable_parameters": trainable,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "targets": ["q_proj", "v_proj"],
            "layers": f"{layers - TOP_LAYERS}-{layers - 1} of {layers}",
        },
        "method": (
            "fp32 feature caching of the frozen bottom layers, fp32 LoRA on the top 8 "
            "with gradient checkpointing and length-bucketed batches"
        ),
        "epochs": args.epochs,
        "effective_batch": args.batch * args.accumulate,
        "lr": args.lr,
        "train_examples": len(train),
        "history": history,
        "best_val_loss": round(best, 4),
        "feature_cache_seconds": round(cached, 1),
        "train_seconds": round(time.perf_counter() - started, 1),
        "seed": SEED,
    }
    (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
