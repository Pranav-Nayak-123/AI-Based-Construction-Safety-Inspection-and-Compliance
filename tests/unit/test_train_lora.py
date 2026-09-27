"""The two-stage LoRA trainer is exactly fine-tuning the full model (tiny random Qwen2)."""

from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

from llm.train_lora import (  # noqa: E402
    TOP_LAYERS,
    answer_loss,
    length_buckets,
    remap_adapter,
    top_model,
)

LAYERS = TOP_LAYERS + 2


@pytest.fixture(scope="module")
def full():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def _bottom_features(full, ids):
    bottom = copy.deepcopy(full)
    bottom.model.layers = bottom.model.layers[: LAYERS - TOP_LAYERS]
    bottom.model.norm = torch.nn.Identity()
    bottom.lm_head = torch.nn.Identity()
    with torch.no_grad():
        return bottom.model(input_ids=ids).last_hidden_state


def _lora():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(TOP_LAYERS)),
        task_type="CAUSAL_LM",
        init_lora_weights=False,  # non-zero B, so the adapter actually changes the output
    )


def test_top_model_on_cached_features_matches_the_full_model(full) -> None:
    ids = torch.randint(0, 97, (2, 12))
    features = _bottom_features(full, ids)
    top = top_model(full, LAYERS).eval()
    with torch.no_grad():
        expected = full(input_ids=ids).logits
        actual = top.lm_head(top.model(inputs_embeds=features).last_hidden_state)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_checkpointing_does_not_change_the_loss_or_gradients(full) -> None:
    ids = torch.randint(0, 97, (2, 12))
    labels = ids.clone()
    labels[:, :6] = -100
    features = _bottom_features(full, ids)
    results = []
    for checkpointing in (False, True):
        torch.manual_seed(1)
        top = top_model(full, LAYERS)
        top.config.use_cache = False
        if checkpointing:
            top.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model = get_peft_model(top, _lora())
        model.train()
        for module in model.modules():  # no dropout, so both runs are deterministic
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        loss = answer_loss(model, features, labels)
        loss.backward()
        grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
        results.append((loss.detach(), grads))
    (loss_a, grads_a), (loss_b, grads_b) = results
    torch.testing.assert_close(loss_a, loss_b)
    for a, b in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_remapped_adapter_loads_onto_the_full_model(full, tmp_path) -> None:
    ids = torch.randint(0, 97, (1, 10))
    torch.manual_seed(2)
    trained = get_peft_model(top_model(full, LAYERS), _lora()).eval()
    trained.save_pretrained(tmp_path)
    remap_adapter(tmp_path, LAYERS - TOP_LAYERS)
    loaded = PeftModel.from_pretrained(copy.deepcopy(full), tmp_path).eval()
    features = _bottom_features(full, ids)
    with torch.no_grad():
        inner = trained.get_base_model()
        expected = inner.lm_head(inner.model(inputs_embeds=features).last_hidden_state)
        actual = loaded(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        assert not torch.allclose(actual, full(input_ids=ids).logits)  # the adapter is live


def test_length_buckets_cover_every_example_once() -> None:
    examples = [([0] * random.Random(i).randint(5, 400), None) for i in range(530)]
    chunks = length_buckets(examples, 8, random.Random(3))
    flat = sorted(i for chunk in chunks for i in chunk)
    assert flat == list(range(530))
    assert all(len(chunk) <= 8 for chunk in chunks)
