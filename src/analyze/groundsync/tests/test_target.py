from __future__ import annotations

import torch

from src.analyze.groundsync.trace_target import (
    attention_to_source_distribution,
    build_parser,
    generate_target_trace,
    locate_subsequence,
    render_document_prompt,
)


class TinyTokenizer:
    eos_token_id = 99
    pad_token = None
    pad_token_id = None

    def _ids(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return {"input_ids": self._ids(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
        return_tensors: str | None = None,
        return_dict: bool = False,
    ):
        assert tokenize and add_generation_prompt and not enable_thinking
        content = messages[0]["content"]
        ids = [101] + self._ids(content) + [102]
        if return_tensors == "pt":
            result = torch.tensor([ids], dtype=torch.long)
            return {"input_ids": result} if return_dict else result
        return ids


def test_locate_subsequence_returns_first_exact_match() -> None:
    assert locate_subsequence([1, 2, 3, 2, 3], [2, 3]) == 1
    assert locate_subsequence([1, 2], [3]) is None


def test_attention_to_source_distribution_averages_layers_and_chunks() -> None:
    # Two layers, one head, one query, four key positions. The first two
    # positions are outside the source span; source positions are [2, 3].
    attentions = [
        [[[0.1, 0.1, 0.6, 0.2]]],
        [[[0.2, 0.0, 0.3, 0.5]]],
    ]
    result = attention_to_source_distribution(
        attentions,
        source_start=2,
        source_end=4,
        chunk_size=1,
    )
    assert result == {"raw": [0.5625, 0.4375], "nosink": [0.5625, 0.4375]}


def test_render_document_prompt_finds_source_token_offsets() -> None:
    tokenizer = TinyTokenizer()
    document = "alpha beta"
    rendered = render_document_prompt(tokenizer, document)
    expected = torch.tensor([ord(char) for char in document])
    actual = rendered.input_ids[0, rendered.source_start : rendered.source_end]
    assert torch.equal(actual, expected)
    assert rendered.source_end - rendered.source_start == len(document)
    assert rendered.input_ids.shape[1] == rendered.source_end + 1


class _Output:
    def __init__(self, logits, attentions=None):
        self.logits = logits
        self.attentions = attentions
        self.past_key_values = object()


class OneStepModel:
    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.calls = []

    def __call__(self, *, input_ids, output_attentions, **kwargs):
        self.calls.append({"output_attentions": output_attentions, **kwargs})
        logits = torch.full((1, input_ids.shape[1], 4), -10.0)
        logits[:, -1, 3] = 10.0  # token 3 is not EOS; trace continues once
        if not output_attentions:
            return _Output(logits)
        key_length = self.prompt_length
        mass = torch.zeros((1, 1, 1, key_length))
        mass[..., -2] = 0.6
        mass[..., -1] = 0.4
        return _Output(logits, attentions=[mass])


class AttentionSwitchModel(OneStepModel):
    def __init__(self, prompt_length: int):
        super().__init__(prompt_length)
        self.attention_implementations = []

    def set_attn_implementation(self, implementation: str):
        self.attention_implementations.append(implementation)


def test_generate_target_trace_keeps_one_attention_vector_per_token() -> None:
    tokenizer = TinyTokenizer()
    rendered = render_document_prompt(tokenizer, "ab")
    model = OneStepModel(rendered.input_ids.shape[1])
    result = generate_target_trace(
        model,
        tokenizer,
        rendered,
        sample_id="s1",
        document_id="d1",
        max_new_tokens=2,
        chunk_size=1,
        skip_source_tokens=0,
        device=torch.device("cpu"),
    )
    assert result["status"] == "ok"
    assert result["output_tokens"] == 2
    assert len(result["attention"]) == 2
    assert all(set(step) == {"raw", "nosink"} for step in result["attention"])
    assert all(call["output_attentions"] is True for call in model.calls[1:])


def test_generate_target_trace_can_capture_chunk_sensitivity_variants() -> None:
    tokenizer = TinyTokenizer()
    rendered = render_document_prompt(tokenizer, "abcdef")
    model = OneStepModel(rendered.input_ids.shape[1])
    result = generate_target_trace(
        model,
        tokenizer,
        rendered,
        sample_id="s1",
        document_id="d1",
        max_new_tokens=1,
        chunk_size=2,
        skip_source_tokens=1,
        device=torch.device("cpu"),
        sensitivity_chunk_sizes=(1, 2),
        sink_sizes=(0, 1),
    )
    step = result["attention"][0]
    assert step["raw"] == step["raw_chunk_2"]
    assert step["nosink"] == step["nosink_1_chunk_2"]
    assert "raw_chunk_1" in step
    assert "nosink_0_chunk_1" in step


def test_generate_target_trace_uses_memory_safe_prefill_attention() -> None:
    tokenizer = TinyTokenizer()
    rendered = render_document_prompt(tokenizer, "ab")
    model = AttentionSwitchModel(rendered.input_ids.shape[1])
    generate_target_trace(
        model,
        tokenizer,
        rendered,
        sample_id="s1",
        document_id="d1",
        max_new_tokens=1,
        chunk_size=1,
        skip_source_tokens=0,
        device=torch.device("cpu"),
    )
    assert model.attention_implementations == ["sdpa", "eager"]


def test_generate_target_trace_chunks_long_prefill_with_bottom_right_mask() -> None:
    tokenizer = TinyTokenizer()
    rendered = render_document_prompt(tokenizer, "abcdef")
    model = AttentionSwitchModel(rendered.input_ids.shape[1])
    generate_target_trace(
        model,
        tokenizer,
        rendered,
        sample_id="s1",
        document_id="d1",
        max_new_tokens=1,
        chunk_size=1,
        skip_source_tokens=0,
        device=torch.device("cpu"),
        prefill_chunk_size=2,
    )
    prefill_calls = [call for call in model.calls if not call["output_attentions"]]
    assert len(prefill_calls) > 1
    assert prefill_calls[1]["attention_mask"].shape[-2:] == (2, 4)
    assert torch.isneginf(prefill_calls[1]["attention_mask"][0, 0, 0, 3])


def test_target_parser_accepts_sensitivity_and_sink_lists() -> None:
    args = build_parser().parse_args([
        "--input", "input.jsonl", "--output", "output.jsonl",
        "--sensitivity-chunk-sizes", "64,128,256",
        "--sink-sizes", "4,8,16",
    ])
    assert args.sensitivity_chunk_sizes == "64,128,256"
    assert args.sink_sizes == "4,8,16"
