from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.qwen3_paired import (  # noqa: E402
    capture_generation_output,
    model_provenance,
)


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del clean_up_tokenization_spaces
        tokens = [int(token) for token in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token != self.eos_token_id]
        return " ".join(map(str, tokens))


def test_capture_preserves_fixed_sequence_and_derives_eos_truncated_quality_text():
    captured = capture_generation_output(
        FakeTokenizer(), [10, 11, 99, 12, 13], expected_output_tokens=5
    )

    assert captured["generated_token_ids"] == [10, 11, 99, 12, 13]
    assert captured["first_eos_index"] == 2
    assert captured["quality_output_tokens"] == 2
    assert captured["full_output_text"] == "10 11 12 13"
    assert captured["quality_text"] == "10 11"
    assert captured["fixed_budget_reached"] is True


def test_capture_handles_no_eos_and_tensor_like_ids():
    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[7, 8]]

    captured = capture_generation_output(
        FakeTokenizer(), TensorLike(), expected_output_tokens=2
    )

    assert captured["generated_token_ids"] == [7, 8]
    assert captured["first_eos_index"] is None
    assert captured["quality_output_tokens"] == 2
    assert captured["quality_text"] == "7 8"
    assert captured["fixed_budget_reached"] is True


def test_model_provenance_records_target_draft_and_tokenizer_identity():
    class ProvenanceTokenizer:
        name_or_path = "/models/tokenizer"
        init_kwargs = {"revision": "tokenizer-rev"}
        eos_token_id = 99
        bos_token_id = 1
        pad_token_id = 0

        def __len__(self):
            return 100

    tokenizer = ProvenanceTokenizer()
    target = SimpleNamespace(config=SimpleNamespace(_commit_hash="target-rev"))
    draft = SimpleNamespace(config=SimpleNamespace(_commit_hash="draft-rev"))

    provenance = model_provenance(target, tokenizer, draft_model=draft)

    assert provenance["target_model_revision"] == "target-rev"
    assert provenance["draft_model_revision"] == "draft-rev"
    assert provenance["tokenizer_name_or_path"] == "/models/tokenizer"
    assert provenance["tokenizer_revision"] == "tokenizer-rev"
    assert provenance["tokenizer_vocab_size"] == 100
    assert provenance["special_token_ids"] == {"bos": 1, "eos": 99, "pad": 0}


def test_quality_decode_preserves_leading_and_trailing_whitespace():
    class WhitespaceTokenizer(FakeTokenizer):
        def decode(self, token_ids, **kwargs):
            del token_ids, kwargs
            return "  indented output\n"

    captured = capture_generation_output(WhitespaceTokenizer(), [10])

    assert captured["quality_text"] == "  indented output\n"
    assert captured["full_output_text"] == "  indented output\n"
