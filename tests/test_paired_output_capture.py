from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.qwen3_paired import capture_generation_output  # noqa: E402


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
