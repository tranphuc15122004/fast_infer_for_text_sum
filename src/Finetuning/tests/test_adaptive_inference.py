from __future__ import annotations

from Finetuning.adaptive_inference import PreparedExample, length_bucket_batches


def test_length_bucketing_respects_token_budget_and_preserves_membership() -> None:
    examples = [PreparedExample(index, length) for index, length in enumerate([9, 2, 8, 3])]

    batches = list(length_bucket_batches(examples, batch_size=3, max_tokens=10, window=4))

    assert [[example.index for example in batch] for batch in batches] == [[1, 3], [2], [0]]
    assert sorted(example.index for batch in batches for example in batch) == [0, 1, 2, 3]
