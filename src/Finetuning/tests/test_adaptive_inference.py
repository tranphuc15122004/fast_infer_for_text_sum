from __future__ import annotations

from Finetuning.adaptive_inference import (
    AdaptiveBatchSelection,
    PreparedExample,
    length_bucket_batches,
    probe_batch_limit,
    runtime_batch_size_after_probe,
)


def test_length_bucketing_respects_token_budget_and_preserves_membership() -> None:
    examples = [PreparedExample(index, length) for index, length in enumerate([9, 2, 8, 3])]

    batches = list(length_bucket_batches(examples, batch_size=3, max_tokens=10, window=4))

    assert [[example.index for example in batch] for batch in batches] == [[1, 3], [2], [0]]
    assert sorted(example.index for batch in batches for example in batch) == [0, 1, 2, 3]


def test_probe_batch_limit_respects_token_budget_for_longest_prompt() -> None:
    examples = [PreparedExample(index, 12_000) for index in range(20)]

    assert probe_batch_limit(examples, configured_maximum=128, max_tokens_per_batch=131_072) == 10


def test_probe_batch_limit_rejects_single_prompt_over_token_budget() -> None:
    examples = [PreparedExample(0, 12_000)]

    try:
        probe_batch_limit(examples, configured_maximum=128, max_tokens_per_batch=8_000)
    except ValueError as error:
        assert "exceeds max_tokens_per_batch" in str(error)
    else:
        raise AssertionError("an impossible probe batch must fail before GPU generation")


def test_runtime_batch_uses_configured_max_when_token_budget_limited_probe_fits() -> None:
    fit_at_probe_limit = AdaptiveBatchSelection(10, 100, 90, 4, True)
    memory_limited_before_probe_limit = AdaptiveBatchSelection(8, 100, 90, 4, False)

    assert runtime_batch_size_after_probe(
        fit_at_probe_limit, configured_maximum=128, probed_maximum=10
    ) == 128
    assert runtime_batch_size_after_probe(
        memory_limited_before_probe_limit, configured_maximum=128, probed_maximum=10
    ) == 8
