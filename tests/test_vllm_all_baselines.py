from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import Benchmark.vllm_all_baselines as evaluator
from Benchmark.vllm_all_baselines import (
    build_speculative_config,
    mean_itl_ms,
    order_methods,
    token_lcs_overlap,
)


def test_json_artifacts_end_with_real_newlines(tmp_path):
    row_path = tmp_path / "rows.jsonl"
    json_path = tmp_path / "report.json"

    evaluator._append_jsonl(row_path, {"sample_id": "a"})
    evaluator._append_jsonl(row_path, {"sample_id": "b"})
    evaluator._write_json(json_path, {"status": "ok"})

    assert row_path.read_text(encoding="utf-8").splitlines() == [
        '{"sample_id": "a"}',
        '{"sample_id": "b"}',
    ]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "ok"}


def test_normalize_vllm_spec_metrics_derives_counters_and_rates():
    metrics = {
        "num_drafts": 2,
        "num_draft_tokens": 8,
        "num_accepted_tokens": 6,
        "num_accepted_tokens_per_pos": [2, 2, 1, 1],
    }

    normalized = evaluator._normalize_vllm_spec_metrics(metrics, method="dflash")

    assert normalized["verification_steps"] == 2
    assert normalized["draft_tokens_accepted"] == 6
    assert normalized["draft_tokens_proposed"] == 8
    assert normalized["acceptance_rate"] == pytest.approx(0.75)
    assert normalized["acceptance_rate_percent"] == pytest.approx(75.0)
    assert normalized["avg_accept_length"] == pytest.approx(4.0)
    assert normalized["draft_proposal_unit"] == "draft_tokens"


def test_normalize_vllm_spec_metrics_preserves_tree_proposal_unit():
    normalized = evaluator._normalize_vllm_spec_metrics(
        {"acceptance_rate": 0.25, "avg_accept_length": 2.0}, method="eagle3"
    )

    assert normalized["acceptance_rate"] == pytest.approx(0.25)
    assert normalized["draft_proposal_unit"] == "draft_tree_nodes"


def test_event_jsonl_is_timestamped_and_keeps_details(tmp_path):
    event_path = tmp_path / "events.jsonl"
    started = evaluator.time.perf_counter()

    evaluator._append_event(
        event_path, "request_finished", started, method="dflash", sample_id="a", output_tokens=7
    )

    event = json.loads(event_path.read_text(encoding="utf-8"))
    assert event["event"] == "request_finished"
    assert event["elapsed_ms"] >= 0
    assert event["method"] == "dflash"
    assert event["sample_id"] == "a"
    assert event["output_tokens"] == 7


def test_nvidia_smi_snapshot_parses_device_telemetry(monkeypatch):
    class Completed:
        returncode = 0
        stdout = (
            "0, NVIDIA H100, 81920, 12345, 69475, 50, 40, 61, 550.54.15\n"
        )
        stderr = ""

    monkeypatch.setattr(evaluator.subprocess, "run", lambda *args, **kwargs: Completed())

    snapshot = evaluator._nvidia_smi_snapshot()

    assert snapshot["available"] is True
    assert snapshot["devices"] == [
        {
            "index": "0",
            "name": "NVIDIA H100",
            "memory_total_mib": "81920",
            "memory_used_mib": "12345",
            "memory_free_mib": "69475",
            "gpu_utilization_percent": "50",
            "memory_utilization_percent": "40",
            "temperature_c": "61",
            "driver_version": "550.54.15",
        }
    ]


def test_method_metrics_summarizes_speculative_counters():
    records = [
        {
            "method": "eagle3",
            "status": "success",
            "quality_valid": True,
            "repetition_flag": False,
            "tpot_ms": 10.0,
            "raw_prefill_ms": 20.0,
            "rouge1": 0.3,
            "rouge2": 0.2,
            "rougeL": 0.25,
            "speculative_metrics_available": True,
            "acceptance_rate": 0.5,
            "avg_accept_length": 2.0,
            "draft_tokens_accepted": 2,
            "draft_tokens_proposed": 4,
            "draft_proposal_unit": "draft_tree_nodes",
        },
        {
            "method": "eagle3",
            "status": "success",
            "quality_valid": True,
            "repetition_flag": False,
            "tpot_ms": 12.0,
            "raw_prefill_ms": 22.0,
            "rouge1": 0.5,
            "rouge2": 0.4,
            "rougeL": 0.45,
            "speculative_metrics_available": True,
            "acceptance_rate": 0.75,
            "avg_accept_length": 4.0,
            "draft_tokens_accepted": 6,
            "draft_tokens_proposed": 8,
            "draft_proposal_unit": "draft_tree_nodes",
        },
    ]

    metrics = evaluator._method_metrics(records, ("eagle3",))["eagle3"]

    assert metrics["mean_acceptance_rate"] == 0.625
    assert metrics["mean_acceptance_rate_percent"] == 62.5
    assert metrics["mean_avg_accept_length"] == 3.0
    assert metrics["total_draft_tokens_accepted_observed"] == 8
    assert metrics["total_draft_tokens_proposed_observed"] == 12
    assert metrics["draft_proposal_units"] == ["draft_tree_nodes"]


def test_prepared_sample_preserves_source_row_for_offline_recalculation():
    class StubTokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    source = {
        "id": "gov_report_0",
        "dataset": "gov_report",
        "context": "Report document text.",
        "input": "Summarize the report.",
        "reference_output": "Gold summary.",
        "document_words": 10,
    }

    samples, excluded = evaluator._prepare_samples(
        [source],
        StubTokenizer(),
        max_input_tokens=4096,
        max_total_tokens=4096,
        max_samples=0,
    )

    assert not excluded
    assert samples[0]["source_record"] == source
    assert samples[0]["reference"] == source["reference_output"]
    assert samples[0]["document_words"] == source["document_words"]


def test_full_cli_flag_is_available_and_wins_over_smoke_flag():
    args = evaluator._parser().parse_args(
        ["--model", "m", "--data-file", "d", "--output-dir", "o", "--full", "--smoke"]
    )

    assert evaluator._resolve_sample_limit(
        full=args.full,
        smoke=args.smoke,
        max_samples=args.max_samples,
    ) == 0


def test_full_mode_ignores_smoke_and_max_sample_caps():
    assert evaluator._resolve_sample_limit(
        full=True,
        smoke=True,
        max_samples=2,
    ) == 0


def test_smoke_mode_still_limits_samples_when_full_is_not_requested():
    assert evaluator._resolve_sample_limit(
        full=False,
        smoke=True,
        max_samples=0,
    ) == 2


def test_order_methods_keeps_vanilla_first_and_preserves_requested_order():
    assert order_methods("domino,vanilla_vllm,eagle3") == (
        "vanilla_vllm",
        "domino",
        "eagle3",
    )


def test_order_methods_requires_vanilla_and_rejects_duplicates():
    with pytest.raises(ValueError, match="must include reference"):
        order_methods("eagle3,domino")
    with pytest.raises(ValueError, match="duplicates"):
        order_methods("vanilla_vllm,eagle3,eagle3")


def test_domino_uses_the_vllm_dflash_adapter_and_keeps_block_size():
    config, metadata = build_speculative_config(
        "domino",
        "/checkpoints/domino",
        {
            "block_size": 16,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "dflash_config": {"projector_type": "domino", "shift_label": True},
        },
        target_num_hidden_layers=36,
    )

    assert config == {
        "method": "dflash",
        "model": "/checkpoints/domino",
        "num_speculative_tokens": 16,
    }
    assert metadata["vllm_method"] == "dflash"
    assert metadata["projector_type"] == "domino"


def test_mean_itl_uses_inter_token_intervals_not_token_count():
    assert mean_itl_ms(first_token_ts=10.000, last_token_ts=10.018, output_tokens=4) == pytest.approx(6.0)
    assert mean_itl_ms(first_token_ts=10.0, last_token_ts=10.0, output_tokens=1) is None


def test_token_lcs_overlap_is_ordered_and_uses_vanilla_length_as_denominator():
    assert token_lcs_overlap([1, 2, 3, 4], [1, 3, 4]) == pytest.approx(0.75)
    assert token_lcs_overlap([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]) == pytest.approx(1.0)
    assert token_lcs_overlap([], []) is None


def test_execution_profile_defaults_to_real_throughput_batches():
    args = evaluator._parser().parse_args(
        ['--model', 'm', '--data-file', 'd', '--output-dir', 'o']
    )
    assert args.measurement == 'throughput'
    assert args.max_num_seqs == 16
    assert args.diagnostics == 'report-only'
    assert evaluator._execution_concurrency(args) == 16
    args.measurement = 'latency'
    assert evaluator._execution_concurrency(args) == 1


def test_generation_batches_submit_multiple_prompts_and_isolate_failed_batches():
    class Engine:
        def generate(self, prompts, sampling, use_tqdm=False):
            if prompts[0]['prompt_token_ids'] == [2]:
                raise RuntimeError('batch failed')
            return [prompt['prompt_token_ids'][0] for prompt in prompts]

    samples = [{'prompt_token_ids': [index]} for index in range(5)]
    observed = list(evaluator._generation_batches(
        Engine(), samples, object(), batch_size=2, synchronize=lambda: None
    ))
    assert [row['output'] for row in observed] == [0, 1, None, None, 4]
    assert [row['batch_size'] for row in observed] == [2, 2, 2, 2, 1]
    assert [row['index'] for row in observed] == [1, 2, 3, 4, 5]
    assert isinstance(observed[2]['error'], RuntimeError)
    assert observed[0]['batch_id'] == observed[1]['batch_id']
    assert observed[-1]['batch_wall_ms'] >= 0


def test_warmup_uses_at_most_three_representative_prompt_lengths():
    samples = [{'input_tokens': n} for n in [99, 5, 40, 6, 300, 50, 12]]
    assert [row['input_tokens'] for row in evaluator._warmup_samples(samples, 3)] == [5, 40, 300]
    assert evaluator._warmup_samples(samples, 0) == []


def test_report_only_diagnostics_do_not_hide_execution_failures():
    assert evaluator._result_exit_code(complete=True, execution=True, quality=False,
                                       correctness=False, diagnostics='report-only') == 0
    assert evaluator._result_exit_code(complete=True, execution=False, quality=True,
                                       correctness=True, diagnostics='report-only') == 1
    assert evaluator._result_exit_code(complete=True, execution=True, quality=True,
                                       correctness=False, diagnostics='strict') == 1


def test_throughput_aggregation_counts_each_batch_wall_once_and_keeps_request_latency():
    records = [
        {'method': 'vanilla_vllm', 'status': 'success', 'quality_valid': True,
         'tpot_ms': 20, 'raw_prefill_ms': 10, 'measurement': 'throughput',
         'batch_id': batch, 'batch_wall_ms': wall, 'output_tokens': tokens,
         'e2e_ms': latency}
        for batch, wall, tokens, latency in [(0, 100, 4, 80), (0, 100, 6, 95), (1, 50, 5, 40)]
    ]
    metrics = evaluator._method_metrics(records, ('vanilla_vllm',))['vanilla_vllm']
    assert metrics['measured_makespan_ms'] == 150
    assert metrics['output_throughput_tok_s'] == 100
    assert metrics['request_throughput_qps'] == 20
    assert metrics['latency_p50_ms'] == 80
    assert metrics['latency_p95_ms'] == pytest.approx(93.5)
    assert metrics['paired_speed_metrics'] is None


def test_request_without_timing_remains_executed_but_diagnostic_invalid():
    from types import SimpleNamespace
    sample = {'sample_id': 'a', 'dataset': 'd', 'input_tokens': 2,
              'prompt': 'p', 'prompt_token_ids': [1, 2], 'reference': None}
    request = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[3, 4], text='ok',
                              finish_reason='length')], metrics=None, prompt_token_ids=[1, 2])
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: 'ok')
    record = evaluator._request_record(method='vanilla_vllm', sample=sample, model='m',
                                       request_output=request, tokenizer=tokenizer,
                                       client_wall_ms=None)
    assert record['execution_status'] == 'success'
    assert record['status'] == 'invalid'
    assert record['e2e_ms'] is None
    assert record['client_wall_ms'] is None


@pytest.mark.parametrize('measurement,expected_sizes', [('throughput', [2, 2, 1]), ('latency', [1] * 5)])
def test_evaluate_uses_selected_concurrency_and_finalizes_real_artifacts(
    tmp_path, monkeypatch, measurement, expected_sizes
):
    from types import SimpleNamespace

    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    data = tmp_path / 'data.jsonl'
    data.write_text('{}\n')
    samples = [dict(sample_id=str(i), dataset='d', input_tokens=2, document_words=1,
                    prompt='p', prompt_token_ids=[i, 2], reference=None, source_record={})
               for i in range(5)]
    engine_calls = []
    engine_config = {}

    class Engine:
        def __init__(self, **kwargs):
            engine_config.update(kwargs)

        def generate(self, prompts, sampling, use_tqdm=False):
            engine_calls.append(len(prompts))
            return [SimpleNamespace(
                outputs=[SimpleNamespace(token_ids=[3, 4], text='ok', finish_reason='length')],
                prompt_token_ids=prompt['prompt_token_ids'],
                metrics=SimpleNamespace(scheduled_ts=1., first_token_ts=1.01,
                                        last_token_ts=1.1, queued_ts=1., first_token_latency=.01),
            ) for prompt in prompts]

        def shutdown(self):
            pass

    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: 'ok')
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(
            num_hidden_layers=36, to_dict=lambda: {})), __version__='5.12.1')
    torch = SimpleNamespace(__version__='2.13.0', cuda=SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda _: 'Fake GPU',
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs), __version__='0.30.0'))
    monkeypatch.setattr(evaluator, '_runtime_snapshot', lambda *args: {})
    monkeypatch.setattr(evaluator, '_gpu_memory_snapshot', lambda *args: {})
    monkeypatch.setattr(evaluator, '_nvidia_smi_snapshot', lambda: {})
    monkeypatch.setattr(evaluator, '_prepare_samples', lambda *args, **kwargs: (samples, []))
    args = evaluator._parser().parse_args([
        '--model', str(checkpoint), '--data-file', str(data), '--output-dir', str(tmp_path / 'out'),
        '--methods', 'vanilla_vllm', '--measurement', measurement, '--max-num-seqs', '2',
        '--warmup-prompts', '0', '--run-id', 'test',
    ])
    assert evaluator._evaluate(args) == 0
    assert engine_calls == expected_sizes
    assert engine_config['max_num_seqs'] == max(expected_sizes)
    output = tmp_path / 'out' / 'test'
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 6
    assert rows[-1]['record_type'] == 'summary'
    assert rows[-1]['execution_status'] == 'complete'
    assert rows[-1]['configuration']['measurement'] == measurement
    assert not (output / 'results.partial.jsonl').exists()
    if measurement == 'throughput':
        assert rows[0]['client_wall_ms'] is None
        assert rows[0]['e2e_ms'] == pytest.approx(100.)


def test_parser_aliases_and_env_fallbacks():
    # Test that --data-path, --max-tokens, --batch-size aliases parse accurately
    args = evaluator._parser().parse_args([
        "--data-path", "sample.jsonl",
        "--max-tokens", "50",
        "--batch-size", "16",
    ])
    assert args.data_file == "sample.jsonl"
    assert args.max_samples == 50
    assert args.max_num_seqs == 16

