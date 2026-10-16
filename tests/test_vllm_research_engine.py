from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@pytest.fixture
def fake_runtime(monkeypatch):
    calls, configs = [], []

    class Engine:
        def __init__(self, **kwargs):
            configs.append(kwargs)

        def generate(self, prompts, sampling, use_tqdm=False):
            calls.append({'size': len(prompts), 'max_tokens': sampling.max_tokens})
            return [SimpleNamespace(
                request_id=str(i), prompt_token_ids=p['prompt_token_ids'],
                outputs=[SimpleNamespace(token_ids=[3, 4], text='ok', finish_reason='length')],
                metrics=SimpleNamespace(queued_ts=1., scheduled_ts=1.01,
                                        first_token_ts=1.02, last_token_ts=1.12,
                                        first_token_latency=.02),
            ) for i, p in enumerate(prompts)]

        def shutdown(self):
            pass

    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: 'ok')
    torch = SimpleNamespace(__version__='2.13.0', cuda=SimpleNamespace(
        is_available=lambda: True, synchronize=lambda: None, empty_cache=lambda: None))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        __version__='5.12.1', AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(__version__='0.30.0', LLM=Engine,
                                                           SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    return calls, configs, Engine


def _inputs():
    job = {'id': 'native', 'experiment': 'E2E', 'variant': 'dflash_full',
           'parameters': {'draft_gamma': 15}, 'kind': 'generation'}
    samples = [{'sample_id': str(i), 'dataset': 'd', 'prompt': 'p',
                'prompt_token_ids': [i, 2], 'input_tokens': 2, 'reference': None,
                'source_group_id': 'source'} for i in range(5)]
    settings = {'target_model': '/target', 'draft_model': '/draft', 'max_new_tokens': 512,
                'max_model_len': 1024, 'measurement': 'throughput', 'max_num_seqs': 2,
                'warmup_prompts': 3, 'warmup_tokens': 64, 'repetitions': 1, 'seed': 42}
    return job, samples, settings


def test_generation_keeps_one_engine_batches_and_resumes_per_request(tmp_path, fake_runtime):
    from Benchmark.vllm_research.engine import run_generation
    calls, configs, _ = fake_runtime
    job, samples, settings = _inputs()
    result = run_generation(job, samples, settings, tmp_path)
    assert result['status'] == 'succeeded'
    assert len(configs) == 1
    assert [row['size'] for row in calls] == [1, 1, 1, 2, 2, 1]
    assert [row['max_tokens'] for row in calls[:3]] == [64] * 3
    assert configs[0]['max_num_seqs'] == 2
    assert configs[0]['attention_config'] == {'backend': 'FLASH_ATTN'}
    assert result['records'][0]['client_wall_ms'] is None
    assert result['records'][0]['e2e_ms'] == pytest.approx(120.)
    assert len(list((tmp_path / 'cells').glob('*.json'))) == 5
    assert json.loads((tmp_path / 'results.jsonl').read_text().splitlines()[-1])['record_type'] == 'summary'
    settings['resume'] = True
    repeated = run_generation(job, samples, settings, tmp_path)
    assert repeated['status'] == 'succeeded'
    assert len(calls) == 6
    assert len(configs) == 1


def test_worker_respects_adapter_serial_requirement_without_silent_profile_change(
    tmp_path, fake_runtime, monkeypatch
):
    from Benchmark.vllm_research import engine
    job, samples, settings = _inputs()
    job.update(variant='fixed_k_7', experiment='B3', parameters={'draft_gamma': 15, 'k': 7})
    monkeypatch.setattr(engine, 'configure_worker_variant', lambda *a, **kw: {
        'env': {}, 'engine_overrides': {'max_num_seqs': 1}})
    result = engine.run_generation(job, samples, settings, tmp_path)
    assert result['status'] == 'unsupported'
    assert not fake_runtime[1]
    assert 'latency' in result['reason']


def test_generation_failures_are_durable_and_do_not_stop_later_batches(tmp_path, fake_runtime):
    from Benchmark.vllm_research.engine import run_generation
    calls, configs, Engine = fake_runtime
    job, samples, settings = _inputs()
    settings['warmup_prompts'] = 0
    original = Engine.generate

    def fail_one(self, prompts, sampling, use_tqdm=False):
        if prompts[0]['prompt_token_ids'][0] == 2:
            raise RuntimeError('controlled failure')
        return original(self, prompts, sampling, use_tqdm=use_tqdm)

    Engine.generate = fail_one
    result = run_generation(job, samples, settings, tmp_path)
    assert result['status'] == 'failed'
    assert len(result['records']) == 5
    assert [r['execution_status'] for r in result['records']] == [
        'success', 'success', 'failed', 'failed', 'success']
    assert len(list((tmp_path / 'cells').glob('*.json'))) == 5
