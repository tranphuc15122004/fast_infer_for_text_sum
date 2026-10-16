"""Worker generation vLLM: engine sống qua corpus, checkpoint từng request.

Module này không import CUDA/vLLM trước khi cấu hình plugin worker. Mỗi process
chạy một job; scheduler và target verification vẫn thuộc engine vLLM.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from Benchmark.common.io_util import JsonlWriter
from Benchmark.vllm_all_baselines import (
    _append_jsonl, _failed_record, _generation_batches, _method_metrics, _request_record,
    _shutdown_engine, _warmup_samples,
)
from .compat import configure_worker_variant
from .execution import atomic_json, identity


def _adapter_variant(job: dict[str, Any]) -> str:
    if job['variant'] == 'full_adapter':
        return 'full_adapter'
    if job['experiment'] == 'A3':
        return 'a3'
    if job['experiment'] == 'B3':
        return 'b3_fixed_k' if job['variant'].startswith('fixed_k_') else 'b3'
    if job['experiment'] == 'B4' and job['variant'].startswith('adaptive_gamma_'):
        return 'b4'
    if job.get('kind') == 'trace':
        return 'trace'
    if job['variant'] in {'ar', 'dflash_full'} or job['variant'].startswith('fixed_gamma_'):
        return 'native'
    raise ValueError(f'unknown generation variant: {job["variant"]}')


def _implementation_identity() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name('compat.py'),
             Path(__file__).with_name('adapter.py'), Path(__file__).with_name('context.py'),
             Path(__file__).parents[1] / 'vllm_all_baselines.py']
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def _model_identity(model: str) -> dict[str, Any]:
    root = Path(model)
    inventory = {}
    if root.is_dir():
        for pattern in ('*.safetensors', '*.bin', '*.json'):
            for path in sorted(root.glob(pattern)):
                stat = path.stat()
                inventory[path.name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                if path.suffix == '.json':
                    inventory[path.name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'path': str(root.resolve()), 'inventory': inventory}


def run_generation(job: dict[str, Any], samples: list[dict[str, Any]],
                   settings: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    """Sinh tất cả request/repetition; scientific diagnostics không chặn job.

    Không tự thay execution profile của custom arm thành batch 1 nếu operator
    yêu cầu throughput nhiều request. Capability ấy được báo unsupported.
    """
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    if job.get('kind', 'generation') not in {'generation', 'trace'}:
        return {'status': 'unsupported', 'reason': 'generation worker cannot execute same-prefix replay',
                'records': [], 'metrics': {}}
    try:
        variant = _adapter_variant(job)
    except ValueError as exc:
        return {'status': 'unsupported', 'reason': str(exc), 'records': [], 'metrics': {}}
    parameters = dict(job.get('parameters') or {})
    layout_value = settings.get('prompt_layouts_file') or settings.get('prompt_layouts') or parameters.get('prompt_layouts_file') or parameters.get('prompt_layouts') or {}
    layout_file = str(layout_value) if isinstance(layout_value, str) else None
    layouts = {} if layout_file else dict(layout_value)
    for sample in samples:
        prompt_hash = hashlib.sha256(json.dumps(sample['prompt_token_ids'], separators=(',', ':')).encode()).hexdigest()
        layout = dict(layouts.get(prompt_hash) or sample.get('layout') or sample.get('prompt_layout') or {})
        if not layout and 'source_positions' in sample and 'global_positions' in sample:
            layout = {key: sample[key] for key in ('source_positions', 'global_positions')}
        # Source boundaries must be supplied exactly by tokenization, never guessed.
        if layout or variant != 'a3':
            layout.update({'prompt_tokens': len(sample['prompt_token_ids']), 'sample_id': sample['sample_id'],
                           'dataset': sample['dataset'], 'source_group_id': sample.get('source_group_id')})
            layouts[prompt_hash] = layout
    if layout_file:
        parameters['prompt_layouts_file'] = layout_file
    elif layouts:
        parameters['prompt_layouts'] = layouts
    requested_concurrency = 1 if settings.get('measurement', 'throughput') == 'latency' else int(settings.get('max_num_seqs', 16))
    if requested_concurrency < 1:
        raise ValueError('max_num_seqs must be positive')
    repetitions = int(settings.get('repetitions', 1))
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    trace_dir = job_dir / 'traces' if variant != 'native' else None
    try:
        configured = configure_worker_variant(variant, parameters, str(trace_dir) if trace_dir else None)
    except ValueError as exc:
        return {'status': 'unsupported', 'reason': str(exc), 'records': [], 'metrics': {}}
    overrides = configured['engine_overrides']
    if overrides.get('max_num_seqs', requested_concurrency) != requested_concurrency:
        return {'status': 'unsupported', 'reason': 'worker policy requires batch 1; run --measurement latency for comparable controls',
                'records': [], 'metrics': {}}

    runtime_metadata = {}
    for package in ('vllm', 'torch', 'transformers'):
        try:
            runtime_metadata[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_metadata[package] = None
    execution_settings = {key: value for key, value in settings.items() if key not in {'resume', 'output_dir'}}
    run_identity = identity({'job': job, 'settings': execution_settings,
                             'implementation': _implementation_identity(), 'packages': runtime_metadata,
                             'target': _model_identity(settings['target_model']),
                             'draft': _model_identity(settings['draft_model']) if job['variant'] != 'ar' else None})
    cells_dir = job_dir / 'cells'
    cells_dir.mkdir(exist_ok=True)
    records = []
    pending_by_repetition = []
    for repetition in range(repetitions):
        pending = []
        for sample in samples:
            cell_id = identity({'execution': run_identity, 'repetition': repetition,
                                'sample_id': sample['sample_id'], 'prompt_token_ids': sample['prompt_token_ids'],
                                'source_group_id': sample.get('source_group_id'), 'layout': sample.get('layout'),
                                'reference': sample.get('reference')})
            path = cells_dir / f'{cell_id}.json'
            previous = json.loads(path.read_text()) if settings.get('resume') and path.is_file() else None
            if previous and (previous.get('execution_status') == 'success' or previous.get('attempts', 0) >= 2):
                records.append(previous)
            else:
                pending.append((sample, cell_id, path, (previous or {}).get('attempts', 0) + 1))
        pending_by_repetition.append(pending)
    engine = None
    collection_id = str(time.time_ns())
    original_env = dict(os.environ)
    original_sys_path = list(sys.path)
    try:
        if any(pending_by_repetition):
            # Native controls use the same runner/attention/eager configuration as
            # custom arms, while leaving signal collection disabled.
            os.environ.pop('FAST_INFER_VLLM_RESEARCH_CONFIG', None)
            plugins = [p for p in os.environ.get('VLLM_PLUGINS', '').split(',')
                       if p and p != 'fast_infer_research']
            if 'VLLM_PLUGINS' in os.environ:
                os.environ['VLLM_PLUGINS'] = ','.join(plugins)
            os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                               'VLLM_USE_V2_MODEL_RUNNER': '0'})
            os.environ.update(configured['env'])
            # Spawned workers inherit sys.path; env-only bootstrap is insufficient
            # for entry-point metadata discovery in multiprocessing children.
            for directory in reversed(configured['env'].get('PYTHONPATH', '').split(os.pathsep)):
                if directory and directory not in sys.path:
                    sys.path.insert(0, directory)
            import torch
            import transformers
            import vllm
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            if not torch.cuda.is_available():
                raise RuntimeError('vLLM research generation requires CUDA on the server')
            if vllm.__version__ != '0.30.0':
                raise RuntimeError(f'research worker requires vLLM 0.30.0; installed={vllm.__version__}')
            tokenizer = AutoTokenizer.from_pretrained(settings['target_model'], local_files_only=True)
            kwargs = {
                'model': settings['target_model'], 'tokenizer': settings['target_model'],
                'dtype': settings.get('dtype', 'bfloat16'),
                'max_model_len': settings.get('max_model_len', 12288),
                'max_num_seqs': requested_concurrency,
                'gpu_memory_utilization': settings.get('gpu_memory_utilization', .85),
                'seed': settings.get('seed', 42), 'enable_prefix_caching': False,
                'trust_remote_code': False, 'disable_log_stats': False,
                'tensor_parallel_size': 1, 'enforce_eager': True,
                'async_scheduling': False, 'attention_config': {'backend': 'FLASH_ATTN'},
            }
            if settings.get('max_num_batched_tokens') is not None:
                kwargs['max_num_batched_tokens'] = settings['max_num_batched_tokens']
            if job['variant'] != 'ar':
                kwargs['speculative_config'] = {
                    'method': 'dflash', 'model': settings['draft_model'],
                    'num_speculative_tokens': int(parameters.get('draft_gamma', 15)),
                }
                kwargs['per_request_spec_decode_metrics'] = 'detailed'
            kwargs.update(overrides)
            atomic_json(job_dir / 'runtime.json', {
                'execution_identity': run_identity, 'engine_kwargs': kwargs,
                'versions': {'vllm': vllm.__version__, 'torch': torch.__version__,
                             'transformers': transformers.__version__},
                'worker_variant': variant, 'trace_dir': str(trace_dir) if trace_dir else None,
                'gpu_available': True, 'worker_started': True,
                'gpu_verified': False, 'semantic_parity_verified': False,
            })
            engine = LLM(**kwargs)
            sampling = SamplingParams(temperature=0., seed=settings.get('seed', 42), n=1,
                                      max_tokens=settings.get('max_new_tokens', 512))
            warmup_sampling = SamplingParams(
                temperature=0., seed=settings.get('seed', 42), n=1, ignore_eos=True,
                max_tokens=min(settings.get('warmup_tokens', 64), sampling.max_tokens),
            )
            for sample in _warmup_samples(samples, int(settings.get('warmup_prompts', 3))):
                warm_outputs = engine.generate([{'prompt_token_ids': sample['prompt_token_ids']}], warmup_sampling, use_tqdm=False)
                for output in warm_outputs:
                    _append_jsonl(job_dir / 'warmup.jsonl', {
                        'is_warmup': True, 'sample_id': sample['sample_id'],
                        'vllm_request_id': getattr(output, 'request_id', None),
                        'collection_id': collection_id, 'max_tokens': warmup_sampling.max_tokens,
                    })
            torch.cuda.synchronize()
            for repetition, pending in enumerate(pending_by_repetition):
                for generated in _generation_batches(
                    engine, [cell[0] for cell in pending], sampling,
                    batch_size=requested_concurrency, synchronize=torch.cuda.synchronize,
                ):
                    sample, cell_id, path, attempts = pending[generated['index'] - 1]
                    try:
                        if generated['error']:
                            raise generated['error']
                        record = _request_record(
                            method='vanilla_vllm' if job['variant'] == 'ar' else 'dflash',
                            sample=sample, model=settings['target_model'], request_output=generated['output'],
                            tokenizer=tokenizer,
                            client_wall_ms=generated['batch_wall_ms'] if requested_concurrency == 1 else None,
                        )
                    except Exception as exc:
                        record = _failed_record(job['variant'], sample, settings['target_model'],
                                                f'{type(exc).__name__}: {exc}')
                        record['error_traceback'] = generated['error_traceback'] or traceback.format_exc()
                    record.update({
                        'method': job['variant'], 'job_id': job['id'], 'cell_id': cell_id,
                        'execution_identity': run_identity, 'repetition': repetition, 'attempts': attempts,
                        'source_group_id': sample.get('source_group_id'),
                        'measurement': settings.get('measurement', 'throughput'),
                        'batch_id': f'{collection_id}:{repetition}:{generated["batch_id"]}',
                        'collection_id': collection_id,
                        'batch_size': generated['batch_size'], 'batch_wall_ms': generated['batch_wall_ms'],
                        'submitted_concurrency': generated['batch_size'],
                        'vllm_request_id': getattr(generated['output'], 'request_id', None),
                    })
                    atomic_json(path, record)
                    records.append(record)
                    print(f'[{job["variant"]}] {sample["sample_id"]} rep={repetition} status={record["status"]}', flush=True)
    except Exception as exc:
        recorded = {row['cell_id'] for row in records}
        for repetition, pending in enumerate(pending_by_repetition):
            for sample, cell_id, path, attempts in pending:
                if cell_id in recorded:
                    continue
                record = _failed_record(job['variant'], sample, settings['target_model'], f'{type(exc).__name__}: {exc}')
                record.update({'cell_id': cell_id, 'execution_identity': run_identity,
                               'repetition': repetition, 'attempts': attempts,
                               'error_traceback': traceback.format_exc(), 'job_id': job['id']})
                atomic_json(path, record)
                records.append(record)
    finally:
        _shutdown_engine(engine)
        # Worker termination may bypass atexit; all JsonlWriter.add calls flush.
        # Close each trace artifact with an explicit summary after engine shutdown.
        if trace_dir and trace_dir.is_dir():
            for trace_path in trace_dir.glob('worker-*.jsonl'):
                lines = trace_path.read_text(encoding='utf-8').splitlines()
                if not lines:
                    continue
                try:
                    last = json.loads(lines[-1])
                except json.JSONDecodeError:
                    last = {}
                if last.get('record_type') != 'summary':
                    _append_jsonl(trace_path, {
                        'record_type': 'summary', 'count': len(lines), 'job_id': job['id'],
                        'finalized_by': 'generation_coordinator',
                        'final_round_verification_may_be_censored': True,
                    })
        os.environ.clear()
        os.environ.update(original_env)
        sys.path[:] = original_sys_path

    metrics = _method_metrics(records, (job['variant'],)).get(job['variant'], {})
    succeeded = bool(records) and all(row.get('execution_status') == 'success' for row in records)
    result = {'status': 'succeeded' if succeeded else 'failed', 'records': records, 'metrics': metrics,
              'execution_identity': run_identity, 'worker_variant': variant,
              'planned_cells': len(samples) * repetitions,
              'succeeded_cells': sum(row.get('execution_status') == 'success' for row in records),
              'failed_cells': sum(row.get('execution_status') != 'success' for row in records)}
    writer = JsonlWriter(job_dir / 'results.jsonl')
    for record in records:
        writer.add(record)
    writer.finalize({key: value for key, value in result.items() if key != 'records'})
    return result
