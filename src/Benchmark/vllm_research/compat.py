"""Capability/version checks và bootstrap plugin cho worker subprocess."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile


def inspect_capabilities(*, version: str | None = None) -> dict:
    if version is None:
        try:
            version = importlib.metadata.version('vllm')
        except Exception:
            try:
                import vllm
                version = getattr(vllm, '__version__', None)
            except Exception:
                version = None
    supported = version == '0.30.0'
    source_reason = None
    if supported:
        try:
            dist = importlib.metadata.distribution('vllm')
            for relative, symbol in (
                ('vllm/v1/spec_decode/dflash.py', 'set_inputs_first_pass'),
                ('vllm/v1/worker/gpu_model_runner.py', 'propose_draft_token_ids'),
                ('vllm/model_executor/models/qwen3_dflash.py', 'DFlashQwen3Attention'),
            ):
                tree = ast.parse(Path(dist.locate_file(relative)).read_text())
                if not any(getattr(node, 'name', None) == symbol for node in ast.walk(tree)):
                    raise ValueError(f'missing internal API {symbol}')
        except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
            supported = False
            source_reason = str(exc)
    reason = source_reason or ('vLLM 0.30 source compatible; requires Qwen3 DFlash, TP=1, eager, FLASH_ATTN, nonquantized full-attention KV, no attention sinks' if supported else f'research worker requires exactly vLLM 0.30.0; installed={version}')
    result = {name: {'available': supported, 'reason': reason, 'gpu_verified': False,
                     'implementation': 'worker_adapter', 'vllm_version': version}
              for name in ('a3', 'full_adapter', 'b3_fixed_k', 'b3', 'b4', 'trace')}
    for name in ('a1', 'a2', 'a3_replay', 'b1', 'b1_v'):
        result[name] = {'available': False, 'reason': 'immutable snapshot/target replay adapter has not been implemented', 'gpu_verified': False}
    result['native'] = {'available': version is not None, 'reason': 'public vLLM API; research patch disabled', 'gpu_verified': False, 'vllm_version': version}
    return result


def configure_worker_variant(variant: str, parameters: dict | None = None, trace_dir: str | None = None) -> dict:
    parameters = dict(parameters or {})
    variant = variant.lower()
    if variant == 'native' and trace_dir is None:
        return {'env': {}, 'engine_overrides': {}}
    if variant not in {'native', 'full_adapter', 'a3', 'b3_fixed_k', 'b3', 'b4', 'trace'}:
        raise ValueError(f'unsupported vLLM research worker variant: {variant}')
    layouts = parameters.get('prompt_layouts', parameters.get('sample_layouts', {}))
    layout_file = parameters.get('prompt_layouts_file')
    if isinstance(layouts, str):
        layout_file, layouts = layouts, {}
    if layout_file:
        layout_file = str(Path(layout_file).expanduser().resolve())
        if not Path(layout_file).is_file():
            raise ValueError(f'prompt layout artifact not found: {layout_file}')
        parameters['prompt_layouts_file'] = layout_file
        layouts = {}
    parameters['prompt_layouts'] = layouts
    if variant == 'a3':
        if not layouts and not layout_file:
            raise ValueError('A3 requires exact per-prompt source/global layouts')
        if parameters.get('selector') not in {'recency', 'previous_dflash'}:
            raise ValueError('A3 worker supports recency and previous_dflash selectors')
        budget = parameters.get('budget', 0)
        if (budget != 'full' and int(budget) < 1) or int(parameters.get('refresh_period', 1)) < 1:
            raise ValueError('budget and refresh_period must be positive')
    if variant == 'b3_fixed_k' and int(parameters.get('k', 0)) not in (3, 7, 11, 15):
        raise ValueError('fixed B3 k must be one of 3/7/11/15')
    if variant == 'b3':
        model = parameters.get('model') or {}
        if model.get('variant') in {'R-H', 'R-C', 'R-HA', 'R-PreAll', 'R-DAll', 'R-DAll-noDH'}:
            raise ValueError('B3 models requiring parent entropy/confidence need a target-logit collector')
        if not model or not parameters.get('cost_profiles'):
            raise ValueError('B3 requires a frozen survival model and measured cost_profiles')
        allowed_b3_variants = {
            'R', 'R-A', 'R-DH', 'R-DC', 'R-A-DH', 'R-A-DC',
            'R-Hazard-Pos', 'R-Hazard-Present', 'R-Hazard-Prefix',
            'R-Hazard-Present-Future',
            'R-Hazard-Suffix-Margin', 'R-Hazard-Suffix-Entropy',
        }
        if model.get('variant') not in allowed_b3_variants:
            raise ValueError('B3 worker does not implement the signals required by this survival variant')
    if variant == 'b4' and not parameters.get('model'):
        raise ValueError('B4 requires a frozen lagged cost model')
    if variant == 'b4':
        model = parameters['model']
        if model.get('schema_version') != 'dflash-speculation-b.b4-cost-model.v1':
            raise ValueError('B4 requires the frozen dflash-speculation-b.b4-cost-model.v1 schema')
    bootstrap = Path(tempfile.mkdtemp(prefix='fast-infer-vllm-research-'))
    metadata = bootstrap / 'fast_infer_vllm_research-0.1.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: fast-infer-vllm-research\nVersion: 0.1.0\n')
    (metadata / 'entry_points.txt').write_text('[vllm.general_plugins]\nfast_infer_research = Benchmark.vllm_research.plugin:register\n')
    (metadata / 'RECORD').write_text('')
    config_path = bootstrap / 'config.json'
    config_path.write_text(json.dumps({'variant': variant, 'parameters': parameters, 'trace_dir': trace_dir}))
    source = str(Path(__file__).resolve().parents[2])
    plugins = [p for p in os.environ.get('VLLM_PLUGINS', 'fast_infer_domino').split(',') if p]
    plugins.append('fast_infer_research')
    env = {'FAST_INFER_VLLM_RESEARCH_CONFIG': str(config_path),
           'VLLM_PLUGINS': ','.join(dict.fromkeys(plugins)),
           'PYTHONPATH': os.pathsep.join((str(bootstrap), source, os.environ.get('PYTHONPATH', ''))),
           'VLLM_USE_V2_MODEL_RUNNER': '0'}
    overrides = {'enforce_eager': True, 'async_scheduling': False,
                 'tensor_parallel_size': 1, 'attention_config': {'backend': 'FLASH_ATTN'}}
    if variant in {'b3', 'b3_fixed_k', 'b4'}:
        overrides['max_num_seqs'] = 1
    return {'env': env, 'engine_overrides': overrides}
