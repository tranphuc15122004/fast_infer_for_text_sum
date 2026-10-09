"""CPU contract checks against the spec and vendored DFlash (no checkpoints)."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'externals/dflash')]
from dflash.model import DFlashDraftModel, build_target_layer_ids
from TrainingFree.context_adaptive.attention import draft_block
from TrainingFree.context_adaptive.cache import DraftContextBank
from TrainingFree.context_adaptive.config import AdaptiveConfig
from TrainingFree.context_adaptive.generation import _sample, generate_adaptive, generate_target_only
from TrainingFree.context_adaptive.report import build_report
from TrainingFree.context_adaptive.schema import validate_round
from TrainingFree.context_adaptive.selection import select_context
from TrainingFree.context_adaptive.signals import TargetQueryCapture, target_parent_scores
from TrainingFree.context_adaptive.splits import build_split_manifest
from TrainingFree.context_adaptive.statistics import AdaptiveStatistics, fit_calibration
from TrainingFree.context_adaptive.types import Action, PromptLayout, RoundState, Selection
from TrainingFree.context_adaptive.types import DraftProposal

torch.set_num_threads(1)


def state(length=8, **updates):
    values = dict(round_index=0, logical_length=length, processed_output_tokens=0,
                  remaining_output_tokens=32, parent_entropy=0.2, source_concentration=0.7,
                  history_acceptance=None, ranking_age=0, refresh_required=False)
    values.update(updates)
    return RoundState(**values)


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(13)
    common = dict(vocab_size=32, hidden_size=32, intermediate_size=64,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=16384, eos_token_id=None, pad_token_id=0,
                  attention_dropout=0.0)
    target_config = Qwen3Config(num_hidden_layers=6, **common)
    target_config._attn_implementation = 'eager'
    draft_config = Qwen3Config(num_hidden_layers=2, num_target_layers=6, block_size=16,
                              dflash_config={'target_layer_ids': [1, 3], 'mask_token_id': 0}, **common)
    draft_config._attn_implementation = 'eager'
    return Qwen3ForCausalLM(target_config).eval(), DFlashDraftModel(draft_config).eval()


def layout(length=8):
    source = tuple(range(1, length - 1))
    return PromptLayout(length, source, (0, length - 1), tuple(tuple(source[i:i+2]) for i in range(0, len(source), 2)),
                        'tiny-prompt', True)


@pytest.mark.parametrize('gamma', [3, 7, 11, 15])
@torch.inference_mode()
def test_full_draft_matches_vendored_logits_and_bank(models, gamma):
    target, draft = models
    torch.manual_seed(21)
    features = torch.randn(1, 8, 64)
    bank = DraftContextBank(draft, 40)
    bank.append(features, torch.arange(8))
    block = torch.zeros(1, gamma + 1, dtype=torch.long)
    block[0, 0] = 3
    positions = torch.arange(8, 8 + gamma + 1)
    reference_cache = DynamicCache()
    reference = draft(target_hidden=features, noise_embedding=target.model.embed_tokens(block),
                      position_ids=torch.arange(8 + gamma + 1).reshape(1, -1),
                      past_key_values=reference_cache, use_cache=True, is_causal=False)
    expected_logits = target.lm_head(reference[:, 1:, :])
    captured = []
    handle = target.lm_head.register_forward_hook(lambda m, a, out: captured.append(out))
    try:
        proposal = draft_block(draft, target, bank, Selection((tuple(range(8)),) * 2, 'full', 'target_parent'),
                               block, positions, Action('full', gamma), layout=layout())
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0], expected_logits, atol=1e-5, rtol=1e-4)
    assert torch.equal(proposal.token_ids, expected_logits.argmax(-1))
    for i, storage in enumerate(bank.layer_storage):
        torch.testing.assert_close(storage.key[:, :, :8], reference_cache.layers[i].keys[:, :, :8])
        torch.testing.assert_close(storage.value[:, :, :8], reference_cache.layers[i].values[:, :, :8])


@torch.inference_mode()
def test_sparse_draft_matches_reference_with_original_positions(models):
    target, draft = models
    torch.manual_seed(22)
    features = torch.randn(1, 9001, 64)
    chosen = torch.tensor([0, 100, 3000, 9000])
    bank = DraftContextBank(draft, 9020)
    bank.append(features, torch.arange(9001))
    block = torch.tensor([[3, 0, 0, 0]])
    positions = torch.arange(9001, 9005)
    reference = draft(target_hidden=features[:, chosen], noise_embedding=target.model.embed_tokens(block),
                      position_ids=torch.cat((chosen, positions)).reshape(1, -1),
                      past_key_values=DynamicCache(), use_cache=True, is_causal=False)
    expected_logits = target.lm_head(reference[:, 1:, :])
    captured = []
    handle = target.lm_head.register_forward_hook(lambda m, a, out: captured.append(out))
    try:
        proposal = draft_block(draft, target, bank, Selection((tuple(chosen.tolist()),) * 2, 4, 'recent_only'),
                               block, positions, Action(4, 3), layout=layout(9001))
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0], expected_logits, atol=1e-5, rtol=1e-4)
    assert torch.equal(proposal.token_ids, expected_logits.argmax(-1))


@pytest.mark.parametrize('gamma', [3, 7, 11, 15])
@pytest.mark.parametrize('selector', ['target_parent', 'draft_refresh', 'recent_only'])
@torch.inference_mode()
def test_greedy_rollout_matches_ar_and_accounts_for_cap(models, gamma, selector):
    target, draft = models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    config = AdaptiveConfig(budgets=(4, 'full'), gammas=(gamma,), selector=selector,
                            source_anchors=0, recent_output=2, source_chunk_size=2)
    ar = generate_target_only(target, ids, max_new_tokens=21, stop_token_ids=())
    # Fixed sparse pair exercises actual intervention; draft_refresh forces periodic full rounds.
    result = generate_adaptive(target, draft, ids, None, layout(), config,
                               max_new_tokens=21, stop_token_ids=(), variant='best_fixed_pair',
                               fixed_budget=4, fixed_gamma=gamma, gamma_reference=gamma)
    assert torch.equal(result.output_ids, ar.output_ids)
    assert result.output_tokens == 21
    assert result.counters['output_accounting_valid']
    assert all(not validate_round(row) for row in result.rounds)


def test_sampling_supports_verification_block_logits():
    samples = _sample(torch.zeros(1, 4, 32), temperature=0.7)
    assert samples.shape == (1, 4)


@torch.inference_mode()
def test_ar_rounds_pass_the_same_cli_schema(models):
    target, _ = models
    ar = generate_target_only(target, torch.tensor([[1, 2, 3]]), max_new_tokens=4, stop_token_ids=())
    errors = []
    for row in ar.rounds:
        row.update(run_id='review', sample_id='one', dataset='tiny', split='dev', repetition=0)
        errors.extend(validate_round(row))
    assert not errors, errors


def test_valid_dflash_config_passes_model_preflight(models, tmp_path):
    target, draft = models
    assert build_target_layer_ids(6, 2) == [1, 3]
    target_dir, draft_dir = tmp_path / 'target', tmp_path / 'draft'
    target.config.save_pretrained(target_dir)
    draft.config.save_pretrained(draft_dir)
    data_file = tmp_path / 'data.jsonl'
    data_file.write_text(json.dumps({'id': 'one', 'prompt': 'hello', 'context': 'hello', 'dataset': 'tiny'}) + '\n')
    spec = importlib.util.spec_from_file_location('cad_review_cli', ROOT / 'scripts/infer_context_adaptive_dflash.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module._parser().parse_args(['--phase', 'preflight', '--data-file', str(data_file),
            '--target-model', str(target_dir), '--draft-model', str(draft_dir), '--model-check',
            '--fixed-gamma', '3', '--gamma-reference', '3', '--gammas', '3'])
    assert module._preflight(args, tmp_path / 'out', [data_file], None) == 0


def test_unseen_context_bucket_has_no_cost_prior():
    action = Action('full', 3)
    cal = {'schema_version': 'cadflash.calibration.v1', 'entropy_cutpoints': [0.1, 0.2, 0.3],
           'prefix_priors': {}, 'cost_priors': {
               f'{action.key()}::context=1::refresh=0': {'action_cost_ms': 1.0, 'repetitions': 3},
               f'{action.key()}::global': {'action_cost_ms': 1.0, 'repetitions': 3}},
           'controller_cost_priors': {'joint': {'choose_ms': 0.1}}}
    statistics = AdaptiveStatistics(cal)
    assert statistics.cost_ms(action, state(length=10000), 'joint') is None


def test_report_includes_joint_against_strong_controls():
    variants = ['ar', 'dflash_full_fixed', 'best_fixed_pair', 'a_only',
                'b_only_history', 'b_only_entropy', 'independent_ab', 'joint']
    rows = [dict(type='sample', status='ok', variant=variant, dataset='tiny', sample_id='one',
                 repetition=0, output_scope='natural_eos', e2e_ms=10.0, generation_temperature=0.0,
                 output_ids_hash='same', run_id='review', source_group_id='source',
                 phase='dev', statistics_update_mode='online') for variant in variants]
    report = build_report(rows, bootstrap_samples=0)
    pairs = {(row['baseline_variant'], row['candidate_variant']) for row in report['comparisons']}
    assert {
        ('dflash_full_fixed', 'a_only'),
        ('dflash_full_fixed', 'b_only_history'),
        ('dflash_full_fixed', 'b_only_entropy'),
        ('best_fixed_pair', 'joint'),
        ('independent_ab', 'joint'),
    } <= pairs, pairs


def test_prefix_prior_does_not_count_greedy_timing_repetitions():
    action = Action('full', 3)
    rows = [dict(status='ok', requested_budget='full', gamma_executed=3, accepted_candidates=accepted,
                 state_hash=f'state-{index}', logical_length=8, action_cost_ms=1.0,
                 parent_entropy=0.2, source_concentration=0.6, history_acceptance=None)
            for index, accepted in enumerate([1, 3]) for _ in range(3)]
    cal = fit_calibration(rows, signature={})
    prior = cal['prefix_priors'][f'{action.key()}::global']
    assert prior['unique_state_count'] == 2
    assert prior['survival'] == [1.0, 0.5, 0.5]


def test_chunk_selection_keeps_protection_and_whole_chunks():
    prompt_layout = PromptLayout(5, (0, 1, 2, 3, 4), (), ((0, 1), (2, 3), (4,)), 'layout', True)
    selected = select_context(state(length=5), prompt_layout, [(1, 2, 0)], 4,
                              processed_output_start=5, bank_length=5, source_anchors=2, recent_output=0)
    assert selected.positions_by_layer == ((0, 1, 2, 3),)
    assert select_context(state(length=5), prompt_layout, [(1, 2, 0)], 1,
                          processed_output_start=5, bank_length=5, source_anchors=2, recent_output=0) is None


def test_source_queries_share_split_and_exposure_stays_in_dev():
    rows = [dict(id=f'tiny::{i}', dataset='tiny', context=f'source-{i}', source_split='test', source_index=str(i))
            for i in range(10)]
    rows.append(dict(id='tiny::query-extra', dataset='tiny', context='source-0', source_split='test', source_index='0'))
    exposure = [{'payload': {'samples': [{'sample_id': 'tiny::0'}]}}]
    manifest = build_split_manifest(rows, exposure)
    assert manifest['status'] == 'complete'
    assert manifest['record_split']['tiny::0'] == manifest['record_split']['tiny::query-extra'] == 'dev'


def cli_module():
    spec = importlib.util.spec_from_file_location('cad_review_cli', ROOT / 'scripts/infer_context_adaptive_dflash.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launcher_args(tmp_path, forwarded, overrides=None):
    master = tmp_path / 'master.env'
    master.write_text('FI_OFFLINE=1\nRUN_MODE=full\nRUN_SAMPLES=2\nRUN_TEMPERATURE=0\n'
                      'MODEL_TARGET=/fixture/master-target\nMODEL_DFLASH_DRAFT=/fixture/master-draft\n'
                      'DATA_INPUT=/fixture/master-data.jsonl\n')
    stub = tmp_path / 'capture-python'
    stub.write_text(f'#!{sys.executable}\nimport json,subprocess,sys\n'
                    'if sys.argv[1] == "-c":\n'
                    '    sys.exit(subprocess.call([sys.executable, *sys.argv[1:]]))\n'
                    'print(json.dumps(sys.argv[1:]))\n')
    stub.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('CAD_', 'FAST_INFER_')) and key not in {
               'MODEL_TARGET', 'MODEL_DFLASH_DRAFT', 'TARGET_MODEL', 'DRAFT_MODEL', 'DATA_FILE',
               'DATA_INPUT', 'RUN_MODE', 'RUN_SAMPLES', 'RUN_TEMPERATURE', 'MAX_SAMPLES',
               'TEMPERATURE', 'SMOKE', 'FULL'}}
    env.update(FAST_INFER_MASTER_CONFIG=str(master), FAST_INFER_PYTHON=str(stub), CUDA_VISIBLE_DEVICES='')
    env.update(overrides or {})
    result = subprocess.run(['bash', str(ROOT / 'scripts/runners/run_context_adaptive_dflash.sh'), *forwarded],
                            cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('phase', ['dev', 'test'])
def test_launcher_does_not_inject_smoke_sample_cap_into_cli_phase(tmp_path, phase):
    argv = launcher_args(tmp_path, ['--phase', phase])
    assert '--max-samples' not in argv, argv


def test_launcher_smoke_keeps_explicit_sample_cap(tmp_path):
    argv = launcher_args(tmp_path, ['--phase', 'smoke'])
    assert argv[argv.index('--max-samples') + 1] == '2'


@pytest.mark.parametrize(('env_name', 'cli_option', 'caller_value'), [
    ('MODEL_TARGET', '--target-model', '/fixture/caller-target'),
    ('MODEL_DFLASH_DRAFT', '--draft-model', '/fixture/caller-draft'),
    ('DATA_INPUT', '--data-file', '/fixture/caller-data.jsonl'),
    ('RUN_TEMPERATURE', '--temperature', '0.7'),
])
def test_launcher_preserves_canonical_caller_overrides(tmp_path, env_name, cli_option, caller_value):
    argv = launcher_args(tmp_path, ['--phase', 'preflight'], {env_name: caller_value})
    assert argv[argv.index(cli_option) + 1] == caller_value, argv


@torch.inference_mode()
def test_target_query_capture_matches_native_eager_attention(models):
    target, _ = models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    cache = DynamicCache()
    capture = TargetQueryCapture(target, 3)
    try:
        capture.enable(last_only=True)
        result = target(ids, past_key_values=cache, use_cache=True, output_attentions=True)
        query = capture.disable()
        scores, _ = target_parent_scores(query, cache, 3, 0, 7, layout().source_chunks,
                                         head_groups=2, scaling=target.model.layers[3].self_attn.scaling)
    finally:
        capture.close()
    reference = result.attentions[3][:, :, -1, :].mean(dim=(0, 1))
    for position, value in scores.items():
        assert value == pytest.approx(float(reference[position]), abs=1e-6)


def monotone_target_and_proposal(models, monkeypatch, *, rejection=None, eos_proposal=False):
    from types import SimpleNamespace
    import TrainingFree.context_adaptive.generation as generation
    target, draft = models
    def forward(ids, *, past_key_values, output_hidden_states=False, logits_to_keep=0, **kwargs):
        count = ids.shape[1]
        logits = torch.full((1, count, 32), -50.0)
        logits.scatter_(2, ((ids + 1) % 32).unsqueeze(-1), 50.0)
        for layer in range(6):
            past_key_values.update(torch.zeros(1, 2, count, 8), torch.zeros(1, 2, count, 8), layer)
        states = tuple(torch.zeros(1, count, 32) for _ in range(7)) if output_hidden_states else None
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits,
                               hidden_states=states, past_key_values=past_key_values)
    def proposal(draft, target, bank, selection, block, positions, action, **kwargs):
        tokens = (block[0, 0] + torch.arange(1, action.gamma + 1)).remainder(32).reshape(1, -1)
        if rejection is not None:
            tokens[0, rejection] = 6 if eos_proposal else (int(tokens[0, rejection]) + 10) % 32
        return DraftProposal(tokens)
    monkeypatch.setattr(target, 'forward', forward)
    monkeypatch.setattr(generation, 'draft_block', proposal)
    return target, draft


@pytest.mark.parametrize('cap', [1, 2, 4, 5, 17])
def test_all_accepted_transaction_handles_output_caps(models, monkeypatch, cap):
    target, draft = monotone_target_and_proposal(models, monkeypatch)
    config = AdaptiveConfig(budgets=('full',), gammas=(3,), selector='recent_only')
    result = generate_adaptive(target, draft, torch.tensor([[0, 1]]), None, layout(2), config,
                               max_new_tokens=cap, fixed_gamma=3, gamma_reference=3, stop_token_ids=())
    assert result.output_ids.tolist() == list(range(2, cap + 2))
    assert result.counters['output_accounting_valid']


@pytest.mark.parametrize(('stop_id', 'rejection', 'eos_proposal', 'expected'), [
    (2, None, False, [2]),
    (4, None, False, [2, 3, 4]),
    (3, 0, False, [2, 3]),
    (6, 0, True, [2, 3, 4, 5, 6]),
])
def test_eos_transaction_keeps_only_target_valid_prefix(models, monkeypatch, stop_id, rejection, eos_proposal, expected):
    target, draft = monotone_target_and_proposal(models, monkeypatch, rejection=rejection, eos_proposal=eos_proposal)
    config = AdaptiveConfig(budgets=('full',), gammas=(3,), selector='recent_only')
    result = generate_adaptive(target, draft, torch.tensor([[0, 1]]), None, layout(2), config,
                               max_new_tokens=17, fixed_gamma=3, gamma_reference=3, stop_token_ids=(stop_id,))
    assert result.output_ids.tolist() == expected
    assert result.counters['stopped_by'] == 'eos'
    assert result.counters['output_accounting_valid']
    assert all(not validate_round(row) for row in result.rounds)


def cpu_runtime(models):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    engine = Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, 'hello': 2}, unk_token='[UNK]'))
    engine.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token='[UNK]', pad_token='[PAD]')
    return dict(target=models[0], draft=None, tokenizer=tokenizer, dtype='torch.float32',
                attention_backend='eager', device='cpu', target_signature='tiny-target',
                draft_signature=None, tokenizer_signature='tiny-tokenizer',
                runtime_signature={'signature': 'review-cpu'}, target_asset=None, draft_asset=None)


def pipeline_fixture(models, monkeypatch, tmp_path, cap):
    import TrainingFree.context_adaptive.benchmark as benchmark
    runtime = cpu_runtime(models)
    monkeypatch.setattr(benchmark, 'load_runtime', lambda *args, **kwargs: runtime)
    data = tmp_path / 'dataset.jsonl'
    data.write_text(''.join(json.dumps({'id': 'same', 'prompt': 'hello', 'context': 'hello', 'dataset': dataset}) + '\n'
                            for dataset in ['d1', 'd2']))
    module = cli_module()
    args = module._parser().parse_args(['--phase', 'smoke', '--variant', 'ar', '--data-file', str(data),
            '--target-model', '/fixture/target', '--max-new-tokens', str(cap), '--repetitions', '1',
            '--warmup-runs', '0', '--fixed-gamma', '3', '--gamma-reference', '3', '--gammas', '3'])
    args.fixed_budget = 'full'
    config = module._config(args, phase='smoke')
    return module, args, config, data, tmp_path / 'run'


def test_resume_does_not_skip_same_id_from_another_dataset(models, monkeypatch, tmp_path):
    module, args, config, data, output = pipeline_fixture(models, monkeypatch, tmp_path, cap=1)
    assert module._run_model_phase(args, 'smoke', 'ar', output, [data], config) == 0
    requests = output / 'smoke/ar/online/rep_0/requests.jsonl'
    samples = [json.loads(line) for line in requests.read_text().splitlines() if json.loads(line).get('type') == 'sample']
    assert len(samples) == 2
    # Simulate interruption after d1 was written, before d2 completed.
    requests.write_text(json.dumps(samples[0]) + '\n')
    args.resume = True
    module._run_model_phase(args, 'smoke', 'ar', output, [data], config)
    samples = [json.loads(line) for line in requests.read_text().splitlines() if json.loads(line).get('type') == 'sample']
    assert {row['dataset'] for row in samples} == {'d1', 'd2'}, samples


def test_pipeline_returns_failure_when_every_request_errors(models, monkeypatch, tmp_path):
    module, args, config, data, output = pipeline_fixture(models, monkeypatch, tmp_path, cap=4)
    import TrainingFree.context_adaptive.generation as generation
    monkeypatch.setattr(generation, '_sample', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('fixture failure')))
    return_code = module._run_model_phase(args, 'smoke', 'ar', output, [data], config)
    requests = output / 'smoke/ar/online/rep_0/requests.jsonl'
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    assert rows[-1]['status'] == 'partial'
    assert rows[-1]['successes'] == 0
    assert rows[-1]['errors'] == 2
    assert return_code != 0, rows[-1]


def test_id_only_exposure_resolves_ids_emitted_by_existing_probes(tmp_path):
    from TrainingFree.context_adaptive.benchmark import load_corpus, split_records
    data = tmp_path / 'data.jsonl'
    data.write_text(''.join(json.dumps({'id': f'sample-{i}', 'dataset': 'tiny',
                                       'prompt': f'source-{i}', 'context': f'source-{i}'}) + '\n'
                            for i in range(10)))
    records = split_records(load_corpus(data_file=data, data_dir=None))
    initial = build_split_manifest(records)
    selected_key = next(key for key, value in initial['record_split'].items() if value == 'test')
    raw_id = selected_key.split('::', 1)[1]
    # Existing probes emit raw sample_id; source_sha256 is optional for an ID registry.
    manifest = build_split_manifest(records, [{'payload': {'samples': [{'sample_id': raw_id}]}}])
    assert manifest['record_split'][selected_key] == 'dev', manifest['unresolved_exposure_sample_ids']
