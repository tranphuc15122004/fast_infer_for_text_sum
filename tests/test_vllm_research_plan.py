"""Kiểm chứng ma trận và quan hệ nhân quả trước khi khởi tạo GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_plan_has_unique_order_independent_ids_and_shared_native_references():
    from Benchmark.vllm_research.plan import build_jobs

    jobs = build_jobs(['A', 'B', 'A3', 'B3'])
    assert len({job.id for job in jobs}) == len(jobs)
    assert {job.id for job in jobs} == {job.id for job in build_jobs(['B', 'A'])}
    assert len([job for job in jobs if job.variant == 'ar']) == 1
    assert len([job for job in jobs if job.variant == 'dflash_full']) == 1
    assert len([job for job in jobs if job.variant == 'full_adapter']) == 1


def test_a1_and_a2_cover_registered_dev_matrix_without_user_a3_grid_filtering():
    from Benchmark.vllm_research.plan import build_jobs
    from TrainingFree.context_interventions.matrix import a1_matrix, a2_matrix

    jobs = build_jobs(['A1', 'A2'], budgets=(2048,))
    a1 = [job for job in jobs if job.experiment == 'A1']
    matrix = a1_matrix('dev')
    assert len(a1) == 52
    assert {job.parameters['control'] for job in a1 if 'control' in job.parameters} == set(matrix['controls'])
    observed = {(job.parameters['attention_scoring'], job.parameters['deletion_count'], job.parameters['deletion_side']) for job in a1 if 'attention_scoring' in job.parameters}
    assert observed == {(score, count, side) for score in matrix['attention_scoring'] for count in matrix['attention_deletions'] for side in ('low', 'high')}
    a2 = [job for job in jobs if job.experiment == 'A2']
    matrix2 = a2_matrix('dev')
    assert len(a2) == 21
    assert {job.parameters['budget'] for job in a2} == set(matrix2['budgets'])
    assert {job.parameters['seed'] for job in a2 if job.parameters['selector'] == 'random'} == set(matrix2['random_seeds'])
    oracle = next(job for job in a2 if job.parameters['selector'] == 'current_dflash_oracle')
    assert oracle.parameters['diagnostic_only'] is True


def test_a3_replay_has_ranking_age_and_e2e_has_real_dense_refresh():
    from Benchmark.vllm_research.plan import build_jobs
    from TrainingFree.context_interventions.matrix import a3_matrix

    jobs = build_jobs(['A3'])
    replay = [job for job in jobs if job.kind == 'replay']
    expected = a3_matrix('dev')
    assert {(j.parameters['budget'], j.parameters['ranking_age']) for j in replay} == {(budget, age) for budget in expected['budgets'] for age in expected['ranking_ages']}
    online = [job for job in jobs if job.experiment == 'A3' and job.kind == 'generation']
    assert {j.parameters['refresh_period'] for j in online} == {2, 4, 8, 16}
    assert all(j.parameters['dense_refresh_proposal_is_verified'] for j in online)
    assert all(j.parameters['target_context'] == 'full' for j in online)
    assert all(not j.dependencies for j in online)


def test_b3_drafts_15_before_k_b4_selects_gamma_before_draft():
    from Benchmark.vllm_research.plan import build_jobs

    jobs = build_jobs(['B'])
    prefixes = [job for job in jobs if job.experiment == 'B1-V']
    assert {j.parameters['k'] for j in prefixes} == {3, 7, 11, 15}
    assert all(j.parameters['draft_gamma'] == 15 for j in prefixes)
    b3 = [j for j in jobs if j.capability == 'adaptive_verification']
    assert b3
    assert all(j.parameters['decision_timing'] == 'after_draft_before_verify' for j in b3)
    assert all(j.parameters['draft_gamma'] == 15 for j in b3)
    b4 = [j for j in jobs if j.capability == 'adaptive_draft']
    assert b4
    assert all(j.parameters['decision_timing'] == 'before_draft' for j in b4)
    assert all(j.parameters['signals_round_offset'] == -1 for j in b4)
    assert all(j.dependencies for j in b3 + b4)
    assert all(j.parameters.get('requires_positive_result') is False for j in b3 + b4)


def test_independent_requests_expose_missing_dependencies_and_keep_fixed_controls():
    from Benchmark.vllm_research.plan import build_jobs, missing_dependencies

    jobs = build_jobs(['B3'])
    missing = missing_dependencies(jobs)
    adaptive = [j for j in jobs if j.capability == 'adaptive_verification']
    assert all(j.id in missing for j in adaptive)
    assert {j.parameters.get('k') for j in jobs if j.variant.startswith('fixed_k_')} == {3, 7, 11, 15}
    assert all(not j.dependencies for j in jobs if j.variant.startswith('fixed_k_'))
    assert not missing_dependencies(build_jobs(['A', 'B']))


@pytest.mark.parametrize('arguments', [dict(experiments=['unknown']), dict(experiments=['B1'], gammas=(5,)), dict(experiments=['A3'], budgets=(0,)), dict(experiments=['A3'], periods=(0,)), dict(experiments=['A3'], selectors=('invented',))])
def test_invalid_global_configuration_is_rejected(arguments):
    from Benchmark.vllm_research.plan import build_jobs

    with pytest.raises(ValueError):
        build_jobs(**arguments)


def _candidate(state='state0', round_index=0, progress=0):
    return {'type': 'candidate_signals', 'state_id': state, 'dataset': 'qmsum',
            'source_group_id': 'document0', 'sample_id': 'sample0',
            'round_index': round_index, 'processed_output_tokens': progress,
            'prompt_tokens': 8192, 'context_tokens': 8192 + progress,
            'accepted_candidates': 3, 'history_ema': None, 'history_missing': True,
            'parent_signal': {'normalized_entropy': .3, 'confidence': .6},
            'draft_uncertainty': {'normalized_entropy': [.4] * 15, 'confidence': [.5] * 15}}


def test_analysis_missing_evidence_never_fabricates_fit():
    from Benchmark.vllm_research.plan import build_jobs
    from Benchmark.vllm_research.analysis import execute_analysis

    job = next(j for j in build_jobs(['B2']) if j.variant == 'survival_fit_R')
    result = execute_analysis(job, {})
    assert result['status'] == 'dependency_missing'
    assert 'payload' not in result
    result = execute_analysis(job, {job.dependencies[0]: [{'type': 'summary'}]})
    assert result['status'] == 'dependency_missing'
    result = execute_analysis(job, {job.dependencies[0]: [{'type': 'candidate_signals'}]})
    assert result['status'] == 'failed'
    assert 'payload' not in result


def test_survival_analysis_serializes_actual_fit_without_positive_gate():
    from Benchmark.vllm_research.plan import build_jobs
    from Benchmark.vllm_research.analysis import execute_analysis

    job = next(j for j in build_jobs(['B2']) if j.variant == 'survival_fit_R-DH')
    result = execute_analysis(job, {job.dependencies[0]: [_candidate()]})
    assert result['status'] == 'succeeded'
    assert result['payload']['training_groups'] == ['document0']
    assert result['payload']['variant'] == 'R-DH'
    assert len(result['payload']['models']) == 15


def test_b4_fit_needs_actual_adjacent_round_and_measured_gamma_costs():
    from Benchmark.vllm_research.plan import build_jobs
    from Benchmark.vllm_research.analysis import execute_analysis

    job = next(j for j in build_jobs(['B4']) if j.variant == 'draft_cost_fit_R-A-LDH')
    artifacts = {job.dependencies[0]: [_candidate('s0', 0, 0), _candidate('s1', 1, 4)]}
    for gamma, dependency in zip((3, 7, 11, 15), job.dependencies[1:]):
        artifacts[dependency] = [{'type': 'draft_shape_action', 'state_id': 's1', 'gamma': gamma,
                                  'repetition_split': 'selection', 'accepted_candidates': 3,
                                  'round_ms': 2 + gamma / 10}]
    result = execute_analysis(job, artifacts)
    assert result['status'] == 'succeeded'
    assert result['payload']['training_groups'] == ['document0']
    artifacts[job.dependencies[0]][1]['round_index'] = 2
    result = execute_analysis(job, artifacts)
    assert result['status'] == 'dependency_missing'
