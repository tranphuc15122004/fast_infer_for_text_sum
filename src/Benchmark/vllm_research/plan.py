"""Lập toàn bộ ma trận nghiên cứu trên CPU, không khởi tạo engine/kernel."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from TrainingFree.context_interventions.matrix import a1_matrix, a2_matrix, a3_matrix
from TrainingFree.speculation_experiments.protocol import SUPPORTED_GAMMAS
from TrainingFree.speculation_experiments.survival import DEFAULT_VARIANTS, SURVIVAL_VARIANTS
from TrainingFree.speculation_experiments.lagged import _COST_VARIANTS


@dataclass(frozen=True)
class ResearchJob:
    id: str
    experiment: str
    variant: str
    kind: str
    parameters: dict[str, Any]
    dependencies: tuple[str, ...]
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _job(experiment: str, variant: str, kind: str, capability: str,
         parameters: Mapping[str, Any] | None = None,
         dependencies: Iterable[str] = ()) -> ResearchJob:
    params = dict(parameters or {})
    signature = json.dumps([experiment, variant, kind, capability, params], sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
    return ResearchJob(f'{experiment.lower()}:{variant}:{digest}', experiment, variant, kind,
                       params, tuple(dependencies), capability)


def _experiments(values: Iterable[str]) -> set[str]:
    if isinstance(values, str):
        values = (values,)
    aliases = {'A': ('A1', 'A2', 'A3'), 'B': ('B0', 'B1', 'B1-V', 'B2', 'B3', 'B4'),
               'ALL': ('A1', 'A2', 'A3', 'B0', 'B1', 'B1-V', 'B2', 'B3', 'B4'),
               'E2E': ('A3-E2E', 'B3', 'B4'), 'B1V': ('B1-V',)}
    valid = {'A1', 'A2', 'A3', 'A3-REPLAY', 'A3-E2E', 'B0', 'B1', 'B1-V', 'B2', 'B3', 'B4'}
    expanded: set[str] = set()
    for value in values:
        for item in str(value).split(','):
            name = item.strip().upper()
            expanded.update(aliases.get(name, (name,)))
    if not expanded or expanded - valid:
        raise ValueError(f'unknown or empty experiment selection: {sorted(expanded - valid)}')
    return expanded


def missing_dependencies(jobs: Iterable[ResearchJob]) -> dict[str, tuple[str, ...]]:
    """Các prerequisite không được chọn vẫn hiện rõ trong plan, không tự bỏ policy."""
    jobs = tuple(jobs)
    ids = {job.id for job in jobs}
    return {job.id: missing for job in jobs
            if (missing := tuple(dep for dep in job.dependencies if dep not in ids))}


def build_jobs(experiments: Iterable[str], *, selectors: Iterable[str] = ('previous_dflash',),
               budgets: Iterable[int | str] = (2048,), periods: Iterable[int] = (2, 4, 8, 16),
               gammas: Iterable[int] = SUPPORTED_GAMMAS) -> list[ResearchJob]:
    """Một job là một arm/config; executor mở rộng tất cả sample và repetition.

    A1/A2 và replay A3 lấy nguyên grid dev đã đăng ký; budgets/periods/selectors
    chọn grid A3 online. Horizon/sample/measurement do executor gắn vào cell ID.
    Việc fit được và kết quả dương là hai điều khác nhau: không có success gate.
    """
    selected = _experiments(experiments)
    selectors = tuple(dict.fromkeys(selectors))
    budgets = tuple(dict.fromkeys(budgets))
    periods = tuple(dict.fromkeys(periods))
    gammas = tuple(sorted(set(gammas)))
    online_selectors = {'previous_dflash', 'target_parent', 'random', 'recency'}
    if not selectors or set(selectors) - online_selectors:
        raise ValueError('online selectors must be previous_dflash, target_parent, random or recency')
    if not budgets or any(value != 'full' and (isinstance(value, bool) or not isinstance(value, int) or value <= 0) for value in budgets):
        raise ValueError('budgets must be positive integer counts or full')
    if not periods or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in periods):
        raise ValueError('refresh periods must be positive integers')
    if not gammas or any(isinstance(value, bool) or value not in SUPPORTED_GAMMAS for value in gammas):
        raise ValueError(f'gammas must belong to {SUPPORTED_GAMMAS}')
    jobs: dict[str, ResearchJob] = {}

    def add(job: ResearchJob) -> ResearchJob:
        jobs.setdefault(job.id, job)
        return job

    def generation(variant: str, params: dict[str, Any] | None = None) -> ResearchJob:
        return _job('E2E', variant, 'generation', 'native_generation', params)

    # E2E controls share the same horizon and measurement namespace in executor.
    if selected & {'A3', 'A3-E2E', 'B0', 'B3', 'B4'}:
        add(generation('ar', {'draft_gamma': 0}))
        add(generation('dflash_full', {'draft_gamma': 15, 'target_context': 'full', 'draft_context': 'full'}))
    if selected & {'A3', 'A3-E2E'}:
        add(_job('E2E', 'full_adapter', 'generation', 'full_adapter',
                 {'draft_gamma': 15, 'budget': 'full', 'selector': 'full', 'target_context': 'full'}))
        for selector in selectors:
            for budget in budgets:
                for period in periods:
                    add(_job('A3', f'{selector}_B{budget}_period{period}', 'generation', 'a3',
                             {'selector': selector, 'budget': budget, 'refresh_period': period,
                              'draft_gamma': 15, 'target_context': 'full',
                              'ranking_source': 'last_verified_dense_draft' if selector == 'previous_dflash' else selector,
                              'dense_refresh_proposal_is_verified': True,
                              'extra_discarded_dense_forward': False}))
    if 'A1' in selected:
        grid = a1_matrix('dev')
        for control in grid['controls']:
            add(_job('A1', control, 'replay', 'same_prefix_replay', {'control': control, 'grid_phase': 'dev'}))
        for count in grid['matched_deletions']:
            for region in grid['matched_regions']:
                add(_job('A1', f'delete_{region}_D{count}', 'replay', 'same_prefix_replay',
                         {'region': region, 'deletion_count': count, 'seed': 42, 'grid_phase': 'dev'}))
        for scoring in grid['attention_scoring']:
            for count in grid['attention_deletions']:
                for side in ('low', 'high'):
                    add(_job('A1', f'{scoring}_{side}_D{count}', 'replay', 'same_prefix_replay',
                             {'attention_scoring': scoring, 'deletion_count': count, 'deletion_side': side, 'grid_phase': 'dev'}))
    if 'A2' in selected:
        grid = a2_matrix('dev')
        for budget in grid['budgets']:
            for selector in grid['selectors']:
                for seed in (grid['random_seeds'] if selector == 'random' else (42,)):
                    add(_job('A2', f'{selector}_B{budget}_seed{seed}', 'replay', 'same_prefix_replay',
                             {'selector': selector, 'budget': budget, 'seed': seed, 'grid_phase': 'dev',
                              'diagnostic_only': selector == 'current_dflash_oracle',
                              'ranking_source_round_offset': -1 if selector == 'previous_dflash' else 0}))
    # B0/B1/B1-V and B4 analyses use the same native trace prerequisite even
    # when the caller selects only the B family or the E2E subset.
    trace = _job('B0', 'reference_trace', 'trace', 'trace',
                 {'draft_gamma': 15, 'collect_adjacent_rounds': True, 'timing_baseline': False})
    if selected & {'A3', 'A3-REPLAY'}:
        grid = a3_matrix('dev')
        for selector in selectors:
            for budget in grid['budgets']:
                for age in grid['ranking_ages']:
                    add(_job('A3', f'replay_{selector}_B{budget}_age{age}', 'replay', 'same_prefix_replay',
                             {'selector': selector, 'budget': budget, 'ranking_age': age, 'grid_phase': 'dev',
                              'ranking_source_round_offset': -age, 'snapshot_shared_across_actions': True}))
    signal = _job('B1', 'candidate_signals', 'analysis', 'cpu_analysis',
                  {'analysis': 'candidate_signal_extract', 'draft_gamma': 15,
                   'collect_adjacent_rounds': True, 'collect_draft_uncertainty': True}, (trace.id,))
    gamma_actions = [_job('B1', f'gamma{gamma}', 'replay', 'same_prefix_replay',
                         {'gamma': gamma, 'draft_gamma': gamma, 'draft_each_action': True,
                          'same_prefix': True}, (trace.id,)) for gamma in gammas]
    prefix_actions = [_job('B1-V', f'k{k}', 'replay', 'same_prefix_replay',
                          {'k': k, 'draft_gamma': 15, 'draft_once_shared_prefixes': True,
                           'decision_timing': 'after_draft_before_verify'}, (trace.id,)) for k in gammas]
    if 'B0' in selected:
        add(trace)
    if 'B1' in selected:
        add(signal)
        for job in gamma_actions:
            add(job)
    if 'B1-V' in selected:
        for job in prefix_actions:
            add(job)
    fits = {variant: _job('B2', f'survival_fit_{variant}', 'analysis', 'cpu_analysis',
                         {'analysis': 'survival_fit', 'model_variant': variant, 'source_group_folds': 4,
                          'requires_positive_result': False}, (signal.id,)) for variant in SURVIVAL_VARIANTS}
    if 'B2' in selected:
        add(_job('B2', 'survival_cross_validation', 'analysis', 'cpu_analysis',
                 {'analysis': 'survival_cross_validation', 'source_group_folds': 4,
                  'requires_positive_result': False}, (signal.id,)))
        add(_job('B2', 'draft_shape_analysis', 'analysis', 'cpu_analysis',
                 {'analysis': 'draft_shape_analysis', 'requires_positive_result': False},
                 (job.id for job in gamma_actions)))
        for fit in fits.values():
            add(fit)
    if 'B3' in selected:
        for k in gammas:
            add(_job('B3', f'fixed_k_{k}', 'generation', 'b3_fixed_k',
                     {'draft_gamma': 15, 'k': k, 'decision_timing': 'after_draft_before_verify'}))
        b3_variants = (
            'R', 'R-A', 'R-DH', 'R-DC',
            'R-Hazard-Pos', 'R-Hazard-Present', 'R-Hazard-Prefix',
            'R-Hazard-Suffix-Margin', 'R-Hazard-Suffix-Entropy',
        )
        for variant in b3_variants:
            if variant in fits:
                add(_job('B3', f'adaptive_k_{variant}', 'generation', 'adaptive_verification',
                         {'draft_gamma': 15, 'allowed_prefixes': list(gammas), 'model_variant': variant,
                          'decision_timing': 'after_draft_before_verify', 'requires_positive_result': False},
                         (fits[variant].id, *(job.id for job in prefix_actions))))
    if 'B4' in selected:
        for gamma in gammas:
            if gamma != 15:
                add(generation(f'fixed_gamma_{gamma}', {'draft_gamma': gamma, 'gamma': gamma, 'decision_timing': 'before_draft'}))
        for variant in _COST_VARIANTS:
            fit = add(_job('B4', f'draft_cost_fit_{variant}', 'analysis', 'cpu_analysis',
                           {'analysis': 'draft_cost_fit', 'model_variant': variant, 'gammas': list(gammas),
                            'signals_round_offset': -1, 'requires_positive_result': False},
                           (signal.id, *(job.id for job in gamma_actions))))
            add(_job('B4', f'adaptive_gamma_{variant}', 'generation', 'adaptive_draft',
                     {'model_variant': variant, 'gammas': list(gammas), 'decision_timing': 'before_draft',
                      'signals_round_offset': -1, 'requires_adjacent_rounds': True,
                      'requires_positive_result': False}, (fit.id,)))
    return list(jobs.values())
