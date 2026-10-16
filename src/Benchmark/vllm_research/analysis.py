"""Cầu nối CPU tới estimator hiện có, chỉ fit từ evidence có schema thật."""
from __future__ import annotations

import math
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from .plan import ResearchJob


class EvidenceMissing(ValueError):
    """Evidence chưa đủ; khác với dữ liệu sai schema."""


def _candidate_signals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ghép proposal/verification liền kề từ native B0 trace thành evidence B1."""
    proposals: dict[tuple[str, int], Mapping[str, Any]] = {}
    verifications: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        request_id = str(row.get('request_id', '')).strip()
        round_index = row.get('round_index')
        if not request_id or isinstance(round_index, bool) or not isinstance(round_index, int):
            continue
        key = (request_id, round_index)
        if (row.get('type') == 'proposal'
                and int(row.get('drafted_candidates', 0) or 0) == 15
                and int(row.get('verified_candidates', 0) or 0) == 15):
            proposals[key] = row
        elif row.get('type') == 'verification':
            verifications[key] = row
    output = []
    for key, proposal in proposals.items():
        verification = verifications.get(key)
        if verification is None:
            continue
        uncertainty = proposal.get('draft_uncertainty') or {}
        if any(not isinstance(uncertainty.get(name), (list, tuple)) or len(uncertainty[name]) != 15
               for name in ('normalized_entropy', 'confidence')):
            continue
        sample_id = str(proposal.get('sample_id', '')).strip()
        source_group = str(proposal.get('source_group_id', '')).strip()
        dataset = str(proposal.get('dataset', '')).strip()
        prompt_tokens = proposal.get('prompt_tokens')
        if not sample_id or not source_group or not dataset or not isinstance(prompt_tokens, int):
            continue
        state_key = {
            'request_id': key[0], 'round_index': key[1],
            'sample_id': sample_id, 'prompt_hash': proposal.get('prompt_hash'),
            'logical_length': proposal.get('logical_length'), 'anchor': proposal.get('anchor'),
        }
        state_id = hashlib.sha256(json.dumps(state_key, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        output.append({
            'type': 'candidate_signals', 'status': 'ok', 'state_id': state_id,
            'sample_id': sample_id, 'source_group_id': source_group, 'dataset': dataset,
            'round_index': key[1], 'processed_output_tokens': int(proposal.get('processed_output_tokens', 0)),
            'prompt_tokens': prompt_tokens, 'context_tokens': int(proposal.get('context_tokens', 0)),
            'accepted_candidates': int(verification.get('accepted_candidates', -1)),
            'draft_uncertainty': dict(uncertainty), 'history_ema': proposal.get('history_ema'),
            'history_missing': proposal.get('history_ema') is None,
            'source_trace_request_id': key[0], 'source_trace_round': key[1],
        })
    return output


def _signals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get('type') == 'candidate_signals' and row.get('status', 'ok') == 'ok']
    seen: set[str] = set()
    for row in selected:
        for name in ('state_id', 'source_group_id', 'sample_id', 'dataset'):
            if not str(row.get(name, '')).strip():
                raise ValueError(f'candidate_signals requires {name}')
        state = str(row['state_id'])
        if state in seen:
            raise ValueError(f'duplicate candidate state {state}')
        seen.add(state)
        for name in ('round_index', 'processed_output_tokens', 'prompt_tokens', 'context_tokens'):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'candidate_signals requires nonnegative integer {name}')
        accepted = row.get('accepted_candidates')
        if isinstance(accepted, bool) or not isinstance(accepted, int) or not 0 <= accepted <= 15:
            raise ValueError('candidate_signals requires accepted_candidates within [0,15]')
        uncertainty = row.get('draft_uncertainty') or {}
        for name in ('normalized_entropy', 'confidence'):
            values = uncertainty.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != 15:
                raise ValueError(f'candidate_signals requires 15 {name} values')
            if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
                raise ValueError(f'candidate_signals requires finite normalized {name}')
    return selected


def _draft_cost_rows(signals: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]],
                     gammas: Sequence[int]) -> list[dict[str, Any]]:
    from TrainingFree.speculation_experiments.lagged import pair_adjacent_round_signals

    paired = pair_adjacent_round_signals(signals)
    by_state: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in actions:
        if row.get('type') != 'draft_shape_action' or row.get('repetition_split') != 'selection':
            continue
        if row.get('status', 'ok') != 'ok' or row.get('acceptance_identity_valid', True) is not True:
            continue
        state = str(row.get('state_id', ''))
        gamma = int(row.get('gamma', -1))
        accepted = int(row.get('accepted_candidates', -1))
        duration = float(row.get('round_ms', math.nan))
        if not state or gamma not in gammas or not 0 <= accepted <= gamma or not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid measured gamma action used for B4 fitting')
        by_state[state][gamma].append(duration / (1 + accepted))
    output = []
    for signal in paired:
        costs = by_state.get(str(signal['state_id']), {})
        if not all(gamma in costs for gamma in gammas):
            continue
        output.append({**signal, 'selection_costs_ms_per_commit_by_gamma':
                       {str(gamma): mean(costs[gamma]) for gamma in gammas}})
    return output


def execute_analysis(job: ResearchJob,
                     artifacts: Mapping[str, Sequence[Mapping[str, Any]]], *,
                     bootstrap_replicates: int = 2000) -> dict[str, Any]:
    """Trả terminal record; không có ngưỡng lợi ích để cho phép fit/chạy policy.

    ``artifacts`` ánh xạ job ID sang raw rows, không phải summary hay generation
    text. Executor phải giữ source groups train/heldout tách riêng khi triển khai
    policy; model artifact giữ training_groups để kiểm tra điều đó.
    """
    if job.kind != 'analysis' or job.capability != 'cpu_analysis':
        raise ValueError('execute_analysis only accepts CPU analysis jobs')
    base = {'job_id': job.id, 'experiment': job.experiment, 'variant': job.variant,
            'capability': job.capability}
    missing = [dependency for dependency in job.dependencies if dependency not in artifacts or not artifacts[dependency]]
    if missing:
        return {**base, 'status': 'dependency_missing', 'missing_dependencies': missing,
                'reason': 'Thiếu raw evidence để phân tích/fit; các arm độc lập vẫn chạy.'}
    rows = [row for dependency in job.dependencies for row in artifacts[dependency]
            if row.get('type') != 'summary']
    if not rows:
        return {**base, 'status': 'dependency_missing', 'reason': 'Evidence chỉ có summary, chưa có measurement.'}
    try:
        operation = job.parameters['analysis']
        if operation == 'candidate_signal_extract':
            payload = _candidate_signals(rows)
            if not payload:
                raise EvidenceMissing('B0 trace không có proposal gamma=15 ghép được với verification và uncertainty đầy đủ.')
        elif operation == 'survival_fit':
            from TrainingFree.speculation_experiments.survival import fit_survival_curve, serialize_survival_curve
            signals = _signals(rows)
            if not signals:
                raise EvidenceMissing('Chưa có candidate_signals hợp lệ.')
            variant = job.parameters['model_variant']
            from TrainingFree.speculation_experiments.survival import DEFAULT_VARIANTS
            if any(feature in DEFAULT_VARIANTS[variant] for feature in ('parent_entropy', 'parent_confidence')) and any(not row.get('parent_signal') for row in signals):
                raise EvidenceMissing(f'{variant} cần parent target-logit signals chưa có trong B0 trace.')
            payload = serialize_survival_curve(fit_survival_curve(signals, variant=variant))
        elif operation == 'survival_cross_validation':
            from TrainingFree.speculation_experiments.survival import cross_validated_survival
            signals = _signals(rows)
            folds = int(job.parameters['source_group_folds'])
            if len({row['source_group_id'] for row in signals}) < folds:
                raise EvidenceMissing(f'Cần ít nhất {folds} source groups độc lập để CV.')
            from TrainingFree.speculation_experiments.survival import DEFAULT_VARIANTS
            variants = tuple(name for name, features in DEFAULT_VARIANTS.items()
                             if not any(feature in features for feature in ('parent_entropy', 'parent_confidence')))
            payload = cross_validated_survival(signals, n_splits=folds,
                                               variants=variants,
                                               bootstrap_replicates=bootstrap_replicates)
        elif operation == 'draft_shape_analysis':
            from TrainingFree.speculation_experiments.analysis import analyze_draft_shape
            actions = [row for row in rows if row.get('type') == 'draft_shape_action'
                       and row.get('status', 'ok') == 'ok']
            if not actions or not {'selection', 'evaluation'}.issubset({row.get('repetition_split') for row in actions}):
                raise EvidenceMissing('Oracle gap cần cả selection/evaluation evidence, không dùng lại cùng một timing.')
            payload = analyze_draft_shape(actions, bootstrap_replicates=bootstrap_replicates)
        elif operation == 'draft_cost_fit':
            from TrainingFree.speculation_experiments.lagged import fit_draft_cost_policy
            signals = _signals(rows)
            gammas = tuple(job.parameters['gammas'])
            cost_rows = _draft_cost_rows(signals, rows, gammas)
            if not cost_rows:
                raise EvidenceMissing('Thiếu cặp round liền kề cùng measured gamma actions để fit B4.')
            payload = fit_draft_cost_policy(cost_rows, variant=job.parameters['model_variant'], gammas=gammas)
        else:
            return {**base, 'status': 'unsupported', 'reason': f'CPU analysis chưa hỗ trợ {operation!r}.'}
    except EvidenceMissing as exc:
        return {**base, 'status': 'dependency_missing', 'reason': str(exc)}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {**base, 'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
    return {**base, 'status': 'succeeded', 'payload': payload,
            'evidence_rows': len(rows), 'requires_positive_result': False}
