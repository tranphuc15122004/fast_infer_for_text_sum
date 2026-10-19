#!/usr/bin/env python3
"""Đánh giá riêng Phase A từ metadata thật; coverage không đại diện acceptance/TPS."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from common.io_util import JsonlWriter
from TrainingFree.context_interventions.selector_evaluation import (
    CANDIDATES, reference_attention, selector_scores, select_chunks,
    spatial_continuity, summarize_rows,
)


def load_snapshot(path, manifest, legacy_layer_ids=None):
    with np.load(path, allow_pickle=False) as archive:
        snapshot = {key: archive[key] for key in archive.files}
    if 'target_layer_ids' not in snapshot:
        ids = manifest.get('target_layers') or legacy_layer_ids
        if not ids or not all(isinstance(layer, int) for layer in ids):
            raise ValueError(f'{path}: thiếu layer IDs trong snapshot/manifest')
        if len(ids) != snapshot['target'].shape[0]:
            raise ValueError(f'{path}: layer metadata/array shape mismatch')
        snapshot['target_layer_ids'] = np.asarray(ids)
    match = re.fullmatch(r'sample_(\d+)_context_(\d+)_(?:round_\d+|first|middle|last)\.npz', path.name)
    if not match:
        raise ValueError(f'Tên snapshot chưa nhận diện: {path.name}')
    index, cap = map(int, match.groups())
    if 'sample_id' not in snapshot:
        samples = manifest.get('samples', [])
        sample = samples[index] if index < len(samples) else f'legacy_sample_{index:02d}'
        snapshot['sample_id'] = sample['sample_id'] if isinstance(sample, dict) else str(sample)
    snapshot['sample_id'] = str(snapshot['sample_id'])
    snapshot['context_cap'] = int(snapshot.get('context_cap', cap))
    snapshot['round'] = int(snapshot['round'])
    snapshot['prefix_length'] = int(snapshot['prefix_length'])
    return snapshot


def evaluate(target_dir, output_dir, *, budgets, all_rounds=False,
             legacy_layer_ids=None, sink=16, recent=32, chunk_sizes=(1, 128)):
    manifest_path = target_dir.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    snapshots, sources = {}, []
    for path in sorted(target_dir.glob('*.npz')):
        if '_first_transition' in path.name or (('_round_' in path.name) != all_rounds):
            continue
        snapshot = load_snapshot(path, manifest, legacy_layer_ids)
        key = (snapshot['sample_id'], snapshot['context_cap'], snapshot['round'])
        if key in snapshots:  # first/middle/last có thể là alias cùng round
            continue
        snapshots[key] = snapshot
        sources.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not snapshots:
        raise ValueError('Không có snapshot phù hợp; kiểm tra --all-rounds và --target-dir')
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'metrics.jsonl').exists():
        raise FileExistsError('Dùng output directory mới để giữ provenance')
    rows, unavailable = [], []
    for (sample, cap, round_index), snapshot in sorted(snapshots.items()):
        prefix = snapshot['prefix_length']
        scores, missing = selector_scores(snapshot, prefix)
        unavailable.extend({'sample_id': sample, 'context_cap': cap, 'round': round_index,
            'method': method, 'reason': reason} for method, reason in missing.items())
        selections = {}
        for budget in budgets:
            if budget > prefix:
                continue
            for method, vector in scores.items():
                for chunk_size in chunk_sizes:
                    try:
                        selections[budget, method, chunk_size] = select_chunks(
                            vector, budget, chunk_size=chunk_size, sink=sink, recent=recent)
                    except ValueError as exc:
                        unavailable.append({'sample_id': sample, 'method': method, 'budget': budget, 'reason': str(exc)})
        references = [('same_round_diagnostic', snapshot)]
        following = snapshots.get((sample, cap, round_index + 1))
        if following is not None:
            references.append(('next_round', following))
        for candidate, layers in CANDIDATES.items():
            for scope, reference in references:
                try:
                    gt = reference_attention(reference, layers, prefix)
                except ValueError as exc:
                    unavailable.append({'sample_id': sample, 'context_cap': cap, 'round': round_index,
                                        'candidate': candidate, 'reason': str(exc)})
                    continue
                for budget in budgets:
                    if budget > prefix:
                        continue
                    for method, vector in scores.items():
                        for chunk_size in chunk_sizes:
                            indices = selections.get((budget, method, chunk_size))
                            if indices is None:
                                continue
                            rows.append({'type': 'phase_a_coverage', 'sample_id': sample,
                                'context_cap': cap, 'round': round_index, 'evaluation_scope': scope,
                                'candidate': candidate, 'method': method, 'chunk_size': chunk_size,
                                'budget': budget, 'selected_tokens': len(indices),
                                'unused_budget': budget-len(indices), 'coverage': float(gt[indices].sum()),
                                'spatial_continuity': spatial_continuity(indices)})
    fields = ('context_cap', 'budget', 'chunk_size', 'method', 'candidate', 'evaluation_scope')
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    summaries = [{**dict(zip(fields, key)), **summarize_rows(values),
                  'mean_selected_tokens': float(np.mean([row['selected_tokens'] for row in values]))}
                 for key, values in sorted(groups.items())]
    report = {'type': 'summary', 'schema_version': 'phase-a.evaluation.v2',
        'status': ('partial' if unavailable else 'ok') if rows else 'unavailable', 'phase': 'A', 'pipeline_e2e': False,
        'unique_documents': len({key[0] for key in snapshots}), 'unique_snapshots': len(snapshots),
        'target_dir': str(target_dir.resolve()), 'source_manifest': manifest, 'source_files': sources,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'metric_code_sha256': hashlib.sha256((ROOT / 'src/TrainingFree/context_interventions/selector_evaluation.py').read_bytes()).hexdigest(),
        'metric': 'document-macro attention coverage conditional on keys visible at mask selection',
        'reference_signal': 'mean-head mean-aligned-15-query target attention; includes speculative suffix, proxy only',
        'same_round_note': 'diagnostic only; selecting and scoring the same vector favors its own top-B',
        'next_round_note': 'observational temporal coverage on dense trajectory; not acceptance under compression',
        'protection': {'sink': sink, 'recent': recent}, 'unavailable': unavailable, 'results': summaries}
    writer = JsonlWriter(output_dir / 'metrics.jsonl')
    for row in rows:
        writer.add(row)
    writer.finalize({key: value for key, value in report.items()
        if key not in {'results', 'unavailable', 'source_files', 'source_manifest'}})
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    if summaries:
        with (output_dir / 'summary.csv').open('w') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            csv_writer.writeheader()
            csv_writer.writerows(summaries)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-dir', type=Path, required=True)
    parser.add_argument('--parent-dir', type=Path, help='Legacy option; unified snapshots required')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--all-rounds', action='store_true')
    parser.add_argument('--legacy-layer-ids')
    parser.add_argument('--budgets', default='256,512,1024,2048,4096')
    parser.add_argument('--chunk-sizes', default='1,128')
    parser.add_argument('--sink', type=int, default=16)
    parser.add_argument('--recent', type=int, default=32)
    args = parser.parse_args()
    if args.parent_dir and args.parent_dir.resolve() != args.target_dir.resolve():
        parser.error('Không ghép parent từ run khác: dùng snapshot unified để giữ đúng trajectory')
    report = evaluate(args.target_dir, args.output_dir,
        budgets=tuple(map(int, args.budgets.split(','))), chunk_sizes=tuple(map(int, args.chunk_sizes.split(','))),
        all_rounds=args.all_rounds, sink=args.sink, recent=args.recent,
        legacy_layer_ids=list(map(int, args.legacy_layer_ids.split(','))) if args.legacy_layer_ids else None)
    print(json.dumps({key: report[key] for key in ('status', 'unique_documents', 'unique_snapshots', 'metric')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
