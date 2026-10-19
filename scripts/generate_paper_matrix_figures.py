#!/usr/bin/env python3
"""Vẽ dữ liệu Phase A đã audit; không chứa số liệu hoặc thứ hạng gán sẵn."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--context-cap', type=int, default=16384)
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--chunk-size', type=int, default=1)
    parser.add_argument('--scope', choices=['next_round', 'same_round_diagnostic'], default='next_round')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get('schema_version') != 'phase-a.evaluation.v2':
        parser.error('Cần report phase-a.evaluation.v2 có provenance')
    rows = [r for r in report['results'] if r['context_cap'] == args.context_cap
            and r['chunk_size'] == args.chunk_size and r['evaluation_scope'] == args.scope]
    if not rows:
        parser.error('Không có dữ liệu phù hợp; không thay thế bằng scope/layer khác')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted({r['candidate'] for r in rows})
    methods = sorted({r['method'] for r in rows})
    budget_rows = [r for r in rows if r['budget'] == args.budget]
    if not budget_rows:
        parser.error('Không có budget đã yêu cầu')
    matrix = np.full((len(methods), len(candidates)), np.nan)
    for row in budget_rows:
        matrix[methods.index(row['method']), candidates.index(row['candidate'])] = 100*row['mean_coverage']
    fig, ax = plt.subplots(figsize=(max(7, len(candidates)*1.4), max(4, len(methods)*.45)))
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('#dddddd')
    img = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(candidates)), candidates, rotation=35, ha='right')
    ax.set_yticks(range(len(methods)), methods)
    for (i,j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:.1f}' if np.isfinite(value) else 'N/A', ha='center', va='center', fontsize=8)
    ax.set_title(f'Phase A: {args.scope}, context={args.context_cap}, budget={args.budget}, chunk={args.chunk_size}')
    fig.colorbar(img, ax=ax, label='Document-macro attention coverage (%)')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'coverage_matrix.{extension}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(candidates), figsize=(max(7, len(candidates)*3.5), 4), squeeze=False)
    for ax, candidate in zip(axes[0], candidates):
        for method in methods:
            values = sorted((r for r in rows if r['candidate']==candidate and r['method']==method), key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in values], [100*r['mean_coverage'] for r in values], marker='.', label=method)
        ax.set_title(candidate)
        ax.set_xlabel('Budget tokens')
        ax.set_ylim(0,100)
    axes[0,0].set_ylabel('Attention coverage (%)')
    axes[0,-1].legend(fontsize=6, bbox_to_anchor=(1.04, 1))
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'budget_curves.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    (args.output_dir / 'figure_provenance.json').write_text(json.dumps({
        'source_report': str(args.report.resolve()), 'source_code_sha256': report['code_sha256'],
        'metric_code_sha256': report.get('metric_code_sha256'),
        'scope': args.scope, 'context_cap': args.context_cap, 'chunk_size': args.chunk_size,
        'note': 'Proxy only; no acceptance, TPS or measured memory-coalescing claim'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
