#!/usr/bin/env python3
"""Microbenchmark CPU pooling/top-B từ snapshot; không đo full selector runtime/GPU.

Không suy diễn GPU latency từ một công thức băng thông hoặc gắn CPU microseconds
thành overhead inference. Dùng profiler runtime riêng để đo QK/softmax/D2H/gather.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from run_candidate_benchmarks import load_snapshot
from TrainingFree.context_interventions.selector_evaluation import pool_queries, protected_topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations phải dương')
    manifest_path = args.snapshot.parent.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    snapshot = load_snapshot(args.snapshot, manifest)
    prefix = int(snapshot['prefix_length'])
    results = []
    for mode in ('mean', 'max'):
        def operation():
            scores = pool_queries(snapshot, 18, mode=mode, scope='accepted')[:prefix]
            return protected_topk(scores, args.budget)
        for _ in range(5):
            operation()
        timings=[]
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            operation()
            timings.append((time.perf_counter_ns()-start)/1000.)
        results.append({'mode': mode, 'median_us': float(np.median(timings)),
                        'p90_us': float(np.quantile(timings,.9))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope':'CPU pooling+protected token top-B only',
        'hardware':platform.processor(), 'snapshot':str(args.snapshot.resolve()),
        'budget':args.budget, 'iterations':args.iterations, 'results':results,
        'excluded':'model forward, QK, softmax, GPU transfer, gather, cache/bank update'},indent=2)+'\n')


if __name__ == '__main__':
    main()
