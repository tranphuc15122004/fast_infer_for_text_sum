# Implementation plan — E29-A Future-Use / Source-Retirement Oracle

## Goal

Implement a reproducible oracle screen that measures future source attention,
projected active-source decode work and target-logit stability after oracle
source-unit retirement.

## Subtask 1: deterministic oracle math and report helpers

- Add pure functions for chunk ranges, future-use matrices, active fractions,
  projected work/savings and safe aggregate statistics.
- Add unit tests with synthetic attention traces, including empty-future and
  non-finite rejection cases.
- Keep raw per-document rows in JSONL/JSON; report only aggregates.

## Subtask 2: target attention collector

- Add a local-only Qwen3 target runner under `src/analyze/source_retirement/`.
- Reuse existing prompt rendering and chunked prefill conventions.
- Generate greedy 256-token traces and store unnormalized source-unit mass,
  generated ids, entropy and runtime metadata.
- Use incremental eager attention and discard full attention tensors after each
  query; never retain a full L×L attention matrix.
- Add a one-document smoke mode and explicit CUDA availability/device metadata.

## Subtask 3: target-retention ablation

- At registered checkpoints and delta values, construct retained-source inputs
  from the same source-unit partition and the same generated prefix.
- Compare full vs retained next-token logits using top-1 agreement, KL,
  target-token NLL delta and optional teacher-forced horizon agreement.
- Mark failures explicitly; do not silently convert OOM or missing model output
  into a pass.

## Subtask 4: final E29-A pipeline [INTEGRATION]

- Wire dataset loading, collection, oracle analysis, ablation, manifest and
  report generation.
- Run L0 checks: compile, unit tests, schema/finite checks and deterministic
  smoke.
- Run L1 one-document runtime smoke where the target model/device is available.
- Run the registered 45-document screen only when the runtime preflight passes;
  otherwise write an explicit blocked report with the exact environment cause.
- Gate E29-B/E29-C from the final aggregate result; no automatic continuation.

## Validation and acceptance

- A GPU result requires both `torch.cuda.is_available()` and a visible CUDA
  device in the run manifest.
- E29-B is opened only if at least two datasets reach 35–40% oracle work
  reduction and the retention ablation meets the registered safety criterion.
- Any CPU-only or partial smoke is reported as incomplete, not as E29-A full.
