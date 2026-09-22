#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep this experimental launcher on the same interpreter/config contract as
# every production baseline launcher. The experiment itself does not need the
# external master values, so it intentionally does not call a baseline config
# namespace loader.
# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh" || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

OUT_ROOT="${HORIZON_OUTPUT_DIR:-$ROOT/outputs/specextend_horizon_cmr/2026-09-13_horizon_cmr}"
PYTHON_BIN="${HORIZON_PYTHON:-$FAST_INFER_PYTHON}"
HORIZON_PYTHONPATH="${HORIZON_PYTHONPATH:-${PYTHONPATH:-}}"
TARGET_MODEL="${SPECEXTEND_BASE_MODEL:-/home/tuantb/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5-16k/snapshots/c8df3ca4436a3bce5c4b5877e0117032081852b4}"
DRAFT_MODEL="${SPECEXTEND_DRAFT_MODEL:-/home/tuantb/.cache/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109}"
DATA_ROOT="${SPECEXTEND_DATA_ROOT:-$ROOT/externals/SpecExtend/specextend/data/govreport}"
LEVEL="${1:-smoke}"

case "$LEVEL" in
  smoke) DATA_FILE="$DATA_ROOT/govreport_512.jsonl"; SAMPLES="${HORIZON_SAMPLES:-1}"; INPUT_LIMIT="${HORIZON_INPUT_LIMIT:-512}"; GEN="${HORIZON_MAX_NEW_TOKENS:-64}" ;;
  1k) DATA_FILE="$DATA_ROOT/govreport_1K.jsonl"; SAMPLES="${HORIZON_SAMPLES:-1}"; INPUT_LIMIT="${HORIZON_INPUT_LIMIT:-1024}"; GEN="${HORIZON_MAX_NEW_TOKENS:-128}" ;;
  2k) DATA_FILE="$DATA_ROOT/govreport_2K.jsonl"; SAMPLES="${HORIZON_SAMPLES:-1}"; INPUT_LIMIT="${HORIZON_INPUT_LIMIT:-2048}"; GEN="${HORIZON_MAX_NEW_TOKENS:-256}" ;;
  4k) DATA_FILE="$DATA_ROOT/govreport_4K.jsonl"; SAMPLES="${HORIZON_SAMPLES:-20}"; INPUT_LIMIT="${HORIZON_INPUT_LIMIT:-4096}"; GEN="${HORIZON_MAX_NEW_TOKENS:-256}" ;;
  8k) DATA_FILE="$DATA_ROOT/govreport_8K.jsonl"; SAMPLES="${HORIZON_SAMPLES:-20}"; INPUT_LIMIT="${HORIZON_INPUT_LIMIT:-8192}"; GEN="${HORIZON_MAX_NEW_TOKENS:-256}" ;;
  *) echo "usage: $0 {smoke|1k|2k|4k|8k}" >&2; exit 2 ;;
esac

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/traces" "$OUT_ROOT/analysis"
PREFLIGHT="$OUT_ROOT/preflight_${LEVEL}.json"
set +e
PYTHONPATH="$ROOT/src${HORIZON_PYTHONPATH:+:$HORIZON_PYTHONPATH}" "$PYTHON_BIN" \
  "$ROOT/src/analyze/specextend_horizon/preflight.py" \
  --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
  --data-dir "$DATA_ROOT" --output "$PREFLIGHT"
PREFLIGHT_RC=$?
set -e
if [[ "$PREFLIGHT_RC" -ne 0 ]]; then
  echo "SpecExtend Horizon-CMR blocked by preflight; no inference was started." >&2
  PYTHONPATH="$ROOT/src${HORIZON_PYTHONPATH:+:$HORIZON_PYTHONPATH}" "$PYTHON_BIN" \
    "$ROOT/src/analyze/specextend_horizon/report.py" \
    --run-root "$OUT_ROOT" --level "$LEVEL" \
    --preflight "$PREFLIGHT" \
    --result "$OUT_ROOT/spec_extend_${LEVEL}.jsonl" \
    --trace "$OUT_ROOT/traces/spec_extend_${LEVEL}.jsonl" \
    --output "$OUT_ROOT/reports/${LEVEL}_report.md" || true
  exit "$PREFLIGHT_RC"
fi

LOG="$OUT_ROOT/logs/spec_extend_${LEVEL}.log"
TRACE="$OUT_ROOT/traces/spec_extend_${LEVEL}.jsonl"
RESULT="$OUT_ROOT/spec_extend_${LEVEL}.jsonl"
export SPECEXTEND_TRACE_FILE="$TRACE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +e
PYTHONPATH="$ROOT/scripts${HORIZON_PYTHONPATH:+:$HORIZON_PYTHONPATH}" "$PYTHON_BIN" "$ROOT/scripts/infer_specextend.py" \
  --script run_classic.py --model-name vicuna_7b \
  --base-model "$TARGET_MODEL" --draft-model "$DRAFT_MODEL" \
  --input-file "$DATA_FILE" --max-samples "$SAMPLES" \
  --max-input-tokens "$INPUT_LIMIT" --max-gen-len "$GEN" \
  --warmup-runs "${HORIZON_WARMUP_RUNS:-1}" --trace-file "$TRACE" \
  --output "$RESULT" --use-specextend 2>&1 | tee "$LOG"
RUN_RC=${PIPESTATUS[0]}
set -e

PYTHONPATH="$ROOT/src${HORIZON_PYTHONPATH:+:$HORIZON_PYTHONPATH}" "$PYTHON_BIN" \
  "$ROOT/src/analyze/specextend_horizon/report.py" \
  --run-root "$OUT_ROOT" --level "$LEVEL" \
  --preflight "$PREFLIGHT" --result "$RESULT" --trace "$TRACE" \
  --output "$OUT_ROOT/reports/${LEVEL}_report.md" || true

exit "$RUN_RC"
