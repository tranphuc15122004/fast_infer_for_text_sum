#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANONICAL_DATA_DIR="/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins"

BACKEND="both"
if [[ "${1:-}" == "fa4" || "${1:-}" == "vllm" || "${1:-}" == "both" ]]; then
  BACKEND="$1"
  shift
fi

MODE="${LENGTH_BINS_MODE:-smoke}"
MASTER_CONFIG="${FAST_INFER_MASTER_CONFIG:-}"
DATA_DIR="${LENGTH_BINS_DATA_DIR:-}"
OUTPUT_ROOT="${LENGTH_BINS_OUTPUT_ROOT:-$ROOT/outputs/length_bins_benchmark}"
RUN_ID="${LENGTH_BINS_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$BASHPID}"
SAMPLES_PER_BIN="${LENGTH_BINS_SAMPLES_PER_BIN:-}"
MAX_NEW_TOKENS="${LENGTH_BINS_MAX_NEW_TOKENS:-}"
MAX_MODEL_LEN="${LENGTH_BINS_MAX_MODEL_LEN:-18432}"
FA4_METHODS="${LENGTH_BINS_FA4_METHODS:-vanilla_hf,dflash,eagle3,domino,dspark}"
VLLM_METHODS="${LENGTH_BINS_VLLM_METHODS:-vanilla_vllm,eagle3,dflash,domino,dspark}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run.sh length_bins [fa4|vllm|both] [options]

Options:
  --mode MODE             preflight, smoke, representative, or full (default: smoke)
  --config PATH           master shell-env file
  --data-dir PATH         directory containing manifest.json and bin*.jsonl
  --output-root PATH      root for this benchmark's artifacts
  --run-id ID             artifact run ID (must be unique)
  --samples-per-bin N     smoke/representative cap; defaults to 2/20
  Inputs are never truncated so each sample stays in its declared length bin.
  --max-new-tokens N      defaults to 8 for smoke, 512 otherwise
  --max-model-len N       vLLM context limit (default: 18432)
  --fa4-methods LIST      FA4 methods (default: all five supported methods)
  --vllm-methods LIST     vLLM methods (default: all five supported methods)
  --preflight-only        shorthand for --mode preflight
  --smoke                 shorthand for --mode smoke
  --full                  shorthand for --mode full
  -h, --help              show this help

Examples:
  bash scripts/run.sh length_bins both --mode smoke
  bash scripts/run.sh length_bins fa4 --mode full --config /path/to/master.env
  bash scripts/run.sh length_bins vllm --mode full --max-model-len 18432
EOF
}

while (($#)); do
  case "$1" in
    --mode) MODE="${2:?--mode requires a value}"; shift 2 ;;
    --config) MASTER_CONFIG="${2:?--config requires a path}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:?--data-dir requires a path}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:?--output-root requires a path}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id requires a value}"; shift 2 ;;
    --samples-per-bin) SAMPLES_PER_BIN="${2:?--samples-per-bin requires a value}"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="${2:?--max-new-tokens requires a value}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:?--max-model-len requires a value}"; shift 2 ;;
    --fa4-methods) FA4_METHODS="${2:?--fa4-methods requires a value}"; shift 2 ;;
    --vllm-methods) VLLM_METHODS="${2:?--vllm-methods requires a value}"; shift 2 ;;
    --preflight-only) MODE="preflight"; shift ;;
    --smoke) MODE="smoke"; shift ;;
    --full) MODE="full"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$BACKEND" in
  fa4|vllm|both) ;;
  *) echo "Backend must be fa4, vllm, or both: $BACKEND" >&2; exit 2 ;;
esac
case "$MODE" in
  preflight|smoke|representative|full) ;;
  *) echo "Mode must be preflight, smoke, representative, or full: $MODE" >&2; exit 2 ;;
esac
if [[ -z "$DATA_DIR" ]]; then
  if [[ -d "$CANONICAL_DATA_DIR" ]]; then
    DATA_DIR="$CANONICAL_DATA_DIR"
  else
    DATA_DIR="$ROOT/data/length_bins"
  fi
fi
if [[ "$DATA_DIR" != /* ]]; then DATA_DIR="$ROOT/$DATA_DIR"; fi
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT"; fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run ID may contain only letters, digits, '_' and '-': $RUN_ID" >&2
  exit 2
fi
if [[ ! -d "$DATA_DIR" || ! -f "$DATA_DIR/manifest.json" ]]; then
  echo "Length-bin data directory or manifest not found: $DATA_DIR" >&2
  exit 2
fi
if [[ -z "$MAX_NEW_TOKENS" ]]; then
  if [[ "$MODE" == "smoke" ]]; then MAX_NEW_TOKENS=8; else MAX_NEW_TOKENS=512; fi
fi
if [[ -z "$SAMPLES_PER_BIN" ]]; then
  case "$MODE" in
    smoke) SAMPLES_PER_BIN=2 ;;
    representative) SAMPLES_PER_BIN=20 ;;
    full|preflight) SAMPLES_PER_BIN=0 ;;
  esac
fi
if [[ "$MODE" == "full" || "$MODE" == "preflight" ]]; then
  SAMPLES_PER_BIN=0
fi
if [[ ! "$MAX_NEW_TOKENS" =~ ^[0-9]+$ || "$MAX_NEW_TOKENS" -lt 2 ]]; then
  echo "--max-new-tokens must be an integer >= 2" >&2
  exit 2
fi
if [[ ! "$SAMPLES_PER_BIN" =~ ^[0-9]+$ ]]; then
  echo "--samples-per-bin must be a non-negative integer" >&2
  exit 2
fi
if [[ ("$MODE" == "smoke" || "$MODE" == "representative") && "$SAMPLES_PER_BIN" -lt 1 ]]; then
  echo "--samples-per-bin must be positive for smoke/representative; use --mode full for all rows" >&2
  exit 2
fi
if [[ ! "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]]; then
  echo "--max-model-len must be a positive integer" >&2
  exit 2
fi
MIN_MODEL_LEN=$((16384 + MAX_NEW_TOKENS))
if ((MAX_MODEL_LEN < MIN_MODEL_LEN)); then
  echo "--max-model-len must be at least $MIN_MODEL_LEN to preserve the full length-bin prompts" >&2
  exit 2
fi

BINS=(bin1_00k_02k bin2_02k_04k bin3_04k_08k bin4_08k_12k bin5_12k_16k)
for bin_name in "${BINS[@]}"; do
  if [[ ! -f "$DATA_DIR/$bin_name.jsonl" ]]; then
    echo "Missing length-bin file: $DATA_DIR/$bin_name.jsonl" >&2
    exit 2
  fi
done

RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Run output already exists: $RUN_ROOT" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
printf '[length-bins] backend=%s mode=%s data=%s output=%s\n' \
  "$BACKEND" "$MODE" "$DATA_DIR" "$RUN_ROOT"

if [[ -n "$MASTER_CONFIG" ]]; then
  export FAST_INFER_MASTER_CONFIG="$MASTER_CONFIG"
fi

if [[ "$BACKEND" == "fa4" || "$BACKEND" == "both" ]]; then
  FA4_ARGS=(
    --data-dir "$DATA_DIR"
    --output-dir "$RUN_ROOT/fa4"
    --datasets all
    --methods "$FA4_METHODS"
    --max-input-tokens 0
    --max-new-tokens "$MAX_NEW_TOKENS"
    --samples-per-dataset "$SAMPLES_PER_BIN"
    --run-id "${RUN_ID}-fa4"
  )
  if [[ "$MODE" == "preflight" ]]; then
    FA4_ARGS+=(--preflight-only)
  else
    FA4_ARGS+=(--mode "$MODE")
  fi
  printf '[length-bins] starting FA4 native run\n'
  bash "$ROOT/scripts/run_fa4_benchmark.sh" "${FA4_ARGS[@]}"
fi

if [[ "$BACKEND" == "vllm" || "$BACKEND" == "both" ]]; then
  for bin_name in "${BINS[@]}"; do
    VLLM_ARGS=(
      --length-spread-samples
      --require-blackwell
    )
    case "$MODE" in
      preflight) VLLM_ARGS+=(--preflight-only) ;;
      smoke) VLLM_ARGS+=(--smoke --max-samples "$SAMPLES_PER_BIN") ;;
      representative) VLLM_ARGS+=(--max-samples "$SAMPLES_PER_BIN") ;;
      full) VLLM_ARGS+=(--full --max-samples 0) ;;
    esac
    printf '[length-bins] starting vLLM run for %s\n' "$bin_name"
    VLLM_DATA_FILE="$DATA_DIR/$bin_name.jsonl" \
    VLLM_OUTPUT_DIR="$RUN_ROOT/vllm" \
    VLLM_RUN_ID="${RUN_ID}-vllm-${bin_name}" \
    VLLM_MAX_INPUT_TOKENS=0 \
    VLLM_MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    VLLM_METHODS="$VLLM_METHODS" \
      bash "$ROOT/scripts/run_vllm_all.sh" "${VLLM_ARGS[@]}"
  done
fi

printf '[length-bins] completed; artifacts: %s\n' "$RUN_ROOT"
