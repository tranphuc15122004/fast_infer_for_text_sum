#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/eagle3_qwen3.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${BASE_MODEL:?BASE_MODEL is required}"
: "${EAGLE_MODEL:?EAGLE_MODEL is required}"
: "${BENCH_NAME:?BENCH_NAME is required}"
: "${QUESTION_BEGIN:?QUESTION_BEGIN is required}"
: "${QUESTION_END:?QUESTION_END is required}"
: "${MAX_NEW_TOKENS:?MAX_NEW_TOKENS is required}"
: "${NUM_CHOICES:?NUM_CHOICES is required}"
: "${TEMPERATURE:?TEMPERATURE is required}"
: "${TOTAL_TOKEN:?TOTAL_TOKEN is required}"
: "${DEPTH:?DEPTH is required}"
: "${TOP_K:?TOP_K is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"

if [[ ! -d "$BASE_MODEL" || ! -f "$BASE_MODEL/config.json" ]]; then
  echo "Qwen3 base model or config.json not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -d "$EAGLE_MODEL" || ! -f "$EAGLE_MODEL/config.json" ]]; then
  echo "EAGLE3 checkpoint or config.json not found: $EAGLE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$EAGLE_MODEL/model.safetensors" && ! -f "$EAGLE_MODEL/pytorch_model.bin" ]]; then
  echo "EAGLE3 checkpoint has no model.safetensors or pytorch_model.bin: $EAGLE_MODEL" >&2
  exit 1
fi

QUESTION_FILE="$ROOT/externals/EAGLE/eagle/data/$BENCH_NAME/question.jsonl"
if [[ ! -f "$QUESTION_FILE" ]]; then
  echo "Question file not found: $QUESTION_FILE" >&2
  exit 1
fi

cd "$ROOT"

# Check dimensions before loading several GB of weights onto the GPU.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/fast_infer_uv_cache}"
uv run --project "$ROOT" --locked python - "$BASE_MODEL/config.json" "$EAGLE_MODEL/config.json" <<'PY'
import json
import sys

base_path, eagle_path = sys.argv[1:]
with open(base_path) as f:
    base = json.load(f)
with open(eagle_path) as f:
    eagle = json.load(f)

checks = {
    "hidden_size": (base.get("hidden_size"), eagle.get("hidden_size")),
    "num_attention_heads": (base.get("num_attention_heads"), eagle.get("num_attention_heads")),
    "num_key_value_heads": (base.get("num_key_value_heads"), eagle.get("num_key_value_heads")),
    "head_dim": (base.get("head_dim"), eagle.get("head_dim")),
    "vocab_size": (base.get("vocab_size"), eagle.get("vocab_size")),
}

print("EAGLE3 compatibility check:")
for key, (left, right) in checks.items():
    print(f"  {key}: base={left}, eagle3={right}")

mismatches = [key for key, (left, right) in checks.items() if left != right]
if mismatches:
    raise SystemExit("Incompatible base/EAGLE3 config fields: " + ", ".join(mismatches))
PY

export PYTHONPATH="$ROOT/externals/EAGLE${PYTHONPATH:+:$PYTHONPATH}"

exec uv run --project "$ROOT" --locked python "$ROOT/scripts/eagle3_infer_qwen3.py" \
  --base-model "$BASE_MODEL" \
  --eagle-model "$EAGLE_MODEL" \
  --question-file "$QUESTION_FILE" \
  --question-begin "$QUESTION_BEGIN" \
  --question-end "$QUESTION_END" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --total-token "$TOTAL_TOKEN" \
  --depth "$DEPTH" \
  --top-k "$TOP_K" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_FILE"
