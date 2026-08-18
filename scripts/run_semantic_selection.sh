#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT/config/semantic_selection.env}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${INPUT_FILE:?INPUT_FILE is required}"
: "${OUTPUT_FILE:?OUTPUT_FILE is required}"
: "${MODEL:?MODEL is required}"

cd "$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

read -r -a selector_args <<< "${SELECTORS:-random lead tfidf textrank mmr}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  read -r -a budget_args <<< "${SMOKE_TOKEN_BUDGETS:-512}"
  max_new_tokens="${MAX_NEW_TOKENS:-128}"
  max_new_tokens=$((max_new_tokens < 32 ? max_new_tokens : 32))
else
  read -r -a budget_args <<< "${TOKEN_BUDGETS:-512 1024 2048}"
  max_new_tokens="${MAX_NEW_TOKENS:-128}"
fi

ARGS=(
  --input "$INPUT_FILE"
  --output "$OUTPUT_FILE"
  --document-field "${DOCUMENT_FIELD:-document}"
  --id-field "${ID_FIELD:-id}"
  --reference-field "${REFERENCE_FIELD:-reference}"
  --limit "${MAX_SAMPLES:-5}"
  --selectors "${selector_args[@]}"
  --token-budgets "${budget_args[@]}"
  --model "$MODEL"
  --device "${DEVICE:-auto}"
  --dtype "${DTYPE:-auto}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}"
  --max-new-tokens "$max_new_tokens"
  --warmup-rounds "${WARMUP_ROUNDS:-0}"
  --random-seed "${RANDOM_SEED:-42}"
  --mmr-lambda "${MMR_LAMBDA:-0.7}"
  --embedding-model "${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  --embedding-device "${EMBEDDING_DEVICE:-cpu}"
  --rouge
)

exec uv run --project "$ROOT" --locked python \
  "$ROOT/externals/Sematic_selection/infer.py" "${ARGS[@]}"
