#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/config/semantic_selection.env"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG_FILE="$1"
  shift
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file not found: $CONFIG_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

# This upstream entrypoint has no --smoke option. Consume the dispatcher flag
# here and translate it into the one-sample/short-generation settings below.
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--smoke" ]]; then
    SMOKE=1
  else
    PASSTHROUGH_ARGS+=("$1")
  fi
  shift
done

# Representative runner injects INPUT_FILE per dataset; direct smoke runs
# still need a deterministic local one-sample input.
# Semantic-selection's upstream loader requires the representative
# ``document`` field.  ``data/smoke_long_docs.jsonl`` is the generic fixture
# for baselines that consume ``text``/``prompt``, so use the matching debug
# fixture for direct smoke runs.
INPUT_FILE="${INPUT_FILE:-data/debug/smoke_real.jsonl}"
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
  limit=1
else
  read -r -a budget_args <<< "${TOKEN_BUDGETS:-512 1024 2048}"
  max_new_tokens="${MAX_NEW_TOKENS:-128}"
  limit="${MAX_SAMPLES:-5}"
fi

ARGS=(
  --input "$INPUT_FILE"
  --output "$OUTPUT_FILE"
  --document-field "${DOCUMENT_FIELD:-document}"
  --id-field "${ID_FIELD:-id}"
  --reference-field "${REFERENCE_FIELD:-reference}"
  --limit "$limit"
  --selectors "${selector_args[@]}"
  --token-budgets "${budget_args[@]}"
  --model "$MODEL"
  --device "${DEVICE:-auto}"
  --dtype "${DTYPE:-auto}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-auto}"
  --max-new-tokens "$max_new_tokens"
  --max-input-tokens "${MAX_INPUT_TOKENS:-0}"
  --warmup-rounds "${WARMUP_ROUNDS:-0}"
  --random-seed "${RANDOM_SEED:-42}"
  --mmr-lambda "${MMR_LAMBDA:-0.7}"
  --embedding-model "${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  --embedding-device "${EMBEDDING_DEVICE:-cpu}"
  --rouge
)

exec "$FAST_INFER_PYTHON" \
  "$ROOT/externals/Sematic_selection/infer.py" "${ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
