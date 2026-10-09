#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/common/runtime.sh
source "$ROOT/scripts/common/runtime.sh"

export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec "$FAST_INFER_PYTHON" "$ROOT/scripts/run_finetuning_b200.py" \
  --phase1-only \
  --generation-backend "${FINETUNE_GENERATION_BACKEND:-hf}" \
  --capture-backend "${FINETUNE_CAPTURE_BACKEND:-hf}" \
  "$@"
