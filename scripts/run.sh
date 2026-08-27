#!/usr/bin/env bash
# Dispatcher: maps a baseline name to its env + run wrapper.
#
# Usage:
#   bash scripts/run.sh <baseline> [extra args...]
#
# Each baseline has a wrapper `scripts/run_<baseline>.sh` that sources
# `config/<baseline>.env` and invokes the shared Python 3.12 interpreter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?usage: run.sh <baseline> [args...]}"
shift || true

case "$BASELINE" in
  eagle3)      WRAPPER="scripts/run_eagle3_qwen3.sh" ;;
  dflash)      WRAPPER="scripts/run_dflash.sh" ;;
  llmlingua)   WRAPPER="scripts/run_llmlingua.sh" ;;
  fastkv)      WRAPPER="scripts/run_fastkv.sh" ;;
  rocketkv)    WRAPPER="scripts/run_rocketkv.sh" ;;
  gemfilter)   WRAPPER="scripts/run_gemfilter.sh" ;;
  specprefill) WRAPPER="scripts/run_specprefill.sh" ;;
  minference)  WRAPPER="scripts/run_minference.sh" ;;
  magicdec)    WRAPPER="scripts/run_magicdec.sh" ;;
  longspec)    WRAPPER="scripts/run_longspec.sh" ;;
  specextend)  WRAPPER="scripts/run_specextend.sh" ;;
  higoe)       WRAPPER="scripts/run_higoe.sh" ;;
  semantic_selection) WRAPPER="scripts/run_semantic_selection.sh" ;;
  flexprefill)     WRAPPER="scripts/run_flexprefill.sh" ;;
  *)
    echo "Unknown baseline: $BASELINE" >&2
    echo "Available: eagle3 dflash llmlingua fastkv rocketkv gemfilter specprefill minference magicdec longspec specextend higoe semantic_selection flexprefill" >&2
    exit 1
    ;;
esac

exec "$ROOT/$WRAPPER" "$@"
