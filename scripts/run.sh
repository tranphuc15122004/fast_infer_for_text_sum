#!/usr/bin/env bash
# Dispatcher: maps a baseline name to its launcher. Every launcher loads the
# same external master through config/master.path.
#
# Usage:
#   bash scripts/run.sh <baseline> [extra args...]
#
# Each baseline has a wrapper in `scripts/runners/` that invokes the shared
# Python 3.12 interpreter after loading the master config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?usage: run.sh <baseline> [args...]}"
shift || true

case "$BASELINE" in
  longbench_200) WRAPPER="scripts/run_longbench_200.sh" ;;
  vanilla_hf)   WRAPPER="scripts/runners/run_vanilla_hf.sh" ;;
  vanilla_fa)   WRAPPER="scripts/runners/run_vanilla_fa.sh" ;;
  eagle3)      WRAPPER="scripts/runners/run_eagle3_qwen3.sh" ;;
  dflash)      WRAPPER="scripts/runners/run_dflash.sh" ;;
  domino)      WRAPPER="scripts/runners/run_domino.sh" ;;
  sssd)       WRAPPER="scripts/runners/run_sssd.sh" ;;
  fafo)       WRAPPER="scripts/runners/run_fafo.sh" ;;
  llmlingua)   WRAPPER="scripts/runners/run_llmlingua.sh" ;;
  fastkv)      WRAPPER="scripts/runners/run_fastkv.sh" ;;
  rocketkv)    WRAPPER="scripts/runners/run_rocketkv.sh" ;;
  gemfilter)   WRAPPER="scripts/runners/run_gemfilter.sh" ;;
  specprefill) WRAPPER="scripts/runners/run_specprefill.sh" ;;
  minference)  WRAPPER="scripts/runners/run_minference.sh" ;;
  magicdec)    WRAPPER="scripts/runners/run_magicdec.sh" ;;
  longspec)    WRAPPER="scripts/runners/run_longspec.sh" ;;
  specextend)  WRAPPER="scripts/runners/run_specextend.sh" ;;
  higoe)       WRAPPER="scripts/runners/run_higoe.sh" ;;
  semantic_selection) WRAPPER="scripts/runners/run_semantic_selection.sh" ;;
  flexprefill)     WRAPPER="scripts/runners/run_flexprefill.sh" ;;
  syncspec)        WRAPPER="scripts/runners/run_syncspec.sh" ;;
  *)
    echo "Unknown baseline: $BASELINE" >&2
    echo "Available: longbench_200 vanilla_hf vanilla_fa eagle3 dflash domino sssd fafo llmlingua fastkv rocketkv gemfilter specprefill minference magicdec longspec specextend higoe semantic_selection flexprefill syncspec" >&2
    exit 1
    ;;
esac

exec "$ROOT/$WRAPPER" "$@"
