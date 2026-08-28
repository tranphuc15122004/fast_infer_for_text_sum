#!/usr/bin/env bash
# Dispatcher: maps a baseline name to its launcher. Every launcher loads the
# same external master through config/master.path.
#
# Usage:
#   bash scripts/run.sh <baseline> [extra args...]
#
# Each baseline has a wrapper `scripts/run_<baseline>.sh` that invokes the
# shared Python 3.12 interpreter after loading the master config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${1:?usage: run.sh <baseline> [args...]}"
shift || true

case "$BASELINE" in
  longbench_200) WRAPPER="scripts/run_longbench_200.sh" ;;
  vanilla_hf)   WRAPPER="scripts/run_vanilla_hf.sh" ;;
  vanilla_fa)   WRAPPER="scripts/run_vanilla_fa.sh" ;;
  eagle3)      WRAPPER="scripts/run_eagle3_qwen3.sh" ;;
  dflash)      WRAPPER="scripts/run_dflash.sh" ;;
  sssd)       WRAPPER="scripts/run_sssd.sh" ;;
  fafo)       WRAPPER="scripts/run_fafo.sh" ;;
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
    echo "Available: longbench_200 vanilla_hf vanilla_fa eagle3 dflash sssd fafo llmlingua fastkv rocketkv gemfilter specprefill minference magicdec longspec specextend higoe semantic_selection flexprefill" >&2
    exit 1
    ;;
esac

exec "$ROOT/$WRAPPER" "$@"
