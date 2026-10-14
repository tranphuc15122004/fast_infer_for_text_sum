#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/run_dflash_context_A_b200.sh"
CORE="$ROOT/scripts/run_dflash_context_experiments.sh"
AUDIT_DIR="$ROOT/data/dflash_context_A_audit"
EXPOSURE_MANIFEST="$AUDIT_DIR/combined_exposure_manifest.json"
EXPOSURE_AUDIT="$AUDIT_DIR/exposure_audit.json"
OUTPUT_ROOT="$ROOT/outputs/dflash_context_interventions"
RUN_ID_PREFIX="dflash_context_A_b200"
GPU_ID="0"

# Use the same external master config and Python 3.12 runtime as the phase
# runner. This also keeps the inline cohort validator on the selected runtime.
# shellcheck disable=SC1091
source "$ROOT/scripts/common/config.sh"
fast_infer_load_config dflash || exit 1
# shellcheck disable=SC1091
source "$ROOT/scripts/common/runtime.sh" || exit 1

usage() {
  cat <<'EOF'
Chạy campaign A1–A3 trên một GPU B200, không cần truyền config hoặc manifest.

  bash scripts/run_dflash_context_A_b200.sh check   # preflight + cohort, không generation
  bash scripts/run_dflash_context_A_b200.sh start   # chạy nền toàn campaign qua các phase gates
  bash scripts/run_dflash_context_A_b200.sh resume RUN_ID    # nối tiếp campaign đã dừng
EOF
}

require_inputs() {
  [[ -f "$CORE" ]] || { echo "Thiếu campaign runner: $CORE" >&2; return 2; }
  [[ -s "$EXPOSURE_MANIFEST" ]] || { echo "Thiếu gói exposure đi kèm repo: $EXPOSURE_MANIFEST" >&2; return 2; }
  [[ -s "$EXPOSURE_AUDIT" ]] || { echo "Thiếu audit exposure đi kèm repo: $EXPOSURE_AUDIT" >&2; return 2; }
}

run_phase() {
  local run_id="$1"
  local mode="$2"
  local -a phase_args=(
    --run-id "$run_id"
    --exposure-manifest "$EXPOSURE_MANIFEST"
    --gpu-ids "$GPU_ID"
    --attention-backend flash_attention_4
    --seed 42
    --mode "$mode"
  )
  if [[ "${DCE_RESUME:-0}" == "1" ]]; then
    phase_args+=(--resume)
  fi
  shift 2
  echo "[$(date -u +%FT%TZ)] phase=$mode run_id=$run_id"
  bash "$CORE" "${phase_args[@]}" "$@"
}

run_phase_resumable() {
  local run_id="$1"
  local mode="$2"
  local status
  shift 2
  if run_phase "$run_id" "$mode" "$@"; then
    return 0
  else
    status=$?
  fi
  # A parity failure is a scientific gate, and an interrupt is an explicit
  # operator action. Retrying either would reuse the same failed evidence.
  if [[ "$status" == "3" || "$status" == "130" ]]; then
    return "$status"
  fi
  echo "[$(date -u +%FT%TZ)] phase=$mode exited $status; retrying once from cell checkpoints" >&2
  export DCE_RESUME=1
  sleep 10
  run_phase "$run_id" "$mode" "$@"
}

verify_prepare() {
  local run_id="$1"
  local run_root="$OUTPUT_ROOT/$run_id"
  "$FAST_INFER_PYTHON" - "$run_root" "$EXPOSURE_MANIFEST" "$EXPOSURE_AUDIT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_root, exposure_path, audit_path = map(Path, sys.argv[1:])
preflight_path = run_root / "preflight.json"
prepare_path = run_root / "prepare_manifest.json"
split_path = run_root / "cohorts" / "split_manifest.json"
if not preflight_path.is_file() or json.loads(preflight_path.read_text()).get("status") != "ready":
    raise SystemExit("CHECK FAILED: B200/FA4/model/data preflight chưa sẵn sàng")
if not prepare_path.is_file() or not split_path.is_file():
    raise SystemExit("CHECK FAILED: prepare chưa tạo được split/cohort")

exposure = json.loads(exposure_path.read_text())
audit = json.loads(audit_path.read_text())
prepare = json.loads(prepare_path.read_text())
split = json.loads(split_path.read_text())
if audit.get("status") != "complete" or exposure.get("review_status") != "complete":
    raise SystemExit("CHECK FAILED: exposure audit chưa complete")
if split.get("status") != "complete":
    raise SystemExit(f"CHECK FAILED: split status={split.get('status')}")

expected_hashes = exposure.get("campaign_corpus_sha256", {})
observed_hashes = {
    Path(item.get("path", "")).name: item.get("sha256")
    for item in prepare.get("source_files", [])
}
hash_mismatches = {
    name: {"expected": digest, "observed": observed_hashes.get(name)}
    for name, digest in expected_hashes.items()
    if observed_hashes.get(name) != digest
}
if hash_mismatches:
    raise SystemExit(f"CHECK FAILED: corpus khác dữ liệu đã audit: {json.dumps(hash_mismatches, ensure_ascii=False)}")

reviewed_ids = {
    str(row.get("sample_id"))
    for row in audit.get("decisions", [])
    if row.get("decision") in {"matched_primary", "absent_verified"}
}
unreviewed = sorted(set(map(str, split.get("unresolved_exposure_sample_ids", []))) - reviewed_ids)
if unreviewed:
    raise SystemExit(f"CHECK FAILED: exposure IDs chưa được audit: {unreviewed[:10]}")

cohorts = {}
for phase in ("pilot", "dev", "confirm"):
    path = run_root / "cohorts" / f"{phase}.json"
    if not path.is_file():
        raise SystemExit(f"CHECK FAILED: thiếu cohort {phase}")
    cohort = json.loads(path.read_text())
    selected = Counter(str(row.get("dataset")) for row in cohort.get("groups", []))
    cohorts[phase] = {
        "status": cohort.get("status"),
        "planned": cohort.get("quotas", {}),
        "available": cohort.get("available_groups", {}),
        "selected": {dataset: selected.get(dataset, 0) for dataset in cohort.get("quotas", {})},
        "shortages": cohort.get("shortages", {}),
    }
    if phase in {"pilot", "dev"} and cohort.get("status") != "complete":
        raise SystemExit(f"CHECK FAILED: cohort {phase} không đủ quota: {cohorts[phase]}")
    if phase == "confirm" and any(selected.get(dataset, 0) < 1 for dataset in cohort.get("quotas", {})):
        raise SystemExit(f"CHECK FAILED: confirm cần ít nhất một nhóm sạch mỗi dataset: {cohorts[phase]}")

result = {
    "status": "ready_partial_confirm" if cohorts["confirm"]["shortages"] else "ready",
    "gpu_generation_started": False,
    "run_id": run_root.name,
    "corpus_hashes": {name: observed_hashes.get(name) for name in expected_hashes},
    "exposure_audit": audit.get("decision_counts", {}),
    "cohorts": cohorts,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

check_command() {
  require_inputs
  local run_id="${RUN_ID_PREFIX}_check_$(date -u +%Y%m%dT%H%M%SZ)"
  run_phase "$run_id" preflight
  run_phase "$run_id" prepare
  verify_prepare "$run_id"
}

campaign_command() {
  local run_id="$1"
  local resume_mode="${2:-0}"
  require_inputs
  if [[ "$resume_mode" == "1" ]]; then
    export DCE_RESUME=1
    echo "[$(date -u +%FT%TZ)] resuming campaign run_id=$run_id"
  else
    unset DCE_RESUME || true
  fi
  run_phase "$run_id" preflight
  run_phase "$run_id" prepare
  verify_prepare "$run_id"
  run_phase_resumable "$run_id" validate
  run_phase_resumable "$run_id" pilot
  run_phase_resumable "$run_id" dev
  # Preregistered default from the B200 runbook; lock still enforces dev gates.
  run_phase "$run_id" lock --a3-selector previous_dflash --a3-budget 2048 --refresh-period 4
  run_phase_resumable "$run_id" confirm \
    --locked-config "$OUTPUT_ROOT/$run_id/locked_config.json" \
    --allow-partial-confirm
  run_phase "$run_id" report --report-phase confirm
  echo "[$(date -u +%FT%TZ)] campaign complete run_id=$run_id"
  echo "Report: $OUTPUT_ROOT/$run_id/reports/confirm_decision.md"
}

campaign_is_running() {
  local run_root="$1"
  local run_id="$2"
  local pid cmdline proc_cmd
  if [[ -s "$run_root/campaign.pid" ]]; then
    pid="$(<"$run_root/campaign.pid")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null \
        && [[ -r "/proc/$pid/cmdline" ]]; then
      cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" || true)"
      if [[ "$cmdline" == *"$SELF"* && "$cmdline" == *"_campaign"* && "$cmdline" == *"$run_id"* ]]; then
        return 0
      fi
    fi
  fi
  # The launcher can die while its Python child is still writing. Scan Linux
  # command lines as well as the PID file so resume cannot start a second
  # writer against the same checkpoint directory.
  for proc_cmd in /proc/[0-9]*/cmdline; do
    [[ -r "$proc_cmd" ]] || continue
    pid="${proc_cmd#/proc/}"
    pid="${pid%/cmdline}"
    cmdline="$(tr '\0' ' ' <"$proc_cmd" 2>/dev/null || true)"
    if [[ "$cmdline" == *"$SELF _campaign $run_id"* \
        || "$cmdline" == *"run_dflash_context_experiments.sh --run-id $run_id"* \
        || "$cmdline" == *"run_dflash_context_experiments.py --run-id $run_id"* ]]; then
      return 0
    fi
  done
  return 1
}

launch_campaign() {
  local run_id="$1"
  local resume_mode="$2"
  local run_root="$OUTPUT_ROOT/$run_id"
  local log_path="$run_root/campaign.log"
  mkdir -p "$run_root"
  command -v flock >/dev/null 2>&1 || {
    echo "Thiếu flock của util-linux; không thể bảo vệ checkpoint khỏi hai campaign cùng RUN_ID." >&2
    return 2
  }
  if campaign_is_running "$run_root" "$run_id"; then
    echo "Campaign $run_id is already running (PID $(<"$run_root/campaign.pid"))." >&2
    return 2
  fi
  local lock_fd
  exec {lock_fd}>"$run_root/campaign.lock"
  if ! flock -n "$lock_fd"; then
    exec {lock_fd}>&-
    echo "Campaign $run_id is already starting or running." >&2
    return 2
  fi
  local pid
  if [[ "$resume_mode" == "1" ]]; then
    printf '\n[%s] resume requested\n' "$(date -u +%FT%TZ)" >>"$log_path"
    nohup bash "$SELF" _campaign "$run_id" --resume >>"$log_path" 2>&1 </dev/null &
  else
    nohup bash "$SELF" _campaign "$run_id" >"$log_path" 2>&1 </dev/null &
  fi
  pid=$!
  printf '%s\n' "$pid" >"$run_root/campaign.pid.tmp.$$"
  mv "$run_root/campaign.pid.tmp.$$" "$run_root/campaign.pid"
  if [[ "$resume_mode" == "1" ]]; then
    printf 'Đã nối tiếp campaign.\nRUN_ID=%s\nPID=%s\nLOG=%s\n' "$run_id" "$pid" "$log_path"
  else
    printf 'Đã treo campaign A1–A3.\nRUN_ID=%s\nPID=%s\nLOG=%s\n' "$run_id" "$pid" "$log_path"
  fi
  printf 'Theo dõi: tail -f %q\n' "$log_path"
}

start_command() {
  require_inputs
  local run_id="${RUN_ID_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)"
  local run_root="$OUTPUT_ROOT/$run_id"
  if [[ -e "$run_root" ]]; then
    echo "Run directory already exists: $run_root; use resume only if its source signature matches." >&2
    return 2
  fi
  mkdir -p "$OUTPUT_ROOT"
  if ! mkdir "$run_root" 2>/dev/null; then
    echo "Run directory was created concurrently: $run_root" >&2
    return 2
  fi
  launch_campaign "$run_id" 0
}

resume_command() {
  require_inputs
  local run_id="${1:-}"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "resume requires a simple RUN_ID (letters, digits, underscore, hyphen)." >&2
    return 2
  fi
  local run_root="$OUTPUT_ROOT/$run_id"
  if [[ ! -d "$run_root" ]]; then
    echo "Run directory not found: $run_root" >&2
    return 2
  fi
  if [[ -s "$run_root/campaign.status" ]] && grep -qx 'exit_code=0' "$run_root/campaign.status"; then
    echo "Campaign $run_id already completed successfully." >&2
    return 2
  fi
  launch_campaign "$run_id" 1
}

case "${1:-}" in
  check)
    shift
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    check_command
    ;;
  start)
    shift
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    start_command
    ;;
  resume)
    shift
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    resume_command "$1"
    ;;
  _campaign)
    shift
    [[ $# -eq 1 || ( $# -eq 2 && "$2" == "--resume" ) ]] || { usage >&2; exit 2; }
    run_id="$1"
    run_root="$OUTPUT_ROOT/$run_id"
    [[ "$run_id" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid RUN_ID: $run_id" >&2; exit 2; }
    mkdir -p "$run_root"
    printf 'running\nstarted_at=%s\n' "$(date -u +%FT%TZ)" >"$run_root/campaign.status.tmp.$$"
    mv "$run_root/campaign.status.tmp.$$" "$run_root/campaign.status"
    campaign_exit_status() {
      local exit_code=$?
      trap - EXIT
      printf 'exit_code=%s\nfinished_at=%s\n' "$exit_code" "$(date -u +%FT%TZ)" \
        >"$run_root/campaign.status.tmp.$$"
      mv "$run_root/campaign.status.tmp.$$" "$run_root/campaign.status" || true
      echo "[$(date -u +%FT%TZ)] campaign exit_code=$exit_code run_id=$run_id" >&2
      exit "$exit_code"
    }
    trap campaign_exit_status EXIT
    if [[ $# -eq 2 ]]; then
      campaign_command "$run_id" 1
    else
      campaign_command "$run_id" 0
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
