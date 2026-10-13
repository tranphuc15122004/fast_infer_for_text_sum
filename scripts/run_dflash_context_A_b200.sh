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

usage() {
  cat <<'EOF'
Chạy campaign A1–A3 trên một GPU B200, không cần truyền config hoặc manifest.

  bash scripts/run_dflash_context_A_b200.sh check   # preflight + cohort, không generation
  bash scripts/run_dflash_context_A_b200.sh start   # chạy nền toàn campaign qua các phase gates
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
  shift 2
  echo "[$(date -u +%FT%TZ)] phase=$mode run_id=$run_id"
  bash "$CORE" \
    --run-id "$run_id" \
    --exposure-manifest "$EXPOSURE_MANIFEST" \
    --gpu-ids "$GPU_ID" \
    --attention-backend flash_attention_4 \
    --seed 42 \
    --mode "$mode" "$@"
}

verify_prepare() {
  local run_id="$1"
  local run_root="$OUTPUT_ROOT/$run_id"
  python3 - "$run_root" "$EXPOSURE_MANIFEST" "$EXPOSURE_AUDIT" <<'PY'
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
    if phase == "confirm" and any(count < 1 for count in selected.values()):
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
  require_inputs
  run_phase "$run_id" preflight
  run_phase "$run_id" prepare
  verify_prepare "$run_id"
  run_phase "$run_id" validate
  run_phase "$run_id" pilot
  run_phase "$run_id" dev
  # Preregistered default from the B200 runbook; lock still enforces dev gates.
  run_phase "$run_id" lock --a3-selector previous_dflash --a3-budget 2048 --refresh-period 4
  run_phase "$run_id" confirm \
    --locked-config "$OUTPUT_ROOT/$run_id/locked_config.json" \
    --allow-partial-confirm
  run_phase "$run_id" report --report-phase confirm
  echo "[$(date -u +%FT%TZ)] campaign complete run_id=$run_id"
  echo "Report: $OUTPUT_ROOT/$run_id/reports/confirm_decision.md"
}

start_command() {
  require_inputs
  local run_id="${RUN_ID_PREFIX}_$(date -u +%Y%m%dT%H%M%SZ)"
  local run_root="$OUTPUT_ROOT/$run_id"
  mkdir -p "$run_root"
  if [[ -e "$run_root/campaign.pid" ]]; then
    echo "Run directory already has a campaign PID: $run_root/campaign.pid" >&2
    return 2
  fi
  nohup bash "$SELF" _campaign "$run_id" \
    >"$run_root/campaign.log" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$run_root/campaign.pid"
  printf 'Đã treo campaign A1–A3.\nRUN_ID=%s\nPID=%s\nLOG=%s\n' \
    "$run_id" "$pid" "$run_root/campaign.log"
  printf 'Theo dõi: tail -f %q\n' "$run_root/campaign.log"
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
  _campaign)
    shift
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    campaign_command "$1"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
