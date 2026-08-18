#!/usr/bin/env bash
# Benchmark runner: chạy infer các baseline có adapter trên
# data/representative_100 và thu thập metric tốc độ + semantic.
#
# Usage:
#   bash scripts/run_representative_100.sh [options]
#
# Options:
#   --baselines a,b,c      baseline có adapter representative (mặc định: 8 baseline)
#   --datasets a,b,c       dataset trong data/representative_100 (mặc định: tất cả)
#   --max-samples N        số mẫu / (baseline, dataset) [smoke=5, full=100]
#   --max-new-tokens N     override độ dài sinh (theo biến của từng baseline)
#   --mode smoke|full      smoke = cấu hình T4-safe, full = cấu hình đầy đủ
#   --config FILE          env defaults (mặc định config/representative_100.env)
#   --output-dir DIR       nơi chứa outputs/logs/configs (mặc định outputs/representative_100)
#   --include-unsupported  chạy thêm smoke probe ngoài benchmark representative
#   --dry-run              chỉ in kế hoạch, không chạy
#   --skip-collect         bỏ qua bước tổng hợp collect_metrics.py
#
# Sau cùng tự động chạy scripts/collect_metrics.py → metrics_summary.{json,csv,md}.
set -uo pipefail   # không dùng -e: vẫn chạy tiếp các baseline khác khi 1 baseline fail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# uv needs a writable cache for its lock files.  The default user cache can be
# read-only in managed/CI environments, so use a writable temporary fallback.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/fast_infer_uv_cache}"
mkdir -p "$UV_CACHE_DIR"

# ---- defaults ---------------------------------------------------------------
MODE="full"
MAX_SAMPLES=""
MAX_NEW_TOKENS=""
OUTPUT_DIR="outputs/representative_100"
CONFIG_FILE="$ROOT/config/representative_100.env"
BASELINES=""
DATASETS=""
INCLUDE_UNSUPPORTED=0
DRY_RUN=0
SKIP_COLLECT=0

# ---- resolve config path before loading defaults -----------------------------
# Parse only --config in this pre-pass so a custom config is actually sourced.
# The full CLI pass below still runs afterwards, therefore CLI values win.
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--config" ]]; then
    if (( i + 1 >= ${#ARGS[@]} )); then
      echo "--config requires a file path" >&2
      exit 2
    fi
    CONFIG_FILE="${ARGS[$((i + 1))]}"
    ((i++))
  fi
done

# ---- optional defaults env (đọc TRƯỚC CLI để CLI thắng) ----------------------
if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

# ---- CLI --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baselines)      BASELINES="$2"; shift 2 ;;
    --datasets)       DATASETS="$2"; shift 2 ;;
    --max-samples)    MAX_SAMPLES="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --mode)           MODE="$2"; shift 2 ;;
    --config)         CONFIG_FILE="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --include-unsupported) INCLUDE_UNSUPPORTED=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --skip-collect)   SKIP_COLLECT=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Documentation accepts comma-separated values; normalize them for the loops.
BASELINES="${BASELINES//,/ }"
DATASETS="${DATASETS//,/ }"

case "$MODE" in
  smoke|full) ;;
  *) echo "bad --mode: $MODE (smoke|full)" >&2; exit 2 ;;
esac

REPRESENTATIVE_BASELINES="llmlingua fastkv gemfilter specprefill minference specextend eagle3 semantic_selection"
UNSUPPORTED_BASELINES="higoe dflash rocketkv magicdec longspec"
ALL_BASELINES="$REPRESENTATIVE_BASELINES $UNSUPPORTED_BASELINES"

if [[ -z "$BASELINES" ]]; then
  BASELINES="$REPRESENTATIVE_BASELINES"
fi

if [[ -z "$DATASETS" ]]; then
  DATASETS=""
  for f in data/representative_100/*_representative.jsonl; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f" _representative.jsonl)
    DATASETS="$DATASETS $name"
  done
  DATASETS="$(echo $DATASETS | xargs)"   # trim
fi

if [[ -z "$MAX_SAMPLES" ]]; then
  if [[ "$MODE" == "full" ]]; then MAX_SAMPLES="100"; else MAX_SAMPLES="5"; fi
fi

if [[ "$OUTPUT_DIR" = /* ]]; then
  OUT_DIR="$OUTPUT_DIR"
else
  OUT_DIR="$ROOT/$OUTPUT_DIR"
fi
mkdir -p "$OUT_DIR/configs" "$OUT_DIR/logs" "$OUT_DIR/data" "$OUT_DIR/smoke"

# Refuse to silently mix non-representative smoke probes into the benchmark.
# They can be requested explicitly for environment diagnostics, but they do
# not count toward representative_100 completeness or metrics.
for b in $BASELINES; do
  case " $ALL_BASELINES " in
    *" $b "*) ;;
    *)
      echo "unknown baseline: $b" >&2
      echo "available: $ALL_BASELINES" >&2
      exit 2
      ;;
  esac
  if [[ "$INCLUDE_UNSUPPORTED" != "1" ]]; then
    case " $UNSUPPORTED_BASELINES " in
      *" $b "*)
        echo "unsupported baseline for representative_100: $b" >&2
        echo "Use a data adapter first, or pass --include-unsupported for a separate smoke probe." >&2
        exit 2
        ;;
    esac
  fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  # Never leave a previous partial report looking like the current run.
  rm -f "$OUT_DIR/metrics_summary.json" \
        "$OUT_DIR/metrics_summary.csv" \
        "$OUT_DIR/metrics_summary.md"
fi

echo "== run_representative_100: mode=$MODE max_samples=$MAX_SAMPLES"
echo "   baselines: $BASELINES"
echo "   datasets : $DATASETS"
echo "   out dir  : $OUT_DIR"

# ---- chuyển data sang format riêng của eagle3 (turns) và specextend (text) --
convert_data() {
  local ds="$1"
  python3 - "$ds" "$MAX_SAMPLES" "$OUT_DIR/data" <<'PY'
import json
import pathlib
import sys

ds, max_samples, out_dir = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
src = pathlib.Path("data/representative_100") / (ds + "_representative.jsonl")
rows = [
    json.loads(line)
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
][:max_samples]
eagle, spec = [], []
for r in rows:
    doc = r.get("document") or r.get("text") or ""
    ref = r.get("reference") or r.get("summary") or r.get("answer")
    rid = r.get("id")
    eagle.append({
        "question_id": rid,
        "turns": ["Summarize the following document.\n\n" + doc],
        "reference": ref,
    })
    spec.append({
        "text": "Summarize the following text into a summary of less than 800 words.\n### Text:\n\n" + doc,
        "reference": ref,
    })
(out_dir / ("eagle3_" + ds + ".jsonl")).write_text(
    "\n".join(json.dumps(e, ensure_ascii=False) for e in eagle) + "\n",
    encoding="utf-8",
)
(out_dir / ("specextend_" + ds + ".jsonl")).write_text(
    "\n".join(json.dumps(s, ensure_ascii=False) for s in spec) + "\n",
    encoding="utf-8",
)
print("  converted " + ds + ": " + str(len(rows)) + " records -> eagle3/specextend")
PY
}

# ---- sinh config cho (baseline, dataset) -------------------------------------
# set_env KEY VALUE  -> dòng 'KEY="VALUE"' cho env file (giá trị không chứa nháy đơn)
set_env() { echo "$1='$2'"; }

# Canonical model matrix for the large-GPU representative benchmark.  Values
# are overridable from config/representative_100.env or a custom --config.
REP_TARGET_MODEL="${REP_TARGET_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
REP_SPEC_MODEL="${REP_SPEC_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
REP_EAGLE_MODEL="${REP_EAGLE_MODEL:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
REP_DFLASH_MODEL="${REP_DFLASH_MODEL:-z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat}"
REP_COMPRESSOR_MODEL="${REP_COMPRESSOR_MODEL:-microsoft/llmlingua-2-xlm-roberta-large-meetingbank}"
REP_EMBEDDING_MODEL="${REP_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
REP_VICUNA_MODEL="${REP_VICUNA_MODEL:-lmsys/vicuna-7b-v1.5-16k}"
REP_SPECEXTEND_DRAFT_MODEL="${REP_SPECEXTEND_DRAFT_MODEL:-double7/vicuna-68m}"

# Resolve a cached HF repo to its snapshot when possible.  This matters for
# EAGLE3, whose wrapper validates that the draft checkpoint is a local dir;
# ordinary Transformers/vLLM baselines can still receive the HF id fallback.
resolve_model_ref() {
  local ref="$1"
  if [[ "$ref" == /* ]]; then
    echo "$ref"
    return
  fi
  python3 - "$ref" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/common").resolve().parent))
from common.paths import snapshot_dir

repo = sys.argv[1]
cached = snapshot_dir(repo)
print(str(cached) if cached is not None else repo)
PY
}

apply_full_model_overrides() {
  local b="$1"
  [[ "$MODE" == "full" ]] || return 0
  case "$b" in
    llmlingua)
      set_env COMPRESSOR_MODEL "$(resolve_model_ref "$REP_COMPRESSOR_MODEL")"
      set_env TARGET_MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      ;;
    fastkv)
      set_env MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      set_env METHOD "fastkv"
      set_env ATTN_IMPL "${FASTKV_FULL_ATTN_IMPL:-flash_attention_2}"
      ;;
    gemfilter)
      set_env MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      set_env SELECT_LAYER_IDX "13"
      ;;
    specprefill)
      set_env TARGET_MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      set_env SPEC_MODEL "$(resolve_model_ref "$REP_SPEC_MODEL")"
      ;;
    minference)
      set_env MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      ;;
    eagle3)
      set_env BASE_MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      set_env EAGLE_MODEL "$(resolve_model_ref "$REP_EAGLE_MODEL")"
      ;;
    specextend)
      set_env MODEL_NAME "vicuna_7b"
      set_env BASE_MODEL "$(resolve_model_ref "$REP_VICUNA_MODEL")"
      set_env DRAFT_MODEL "$(resolve_model_ref "$REP_SPECEXTEND_DRAFT_MODEL")"
      ;;
    semantic_selection)
      set_env MODEL "$(resolve_model_ref "$REP_TARGET_MODEL")"
      set_env EMBEDDING_MODEL "$(resolve_model_ref "$REP_EMBEDDING_MODEL")"
      ;;
  esac
}

# tên config của baseline (một số baseline có tên khác với tên dispatcher)
config_for() {
  case "$1:$MODE" in
    fastkv:smoke)      echo "fastkv_smoke" ;;
    gemfilter:smoke)   echo "gemfilter_smoke" ;;
    specprefill:smoke) echo "specprefill_smoke" ;;
    specextend:smoke)  echo "specextend_smoke" ;;
    eagle3:*)          echo "eagle3_qwen3" ;;
    dflash:*)          echo "dflash_smoke" ;;
    *)                 echo "$1" ;;
  esac
}

gen_config() {
  local b="$1" ds="$2"
  local base_cfg="$ROOT/config/$(config_for "$b").env"
  if [[ ! -f "$base_cfg" ]]; then
    echo "  missing config: $base_cfg (skip)" >&2
    return 1
  fi
  local cfg="$OUT_DIR/configs/${b}_${ds}.env"
  {
    cat "$base_cfg"
    echo
    echo "# ---- overrides by run_representative_100.sh ----"
    if [[ "$MODE" == "smoke" ]]; then set_env SMOKE 1; else set_env SMOKE 0; fi
    set_env OUTPUT_FILE "$OUT_DIR/${b}_${ds}.jsonl"
    set_env MAX_SAMPLES "$MAX_SAMPLES"
    apply_full_model_overrides "$b"
    case "$b" in
      llmlingua)
        set_env DOC_FILE "data/representative_100/${ds}_representative.jsonl"
        ;;
      fastkv|gemfilter|specprefill|minference)
        set_env DATA_FILE "data/representative_100/${ds}_representative.jsonl"
        ;;
      eagle3)
        set_env DATA_FILE "$OUT_DIR/data/eagle3_${ds}.jsonl"
        set_env QUESTION_BEGIN "0"
        set_env QUESTION_END "$MAX_SAMPLES"
        ;;
      semantic_selection)
        set_env INPUT_FILE "data/representative_100/${ds}_representative.jsonl"
        ;;
      specextend)
        set_env INPUT_FILE "$OUT_DIR/data/specextend_${ds}.jsonl"
        ;;
    esac
    if [[ -n "$MAX_NEW_TOKENS" ]]; then
      case "$b" in
        gemfilter|specextend) set_env MAX_GEN_LEN "$MAX_NEW_TOKENS" ;;
        specprefill)          set_env MAX_TOKENS "$MAX_NEW_TOKENS" ;;
        *)                    set_env MAX_NEW_TOKENS "$MAX_NEW_TOKENS" ;;
      esac
    fi
  } > "$cfg"
  echo "$cfg"
}

# ---- chạy -------------------------------------------------------------------
PASSED=0
FAILED=0
FAILED_LIST=""
COLLECT_FAILED=0

run_pair() {
  local b="$1" ds="$2" cfg="$3" output_file="${4:-}"
  [[ -n "$output_file" ]] || output_file="$OUT_DIR/${b}_${ds}.jsonl"
  local log="$OUT_DIR/logs/${b}_${ds}.log"
  # Infer scripts append JSONL records. Remove only this run's generated
  # artifact so rerunning the same pair does not duplicate samples/summaries.
  mkdir -p "$(dirname "$output_file")"
  echo
  echo "== [$b / $ds] =="
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  would run: bash scripts/run.sh $b $cfg"
    echo "  log       : $log"
    return
  fi
  rm -f "$output_file"
  local start=$(date +%s)
  if bash scripts/run.sh "$b" "$cfg" > "$log" 2>&1; then
    PASSED=$((PASSED + 1))
    echo "  PASS ($(( $(date +%s) - start ))s) — log: $log"
  else
    FAILED=$((FAILED + 1))
    FAILED_LIST="$FAILED_LIST $b/$ds"
    echo "  FAIL — log: $log (tail:)"
    tail -8 "$log" | sed 's/^/    /'
  fi
}

# convert data trước (chỉ khi có baseline cần)
if [[ "$BASELINES" == *eagle3* || "$BASELINES" == *specextend* ]]; then
  echo "== converting data formats (eagle3/specextend) =="
  for ds in $DATASETS; do
    convert_data "$ds"
  done
fi

for b in $BASELINES; do
  if [[ " $REPRESENTATIVE_BASELINES " != *" $b "* ]]; then
    # baseline smoke/pipeline riêng: chạy 1 lần theo config smoke của nó
    if [[ "$DRY_RUN" == "1" ]]; then
      echo
      echo "== [$b (smoke-only: không đọc DATA_FILE)] =="
      echo "  would run: bash scripts/run.sh $b (smoke probe riêng, không tính metric)"
    else
      base_cfg="$ROOT/config/$(config_for "$b").env"
      if [[ ! -f "$base_cfg" ]]; then
        echo "  missing config: $base_cfg (skip)" >&2
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $b/smoke"
        continue
      fi
      cfg="$OUT_DIR/configs/${b}_smoke.env"
      {
        cat "$base_cfg"
        echo
        echo "# ---- overrides by run_representative_100.sh ----"
        set_env SMOKE 1
        set_env OUTPUT_FILE "$OUT_DIR/smoke/${b}_smoke.jsonl"
      } > "$cfg"
      run_pair "$b" "smoke" "$cfg" "$OUT_DIR/smoke/${b}_smoke.jsonl"
    fi
    continue
  fi
  for ds in $DATASETS; do
    cfg=$(gen_config "$b" "$ds") || { FAILED=$((FAILED + 1)); FAILED_LIST="$FAILED_LIST $b/$ds"; continue; }
    run_pair "$b" "$ds" "$cfg"
  done
done

echo
echo "================ benchmark summary ================"
echo "PASS=$PASSED  FAIL=$FAILED"
if [[ -n "$FAILED_LIST" ]]; then
  echo "failed runs:$FAILED_LIST"
fi

# ---- tổng hợp metric ----------------------------------------------------------
if [[ "$SKIP_COLLECT" == "1" || "$DRY_RUN" == "1" ]]; then
  [[ "$DRY_RUN" == "1" ]] && echo "(dry-run: bỏ qua collect_metrics.py)"
  exit $((FAILED > 0 ? 1 : 0))
fi

echo
echo "== collecting metrics =="
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
EXPECTED_BASELINES=""
for b in $BASELINES; do
  case " $REPRESENTATIVE_BASELINES " in
    *" $b "*) EXPECTED_BASELINES="$EXPECTED_BASELINES $b" ;;
  esac
done
EXPECTED_BASELINES="$(echo "$EXPECTED_BASELINES" | xargs)"

if [[ -z "$EXPECTED_BASELINES" ]]; then
  echo "ERROR: no representative baseline selected; nothing to collect" >&2
  COLLECT_FAILED=1
elif uv run --project "$ROOT" --locked python "$ROOT/scripts/collect_metrics.py" \
     --outputs-dir "$OUT_DIR" --data-dir "data/representative_100" \
     --strict \
     --expected-baselines "$EXPECTED_BASELINES" \
     --expected-datasets "$DATASETS" \
     --expected-samples "$MAX_SAMPLES"; then
  echo "Metrics saved to $OUT_DIR/metrics_summary.{json,csv,md}"
else
  echo "ERROR: collect_metrics.py failed (xem output bên trên)" >&2
  COLLECT_FAILED=1
fi

exit $((FAILED > 0 || COLLECT_FAILED > 0 ? 1 : 0))
