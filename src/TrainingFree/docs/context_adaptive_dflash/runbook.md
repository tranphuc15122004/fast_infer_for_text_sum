# Runbook Context-Adaptive DFlash

Ngày: **2026-10-09**. Trạng thái: **executor/launcher đã có; chưa có kết quả GPU hoặc parity report**. Đọc [ma trận thực nghiệm](experiment_matrix.md), [kiểm chứng GPU](gpu_validation.md), [mẫu báo cáo](results_template.md), [protocol](experiment_protocol.md) và [measurement contract](measurement_contract.md).

Các block dưới đây dùng chung một shell, chạy theo thứ tự. Review gate trước bước tiếp theo; không paste toàn runbook thành job chưa kiểm chứng.

## 1. Runtime và asset

Trên server dùng Python 3.12 từ PATH, model snapshots local và dependency đã mirror. Không cài online hoặc tạo venv riêng cho phương pháp.

```bash
set -euo pipefail
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
python3 scripts/setup_server_env.py --check
python3 scripts/check_shared_env.py

export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
export CUDA_VISIBLE_DEVICES=0
export SMOKE=0 FULL=0
export RUN_TEMPERATURE=0 LONG_BENCH_SEED=42 RUN_MAX_INPUT_TOKENS=0
export CAD_MAX_NEW_TOKENS=2048 RUN_MAX_NEW_TOKENS=2048
export CAD_BUDGETS=1024,2048,4096,full CAD_GAMMAS=3,7,11,15
export CAD_SELECTOR=target_parent CAD_TARGET_LAYER=middle
export CAD_REFRESH_PERIOD=4 CAD_SIGNAL_UPDATE_PERIOD=1
export CAD_SOURCE_CHUNK_SIZE=128 CAD_SOURCE_ANCHORS=64 CAD_RECENT_OUTPUT=256
export CAD_PRIOR_STRENGTH=16 CAD_MIN_STATE_SUPPORT=8 CAD_STATISTICS_DECAY=0.95
export CAD_HISTORY_ALPHA=0.1 CAD_LATENCY_EMA_ALPHA=0.1 CAD_CONTROLLER_MARGIN=0.02
export CAD_ENTROPY_SIGNAL_TEMPERATURE=1 CAD_TIMING_MODE=wall
export CAD_LENGTH_MODE=draft_shape CAD_COST_UPDATE_MODE=frozen_cost
export CAD_STATISTICS_UPDATE_MODE=online
```

Master config ở `config/master.path` trỏ tới shell-env ngoài repository. Cấu hình `MODEL_TARGET`, `MODEL_DFLASH_DRAFT`, `DATA_INPUT` (hoặc `CAD_DATA_DIR`), `RUN_MAX_NEW_TOKENS`, `RUN_MAX_INPUT_TOKENS`, `RUN_TEMPERATURE` và `LONG_BENCH_SEED` như DFlash baseline. Các `CAD_*` caller overrides được runner giữ lại khi load master. Token HF chỉ dùng nếu snapshot local cần xác thực; runner bật offline mode khi load model.

Master phải dùng Qwen3-4B + Qwen3-4B-DFlash-b16 snapshots local. Kiểm aliases `TARGET_MODEL`/`DRAFT_MODEL` và `TEMPERATURE`/`MAX_INPUT_TOKENS` nếu có caller override. Manifest ghi effective settings; không dump master-env chứa secrets. Backend/dtype tự chọn theo GPU/kernels, chưa có CLI ép backend/dtype.

`prepare`, `preflight`, `lock`, `report` không cần model forward. `preflight --model-check` đọc config snapshot local. `smoke`, `calibrate`, `dev`, `test` yêu cầu CUDA và load model; máy local T4 chỉ chạy CPU, không dùng để benchmark.

## 2. Chuẩn bị split và preflight

Đặt `CAD_RUN_ID` duy nhất cho campaign cùng code/data/runtime. CLI đọc mọi JSONL trong data-dir; stage chỉ ba datasets summarization. Giữ canonical files và symlinks bất biến suốt run. Audit 10 probe IDs theo [matrix §3](experiment_matrix.md#3-cohort-primary-và-exposure-audit) và lưu preregistration trước forward.

```bash
export CAD_RUN_ID=cadflash_primary_20261009_001
export CAD_RUN_ROOT=outputs/context_adaptive_dflash/$CAD_RUN_ID
export CAD_OUTPUT_ROOT="$CAD_RUN_ROOT"
export CAD_CANONICAL_DATA=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k
export CAD_DATA_DIR="$CAD_RUN_ROOT/inputs/primary"
test ! -e "$CAD_RUN_ROOT"
mkdir -p "$CAD_DATA_DIR" "$CAD_RUN_ROOT/evidence"
for cad_dataset in gov_report qmsum multi_news; do
  test -f "$CAD_CANONICAL_DATA/$cad_dataset.jsonl"
  ln -s "$CAD_CANONICAL_DATA/$cad_dataset.jsonl" "$CAD_DATA_DIR/$cad_dataset.jsonl"
done

# File bổ sung phải được tạo từ đối soát thật theo matrix §3.
CAD_EXPOSURE_ARGS=(
  --exposure-manifest outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/manifest.json
  --exposure-manifest "$CAD_RUN_ROOT/evidence/resolved_exposure_manifest.json"
)

bash scripts/run.sh context_adaptive_dflash --phase prepare \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  "${CAD_EXPOSURE_ARGS[@]}"

bash scripts/run.sh context_adaptive_dflash --phase preflight \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" --model-check
```

`prepare` ghi `split_manifest.json` và checksum dataset. Nó trả exit code khác 0 khi mỗi dataset/source stratum không đủ nhóm cho calibration/dev/test; không được tiếp tục bằng split tổng hợp. `preflight` kiểm tra vocab, hidden size, target feature layers, mask token, DFlash block size và sliding-window config. Nếu model chưa được mirror thì bỏ `--model-check` để chỉ kiểm dữ liệu.

Nếu audit xác minh mọi probe IDs absent, chỉ truyền manifest gốc và lưu `absent_verified` evidence; không dùng file rỗng thay audit. CLI không fail vì unresolved IDs: reviewer phải xác minh từng entry, exposed groups chỉ dev, không overlap, status/counts split đầy đủ. Còn `unresolved` thì chưa claim untouched heldout. Config preflight thành công chưa đủ G0.

## 3. GPU smoke và calibration

Smoke tối đa 32 token, hai source groups/dataset, một query/group. **Phase smoke hiện không tự lọc dev bằng split manifest**; xuất dev cohort riêng trước khi chạy:

```bash
export CAD_SMOKE_DATA_DIR="$CAD_RUN_ROOT/inputs/smoke_dev"
PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path
from TrainingFree.context_adaptive.benchmark import load_corpus, select_split

root = Path(os.environ["CAD_RUN_ROOT"])
split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
samples = load_corpus(data_file=None, data_dir=Path(os.environ["CAD_DATA_DIR"]))
dev = select_split(samples, split, "dev")
destination = Path(os.environ["CAD_SMOKE_DATA_DIR"])
destination.mkdir(parents=True, exist_ok=False)
groups, rows = defaultdict(set), defaultdict(list)
for sample in dev:
    dataset, group = sample["dataset"], sample["_cad_source_group_id"]
    if group in groups[dataset] or len(groups[dataset]) >= 2:
        continue
    groups[dataset].add(group)
    row = dict(sample["raw"])
    row.update(id=sample["id"], dataset=dataset)
    rows[dataset].append(row)
assert set(rows) == {"gov_report", "qmsum", "multi_news"}
assert all(len(values) == 2 for values in rows.values())
for dataset, values in rows.items():
    path = destination / f"{dataset}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
print({dataset: len(values) for dataset, values in rows.items()})
PY

for cad_variant in ar dflash_full_fixed; do
  bash scripts/run.sh context_adaptive_dflash --phase smoke --variant "$cad_variant" \
    --data-dir "$CAD_SMOKE_DATA_DIR" --output-root "$CAD_RUN_ROOT/smoke_check" \
    --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
    --max-samples 6 --max-new-tokens 32 --repetitions 1 --warmup-runs 1 \
    --fixed-budget full --fixed-gamma 15 --gamma-reference 15
done
bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$CAD_RUN_ROOT/smoke_check" \
  --report-phase smoke --report-statistics-mode online --bootstrap-samples 10000
```

Smoke root riêng vì corpus signature khác primary. Kiểm actual groups/IDs đều dev. Gamma sweep cần roots riêng; không đổi setting trong cell đã có manifest. Smoke report chỉ kiểm paired outputs trên cohort này. Hoàn thành harness checkpoint thật và mở rộng 8 GovReport + 8 Multi-News + 4 QMSum dev groups, cap 256 theo [GPU validation](gpu_validation.md) trước matrix lớn. Chưa có phase parity CLI; không coi smoke exit 0 là G1/G2 pass.

V1 calibration hiện chỉ dùng greedy target (`--temperature 0`) và full fixed trajectory. Mỗi checkpoint chạy lại từng feasible `(budget,gamma)` trên bản clone target cache của cùng prefix; clone setup được ghi riêng, còn controller cost được profile sau khi action table đã fit.

```bash
bash scripts/run.sh context_adaptive_dflash --phase calibrate \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT/pilot" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --max-samples 16 \
  --max-new-tokens 2048 --calibration-checkpoints 0,512,1024,1536 \
  --fixed-budget full --fixed-gamma 15 --gamma-reference 15 \
  --repetitions 3 --warmup-runs 3
```

Đây là **pilot**, tối đa 16 records, ưu tiên source groups/cân bằng dataset; manifest khóa cohort. Kiểm actual checkpoint positions, unique prefix support và context/refresh cost buckets. Repetitions greedy không tăng outcome support; EOS sớm thiếu mốc là hợp lệ. Sau pilot, full calibration không sample cap:

```bash
bash scripts/run.sh context_adaptive_dflash --phase calibrate \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --max-new-tokens 2048 --calibration-checkpoints 0,512,1024,1536 \
  --fixed-budget full --fixed-gamma 15 --gamma-reference 15 \
  --repetitions 3 --warmup-runs 3
```

Output ở `calibration/calibration.json`, `action_observations.jsonl`, `run_manifest.json`. Selector/layer/grid/protection/entropy/output/runtime phải giống calibration signature. Calibration tự dùng diagnostic profile; wall rollout vẫn giữ production kernel/collector. Cost cần đúng context bucket và refresh mode; thiếu support phải fallback. Calibration hiện không resume, artifact tồn tại thì dùng root mới và lưu lần thử mới.

## 4. Dev matrix, report và lock

### 4.1. Fixed-pair search

Chọn actions bằng dev theo [matrix](experiment_matrix.md), không mặc định 4096/7/15 là tốt nhất. Mỗi setting có root riêng; run ID/calibration/config `CAD_OUTPUT_ROOT` chung trong campaign. AR/full/fixed phải cùng fixed-action provenance trong mỗi root:

```bash
for cad_budget in 1024 2048 4096 full; do
  for cad_gamma in 3 7 11 15; do
    cad_cell="$CAD_RUN_ROOT/sweeps/b${cad_budget}_g${cad_gamma}"
    for cad_variant in ar dflash_full_fixed best_fixed_pair; do
      bash scripts/run.sh context_adaptive_dflash --phase dev --variant "$cad_variant" \
        --data-dir "$CAD_DATA_DIR" --output-root "$cad_cell" \
        --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
        --calibration-file "$CAD_RUN_ROOT/calibration/calibration.json" \
        --fixed-budget "$cad_budget" --fixed-gamma "$cad_gamma" --gamma-reference "$cad_gamma" \
        --max-new-tokens 2048 --repetitions 3 --warmup-runs 3
    done
    bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$cad_cell" \
      --report-phase dev --report-statistics-mode online --bootstrap-samples 10000
  done
done
```

Lệnh sweep trên có thứ tự tuần tự; đo chính thức lưu/shuffle cell blocks theo matrix. Full cùng gamma ở bốn roots là replication; không chọn timing tốt nhất trong bốn lần. Selector study và dev prefix intervention theo matrix/validation; đổi selector cần calibration tương ứng cho adaptive rollout.

### 4.2. Final matrix

Lưu `evidence/dev_selection.json`, đặt ba biến sau theo tuning table. Nếu selector cuối đổi, cập nhật `CAD_SELECTOR` và calibration trước final matrix. Chạy toàn dev split, cùng ba repetitions/warmups, actions và output scope cho mọi variant. `dflash_full_fixed` dùng gamma reference; fixed pair/A-only dùng fixed gamma. Ngay cả AR cũng truyền cùng actions/calibration để pair provenance.

```bash
: "${CAD_BEST_BUDGET:?Chua chon fixed budget tu dev}"
: "${CAD_BEST_GAMMA:?Chua chon fixed gamma tu dev}"
: "${CAD_BEST_FULL_GAMMA:?Chua chon full gamma tu dev}"
CAD_ACTION_ARGS=(
  --fixed-budget "$CAD_BEST_BUDGET" --fixed-gamma "$CAD_BEST_GAMMA"
  --gamma-reference "$CAD_BEST_FULL_GAMMA"
)

python3 - "$CAD_RUN_ROOT/evidence/schedule.json" <<'PY'
import json
import random
import sys
from pathlib import Path
variants = ["ar", "dflash_full_fixed", "best_fixed_pair", "a_only",
            "b_only_history", "b_only_entropy", "independent_ab", "joint",
            "joint_no_entropy", "joint_no_source_relevance"]
random.Random(42).shuffle(variants)
with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
    json.dump({"seed": 42, "scheduling": "variant_blocks", "variants": variants}, stream, indent=2)
PY
mapfile -t CAD_DEV_VARIANTS < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1]))["variants"], sep="\n")' "$CAD_RUN_ROOT/evidence/schedule.json")

for cad_mode in online frozen; do
  for variant in "${CAD_DEV_VARIANTS[@]}"; do
    bash scripts/run.sh context_adaptive_dflash --phase dev --variant "$variant" \
      --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
      --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
      --calibration-file "$CAD_RUN_ROOT/calibration/calibration.json" \
      "${CAD_ACTION_ARGS[@]}" --statistics-update-mode "$cad_mode" \
      --max-new-tokens 2048 --repetitions 3 --warmup-runs 3
  done
  bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$CAD_RUN_ROOT" \
    --report-phase dev --report-statistics-mode "$cad_mode" --bootstrap-samples 10000
done
```

`report` ghi `dev/<mode>/comparison.json` và `report.md`. Nó pair dataset/sample/repetition/output scope; kiểm provenance, errors/coverage/token hashes và bootstrap source clusters. Không gộp online/frozen. Đối chiếu planned cells và full dev IDs bên ngoài reporter: valid comparison trên pilot hoặc thiếu cả variant chưa đủ full matrix. G3–G5 và signal comparisons bổ sung phải review riêng; `headline_valid=true` chưa chứng nhận ngưỡng gain.

CLI chạy repetitions trong từng variant block, chưa shuffle từng repetition. Warmup hiện dùng selected records, cap 16 ngoài timing; chưa phủ mọi shape/bucket. Ghi scheduling và shape-warmup diagnostic vào hồ sơ. Lưu first mismatch/output IDs, không dùng ROUGE thay exactness.

### 4.3. Review và lock

Exposure/G0/G1/G2 phải hoàn tất; ghi verdict G3–G5 và phạm vi claim trước heldout. Lock yêu cầu selected dev config/signatures/coverage/exactness, chưa tự enforce mọi gate. Tạo locks cho cả hai modes:

```bash
for cad_mode in online frozen; do
  cad_lock="$CAD_RUN_ROOT/locked_config.json"
  if [[ "$cad_mode" == frozen ]]; then
    cad_lock="$CAD_RUN_ROOT/locked_config_frozen.json"
  fi
  bash scripts/run.sh context_adaptive_dflash --phase lock --variant joint \
    --output-root "$CAD_RUN_ROOT" --statistics-update-mode "$cad_mode" \
    --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
    --calibration-file "$CAD_RUN_ROOT/calibration/calibration.json" \
    --dev-report "$CAD_RUN_ROOT/dev/$cad_mode/comparison.json" \
    --locked-config-out "$cad_lock" "${CAD_ACTION_ARGS[@]}" \
    --max-new-tokens 2048 --repetitions 3 --warmup-runs 3
done
```

Lock ghi `run_id`, implementation hash và danh sách `test_variants` có paired comparison exact, đủ coverage trên dev. Test cũng xác thực target/tokenizer/runtime signature từ calibration, kể cả với AR. Lock là candidate config cho heldout, không tuyên bố G0–G5 đã pass.

Kiểm cả hai locks có đầy đủ planned variants; thiếu cả một dev cell thì lock vẫn có thể tạo danh sách nhỏ hơn. Trạng thái là `candidate_locked_pending_heldout`; không sửa sau xem test.

## 5. Heldout test và report

Test không nhận `--max-samples`; runner bắt buộc dùng toàn bộ test split. Chỉ `--variant`, device/runtime output path và resume được thay theo lock contract. Chạy cùng các test variants, cùng `CAD_RUN_ID` và cùng master config:

```bash
python3 - "$CAD_RUN_ROOT" "${CAD_DEV_VARIANTS[@]}" <<'PY'
import json
import sys
from pathlib import Path
root, expected = Path(sys.argv[1]), set(sys.argv[2:])
assert len(expected) == 10, "Planned matrix must contain all ten variants"
for name in ("locked_config.json", "locked_config_frozen.json"):
    locked = json.loads((root / name).read_text(encoding="utf-8"))
    assert set(locked["test_variants"]) == expected, f"Incomplete locked matrix: {name}"
print("Both locked matrices cover every planned variant")
PY

for cad_mode in online frozen; do
  cad_lock="$CAD_RUN_ROOT/locked_config.json"
  if [[ "$cad_mode" == frozen ]]; then
    cad_lock="$CAD_RUN_ROOT/locked_config_frozen.json"
  fi
  mapfile -t CAD_TEST_VARIANTS < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1], encoding="utf-8"))["test_variants"], sep="\n")' "$cad_lock")
  for variant in "${CAD_DEV_VARIANTS[@]}"; do
    if [[ ! " ${CAD_TEST_VARIANTS[*]} " == *" $variant "* ]]; then
      printf 'Planned variant missing from lock: %s\n' "$variant" >&2
      exit 1
    fi
    bash scripts/run.sh context_adaptive_dflash --phase test --variant "$variant" \
      --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
      --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
      --locked-config "$cad_lock" --statistics-update-mode "$cad_mode"
  done
  bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$CAD_RUN_ROOT" \
    --report-phase test --report-statistics-mode "$cad_mode" --bootstrap-samples 10000
done
```

Calibration path/hash nạp từ lock. `test` từ chối đổi signatures/scope/seed/repetitions/warmups/policy/actions; giữ master/runtime đã đăng ký. Report ở `test/<mode>/`. Review G6 đối chiếu toàn planned/locked matrix và full test IDs/groups×repetitions; reporter chỉ thấy cells hiện diện. Điền [mẫu báo cáo](results_template.md). Không đổi lock sau xem test; đổi method cần protocol/cohort mới.

## 6. Artifact layout và giới hạn V1

```text
<run_root>/
  inputs/{primary,smoke_dev}/
  evidence/{preregistration.md,exposure_audit.json,schedule.json,dev_selection.json,...}
  split_manifest.json
  prepare_manifest.json
  preflight.json
  calibration/{calibration.json,action_observations.jsonl,run_manifest.json}
  locked_config.json
  locked_config_frozen.json
  smoke_check/smoke/<variant>/online/...
  pilot/calibration/...
  sweeps/b<B>_g<gamma>/dev/...
  {smoke,dev,test}/<variant>/<online|frozen>/
    manifest.json
    rep_N/{requests.jsonl,rounds.jsonl,output_token_ids.jsonl}
  {dev,test}/<online|frozen>/{comparison.json,report.md}
```

Mỗi JSONL có summary cuối; `--resume` giữ các sample success tương thích và retry lỗi. Mỗi request ghi prompt/output hashes, runtime signatures, counters, timing và ROUGE khi có reference. Manifest hashing weights đầy đủ trước khi timing nên có thể làm startup lâu; thời gian này ngoài E2E.

`evidence/*` do người vận hành/harness lưu, chưa tự sinh bởi executor. Error/partial phase trả exit khác 0 sau finalize; giữ logs/exit codes và `pipefail` khi dùng tee. Resume cùng config/signature theo dataset/sample/repetition, không dùng để đổi cap/policy/seed/calibration. Sau resume tạo report lại; calibration chưa hỗ trợ resume.

```bash
bash scripts/run.sh context_adaptive_dflash --phase test --variant joint \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --locked-config "$CAD_RUN_ROOT/locked_config.json" --resume
```

V1 chỉ hỗ trợ `length_mode=draft_shape`, `cost_update_mode=frozen_cost`, batch 1, greedy-calibrated action priors và request-local online/frozen prefix statistics. `verify_prefix` và `async_event` chưa triển khai nên bị parser từ chối. Sampling có thể chạy theo verifier DFlash nhưng calibration action priors cần greedy. Sliding-window draft attention dùng mask theo absolute logical positions. Không có G0–G6 result cho đến khi các lệnh GPU trên server hoàn tất và được review.

Secondary fixed-output dùng từng giá trị `--fixed-output-tokens 256`, `512`, `1024`, `2048`, root/calibration/lock riêng; EOS disabled đồng nhất và quality scope `not_natural_summary`. Không reuse calibration natural hoặc gộp headline. Code datasets, model khác, 16K/32K là campaign riêng; thiếu assets/data thì unavailable. GPU parity, dev intervention replay, entropy-threshold variant, sampling audit và scheduler shuffle từng repetition chưa có phase CLI; xem matrix/validation. Việc hoàn thiện tài liệu chưa tạo measured results.
