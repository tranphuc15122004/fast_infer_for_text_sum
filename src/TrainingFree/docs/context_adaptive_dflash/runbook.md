# Runbook Context-Adaptive DFlash

Ngày: **2026-10-08**. Trạng thái: **executor/launcher đã có trong repository; chưa có kết quả GPU hoặc parity report**. Xem [integration](integration.md), [protocol](experiment_protocol.md), [measurement contract](measurement_contract.md).

## 1. Runtime và asset

Trên server dùng Python 3.12 từ PATH, model snapshots local và dependency đã mirror. Không cài online hoặc tạo venv riêng cho phương pháp.

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
python3 scripts/setup_server_env.py --check
python3 scripts/check_shared_env.py
```

Master config ở `config/master.path` trỏ tới shell-env ngoài repository. Cấu hình `MODEL_TARGET`, `MODEL_DFLASH_DRAFT`, `DATA_INPUT` (hoặc `CAD_DATA_DIR`), `RUN_MAX_NEW_TOKENS`, `RUN_MAX_INPUT_TOKENS`, `RUN_TEMPERATURE` và `LONG_BENCH_SEED` như DFlash baseline. Các `CAD_*` caller overrides được runner giữ lại khi load master. Token HF chỉ dùng nếu snapshot local cần xác thực; runner bật offline mode khi load model.

`prepare`, `preflight`, `lock`, `report` không cần model forward. `preflight --model-check` đọc config snapshot local. `smoke`, `calibrate`, `dev`, `test` yêu cầu CUDA và load model; máy local T4 chỉ chạy CPU, không dùng để benchmark.

## 2. Chuẩn bị split và preflight

Đặt `CAD_RUN_ID` duy nhất cho một protocol run. Exposure manifest đưa các document đã dùng để phát triển ý tưởng vào dev trước khi chia split.

```bash
export CAD_RUN_ID=cadflash_v1_001
export CAD_DATA_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k
export CAD_RUN_ROOT=outputs/context_adaptive_dflash/$CAD_RUN_ID

bash scripts/run.sh context_adaptive_dflash --phase prepare \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --exposure-manifest outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/manifest.json

bash scripts/run.sh context_adaptive_dflash --phase preflight \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" --model-check
```

`prepare` ghi `split_manifest.json` và checksum dataset. Nó trả exit code khác 0 khi mỗi dataset/source stratum không đủ nhóm cho calibration/dev/test; không được tiếp tục bằng split tổng hợp. `preflight` kiểm tra vocab, hidden size, target feature layers, mask token, DFlash block size và sliding-window config. Nếu model chưa được mirror thì bỏ `--model-check` để chỉ kiểm dữ liệu.

## 3. GPU smoke và calibration

Smoke tối đa 32 token, lấy tối đa hai source groups mỗi dataset. Đây là kiểm tra đường chạy, không phải bằng chứng correctness hoặc speedup.

```bash
bash scripts/run.sh context_adaptive_dflash --phase smoke --variant dflash_full_fixed \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json"
```

V1 calibration hiện chỉ dùng greedy target (`--temperature 0`) và full fixed trajectory. Mỗi checkpoint chạy lại từng feasible `(budget,gamma)` trên bản clone target cache của cùng prefix; clone setup được ghi riêng, còn controller cost được profile sau khi action table đã fit.

```bash
bash scripts/run.sh context_adaptive_dflash --phase calibrate \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --max-samples 16 \
  --max-new-tokens 2048 --calibration-checkpoints 0,512,1024,1536 \
  --repetitions 3 --warmup-runs 3
```

Lệnh trên đăng ký trước cohort tối đa 16 records, ưu tiên một đại diện cho mỗi source group và cân bằng theo dataset; `run_manifest.json` khóa IDs/hash và cỡ cohort. Output ở `calibration/calibration.json`, `action_observations.jsonl` và `run_manifest.json`. Mọi selector, layer, action grid, source chunk/protection và entropy temperature dùng ở dev/test phải giống signature calibration.

## 4. Dev matrix, report và lock

Chạy AR và tất cả controls/candidate trên đúng split/config/output scope. Có thể chạy từng variant bằng lệnh sau; mặc định dev có 3 repetitions và 3 warmups. Dùng cùng `--run-id`, action settings, split và calibration cho mọi variant. `dflash_full_fixed` dùng `gamma_reference`; `best_fixed_pair` và `a_only` dùng `fixed_gamma` cùng `fixed_budget`.

```bash
for variant in ar dflash_full_fixed best_fixed_pair a_only b_only_history b_only_entropy independent_ab joint joint_no_entropy joint_no_source_relevance; do
  bash scripts/run.sh context_adaptive_dflash --phase dev --variant "$variant" \
    --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
    --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
    --calibration-file "$CAD_RUN_ROOT/calibration/calibration.json" \
    --fixed-budget 4096 --fixed-gamma 7 --gamma-reference 15
done

bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$CAD_RUN_ROOT" \
  --report-phase dev --report-statistics-mode online --bootstrap-samples 10000
```

`report` ghi `dev/online/comparison.json` và `report.md`. Nó pair theo dataset/sample/repetition/output scope; kiểm prompt/model/runtime/source group/implementation/fixed-action provenance, lỗi, độ phủ và token hash greedy; bootstrap theo source-document cluster. `headline_valid=false` khi thiếu request, mismatch, có lỗi hoặc trộn provenance. Tạo report riêng cho `--report-statistics-mode frozen`; không gộp frozen và online.

Chọn fixed gamma/budget và candidate chỉ từ dev. Khi sweep nhiều fixed pairs, dùng output root riêng cho mỗi setting (kèm AR trong cùng root), sau đó chọn cặp tốt nhất theo report; không ghi nhiều setting vào cùng cell. Sau khi chọn, chạy lại đầy đủ dev matrix với fixed settings cuối và tạo report mới. Lock yêu cầu report dev có paired exact comparison đầy đủ cho candidate, calibration khớp split và cùng statistics mode:

```bash
bash scripts/run.sh context_adaptive_dflash --phase lock --variant joint \
  --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --calibration-file "$CAD_RUN_ROOT/calibration/calibration.json" \
  --dev-report "$CAD_RUN_ROOT/dev/online/comparison.json" \
  --locked-config-out "$CAD_RUN_ROOT/locked_config.json" \
  --fixed-budget 4096 --fixed-gamma 7 --gamma-reference 15
```

Lock ghi `run_id`, implementation hash và danh sách `test_variants` có paired comparison exact, đủ coverage trên dev. Test cũng xác thực target/tokenizer/runtime signature từ calibration, kể cả với AR. Lock là candidate config cho heldout, không tuyên bố G0–G5 đã pass.

## 5. Heldout test và report

Test không nhận `--max-samples`; runner bắt buộc dùng toàn bộ test split. Chỉ `--variant`, device/runtime output path và resume được thay theo lock contract. Chạy cùng các test variants, cùng `CAD_RUN_ID` và cùng master config:

```bash
mapfile -t CAD_TEST_VARIANTS < <(python3 -c 'import json,sys; print(*json.load(open(sys.argv[1], encoding="utf-8"))["test_variants"], sep="\n")' "$CAD_RUN_ROOT/locked_config.json")
for variant in "${CAD_TEST_VARIANTS[@]}"; do
  bash scripts/run.sh context_adaptive_dflash --phase test --variant "$variant" \
    --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
    --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
    --locked-config "$CAD_RUN_ROOT/locked_config.json"
done

bash scripts/run.sh context_adaptive_dflash --phase report --output-root "$CAD_RUN_ROOT" \
  --report-phase test --report-statistics-mode online --bootstrap-samples 10000
```

Calibration path/hash được nạp từ lock. `test` từ chối đổi model/data split/output cap/temperature/input cap/seed/repetitions/warmups, config hoặc action settings. Report heldout nằm ở `test/online/`. Không có permission để đổi lock sau khi xem test results; thay đổi method cần run ID/split mới.

## 6. Artifact layout và giới hạn V1

```text
<run_root>/
  split_manifest.json
  prepare_manifest.json
  preflight.json
  calibration/{calibration.json,action_observations.jsonl,run_manifest.json}
  locked_config.json
  {smoke,dev,test}/<variant>/<online|frozen>/
    manifest.json
    rep_N/{requests.jsonl,rounds.jsonl,output_token_ids.jsonl}
  {dev,test}/<online|frozen>/{comparison.json,report.md}
```

Mỗi JSONL có summary cuối; `--resume` giữ các sample success tương thích và retry lỗi. Mỗi request ghi prompt/output hashes, runtime signatures, counters, timing và ROUGE khi có reference. Manifest hashing weights đầy đủ trước khi timing nên có thể làm startup lâu; thời gian này ngoài E2E.

V1 chỉ hỗ trợ `length_mode=draft_shape`, `cost_update_mode=frozen_cost`, batch 1, greedy-calibrated action priors và request-local online/frozen prefix statistics. `verify_prefix` và `async_event` chưa triển khai nên bị parser từ chối. Sampling có thể chạy theo verifier DFlash nhưng calibration action priors cần greedy. Sliding-window draft attention dùng mask theo absolute logical positions. Không có G0–G6 result cho đến khi các lệnh GPU trên server hoàn tất và được review.
