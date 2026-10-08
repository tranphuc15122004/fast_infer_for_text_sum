# Runbook và cấu hình triển khai

Ngày: **2026-10-08**. [Index](../../README.md), [integration](integration.md), [protocol](experiment_protocol.md).

## 1. Trạng thái các lệnh

`scripts/infer_dflash.py` và `bash scripts/run.sh dflash` đã có trong repo. `scripts/infer_context_adaptive_dflash.py`, runner tương ứng và case dispatcher mới **chưa tồn tại**; các lệnh CAD bên dưới là CLI contract cần hoàn thành theo plan. Không chạy chúng như một implementation đã sẵn sàng.

## 2. Runtime và asset preflight hiện có

Trên server:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
python3 scripts/setup_server_env.py --check
python3 scripts/check_shared_env.py
```

Master canonical ở `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env`, pointer `config/master.path`. Dùng `FI_PYTHON=python3`, `FI_DEVICE=cuda`, `FI_OFFLINE=1`; giữ `MODEL_TARGET` và `MODEL_DFLASH_DRAFT` theo snapshot operator đã có. Không suy đường dẫn model từ máy local; ví dụ model root trong các run cũ khác nhau. Không tải Hub, cài online hoặc thêm dependency cho controller.

Trên local CPU, chỉ chạy các checks pure Python sau khi implementation tương ứng có; preflight GPU/timing thực chạy trên server. Không cố sửa driver T4/cu130 hoặc cài profile server vào venv local.

Baseline smoke **hiện chạy được khi master/model/GPU server hợp lệ**:

```bash
bash scripts/run.sh dflash \
  --data-file data/longbench_100_14k/gov_report.jsonl \
  --max-samples 1 --block-size 16 --smoke \
  --output outputs/context_adaptive_dflash/baseline_smoke/requests.jsonl
```

Baseline writer append; dùng output path mới cho mỗi lần smoke, không chạy lại trên file đã có rồi coi duplicated rows là samples mới. Không có lệnh GPU nào trong runbook được thực thi khi viết tài liệu.

## 3. Namespace master-env dự kiến

Operator thêm block này vào master **ngoài repository** sau khi launcher được triển khai. Không tạo file runtime mới trong `config/`.

```bash
CAD_PHASE=smoke
CAD_VARIANT=joint
CAD_SELECTOR=target_parent
CAD_BUDGETS=1024,2048,4096,full
CAD_GAMMAS=3,7,11,15
CAD_FIXED_GAMMA=15
CAD_FIXED_BUDGET=4096
CAD_LENGTH_MODE=draft_shape
CAD_SOURCE_CHUNK_SIZE=128
CAD_SOURCE_ANCHORS=64
CAD_RECENT_OUTPUT=256
CAD_TARGET_LAYER=middle
CAD_REFRESH_PERIOD=4
CAD_SIGNAL_UPDATE_PERIOD=1
CAD_CONTROLLER_MARGIN=0.02
CAD_ENTROPY_SIGNAL_TEMPERATURE=1.0
CAD_PRIOR_STRENGTH=16
CAD_MIN_STATE_SUPPORT=8
CAD_STATISTICS_DECAY=0.95
CAD_LATENCY_EMA_ALPHA=0.1
CAD_COST_UPDATE_MODE=frozen_cost
CAD_TIMING_MODE=wall
CAD_OUTPUT_ROOT=outputs/context_adaptive_dflash
```

`MODEL_TARGET`, `MODEL_DFLASH_DRAFT`, `DATA_INPUT`, `RUN_SAMPLES`, `RUN_MAX_NEW_TOKENS`, `RUN_TEMPERATURE` và `LONG_BENCH_SEED` vẫn lấy từ master chung. `CAD_DATA_DIR` nếu đặt dùng canonical dataset directory, không thay namespace data chung. `CAD_PHASE=smoke` áp cap 32 và 2 source groups/dataset; primary cap 2048 được khóa trong test config, không lấy cap smoke cho full run.

Thứ tự precedence: CLI > caller environment > master values > defaults trong `AdaptiveConfig`. Launcher phải bảo toàn caller overrides qua `fast_infer_load_master`; helper hiện tại không tự bảo đảm mọi `CAD_*` override nên wrapper phải snapshot/restore các keys liên quan khi source master. `--phase`/`--variant` là nguồn quyết định cuối. `SMOKE=1` hoặc `FULL=1` chỉ đặt default phase khi CAD phase chưa được chọn; nếu cả hai bật phải báo configuration error.

Validation: budgets số nguyên dương hoặc `full`, gammas trong checkpoint/parity support, `gamma>=1` cho draft actions; zero chỉ AR. Margin `[0,1)`, EMA alpha `(0,1]`, decay `(0,1)`, entropy signal temperature dương, protection/chunk integers hợp lệ. Không silently clip checkpoint-incompatible gamma.

## 4. CLI phải xây

Entry point `scripts/infer_context_adaptive_dflash.py` với các options:

| Option | Hợp đồng |
|---|---|
| `--phase` | `prepare|preflight|smoke|calibrate|dev|test|report`; default smoke |
| `--variant` | Các ID trong protocol, thêm `joint_no_entropy`, `joint_no_source_relevance` |
| `--target-model`, `--draft-model` | Snapshot local bắt buộc cho model phases |
| `--data-dir`, `--data-file` | Canonical multi-dataset hoặc single file; không truyền cả hai |
| `--output-root`, `--run-id` | Artifact root và ID duy nhất |
| `--split-manifest` | Bắt buộc calibration/dev/test; prepare tạo manifest |
| `--exposure-manifest` | Có thể lặp để import các cohort trước |
| `--calibration-file` | Bắt buộc dev/test joint; schema/signature phải match |
| `--locked-config` | Bắt buộc test/report, không cho runtime đổi hyperparameters |
| `--budgets`, `--gammas`, `--fixed-budget`, `--fixed-gamma` | Actions theo design |
| `--selector`, `--target-layer`, `--refresh-period` | Selection/collector configuration |
| `--max-new-tokens`, `--temperature`, `--seed` | Sampling/stopping signature |
| `--fixed-output-tokens` | Performance stress scope riêng, disable EOS đồng nhất |
| `--timing-mode`, `--cost-update-mode` | Diagnostic/wall và frozen/async cost |
| `--smoke`, `--full`, `--resume` | Smoke alias; full alias test cần locked config; resume kiểm tra compatibility |

`prepare`/`report` là model-free, chạy CPU; parser phải dispatch trước model load/CUDA requirements. Core benchmark dùng evaluator chung cho live generation và artifact replay/report. Prefix replay/calibration là offline experimental mode, không phải online extra target forwards.

## 5. Các lệnh sau khi implementation pass

**Các lệnh trong mục này chưa chạy được trước khi task launcher/integration hoàn tất.** `CAD_RUN_ID` dưới đây phải đổi khi chạy experiment mới; không dùng ID cũ nếu không resume.

```bash
export CAD_RUN_ID=cadflash_v1_dev_001
export CAD_DATA_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k
export CAD_RUN_ROOT=outputs/context_adaptive_dflash/$CAD_RUN_ID

bash scripts/run.sh context_adaptive_dflash --phase prepare \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --exposure-manifest outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/manifest.json

bash scripts/run.sh context_adaptive_dflash --phase preflight \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json"

bash scripts/run.sh context_adaptive_dflash --phase smoke \
  --variant dflash_full_fixed --data-dir "$CAD_DATA_DIR" \
  --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json"

bash scripts/run.sh context_adaptive_dflash --phase calibrate \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json"

bash scripts/run.sh context_adaptive_dflash --phase dev --variant joint \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --calibration-file "$CAD_RUN_ROOT/calibration.json"
```

Dev phải chạy toàn bộ variants của protocol, không chỉ joint trong ví dụ. Chạy report dev để chọn config và tạo `locked_config.json` có selector/actions/splits/signatures/gate decisions. Test không tự khóa hyperparameters từ test result.

```bash
bash scripts/run.sh context_adaptive_dflash --phase test --variant joint \
  --data-dir "$CAD_DATA_DIR" --output-root "$CAD_RUN_ROOT" \
  --split-manifest "$CAD_RUN_ROOT/split_manifest.json" \
  --calibration-file "$CAD_RUN_ROOT/calibration.json" \
  --locked-config "$CAD_RUN_ROOT/locked_config.json"

bash scripts/run.sh context_adaptive_dflash --phase report \
  --output-root "$CAD_RUN_ROOT" \
  --locked-config "$CAD_RUN_ROOT/locked_config.json"
```

Artifact cells ở `<run_root>/<phase>/<variant>/rep_<index>/`; mỗi cell có manifest/request/round stream theo measurement contract. Run root chứa split/calibration/locked config và aggregate reports. `calibrate` ghi riêng signatures cho các selector/modes đã profile; `dev`/`test` từ chối missing prior, không âm thầm lấy số đo của selector khác.

## 6. Lock, resume và kết luận

Locked config chứa model/runtime/data/split hashes, action set và shape capabilities, selector/refresh/protection, entropy cutpoints/priors, cost support buckets, seeds/repetitions, output scopes, baseline fixed pair và G0–G5 reports. Người triển khai tạo nó từ completed dev artifacts; file cấu hình ví dụ trong tài liệu không thay artifact đã đo.

Resume kiểm tra từng cell trước model load, giữ request IDs và repetition order. Summary chỉ finalize khi cell đã đủ expected rows. Error/mismatch làm cell partial/error; không bỏ request để report trông đẹp hơn.

Hoàn tất thực nghiệm nghĩa là G6 có completed heldout artifacts và report với uncertainty/correctness, không chỉ launcher chạy hoặc GPU smoke pass.
