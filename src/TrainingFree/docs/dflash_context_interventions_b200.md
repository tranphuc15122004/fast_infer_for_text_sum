# Chạy campaign A1–A3 trên B200

Launcher của kế hoạch [A 13/10/2026](../plans/experiment_plan_A_13_10_26.md)
đã được nối vào runtime dùng chung của repo. Campaign chạy offline trên một
B200, một worker, batch 1, BF16, DFlash b16, gamma 15, block 16 và FA4. Không
cần Modal hoặc package Python mới.

## Điều kiện đầu vào

Trên server phải có:

- Repo ở `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum`
  và master env ở `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env`.
- Python hệ thống 3.12, CUDA/Torch/Transformers và FA4/CUTLASS tương thích với
  image B200; runtime đã cài các package theo `requirements.txt`.
- Target Qwen3-4B, DFlash Qwen3-4B-b16, LongBench ba dataset `gov_report`,
  `multi_news`, `qmsum`; mọi asset phải là snapshot local hoàn chỉnh.
- Split manifest hiện có và một exposure manifest đã rà soát bao gồm probe
  trước đây cùng các pilot đã chạy. Exposure manifest dùng cấu trúc
  `{"samples": [{"sample_id": ..., "dataset": ..., "source_sha256": ...}]}`
  hoặc danh sách `source_fingerprints`/`context_fingerprints` mà helper split
  hiện có hỗ trợ. Không dùng audit một phần làm bằng chứng rằng heldout sạch.

Repo local có [exposure audit của probe](../../../outputs/context_adaptive_dflash/m0_local_primary_cpu_20261009/evidence/exposure_audit.json)
và split manifest đi kèm. Audit đó ghi probe absent-verified so với corpus
primary, nhưng chưa thay thế exposure audit của các pilot trước và còn ghi
đường dẫn local `/home/tuantb`. Hãy chuyển các manifests lên server, kiểm tra
hash/data identity và gộp đủ exposure của probe + pilot; sau đó dùng manifest
đã rebase sang server. Launcher từ chối tạo cohort nếu không có exposure
manifest, và kiểm tra group đã lộ không bị đặt vào test.

Không có Python dependency bổ sung cho harness: bootstrap dùng thư viện chuẩn,
ROUGE và JSONL dùng helper của repo. FA4 kernel phải qua probe ngay trên server;
không tự fallback sang SDPA trong primary results.

## Lệnh trên B200

Điền thư mục đã staging hai manifest audit trước khi bắt đầu. Ví dụ dưới đây
giữ nguyên một `RUN_ID` xuyên suốt pilot/dev/lock/confirm:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

RUN_ID=dflash_context_A_20261013
AUDIT_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/dflash_context_A_audit
SPLIT_MANIFEST="$AUDIT_DIR/split_manifest.json"
EXPOSURE_MANIFEST="$AUDIT_DIR/combined_exposure_manifest.json"
COMMON=(--config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
  --run-id "$RUN_ID" --split-manifest "$SPLIT_MANIFEST" \
  --exposure-manifest "$EXPOSURE_MANIFEST" --gpu-ids 0)

# In kế hoạch, chưa load model hoặc chạy generation.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode preflight --plan-only

# Kiểm asset, Python, B200/SM100 và FA4 shim/kernel. Chưa load target/draft.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode preflight

# Chuẩn bị source groups/split/cohort manifests; không load model.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode prepare

# Parity gate nhỏ trước ma trận.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode validate

# Pilot 2 groups mỗi dataset, tối đa 6 prefix/group, cap 1024 token.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode pilot

# Dev gồm 24 groups, cap 2048; mặc định chọn previous_dflash, B=2048.
# Có thể chọn selector/B/period khác trước khi chạy dev.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode dev \
  --a3-selector previous_dflash --a3-budget 2048 --refresh-period 4

# Lock lựa chọn sau khi đọc reports/dev_decision.md.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode lock \
  --a3-selector previous_dflash --a3-budget 2048 --refresh-period 4

# Chỉ chạy heldout sau khi lock; confirm yêu cầu đúng run-id, source code,
# runtime, model/tokenizer và split signatures của dev.
bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode confirm \
  --locked-config "outputs/dflash_context_interventions/$RUN_ID/locked_config.json"

bash scripts/run_dflash_context_experiments.sh "${COMMON[@]}" --mode report \
  --report-phase confirm
```

Nếu job bị ngắt, chạy lại đúng lệnh pha đó và thêm `--resume`. Không sửa code,
model, split, seed, selector, budget hoặc backend giữa resume; signature không
khớp sẽ dừng. Sau khi code hoặc input thay đổi, dùng `RUN_ID` mới. `validate`
chỉ qua khi native Full DFlash và adapter Full có output token IDs bằng greedy
AR trên toàn bộ cohort validate. Parity fail thì pilot/dev/confirm không mở
ma trận hoặc ghi nhận speedup.

## Artifact và giới hạn triển khai hiện tại

Mỗi cell được checkpoint nguyên tử dưới `outputs/dflash_context_interventions/<run_id>/`;
JSONL pha được ghép qua `JsonlWriter` dùng chung. Report giữ cell infeasible,
not-eligible và error thay vì tính chúng thành zero. Confirm dùng test split,
kiểm hash split/asset/runtime/code đã khóa, rồi mới sinh report cuối.

Launcher hiện cho phép một `--a3-selector` và một `--a3-budget` cho mỗi lần
chạy dev. Replay A3 vẫn chạy budget curve của selector đó; rollout latency
đo budget/selector được truyền vào, và refresh periods 2/4/8/16 được đo riêng.
Nếu cần so sánh nhiều policy end-to-end trước lock, chạy các cấu hình dev ở
các `run-id` riêng rồi tổng hợp chọn thủ công trên dev; không gộp records khác
signature. Confirm chỉ được chạy một cấu hình đã lock. Campaign này cố ý chỉ
dùng một GPU; data-parallel nhiều B200 chưa được triển khai.

Đây mới là code và runbook. Chưa chạy trên B200; các lệnh thực nghiệm cần được
chạy trên server sau khi hoàn thành preflight và xác nhận exposure manifests.
