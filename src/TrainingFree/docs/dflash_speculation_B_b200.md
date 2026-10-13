# Thực nghiệm B — DFlash uncertainty và verification depth trên B200

Ngày cập nhật: **2026-10-13**.

Harness đã có CLI cho B0, B1, B2 và B1-V. Test logic/schema chạy trên CPU; chưa
chạy parity, cache-prefix validation hay timing GPU. B1/B2/B1-V chỉ được xem là
bằng chứng hợp lệ sau khi B0 exact greedy parity qua trên đúng checkpoint và
backend của server.

## Bố cục

| Mode | Làm gì | Kết quả |
|---|---|---|
| `prepare` | Đọc corpus, exposure manifests và tạo source-group cohort/split | `campaign.json`, `cohort_manifest.json`, `split_manifest.json` |
| `preflight` | Nạp target/DFlash từ local snapshot, kiểm runtime/checkpoint/backend | `runtime.json` |
| `validate` | So target-only AR, native DFlash và adapter trên gamma 3/7/11/15; gamma 15 tạo reference rollout | `evidence/parity.json`, `traces/*.json` |
| `b1` | Replay từng gamma bằng draft forward riêng trên cùng prefix/cache/bank; gamma 15 thu signal ở lượt riêng | Bảng B1, vector DFlash/parent signals |
| `b1v` | Draft gamma 15 rồi verify prefix k=3/7/11/15 trên cache fork riêng; kiểm tra L và pending anchor | Bảng B1-V |
| `analyze` | Tính oracle/fixed gaps, bootstrap theo source group, grouped OOF survival/Brier và quyết định screening | `reports/*`, không nạp model |
| `run` | Chạy prepare, B0, B1, B1-V rồi analyze tuần tự | Toàn bộ artifacts trên |

Không có adaptive controller B3/B4 trong harness này. B3 chỉ nên bắt đầu khi
B1-V cho thấy headroom, B2 có predictive value ngoài regime/history controls,
và decision-regret đánh giá heldout đã được bổ sung/khóa. Vì vậy kết quả của
harness hiện tại là opportunity/predictability, chưa phải E2E speedup.

## Chạy trên server

Đảm bảo master config trỏ tới Python 3.12, local target/DFlash snapshots, FA4
và corpus. Mặc định corpus là
`/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k`;
có thể truyền `--data-dir` hoặc `--data-file`. Exposure manifest phải được rà soát
chung với campaign A/B. Manifest đã có trong repo để bắt đầu là
`data/dflash_context_A_audit/combined_exposure_manifest.json`; nếu có audit
mới hơn, dùng manifest mới đó.

Lệnh pilot một lượt:

```bash
bash scripts/run_dflash_speculation_B_b200.sh \
  --config /path/to/master.env \
  --gpu-ids 0 \
  --mode run \
  --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --max-new-tokens 1024 \
  --validation-tokens 128 \
  --max-prefixes 6 \
  --warmups 3 \
  --repetitions 6 \
  --selection-repetitions 3
```

`pilot` mặc định chọn 2 source groups/dataset. Lệnh in `run_id` và thư mục
artifact ngay khi tạo cohort. Để chạy từng phase, dùng cùng `--run-id`, corpus,
exposure manifests, `--cohort-phase` và protocol options:

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode prepare --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode preflight --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode validate --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b1 --run-id B_pilot_01 --cohort-phase pilot --resume \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b1v --run-id B_pilot_01 --cohort-phase pilot --resume \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --mode analyze --run-id B_pilot_01
```

`--resume` chỉ tái dùng cell khớp campaign/code/runtime signature. Thay model,
backend, cohort, gamma grid, prompt hoặc protocol cần run mới và chạy lại B0.
Lưới gamma hiện khóa ở 3/7/11/15; CLI từ chối lưới rút gọn để tránh ngầm đổi
protocol.

## Đọc artifacts và kết luận

- `evidence/parity.json`: parity theo document/gamma. Bất kỳ mismatch nào làm
  `invalid_correctness`; B1/B1-V bị chặn.
- `b1/draft_shape.jsonl`: cùng `state_id` có bốn gamma, proposals riêng từng
  shape, accepted candidates và round/component timing. `round_ms` gồm draft,
  full-context gather, verification, cache crop và draft-bank append; cache
  fork/rebuild nằm ngoài span và được ghi riêng.
- `b2/signals.jsonl`: entropy thô/normalized, confidence, probability/logit
  margin ở từng candidate; parent-target signals; history chỉ từ reference
  gamma 15 trước round. Candidate sau rejection chỉ tham gia survival label,
  không được tính là rejection độc lập ở vị trí không reach.
- `b1v/verify_prefix.jsonl`: cùng proposals gamma 15 cho cả k, verify riêng
  prefix và cache fork. Mỗi cell phải thỏa `L_k=min(L_15,k)`, cache crop length
  và pending anchor phù hợp reference; mismatch dừng phase.
- `reports/b1_metrics.json`: oracle gamma chọn bằng 3 selection repetitions,
  gap/regret đo trên 3 evaluation repetitions, near-optimal fixed coverage,
  tail acceptance, proposal-prefix equality và CI theo source group.
- `reports/b2_metrics.json`: Brier/survival out-of-fold theo document group,
  fallback/support counts, candidate risk set và tỷ lệ confident-but-wrong.
  So entropy với regime R, rồi so `R-A-DH` với `R-A` để kiểm tra entropy có
  thêm thông tin sau history hay không.
- `reports/b1v_metrics.json`: oracle k-gap dùng cùng cách tách selection/evaluation
  như B1. Đây là local cost opportunity, chưa tính chi phí chọn k ở controller.
- `reports/decision.md`: screening theo ngưỡng đã khai báo trong campaign;
  CI rộng hoặc cohort thiếu sẽ cho `inconclusive`, không tự biến thành kết quả âm.

Ngưỡng B1 đề xuất: gap J ít nhất 5% và CI lower trên 2% để mở nhánh
opportunity; CI upper không quá 2% là ít headroom trong grid/workload hiện tại.
B2: entropy cần Brier skill tối thiểu 5% so R với CI improvement dương; để nói
entropy bổ sung sau history cần tối thiểu 2% so `R-A` và CI dương. Bootstrap
đơn vị source group/document, không phải round hay timing repetition.

Pilot 6 documents chỉ dùng để debug runtime/ước lượng chi phí; grouped folds
và CI từ pilot rất thô. Không dùng pilot làm paper claim. Khi B0 parity fail,
điều tra backend/cache/accounting trước khi đọc acceptance như một thuộc tính
của DFlash. B3 E2E, output exactness của controller, ROUGE và B4 lagged
draft-length policy chưa được triển khai trong harness này.
