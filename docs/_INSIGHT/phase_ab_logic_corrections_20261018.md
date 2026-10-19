# Nhật ký sửa logic đánh giá riêng Phase A/B — 18/10/2026

## 1. Phạm vi

Công việc này sửa logic phương pháp, thu thập và đánh giá từng phase. Thực nghiệm
pipeline A+B chưa được thực hiện. Việc chưa chạy A+B không phải lỗi của giai đoạn
screening hiện tại. Các claim mới giới hạn theo bằng chứng từng phase.

## 2. Các sửa đổi

### Phase A

- Snapshot v2 lưu layer IDs và [layer,query,key] attention; giữ accepted length,
  sample ID, round và parent query thật. Không tạo parent bằng vector mean.
- Causal positions giữ riêng theo query; scalar API được định nghĩa là vị trí
  cuối của dãy liên tiếp, không gán vị trí cuối cho mọi hàng.
- Không lấy L18 thay L27/band hoặc L0 thay L12. Dữ liệu legacy chỉ được đánh giá
  theo tầng trong manifest thật; missing features/layers là unavailable.
- Parent control dùng query q[L] của cùng round t để chọn cho t+1, cùng thời điểm
  với max/mean-query. Không dùng parent_target từ t-1 để so lệch một vòng.
- Metric cùng round mang nhãn diagnostic. Metric next-round chỉ chọn bằng quá
  khứ; reference conditional trên keys đã tồn tại lúc chọn.
- Macro-average theo document, đếm snapshot riêng; deduplicate alias cùng round.
- Protection có thể làm action infeasible; không cắt mất protection để đạt budget.
  Chunk mode không bù token rời rạc và có ghi budget chưa dùng.
- Các script selector/layer/trade-off dùng chung evaluator có provenance;
  figure đọc report thật, bỏ bảng số và thứ hạng gán sẵn.
- Đổi nhãn baseline thành parent/spatial-pool/ensemble adaptations. Bỏ claim
  tái hiện 1:1 FastKV/SnapKV/RocketKV và oracle DuoAttention.
- Microbenchmark được ghi đúng scope CPU pooling/top-B; không suy diễn GPU
  coalescing, PagedAttention, VRAM hay full-runtime overhead từ proxy.
- Pilot selector ghi rõ reference token-match là dense DFlash, có exact length
  match riêng và phân biệt E[L] với E[1+L].

### Phase B

- Parent entropy/confidence online được truyền cùng schema parent_signal với
  offline fit; confidence nằm trong RoundState và được tính cùng entropy.
  Callback cũng giữ margin fields khi chuyển draft signals.
- Sửa opportunity screening: adaptive policy chịu extraction/controller cost;
  fixed-k baseline không chịu chi phí đó. Best fixed-k được chọn bằng selection
  repeats không có adaptive overhead; chấm bằng evaluation repeats độc lập.
- Tài liệu hazard dùng risk set/censoring đúng; bỏ claim hidden-state mixture,
  ba thành phần trực giao, history-to-probability không có cơ sở và hội tụ Newton
  tức thời. Feature/offset phải đồng nhất giữa fit/predict.
- Gate dựa min cost trên toàn bộ action được hỗ trợ, không giả định k=3 rẻ nhất.
  Bỏ “0 ms tuyệt đối”, “99.98% traffic” và kết quả ablation điền trước.
- Hazard/gates mới vẫn là đề xuất; chưa biến chúng thành kết quả GPU đã xác nhận.

## 3. Tái đánh giá dữ liệu legacy

Đường dẫn report canonical:

`outputs/phase_a_logic_audit_20261018/legacy_corrected/report.json`

Lệnh:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_candidate_benchmarks.py \
  --target-dir outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/vector_snapshots \
  --output-dir outputs/phase_a_logic_audit_20261018/legacy_corrected \
  --budgets 256,512,1024,2048,4096 --chunk-sizes 1,128
```

Bộ này có 10 document và 120 unique snapshot. Chỉ L0/L18/L35 và ensemble L18/L35
có reference hợp lệ. Max-query và các tầng chưa thu được ghi unavailable, nên
report ở trạng thái partial. Dữ liệu mean cũ vẫn giữ giá trị diagnostic nhưng
không còn được gắn thành kết quả M1 max-query hoặc bằng chứng đa tầng.

## 4. Probe GPU riêng Phase A

Modal profile đã kiểm tra: phuclk30. Workspace này có volume nhưng chưa có
model/experiment cache của các run lịch sử; không truy cập workspace khác.
Probe dùng public model Qwen/Qwen3-4B và z-lab/Qwen3-4B-DFlash-b16, cache vào
volume của phuclk30, cùng tokenized prompts đã lưu từ bộ GovReport nguồn.

Lệnh:

```bash
MODAL_PROFILE=phuclk30 .venv/bin/modal run --profile phuclk30 \
  scripts/modal_phase_a_probe.py \
  --run-id phase-a-query-audit-20261018 --samples 2 --max-rounds 8
```

[Modal run](https://modal.com/apps/phuclk30/main/ap-d2DqYMCFbDX3MACuLtRVYd).
Đây là probe attention trên một L40S, không đo tốc độ hệ thống A+B.

- 2 document × 2 context cap (8192,16384) × 8 round = 32 round/snapshot độc lập.
- Snapshot lưu đủ tầng 0,12,14,16,18,20,27,35 và 16 query trong mỗi verifier block.
- Có 36 file snapshot tính cả alias mốc; evaluator chỉ tính 32 round duy nhất.
- 28 transition vòng kế tiếp; không thiếu layer/query trong evaluation cuối.
- Future-key attention bằng 0 trong các snapshot thu được.
- Tiền tố output khớp reference ở cả 4 request ngắn (13–15 output tokens/request).
  Đây là partial-reference replay, không phải exact full-output parity hay
  acceptance dưới context nén.

Artifacts local:

`outputs/phase_a_logic_audit_20261018/phase-a-query-audit-20261018/`

Report dùng để đọc kết quả cuối: `evaluation_final/report.json` và
`evaluation_final/summary.csv`. `evaluation/` và `evaluation_corrected/` là
intermediate audit trước khi khóa evaluator cuối; giữ để truy vết, không dùng
làm bảng báo cáo hiện tại.

### Coverage proxy sau sửa

B=2048, chọn token, protection 16/32; mask từ round t chấm mean-15-query target
attention L18 ở t+1 trên keys đã tồn tại ở t. Mean được tính macro theo document.
Mỗi ô có 2 document và 14 transition, cùng support giữa ba phương pháp.

| Context | Parent query cùng round | Max accepted-prefix queries | Mean accepted-prefix queries |
|---|---:|---:|---:|
| 8192 | 91.6554% | 91.6380% | 91.6574% |
| 16384 | 85.0968% | 85.5129% | 85.6142% |

Các số này cho thấy cần giữ cả mean và max để đánh giá tiếp; không xác nhận max
là phương pháp tốt nhất. Chỉ 2 document/8 round đầu, chưa có CI xác nhận, và
reference vẫn gồm speculative suffix; không suy ra acceptance/speedup. Các điều
kiện khác bộ legacy nên không dùng chênh lệch hai bảng để suy ra suy giảm phương pháp.

Figures được xuất từ report cuối tại `outputs/phase_a_logic_audit_20261018/figures_final/`.
CPU pooling timing được ghi riêng tại `pooling_cpu_timing.json`, không đưa vào
bảng GPU hoặc full selector overhead.

## 5. Kiểm tra hồi quy

Các kiểm tra tập trung vào:

- layer/query missing, mean-vs-max, budget protection và document aggregation;
- lựa chọn mask từ quá khứ, parent control cùng round và causal query positions;
- parent signal schema offline/online;
- adaptive overhead so với fixed-k không overhead;
- các luồng survival, uncertainty, history và context replay liên quan.

Lượt kiểm tra cuối đạt 63/63 tests trong 6.66 giây; có một cảnh báo CUDA driver
đã biết của máy local CPU, không có test thất bại. Các test tập trung trong
`test_phase_a_evidence.py`, `test_phase_b_observational.py`,
`test_proposed_context_selectors.py`, `test_speculation_controller.py`, các test
signals/survival/history và context intervention/replay liên quan.
`git diff --check` sạch; các script/module đã sửa được parse bằng AST.
Probe GPU không lỗi, 4/4 request khớp reference prefix. Figures lấy từ report
cuối với hash evaluator và metric helper, không có bảng số gán sẵn.
Các artifacts kết quả nằm trong outputs (gitignored); tài liệu và mã là phần
có thể đưa vào version control.

## 6. Tái phân tích quan sát riêng Phase B

Script `run_claim1_pre_method_experiment.py` trước đây chuẩn hóa top-16 logits
thành probability margin như thể là toàn vocabulary, gán nhãn cả suffix sau
first failure và suy diễn p-value IID thành bằng chứng nhân quả. Script đã sửa:

- Thiếu full logsumexp thì không tạo confidence/probability margin full-vocab.
- Dùng top-two logit gap và ratio exp(-gap), các đại lượng xác định được từ trace.
- Chỉ lấy baseline stage; bỏ redraft interventions khỏi phân tích chính.
- Chỉ dùng risk set và chia 5 fold theo document gốc; chuẩn hóa feature trên
  training fold, chấm NLL/Brier continuation trên document held-out.
- CI bootstrap theo document cho chênh NLL; bỏ Wald/token t-test IID và các hình
  được đặt tên là “chứng minh leakage” từ dữ liệu quan sát.

Lệnh đã chạy:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_claim1_pre_method_experiment.py \
  --output-dir outputs/phase_b_logic_audit_20261018/observational_corrected \
  --bootstrap 2000
```

Dữ liệu lịch sử GovReport/CNN-DM/MultiNews ở context khoảng 1K: 90 document,
270 baseline blocks, 700 risk-set positions; loại 8100 intervention positions.
Đây là phân tích dữ liệu cũ trên CPU, không phải benchmark GPU mới của hazard.

| Mô hình continuation | Document-macro OOF NLL | OOF continuation Brier |
|---|---:|---:|
| Logit margin + position | 0.535274 | 0.180067 |
| Thêm suffix top-two-ratio exposure | 0.536777 | 0.181578 |

Chênh NLL (suffix − baseline) = +0.001503; CI95 theo document
[-0.003447, +0.006472]. Chưa có bằng chứng held-out cho lợi ích của feature
suffix-ratio trên cấu hình này. Kết quả chỉ áp dụng cho proxy logit-gap/ratio,
trace lịch sử và model/control đã chạy; chưa đánh giá probability-margin hazard
hoặc context dài 8K–32K. Report có source hashes và ghi causal_evidence=false.
