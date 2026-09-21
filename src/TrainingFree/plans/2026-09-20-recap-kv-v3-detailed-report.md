# Báo cáo triển khai và đánh giá RECAP-KV V3

Ngày lập báo cáo: 2026-09-20

Phạm vi: `src/TrainingFree`, Phase 1 của ý tưởng **Global Observability + Local Exactness**.

## 1. Kết luận điều hành

Đã triển khai được prototype V3 và chạy pilot thật trên Modal với Qwen3-0.6B:

- 40/40 tài liệu hoàn tất, gồm 20 `gov_report` và 20 `multi_news`.
- GPU xác nhận trong manifest: NVIDIA A10, CUDA hoạt động, 0 sample lỗi.
- Upper bound không có vi phạm và missed attention mass bằng 0.
- Tuy nhiên router đã phải mở rộng chính xác **100% source tokens** ở mọi bước decode.
- Vì vậy method chưa đạt mục tiêu giảm chi phí attention; gate dừng trước physical KV executor: `STOP_BEFORE_PHYSICAL`.

Kết luận đúng của thí nghiệm là: **prototype và pipeline đo lường đã chạy đúng, nhưng bound hiện tại quá lỏng để tạo sparsity thực tế**. Chưa có cơ sở để tuyên bố tăng tốc hoặc triển khai physical KV mutation.

## 2. Ý tưởng đã triển khai

V3 thay đổi trọng tâm từ dự đoán future importance sang:

> Global observability + local exactness: full prefill, xây hierarchy theo document → region → block, dùng representative để định tuyến, chỉ cho phép exact attention trên active blocks sau khi qua gate an toàn.

Thiết kế cụ thể:

1. Full prefill tạo K/V như baseline.
2. Mỗi KV head có hierarchy gồm region 1024 token và block 128 token.
3. Mỗi block và region có 4 representative được chọn bằng deterministic farthest-point sampling.
4. Với query `q`, tính upper bound:

   `U_i(q) = max_r q·p_ir + ||q|| ε_i`

   trong đó `ε_i` là coverage radius của cluster.
5. Chọn region rồi refine xuống block theo upper bound và mass budget 1%.
6. Trong Phase 1, active set chỉ được dùng để audit; không sửa KV cache thật.
7. So sánh active set với exact attention để đo missed mass, expansion và soundness.

Gate được đăng ký trước khi chạy:

| Tiêu chí | Ngưỡng |
|---|---:|
| Maximum missed attention mass | ≤ 1% |
| Mean exact expansion | ≤ 30% |
| Index overhead | ≤ 10% |
| Upper-bound violations | 0 |
| Coverage | ≥ 10 tài liệu/dataset trên ≥ 2 dataset |

## 3. Các thành phần đã triển khai

- `src/TrainingFree/hierarchy.py`: hierarchy, representative selection, coverage radius, upper-bound primitives.
- `src/TrainingFree/hierarchy_routing.py`: coarse-to-fine routing và exact audit metrics.
- `src/TrainingFree/hierarchy_collector.py`: lấy hidden state/query trong inference, xây hierarchy và ghi trace theo từng decode step.
- `src/TrainingFree/hierarchy_evaluation.py`: aggregation, dataset coverage gate và quyết định follow-up.
- `src/TrainingFree/schema.py`: schema/version và validation cho hierarchy trace.
- `src/TrainingFree/run.py`: thêm experiment `hierarchy`, CLI options và report artifacts.
- `scripts/modal_trainingfree.py`: forwarding cấu hình hierarchy và output root V3 trên Modal.
- `src/TrainingFree/tests/` và `tests/test_trainingfree_hierarchy_contract.py`: unit/contract tests cho hierarchy, routing, collector, schema và Modal launcher.
- `src/TrainingFree/plans/2026-09-18-recap-kv-v0-design.md`: cập nhật design V3.
- `src/TrainingFree/plans/2026-09-18-recap-kv-v0.md`: cập nhật plan và acceptance checklist V3.

Physical KV mutation chưa được triển khai có chủ ý. Điều này phù hợp với safety gate: chỉ chạy physical executor sau khi routing audit cho thấy active set nhỏ nhưng vẫn giữ missed mass trong giới hạn.

## 4. Kiểm thử cục bộ trước khi chạy GPU

Quy trình kiểm thử theo TDD:

- RED: test mới fail vì các module V3 chưa tồn tại.
- GREEN: sửa lần lượt hierarchy primitive, zero-budget behavior, overhead đếm representative trùng, import collector và Modal forwarding.
- Kết quả bộ test V3/core sau khi sửa: 13 test pass.
- Contract test Modal và hierarchy: 8 test pass.
- Full test suite liên quan ở bước final verification: 53 test pass, compileall không lỗi.

Một lỗi hiệu năng được phát hiện trong smoke đầu tiên: routing dùng nhiều torch operation nhỏ trong vòng lặp, khiến một step mất khoảng 5.4 giây. Đã vector hóa phần bound computation; synthetic route giảm còn khoảng 0.03 giây/query ở trace 12,346 token. Tuy vậy audit exact bound vẫn là chi phí đáng kể và được ghi riêng trong `routing_time_ms`.

## 5. Các lần chạy Modal

| Run | Cấu hình | Kết quả |
|---|---|---|
| `recap-v3-hierarchy-smoke-qwen06b-a10g` | 1 sample, bản routing chưa vector hóa | Hoàn tất; dùng để phát hiện bottleneck 5.4 s/step |
| `recap-v3-hierarchy-smoke-qwen06b-a10g-v2` | 1 sample, routing vectorized | 1/1, 28.401 s, 0 violation, expansion 100% |
| `recap-v3-sweep-r1024-b128-r8` | 1 sample, reps block/region = 8 | 1/1, overhead tăng lên 6.56%, expansion vẫn 100% |
| `recap-v3-hierarchy-pilot-qwen06b-a10g-default` | 40 samples, 20 + 20, reps = 4 | 40/40, 550.91 s, 0 error, `STOP_BEFORE_PHYSICAL` |

Pilot chính được chạy bằng:

```bash
MODAL_GPU=A10G modal run scripts/modal_trainingfree.py \
  --experiment hierarchy --mode pilot --max-new-tokens 32 \
  --run-id recap-v3-hierarchy-pilot-qwen06b-a10g-default \
  --region-size 1024 --block-size 128 \
  --reps-per-block 4 --reps-per-region 4 \
  --hierarchy-mass-budget 0.01 --hierarchy-layers last
```

Lần invoke đầu của pilot bị preempt sau sample 5; Modal tự khởi động lại function và lần chạy sau hoàn tất toàn bộ 40 sample. Đây là gián đoạn vận hành, không phải sample failure. Manifest cuối cùng là nguồn sự thật cho số lượng sample/GPU.

App Modal của pilot: <https://modal.com/apps/tdphuc-work/main/ap-JJqtnin2b0Jcs7PgfEl1qA>.

## 6. Kết quả pilot chính

Model và runtime:

| Trường | Giá trị |
|---|---|
| Model | `/opt/models/Qwen3-0.6B` |
| Layer | layer cuối, id 27 |
| GPU | `NVIDIA A10` theo manifest |
| Region/block | 1024 / 128 token |
| Representatives | 4 / block, 4 / region |
| Mass budget | 0.01 |
| Max new tokens | 32 |
| Documents | 40 |
| Decode steps trung bình | 31.1/document |

### 6.1 Aggregate

Các metric dưới đây được tính lại từ `hierarchy_trace.jsonl` bằng evaluator hiện tại, theo mean trên document; do đó có thêm `routing_time_ms` so với metrics artifact được ghi ở thời điểm pilot.

| Metric | Kết quả | Gate |
|---|---:|---|
| Missed attention mass | 0.0000 | PASS |
| Maximum missed attention mass | 0.0000 | PASS |
| Exact expansion fraction | 1.0000 | FAIL, ngưỡng 0.30 |
| Index overhead | 0.03377 = 3.38% | PASS, ngưỡng 10% |
| Routing representative fraction | 0.03645 | Chỉ là chi phí index, không phải KV reduction |
| Active QK tokens | 5,835.18 | Không giảm |
| Full QK tokens | 5,835.18 | Baseline audit |
| Upper-bound violations | 0 | PASS |
| Upper missed-mass bound | 0.0000 | PASS |
| Routing/audit time | 288.28 ms/step | Chưa phải speedup |

### 6.2 Theo dataset

| Dataset | Docs | Missed mass | Expansion | Index overhead | Routing/audit time |
|---|---:|---:|---:|---:|---:|
| `gov_report` | 20 | 0.0000 | 1.0000 | 3.33% | 431.52 ms/step |
| `multi_news` | 20 | 0.0000 | 1.0000 | 3.43% | 145.05 ms/step |

`active_qk_tokens` bằng `full_qk_tokens` ở cả hai dataset. Đây là kết quả quan trọng nhất: missed mass bằng 0 không chứng minh router hiệu quả, vì router đã giữ lại toàn bộ block.

### 6.3 Representative sweep

Ở smoke `recap-v3-sweep-r1024-b128-r8`, tăng representatives từ 4 lên 8 cho block/region:

- expansion vẫn bằng 1.0;
- missed mass vẫn bằng 0;
- index overhead tăng lên khoảng 6.56%;
- routing fraction tăng lên khoảng 7.13%.

Do đó chỉ tăng số representative không giải quyết được vấn đề bound lỏng; nó chỉ tăng chi phí index.

## 7. Diễn giải và giới hạn

### Đã xác nhận

- V3 hierarchy có thể tích hợp vào pipeline `TrainingFree` và chạy trên model/checkpoint hiện có.
- Modal launcher truyền đúng các tham số hierarchy và ghi manifest/trace/metrics.
- Trace schema, exact attention audit và upper-bound soundness hoạt động trên 40 tài liệu thật.
- Với cấu hình hiện tại, bound không có violation trên pilot.
- Index overhead nằm dưới 10%.

### Chưa được xác nhận

- Chưa có reduction thực tế về QK/KV bytes, latency, VRAM hoặc throughput.
- Chưa có so sánh latency end-to-end với baseline dense hoặc RECAP-KV V0/V2 trong cùng run.
- Chưa chạy ROUGE/quality comparison; pilot này đánh giá routing safety/coverage, không đánh giá chất lượng summary.
- Chưa chạy physical KV executor và chưa có lý do an toàn để chạy nó.

### Nguyên nhân kỹ thuật của gate fail

Coverage radius của block trong không gian key nhiều chiều tạo upper bound quá bảo thủ. Với budget 1%, tổng upper mass của các block bị loại không giảm đủ, nên router chọn toàn bộ block. Kết quả này phù hợp cả với synthetic diagnostic và pilot thật: thay đổi block size/replication đơn giản hoặc tăng representatives không tạo sparsity.

`routing_time_ms` hiện bao gồm routing cộng exact upper-bound audit phục vụ nghiên cứu; không được diễn giải là latency của executor tối ưu. Việc này làm con số routing/audit thận trọng hơn nhưng không thay đổi kết luận expansion = 100%.

## 8. Trạng thái theo validation pyramid

- **L0 static/unit:** đạt.
- **L1 runtime smoke trên Modal:** đạt, có phát hiện và sửa bottleneck vectorization.
- **R5 short pilot:** đạt về mặt hoàn tất dữ liệu và audit soundness, 40/40 sample.
- **Physical KV / systems comparison:** chưa được phép chạy vì fail expansion gate.
- **Kết luận nghiên cứu hiện tại:** `STOP_BEFORE_PHYSICAL`.

## 9. Khuyến nghị bước tiếp theo

Chỉ nên thực hiện một vòng routing follow-up có kiểm soát, giữ nguyên 40 tài liệu và các gate hiện tại, thay coverage radius toàn cục bằng bound query-directional chặt hơn (ví dụ projected radius hoặc compact score sketch theo query direction). Mục tiêu duy nhất là đưa exact expansion xuống dưới 30% mà vẫn giữ missed mass ≤1% và zero upper-bound violation.

Không nên thêm learned router hoặc physical KV mutation trước khi vòng bound mới chứng minh được active set nhỏ. Nếu bound mới vẫn cho expansion gần 100%, nên dừng nhánh V3 này thay vì tiếp tục tối ưu executor.

## 10. Artifact và bằng chứng

Artifact pilot đã tải về local trong:

- `/tmp/recap-v3-hierarchy-pilot-final/manifest.json`
- `/tmp/recap-v3-hierarchy-pilot-final/hierarchy_trace.jsonl`
- `/tmp/recap-v3-hierarchy-pilot-final/hierarchy_metrics.json`
- `/tmp/recap-v3-hierarchy-pilot-final/hierarchy_metrics.jsonl`

Báo cáo này ghi kết quả thực nghiệm, không thay đổi kết luận gate trong manifest: prototype đã chạy xong, nhưng method hiện tại chưa đạt điều kiện để triển khai physical KV.
