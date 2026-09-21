# Workboard nghiên cứu và triển khai của repo

> Tài liệu điều hành trung tâm cho các ý tưởng và chức năng quan trọng trong
> `fast_infer_text_sum`. Mục tiêu là đọc một file này để biết repo đang làm gì,
> phần nào đã có code, phần nào đã test/chạy được, bằng chứng hiện tại mạnh đến
> đâu và bước tiếp theo là gì.

Benchmark reference để đối chiếu Training-Free: [Modal reference](experiments/2026-09-21_modal_trainingfree_reference.md)
và [metadata/config JSON](experiments/2026-09-21_modal_trainingfree_reference.json).

**Ngày snapshot:** 2026-09-21  
**Commit nền:** `ffe40fc` (`cap nhat quá trình Eval`)  
**Lưu ý trạng thái:** working tree hiện có thay đổi chưa commit ở các nhánh
`MR_DFlash`, `TrainingFree` và các tài liệu/runtime liên quan. Những phần này
được ghi theo code hiện tại, không chỉ theo commit nền.

## Cách đọc trạng thái

Mỗi workstream được đánh giá theo ba trục độc lập:

| Trục | Ý nghĩa |
|---|---|
| **Code** | Ý tưởng đã được hiện thực đến đâu trong source/script. |
| **Validation** | Đã qua static/unit, CPU/synthetic, GPU smoke, pilot dữ liệu thật hay full benchmark. |
| **Evidence** | Có thể kết luận khoa học/production đến đâu; artifact chạy được không đồng nghĩa với claim speedup. |

Nhãn dùng trong tài liệu:

- **Đã triển khai:** có code và đường chạy chính.
- **Đã kiểm chứng:** có test hoặc artifact tương ứng; phần phạm vi được ghi rõ.
- **Đang triển khai:** còn thay đổi/code hoặc experiment chưa đóng.
- **Chưa đủ bằng chứng:** pipeline chạy được nhưng chưa đủ để kết luận chất lượng, speedup hoặc production readiness.
- **Dừng có chủ đích:** scientific gate đã fail hoặc chưa đạt điều kiện an toàn; không coi là lỗi runtime.

## Bản đồ chức năng chính

| ID | Workstream | Nơi chính trong repo | Trạng thái điều hành hiện tại |
|---|---|---|---|
| `S0` | Runtime chung và benchmark orchestration | `scripts/common/`, `config/`, `scripts/run_longbench_200.py`, `scripts/collect_metrics.py` | **Đã triển khai; đang hoàn thiện audit/final report** |
| `B1` | Benchmark các baseline tăng tốc inference | `scripts/infer_*.py`, `scripts/run_*.sh`, `externals/`, `docs/baselines/` | **Đã có matrix, smoke và full B200 artifacts; cần chuẩn hóa diễn giải** |
| `F1` | DFlash fine-tuning cho tóm tắt tiếng Việt | `src/Finetuning/` | **Core và synthetic pipeline đã chạy; chưa đóng real Vietnamese quality/speedup** |
| `M1` | MR-DFlash: memory-aware learned drafter | `src/MR_DFlash/`, `scripts/mr_dflash/` | **Prototype/pipeline source-to-cache đã triển khai; pilot train/eval thật còn là việc chính** |
| `T1` | Training-Free RECAP-KV | `src/TrainingFree/`, `scripts/modal_trainingfree.py` | **E41 xác nhận head heterogeneity; E42 oracle hybrid-head không đạt đồng thời các gate; dừng trước E43/router/physical KV** |
| `Y1` | SyncSpec-v1 speculative decoding | `src/SyncSpec/`, `docs/baselines/syncspec.md` | **Core, test và smoke đã có; chưa có full benchmark canonical để claim** |
| `A1` | Phân tích chẩn đoán và quyết định nghiên cứu | `src/analyze/`, `outputs/dflash_residual/`, `outputs/safe_budget_sum/`, `outputs/specextend_*` | **Nhiều thí nghiệm đã chạy; kết quả dùng để lọc giả thuyết, không tự động là baseline** |

## Thứ tự ưu tiên hiện tại

1. Đóng báo cáo benchmark canonical: kiểm tra coverage, metric contract,
   fairness và tách rõ kết quả đã đủ kết luận với kết quả chỉ là smoke.
2. Hoàn tất pilot MR-DFlash trên server B200: preprocess/cache → train các
   cấu hình control → exactness/evaluation → so sánh acceptance và cost.
3. Với TrainingFree V3, chỉ chạy một vòng cải thiện bound có kiểm soát; không
   triển khai physical KV executor khi expansion vẫn gần 100%.
4. Đưa SyncSpec và Finetuning qua target/model snapshot thật theo đúng gate của
   từng nhánh trước khi đưa vào bảng claim chung.

Các mục trên là định hướng kỹ thuật rút ra từ trạng thái repo. Khi có giao việc
cụ thể mới, ghi giao việc đó vào trường **Việc tiếp theo được giao** của
workstream tương ứng.

---

## `S0` — Runtime chung và benchmark orchestration

### Ý tưởng và chức năng

Repo dùng một runtime/config contract chung để mọi baseline chạy trên cùng
interpreter, master shell-env, data schema và output schema. Lớp này tách
benchmark thành các stage có thể kiểm tra độc lập:

```text
master config → preflight/runtime → load data → run baseline → JSONL output
               → metric audit → aggregate report
```

Các thành phần quan trọng:

- `scripts/common/runtime.sh`, `scripts/common/config.sh`: chọn Python 3.12,
  load master config và giữ quy tắc runtime chung.
- `scripts/common/io_util.py`, `scripts/common/metrics.py`,
  `scripts/common/rouge.py`: output JSONL, metric tốc độ/speculative và chất
  lượng task-aware.
- `scripts/run_longbench_200.py`: runner canonical, preflight GPU, data
  parallel, shard/merge và giới hạn VRAM.
- `scripts/collect_metrics.py`, `scripts/audit_benchmark_metrics.py`: tổng hợp
  và audit artifact sau khi chạy, không cần chạy inference lại.

### Đã triển khai

- Có master config pointer tại `config/master.path` và hướng dẫn server tại
  [`docs/server_environment.md`](server_environment.md).
- Có schema chung cho record/summary, status từng sample và paired speedup.
- Có canonical LongBench 5 dataset, profile `smoke`/`representative`/`full`,
  data-parallel nhiều GPU và nhiều process/card với VRAM budget.
- Có kiểm thử contract cho runtime, dataset, launcher, metric và output.

### Đã test/chạy đến đâu

- Unit/contract tests bao phủ shared runtime, dataset canonical, collector,
  metric audit và launcher.
- Có artifact full B200 5 dataset, 100 sample/dataset, gồm manifest và
  `metrics_summary.json` trong `outputs/Benchmark_results/`.
- Có run bổ sung data-parallel trên 2 B200 và run 8 GPU; các manifest cuối ghi
  `status: success` ở cấp run. Một số baseline như FAFO chỉ có semantics
  aggregate-only, nên không được đọc như per-sample timing đầy đủ.

### Khoảng trống và giới hạn

- `outputs/` là gitignored và artifact có thể đến từ các lần chạy/config khác
  nhau; trước khi kết luận phải khóa run id, model, dataset manifest, GPU,
  seed, max output và metric scope.
- Full benchmark thành công về mặt runtime không đồng nghĩa mọi method đã có
  cùng loại metric hoặc cùng semantics quality.
- Một số baseline trong catalog chưa nằm trong cùng full B200 matrix; xem
  [`docs/model_baseline_matrix.md`](model_baseline_matrix.md) và
  [`docs/longbench_200_benchmark.md`](longbench_200_benchmark.md).

### Việc tiếp theo dự kiến

- Audit các run full bằng `audit_benchmark_metrics.py`, xác nhận metric thiếu
  là do semantics baseline hay do lỗi pipeline.
- Tạo báo cáo cuối cùng phân biệt per-sample, aggregate-only, smoke và full;
  không gộp các scope không tương đương.
- Khi thêm baseline hoặc sửa schema, cập nhật contract test và workboard này.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`docs/README.md`](README.md)
- [`docs/longbench_200_benchmark.md`](longbench_200_benchmark.md)
- [`docs/representative_100_benchmark.md`](representative_100_benchmark.md)
- [`scripts/run_longbench_200.py`](../scripts/run_longbench_200.py)
- [`scripts/collect_metrics.py`](../scripts/collect_metrics.py)
- [`scripts/audit_benchmark_metrics.py`](../scripts/audit_benchmark_metrics.py)

---

## `B1` — Benchmark các baseline tăng tốc inference

### Ý tưởng và chức năng

Đây là mặt bằng so sánh công bằng cho long-context summarization. Repo gom các
method theo nơi chúng giảm chi phí:

| Nhóm | Baseline tiêu biểu | Câu hỏi so sánh |
|---|---|---|
| Semantic/context reduction | LLMLingua, GemFilter, HiGOE, semantic selection, Speculative Prefill | Giảm input có giữ quality và giảm prefill/KV không? |
| Sparse prefill/attention | MInference, FlexPrefill | Không xóa input nhưng giảm attention interaction có hiệu quả ở context dài không? |
| KV/token optimization | FastKV, RocketKV | Giảm chi phí/KV memory trong decode đến đâu? |
| Learned/generic speculative | EAGLE-3, DFlash, SyncSpec | Drafter/selector có tăng acceptance và throughput mà vẫn exact không? |
| Long-context speculative | MagicDec, LongSpec, SpecExtend, SSSD, FAFO | Long-context/batch-aware speculation có lợi hơn generic SD không? |

`Dense AR + FlashAttention/vLLM/SGLang` là control/runtime baseline;
`SpecForge` là infrastructure tham chiếu, không phải một algorithmic baseline
độc lập.

### Đã triển khai

- Có launcher theo quy ước `infer_<baseline>.py` + `run_<baseline>.sh`, được
  nối vào `scripts/run.sh` khi baseline có adapter.
- Có docs riêng tại [`docs/baselines/`](baselines/) và taxonomy tại
  [`externals/baseline_repo_guide.md`](../externals/baseline_repo_guide.md).
- Có output schema chung, ROUGE khi baseline sinh text, acceptance metrics cho
  speculative method và paired reference cho một số adapter.
- Có smoke/representative outputs; full B200 artifacts hiện có các baseline
  `vanilla_hf`, `vanilla_fa`, `magicdec`, `eagle3`, `dflash`, `specextend`,
  `fafo` trên 5 dataset canonical ở các run đã ghi trong `outputs/`.

### Đã test/chạy đến đâu

- Contract tests đã bao phủ nhiều launcher và compatibility path: DFlash,
  EAGLE, FastKV, GemFilter, MInference, MagicDec, SpecExtend, SSSD/FAFO,
  semantic selection, SyncSpec và các baseline khác.
- Có smoke 1 sample/CPU hoặc GPU tùy baseline; smoke chỉ xác nhận đường chạy,
  không dùng để xếp hạng speedup.
- Có full B200 run thành công ở cấp runtime và có thể truy nguyên bằng
  `run_manifest.json` + `metrics_summary.json`.
- Có báo cáo kết quả riêng cho semantic selection; kết luận hiện tại là
  `lead-512` có trade-off tốt trên tập 16 mẫu, nhưng cần representative/full
  validation trước khi biến thành policy.

### Screening Modal mới nhất

Run `tf-screen-l50-20260921-core` đã chạy trên Modal A100 80GB với Llama 3.1
8B Instruct, `gov_report` + `multi_news` + `qmsum`, 20 mẫu/dataset và tối đa
128 token output. Kết quả đã được aggregate và lưu trong Modal Volume
`fast-infer-text-sum-cache` tại
`outputs/longbench_100_14k/tf-screen-l50-20260921-core/`.

| Dataset | Method | n hợp lệ | E2E ms | tok/s | ROUGE-2 F | ROUGE-L F | Trạng thái |
|---|---|---:|---:|---:|---:|---:|---|
| gov_report | Vanilla HF | 20 | 6663.0 | 22.26 | 0.0903 | 0.1400 | đủ |
| gov_report | EAGLE-3 | 20 | 4802.3 | 32.19 | 0.0862 | 0.1242 | đủ; accept 4.23 |
| gov_report | DFlash | 20 | 3465.7 | 41.32 | 0.0960 | 0.1441 | đủ; output checks pass |
| multi_news | Vanilla HF | 20 | 3037.2 | 43.17 | 0.0759 | 0.1526 | đủ |
| multi_news | EAGLE-3 | 20 | 1872.0 | 73.45 | 0.0573 | 0.1404 | đủ; accept 4.25 |
| multi_news | DFlash | 20 | 1488.0 | 95.74 | 0.0791 | 0.1538 | đủ; output checks pass |
| qmsum | Vanilla HF | 4 | 6643.5 | 20.30 | 0.0915 | 0.2087 | OOM từ mẫu 5 |
| qmsum | EAGLE-3 | 5 | 5967.1 | 19.50 | 0.0634 | 0.1835 | OOM từ mẫu 6 |
| qmsum | DFlash | 20 | 4593.4 | 29.17 | 0.0829 | 0.2032 | đủ; output checks pass |

Đây là screening systems/quality, chưa phải bảng claim cuối: `vanilla_fa`
chưa chạy vì image FlashAttention source build không hoàn tất; RECAP-KV V3
chạy riêng trên Qwen3-0.6B nên không được gộp vào các số Llama 3.1 8B. Hai ô
qmsum bị OOM phải được giữ nguyên là thiếu coverage, không xem mean partial
như kết quả đầy đủ. `metrics_summary.json` còn cho speedup ghép cặp với
Vanilla HF; chỉ diễn giải speedup khi coverage pair đủ.

### Khoảng trống và giới hạn

- Không phải baseline nào cũng trả cùng metric: một số kernel smoke không sinh
  text; FAFO có thể aggregate-only; vì vậy phải đọc metric contract trước khi
  so sánh.
- Chưa nên dùng một bảng aggregate duy nhất để kết luận mọi baseline thắng
  nhau ở cả TTFT, TPOT, E2E, memory và quality.
- Chất lượng hiện chủ yếu là ROUGE/metric task-aware; factuality/human quality
  chưa phải gate chung.

### Việc tiếp theo dự kiến

- Hoàn chỉnh matrix baseline theo cùng model, data, output budget và reference
  semantics; ghi các ô `unsupported`, `blocked`, `aggregate-only` thay vì bỏ
  qua.
- Chạy lại `vanilla_hf/qmsum` và `eagle3/qmsum` bằng context/memory policy
  tương thích, hoặc chọn GPU/profile đủ VRAM; không điền số giả vào bảng.
- Quyết định có cần build FlashAttention wheel/cache ổn định để đưa
  `vanilla_fa` vào cùng matrix hay giữ Vanilla HF làm dense control.
- Chạy lại các ô bị thiếu metric sau khi audit; sau đó viết báo cáo speed/
  quality theo từng nhóm phương pháp.
- Khi thêm method mới, cập nhật cả adapter, contract test, docs baseline và
  workboard.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`docs/model_baseline_matrix.md`](model_baseline_matrix.md)
- [`externals/baseline_repo_guide.md`](../externals/baseline_repo_guide.md)
- [`docs/semantic_selection_analysis.md`](semantic_selection_analysis.md)
- [`scripts/run.sh`](../scripts/run.sh)

---

## `F1` — DFlash fine-tuning cho tóm tắt tiếng Việt

### Ý tưởng và chức năng

`src/Finetuning` port quy trình train DFlash theo hướng offline, target Qwen3
được freeze và optimizer chỉ cập nhật draft model. Training dùng trajectory do
target sinh, hidden-state feature cache và DFlash objective; gold summary giữ
lại để đánh giá quality sau cùng.

Luồng chính:

```text
raw Vietnamese JSONL
  → target generation/trajectory
  → hidden feature capture
  → DFlash objective + trainer
  → checkpoint/export
  → generation evaluation với target_exact_rate + ROUGE + paired speedup
```

### Đã triển khai

- DFlash model/objective, strategy, schedule, trainer, checkpoint và feature
  manifest trong `src/Finetuning/`.
- CLI tách stage: `generate_targets.py`, `capture_features.py`, `run_train.py`,
  `generation_evaluation.py`.
- Contract bảo đảm target frozen, dùng local snapshot khi offline, cache atomic,
  resume/checkpoint có provenance và không dùng raw JSONL thay cho feature store
  trong bước train.
- Có config Qwen3-4B, Qwen3-8B và synthetic smoke.

### Đã test/chạy đến đâu

- Unit/contract tests bao phủ model, objective, strategy, data/features,
  trainer, checkpoint, evaluation và teacher generation.
- Synthetic training lifecycle đã có artifact trong
  `outputs/finetuning-synthetic/` với các checkpoint `synthetic-step1` đến
  `synthetic-step4` và marker `COMPLETE`.
- Có quy định rõ: synthetic smoke chỉ xác nhận lifecycle, không phải bằng chứng
  về chất lượng tóm tắt tiếng Việt, VRAM hay speedup.

### Chưa đủ bằng chứng

- Chưa có trong repo một kết quả final trên dataset tiếng Việt thật với
  `target_exact_rate=1.0`, ROUGE và paired speedup đủ để claim production.
- GPU/Qwen3 local-snapshot smoke là opt-in và phụ thuộc checkpoint/model cache
  trên server.

### Việc tiếp theo dự kiến

- Chuẩn bị dataset tiếng Việt thật và snapshot Qwen3 local; chạy generation →
  capture → train trên quy mô nhỏ trước.
- Kiểm tra `target_exact_rate` trước khi diễn giải speedup; nếu exactness fail,
  dừng ở checkpoint/contract debugging.
- Sau pilot mới quyết định scale data, layer count và benchmark inference.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`src/Finetuning/README.md`](../src/Finetuning/README.md)
- [`src/Finetuning/plans/2026-09-07-dflash-specforge-training.md`](../src/Finetuning/plans/2026-09-07-dflash-specforge-training.md)
- [`src/Finetuning/run_train.py`](../src/Finetuning/run_train.py)
- [`src/Finetuning/generation_evaluation.py`](../src/Finetuning/generation_evaluation.py)

---

## `M1` — MR-DFlash: memory-aware learned drafter

### Ý tưởng và chức năng

MR-DFlash là nhánh nghiên cứu sửa draft model DFlash để tận dụng memory ở độ
phân giải hỗn hợp:

- **HCA** gom context theo nhóm lớn để giữ global evidence.
- **CSA** index/select vùng chi tiết hơn theo query để giảm context phải dùng.
- Draft vẫn giữ block-parallel objective, target verifier và accepted-only memory
  update của DFlash.

Mục tiêu không chỉ là có model chạy, mà là kiểm tra liệu heterogeneous-resolution
memory có đem lại acceptance/cost tốt hơn DFlash control ở cùng data, feature,
loss và ngân sách tham số hay không.

### Đã triển khai

- `src/MR_DFlash` là workspace self-contained: model, MR memory, training,
  inference, data/feature store, checkpoint, trainer và config.
- Có các config Qwen3-4B pilot, Qwen3-8B và Llama 3.1 8B; có ma trận
  DFlash-2L / MR-2S / DFlash-5L cho pilot.
- Có preprocess từ source data đến regenerated target, tokenize, hidden feature
  cache, training/evaluation và report.
- Có `--cache-auto-batch`, adaptive batch theo length bucket, shared lease,
  resume khác host, parallel GPU stage và heartbeat/progress.
- Có regenerate bằng vLLM trong
  [`scripts/mr_dflash/vllm_regenerate.py`](../scripts/mr_dflash/vllm_regenerate.py)
  cùng tài liệu chuyển tiếp sang SGLang.
- Giữ DFlash baseline inference ngoài `src/MR_DFlash` để không trộn baseline
  production với workspace nghiên cứu.

### Đã test/chạy đến đâu

- Có nhiều unit/contract/smoke tests cho model, memory, loss, checkpoint,
  cache, scheduler, target cache và pilot scripts.
- CPU/synthetic tests xác nhận shape, semantics, checkpoint và scheduler; các
  test GPU được guard theo hardware.
- Pipeline Phase 1 source-to-cache có runbook đầy đủ cho B200, gồm dry-run,
  resume, adaptive batch, parallel worker và fairness check.
- Có tài liệu/protocol cho pilot train/eval, nhưng trong artifact hiện thấy
  chưa có một output MR-DFlash final được đóng cùng report speed/quality để
  gọi là benchmark result hoàn chỉnh.

### Chưa đủ bằng chứng

- `src/MR_DFlash/README.md` ghi rõ chưa có claim speedup/quality GPU trong giai
  đoạn CPU smoke và MR-DFlash chưa đăng ký vào baseline benchmark matrix.
- Cần phân biệt pipeline đã sẵn sàng chạy với checkpoint/training result đã
  được nghiệm thu. Không dùng DFlash baseline full run để suy ra MR-DFlash.

### Việc tiếp theo dự kiến

1. Chạy/hoàn tất preprocess và cache Qwen3-4B theo cùng feature contract cho
   DFlash-2L, MR-DFlash-2S và DFlash-5L.
2. Train pilot cùng seed/data/loss, ghi trainable params, peak VRAM,
   step time, tokens/s và checkpoint provenance.
3. Evaluate với target full-prefix verifier, exactness-check và summary report;
   chỉ sau đó mới đưa vào benchmark inference.
4. Nếu Qwen3 pilot có tín hiệu, lặp lại ma trận Llama 3.1 8B theo config đã
   khóa, không trộn feature cache khác target/layer.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`src/MR_DFlash/README.md`](../src/MR_DFlash/README.md)
- [`docs/mr_dflash.md`](mr_dflash.md)
- [`docs/mr_dflash_pilot_pipeline.md`](mr_dflash_pilot_pipeline.md)
- [`docs/mr_dflash_phase1_pipeline_v2.md`](mr_dflash_phase1_pipeline_v2.md)
- [`docs/mr_dflash_b200_profile.md`](mr_dflash_b200_profile.md)
- [`docs/mr_dflash_vllm_regenerate.md`](mr_dflash_vllm_regenerate.md)

---

## `T1` — Training-Free RECAP-KV

### Ý tưởng và chức năng

Nhánh `TrainingFree` thử giảm chi phí source attention/KV mà không train thêm
model target. Repo đã đi qua ba vòng ý tưởng:

| Vòng | Ý tưởng | Kết luận hiện tại |
|---|---|---|
| V0 | Dùng future-use/hindsight và current-attention để tìm residual evidence, làm negative control cho KV deletion. | Pipeline chạy; scientific gate fail, chưa có cơ sở physical eviction. |
| V2 | Source-state lease với anchor/current query, certificate query-drift và headroom/refresh policy. | Chạy pilot 40 document; refresh quá cao, lease hữu ích quá ngắn; `STOP_BEFORE_PHYSICAL`. |
| V3 | Global observability + local exactness: hierarchy document → region → block, representative và upper-bound routing. | Pilot 40 document chạy đúng/sound nhưng exact expansion 100%; dừng trước physical executor. |

V3 hiện là prototype audit: full prefill và exact attention vẫn là nguồn đối
chiếu; active set chưa được dùng để mutate KV thật.

### Đã triển khai

- V0/V2 schema, policy, collector, evaluator, lease certificate và Modal
  launcher.
- V3 hierarchy, representative selection, coverage radius, upper bound,
  coarse-to-fine routing, exact audit, aggregation/gate và report artifacts.
- CLI `src/TrainingFree/run.py` có các experiment tương ứng; Modal forwarding
  có output root riêng cho V2/V3.
- Contract/unit tests cho policy, schema, collector, evaluator, hierarchy và
  Modal launcher.

### Đã test/chạy đến đâu

- V0 và V2 đã có Modal smoke/pilot; V2 pilot chính đạt 40/40 về runtime nhưng
  source-score avoidance chỉ khoảng 4.84%, chưa có systems value đáng kể.
- V3 có L0 static/unit, L1 Modal smoke và pilot Qwen3-0.6B trên A10:
  40 document, 20 `gov_report` + 20 `multi_news`, 0 error.
- V3 pilot đạt zero upper-bound violation và missed mass 0, index overhead dưới
  ngưỡng; nhưng exact expansion = 1.0, active QK = full QK.
- Kết luận đã khóa trong report là `STOP_BEFORE_PHYSICAL`.

Latest Modal screening `recap-v3-screen-20260921` đã mở rộng lên 60 document
(20 mỗi `gov_report`, `multi_news`, `qmsum`) trên A10G, target Qwen3-0.6B,
`max_new_tokens=32`. Runtime 60/60, error 0; missed attention mass = 0,
upper-bound violation = 0, exact expansion = 1.0, active QK = full QK,
index overhead = 3.36%, routing fraction = 3.61%. Artifact nằm trong Modal
Volume `fast-infer-text-sum-cache` tại
`outputs/recap_kv_v3/recap-v3-screen-20260921/`. Run này củng cố kết luận
audit nhưng không tạo ra số speedup/quality để ghép với bảng Llama 3.1 8B.

Controlled empirical search mới nhất đã chạy E41/E42 trên 40 document
(20 `gov_report` + 20 `multi_news`) với Qwen3-0.6B trên Modal A10. E41 cho
thấy K95 fraction mean lần lượt 0.2429/0.3847 và head heterogeneity đủ để mở
E42. E42 đo toàn bộ sweep global-head 10/20/30/40% × routed-source
5/10/20/30%; không candidate nào đạt đồng thời expansion ≤30%, mean missed
mass ≤1% và P99 missed mass ≤5% trên cả hai dataset. Cấu hình gần nhất là
global 10% + routed 10% (expansion khoảng 21.3%) nhưng P99 missed mass còn
8.75%/9.95%; routed 20% vượt nhẹ expansion 30% và `multi_news` vẫn P99
5.81%. Kết luận là `STOP_BEFORE_E43`; chưa có router hay physical KV.
Chi tiết bảng/config/artifact nằm ở
[`docs/experiments/2026-09-21_trainingfree_controlled_search.md`](experiments/2026-09-21_trainingfree_controlled_search.md)
và JSON đi kèm.

### Chưa đủ bằng chứng / không được diễn giải sai

- Chưa có reduction thật về KV bytes, VRAM, latency hoặc throughput.
- Chưa có quality/ROUGE comparison trong V3 pilot.
- Không được coi `routing_time_ms` là latency của executor production; nó bao
  gồm routing và exact audit phục vụ nghiên cứu.
- Không triển khai physical KV mutation trước khi active set nhỏ mà vẫn giữ
  missed mass trong gate.

### Việc tiếp theo dự kiến

- Không mở E43 block-score oracle, learned router hoặc physical KV executor ở
  trạng thái hiện tại vì E42 chưa có headroom dưới gate.
- Nếu có giả thuyết mới, phải ghi rõ thay đổi bound/gate/quality target rồi
  chạy lại controlled search; không tune threshold trên holdout.
- Chỉ promote một candidate khi đạt đồng thời expansion ≤30%, mean missed mass
  ≤1%, P99 missed mass ≤5%, index overhead ≤10% và có bằng chứng quality/logit
  fidelity tương ứng.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`src/TrainingFree/plans/2026-09-18-recap-kv-v0-detailed-report.md`](../src/TrainingFree/plans/2026-09-18-recap-kv-v0-detailed-report.md)
- [`src/TrainingFree/plans/2026-09-18-recap-kv-v2-detailed-report.md`](../src/TrainingFree/plans/2026-09-18-recap-kv-v2-detailed-report.md)
- [`src/TrainingFree/plans/2026-09-20-recap-kv-v3-detailed-report.md`](../src/TrainingFree/plans/2026-09-20-recap-kv-v3-detailed-report.md)
- [`src/TrainingFree/run.py`](../src/TrainingFree/run.py)
- [`scripts/modal_trainingfree.py`](../scripts/modal_trainingfree.py)
- [`docs/experiments/2026-09-21_trainingfree_controlled_search.md`](experiments/2026-09-21_trainingfree_controlled_search.md)
- [`src/TrainingFree/plans/2026-09-21-controlled-empirical-search.md`](../src/TrainingFree/plans/2026-09-21-controlled-empirical-search.md)

---

## `Y1` — SyncSpec-v1: synchronized lossless speculative decoding

### Ý tưởng và chức năng

SyncSpec nghiên cứu speculative decoding cho summarization dài với hai tầng
điều khiển:

1. drafter sinh candidate rộng;
2. selector dùng source evidence/ngram/hidden features để chọn candidate;
3. survival head ước lượng prefix sống sót;
4. controller chọn budget draft/verify (`K_d`, `K_v`);
5. target verifier kiểm tra exact và commit cache theo transaction.

Mục tiêu là tăng hiệu quả draft/verify nhưng giữ semantics lossless so với target
đầy đủ. Đây là nhánh speculative decoding có training, khác TrainingFree vốn
không train target/drafter.

### Đã triển khai

- Model/drafter, selector, survival head, controller, trajectory cache,
  training stages, prompt/schema, Transformers adapter và exact verifier.
- Có synthetic target/drafter để kiểm tra deterministic; có CPU pipeline và
  CLI stage 0 → train → infer.
- Có cache fingerprint/resume, source evidence và metrics/profile cho batch,
  acceptance, draft/verify cost.
- Có launcher riêng và contract tests; CUDA unavailable được trả về `BLOCKED`
  có cấu trúc thay vì giả lập speedup CPU.

### Đã test/chạy đến đâu

- Unit/contract tests bao phủ model, objective, selector, survival, verifier,
  trajectory, profile, launcher và transformer adapter.
- Có CPU synthetic smoke và B200 train-smoke artifact trong `outputs/`.
- Docs đã có quy trình train drafter, inference, correctness và profile B200.
- Chưa thấy SyncSpec nằm trong full LongBench matrix 7 baseline hiện tại; do đó
  chưa có claim speedup/quality canonical cho SyncSpec.

### Chưa đủ bằng chứng

- Chưa đủ một run target/model/dataset cố định để công bố exactness, acceptance,
  TTFT/TPOT/E2E và quality trên full benchmark.
- CPU timing chỉ là correctness/runtime smoke, không dùng để kết luận B200.

### Việc tiếp theo dự kiến

- Chốt một target snapshot và drafter checkpoint local; chạy preflight B200.
- Chạy representative trước, kiểm tra exact verifier và acceptance distribution;
  chỉ sau đó đưa vào LongBench full.
- So sánh cùng control dense AR và báo riêng chi phí training/checkpoint.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`src/SyncSpec/SyncSpec_v1_design_complete.md`](../src/SyncSpec/SyncSpec_v1_design_complete.md)
- [`docs/baselines/syncspec.md`](baselines/syncspec.md)
- [`src/SyncSpec/verifier.py`](../src/SyncSpec/verifier.py)
- [`src/SyncSpec/transformers_adapter.py`](../src/SyncSpec/transformers_adapter.py)

---

## `A1` — Phân tích chẩn đoán và lọc giả thuyết

### Ý tưởng và chức năng

`src/analyze` không phải một baseline inference đơn lẻ. Đây là khu vực dùng để
trả lời các câu hỏi nhân quả trước khi đầu tư vào model/kernel mới:

- DFlash residual/headroom: token mất khỏi candidate set hay selector sai dù
  token còn trong candidate set?
- source retirement/safe budget: source evidence còn hữu ích đến khi nào và
  policy budget nào không làm mất quality?
- SpecExtend horizon/CMR: source retrieval/cycle intervention có tạo lợi ích
  ngoài artifact upstream không?
- full-infer/profile/output-stopping/groundsync: phân rã runtime, dừng output
  và kiểm tra giả định benchmark.

### Đã triển khai và đã chạy

- Có các package/phân tích chuyên biệt trong `src/analyze/` cùng run manifest,
  trace, metrics và report dưới `outputs/`.
- Có các chuỗi thực nghiệm DFlash residual đến E22/E29, safe-budget và
  SpecExtend horizon/CMR; nhiều run có report chi tiết và regression tests.
- Một số kết quả đã đủ để dừng hoặc hạ ưu tiên hướng nghiên cứu, ví dụ không
  dùng một oracle/diagnostic đơn lẻ để biện minh cho fixed-budget training.

### Trạng thái bằng chứng

- Đây là bằng chứng chẩn đoán/giả thuyết, không tự động là kết quả sản phẩm.
- Mọi claim cần trỏ tới report + manifest + metric definition cụ thể; không chỉ
  trỏ tới tên thư mục output.
- Các kết luận cần được tổng hợp trở lại vào workstream gốc (`M1`, `T1`, `Y1`)
  để không tạo nhánh nghiên cứu bị quên.

### Việc tiếp theo dự kiến

- Mỗi experiment mới phải có hypothesis, gate, input manifest, report và quyết
  định `continue`, `stop` hoặc `defer`.
- Khi một diagnostic đã đủ kết luận, cập nhật workstream gốc và không tiếp tục
  tối ưu một nhánh đã fail gate nếu chưa có giả thuyết mới.

### Việc tiếp theo được giao

<!-- Để trống nếu chưa có giao việc cụ thể. -->

### Nguồn sự thật

- [`src/analyze/`](../src/analyze/)
- [`docs/experiments/`](experiments/)
- [`outputs/dflash_residual/`](../outputs/dflash_residual/)
- [`outputs/safe_budget_sum/`](../outputs/safe_budget_sum/)
- [`outputs/specextend_horizon_cmr/`](../outputs/specextend_horizon_cmr/)

---

## Quy tắc cập nhật workboard sau mỗi đầu việc

Mỗi khi hoàn tất một task/experiment, cập nhật đúng workstream, không chỉ thêm
comment rời rạc vào commit:

```text
Ngày / run id:
Ý tưởng hoặc câu hỏi:
Đã triển khai:
Đã test/chạy:
Artifact và commit:
Kết quả đã xác nhận:
Giới hạn / evidence gap:
Việc tiếp theo dự kiến:
Việc tiếp theo được giao:
Quyết định: continue | stop | defer | promote-to-benchmark
```

### Tiêu chuẩn để đổi trạng thái

- **Code → Đã triển khai:** source/script và entry point đã tồn tại, có docs tối
  thiểu.
- **Validation → Đã kiểm chứng:** test/command đã chạy và artifact còn truy
  nguyên được; ghi rõ CPU, GPU, smoke, pilot hay full.
- **Evidence → Có thể claim:** metric scope, control, dataset, model, hardware
  và quality/correctness gate đã phù hợp với claim.
- **Hoàn tất workstream:** không chỉ code xanh; phải có quyết định cuối và đóng
  evidence gap chính.

## Liên kết điều hướng

- Hướng dẫn repo và baseline: [`docs/README.md`](README.md)
- Môi trường server: [`docs/server_environment.md`](server_environment.md)
- Taxonomy baseline: [`externals/baseline_repo_guide.md`](../externals/baseline_repo_guide.md)
- Dữ liệu: [`data/README.md`](../data/README.md)
- Các kế hoạch/spec chi tiết: [`docs/superpowers/`](superpowers/)
