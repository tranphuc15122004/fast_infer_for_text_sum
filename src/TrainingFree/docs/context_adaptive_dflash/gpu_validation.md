# Kiểm chứng GPU trước thực nghiệm Context-Adaptive DFlash

Ngày: **2026-10-09**. Trạng thái: **chưa có báo cáo G0/G1/G2 trên checkpoint thật**. CPU regression đã có trong [review](implementation_review_2026-10-08.md); chúng không thay kiểm chứng GPU. Lệnh có sẵn ở [runbook](runbook.md), gate nghiên cứu ở [protocol](experiment_protocol.md).

## 1. Công cụ hiện có và phần cần bổ sung

| Kiểm chứng | Công cụ hiện tại | Giới hạn |
|---|---|---|
| Corpus/split/model config | `--phase prepare`, `preflight --model-check` | Chưa forward, chưa chứng minh kernel/collector hoạt động |
| Đường chạy GPU ngắn | `--phase smoke` cho AR và full fixed | Phải truyền corpus dev riêng; phase này không tự lọc dev |
| Token equality AR↔adapter | `--phase report --report-phase smoke/dev` | Hash equality trên requests đã chạy; chưa so tensor/cache với vendored |
| Full/sparse tensor parity, rollback, EOS cases | Logic CPU ở [regression tests](../../../../tests/test_context_adaptive_dflash_regressions.py) | Fixture nhỏ, eager/CPU; cần harness checkpoint/GPU riêng |
| Calibration action replay | `--phase calibrate` | Thu priors từ calibration; chưa phải diagnostic intervention study trên dev |
| Sampling distribution / instrumented controls | Chưa có phase CLI | Cần harness riêng; không gọi smoke là audit sampling |

Không có `--phase parity`, `--phase replay`, `--dataset`, hay CLI chọn backend/dtype. Backend/dtype được chọn bởi `load_runtime`/DFlash helper theo GPU và kernels đã cài; lưu kết quả thực vào manifest. Diagnostic eager chỉ để tìm sai lệch, production backend phải được kiểm riêng.

## 2. G0: asset, prompt và runtime

Trước forward, kiểm và lưu:

- Python 3.12, dependency offline, snapshots đầy đủ, target/draft/tokenizer content hashes; config checkpoint không bị sửa để qua preflight.
- Target depth = `draft.num_target_layers`; `target_layer_ids` là danh sách selected layers hợp lệ, không dùng độ dài danh sách thay target depth.
- Vocabulary, hidden size, mask token, block size ≥16 cho gamma 15, RoPE/position capacity và sliding-window config tương thích.
- Prompt IDs giống DFlash baseline; no thinking; actual target-tokenizer length được ghi. Source span map từ final rendered string, giữ instruction/special/boundary tokens; toàn source thuộc document đúng.
- Target/draft eval mode, no gradients; batch 1; backend và dtype giống các đối chứng. Kiểm Q/K capture và parent collector không bật eager hoặc `output_attentions` làm đổi target kernel của primary run.
- Nhóm exposed ở dev hoặc có hồ sơ absent đã xác minh; không forward test trước lock. Cần đủ VRAM cho weights, full target KV, full draft bank và temporary block/gather.

`preflight.json` là một phần G0; reviewer bổ sung checks ngoài config vào `evidence/gpu_validation.json`. Check không hỗ trợ phải ghi `unsupported` cùng lý do, không tính như pass.

## 3. Thiết kế harness checkpoint thật cho G1/G2

Harness dùng API production và vendored làm reference, không thay core bằng fixture. Source tham khảo:

- [benchmark.py](../../context_adaptive/benchmark.py): `load_runtime`, `load_corpus`, `select_split`.
- [prompt.py](../../context_adaptive/prompt.py): `prepare_prompt`.
- [attention.py](../../context_adaptive/attention.py), [cache.py](../../context_adaptive/cache.py): `draft_block`, `DraftContextBank`.
- [generation.py](../../context_adaptive/generation.py): `generate_adaptive`, `generate_target_only`.
- [vendored model](../../../../externals/dflash/dflash/model.py): `extract_context_feature`, `dflash_generate`, `DFlashDraftModel.forward`.

Đặc tả flow dưới đây là **pseudocode cho harness cần viết**, chưa phải lệnh CLI đã triển khai:

```text
load local target/draft/tokenizer bằng loader production; khóa backend/dtype
chọn dev groups đã đăng ký; tokenize bằng prepare_prompt
prefill target full; extract selected target hidden features
build DraftContextBank từ cùng features và absolute positions
cho gamma ∈ {3,7,11,15}:
    tạo cùng anchor + gamma mask tokens
    reference: draft.forward với context features và DynamicCache mới
    adapter: draft_block với full Selection và bank không đổi
    so K/V, draft logits, argmax và bank mutation
    chạy AR, vendored dflash_generate(block_size=gamma+1), full adapter
    so output IDs, first mismatch và boundary accounting
cho sparse positions/budgets/refresh cases:
    reference dùng selected target features cùng absolute positions
    adapter gather từ bank; so K/V/logits rồi rollout so AR
kiểm snapshots trước/sau verify: cache, committed features, parent row
lưu mỗi case cùng hashes/metrics/exception; exit khác 0 nếu required case fail
```

Chọn 2 dev groups/dataset, một query/group để debug; mở rộng tối thiểu 8 GovReport, 8 Multi-News, 4 QMSum groups, cap 256 trước matrix. Export IDs/groups cụ thể vào `validation_cohort.json`; `--max-samples 20` không bảo đảm quota này. Correctness cohorts nằm trong dev/exploratory, không chuyển các nhóm đã xem sang test.

## 4. Cases bắt buộc

| Case | Kiểm trực tiếp | Tiêu chí |
|---|---|---|
| Full bank, gamma 3/7/11/15 | Cùng context/block features, K/V post-RoPE, draft logits/argmax so vendored | Tensor close theo tolerance đã đăng ký; argmax audit; output IDs exact với AR và reference |
| AR / gamma 0 | `variant=ar`, cap/EOS và round schema | Không truyền 0 vào `--gammas`; AR output và schema hợp lệ |
| Sparse absolute positions | `{0,100,3000,9000}` khi P≥9001; nếu nguồn ngắn chọn khoảng cách phù hợp | RoPE dùng original positions; không `arange(B)` hay double rotation |
| GQA và sliding attention | Heads→KV mapping; masks theo khoảng cách logical | Khớp reference mask/backend; không dùng khoảng cách sau gather |
| Reject đầu, accept một phần, accept hết | `L=0`, `0<L<gamma`, `L=gamma` | Target cache/bank trước round sau có length `n+1+L` |
| Commit/rollback | Features rows `0:L+1`; rejected suffix và temporary draft K/V | Không append rejected/pending correction features vào committed bank |
| Parent signals | Query/logits row `L` của verify trước; next pending anchor | Entropy/relevance và origin IDs đúng; không dùng row cuối hay future outcome |
| Per-layer selection | Bank/gather/layer IDs, instructions/anchors/recent output | Protection luôn được giữ; actual valid/physical keys và budget đúng |
| Refresh / fallback | Dense-refresh rounds; protection>B; missing/ambiguous source; unsupported cost bucket | Deterministic full fallback có reason; không đổi target context |
| EOS và cap | Pending EOS, accepted EOS, EOS trong rejected suffix; remaining 1/2; caps 1/2/16/17/31/32/256 | Chỉ emitted accepted/pending EOS dừng; count invariant và target IDs đúng |
| Isolation / resume | Request thứ hai; cùng raw ID ở hai datasets; failed request/retry | Không rò cache/statistics giữa requests; identity gồm dataset/id/repetition |

Chỉ đánh dấu một case pass nếu đã thực sự quan sát hoặc force đúng branch trong diagnostic harness. Real trajectory chưa có all-accept/EOS case thì ghi `not_covered`, bổ sung controlled proposal/stop IDs để audit branch. Forced cases tách khỏi natural quality/timing results.

### Tensor tolerance và greedy exactness

Đăng ký tolerance theo dtype/backend trước chạy. Điểm khởi đầu diagnostic: FP32 `atol=1e-5, rtol=1e-4`; FP16 `atol=1e-3, rtol=1e-3`; BF16 `atol=1e-2, rtol=1e-2`. Lưu max absolute/relative error, tỷ lệ tensor elements vượt tolerance và top-1/top-2 margins. Đây là ngưỡng chẩn đoán, không cho phép output mismatch.

Greedy output IDs yêu cầu **100% trùng** trên toàn cohort, kể cả EOS/cap. Argmax logits draft khác nhau cần audit kernel/semantics dù cuối cùng target sửa đúng. Nếu có output mismatch, lưu vị trí đầu tiên, prefix IDs, action, target logits/margins và cache state; dừng gate. Không nới tolerance sau nhìn kết quả để hợp thức hóa cell. Reference vendored có boundary behavior khác cũng cần ghi lỗi/giới hạn riêng, chưa coi G1 pass bằng việc cắt bỏ mismatch.

## 5. Memory, timing và collector capability

Sau correctness, chạy diagnostic memory/timing trên dev trước rollout dài:

- Reset peak stats sau load; đo allocated/reserved GiB với đầy đủ model, cache, bank, gather và scratch. Ghi peak/bank bytes theo P, B, gamma; sparse attention chưa đồng nghĩa giảm resident memory.
- Primary E2E đồng bộ CUDA ở hai biên request, gồm prefill, bank projection, collector, selection/gather, controller, refresh, draft, verify và finalization. JSON serialization/model load/hash/warmup nằm ngoài E2E; request preparation ghi riêng.
- Kiểm component events không âm/nonfinite; host timings và GPU timings có thể overlap, không dùng tổng components thay E2E.
- So target baseline có/không instrumentation để phát hiện collector đổi kernel hoặc overhead. Diagnostic full-attention export tách khỏi primary wall runs.
- Warmup CLI hiện cap 16, chưa phủ mọi gamma/B/bucket. Ghi JIT/cold-shape effects; nếu thêm shape warmups ngoài timing thì lưu schedule và áp dụng cùng cách cho controls.

## 6. Sampling audit riêng

Primary dùng temperature 0; sampling không chặn việc nghiên cứu greedy nhưng chặn claim distribution exactness cho temperature>0 chưa audit.

Verifier dùng deterministic argmax draft (delta proposal), target sample theo từng vocabulary row, giữ longest matching prefix và target correction/bonus. Audit: toy exact enumeration; tensor logits `[batch,block,vocab]` lấy sample từng row đúng shape; ≥10.000 independent draws/toy state; empirical frequencies và TV error; khóa toy distributions, seed, ngưỡng và multiple-test correction trước chạy. Same-seed production outputs có thể khác AR vì RNG consumption; không dùng token-hash equality để kết luận sampling distribution sai/đúng.

Coupled token-position uniforms có thể giúp debug prefix parity; sampler này thuộc diagnostic scope, không so latency với production sampler khác. Toy/empirical checks bổ sung algorithm/implementation review, không tự chứng minh phân phối bằng nhau tuyệt đối.

## 7. Hồ sơ và quyết định

```text
<run_root>/evidence/
  preregistration.md
  exposure_audit.json
  validation_cohort.json
  gpu_validation.json
  gpu_validation.md
  memory_timing_audit.json
  gates.json
  diagnostic/first_mismatch_...    # chỉ khi cần; artifact gitignored
```

Đây là layout đề nghị cho bằng chứng ngoài executor; CLI chưa tự sinh các file `evidence/*`. Mỗi validation case ghi `case_id`, code/model/runtime/cohort hashes, gamma/B/positions, expected/observed branch, tensor errors/tolerance, exact mismatch count, invariant violations, status, reason và log path. Summary ghi cases expected/executed/passed/failed/not covered/unsupported, CUDA errors và exit status.

Chỉ đánh dấu **G0/G1/G2 pass** sau review đủ required cases trên checkpoint/backend/GPU thật. Thiếu harness hoặc case là `pending`, lỗi là `fail`; không chuyển `pending` thành pass vì CLI smoke exit 0 hoặc report `headline_valid=true`. Sau sửa core/loader/kernel, kiểm lại các gates ảnh hưởng và thu calibration mới khi signature thay đổi.
