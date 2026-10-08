# Đặc tả kỹ thuật AMR-DFlash V0

Ngày: **08/10/2026**. Trạng thái: **V0 code/CPU synthetic tests đã có; model thật/B200 và kết quả khoa học chưa có**.

Nguồn: [proposal](../../../src/ARMdflash/AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md); [index tài liệu triển khai](../../../src/ARMdflash/AMR_DFlash_Implementation/README.md).

## 1. Mục tiêu và phạm vi

Xây memory interface cho pretrained DFlash để thử giả thuyết selection theo acceptance và global compression bổ trợ làm tăng accepted prefix trên long context với overhead đủ thấp.

V0 giữ checkpoint DFlash-5L, block 16, target full-context, greedy temperature 0 và batch 1. Ưu tiên một backend HF/PyTorch cached có dense control chung. Batch lớn, layer-specific routing, adaptive block size, LoRA và speculative sampling là extension sau pilot.

**Ràng buộc toàn cục:**

- Tài liệu tiếng Việt; Python **3.12**; server dùng `python3` hệ thống.
- Local chỉ CPU qua `.venv/bin/python` hoặc executable do `FAST_INFER_PYTHON` chọn. Không sửa GPU/driver local.
- Không thêm dependency mới, installer online hoặc venv từng baseline.
- Model/tokenizer/dataset dùng local snapshot; đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` và `local_files_only=True`.
- Giữ nguyên toàn bộ **5 draft layer**, weights pretrained, block **16** và feature-layer metadata của checkpoint. Bộ `[1,9,17,25,33]` là cấu hình của cohort đã đo, không áp đặt mù cho checkpoint khác.
- Trọng số target và backbone frozen; optimizer chỉ chứa selector/compressor/adapters được khai báo.
- Raw keys giữ original absolute positions/RoPE; live block không bị prune.
- Mọi generation record qua `common.io_util.JsonlWriter`, kết thúc bằng summary; gọi `rouge.add_rouge`/`aggregate_rouge` khi có reference.
- Không sửa trực tiếp `externals/dflash` và không làm thay đổi default DFlash baseline.
- Không dùng full-prefix reference timing, dense mask hoặc attention collector timing để claim acceleration.

## 2. Những điểm phải xử lý từ code hiện có

| Quan sát code | Hệ quả |
|---|---|
| `DFlashDraftModel.forward` dùng `hidden_norm(fc(target_hidden))` rồi cùng context features cho năm layer | AMR chọn/nén trong không gian context đã project, giữ nguyên backbone |
| `Qwen3DFlashAttention` nối context K/V với live-block K/V và dùng RoPE | Selected raw keys, slot keys và live keys phải có đường vị trí tách rõ |
| `dflash_generate` dùng `past_key_values_draft.get_seq_length()` để suy đoạn vị trí | Số entry được giữ không còn bằng logical prefix length; AMR không dùng compact cache length để suy absolute position |
| Dense generator lưu `acceptance_lengths = A + 1` | Không dùng field này như accepted proposals A; canonical AMR ghi riêng A và G |
| `src/MR_DFlash/inference.py` chạy target `use_cache=False` | Chỉ tham khảo correctness, không làm dense control cho latency production |
| `common.io_util.throughput` lấy nghịch đảo mean TPOT | Primary AMR dùng `sum(G)/sum(T)` và field riêng, không thay semantics metric chung |

## 3. Sơ đồ dữ liệu

~~~mermaid
flowchart LR
    T["Target full-context + KV"] --> H["Committed target features"]
    H --> P["Frozen fc + hidden_norm"]
    P --> BANK["Raw memory bank + compact index"]
    P --> CMP["Incremental compressor"]
    Z["Pre-draft state"] --> SEL["Selector"]
    BANK --> SEL
    SEL --> W["Local + selected raw K/V"]
    CMP --> W
    W --> D["Same DFlash-5L + full live block"]
    D --> V["Full-context target verify"]
    V --> U["Commit / rollback / EOS"]
    U --> T
    U --> BANK
    U --> CMP
~~~

Dense bypass dùng checkpoint và target verifier cùng backend. Memory modules không đọc proposal tương lai hoặc current-round verification để quyết định selection hiện tại.

## 4. Hợp đồng state và tensor

Các primitive/config nằm trong `src/AMR_DFlash/core.py` và `config.py`; batch 1 ở V0. V0 chưa có một `DraftState` dataclass riêng: index JSONL và bundle tensor giữ state fields.

| Kiểu / field | Shape / nội dung |
|---|---|
| `DraftState.state_id, document_id, split` | Định danh state và document; split khóa theo document |
| `context_ids` | int64 `[1,N]`; prefix **đã được target xử lý**, trước live anchor |
| `context_positions` | int64 `[1,N]`; original absolute positions |
| `target_features` | `[1,N,d_H]`; concat target hidden layers theo manifest |
| `anchor_id, anchor_position` | Token target đã chọn ở vị trí `N`, chưa thuộc context bank |
| `prompt_length, emitted_output_ids, max_new_tokens_remaining` | Scope prompt/output và bookkeeping dừng |
| `MemoryBank.projected_context` | `[1,N,d]` sau frozen fc/norm, tùy storage policy |
| `MemoryBank.index` | `[1,N,64]` dùng selector; token index gắn absolute position |
| `WorkingMemory.raw_positions` | `[1,B_raw_actual]`, unique, tăng dần, thuộc context |
| `WorkingMemory.raw_features` | `[1,B_raw_actual,d]` nếu chế độ feature projection |
| `WorkingMemory.slots, slot_valid` | `[1,M,d]` và bool `[1,M]` |
| `DraftOutput.proposal_ids, logits` | `[1,15]` và `[1,15,V]` |
| `VerifyResult` | accepted proposal count, emitted IDs, processed committed features, pending anchor, EOS, target-cache cursor |

**Anchor bookkeeping:** một token đã biết/đã emit có thể chưa được target xử lý trong cache. Vì vậy `emitted_output_ids`, `context_ids` và pending anchor là ba khái niệm riêng. Adapter của generator phải chuẩn hóa snapshot sang hợp đồng này. Không giả định `len(context_ids) = prompt_length + output_tokens` ở mọi boundary.

**A/G:** `A_raw` là matched-prefix proposals trước stop clipping, `A_committed` là proposals thực sự emit trước EOS/output cap, `G` là số token mới emit ở round. `G` lấy từ emitter, không suy bằng `A+1`. Anchor đã emit không được đếm lại; pending correction/bonus không append features trước khi được xử lý.

**Emitter V0:** emit anchor đầu ngay sau target prefill; sau verification emit accepted proposals rồi correction/bonus nếu còn budget và chưa gặp EOS. Correction/bonus đã emit trở thành pending anchor chưa được xử lý cho round kế. Initial event có `phase=prefill`; round/terminal events có phase riêng. Tổng G của mọi event bằng output tokens; throughput decode chỉ lấy G và thời gian thuộc decode. Nếu anchor đầu là EOS, dừng mà không draft block.

### Interface lõi

~~~python
# Chữ ký dự kiến; dependencies pretrained/config được bind trong constructor.
def project_context(target_features: Tensor) -> Tensor: ...
def build_bank(state: DraftState) -> MemoryBank: ...
def append_processed(bank: MemoryBank, features: Tensor, positions: Tensor) -> MemoryBank: ...
def select_context(state: DraftState, bank: MemoryBank, raw_budget: int) -> Tensor: ...
def build_working_memory(state: DraftState, bank: MemoryBank, config: AMRConfig) -> WorkingMemory: ...
def draft_forward(state: DraftState, memory: WorkingMemory) -> DraftOutput: ...
def verify_greedy(state: DraftState, draft: DraftOutput, target_cache: TargetCacheState) -> VerifyResult: ...
def evaluate_model(model: AMRModel, states: Sequence[DraftState], mode: str) -> EvalReport: ...
~~~

`Tensor` là `torch.Tensor`. V0 dùng `MemoryConfig` trong `core.py`, `AMRDFlashDraft` trong `model.py`, target cache trực tiếp trong `inference.py`, và fixed-state evaluator trong `evaluation.py`; CLI có fixed-state và rollout evaluation. Train-time cadence/aggregate `EvalReport` chưa được triển khai.

## 5. Selector V0

- Input index: `k_i = W_k h'_i`, dim 64, với `h'_i = hidden_norm(fc(h_i))`.
- Query: concat anchor embedding, projected feature mới nhất và mean của tối đa 16 projected features gần nhất; MLP Linear → SiLU → Linear tới dim 64. Tất cả đều thuộc prefix đã commit; không dùng current DFlash query.
- Score: dot product / sqrt(64), cộng scalar bias theo normalized absolute position. Prompt/output flag chưa có trong V0.
- Local guard mặc định 128 recent positions; sink guard mặc định 0. Các guard nằm **trong raw budget**.
- Loại guard khỏi miền Top-K; chọn phần budget còn lại; sort selected positions theo absolute position sau lựa chọn.
- Nếu N nhỏ hơn raw budget, giữ tất cả. Padding bị mask; zero valid context hoặc budget nhỏ hơn guards phải có xử lý đã định nghĩa: clamp guards theo budget, không tạo index âm.
- Preference scorer V0 dùng mean token score của candidate set đã loại padding. Additive scoring không mô hình hóa mọi tương tác; chunk-swap/set-level labels vẫn là nguồn reward.
- V0 tính lại scores mỗi round, giữ compact selector-key index và append key của feature mới đã commit. Key bị loại ở round trước vẫn có thể quay lại; reuse mỗi 2/4 round chưa triển khai.

## 6. Compressor thiết kế: pooling học được với cập nhật chính xác

Chọn một baseline nhỏ có thể triển khai và replay streaming; chưa dùng HCA/CSA hoặc recurrent transformer phức tạp.

Thiết kế đề xuất chia absolute timeline thành M bucket cố định trong run. `span = ceil((max_input_tokens + max_new_tokens)/M)`; bucket của token i là `floor(position_i/span)`. Đây là allocation toàn lịch sử có vị trí ổn định; nếu vượt capacity, báo lỗi cấu hình, không silently truncate.

Trên projected context:

~~~text
a_i = w2 · SiLU(W1 h'_i)                   # hidden dim 64
v_i = h'_i + U SiLU(V h'_i)                # residual rank 16
R_b = sum_i exp(a_i) v_i / sum_i exp(a_i)   # i thuộc bucket b
~~~

Mặc định M=128. Bucket rỗng có `slot_valid=False` và không tham gia softmax. Accumulator dùng FP32 và log-sum-exp ổn định: giữ max log-weight, scaled numerator/denominator; append chỉ cập nhật bucket chứa token mới. Không nén lại complement mỗi round. Khi weights compressor thay đổi, accumulator cũ không hợp lệ; rebuild từ feature store cho training/evaluation checkpoint mới.

Global slots chứa cả information trùng selected raw states. Vai trò bổ trợ được học qua loss khi raw selection cùng hiện diện; không gọi đây là phân hoạch disjoint hoặc trừ attention mass bằng heuristic.

### Slot adapter và vị trí

Mỗi layer dùng pretrained k_proj/v_proj với residual adapter rank 16 cho slots; giữ k_norm của layer. Raw keys dùng nguyên RoPE tại absolute position.

V0 chọn slot position là integer midpoint của span các **token thực sự đã xử lý** trong bucket, không dùng vị trí token tương lai trong bucket. Slot K dùng cùng RoPE tại midpoint; V không rotate. Đây là convention phải ablate, không bảo đảm đại diện tương đương nhiều raw keys.

Thêm logit bias `log(sigmoid(g_layer))` trên slot keys, khởi tạo `g_layer=-4`, để hạn chế distribution shift khi bắt đầu. Bias không đặt trên raw/live keys. Khi compressor disabled, bỏ hẳn slot entries/bias; dense parity không dựa vào gate gần zero.

Compressor/adapter được train bằng target alignment qua frozen DFlash. Không bọc draft forward trong `no_grad` khi train compressor; chỉ frozen feature extraction được detach.

**Phần hiện thực V0:** `ComplementaryCompressor` dùng learned soft assignment trên toàn bộ projected context và weighted-mean slot values, cập nhật bằng running sums. Slot positions là weighted centroid của feature positions đã quan sát. V0 chưa hiện thực fixed timeline buckets, residual rank-16 projection hoặc per-layer slot K/V adapters; DFlash gốc chiếu cùng slot feature qua K/V projections của từng layer. Đây là scope rút gọn cần ghi khi báo cáo.

## 7. Memory interface của DFlash

`AMRDFlashDraft` trong `model.py` giữ reference tới backbone pretrained; không copy/giảm layer và không thay parameter names để load weights.

Trên mỗi draft layer:

1. Tính Q từ live hidden states bằng projections/norm pretrained.
2. Lấy raw context K/V từ bank hoặc project selected features; K đã rotate theo absolute positions.
3. Tạo slot K/V theo mục 6; tạo đủ live-block K/V theo DFlash gốc.
4. Attention không causal giữa các live positions, giữ visibility gốc; concat raw + valid slots + live block.
5. Dùng o_proj, residual, MLP, layer norms và final norm gốc.

**Dense identity test:** với mode dense hoặc all-context raw selection, slots disabled, cùng dtype/backend/mask, logits phải khớp original DFlash trong tolerance số học và argmax phải khớp. Test bao gồm positions không liên tiếp để bắt lỗi compact renumbering. `fc/hidden_norm` không được chạy hai lần.

## 8. Verifier và lifecycle

V0 verifier chỉ greedy. Không dùng equality với target sampled tokens để claim exact speculative sampling.

Đường V0 hiện có:

- Labeling dùng target DynamicCache có prefix copy/crop về cùng state sau mỗi candidate.
- Rollout inference dùng target KV cache full prefix, verify block và crop sau rejection; dense và AMR đi qua **cùng** engine. Prefix cursor dùng logical absolute position, độc lập compact draft memory.

Fixed-state evaluation command hiện có trong `evaluation.py`; timing của nó là correctness-oriented và chưa được dùng làm production latency. Target verify candidate proposals trên history tương ứng; chỉ accepted-prefix target features được persist. Correction/bonus có thể còn pending: xử lý token đó trước khi đưa feature vào context bank. Crop cache sau rejection; prune emitter theo EOS và remaining output cap; không giữ rejected hidden features.

Bootstrap/build bank, first draft, switching gate, compression update và terminal work đều nằm trong timing. Target prefill tách khỏi decode, nhưng cả hai thuộc E2E.

## 9. Storage và gate

V0 quality path lưu projected features rồi project selected keys; production prototype thử full per-layer raw KV bank + compact selector index. Cả hai vẫn có full resident bank để retrieval; không claim giảm tổng resident VRAM chỉ vì attention đọc ít key.

Đo bytes cho target KV, full draft bank, index, projected feature bank nếu giữ, slots, temporary gathers và peak toàn hệ thống. Mọi storage variant báo rõ có giữ đồng thời features và KV không.

Gate `dense|amr|auto`: ban đầu benchmark force dense/amr. Auto dùng bảng crossover theo hardware × backend × dtype × batch × context cap từ calibration; ngoài vùng đã đo quay dense. Nếu chưa có calibration artifact, auto báo rõ dense fallback; artifact có checkpoint/config fingerprint sai phải fail. Mode đã force không được gate thay đổi. Short dense bypass không build index/slots; lazy switching phải đo cả build cost.

## 10. Config V0 và extension

Default pilot: raw budget 4096, slots 128, local 128, sink 0, index dim 64, compressor hidden 64, residual rank 16, rescore mỗi round, temperature 0, batch 1.

Raw+slot budget accounting: `B_total = B_raw + M`; live block 16 nằm ngoài budget. Heuristic/selection-only matched-budget dùng `B_total` raw entries; thêm so fixed raw budget để cô lập lợi ích slots. Không so hybrid raw 4K + 128 slots với selection-only raw 4K rồi gọi matched-total-budget.

Config/manifest từ chối checkpoint khác depth/block/feature metadata, sampling, per-layer routing hoặc batch>1 chưa hỗ trợ. Không âm thầm rơi về một engine khác.

Extension chỉ sau full pilot: layer-specific support, advanced compressor, differentiable selector estimator, adaptive budgets, batched cache executor và sampling đúng phân phối.

## 11. File dự kiến và trách nhiệm

| File mới | Trách nhiệm |
|---|---|
| `core.py`, `config.py` | Memory config, selector/compressor, loss và primitive validation |
| `model.py`, `memory.py` | Pretrained DFlash adapter, projected-feature memory interface |
| `inference.py` | Cached greedy verifier, cache crop, pending anchor và rollout trace |
| `pipeline.py`, `candidates.py`, `artifacts.py` | Capture, candidate labels/preferences, train phases và artifacts |
| `checkpoint.py`, `runtime.py`, `budget.py` | Local model loading, strict fingerprints, checkpoint và GPU-hour ledger |
| `tests/test_amr_dflash_*.py` | Semantic/gradient/launcher CPU synthetic tests |
| `scripts/amr_dflash/cli.py`, `scripts/runners/run_amr_dflash.sh` | Preflight, capture, label, train và inference CLI |

Fixed-state evaluation hiện có; full validation cadence, checkpoint resume, profiling CLI, calibrated gate và B200 experiment runner chưa có trong V0.

Chữ ký, artifact và criteria chi tiết: [dữ liệu/train](../../../src/ARMdflash/AMR_DFlash_Implementation/data_training_contract.md), [protocol](../../../src/ARMdflash/AMR_DFlash_Implementation/experiment_protocol.md), [kế hoạch](../plans/2026-10-08-amr-dflash-implementation.md).
