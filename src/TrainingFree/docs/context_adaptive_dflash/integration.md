# Hợp đồng tích hợp DFlash và tính đúng đắn

Ngày: **2026-10-08**. Trạng thái: **executor dùng các interface dưới đây đã có trong repository; parity/correctness GPU vẫn pending**. [Đặc tả](design.md), [đo lường](measurement_contract.md), [runbook](runbook.md).

## 1. Điểm tích hợp có thật trong repo

| Source hiện có | Chức năng có thể tái sử dụng |
|---|---|
| [dflash/model.py](../../../../externals/dflash/dflash/model.py) | `extract_context_feature`, `DFlashDraftModel`, Q/K/V và loop `dflash_generate` |
| [infer_dflash.py](../../../../scripts/infer_dflash.py) | Model load, offline compatibility, prompt format, E2E timing và JSONL |
| [data_loader.py](../../../../scripts/common/data_loader.py) | `load_prompts`, normalized record và `raw` canonical source |
| [paired_generation.py](../../../../scripts/common/paired_generation.py) | Đồng bộ CUDA tại biên request khi đo generation |
| [probe_dflash_target_comparison.py](../../../../scripts/probe_dflash_target_comparison.py) | GQA/mask audit và chọn đúng parent query; chỉ tham khảo logic |
| [probe_dflash_attention.py](../../../../scripts/probe_dflash_attention.py) | Attention scope và proposal-query aggregation; collector eager chỉ cho diagnostic |
| [io_util.py](../../../../scripts/common/io_util.py), [rouge.py](../../../../scripts/common/rouge.py) | Writer, schema và ROUGE chung |

`dflash_generate` hiện chỉ có một block size cho toàn request, dùng `DynamicCache` với cache length liên tục. Nó chưa hỗ trợ dynamic sparse draft views, controller hoặc round schema mới. Tạo adapter mới trong `src/TrainingFree/context_adaptive/`; giữ baseline vendored làm reference độc lập. Model weights/load format không đổi.

Không import `scripts/probe_*` hoặc `src/TrainingFree/tests` từ core inference. Logic collector cần được port thành module production, với cost và capability checks riêng.

## 2. Module và interface đã triển khai

| Module dưới `context_adaptive/` | Trách nhiệm |
|---|---|
| `config.py` | `AdaptiveConfig`, validation và signature |
| `types.py` | Các dataclass/tensor contracts dưới đây |
| `prompt.py` | Tokenize cùng baseline và xây `PromptLayout` |
| `cache.py` | `DraftContextBank`, append context, gather và positions |
| `attention.py` | Adapter attention với selected bank + temporary block |
| `signals.py` | Parent entropy, target-parent scores và collector capability |
| `selection.py` | Protection/chunk ranking và dense-refresh ranking |
| `controller.py` | `AdaptiveController` cho fixed, one-axis, independent và joint policies |
| `statistics.py` | Prefix survival, cost priors, EMA và serialization |
| `generation.py` | Exact loop, boundary accounting và request timing |
| `schema.py` | Validation manifest/request/round artifacts |
| `benchmark.py` | Corpus/model signatures, runtime loading và request record |
| `calibration.py` | Thu states, action-grid observations và locked priors |
| `report.py` | Paired aggregation/bootstrap và gate reports |

Các type bắt buộc:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Action:
    budget: int | Literal["full"]
    gamma: int

@dataclass(frozen=True)
class PromptLayout:
    prompt_tokens: int
    source_positions: tuple[int, ...]
    global_positions: tuple[int, ...]
    source_chunks: tuple[tuple[int, ...], ...]
    prompt_hash: str

@dataclass(frozen=True)
class RoundState:
    round_index: int
    logical_length: int
    processed_output_tokens: int
    remaining_output_tokens: int
    parent_entropy: float | None
    source_concentration: float | None
    history_acceptance: float | None
    ranking_age: int
    refresh_required: bool

@dataclass(frozen=True)
class Selection:
    positions_by_layer: tuple[tuple[int, ...], ...]
    requested_budget: int | Literal["full"]
    ranking_origin_round: int | None
    selector_id: str
```

Tensor contracts và các method dưới đây là yêu cầu API tương lai:

| Interface | Input → output |
|---|---|
| `prepare_prompt(record, tokenizer) -> (input_ids, PromptLayout)` | Input `[1,P]` long; giữ hash của IDs thật và source mapping |
| `DraftContextBank.append(target_features, positions) -> None` | Features `[1,delta,F]`, positions `[delta]`; append liên tục logical prefix |
| `DraftContextBank.gather(Selection) -> tuple[LayerContext,...]` | Mỗi layer: post-RoPE K `[1,Hkv,B_l,Dh]`, V cùng shape, absolute IDs `[B_l]` |
| `select_context(state, layout, rankings, action, config) -> Selection` | Chọn được từng layer, protection/budget/determinism |
| `JointController.choose(state, feasible_actions) -> Action` | Không forward model, không đọc artifact tương lai |
| `JointController.observe(state, action, outcome, completed_cost) -> None` | Chỉ cập nhật outcome đúng executed action; cost có thể chưa sẵn sàng |
| `draft_block(anchor, positions, layer_contexts, gamma) -> DraftProposal` | Noise `[1,gamma+1,H]`; IDs `[1,gamma]`, raw logits nếu cần |
| `verify_block(anchor, proposal, target_cache, positions) -> VerificationOutcome` | Full target forward; `accepted_prefix`, pending anchor, retained features, parent logits |
| `generate_adaptive(...) -> GenerationResult` | Output IDs, counters, timings, round events, correctness metadata |

`LayerContext`, `DraftProposal` và `GenerationResult` nằm trong `types.py`. Interface thực tế giữ verifier trong generation loop; không có public `verify_block` riêng. `generate_adaptive` trả output/counters/timings/rounds, bao gồm processed commits, final pending token, EOS trimming và output-accounting invariant.

```python
prepare_prompt(sample, tokenizer, *, chunk_size=128, max_tokens=0)
DraftContextBank.append(target_features, positions)
DraftContextBank.gather(selection) -> tuple[LayerContext, ...]
select_context(state, layout, rankings, budget, *, processed_output_start, bank_length, ...)
AdaptiveController.choose(state, feasible_actions) -> (Action, reason, elapsed_ms)
AdaptiveController.observe(state, action, accepted_prefix, action_cost_ms=None)
draft_block(draft, target, bank, selection, anchor_and_masks, position_ids, action, *, layout, ...)
generate_adaptive(target, draft, input_ids, tokenizer, layout, config, *, max_new_tokens, ...)
```

`select_context` trả `None` khi protection set không vừa budget; generation ghi fallback về full context. V1 nhận batch 1 và action `draft_shape`; `async_event` cùng `verify_prefix` bị từ chối trong config validation.

## 3. Prompt/source mapping

Canonical LongBench dùng `raw.context` làm source; `raw.input` là query/instruction, không gom vào source. Dùng đúng renderer và `_format_prompt` của baseline; canonical prompt đã render không áp thêm chat template. Thinking mode phải giống baseline.

Xây source character span từ formatter và map sang tokens bằng offset mapping của tokenizer fast trên chính rendered string. Với template wrapping, tìm source span trên final string; yêu cầu occurrence duy nhất và exact substring. Token giao ranh giới được giữ trong source **và** global protection để không mất instruction; token special offset `(0,0)` thuộc global. IDs từ mapping phải khớp token IDs baseline; không token hóa từng đoạn rồi ghép vì subword boundary có thể đổi.

Input chỉ có `prompt` mà không có source span đáng tin cậy: full-context/B-only vẫn được phép chạy; A/C báo `source_span_unavailable` hoặc explicit full fallback. Không tự đoán source span. Đường tokenize tensor mới dùng `.input_ids` của `BatchEncoding`, hoặc `apply_chat_template(..., return_dict=False)` nếu gọi API tensor trực tiếp.

## 4. Bank và projection

Feature gốc concat target layers theo `draft.target_layer_ids`, với offset hidden-states indexing đúng `extract_context_feature`. Thực hiện `draft.hidden_norm(draft.fc(features))` trước context projections, giống model gốc.

Trong mỗi layer, context K dùng `k_proj`, `k_norm` và RoPE tại original position; V dùng `v_proj`. Bank lưu sau RoPE để gather không xoay lần hai. Append features của target một lần sau prefill và sau mỗi verify cho các input rows `0:L+1`; không append rejected suffix hoặc hidden features của pending correction mới.

Bank được cấp phát theo max logical prompt + output capacity, không reallocate/copy toàn prefix mỗi round. Log resident/capacity bytes thật. Temporary current-block K/V nằm trong scratch riêng; attention không append chúng vào full bank. Khi target verify xong, temporary proposal K/V bị bỏ; accepted tokens có context K/V mới từ target features.

V1 gather materialize selected context mỗi layer, giữ dtype/device giống baseline. Không tính RoPE bằng `arange(B_l)` và không gọi `crop(B_l)` trên full target/draft bank. Layout gather và `contiguous()` phải được tính vào selection latency/bytes.

## 5. Attention và dynamic shapes

Reference attention cho layer `l`:

```text
Q = q_norm(q_proj(current_noise_states)), RoPE tại n_t ... n_t+gamma
K_block = k_norm(k_proj(current_noise_states)), RoPE cùng absolute positions
V_block = v_proj(current_noise_states)
K = concat(K_bank[S_l,t], K_block), theo token axis
V = concat(V_bank[S_l,t], V_block), cùng thứ tự
attention(Q, K, V, is_causal=False) -> output projection và MLP gốc
```

Noise block vẫn chứa một anchor và `gamma` masks. Mỗi layer có thể dùng selected positions khác nhau; all block keys luôn có mặt. GQA mapping và Q/K normalization giữ nguyên. Full selection phải tái tạo bank semantics của DFlash baseline trước khi bật sparsity.

Prototype không dùng CUDA graph chung cho mọi gamma. Nếu có graph optimization, capture buckets riêng cho shapes đã hỗ trợ và ghi compile/capture warmup ngoài steady state. Không chạy shape 16 rồi log như chỉ tốn shape 4. Sparse B buckets có thể cần padded capacity; ghi cả valid/physical keys và tính padding cost.

## 6. Greedy và sampling exactness

### 6.1. Verifier V1

Drafter dùng argmax, **kể cả khi target sampling**. Target tạo distribution tại mỗi input row của block. Giữ longest candidate prefix trùng target token được chọn ở các row tương ứng; tại mismatch lấy token của target làm pending anchor; nếu accept hết, lấy bonus token của target.

Lý do sampling có thể exact với token-ID comparison ở scheme này: greedy draft tại một vị trí là proposal distribution delta tại `d`. Acceptance probability là `p_target(d)`. Nếu mismatch, target sample đã có phân phối `p_target` có điều kiện token khác `d`; tổng xác suất của accept và correction khôi phục `p_target`. Prefix được giữ liên tiếp bảo đảm target probabilities dùng đúng realized prefix. Không áp luận chứng này cho stochastic draft có `q` không phải delta; nhánh đó cần acceptance/rejection correction phù hợp và proof riêng.

Controller chọn action trước target samples của current round và chỉ dùng prefix/pending anchor đã có; adaptation không thay target distribution. Selected draft context không được chuyển sang target model.

### 6.2. RNG và numerical equivalence

Production sampling log seed/RNG mode. `torch.multinomial` theo block có thể tiêu thụ random numbers khác AR; cùng seed không bắt buộc cùng trajectory. Distribution check dùng toy exact enumeration và empirical sampling, không dùng ROUGE.

Debug sampling parity có thể dùng token-position-indexed uniforms và inverse CDF sampler chung cho AR/speculation; random cho vị trí bị bỏ không làm dịch RNG của tokens sau. Debug sampler không được dùng timing để so với production sampler khác.

Greedy output token IDs phải bằng AR và full DFlash. Nếu shape/backend dẫn đến numerical argmax mismatch, đánh dấu exactness fail trên cell; lưu first mismatch và margin/logits cho audit. Không hợp thức hóa mismatch bằng quality score tốt.

## 7. Round transaction và boundary

Giữ authoritative state `(n_t, full_target_cache, bank_processed_length, pending_anchor)`. Trước draft: `target_cache.length == bank_processed_length == n_t`. DFlash noise positions bắt đầu `n_t`, độc lập `B_l`.

Sau verify: `n_next = n_t + 1 + L`; crop target KV tới `n_next`; bank append committed target features ở absolute positions `[n_t,n_next)`; parent row `L` tạo pending anchor tại `n_next`. Việc cập nhật phải hoàn tất trước round sau. Checkpoint request/replay state không lưu rejected suffix như committed state.

Output budget remaining tính `max_new_tokens - (n_t-P)`, bao gồm slot pending anchor hiện tại. Candidate gamma tối đa `remaining-1`. Nếu remaining=1, finalize pending anchor hoặc AR step theo common boundary implementation. EOS pending có thể kết thúc ngay; EOS trong accepted segment trim output và ghi số processed commits vượt EOS. Không dùng EOS nằm trong rejected suffix để dừng.

AR emergency cần target hidden states khi sẽ quay lại DFlash; cập nhật bank/parent signal đúng như round gamma 0. Một runtime không hỗ trợ trở lại từ AR phải đánh dấu fallback-to-AR-until-end rõ ràng, không tiếp tục với bank thiếu features.

## 8. Capability/preflight và acceptance checks

Preflight phải khóa local snapshots, checkpoint config/hash, tokenizer/vocabulary/head compatibility, feature layer IDs, block limit, mask token, positional capacity, attention backend, Q/K capture, dtype và đủ VRAM cho full bank + scratch. `unsupported` khác `error` và không đóng góp speedup.

Checks cần có trước GPU matrix:

- Full adapter parity với vendored: gamma 3/7/11/15, early reject, all accept và multiple rounds.
- Sparse positions `{0,100,3000,9000}` giữ đúng RoPE; full selection cho cùng logits trong tolerance chẩn đoán.
- Bank không còn proposal K/V; output accepted được reproject từ target features.
- Protection vượt budget, source span rỗng/ambiguous, dedup và shape clip đều có đường xử lý xác định.
- Parent entropy/relevance dùng row `L`, không row cuối; outcome chỉ cập nhật executed action.
- Greedy tới EOS/cap và request lỗi không tạo success record.
- Sampling delta-proposal verifier khôi phục toy target distribution; RNG debug parity không bị suffix draws ảnh hưởng.

Các checks runtime/GPU vẫn là gate trong [kế hoạch triển khai](../../plans/2026-10-08-context-adaptive-dflash-implementation.md); việc có module và CLI không đồng nghĩa chúng đã pass.
