# Đặc tả Context-Adaptive DFlash V1

Ngày: **2026-10-08**. Trạng thái: **đặc tả canonical cho executor experimental; chưa có GPU parity hoặc benchmark result**. [Index](../../README.md), [tích hợp](integration.md), [protocol](experiment_protocol.md), [runbook](runbook.md).

## 1. Mục tiêu, giới hạn và bất biến

Tối ưu E2E latency của summarization bằng hai tài nguyên của drafter: lượng context đọc và số speculative tokens. Target luôn xử lý prompt đầy đủ, dùng full KV và exact verification. Training-free nghĩa là không cập nhật trọng số hoặc train predictor; calibration chỉ thống kê latency, quantiles và prefix acceptance.

V1: một request/process, batch size 1, Transformers adapter của Qwen3 + checkpoint DFlash tương thích. Giữ tokenizer, chat template, thinking mode, dtype, sampling và stopping giống các control. vLLM/SGLang, block-indexed kernel, KV offload và giảm resident memory là các bước sau prototype, không phải điều kiện đã có.

Bất biến:

- Target KV chỉ chứa full prompt và output prefix đã xử lý đúng.
- Full draft-context bank chứa K/V chiếu từ **target hidden features**, không chứa K/V mask/anchor tạm của draft block.
- Selected keys giữ absolute positions và RoPE gốc; không dùng physical sparse length để suy logical position.
- Controller chỉ đọc signals đã có trước action; oracle cùng round chỉ dùng phân tích offline.
- Không thêm target forward chỉ để lấy entropy/relevance.
- Không dùng gold summary, teacher tương lai hoặc verification suffix sau rejection làm online evidence.

## 2. Ký hiệu và ranh giới round

| Ký hiệu | Định nghĩa |
|---|---|
| `P` | Số prompt tokens thực, gồm instructions/template/source/query |
| `n_t` | Số tokens target đã xử lý, tương ứng full logical KV length trước round |
| `z_t` | Pending anchor được target chọn; token tại vị trí `n_t`, chưa được xử lý |
| `B_t` | Trần context keys mỗi draft layer, **không gồm current block** |
| `gamma_t` | Số speculative candidates; `block_size = gamma_t + 1` gồm anchor |
| `L_t` | Accepted candidate prefix length, từ 0 tới `gamma_t` |
| `C_t` | Tokens mới được xử lý/commit trong round, trước EOS cắt: `1 + L_t` |
| `S_l,t` | Sorted unique logical positions trong `[0, n_t)` của draft layer `l` |

Trong vòng lặp vendored, target xử lý anchor và candidates, giữ anchor + accepted prefix, chọn correction/bonus làm pending anchor mới, rồi crop target cache tới `n_t + 1 + L_t`. Feature của pending anchor mới chỉ có sau khi token đó được target xử lý ở round kế tiếp. Finalization được phép xuất pending anchor đã được target chọn, nhưng phải hạch toán boundary riêng theo measurement contract.

## 3. A — Chọn draft context

### 3.1. Metadata và budget

`PromptLayout` phải biết source token positions trên **đúng prompt đã tokenize**. Instructions, query QMSum và template tokens không thuộc source đều được bảo vệ. Không tự suy source bằng cách xem toàn prompt là tài liệu.

Tập bảo vệ:

```text
G_t = non_source_prompt_positions
      ∪ first min(64, source_length) source positions
      ∪ last min(256, processed_output_length) processed output positions
```

Các nhóm loại trùng. `B_t` bao gồm `G_t`; anchor và `gamma_t` mask positions nằm ngoài budget. Action reduced không khả thi khi `|G_t| > B_t`; controller loại action đó, không bỏ instructions. `full` giữ mọi vị trí `[0,n_t)`, kể cả output cũ ngoài recent window. `B >= n_t` được canonicalize thành `full` để tránh action trùng.

Chia source theo thứ tự thành chunk tối đa 128 source tokens, không vượt source span. Xếp chunk theo score giảm dần, tie theo vị trí đầu nhỏ hơn. Bổ sung **toàn bộ chunk** chưa chọn nếu số vị trí mới không vượt budget; chunk không vừa bị bỏ qua. Không cắt tùy tiện một chunk cuối. Budget thực có thể nhỏ hơn trần và phải log `selected_context_tokens`/unused budget. Source chunk overlap với anchor protection chỉ tính số vị trí chưa được giữ.

### 3.2. Selector chính và đối chứng

| ID | Signal và cách chọn | Causality/chi phí |
|---|---|---|
| `target_parent` | Mean attention qua heads của target parent query; sum mass theo source chunk | Signal từ prefill hoặc verification trước; thu Q/K có overhead |
| `draft_refresh` | Mean attention qua proposal queries và heads, riêng từng draft layer, tại dense draft gần nhất | Dense bootstrap/refresh là round sinh thật, tính đủ chi phí |
| `recent_only` | Fill budget từ source chunks gần cuối nhất | Không gọi collector |
| `random` | Hoán vị chunks bằng seed gắn request/round | Cùng budget và protection |
| `lexical` | BM25-style ranking source chunks bằng tối đa 64 processed output tokens gần nhất | Ablation CPU; first round dùng task query hoặc recency |
| `feature_cosine` | Cosine giữa pooled source target features và last processed target feature | Heuristic; không mặc định là grounding score |
| `current_draft_oracle` | Ranking từ dense forward của chính trạng thái cần draft | Chỉ offline quality upper reference; không làm speed baseline |

`target_parent` là phiên bản chuyển giao cơ chế CMR để triển khai đầu tiên. Trong Qwen3-4B chọn target layer 18 (quy tắc mặc định `num_hidden_layers // 2`); layer 0 và layer cuối là ablation. Không thay selector trong một cell. Dev chọn selector cho C bằng E2E latency sau correctness gate, giữ các đối chứng kể cả khi target-parent không tốt nhất.

Repo đã đo target-parent và DFlash attention khác nhau trên cohort 10 GovReport, đặc biệt ở context dài; phép đo này chưa là can thiệp cache. Vì vậy bổ sung `draft_refresh`, không suy rằng CMR chắc chắn giữ nguyên lợi ích khi chuyển sang DFlash. Xem [báo cáo parent](../../../../docs/experiments/2026-10-06_dflash_parent_attention_next_draft.md).

### 3.3. Target-parent signal

Sau prefill, dùng query của prompt token cuối. Sau verification có `L_t` accepted candidates, dùng **query index `L_t`**; query đó tạo correction/bonus cho round sau. Không dùng last verifier row khi có early rejection. Query này chỉ nhìn prefix đã được giữ.

Reconstruct attention từ Q/K đã qua normalization/RoPE của selected target layer, áp đúng GQA, scale, causal/sliding mask và FP32 softmax trên full allowed key scope. Chỉ sau softmax mới sum source mass thành chunk scores. Không chuẩn hóa source trước softmax toàn scope. V1 lấy signal mỗi round; update frequency 2/4/8 là ablation có cost riêng.

Target kernel output vẫn dùng SDPA/FA backend đã khóa. Collector không được âm thầm đổi target sang eager. Collector nghiên cứu trong probe là hướng dẫn audit, không import trực tiếp vào core. Nếu kernel không cung cấp Q/K hook đáng tin cậy, đánh dấu backend unsupported; không bịa signal hoặc bỏ overhead.

Hook chỉ gắn vào selected **target module**, không sửa attention registry dùng chung khiến draft cũng bị instrument. Prefill chỉ giữ last Q; verification giữ tối đa `gamma+1` Q rows đến khi biết L rồi reconstruct row L với K/cache view đúng forward. Future keys trong view phải bị causal mask loại. Release tensor references/scratch trước forward kế tiếp; không lưu toàn Q/K/attention theo round vào artifacts. Hook phải được tháo cả khi request lỗi.

### 3.4. Draft-refresh signal

Round đầu dùng `full`; subsequent dense refresh mỗi 4 round theo `(round_index % 4 == 0)`, với round 0 là bootstrap. So period 2/4/8 trên dev. Refresh ép `B=full` nhưng controller vẫn được chọn `gamma` có support. Lượt đó sinh và verify bình thường, không chạy thêm dense draft để lấy ranking rồi bỏ candidates.

Chỉ mean proposal query rows `1:gamma+1`, giữ current block trong softmax denominator rồi loại block khi xếp source chunks. Giữ ranking riêng từng layer; shared ranking từ mean layers là ablation. Output mới vào recent protection ngay, feature/KV được tạo lại từ target. Sparse attention không thấy omitted keys, nên không cập nhật global ranking bằng cách gán chúng score 0. Khác gamma ở refresh có thể tạo ranking khác; log gamma/age của ranking, khóa period trên dev.

## 4. B — Uncertainty và độ dài block

Gamma candidates ban đầu: `{3,7,11,15}`, chỉ bật shape đã qua parity/runtime gate và không vượt checkpoint block size. Gamma 0 (`block_size=1`) là AR control/emergency mode có engine riêng; không gọi draft head với slicing `1-block_size` khi block size 1.

V1 **thực sự draft với shape `gamma+1`** và verify đúng shape đó. Variant giữ draft block 16 rồi cắt verification prefix có ID `verify_prefix`, được đánh giá riêng. DFlash dùng attention hai chiều nên prefix của block dài không được coi là kết quả draft block ngắn.

Entropy signal dùng logits tạo pending anchor: prefill last logits hoặc verification row `L_t`. Nó dự đoán anchor đã chọn, không phải phân phối sau anchor. Đặt tên `parent_entropy`, dùng

```text
p = softmax(parent_logits / 1.0), tính FP32
H = -sum(p * log p), H_normalized = H / log(vocab_size)
```

Signal temperature 1.0 độc lập generation temperature; greedy vẫn có entropy hợp lệ từ raw logits. Không lấy entropy từ phân phối one-hot sau argmax. Tính trên GPU, chỉ chuyển scalar tại synchronization point cần cho acceptance. Draft confidence sau drafting chỉ có thể quyết định verification-prefix variant, không dùng ngược thời gian để chọn current draft shape.

## 5. C — Policy thống kê training-free

### 5.1. State và calibration

`s_t` gồm parent entropy, source relevance concentration, recent acceptance history, logical context length, remaining output budget và refresh schedule. Không sử dụng target entropy như hàm của draft budget: target context giữ nguyên.

- Entropy bins: 4 bins theo 25/50/75% quantiles trên calibration, giữ cutpoints cố định khi chạy dev/test.
- Source concentration: mass của tối đa 8 source chunks cao nhất / total source mass; bins `<0.5`, `>=0.5`, `unknown` nếu không có source mass.
- Với `draft_refresh`, tính concentration trên từng layer có source mass rồi mean đều các layer hợp lệ; scalar giữ origin/age của dense ranking. Không tính lại global concentration từ sparse observations.
- History: EMA `L/gamma`, alpha 0.1; bins `<0.5`, `>=0.5`, `unknown` trước observation đầu.
- Cost context buckets: `ceil(n_t / 2048)`; không dùng chi phí context ngắn cho context dài ngoài support.
- Statistics key luôn chứa checkpoint/runtime signature, selector ID, length mode, budget, gamma và refresh/non-refresh.

Calibration thu cost và prefix outcomes cho các action trên cohort riêng. Replay states giống nhau khi so action: tái draft/verify từng gamma; không cắt trace block dài. Cost prior không được chứa timing của oracle score extraction hoặc model load.

### 5.2. Expected commit và cập nhật

Với action `a=(B,gamma)`, ước lượng prefix survival

```text
p_i = P(L >= i | state, selector, B, gamma), 1 <= i <= gamma
E[C | s,a] = 1 + sum_i p_i
```

Mỗi observation cập nhật cho mọi `i<=gamma`: success khi `L>=i`, failure khi `L<i`. Đây là nhãn **prefix survival**, không phải kết luận từng suffix token độc lập bị reject. Khi `L=gamma`, observation right-censored: không cập nhật vị trí `i>gamma` hoặc action có shape khác.

Ước lượng thống kê V1:

```text
n <- 0.95*n + 1
u_i <- 0.95*u_i + indicator(L >= i)
p_i <- (u_i + 16*p_calibration_i) / (n + 16)
```

Prior `p_calibration_i` lấy từ calibration cùng action; n/u ban đầu bằng 0 là request-local online counts. Bucket support là `calibration_unique_state_count + n_request_effective`, không đếm timing repetitions. Với support ít hơn 8, backoff lần lượt bỏ concentration/history, rồi entropy; cuối cùng dùng action-global prior. Không chia sẻ outcome giữa gamma khác nhau. Repair `p_i=min(p_i,p_(i-1))`, clamp `[0,1]`. Không có prior hợp lệ cho action thì action không có support.

Latency prior là arithmetic mean các profiling repetitions cho cùng signature/action/context bucket; record đủ total cost và components. Cập nhật EMA alpha 0.1 bằng measurements đã hoàn tất; CUDA event chưa ready không được đọc/chờ chỉ để ra quyết định. V1 có mode `frozen_cost` khi backend chưa cung cấp reliable async timing; vẫn adapt acceptance online và phải báo mode đó. Đo E2E thực để kiểm chứng surrogate.

Cost table dùng chung giữa policies lưu action cost không gồm controller. Sau khi dựng table, profile `choose` trên causal states với đủ action set để có controller overhead riêng từng policy/signature; thêm overhead đó vào J. Fixed-action replay không được giả chi phí joint controller bằng 0. Nếu lưu total-round prior/EMA thay vì components, key phải chứa controller ID và chỉ dùng cho đúng policy; không cộng controller lần hai. `cost_update_mode` có hai giá trị `frozen_cost|async_event`, async chỉ bật khi completed total/component observations có accounting hợp lệ.

`statistics_update_mode=online` là mặc định: cập nhật prefix statistics đúng executed action. Mode `frozen` giữ nguyên prefix/cost priors trong request để làm control; causal history scalar vẫn cập nhật từ các round thực. `frozen` bắt buộc đi cùng `cost_update_mode=frozen_cost`. Không dùng outcome ở `(B,gamma)` để cập nhật action giả định `(full,gamma)` hoặc `(B,gamma_reference)`. Independent A+B truy vấn các reference axes trong cùng table nhưng chỉ cập nhật cell thực đã execute; reference cells có thể vẫn dùng prior. So joint/independent ở cùng mode, thêm frozen comparison để tách joint decision khỏi khác biệt học thống kê online.

### 5.3. Chọn action

```text
J(s,a) = (T_controller + T_signal + T_selection + T_bank_update
          + T_draft(B,gamma) + T_verify(n_t,gamma)) / E[C | s,a]
```

Costs gồm overhead collector/refresh và bootstrap; số đo hardware timing theo measurement contract. Full baseline action là `(full,15)` hoặc gamma full-context tốt nhất được khóa trên dev.

1. Loại action vượt checkpoint, positional limit, budget protection, cost support hoặc remaining output budget.
2. Draft-refresh round chỉ xét `B=full`.
3. Chọn `J` nhỏ nhất. Tie trong `1e-9` ưu tiên budget thực lớn hơn rồi gamma nhỏ hơn.
4. Chỉ đổi khỏi full fallback khi predicted cost thấp hơn fallback ít nhất 2%; `controller_margin=0.02` khóa trên dev.
5. Thiếu signal/statistics dùng full fallback. Khi full cost chưa được calibrated, dùng full fixed action trực tiếp và log `uncalibrated`.
6. Reset request EMA/history về calibration prior cho mỗi request; streaming cross-request learning là experiment riêng.

Near EOS/output cap: gamma được clip tới số candidate slots còn hợp lệ; nếu shape clip không có parity/support thì dùng AR step. Boundary rounds ghi đúng executed gamma và không gộp vào regular cost calibration.

## 6. Execution và xử lý lỗi

```text
target_prefill(full_prompt) -> processed prefix, pending anchor, parent signal
append/project target context features vào full draft bank
while còn output budget và chưa EOS:
    state <- signals đã có + history + n_t
    action <- controller(state), áp refresh và boundary constraints
    selection <- selector(state, action.B), giữ logical positions
    candidates <- DFlash(selected bank + anchor/masks, action.gamma)
    outcome <- target verify trên full KV
    commit anchor + accepted prefix; lưu correction/bonus pending anchor
    crop target KV; append chỉ target features của committed rows vào bank
    lấy parent logits/attention tại row L; cập nhật statistics đã sẵn sàng
finalize pending anchor nếu còn budget; cắt EOS và tính boundary accounting
```

Nếu selector thất bại trước verification, bỏ scratch, giữ nguyên full bank/target KV và redraft full; tính wasted work vào E2E và log fallback. Nếu target KV/position invariants hỏng, abort request với error record; không tiếp tục trên cache có thể sai. Device-side assertion, invalid CUDA state và checkpoint mismatch là lỗi request/run, không phải tín hiệu để policy thử budget khác.

Exactness áp dụng dưới verifier/sampler hợp lệ. Greedy phải khớp target-only token IDs. Sampling với draft greedy có thể dùng target-sampled prefix match của vendored DFlash; xem proof và RNG contract ở [integration](integration.md). Không suy distribution equivalence từ ROUGE hoặc cùng seed.

## 7. Quyết định đã khóa và phần sẽ chọn bằng dev

Đã khóa: full target context, frozen weights, bank/temporary-block separation, original positions, actual draft shapes, causal signals, request reset, total cost accounting, error rows không giả success.

Chọn bằng dev trước heldout: selector chính, target layer, refresh period, protected recent size (128/256/512), supported gamma set, best fixed pair và margin nếu thay default. Ghi toàn bộ lựa chọn vào locked config; không chỉnh bằng test result.

Full bank + gather V1 nhắm giảm draft reads/compute. Resident memory reduction và source factuality improvement không phải claim mặc định. [Protocol](experiment_protocol.md) quy định bằng chứng cần có trước mỗi claim.
