# Ghi chép hình thành ý tưởng Context-Adaptive DFlash

> Cập nhật **2026-10-08**: bản triển khai canonical đã được hoàn thiện tại
> [TrainingFree README](../README.md), gồm [đặc tả](../docs/context_adaptive_dflash/design.md),
> [hợp đồng tích hợp](../docs/context_adaptive_dflash/integration.md),
> [protocol](../docs/context_adaptive_dflash/experiment_protocol.md) và
> [kế hoạch implementation](2026-10-08-context-adaptive-dflash-implementation.md).
> Phần dưới giữ nguyên các phiên bản thảo luận ban đầu; pseudocode chưa phải API.
> Dùng bộ canonical để xử lý budget/gamma, parent entropy, sampling exactness,
> timing và novelty. Các giả thuyết/illustrations ở đây chưa là kết quả của executor mới.

Tôi đề xuất chốt phương pháp hiện tại thành một hướng nghiên cứu có trọng tâm rõ ràng:

# Context-Adaptive DFlash for Long-Context Summarization

Working title: Jointly Optimizing Draft Context and Speculation Length for Efficient Long-Context Generation

Ý tưởng cốt lõi: Thay vì thay đổi kiến trúc hoặc huấn luyện lại DFlash, chúng ta xây dựng một training-free adaptive layer gắn vào DFlash, giúp nó sử dụng lượng context phù hợp và chọn độ dài speculative block phù hợp tại từng thời điểm sinh summary.

Mục tiêu là giảm end-to-end inference latency, đặc biệt với workload khoảng 10K input tokens → 2K output tokens, trong khi giữ nguyên hành vi của target model thông qua exact verification.

## 1. Paper story: Từ vấn đề đến phương pháp

1. Problem — Long context makes speculative generation expensive

   DFlash tăng tốc decoding bằng block-diffusion drafting, nhưng khi source document dài, draft context lớn có thể làm tăng chi phí drafting và giảm hiệu quả speculation. Đồng thời, mức độ khó và nhu cầu truy xuất thông tin nguồn thay đổi trong quá trình sinh summary.
2. Observation — Not every generation step needs the same context

   Một số đoạn summary dễ dự đoán từ recent tokens, trong khi những đoạn chứa facts, entities hoặc numbers có thể cần thông tin từ các vị trí xa trong tài liệu.

   Đây là research hypothesis cần kiểm chứng, không phải giả định mặc định.
3. Gap — Existing optimizations address separate decisions

   SpecExtend nghiên cứu draft-context/KV retrieval; AdaFlash nghiên cứu adaptive drafting length. Nhưng câu hỏi về jointly allocating draft-context budget and diffusion block size, dựa trên source grounding và chi phí thực tế, vẫn là hướng cần khảo sát và kiểm chứng thêm.
4. Method — Context-adaptive diffusion drafting

   Một lightweight controller quyết định tại mỗi speculation round: DFlash nên nhìn bao nhiêu context, nhìn phần nào, và draft bao nhiêu tokens?
5. Outcome — Lower latency without changing target context

   DFlash sử dụng selected context để tạo candidates; target model vẫn verify trên full context. Nếu giữ đúng exact speculative verification, thay đổi drafter không làm thay đổi phân phối output mục tiêu.

## 2. Kiến trúc phương pháp đề xuất

Hai quyết định thích nghi tác động lên DFlash; target model luôn sử dụng full context để verification.

Phương pháp gồm ba thành phần:

| Component                    | Chức năng                                                                  | Vai trò trong paper                     |
| ---------------------------- | -------------------------------------------------------------------------- | --------------------------------------- |
| A. Selective Draft Context   | Chọn source blocks/features liên quan, giữ recent tokens và global anchors | Giảm draft computation và memory        |
| B. Adaptive Block Controller | Chọn số tokens DFlash sẽ draft, $\gamma_t$                             | Tránh draft quá dài khi acceptance thấp |
| C. Joint Cost-Aware Policy   | Phối hợp context budget $B_t$ và block length $\gamma_t$           | Contribution trung tâm                  |

Trong đó, source-grounding signal là phần làm phương pháp mang tính long-document summarization-specific, thay vì một controller chung cho mọi tác vụ.

## 3. Formulation: Tối ưu điều gì?

Ở speculation round $t$, controller chọn:

$$ a_t=(B_t,\gamma_t) $$

với:

- $B_t$: draft-context budget.
- $\gamma_t$: số candidate tokens trong một diffusion block.
- $A_t$: số tokens được target chấp nhận.
- $s_t$: trạng thái gồm acceptance history, source relevance, context length và system cost.

Mục tiêu:

$$ \boxed{ \min\_{\pi} \mathbb{E} \left[ \frac{ T\_{\text{selection}} + T\_{\text{draft}}(B_t,\gamma_t) + T\_{\text{verify}}(\gamma_t) }{ A_t } \right] } $$

Đây là surrogate objective theo từng round. Metric cuối cùng vẫn phải là end-to-end latency trên toàn request, vì việc chọn context và drafting có overhead, còn prefill không biến mất.

Controller nên bắt đầu bằng training-free online adaptation, dùng thống kê latency và accepted length quan sát được. Source-grounding có thể ước lượng bằng lexical/retrieval matching hoặc cached target signals; tránh thêm target forward pass đắt tiền chỉ để tính feature.

## 4. Novelty nên đặt ở đâu?

Tôi sẽ không trình bày contribution là “DFlash with context compression”, vì quá gần SpecExtend. Cũng không nên chỉ là “DFlash with adaptive block size”, vì đã có prior art liên quan.

Paper claim hợp lý hơn là:

> A training-free, source-grounding-aware controller that jointly adapts the effective draft context and diffusion block length for long-context speculative generation.

Điểm khác biệt dự kiến:

| Prior work     | Đã giải quyết                        | Phần đề xuất cần chứng minh thêm                             |
| -------------- | ------------------------------------ | ------------------------------------------------------------ |
| DFlash         | Block-diffusion speculative drafting | Context-budget adaptation                                    |
| SpecExtend     | Long-context draft KV retrieval      | Joint control với diffusion block length                     |
| AdaFlash       | Adaptive diffusion drafting length   | Source-aware context budgeting                               |
| UniSpec / SSSD | Training-free, efficient speculation | Integration với diffusion drafter và long-document grounding |

Đây là định vị novelty dự kiến, chưa phải khẳng định rằng chưa có công trình nào kết hợp các thành phần này.

## 5. Paper contributions nên viết thành ba ý

1\. Empirical characterization: Phân tích tác động của input length, output length và draft-context budget lên DFlash latency, acceptance và verification cost, đặc biệt ở workload 10K → 2K.

2\. Joint adaptive method: Đề xuất lightweight training-free controller phối hợp context selection với diffusion block sizing, tối ưu chi phí trên mỗi accepted token.

3\. Long-context evaluation: Chứng minh trade-off giữa latency, memory, acceptance và output correctness trên summarization benchmarks, với target full-context verification.

Nếu source-grounding signal thực sự giúp ích, đó sẽ là một contribution thứ tư hoặc một phần nổi bật của contribution 2.

## 6. Experimental story cần có để paper thuyết phục

Primary experiment: 10K input → 2K output

Các ablation tối thiểu:

| Variant                   | Context budget        | Block length | Mục đích                           |
| ------------------------- | --------------------- | ------------ | ---------------------------------- |
| Vanilla DFlash            | Full                  | Fixed        | Baseline                           |
| DFlash + Fixed Selection  | Fixed reduced         | Fixed        | Kiểm tra lợi ích context reduction |
| DFlash + Adaptive Context | Dynamic               | Fixed        | Đo đóng góp của A                  |
| DFlash + Adaptive Block   | Full                  | Dynamic      | Đo đóng góp của B                  |
| Proposed Joint Method     | Dynamic               | Dynamic      | Kiểm tra hiệu ứng phối hợp         |
| Joint + Grounding         | Dynamic, source-aware | Dynamic      | Kiểm tra đóng góp semantic signal  |

Các metrics quan trọng nhất là E2E latency, TTFT, draft latency, verification latency, accepted length, ms/accepted token, peak memory và output equivalence. Với sampling, cần kiểm tra rằng verifier giữ đúng target distribution, không chỉ kiểm tra greedy output match.

Ngoài 10K → 2K, nên sweep context 4K–32K và output 256–2048 tokens để xác định khi nào phương pháp có lợi. Với summarization datasets có reference ngắn, đánh giá chất lượng trên output length tự nhiên, không ép mọi mẫu sinh đủ 2K tokens.

## 7. Điều kiện quyết định hướng nghiên cứu có khả thi không

Trước khi xây controller đầy đủ, cần xác nhận hai giả thuyết:

H1: Giảm effective context của DFlash thực sự làm giảm draft latency hoặc memory đáng kể mà không khiến accepted length suy giảm quá mạnh.

H2: Optimal context budget và optimal block length thay đổi đủ nhiều giữa các generation states để adaptive controller vượt qua một fixed configuration đã tuning tốt.

Nếu H1 không đúng, hướng context selection không phải bottleneck cần giải quyết. Nếu H2 không đúng, joint controller có thể chỉ thêm overhead mà không đem lại speedup.

Đặc biệt cần kiểm tra khả năng sử dụng sparse selected target features trong DFlash, bảo toàn positional information và KV-cache consistency. Không thể giả định việc cắt ngắn feature sequence sẽ hoạt động đúng mà không sửa attention path.

## 8. Paper story cô đọng — phiên bản có thể dùng để trình bày đề tài

Proposed paper narrative

Long-context summarization creates an inference regime in which a large source document must be processed while thousands of output tokens are generated sequentially. Although diffusion-based speculative decoding such as DFlash accelerates generation, its drafting efficiency can be sensitive to the amount of source context and the chosen speculation length.

We hypothesize that these two resources should not remain fixed throughout summarization: different generation states require different amounts of source evidence and offer different opportunities for speculation.

We therefore propose a training-free, context-adaptive extension to DFlash that jointly selects relevant draft context and adjusts diffusion block length using source-grounding and online latency signals. The target model retains its full context and performs exact verification.

Our central hypothesis is that joint context-budget and speculation-length adaptation can reduce end-to-end latency more effectively than optimizing either component independently, without changing the target model's generation semantics.

Kết luận: Tôi sẽ chọn Joint Context-Budget and Block-Size Adaptation for DFlash làm core method, còn source-grounding-aware selection là yếu tố tạo bản sắc cho long-context summarization. MInference, speculative prefill và n-gram routing nên để làm extensions sau khi chứng minh được core hypothesis.

Câu hỏi nghiên cứu cần trả lời trong paper là:

“How much context does a diffusion drafter actually need at each generation step, and can jointly adapting context budget and speculation length deliver faster exact long-context summarization?”

Để triển khai hai thành phần này trên DFlash, tôi đề xuất một thiết kế training-free ở mức controller, không cần train lại target model hoặc DFlash drafter đã có checkpoint.

Điểm quan trọng là A và B giải quyết hai câu hỏi khác nhau:

- A. Selective Draft Context: DFlash cần nhìn những phần nào của 10K input để dự đoán block tiếp theo?
- B. Adaptive Block Controller: Với context đã chọn, DFlash nên dự đoán bao nhiêu tokens trong một lần?

Hai quyết định này cần phối hợp, vì context càng ít thì drafting có thể càng rẻ, nhưng block dài lại đòi hỏi khả năng dự đoán chính xác nhiều tokens liên tiếp.

## 1. A — Selective Draft Context

### 1.1. Chính xác chúng ta thay đổi gì trong DFlash?

DFlash sử dụng hidden states từ nhiều layer của target model, fusion chúng thành context features, rồi đưa vào draft model thông qua KV injection. Draft model dùng các features này để dự đoán đồng thời một block tokens. Đây là cơ chế được mô tả trong paper DFlash và tài liệu triển khai của NVIDIA/vLLM.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press\&sz=32)

proceedings.mlr.press

+2



Giả sử target đã xử lý 10K source tokens:

$$ H = [h_1,h_2,\ldots,h\_{10000}] $$

DFlash thông thường được conditioning trên toàn bộ context features:

$$ P_D(y\_{t+1:t+\gamma}\mid H,y\_{\le t}) $$

Phương pháp A thay bằng:

$$ P_D(y\_{t+1:t+\gamma}\mid H\_{S_t},y\_{\le t}) $$

trong đó $S_t$ là tập vị trí context được chọn tại speculation round $t$.

Chúng ta không cắt source input của target model. Target vẫn giữ toàn bộ context và thực hiện verification như DFlash gốc.

### 1.2. Chọn context như thế nào?

Tôi đề xuất chia thành ba nhóm:

Full target context: 10K source + generated output

Giữ nguyên trong target model

Recent

Recent summary tokens

Luôn giữ

Relevant

Retrieved source chunks

Chọn động

Global

Instructions, anchors

Luôn giữ

Selected draft context

Ví dụ: 2K–4K features thay vì full 10K+

DFlash → Target verification trên full context

Các budget này là cấu hình thử nghiệm, chưa phải thông số tối ưu đã được chứng minh.

### 1.3. Retrieval bằng chính target features

Không nhất thiết phải dùng thêm embedding model. Ta có thể tận dụng fused target features đã có từ DFlash.

Bước 1 — Chunking: Chia 10K source thành các block 128 tokens:

$$ C_1,C_2,\ldots,C_M,\qquad M\approx79 $$

Bước 2 — Tạo block representations: Mean-pooling hidden features trong mỗi chunk:

$$ k_i=\operatorname{Normalize} \left( \frac{1}{|C_i|}\sum\_{j\in C_i}h_j \right) $$

Việc pooling này chỉ cần thực hiện một lần sau prefill. Có thể thử mean pooling, max pooling hoặc last-token pooling trong ablation.

Bước 3 — Tạo query cho generation state: Lấy hidden feature của token vừa được target xác nhận:

$$ q_t=\operatorname{Normalize}(h_t) $$

Bước 4 — Chấm điểm source chunks:

$$ s_i(t)=q_t^\top k_i $$

Sau đó chọn top-$K$ chunks theo budget:

$$ S_t^{source}=\operatorname{TopK}\_{i}(s_i(t)) $$

Cuối cùng hợp nhất với recent output và global anchors:

$$ S_t=S_t^{source}\cup S_t^{recent}\cup S^{global} $$

Lưu ý: Đây là một heuristic training-free. Cosine similarity giữa target features chưa chắc phản ánh đúng source evidence relevance. Phải kiểm tra nó so với lexical retrieval, random selection, recency-only và oracle selection. Không nên mặc định feature similarity là grounding score tốt.

### 1.4. Gắn vào KV injection của DFlash

Đây là phần triển khai quan trọng nhất.

Trong mỗi draft layer $l$, DFlash có context keys và values:

$$ K_l^H=W\_{K,l}H,\qquad V_l^H=W\_{V,l}H $$

Thay vì cho draft queries attend đến toàn bộ $K_l^H,V_l^H$, ta dùng:

$$ K\_{l,t}^{selected}=K_l^H[S_t] $$

$$ V\_{l,t}^{selected}=V_l^H[S_t] $$

Draft attention sẽ thực hiện trên selected context KV cộng với KV của current diffusion block.

Có hai cách implement:

| Cách                              | Triển khai                                                               | Ưu/nhược điểm                                          |
| --------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| Prototype: KV gather              | Cache full draft-context KV, dùng `index_select` theo selected positions | Dễ kiểm chứng; chưa chắc nhanh do gather/copy overhead |
| Optimized: block-sparse attention | Giữ KV theo blocks, kernel chỉ đọc selected blocks                       | Tránh gather lớn; phức tạp hơn, phù hợp paper systems  |

Chi tiết bắt buộc: Phải bảo toàn original positional indices/RoPE. Không được lấy các tokens ở vị trí 100, 3000, 9000 rồi gán lại vị trí 0, 1, 2, vì như vậy thay đổi positional semantics của DFlash.

Ngoài ra, nếu vẫn lưu full KV để phục vụ retrieval động, ta chủ yếu giảm draft attention bandwidth/compute, chưa chắc giảm peak memory. Muốn giảm memory thật sự cần thêm eviction, offloading hoặc lazy KV materialization.

### 1.5. Pseudocode của A

```
# Conceptual pseudocode — not a drop-in DFlash API# After target prefill:H = fuse_target_hidden_states(target_outputs)source_chunks = split_source_positions(chunk_size=128)chunk_keys = pool_and_normalize(H, source_chunks)# Project and cache draft-context KV once.draft_kv_bank = project_context_kv(H, dflash_layers)def select_draft_context(last_target_feature, budget):    q = normalize(last_target_feature)    scores = chunk_keys @ q    chosen_chunks = topk_chunks(scores, budget)    positions = union_positions(        chosen_chunks,        recent_committed_output_positions(),        global_anchor_positions(),    )    positions = sort_unique(positions)    # Preserve original positional information.    selected_kv = gather_draft_kv(        draft_kv_bank, positions    )    return selected_kv, positions
```

Ở phiên bản đầu, tôi khuyên chỉ implement A với fixed budget, ví dụ 1K, 2K, 4K và full context. Mục tiêu là đo xem giảm draft-context length có thực sự giảm thời gian và giữ được acceptance hay không, trước khi xây adaptive controller.

## 2. B — Adaptive Block Controller

Nếu A quyết định DFlash nhìn bao nhiêu context, thì B quyết định DFlash dự đoán bao xa.

### 2.1. Block size trong DFlash có ý nghĩa gì?

DFlash sử dụng một anchor token đã được xác nhận và một số mask positions để dự đoán các speculative tokens song song.

Ví dụ, với block size 16:

```
[ANCHOR] [MASK] [MASK] ... [MASK]
    1      2      3          16

         DFlash forward
                ↓
[ANCHOR] [tok1] [tok2] ... [tok15]
                ↓
         Target verification
```

Theo cơ chế DFlash mặc định, block size 16 tương ứng với 15 speculative tokens, vì vị trí đầu tiên là anchor.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Ta ký hiệu:

$$ \gamma_t = \text{số speculative tokens tại round }t $$

Một checkpoint hỗ trợ tối đa 15 speculative tokens có thể được thử với:

$$ \gamma_t\in\\{3,7,11,15\\} $$

Các giá trị này là candidate configurations; cần xác nhận checkpoint và inference backend hỗ trợ chúng chính xác.

### 2.2. Vì sao cần thay đổi block size?

Giả sử có hai trạng thái:

| Trạng thái  | DFlash dự đoán        | Kết quả giả định       |
| ----------- | --------------------- | ---------------------- |
| Dễ dự đoán  | 15 speculative tokens | Target chấp nhận 12    |
| Khó dự đoán | 15 speculative tokens | Target chỉ chấp nhận 2 |

Ở trường hợp thứ hai, phần lớn verification work dành cho các candidates không được sử dụng.

Nếu giảm block size xuống 4–8, ta có thể tiết kiệm verification work. Tuy nhiên, block ngắn cũng khiến số speculation rounds tăng lên.

Vì vậy, không nên chọn block size chỉ dựa vào acceptance rate.

### 2.3. Objective của B

Gọi $L_t$ là số speculative tokens liên tiếp được chấp nhận trước rejection.

Định nghĩa:

$$ p_i(s_t,B_t)=P(L_t\ge i\mid s_t,B_t) $$

Khi đó:

$$ \mathbb E[L_t] = \sum\_{i=1}^{\gamma_t}p_i(s_t,B_t) $$

Trong speculative decoding thông thường có correction/bonus token, số tokens mới được commit ở một round xấp xỉ $1+L_t$, bỏ qua EOS và giới hạn output.

Ta ước lượng:

$$ \boxed{ J(B,\gamma\mid s_t)= \frac{ \widehat T\_{\text{select}}(B) +\widehat T\_{\text{draft}}(B,\gamma) +\widehat T\_{\text{verify}}(N_t,\gamma) }{ 1+\sum\_{i=1}^{\gamma} \widehat p_i(s_t,B) } } $$

Trong đó $N_t$ là full target context length hiện tại.

Chọn action có estimated milliseconds per committed token nhỏ nhất, không phải action có acceptance length lớn nhất.

### 2.4. Controller lấy tín hiệu từ đâu?

Tôi sẽ ưu tiên các signals không cần thêm model forward pass:

| Signal                             | Nguồn                             | Ý nghĩa                              |
| ---------------------------------- | --------------------------------- | ------------------------------------ |
| Recent accepted length             | Verification history              | Đo drafting có đang hiệu quả không   |
| Prefix acceptance statistics       | Các rounds trước                  | Ước lượng lợi ích của block dài      |
| Retrieval concentration            | Similarity scores từ A            | Context relevance có tập trung không |
| Last available target confidence   | Logits đã tính trong verification | Proxy cho generation difficulty      |
| Measured draft/verify latency      | CUDA timing                       | Chi phí thực tế trên GPU             |
| Current context length, batch size | Serving runtime                   | Phản ánh system cost                 |

Có thể dùng exponential moving average (EMA) để cập nhật:

$$ \hat T\_{a,t} = (1-\alpha)\hat T\_{a,t-1} + \alpha T\_{a,t} $$

với action $a=(B,\gamma)$.

Tương tự, lưu acceptance statistics theo từng action. Với dữ liệu online ít, nên dùng smoothing hoặc chia sẻ thống kê giữa các block sizes để tránh estimates không ổn định.

Không cần train một neural controller ngay từ đầu. Một cost-aware rule-based policy hoặc contextual bandit nhẹ là đủ để kiểm chứng giả thuyết.

## 3. A và B kết hợp với nhau như thế nào?

Đây là phần quan trọng nhất của phương pháp đề xuất.

Ví dụ minh họa cho controller:

| Generation state           | Context budget $B_t$ | Speculative tokens $\gamma_t$ |
| -------------------------- | ------------------------ | --------------------------------- |
| Predictable continuation   | 1K                       | 15                                |
| Strong source match        | 2K                       | 11                                |
| Uncertain factual content  | 4K                       | 3                                 |
| Uncertain / poor retrieval | Full                     | 3                                 |

Đây chỉ là policy intuition. Controller thực tế không nên hard-code bảng này; nó phải chọn theo estimated latency và acceptance, vì đôi khi context nhỏ + block dài vẫn có thể là lựa chọn tệ.

## 4. Pseudocode của Joint Controller

```
# Conceptual algorithm# Assumes pretrained DFlash and exact target verifierCONTEXT_BUDGETS = [1024, 2048, 4096, "full"]DRAFT_LENGTHS = [3, 7, 11, 15]stats = OnlineCostAndAcceptanceStats()while not generation_finished:    state = collect_cached_signals(        last_target_features,        previous_acceptance,        retrieval_scores,        context_length,        batch_size,    )    # Joint action selection    actions = [        (budget, gamma)        for budget in CONTEXT_BUDGETS        for gamma in DRAFT_LENGTHS    ]    budget, gamma = min(        actions,        key=lambda a: stats.estimated_cost_per_token(            action=a, state=state        ),    )    # A: Select only the draft-side context.    draft_kv, positions = select_draft_context(        last_target_feature=state.last_feature,
```

`OnlineCostAndAcceptanceStats`, `dflash.draft` và `target.verify_exact` ở đây là interfaces thiết kế đề xuất, không phải APIs có sẵn của repository.

Trong thực tế, nên có warm-up calibration và safe fallback: khi controller thiếu dữ liệu hoặc predicted cost không tốt hơn baseline, quay về fixed DFlash configuration.

## 5. Những vấn đề kỹ thuật có thể khiến phương pháp thất bại

Đây là bốn điểm tôi sẽ kiểm chứng sớm nhất.

Thứ nhất — KV gather có thể đắt hơn lượng compute tiết kiệm. Với 10K context, việc chỉ chọn 2K KV không tự động tạo speedup. Nếu mỗi round phải gather/copy hàng chục MB, lợi ích attention có thể biến mất. Vì vậy, cần đo riêng `selection_ms`, `draft_ms` và GPU memory bandwidth. Prototype có thể dùng gather; phiên bản tối ưu nên cân nhắc block-indexed attention.

Thứ hai — DFlash đã được train với cách conditioning cụ thể. Đột ngột bỏ phần lớn target features có thể làm acceptance giảm mạnh. Đây là lý do phải thử fixed selection trước. Nếu training-free selection thất bại, một lightweight fine-tuning để làm drafter robust với sparse context có thể là phương án dự phòng, nhưng sẽ thay đổi phạm vi contribution.

Thứ ba — Dynamic block size phải thực sự thay đổi computation. Nếu runtime vẫn chạy CUDA graph cho block size 16 rồi chỉ sử dụng 4 candidates, chi phí drafting/verification có thể không giảm tương ứng. Cần hỗ trợ các bucketed shapes và đo overhead của kernel dispatch/CUDA graphs. Repository DFlash hiện đã có tích hợp vLLM và SGLang, nhưng joint dynamic context selection không phải tính năng có thể giả định là được hỗ trợ sẵn.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Thứ tư — Exact verification là điều kiện bắt buộc. Với greedy decoding, target phải xác nhận prefix đúng theo greedy target. Với sampling, cần acceptance/rejection correction bảo toàn target distribution, không chỉ so sánh token IDs. Đồng thời phải rollback rejected target KV và cập nhật draft features đúng vị trí. Nếu làm sai, phương pháp không còn lossless.

## 6. Thứ tự implement tôi đề xuất

Không nên implement A và B cùng lúc ngay từ đầu. Nên chia thành ba milestones có thể đo lường độc lập.

Milestone 1 — Fixed Selective Draft Context

Giữ $\gamma=15$, sweep $B\in\\{1K,2K,4K,\text{Full}\\}$. So sánh recent-only, retrieval, random và full context.

Go/no-go: Có giảm draft latency mà không mất quá nhiều accepted tokens không?

Milestone 2 — Adaptive Block Controller

Giữ full draft context, thay đổi $\gamma$ theo measured cost và acceptance statistics.

Go/no-go: Adaptive $\gamma$ có vượt được fixed $\gamma$ tốt nhất trên cùng workload không?

Milestone 3 — Joint A+B

Chọn $(B_t,\gamma_t)$ bằng online cost model, thêm source-grounding features và đo interaction giữa hai quyết định.

Go/no-go: Joint controller có vượt được A-only, B-only và best fixed pair không?

## 7. Paper contribution sẽ hình thành như thế nào?

Nếu ba milestones thành công, paper sẽ có một logic rất chặt:

- Finding 1: DFlash không cần cùng một lượng draft context ở mọi generation state.
- Finding 2: Optimal speculation length phụ thuộc không chỉ acceptance history mà còn phụ thuộc draft-context budget.
- Method: Joint cost-aware context-and-block adaptation.
- Result: Cải thiện end-to-end latency trên long-context summarization, giữ target full-context verification.

Tuy nhiên, Finding 1 và 2 hiện là giả thuyết cần đo, không phải kết quả đã được chứng minh.

Điểm tôi muốn nhấn mạnh nhất: A mới là nơi cần kiểm tra feasibility trước, còn B là phần giúp biến context selection thành một phương pháp adaptive hoàn chỉnh. Nếu A tạo được Pareto trade-off tốt giữa draft cost và acceptance, việc tối ưu phối hợp $B_t$ và $\gamma_t$ sẽ trở thành một research question có cơ sở thực nghiệm rất rõ.


Tôi đồng ý với cả hai ý của bạn. Đặc biệt, kết quả khảo sát cho thấy entropy càng cao thì xác suất rejection càng lớn là một cơ sở thực nghiệm tốt để thiết kế B — Adaptive Block Controller.

Tuy nhiên, tôi đề xuất một điều chỉnh quan trọng:

Dùng entropy để dự đoán khả năng acceptance, nhưng dùng latency cost để quyết định block size.

Và về thực nghiệm, nên đánh giá A và B độc lập trước, sau đó mới xây dựng C — Joint Controller. Đây cũng là cách tổ chức paper story thuyết phục hơn.

## 1. B — Entropy-Guided Adaptive Block Controller

### 1.1. Vì sao entropy phù hợp?

Với target model, tại decoding step $t$:

$$ H_t=-\sum\_{v\in V}p_T(v\mid x,y\_{\<t}) \log p_T(v\mid x,y\_{\<t}) $$

Entropy phản ánh mức độ phân tán của next-token distribution.

Theo quan sát thực nghiệm của bạn:

$$ H_t\uparrow \quad\Longrightarrow\quad P(\text{rejection})\uparrow $$

Ta có thể hình thành giả thuyết:

- Entropy thấp → target distribution tập trung → drafter dễ dự đoán đúng → nên thử block dài.
- Entropy cao → khả năng mismatch lớn hơn → nên thử block ngắn để tránh verification lãng phí.

Tuy nhiên, cần phân biệt correlation với quan hệ tất yếu. Rejection còn phụ thuộc vào mức độ phù hợp giữa draft distribution và target distribution. Nếu cả hai đều dự đoán tốt cùng một phân phối có entropy cao, acceptance vẫn có thể tốt.

Vì vậy, entropy là một predictive signal, không phải thước đo rejection hoàn hảo.

### 1.2. Thiết kế B phiên bản đầu tiên

Tôi đề xuất bắt đầu bằng một controller đơn giản, có thể giải thích được.

Target next-token logits

Tận dụng logits có sẵn từ prefill hoặc verification trước đó

Compute entropy $H_t$

Low entropy

Long block

Ví dụ γ = 15

Medium

Medium block

Ví dụ γ = 7

High entropy

Short block

Ví dụ γ = 3

DFlash draft → Target verification

Ghi nhận accepted length, draft latency, verification latency

Các giá trị $\gamma$ trên chỉ mang tính minh họa và phải nằm trong giới hạn mà checkpoint/runtime DFlash hỗ trợ.

Ngưỡng entropy không nên đặt tùy ý. Bạn có thể dùng quantiles trên một tập calibration tách biệt, hoặc ước lượng online bằng running statistics.

### 1.3. Một vấn đề rất quan trọng: entropy nào có thể dùng tại inference?

Bạn cần phân biệt hai loại entropy.

| Loại                                                              | Có thể dùng khi chọn block hiện tại? | Mục đích                            |
| ----------------------------------------------------------------- | ------------------------------------ | ----------------------------------- |
| Entropy tại các vị trí candidate, tính sau target verification    | Không, vì lúc chọn block chưa biết   | Phân tích offline entropy–rejection |
| Entropy của next-token distribution đã có ở đầu speculation round | Có, nếu runtime cung cấp logits      | Online block-size selection         |
| Entropy trung bình của các bước đã verify trước đó                | Có                                   | Predictive history signal           |

Nếu bạn đã khảo sát bằng entropy của chính token bị reject, correlation đó rất có giá trị, nhưng chưa chứng minh rằng entropy có sẵn trước speculation dự đoán tốt rejection của cả block.

Đây là một research question riêng:

> Can entropy available before a speculation round predict the accepted length of the next diffusion block?

Tôi nghĩ đây nên là thí nghiệm đầu tiên của B.

Ngoài ra, không nên thêm một target forward pass chỉ để tính entropy, vì chi phí này có thể triệt tiêu speedup. Nên lấy logits từ target verification đã thực hiện và đo cả overhead của phép tính entropy.

## 2. Nâng B từ Entropy Threshold lên Cost-Aware Controller

Entropy threshold là baseline tốt, nhưng chưa đủ mạnh để trở thành contribution chính.

Ví dụ, với entropy cao, block ngắn có thể giảm wasted verification. Tuy nhiên, nếu block ngắn khiến DFlash phải chạy quá nhiều rounds, tổng latency vẫn có thể tăng.

Vì vậy, controller cuối cùng nên chọn:

$$ \boxed{ \gamma_t^\* = \arg\min\_{\gamma\in\Gamma} \frac{ \widehat T_D(\gamma) + \widehat T_V(\gamma) }{ \widehat N\_{\mathrm{commit}}(\gamma\mid H_t,\mathcal H_t) } } $$

Trong đó:

- $H_t$: entropy hiện tại.
- $\mathcal H_t$: acceptance history của những rounds trước.
- $\widehat T_D,\widehat T_V$: estimated draft và verification latency.
- $\widehat N\_{\mathrm{commit}}$: số tokens dự kiến được commit sau một round.

Có thể bắt đầu với lookup table thống kê theo entropy buckets, không cần neural network.

Ví dụ:

```
# Conceptual training-free controllerentropy = get_available_target_entropy()history = recent_acceptance_statistics()best_gamma = Nonebest_cost = float("inf")for gamma in supported_block_sizes:    expected_committed = estimate_committed_tokens(        entropy, history, gamma    )    expected_time = (        estimated_draft_time(gamma)        + estimated_verify_time(gamma)    )    cost = expected_time / max(expected_committed, 1)    if cost < best_cost:        best_cost = cost        best_gamma = gamma
```

Đây là một phiên bản B có thể triển khai tương đối độc lập với A.

## 3. Thực nghiệm A và B riêng trước C: hoàn toàn hợp lý

Tôi còn cho rằng đây là thiết kế thực nghiệm nên dùng cho paper.

A — Selective Draft Context

Fixed block size

Optimize $B_t$

B — Entropy-Guided Block

Full draft context

Optimize $\gamma_t$

C — Joint Context & Block Controller

Optimize $(B_t,\gamma_t)$ cùng lúc

Entropy + source relevance + online latency

### Phase A — Selective Draft Context

Giữ block size cố định ở một cấu hình baseline đã được tuning, rồi thay đổi draft-context budget.

| Experiment | Context selection           | Block size |
| ---------- | --------------------------- | ---------- |
| A0         | Full context                | Fixed      |
| A1         | Recent-only                 | Fixed      |
| A2         | Random chunks               | Fixed      |
| A3         | Retrieval-based chunks      | Fixed      |
| A4         | Retrieval + recent + global | Fixed      |

Mục tiêu là trả lời:

Có thể giảm draft context mà vẫn giữ được acceptance đủ cao để giảm E2E latency không?

### Phase B — Entropy-Guided Adaptive Block

Giữ full draft context, chỉ thay đổi block size.

| Experiment | Controller                       | Context |
| ---------- | -------------------------------- | ------- |
| B0         | Best fixed $\gamma$          | Full    |
| B1         | Entropy threshold                | Full    |
| B2         | Acceptance-history only          | Full    |
| B3         | Entropy + acceptance history     | Full    |
| B4         | Entropy + history + latency cost | Full    |

Mục tiêu là trả lời:

Entropy có thực sự giúp chọn block size tốt hơn fixed block và acceptance-history-only không?

Đặc biệt, B1 so với B2 là ablation quan trọng. Nếu B3/B4 tốt hơn B2, bạn mới có bằng chứng rằng entropy mang lại thông tin bổ sung chứ không chỉ lặp lại tín hiệu acceptance history.

### Phase C — Joint Adaptive Controller

Sau khi A và B đã có kết quả độc lập, ta kết hợp:

$$ (B_t^\*,\gamma_t^\*) = \arg\min\_{B,\gamma} \widehat{\text{CostPerCommittedToken}} (B,\gamma\mid s_t) $$

Ở đây cần so sánh naive A+B (hai controller chạy độc lập) với joint A+B (quyết định phối hợp). Nếu joint không vượt naive combination, đóng góp về joint optimization sẽ yếu.

| Method               | Adaptive context | Adaptive block |
| -------------------- | ---------------- | -------------- |
| DFlash baseline      | Không            | Không          |
| A-only               | Có               | Không          |
| B-only               | Không            | Có             |
| A+B independent      | Có               | Có, độc lập    |
| C — Joint Controller | Có               | Có, phối hợp   |

Điểm cần chứng minh là context budget làm thay đổi optimal block size. Nếu hai quyết định không có interaction đáng kể, việc tối ưu phối hợp có thể không cần thiết.

## 4. Tôi đề xuất thêm một thí nghiệm trước khi triển khai B

Vì bạn đã có quan sát entropy–rejection, hãy biến nó thành một empirical finding có tính định lượng.

Cần đo hai quan hệ:

Quan hệ 1 — Token-level entropy và rejection

$$ P(\text{reject at position }i\mid H_i) $$

Đây là phân tích offline, có thể dùng entropy được tính trong verification. Nên phân tích theo vị trí trong block, vì token ở cuối block chỉ được xét nếu các tokens trước chưa bị reject.

Quan hệ 2 — Pre-round entropy và accepted length

$$ \mathbb E[L_t\mid H_t^{\mathrm{start}}] $$

Đây mới là quan hệ quyết định entropy có hữu ích cho online controller hay không.

Giả thuyết thực nghiệm cần kiểm chứng

0481216LowMedium-lowMediumMedium-highHigh

Đường cong minh họa giả thuyết, không phải dữ liệu thực nghiệm.

Nếu quan hệ này tồn tại đủ mạnh ở entropy có sẵn trước round, ta có cơ sở để xây B. Các giá trị trên chỉ minh họa xu hướng cần kiểm tra.

Nên đo Spearman correlation, acceptance length theo entropy quantiles và khả năng dự đoán rejection/accepted length trên held-out samples. Quan trọng hơn, phải chứng minh entropy-based policy thực sự giảm wall-clock latency, không chỉ cải thiện dự đoán rejection.

## 5. Paper story sau khi bổ sung entropy

Tôi sẽ cập nhật cách trình bày paper như sau:

Observation 1: Long-context drafting không cần cùng một lượng source context tại mọi generation state.

Observation 2: Target-model uncertainty, được ước lượng bằng entropy, có liên hệ với speculative acceptance và có thể giúp quyết định block length.

Observation 3 — cần kiểm chứng: Optimal speculation length phụ thuộc đồng thời vào uncertainty và lượng draft context được cung cấp.

Từ đó:

Method A: Source-aware selective draft context.

Method B: Entropy-guided, cost-aware adaptive diffusion block sizing.

Method C: Joint context-and-block optimization for long-context speculative generation.

Tôi đề xuất đặt tên working title:

ECA-Flash: Entropy- and Context-Adaptive Speculative Decoding for Long-Context Generation

Tên này nhấn mạnh hai tín hiệu chính, nhưng nên giữ là tên tạm cho đến khi chứng minh được lợi ích của cả hai.

### Kết luận và thứ tự tôi khuyên triển khai

Thứ tự về mặt paper: Baseline → A-only → B-only → C (joint).

Thứ tự về mặt kỹ thuật: Tôi sẽ làm B trước A, bởi bạn đã có quan sát entropy–rejection và B ít đụng đến kiến trúc KV injection của DFlash hơn. B có thể cho ra kết quả thực nghiệm đầu tiên nhanh hơn. Sau đó thực hiện A độc lập và cuối cùng tích hợp C.

Điểm quyết định tính thuyết phục của paper không chỉ là A và B đều tăng tốc, mà là chứng minh được:

$$ \boxed{ \text{Optimal block length} = f(\text{entropy},\text{draft context},\text{system cost}) } $$

Nếu dữ liệu ủng hộ quan hệ này, C sẽ là contribution trung tâm, còn A và B vừa là các module của phương pháp vừa là những ablation độc lập, tạo thành một paper story rất mạch lạc.
