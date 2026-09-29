# Efficient Inference for Long-Context Text Summarization: A Unified Literature Review

## 1. Bối cảnh và bài toán nghiên cứu

Large Language Models ngày càng được sử dụng cho long-document summarization, trong đó tài liệu đầu vào có thể dài hàng nghìn đến hàng trăm nghìn token, trong khi summary thường ngắn hơn đáng kể. Đây là một workload có profile hệ thống khác với chat, code generation hay long-chain reasoning:

$$
L_{\text{source}}\gg L_{\text{output}}.
$$

Do đó, latency của một request không chỉ đến từ autoregressive decoding mà còn từ việc đọc và biểu diễn tài liệu nguồn.

Một decomposition phù hợp là:

$$
T_{\mathrm{E2E}}
=
T_{\mathrm{queue}}
+
T_{\mathrm{preprocess}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{decode}}.
$$

Nếu có semantic selector:

$$
T_{\mathrm{preprocess}}
=
T_{\mathrm{selector}}.
$$

Nếu có speculative decoding:

$$
T_{\mathrm{decode}}
\approx
N_{\mathrm{verify}}
\left(
T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{controller}}
+
T_{\mathrm{system}}
\right).
$$

Trong đó:

- **TTFT — Time To First Token:** thời gian từ khi request tới khi target sinh token đầu tiên; chủ yếu phản ánh preprocessing + prefill.
- **TPOT — Time Per Output Token:** chi phí trung bình sinh mỗi token sau prefill.
- **E2E latency:** thời gian hoàn thành toàn bộ summary.
- **Throughput/QPS:** số request hoặc token hệ thống xử lý trong một đơn vị thời gian.
- **Goodput/QPS@SLO:** throughput đáp ứng latency constraint.
- **KV-cache footprint:** lượng memory cần giữ source/generated states.
- **Quality:** ROUGE, BERTScore và các semantic metrics.
- **Factuality:** SummaC, FactCC, entity/number preservation hoặc source-evidence consistency.

Điểm quan trọng là **peak decode speedup không đồng nghĩa với end-to-end summarization speedup**. Với long-input–short-output workloads, giảm prefill có thể quan trọng ngang hoặc quan trọng hơn giảm decoding. Đây là kết luận xuyên suốt của các review trước.

---

# 2. Hai track cơ bản: exact và approximate inference

Literature nên được phân chia trước hết theo guarantee thay vì chỉ theo vị trí optimization.

| Track | Định nghĩa | Ví dụ | Cách đánh giá |
|---|---|---|---|
| **Exact / lossless** | Không cố ý thay đổi target distribution; accelerated decoding được target xác minh đầy đủ | Standard SD, DistillSpec+SD, EAGLE, SSSD, LongSpec, SpecExtend | Latency, throughput, exactness, acceptance |
| **Approximate / quality-bounded** | Input, KV, weights hoặc computation bị bỏ/nén, do đó output có thể thay đổi | Semantic selection, SpecPrefill, SwiftKV, MInference, FastKV, RocketKV, quantization | Quality–latency–factuality Pareto |
| **Hybrid** | Một phần pipeline exact nhưng một phần làm thay đổi conditioning/computation | Retrieval + SD, selector + exact SD, QuantSpec variants | Phải chỉ rõ guarantee tại từng stage |

Với greedy decoding, strict lossless có thể viết:

$$
\hat{y}_{\mathrm{accelerated}}(x)
=
\hat{y}_{\mathrm{target}}(x).
$$

Với sampling:

$$
P_{\mathrm{accelerated}}(y|x)
=
P_{\mathrm{target}}(y|x).
$$

Do đó:

> “ROUGE không giảm” không phải bằng chứng lossless.

Đây là distinction đặc biệt quan trọng khi so speculative decoding với input pruning, KV compression hoặc semantic selection.

---

# 3. Taxonomy tổng thể

```text
Efficient Inference for Long-Context Summarization
│
├── A. Source / Input Reduction
│   ├── Semantic content selection
│   ├── Token / prompt compression
│   └── Hierarchical processing
│
├── B. Prefill Acceleration
│   ├── Exact attention kernels
│   ├── Sparse attention
│   └── Layer/model transformation
│
├── C. KV-Cache Optimization
│   ├── Eviction / sparse KV
│   ├── Cross-layer sharing
│   ├── Quantization
│   └── Factuality-aware KV preservation
│
├── D. Decode Acceleration
│   ├── Draft-model speculative decoding
│   ├── Feature/multi-head drafting
│   ├── Self-speculation
│   ├── Retrieval / n-gram drafting
│   └── Long-context speculative decoding
│
├── E. Model Compression
│   ├── Weight quantization
│   ├── KV quantization
│   └── Pruning / low-bit self-drafting
│
└── F. Adaptive / Joint / Serving Optimization
    ├── Adaptive speculation
    ├── Hardware/batch-aware control
    ├── Joint prefill–decode optimization
    └── Hybrid semantic + systems optimization
```

Một cách nhìn tổng quát hơn là mỗi hướng giảm một thành phần khác nhau:

$$
\begin{aligned}
\text{Semantic selection}:&\quad L_{\mathrm{source}}\downarrow\\
\text{Sparse prefill}:&\quad C_{\mathrm{prefill/token}}\downarrow\\
\text{KV optimization}:&\quad M_{\mathrm{KV}},BW_{\mathrm{KV}}\downarrow\\
\text{Speculative decoding}:&\quad N_{\mathrm{target\ decoding\ calls}}\downarrow\\
\text{Quantization}:&\quad BW_{\mathrm{weights/KV}}\downarrow.
\end{aligned}
$$

Vì vậy các hướng này phần lớn **không thay thế hoàn toàn lẫn nhau**.

---

# 4. Semantic Content Selection và Input Reduction

## 4.1. Định nghĩa

Semantic content selection lựa chọn các đơn vị có ý nghĩa trong tài liệu:

$$
\text{sentence},
\text{section},
\text{chunk},
\text{key point},
\text{proposition},
\text{evidence},
$$

sau đó chỉ đưa phần được chọn vào target:

$$
D
\xrightarrow{S}
D',
\qquad
|D'|\ll |D|,
$$

$$
Y\sim p_T(Y|D').
$$

Khác với sparse attention, ở đây target **không còn conditioning trên full source**:

$$
p_T(Y|D')
\neq
p_T(Y|D)
$$

nói chung.

Do đó semantic selection là **approximate acceleration**.

Objective phù hợp hơn là:

$$
\min_S
T_{\mathrm{selector}}
+
T_{\mathrm{target}}(S(D))
$$

subject to:

$$
Q(S(D))\ge Q(D)-\epsilon,
$$

$$
F(S(D))\ge F(D)-\delta.
$$

---

## 4.2. Cheap extractive selectors

### Lead-k

Chọn các sentence đầu tiên đến khi đạt token budget.

**Ưu điểm**

- gần như không có runtime overhead;
- rất mạnh trên dataset có lead bias;
- là baseline bắt buộc để chứng minh một semantic method thực sự cần thiết.

**Limitation**

- positional bias;
- kém trên scientific reports, meetings hoặc narrative, nơi facts phân tán;
- không kiểm soát redundancy.

---

### TF-IDF / centroid

Xếp sentence dựa trên lexical information density hoặc similarity với document centroid.

**Insight**

Một selector không nhất thiết cần LLM. Classical lexical signal có thể cung cấp relevance đủ tốt với overhead cực thấp.

**Limitation**

- synonym/paraphrase khó xử lý;
- lexical relevance không đồng nghĩa summary utility;
- ít model discourse/coverage.

---

### TextRank / LexRank

Xây graph sentence similarity và dùng graph centrality/PageRank.

Một implementation dense có complexity gần:

$$
O(n^2d)+O(I|E|).
$$

**Insight**

Centrality là tín hiệu mạnh hơn raw position cho structured documents.

**Limitation**

- pairwise graph cost tăng nhanh khi số sentence lớn;
- central sentence chưa chắc chứa rare-but-important facts;
- tail latency có thể trở thành vấn đề cho BookSum hoặc meeting transcript cực dài.

---

## 4.3. Lightweight semantic selection: Embedding + MMR

Một encoder nhỏ sinh sentence representations:

$$
e_i=E(s_i).
$$

Relevance:

$$
r_i=\cos(e_i,e_D).
$$

MMR chọn:

$$
s^*
=
\arg\max_{s_i\notin S}
\left[
\lambda r_i
-
(1-\lambda)
\max_{s_j\in S}
\cos(e_i,e_j)
\right].
$$

Nó trực tiếp cân bằng:

$$
\text{relevance}
-
\text{redundancy}.
$$

Budget-Aware Routing 2026 cung cấp evidence đáng chú ý: trong MIMIC setting với GPT-4o, sentence+MMR ở **512 selected tokens** đạt ROUGE-1 tương đương full context và BERTScore còn cao hơn trong reported experiment. Điều này không chứng minh universal speedup, nhưng chứng minh rằng trong một số regime:

$$
\text{less context}
\not\Rightarrow
\text{worse summary}.
$$



**Insight**

Embedding+MMR hiện là một trong những baseline có tỷ lệ tốt nhất giữa:

$$
\frac{\text{semantic quality}}
{\text{implementation/runtime complexity}}.
$$

**Limitation**

- vẫn cần embedding encoder;
- naive pairwise/greedy computation có thể gần $O(n^2)$;
- relevance với document centroid chưa trực tiếp model factual importance;
- paper hiện tại thường không tính selector latency trong E2E serving.

---

## 4.4. Coverage- và budget-aware selection

### RCD / Budget-Aware Routing

RCD kết hợp ba mục tiêu:

$$
\text{relevance}
+
\text{coverage}
+
\text{diversity}
$$

dưới explicit token budget.

Một insight quan trọng của Budget-Aware Routing là **optimal selector thay đổi theo budget**:

$$
\text{very small budget}
\rightarrow
\text{Lead/positional heuristic},
$$

$$
\text{moderate budget}
\rightarrow
\text{MMR},
$$

$$
\text{larger budget}
\rightarrow
\text{coverage-aware RCD}.
$$

Điều này gợi ý rằng **một selector cố định không tối ưu cho mọi request**.

**Limitation**

- graph/kernel operations có thể đắt hơn MMR;
- paper tập trung quality-vs-budget, không cung cấp full serving decomposition;
- policy được quan sát trên một số domain chưa chắc transfer thẳng sang GovReport, BookSum hoặc XSum.

---

### StrucSum-CGM

StrucSum dùng sentence embeddings và document graph để khai thác structural centrality, sau đó Content Graph Masking loại các sentence có centrality thấp.

Review trước ghi nhận phương pháp giảm khoảng **40–50% prompt tokens** trong nhiều setting trên ArXiv, PubMed và Multi-News trong khi quality cạnh tranh hoặc cải thiện.

**Insight**

Raw salience chưa đủ; document structure có thể cung cấp một prior phù hợp cho summarization hơn token-level importance.

**Limitation**

- $O(n^2)$-style semantic graph;
- chưa benchmark selector-inclusive TTFT/E2E;
- graph centrality vẫn có thể bỏ rare evidence không central.

---

## 4.5. Learned extract-then-abstract

### LoBART + MCS

Chọn sentence trước rồi đưa vào long-context BART.

**Contribution**

Cho thấy explicit content selection giúp long summarization từ trước thời LLM inference hiện đại.

**Limitation**

- supervised task-specific selector;
- stack cũ;
- không phản ánh vLLM/SGLang production serving hiện tại.

---

### REFLECT

Tối ưu extractor theo **credit đối với downstream abstractor**, thay vì extraction score đơn thuần.

Insight cốt lõi:

$$
\boxed{
\text{selector quality}
\neq
\text{extractive ROUGE}.
}
$$

Một sentence có vẻ salient chưa chắc giúp generator tạo summary tốt.

**Limitation**

- extractor/abstractor training phức tạp;
- domain shift;
- không drop-in cho off-the-shelf target LLM.

---

### DYLE

Selection là latent và thay đổi theo decoding position.

**Insight**

Static context selection có thể suboptimal vì information needs thay đổi khi summary đang được sinh.

Đây là tiền đề quan trọng cho **generation-aware or phase-aware semantic selection**.

**Limitation**

- joint training;
- architecture-specific;
- runtime complexity lớn hơn pre-filter một lần.

Các learned extract-then-abstract methods cung cấp insight mạnh về downstream utility, nhưng không phải lựa chọn đầu tiên nếu mục tiêu là **production acceleration với target LLM có sẵn**.

---

## 4.6. Evidence retrieval và graph-based selection

### Graph of Records / HiGoE

Các phương pháp này chuyển từ sentence similarity đơn giản sang:

$$
\text{proposition}
\leftrightarrow
\text{evidence}
\leftrightarrow
\text{theme}.
$$

HiGoE đặc biệt nhấn mạnh source-grounded evidence thay vì generated pseudo-evidence.

**Insight**

Đối với summarization, mục tiêu không chỉ là “salience” mà còn là:

$$
\text{coverage}
+
\text{evidence grounding}.
$$

**Limitation**

- graph construction/preprocessing đắt;
- đôi khi yêu cầu additional model inference;
- không hấp dẫn cho one-shot high-QPS requests nếu preprocessing không reuse được;
- ít systems evidence về E2E latency.

---

## 4.7. Planning và hierarchical summarization

Self-Planning/highlight methods tạo content plan trước khi target viết final summary.

Hierarchical methods:

$$
D
\rightarrow
\{D_1,\dots,D_m\}
\rightarrow
\{S_1,\dots,S_m\}
\rightarrow
S.
$$

Chúng đặc biệt hữu ích khi:

$$
L_{\mathrm{source}}
>
100K
$$

hoặc model không thể xử lý toàn source trong một request.

**Ưu điểm**

- mở rộng tới extreme context;
- có thể tăng coverage;
- intermediate evidence giúp giảm lost-in-the-middle.

**Limitation**

- nhiều LLM calls;
- E2E latency có thể tăng thay vì giảm;
- error accumulation/factual drift giữa levels;
- không phù hợp làm default acceleration ở context 8K–32K.

Một policy hợp lý hơn là:

$$
L<L_1
\rightarrow
\text{full context},
$$

$$
L_1\leq L<L_2
\rightarrow
\text{semantic selection},
$$

$$
L\geq L_2
\rightarrow
\text{hierarchical processing}.
$$

---

# 5. Lightweight Non-LLM Semantic Selection: hệ quả mới nhất

Supplementary review về lightweight selection bổ sung một insight systems rất quan trọng:

$$
\boxed{
T_{\mathrm{selector}}
<
T_{\mathrm{full-target}}
-
T_{\mathrm{selected-target}}
}
$$

là điều kiện tối thiểu để selection thực sự tăng tốc E2E.

Đây tưởng như hiển nhiên nhưng phần lớn semantic-selection literature **không đo đầy đủ đại lượng này**.

Các selector có thể chia theo cost thành:

| Loại | Ví dụ | Runtime characteristics |
|---|---|---|
| Pure algorithmic | Lead, TF-IDF, TextRank, LexRank | CPU, không neural inference |
| Light neural | MiniLM + MMR/RCD/graph | one-shot batched encoder |
| LLM-based | planning, causal scoring, abstractive compressor | expensive generation/model pass |

Điều này dẫn đến một design principle:

$$
\boxed{
\text{use the cheapest semantic signal sufficient for the regime}.
}
$$

Một engineering precedent từ vLLM Semantic Router cho thấy classical CPU compression có thể nằm ở mức vài–vài chục millisecond đến 16K tokens trong workload được paper đo, nhưng đây **không phải summarization evidence**. Nó chỉ cho thấy break-even condition về hệ thống là khả thi.

### Limitation chung của semantic-selection literature

Khoảng trống nổi bật nhất là thiếu measurement đồng thời:

$$
(
T_{\mathrm{selector}},
T_{\mathrm{prefill}},
T_{\mathrm{decode}},
T_{\mathrm{E2E}},
P50/P95/P99,
QPS
).
$$

Các paper thường chứng minh:

$$
\text{token reduction}
+
\text{quality},
$$

nhưng chưa chứng minh đầy đủ:

$$
\text{token reduction}
\Rightarrow
\text{production E2E speedup}.
$$

---

# 6. Prefill Acceleration

## 6.1. Exact attention kernels

### FlashAttention / FlashAttention-2

Thay vì thay đổi attention semantics, FlashAttention giảm HBM↔SRAM traffic bằng IO-aware tiling.

Đây là baseline hệ thống bắt buộc:

$$
\text{Dense AR + optimized attention}
$$

phải là denominator thay cho naïve Hugging Face attention.

**Ưu điểm**

- exact attention;
- broadly deployed;
- orthogonal với nhiều methods khác.

**Limitation**

- không giảm số attention interactions;
- khi context cực dài, quadratic computation vẫn tồn tại.

---

## 6.2. Dynamic sparse attention — MInference

MInference quan sát long-context attention thường rơi vào ba structured patterns:

- A-shape,
- Vertical-Slash,
- Block-Sparse.

Pattern được assign theo attention head; inference xây sparse indices động và chạy optimized kernels.

Paper báo cáo tới khoảng **10× prefill speedup ở 1M-token context** trong headline A100 setting.

**Insight**

Long-context attention không phải dense uniform computation; structure có thể được khai thác.

**Limitation đối với summarization**

- gains lớn nhất ở extreme context;
- chỉ giảm attention, trong khi MLP có thể dominate ở medium context;
- “maintained benchmark accuracy” không phải strict output equivalence;
- importance pattern cho QA/retrieval không đảm bảo global summary coverage.

---

## 6.3. Speculative Prefill

Một lightweight model ước lượng token importance rồi chỉ truyền selected tokens tới target.

Nếu keep rate là $r$:

$$
L
\rightarrow
rL.
$$

Paper báo cáo tới **7.66× TTFT** và **7× maximal QPS** trong favorable large-scale setting. Tuy nhiên summarization được ghi nhận là một trong những workload khó compress hơn khi keep rate giảm.

**Insight**

Không phải tất cả prompt tokens đều có computational value như nhau.

**Limitation**

- approximate;
- token-level pruning có thể phá discourse;
- entity/number/context dependency dễ bị mất;
- selector itself là lightweight model nên có overhead;
- global summarization khác retrieval: gần như mọi section đều có khả năng đóng góp.

---

## 6.4. GemFilter

Dùng early layers của target để xác định important tokens, tránh separate selector model.

**Insight**

$$
\boxed{
\text{reuse signals already paid for}
>
\text{add an expensive controller}
}
$$

là một design principle hấp dẫn.

**Limitation**

- vẫn là target-computation-dependent;
- early-layer token importance chưa chắc tương ứng summary evidence;
- output có thể thay đổi khi context bị lọc.

---

## 6.5. SwiftKV

SwiftKV nhắm trực tiếp vào long-prompt/short-generation workloads như summarization.

Later-layer KV được tạo từ representation của một layer sớm hơn, cho phép prompt tokens bỏ qua một phần later layers. Distillation nhẹ được dùng để recover knowledge.

Reported prefill compute reduction:

$$
25\%-50\%.
$$

Aggregate serving throughput có thể đạt khoảng **2×** trong reported configurations.

**Insight**

Attention không phải cost duy nhất. Ở medium-long context, MLP và layer-wide compute vẫn lớn; vì vậy skip complete layer work có thể tốt hơn chỉ sparsify attention.

**Limitation**

- cần model transformation;
- cần distillation;
- không plug-and-play cho arbitrary checkpoint;
- output không có strict exactness guarantee relative to original target.

---

# 7. KV-Cache Optimization

KV cache có memory complexity gần:

$$
O(L_{\mathrm{context}}\times N_{\mathrm{layers}}\times d_{\mathrm{KV}}).
$$

Khi context hoặc batch tăng, KV bandwidth/memory trở thành bottleneck.

---

## 7.1. KV eviction / pruning

Các dòng H2O, SnapKV, SCOPE, SpindleKV, RocketKV giữ một tập token KV được xem là quan trọng.

General idea:

$$
K,V
\rightarrow
K_S,V_S,
\qquad |S|\ll L.
$$

### RocketKV

Dùng multi-stage compression/sparse attention để giảm decode memory traffic.

**Insight**

KV bandwidth có thể quan trọng hơn arithmetic trong long decode.

**Limitation**

- token importance có thể thay đổi theo output position;
- aggressive compression có thể mất evidence;
- generic long-context score không đủ chứng minh summary factuality.

---

## 7.2. HalluKV và factuality

HalluKV là một paper quan trọng vì chỉ ra KV compression có thể tăng hallucination mạnh trong aggressive regimes; review ghi nhận mức tăng hallucination score tới khoảng **3.36×** trong reported analysis.

Điều này chuyển objective từ:

$$
\min T
$$

sang:

$$
\boxed{
\min T
\quad
\text{s.t.}
\quad
Q\ge Q_0-\epsilon,
\quad
F\ge F_0-\delta.
}
$$

Một insight đặc biệt quan trọng cho summarization:

$$
\boxed{
\text{source KV}
\neq
\text{generated-token KV}.
}
$$

Source KV chứa evidence grounding và có thể cần policy bảo vệ khác.

---

## 7.3. KV quantization — KVTuner và related methods

Giảm bitwidth của K/V theo layer/head.

**Ưu điểm**

- tăng effective batch;
- giảm memory bandwidth;
- dễ compose với techniques khác.

**Limitation**

- không strict lossless;
- speedup phụ thuộc kernel/hardware;
- low-bit error có thể tác động facts hiếm;
- không trực tiếp giải prefill compute.

---

# 8. Speculative Decoding

## 8.1. Định nghĩa

Draft model $q$ sinh $\gamma$ candidate tokens:

$$
x_{t+1:t+\gamma}\sim q.
$$

Target $p$ verify chúng trong một forward pass.

Nếu verification/correction chuẩn được sử dụng:

$$
P_{\mathrm{SD}}(Y|X)=P_T(Y|X).
$$

Hiệu quả phụ thuộc không chỉ acceptance:

$$
A=\text{accepted tokens},
$$

mà vào:

$$
\boxed{
\frac{\mathbb E[A]}
{T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{system}}}
}.
$$



---

## 8.2. Classical speculative decoding

Leviathan et al. là mốc nền tảng; survey lossless ghi nhận direct summarization evidence trên CNN/DailyMail với T5-XXL và speedup khoảng 3.1× greedy, 2.3× sampling trong specific batch-1 TPU setting.

**Strength**

- theoretical exactness;
- clean baseline.

**Limitation**

- cần draft-target compatible model;
- draft KV và weights thêm memory;
- long-context draft itself trở nên đắt.

---

## 8.3. DistillSpec

Thay vì train drafter để tối đa standalone LM accuracy, DistillSpec tối đa **draft-target alignment** tại states gặp trong inference.

Hai lessons:

1. on-policy data quan trọng;
2. optimal divergence phụ thuộc task và decoding strategy.

**Insight**

$$
\boxed{
\text{good draft LM}
\neq
\text{good speculative drafter}.
}
$$

Draft latency và alignment quan trọng hơn generic language-modeling score.

**Limitation**

- phải distill;
- training recipe task-sensitive;
- domain/model changes có thể cần adaptation.

---

## 8.4. EAGLE family

EAGLE dự đoán future hidden features; EAGLE-3 khai thác multi-layer target features và direct token prediction/tree verification.

Generic benchmark peak speedup rất cao, nhưng các review nhấn mạnh phần lớn evidence là chat/reasoning/code chứ không long-document summarization.

**Insight**

Target representations là rich side-information cho drafter.

**Limitation**

- training;
- extra parameters/model maintenance;
- generic acceptance không đảm bảo summarization gain;
- batch/context behavior cần benchmark riêng.

---

## 8.5. Training-free retrieval/n-gram speculation — SSSD

SSSD sử dụng:

- prompt n-grams;
- self-output history;
- external/live datastore;
- CPU lookup;
- hardware-aware speculation budget.

Paper báo cáo tới khoảng **2.9× latency reduction** trong favorable settings.

**Ưu điểm**

- không training;
- low draft compute;
- robust hơn learned drafter với language/domain shift;
- dễ deploy.

**Limitation đối với summarization**

Hiệu quả phụ thuộc lexical recurrence:

$$
\text{extractive summary}
\rightarrow
\text{high source-copy opportunity},
$$

$$
\text{abstractive summary}
\rightarrow
\text{low n-gram opportunity}.
$$

Do đó XSum-like tasks có thể là failure regime.

---

# 9. Adaptive Speculative Decoding

Static speculation window:

$$
\gamma=\mathrm{constant}
$$

không thể tối ưu mọi position/request/task.

On-the-Fly Adaptive SD điều chỉnh:

$$
\gamma_t
$$

và/hoặc chọn drafter dựa trên online statistics.

Một observation đặc biệt quan trọng từ summarization là trên XSum có regime trong đó speculation window nên giảm rất nhỏ, thậm chí:

$$
\boxed{\gamma=0}.
$$



Do đó:

$$
\boxed{\text{no speculation}}
$$

phải là một first-class action của controller.

Một runtime policy tổng quát có thể là:

$$
a_t,\gamma_t
=
\pi(
L_{\mathrm{source}},
L_{\mathrm{generated}},
B,
H,
\text{acceptance history},
\text{source overlap},
\text{confidence}
).
$$

**Limitation**

- controller overhead;
- online adaptation cần phản ứng nhanh;
- policy có thể overfit hardware;
- acceptance maximization không nhất thiết maximize E2E speedup.

---

# 10. Long-Context Speculative Decoding

Generic SD degrade khi source dài vì:

1. draft KV cache lớn;
2. draft model được train ở short context;
3. target verification trở nên đắt;
4. tree attention không map tốt lên optimized kernels.

---

## 10.1. LongSpec

LongSpec giải quyết ba vấn đề:

- constant-sized draft KV;
- Anchor-Offset positional scheme;
- Hybrid Tree Attention.

Trên long summarization datasets như GovReport, QMSum và Multi-News, review tổng hợp mức speedup khoảng **1.5–2.7× tùy target/dataset/configuration**, trong khi headline overall của paper cao hơn ở các workload khác.

**Insight**

Long-context SD không thể chỉ lấy short-context EAGLE rồi tăng context window.

**Limitation**

- draft architecture/training mới;
- implementation custom;
- target verification vẫn tăng chi phí theo context.

---

## 10.2. SpecExtend

SpecExtend tập trung vào moderate-to-long input regime, nơi SD đã degrade trước khi KV cache hoàn toàn trở thành memory bottleneck.

Cross-model Retrieval dùng **target attention scores** để giữ relevant source chunks trong draft KV cache.

Reported acceleration tới **2.84× trên 16K long-document summarization** trong paper.

Đây là baseline đặc biệt quan trọng vì nối ba ý tưởng:

$$
\boxed{
\text{source relevance}
+
\text{target-derived feedback}
+
\text{lossless target verification}
}
$$

**Limitation**

- target vẫn cần full context;
- attention-based relevance không đồng nghĩa semantic coverage;
- source selection chỉ giảm drafter context, không giảm full target prefill.

---

## 10.3. RAPID

Retrieval tạo context rút gọn cho drafter:

$$
D
\xrightarrow{\text{retrieve}}
D_{\mathrm{draft}}
$$

trong khi target vẫn kiểm soát output.

**Insight**

Source retrieval có thể dùng không chỉ để improve answer quality mà còn như **computational primitive cho drafting**.

**Limitation**

- hybrid variants có quality modifications;
- retrieval overhead;
- không giải full-target prefill cost;
- retrieved context có thể miss globally distributed summary evidence.

---

## 10.4. MagicDec và batch/context interaction

Conventional wisdom:

$$
B\uparrow
\Rightarrow
\text{SD speedup}\downarrow
$$

không phải luôn đúng.

Trong long-context KV-bound regimes, MagicDec chỉ ra speculative techniques có thể trở nên favorable ở large batch vì KV movement trở thành bottleneck.

Điều này tạo một **phase diagram**:

$$
S
=
S(
B,
L_{\mathrm{context}},
L_{\mathrm{output}},
M,
H
).
$$

**Insight**

Không tồn tại một method ranking cố định cho mọi batch/context.

---

# 11. Verification là bottleneck mới

Production-oriented study *Speculative Decoding: Performance or Illusion?* cho thấy:

- target verification thường chiếm phần lớn runtime;
- acceptance biến thiên theo token position;
- biến thiên mạnh theo request/dataset;
- measured speedup cách xa theoretical upper bound;
- các proposer khác nhau thắng ở những position khác nhau.

Review xem đây là methodology evidence quan trọng hơn một competitor trực tiếp.

Điều này dẫn tới hai conclusions.

Thứ nhất:

$$
\boxed{
\text{maximize acceptance rate}
\neq
\text{maximize speedup}.
}
$$

Thứ hai, proposer complementarity tạo motivation cho:

$$
\boxed{
\text{source proposer}
+
\text{neural proposer}
+
\text{no-spec}.
}
$$

Một controller có thể chọn proposer theo token/phase thay vì dùng một drafter duy nhất.

---

# 12. Joint Prefill–Decode Optimization

## FastKV

FastKV tách hai budget:

$$
k_{\mathrm{prefill}}
\neq
k_{\mathrm{KV}}.
$$

Một token cần thiết để xây representation trong prefill chưa chắc cần được giữ/read lại ở mọi decode step. Review trước ghi nhận paper báo tới khoảng **1.82× prefill** và **2.87× decode** trong reported settings.

**Insight quan trọng cho summarization**

Có thể cần đọc phần lớn source để hiểu global structure:

$$
D\rightarrow H,
$$

nhưng decode chỉ cần một subset source anchors để grounding:

$$
H_{\mathrm{source}}
\rightarrow
H_{\mathrm{anchor}}.
$$

Đây là bridge tự nhiên giữa semantic selection và KV policy.

**Limitation**

- approximate;
- internal token importance không trực tiếp đảm bảo semantic evidence coverage;
- interaction với external source selection chưa được khảo sát đầy đủ.

---

# 13. Model Compression và Quantization

## AWQ/GPTQ

Weight quantization:

$$
W_{\mathrm{FP16}}
\rightarrow
W_{\mathrm{INT4}}.
$$

**Strength**

- giảm VRAM;
- tăng batch capacity;
- production practical.

**Limitation**

- không summarization-specific;
- latency gain phụ thuộc hardware/kernel;
- có thể thay đổi output;
- không giải quyết input-length scaling.

---

## KV quantization

Các phương pháp như KVTuner chọn bitwidth theo layer/head.

Đây là optimization orthogonal:

$$
\text{semantic selection}
+
\text{weight quantization}
+
\text{KV quantization}
$$

có thể cùng tồn tại.

Tuy nhiên quantization nên được xem là **deployment control**, không phải central methodological baseline nếu research question tập trung source-aware summarization.

---

# 14. So sánh các hướng

| Hướng | Bottleneck giảm | Exact? | Lợi thế lớn nhất | Limitation chính |
|---|---|---:|---|---|
| Dense FA | Kernel/I/O | Có | baseline production mạnh | không giảm asymptotic work |
| MInference | Prefill attention | Không strict | cực mạnh ở extreme context | coverage/MLP/context dependence |
| SwiftKV | Prefill layers | Không | long-prompt/short-output | transformation + distillation |
| SpecPrefill | Input/prefill | Không | TTFT/QPS gain lớn | mất source information |
| Cheap semantic selection | Input/prefill/KV | Không | task-aware, CPU-friendly | selector coverage + E2E chưa được chứng minh |
| RCD/StrucSum | Input + coverage | Không | semantic structure | $O(n^2)$, serving evidence thiếu |
| RocketKV/FastKV | KV/decode | Không | memory/throughput | factuality risk |
| Vanilla SD | Decode | Có | theoretical guarantee | draft overhead |
| EAGLE | Decode | Có với full verification | high acceptance | train + deployment complexity |
| SSSD | Decode | Có | training-free, cheap | phụ thuộc source/output recurrence |
| LongSpec | Long-context decode | Có | purpose-built long SD | custom training/architecture |
| SpecExtend | Long-context draft | Có | target-guided source selection | target prefill vẫn full |
| Adaptive SD | Decode/controller | Có | tránh bad speculation regimes | controller complexity |
| Hierarchical | Extreme input | Không | >100K scalability | many LLM calls, drift |
| Quantization | Weight/KV BW | Không strict | deployment/memory | hardware-dependent |

Không nên xếp các headline speedup trong bảng thành ranking trực tiếp. Ví dụ:

$$
7.66\times\ \mathrm{TTFT}
$$

của SpecPrefill và

$$
2.84\times
$$

của SpecExtend giảm **hai thành phần khác nhau của pipeline**, trên hardware/model khác nhau.

---

# 15. Các insight xuyên suốt literature

## Insight 1 — Summarization là long-input–short-output workload

Do:

$$
L_{\mathrm{source}}\gg L_{\mathrm{output}},
$$

chỉ tối ưu decoding có nguy cơ bỏ qua bottleneck lớn nhất.

Do đó central target phải là:

$$
\boxed{
\text{End-to-End Summarization Inference}
}
$$

chứ không chỉ “faster decoding”.

---

## Insight 2 — Source không chỉ là dữ liệu; source là computational resource

Literature cho thấy source có thể được dùng ở nhiều vai trò:

$$
\text{Source}
\rightarrow
\begin{cases}
\text{semantic selection}\\
\text{n-gram drafting}\\
\text{retrieval drafting}\\
\text{target-guided KV selection}\\
\text{factual grounding}
\end{cases}
$$

SSSD, RAPID, SpecExtend, semantic selectors và SpecPrefill đều khai thác cùng một fact theo những cách khác nhau.

---

## Insight 3 — “Less context” đôi khi tốt hơn

Noise, redundancy và positional bias có nghĩa:

$$
Q(S(D))
$$

có thể bằng hoặc thậm chí cao hơn:

$$
Q(D).
$$

Tuy nhiên result này phụ thuộc dataset và retention budget. Không thể generalize từ MIMIC hay Multi-News sang BookSum/XSum mà không kiểm chứng.

---

## Insight 4 — Coverage quan trọng hơn raw salience

Summarization khác retrieval.

Một QA query có thể cần một local span; summary cần:

$$
\text{global topical coverage}.
$$

Do đó attention top-k/token importance đơn giản có thể kém một sentence-level selector có diversity/coverage constraint.

---

## Insight 5 — Selector phải rẻ hơn computation nó tiết kiệm

Điều kiện thực sự là:

$$
T_{\mathrm{selector}}
<
T_{\mathrm{saved}}.
$$

Đây là điểm mà nhiều semantic-selection papers hiện chưa chứng minh.

---

## Insight 6 — Không có universal speculation policy

Optimal policy phụ thuộc:

$$
(\text{task},L,B,H,\text{output phase}).
$$

“No speculation” phải là lựa chọn hợp lệ.

---

## Insight 7 — Verification chứ không chỉ drafting là bottleneck

Improving drafter thêm nữa có diminishing return nếu:

$$
T_{\mathrm{verify}}
\gg
T_{\mathrm{draft}}.
$$

Future SD cần tối ưu accepted useful work trên total system cost.

---

## Insight 8 — Reuse target-derived signals

Target đã tạo:

- attention statistics,
- confidence,
- accepted/rejected history,
- KV activity.

Các signals này có thể tái sử dụng cho routing hoặc context selection thay vì chạy thêm một large controller.

---

## Insight 9 — Factuality phải là constraint

ROUGE/BERTScore có thể không phát hiện mất rare numbers/entities.

Do đó:

$$
\Delta Q\approx0
$$

không đủ nếu:

$$
\Delta F\gg0.
$$

HalluKV là evidence rõ nhất cho problem này.

---

## Insight 10 — Batch size × context length tạo phase diagram

Performance nên được mô tả bằng:

$$
S(B,L,\mathrm{task},H)
$$

thay vì một “2.3× average speedup”.

Đây là đặc biệt quan trọng khi chuyển từ prototype batch 1 sang continuous batching.

---

# 16. Limitations của literature hiện tại

### Thiếu true E2E evaluation

Nhiều paper chỉ report:

- prefill speedup;
- decode tokens/s;
- token reduction;
- acceptance rate.

Nhưng production question là:

$$
\boxed{
T_{\mathrm{E2E,new}}
<
T_{\mathrm{E2E,baseline}}?
}
$$

---

### Thiếu standardized production baseline

Speedup so với naïve Hugging Face có thể lớn hơn đáng kể so với speedup so với:

$$
\text{vLLM/SGLang + FlashAttention/FlashInfer}.
$$

---

### Summarization-specific evidence còn mỏng

Phần lớn generic SD research đánh giá code, dialogue, reasoning.

Direct long summarization evidence tập trung ở tương đối ít work như LongSpec và SpecExtend.

---

### Batch/continuous serving chưa được nghiên cứu đủ

Batch 1 không phản ánh production.

Cần:

$$
B\in\{1,4,8,16,32,64\}
$$

hoặc request-rate based continuous batching.

---

### Semantic selection thiếu selector-inclusive timing

Đây là gap nổi bật nhất của recent semantic-selection literature.

---

### Quality metrics chưa đủ

ROUGE/BERTScore không kiểm tra đầy đủ:

- entity omission;
- number corruption;
- unsupported facts;
- evidence coverage;
- section coverage.

---

### Hardware khác nhau làm headline speedups không so sánh được

A100, H100, H200, TPU, tensor parallelism và engine khác nhau tạo denominator khác nhau.

---

# 17. Research gap tổng hợp

Từ toàn bộ literature, gap lớn nhất không phải “thiếu thêm một speculative drafter”.

Khoảng trống rộng hơn là:

$$
\boxed{
\textbf{How should compute be allocated across source processing and generation}
}
$$

để đạt:

$$
\boxed{
\text{minimum E2E latency}
}
$$

dưới:

$$
\boxed{
\text{quality + factuality constraints}.
}
$$

Một formulation tổng quát là:

$$
\min_{\pi,S,\gamma,K}
T_{\mathrm{selector}}
+
T_{\mathrm{prefill}}
+
T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{decode}}
$$

subject to:

$$
Q\ge Q_{\mathrm{full}}-\epsilon,
$$

$$
F\ge F_{\mathrm{full}}-\delta,
$$

$$
P95\le \mathrm{SLO}.
$$

Trong đó:

- $S$: source context policy;
- $\gamma$: speculation budget;
- $K$: KV budget;
- $\pi$: runtime controller.

---

# 18. Hướng nghiên cứu nổi bật nhất từ meta-review

Từ các nhánh trên, một research identity hợp lý là:

$$
\boxed{
\textbf{Source-Aware Adaptive Inference for Long-Document Summarization}
}
$$

với hai level.

### Level 1 — Lightweight semantic compute allocation

Một rule/lightweight controller chọn:

$$
a
\in
\{
\text{Full},
\text{Lead},
\text{MMR},
\text{RCD},
\text{Structure}
\}
$$

và retention:

$$
r\in(0,1].
$$

Policy dựa trên:

$$
r^*
=
f(
L_{\mathrm{source}},
\text{redundancy},
\text{document structure},
\text{budget},
B,H
).
$$

Mục tiêu không phải phát minh selector phức tạp nhất mà tìm **minimum semantic context đủ để target summarizer hoạt động tốt**.

---

### Level 2 — Compose với internal inference acceleration

Sau semantic selection:

$$
D
\xrightarrow{S}
D'
$$

có thể dùng:

$$
D'
\xrightarrow{\text{SwiftKV/FastKV}}
\text{efficient prefill/KV}
$$

và:

$$
\xrightarrow{\text{SSSD/SpecExtend}}
\text{efficient decode}.
$$

Một pipeline có thể là:

```text
Long Document
      ↓
Cheap document statistics
      ↓
Budget / regime controller
      ↓
Full / Lead / MMR / RCD selection
      ↓
Restore original document order
      ↓
Target prefill
      ↓
Optional SwiftKV / FastKV
      ↓
Adaptive source/neural/no-spec decoding
      ↓
Summary
```

Điểm khoa học thú vị nằm ở **interaction**, vì gains có thể không additive.

Ví dụ selector giảm source từ:

$$
L\rightarrow0.4L,
$$

khi đó headroom cho SwiftKV/MInference có thể giảm.

Ngược lại, selector có thể giảm KV footprint đủ để tăng batch capacity, tạo throughput gain lớn hơn latency gain.

---

# 19. Hypothesis trung tâm đáng kiểm chứng

Một hypothesis rất rõ và falsifiable là:

$$
\boxed{
\begin{aligned}
&\exists S,\ r<1:\\
&Q(S_r(D))
\ge
Q(D)-\epsilon,\\
&F(S_r(D))
\ge
F(D)-\delta,\\
&T_{\mathrm{selector}}
+
T_{\mathrm{target}}(S_r(D))
<
T_{\mathrm{target}}(D).
\end{aligned}
}
$$

Nếu đúng across dataset/model/load regimes, contribution không còn đơn giản là “semantic summarization”.

Nó trở thành:

$$
\boxed{
\textbf{quality-constrained semantic compute allocation}.
}
$$

---

# 20. Benchmark framework phù hợp với literature gap

Một evaluation đầy đủ nên dùng tối thiểu các regimes:

| Dataset | Vai trò |
|---|---|
| CNN/DailyMail | relatively extractive control |
| XSum | highly abstractive / low-copy failure regime |
| GovReport | long structured reports |
| Multi-News | redundancy + multi-document coverage |
| QMSum | sparse/query-focused meeting evidence |
| ArXiv | long structured technical document |
| PubMed | factual/entity-sensitive scientific text |
| BookSum | distributed narrative information |

Core experimental set có thể rút xuống:

$$
\boxed{
\text{GovReport}
+
\text{Multi-News}
+
\text{QMSum}
+
\text{ArXiv}.
}
$$

### Systems metrics

$$
T_{\mathrm{selector}},
T_{\mathrm{prefill}},
TTFT,
TPOT,
T_{\mathrm{E2E}},
P50,
P95,
P99,
QPS,
\text{VRAM}.
$$

### Quality metrics

$$
ROUGE,
BERTScore,
SummaC,
FactCC,
EntityRecall,
NumberRecall,
EvidenceCoverage.
$$

### Context regimes

$$
L
\in
\{
2K,4K,8K,16K,32K,64K+
\}.
$$

### Retention regimes

$$
r
\in
\{
0.2,0.3,0.5,0.7,1.0
\}.
$$

### Batch regimes

$$
B
\in
\{
1,8,32,64
\}
$$

kèm continuous batching/request-rate experiment.

---

# 21. Baseline taxonomy đề xuất

Một final comparison không cần reproduce mọi paper trong literature.

### Dense production denominator

$$
\boxed{
\text{AR + FlashAttention + vLLM/SGLang}
}
$$

### Lightweight source-selection family

$$
\boxed{
\text{Lead}
\rightarrow
\text{TextRank/TF-IDF}
\rightarrow
\text{Embedding+MMR}
\rightarrow
\text{RCD}
\rightarrow
\text{StrucSum-CGM}
}
$$

### Aggressive approximate systems baselines

$$
\boxed{
\text{SpecPrefill},
\text{SwiftKV},
\text{FastKV}
}
$$

### Lossless decode baselines

$$
\boxed{
\text{Vanilla SD},
\text{EAGLE-3},
\text{SSSD},
\text{Adaptive SD}
}
$$

### Long-context controls

$$
\boxed{
\text{LongSpec},
\text{SpecExtend},
\text{MagicDec},
\text{RAPID}
}
$$

### Factuality control

$$
\boxed{
\text{HalluKV-inspired source/KV preservation}
}
$$

Cấu trúc baseline này cho phép trả lời tách biệt:

1. liệu source reduction có đáng làm hay không;
2. liệu semantic selection tốt hơn token-level pruning hay không;
3. liệu decode-only acceleration đủ hay không;
4. liệu các hướng có complementary hay overlap;
5. factuality bị ảnh hưởng ở đâu.

---

# 22. Kết luận

Literature về LLM inference acceleration đã phát triển nhanh, nhưng **efficient inference for long-document summarization chưa phải một bài toán đã được giải quyết**.

Speculative decoding giải quyết autoregressive dependency nhưng không trực tiếp giảm expensive long-source prefill. Sparse attention và SwiftKV giảm source processing nhưng không khai thác mục tiêu semantic riêng của summarization. KV compression giảm memory/bandwidth nhưng có factuality risk. Semantic selection trực tiếp giảm lượng source computation, nhưng literature hiện tại thiếu bằng chứng selector-inclusive E2E serving. Hierarchical summarization xử lý extreme context nhưng thêm nhiều inference stages. Các adaptive systems cho thấy không có một configuration tối ưu cho mọi request.

Bức tranh chung vì vậy không chỉ ra một “winner” đơn lẻ. Nó chỉ ra một nguyên lý rộng hơn:

$$
\boxed{
\textbf{Efficient summarization requires allocating computation according to semantic utility and runtime regime.}
}
$$

Hướng nghiên cứu có cơ sở nhất từ các survey là kết hợp:

$$
\boxed{
\text{lightweight semantic selection}
+
\text{coverage/factuality constraints}
+
\text{runtime adaptation}
+
\text{efficient prefill/KV}
+
\text{adaptive decoding}.
}
$$

Research gap cụ thể nhất trước mắt là xác định **quality–factuality–throughput frontier** và **break-even surface** cho source reduction:

$$
r^*
=
f(
L_{\mathrm{source}},
\text{dataset},
\text{redundancy},
\text{abstraction},
B,
H,
M
),
$$

sao cho:

$$
r<r^*
\Rightarrow
\text{information/factuality loss},
$$

trong khi:

$$
r>r^*
\Rightarrow
\text{unnecessary computation}.
$$

Nếu một lightweight policy có thể dự đoán gần $r^*$, đồng thời quyết định khi nào nên dùng full context, semantic selection, speculative decoding hay không speculate, thì đó sẽ là bước chuyển từ một collection các inference tricks sang một **task-aware adaptive inference framework dành riêng cho long-document summarization**.