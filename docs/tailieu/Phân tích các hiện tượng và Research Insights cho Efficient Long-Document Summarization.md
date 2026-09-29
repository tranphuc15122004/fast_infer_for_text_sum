# Phân tích các hiện tượng và Research Insights cho Efficient Long-Document Summarization

## 1. Mục tiêu của báo cáo

Báo cáo này không nhằm liệt kê tuần tự các paper về efficient inference hay speculative decoding. Mục tiêu là tổng hợp các **behavioral phenomena**, **systems phenomena** và **structural properties của long-document summarization** xuất hiện lặp lại trong literature, từ đó tìm ra các nguyên lý có thể dẫn đến một phương pháp mới.

Toàn bộ evidence hiện tại cho thấy một pattern khá nhất quán:

$$
\boxed{
\text{Utility của computation trong LLM inference không đồng đều.}
}
$$

Sự không đồng đều này tồn tại trên nhiều chiều:

$$
\begin{aligned}
&\text{decoding timestep},\\
&\text{source position},\\
&\text{attention head},\\
&\text{transformer layer},\\
&\text{proposer},\\
&\text{speculation length},\\
&\text{context length},\\
&\text{batch size},\\
&\text{hardware/runtime regime}.
\end{aligned}
$$

Long-document summarization đặc biệt thú vị vì tất cả các dạng heterogeneity trên cùng tồn tại trong một workload:

$$
L_{\mathrm{source}}\gg L_{\mathrm{output}},
$$

trong khi output vừa cần **grounding vào source**, vừa cần **abstraction**, vừa chứa các đoạn rất predictable.

Do đó câu hỏi trung tâm dần chuyển từ:

> Làm thế nào để xây một drafter mạnh hơn?

sang:

> **Ở mỗi trạng thái inference, computation nào đáng thực hiện, source context nào cần sử dụng, proposer nào phù hợp và bao nhiêu speculative tokens nên được verify?**

---

# 2. Quy ước mức độ bằng chứng

Để tránh biến những suy luận hấp dẫn thành các claim quá mạnh, báo cáo phân biệt ba loại kết luận.

### Established evidence

Hiện tượng được paper trực tiếp đo, phân tích hoặc chứng minh.

### Cross-paper synthesis

Không có một paper duy nhất chứng minh toàn bộ kết luận, nhưng nhiều kết quả độc lập cùng hỗ trợ một interpretation chung.

### Open hypothesis

Ý tưởng hợp lý từ evidence hiện tại nhưng chưa được kiểm chứng trực tiếp. Đây thường là nơi research contribution có thể xuất hiện.

Việc phân biệt ba mức này rất quan trọng cho hướng nghiên cứu hiện tại.

---

# 3. Token difficulty không đồng đều

## 3.1. Hiện tượng

LLM không gặp mức độ khó giống nhau tại mọi decoding timestep.

Với target distribution:

$$
p_T(v\mid X,y_{<t}),
$$

ta có entropy:

$$
H_t
=
-\sum_v
p_T(v\mid X,y_{<t})
\log p_T(v\mid X,y_{<t}).
$$

Nếu:

$$
H_t\approx0,
$$

target gần như chắc chắn token tiếp theo.

Nếu:

$$
H_t\gg0,
$$

nhiều continuation cạnh tranh với nhau.

Ví dụ:

> The report found **that the program reduced costs by 17.3%**.

Các token function-word như:

`the`, `that`, `by`

thường dễ predict hơn nhiều lexical decisions.

Tuy nhiên cần sửa một interpretation quan trọng:

$$
\boxed{
\text{high entropy}
\neq
\text{high information importance}.
}
$$

Một token `"17.3"` có thể cực kỳ quan trọng nhưng entropy thấp nếu source làm continuation gần như deterministic.

Ngược lại, một discourse transition không chứa fact quan trọng vẫn có thể có nhiều cách diễn đạt:

> therefore / consequently / as a result / hence

và entropy cao.

## 3.2. Quan hệ với speculative decoding

Một drafter $D$ cố approximate:

$$
q_D(v\mid X,y_{<t})
\approx
p_T(v\mid X,y_{<t}).
$$

Khi target distribution sharp:

$$
H_t\downarrow,
$$

cả small model và retrieval-based proposal có khả năng đoán đúng cao hơn.

Khi:

$$
H_t\uparrow,
$$

target–draft divergence có xu hướng tăng.

EDSD 2026 khai thác trực tiếp relation giữa target entropy và speculative difficulty.

### Research implication

Entropy không nên được xem là grounding signal.

Thay vào đó cần tách ít nhất:

$$
\boxed{
\text{difficulty}_t
}
$$

và:

$$
\boxed{
\text{grounding}_t.
}
$$

Điều này tạo bốn trường hợp:

| Grounding | Entropy | Interpretation |
|---|---|---|
| cao | thấp | source-copyable/easy |
| cao | cao | grounded nhưng cần abstraction/synthesis |
| thấp | thấp | generic predictable continuation |
| thấp | cao | uncertain/free generation |

Các region này có thể tương ứng với proposer khác nhau.

**Mức bằng chứng:** Established về entropy–difficulty; hypothesis về mapping sang proposer.

---

# 4. Batch size không làm acceptance tự nhiên giảm — nó thay đổi giá trị kinh tế của speculation

Trong báo cáo gốc, tiêu đề “batch size làm acceptance length giảm mạnh” cần được sửa.

Vấn đề cốt lõi không nhất thiết nằm ở $L_{\mathrm{acc}}$, mà ở **cost của verification**.

## 4.1. Batch nhỏ

Autoregressive decoding của một large model thường memory-bandwidth bound.

Trong mỗi token:

1. weights được đọc từ HBM;
2. chỉ một số lượng FLOPs tương đối nhỏ được thực hiện;
3. GPU compute units chưa được sử dụng hết.

Ta có thể hình dung arithmetic intensity thấp:

$$
I_{\mathrm{AR}}
=
\frac{\mathrm{FLOPs}}
{\mathrm{bytes\ moved}}.
$$

Speculative verification biến target call từ:

$$
1\text{ token}
$$

thành:

$$
k\text{ candidate tokens}
$$

được process song song.

Weight loading được amortize tốt hơn và GPU utilization tăng.

## 4.2. Batch lớn

Khi batch tăng:

$$
B\uparrow,
$$

AR baseline tự nó đã có nhiều independent sequences để process song song.

Do đó GPU dần chuyển:

$$
\text{memory-bound}
\rightarrow
\text{compute-bound}.
$$

Trong regime này, verify thêm speculative tokens không còn gần như “free”.

Rejected token giờ tạo real FLOPs:

$$
C_{\mathrm{waste}}
\approx
N_{\mathrm{rejected}}
C_{\mathrm{verify/token}}.
$$

*Performance or Illusion?* cho thấy speedup của nhiều SD methods giảm rõ theo batch trong production-grade vLLM. Đây là một trong những warning quan trọng nhất đối với các headline speedup đo ở batch 1.

## 4.3. Nhưng batch lớn không luôn xấu cho SD

MagicDec cho một seemingly conflicting result.

Ở:

$$
L_{\mathrm{ctx}}\gg0
$$

và large batch, decode có thể bị chi phối bởi **KV-cache loading**, không chỉ model weights.

Một drafter với sparse KV có thể cheap hơn target rất nhiều, khiến SD lại hấp dẫn.

Do đó không tồn tại rule:

$$
B\uparrow
\Rightarrow
S_{\mathrm{SD}}\downarrow.
$$

Rule chính xác hơn là:

$$
\boxed{
S_{\mathrm{SD}}
=
f(
B,
L_{\mathrm{ctx}},
M_{\mathrm{target}},
M_{\mathrm{draft}},
\text{KV policy},
\text{GPU}
).
}
$$

### Research implication

Router không chỉ cần semantic state.

Nó còn cần **systems state**.

$$
s_t^{\mathrm{system}}
=
[
B,
L_{\mathrm{ctx}},
T_{\mathrm{verify}},
T_{\mathrm{draft}},
\text{KV pressure}
].
$$

**Mức bằng chứng:** Established.

---

# 5. Target verification có thể là bottleneck thật sự

Một speculative round có thể viết:

$$
T_{\mathrm{round}}
=
T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{sample}}
+
T_{\mathrm{system}}.
$$

Các empirical analyses gần đây cho thấy:

$$
T_{\mathrm{verify}}
$$

có thể chiếm phần lớn round latency.

Điều này thay đổi cách ta nghĩ về speculative decoding.

## 5.1. Traditional view

Thông thường ta nhìn vào:

$$
L_{\mathrm{acc}}
$$

hoặc:

$$
\alpha
=
\frac{L_{\mathrm{acc}}}{k}.
$$

Acceptance càng cao → method càng tốt.

Nhưng đây không phải objective cuối.

Giả sử:

### Drafter A

$$
L_{\mathrm{acc}}=5,
\qquad
T_{\mathrm{draft}}=10\text{ ms}.
$$

### Drafter B

$$
L_{\mathrm{acc}}=4,
\qquad
T_{\mathrm{draft}}=1\text{ ms}.
$$

A có acceptance tốt hơn nhưng B có thể cho wall-clock throughput tốt hơn.

## 5.2. Objective thích hợp hơn

Đại lượng gần với system performance hơn là:

$$
\boxed{
U_t
=
\frac{
\mathbb E[\text{committed tokens}]
}{
T_{\mathrm{draft}}
+
T_{\mathrm{verify}}
+
T_{\mathrm{routing}}
+
T_{\mathrm{system}}
}.
}
$$

Hoặc tương đương minimize:

$$
\boxed{
C_t
=
\frac{
T_{\mathrm{round}}
}{
\mathbb E[\text{committed tokens}]
}.
}
$$

Đây có thể được xem là:

> milliseconds per committed target token.

## 5.3. Rejected verification work

Nếu target verify $k$ tokens nhưng chỉ accept $A_t$:

$$
A_t<k,
$$

thì:

$$
k-A_t
$$

draft tokens không làm sequence tiến thêm.

Khi compute của verification tăng:

$$
C_{\mathrm{waste}}
\propto
(k-A_t)
T_{\mathrm{verify/token}}.
$$

### Research implication

Một method tốt không chỉ cần:

$$
A_t\uparrow
$$

mà còn phải:

$$
\boxed{
\text{avoid verifying tokens likely to be rejected}.
}
$$

Đây chính là motivation cho adaptive $k_t$.

**Mức bằng chứng:** Established.

---

# 6. Tree verification không tự nhiên tốt hơn chain

Tree speculative decoding được motivate rất hấp dẫn.

Thay vì chỉ proposal:

$$
x_1\rightarrow x_2\rightarrow x_3,
$$

ta proposal nhiều branches:

```text
                x1
          /      |      \
        x2       y2      z2
       /  \     /  \
     x3   y3   z3   w3
```

Target verify toàn bộ tree trong một forward.

## 6.1. Lợi ích

Probability rằng ít nhất một branch match target tăng.

Do đó:

$$
E[L_{\mathrm{accepted\ path}}]
\uparrow.
$$

## 6.2. Chi phí

Nhưng số verified nodes:

$$
N_{\mathrm{tree}}
$$

có thể tăng rất nhanh.

Ví dụ chain $k=3$:

$$
N=3.
$$

Một tree depth 3 có thể cần:

$$
N=21.
$$

Nếu accepted path chỉ tăng:

$$
2.25\rightarrow2.92,
$$

nhưng number verified tokens:

$$
3\rightarrow21,
$$

verification efficiency giảm mạnh.

## 6.3. Distinction quan trọng

Ta cần phân biệt:

### Accepted length

$$
L_{\mathrm{acc}}.
$$

### Verification acceptance density

$$
\rho
=
\frac{
L_{\mathrm{acc}}
}{
N_{\mathrm{verified}}
}.
$$

### Actual throughput

$$
U
=
\frac{
\text{committed tokens}
}{
T_{\mathrm{tree verification}}
}.
$$

Một tree có thể:

$$
L_{\mathrm{acc}}\uparrow
$$

nhưng:

$$
\rho\downarrow,
\qquad
U\downarrow.
$$

### Research implication

Tree size nên được xem là **adaptive compute budget**, không chỉ decoding hyperparameter.

$$
N_t^*
=
f(
s_t,
B,
L_{\mathrm{ctx}},
\text{hardware}
).
$$

UniSpec thực tế đi theo intuition này bằng device-aware calibration của tree budget.

**Mức bằng chứng:** Established.

---

# 7. Acceptance length không phải hằng số

Một lỗi conceptual thường gặp trong SD benchmark là chỉ report:

$$
\bar A
=
\frac1T
\sum_tA_t.
$$

Nhưng:

$$
A_t
$$

có variance rất lớn.

Ta có thể gặp trace:

$$
1,1,2,7,6,1,1,5,8,2,\ldots
$$

mặc dù:

$$
\bar A\approx3.4.
$$

Average làm mất toàn bộ temporal structure.

*Performance or Illusion?* cho thấy variability tồn tại ở ít nhất ba cấp:

$$
\boxed{
\text{token position}
\times
\text{request}
\times
\text{dataset}.
}
$$



## 7.1. Different proposer → different distribution

Learned proposer như EAGLE thường:

- stable hơn;
- accepted span tương đối consistent.

N-gram:

- nhiều short failures;
- đôi lúc extremely long accepted burst.

Nó tạo ra hai rất khác nhau dù average acceptance có thể gần nhau.

### Research implication

Ta nên log:

$$
P(A_t=k),
$$

không chỉ:

$$
E[A_t].
$$

Và đặc biệt:

$$
P(A_t\mid s_t,m).
$$

Đây mới là quantity router cần học.

---

# 8. Prompt–output overlap tạo một speculation regime riêng

Đây là một trong những insight quan trọng nhất đối với summarization.

N-gram proposer có cost cực nhỏ:

$$
T_{\mathrm{ngram}}
\ll
T_{\mathrm{neural\ draft}}.
$$

Nhưng success của nó phụ thuộc vào việc continuation đã tồn tại trong:

- source;
- self-output;
- datastore.

*Performance or Illusion?* cho thấy trên InstructCoder, khi prompt-output lexical overlap cao, n-gram có thể vượt EAGLE/EAGLE-3 rõ rệt.

SSSD khai thác chính prompt + self-output + datastore làm nguồn n-gram proposals và cho thấy training-free speculation có thể cạnh tranh với learned drafter khi workload phù hợp.

## 8.1. Deeper interpretation

Không phải:

$$
\text{n-gram}>\text{EAGLE}.
$$

Mà:

$$
\boxed{
\text{Prediction mechanism nên match structure của workload.}
}
$$

Nếu continuation đã nằm trong source:

$$
\text{retrieve}
$$

có thể hợp lý hơn:

$$
\text{infer}.
$$

Ta có thể xem đây là:

$$
\boxed{
\text{retrieval vs prediction trade-off}.
}
$$

## 8.2. Summarization đặc biệt phù hợp

Summary thường cần giữ chính xác:

- tên người;
- tổ chức;
- số liệu;
- ngày tháng;
- thuật ngữ;
- location;
- quoted phrase;
- product/program names.

Ví dụ source:

> The program reduced annual operating costs by **17.3 percent**.

Summary:

> The program cut costs by **17.3 percent**.

Một part là abstractive:

`reduced annual operating costs` → `cut costs`

nhưng span:

`17.3 percent`

là pure copy opportunity.

### Research implication

Một output không phải toàn bộ extractive hay toàn bộ abstractive.

Ngay trong một câu:

$$
\text{paraphrase}
\rightarrow
\text{copy}
\rightarrow
\text{paraphrase}.
$$

Đây là motivation mạnh cho token/segment-level proposer switching.

**Mức bằng chứng:** Established về overlap; synthesis cho summarization.

---

# 9. Summarization có thể gồm nhiều generation regimes, không chỉ COPY và GENERATE

Report ban đầu dùng:

$$
\text{COPY}
\leftrightarrow
\text{GENERATE}.
$$

Đây là abstraction tốt để bắt đầu nhưng có thể mở rộng thành ít nhất năm trạng thái.

## 9.1. Lexically source-copyable

Ví dụ:

> World Health Organization

hay:

> 17.3 percent.

Preferred proposer có thể là:

$$
\text{source n-gram}.
$$

## 9.2. Source-grounded paraphrase

Source:

> profits declined by 8 percent.

Summary:

> earnings fell 8%.

Source strongly determines semantics nhưng lexical overlap không hoàn toàn.

Neural drafter có thể phù hợp hơn retrieval.

## 9.3. Multi-evidence synthesis

Source chunks:

$$
C_3,\quad C_{17},\quad C_{28}
$$

cùng cần để tạo một sentence.

Đây là difficult region:

$$
H_t\uparrow
$$

và local n-gram proposal có thể yếu.

## 9.4. Structural/discourse generation

Ví dụ:

> However, ...

> Overall, ...

> The report also found ...

Những token này có thể highly predictable nhưng source grounding thấp.

## 9.5. High-uncertainty transition

Khi model chuyển topic/section, nhiều continuation hợp lệ.

Có thể:

$$
\text{no-spec}
$$

hoặc:

$$
k_t\downarrow
$$

là optimal.

### Hypothesis

Ta có thể đặt latent state:

$$
z_t
\in
\{
C,
P,
S,
D,
U
\}.
$$

Router không nhất thiết cần predict $z_t$ explicitly.

Chỉ cần state features đủ tốt để predict action:

$$
a_t^*.
$$

**Mức bằng chứng:** Open hypothesis.

Đây là một trong những hypotheses quan trọng nhất cần oracle study.

---

# 10. Oracle proposal length cho thấy fixed $k$ lãng phí đáng kể

Standard SD thường chọn:

$$
k=3,5,7,\ldots
$$

và giữ cố định trong toàn request.

Nhưng actual accepted length:

$$
A_t
$$

thay đổi.

Oracle biết tương lai và đặt:

$$
k_t=A_t.
$$

Khi đó:

$$
N_{\mathrm{rejected}}=0
$$

cho proposal block.

Trong *Performance or Illusion?*, oracle proposal length tạo gap đáng kể so với best fixed $k$, và gap tăng trong một số large-batch regimes.

## 10.1. Ý nghĩa thật sự

Oracle không phải algorithm thực tế.

Không thể biết:

$$
A_t
$$

trước verification.

Do đó oracle chỉ trả lời:

> Có đủ headroom để adaptive-$k$ đáng nghiên cứu không?

Nếu:

$$
S_{\mathrm{oracle}}
\approx
S_{\mathrm{fixed}},
$$

adaptive controller ít giá trị.

Nếu:

$$
S_{\mathrm{oracle}}
\gg
S_{\mathrm{fixed}},
$$

ta mới hỏi:

> Có feature nào predict $A_t$ đủ tốt?

Possible features:

$$
H_t,
$$

$$
A_{t-1},
$$

$$
\text{draft confidence},
$$

$$
\text{source overlap},
$$

$$
\text{grounding},
$$

$$
\text{recent rejection history}.
$$

### Research implication

Research sequence đúng là:

$$
\boxed{
\text{Oracle headroom}
\rightarrow
\text{Predictability}
\rightarrow
\text{Controller}.
}
$$

Không nên xây controller trước.

---

# 11. Không tồn tại universal best proposer

Hiện có nhiều proposer families.

### Neural feature proposer

EAGLE/EAGLE-3.

Strength:

- robust;
- semantic generalization;
- high acceptance.

Weakness:

- training;
- GPU cost;
- context sensitivity.

### N-gram/source retrieval

SSSD/PLD/UniSpec.

Strength:

- extremely cheap;
- training free;
- bursty long acceptance khi copyable.

Weakness:

- paraphrase;
- novel synthesis;
- low-overlap generation.

### Separate small LM

Strength:

- generic;
- conceptually simple.

Weakness:

- AR drafting cost;
- duplicate KV/model footprint.

### Diffusion/parallel drafter

Strength:

- draft block parallelism.

Weakness:

- suffix decay;
- weaker intra-block causal modeling.

### No speculation

$$
k=0.
$$

Đây cũng phải được coi là một expert.

Nếu:

$$
U_{\mathrm{spec}}<U_{\mathrm{AR}},
$$

best action là:

$$
a_t=\mathrm{AR}.
$$

On-the-Fly Adaptive SD cũng cho thấy fixed speculation có thể không tối ưu và no-spec cần tồn tại trong action space.

---

# 12. Proposer complementarity có thể tạo headroom rất lớn — nhưng không phải ở mọi workload

Oracle experiments trong *Performance or Illusion?* cho thấy các methods thắng ở các output positions khác nhau.

Một idealized combined oracle đạt tới:

$$
\sim4.9\times
$$

trong một số setup.

Nhưng cần hiểu oracle này biết cả:

1. proposer nào tốt nhất;
2. accepted length tối ưu.

Do đó:

$$
4.9\times
$$

không phải performance target trực tiếp.

Điều quan trọng hơn là:

$$
\boxed{
\exists t_1,t_2:
m_{t_1}^*\neq m_{t_2}^*.
}
$$

Nhưng gap giữa oracle combination và best proposer thay đổi mạnh theo workload.

Ví dụ:

$$
\Delta_{\mathrm{oracle}}
$$

có thể lớn trên code/editing nhưng nhỏ trên GSM8K.

### Research implication

Trước khi xây summarization router:

$$
\boxed{
\text{Measure oracle proposer complementarity on summarization first.}
}
$$

Nếu:

$$
\Delta_{\mathrm{oracle}}
\approx0,
$$

dừng hướng multi-proposer.

Nếu:

$$
\Delta_{\mathrm{oracle}}
\gg0,
$$

router có empirical justification.

---

# 13. Source relevance thay đổi trong quá trình sinh summary

Đây là một insight rất mạnh từ summarization literature.

Giả sử document:

$$
X=
C_1\cup C_2\cup C_3\cup C_4
$$

với:

- $C_1$: background;
- $C_2$: methods;
- $C_3$: results;
- $C_4$: recommendations.

Summary có thể generate:

$$
y_{1:25}
\leftarrow C_1,
$$

sau đó:

$$
y_{26:60}
\leftarrow C_3,
$$

rồi:

$$
y_{61:90}
\leftarrow C_4.
$$

Vì vậy relevance nên được viết:

$$
R(C_i,t),
$$

không phải:

$$
R(C_i).
$$

DYLE cung cấp evidence rằng snippet relevance trong long-input summarization cần dynamic weighting theo decoding state.

## 13.1. Relation với speculative decoding

Nếu drafter luôn giữ:

$$
X_{\mathrm{full}},
$$

nó trả memory/attention cost cho nhiều chunks không hữu ích ở timestep hiện tại.

Ta có thể thay bằng:

$$
C_t
=
\operatorname{RelevantSource}(X,y_{<t}).
$$

SpecExtend tiến rất gần ý tưởng này.

Nó reuse target attention để update source chunks trong draft KV cache.

### Research implication

SpecExtend giải:

$$
C_t^*.
$$

Một generalization tự nhiên là jointly solve:

$$
\boxed{
(m_t,C_t,k_t).
}
$$

---

# 14. Long context không có nghĩa mọi context interaction đều có giá trị ngang nhau

MInference cho thấy attention matrices của long-context LMs có strong structured sparsity.

Các patterns như:

- A-shape;
- Vertical-Slash;
- Block-Sparse

xuất hiện thường xuyên và có thể được dùng để giảm computation.

Điểm quan trọng hơn đối với chúng ta là sparsity **dynamic theo input**.

$$
M(X_1)
\neq
M(X_2).
$$

Do đó fixed sparse mask khó tối ưu universal.

## 14.1. Nhưng attention sparsity ≠ semantic relevance

Đây là distinction rất quan trọng.

Nếu:

$$
A_{ij}\approx0,
$$

không thể tự động kết luận:

> token $j$ semantically irrelevant cho summary.

Attention head có nhiều chức năng:

- syntax;
- position;
- induction;
- retrieval;
- normalization;
- sink behavior.

Do đó MInference hỗ trợ claim:

$$
\boxed{
\text{all context interactions need not receive equal compute},
}
$$

nhưng không trực tiếp chứng minh:

$$
\text{top-attention token}
=
\text{summary-important token}.
$$

### Research implication

Sparsity literature cung cấp **compute-allocation principle** chứ không trực tiếp cung cấp summarization selector.

---

# 15. Early layers đã chứa relevance signal

GemFilter đưa ra observation hấp dẫn:

> Relevant tokens có thể được phát hiện từ early layers trước khi toàn bộ target computation hoàn thành.

Điều này thay đổi architecture của semantic selection.

Traditional pipeline:

$$
X
\rightarrow
\text{MiniLM/BERT selector}
\rightarrow
X'
\rightarrow
T.
$$

Possible target-derived pipeline:

$$
X
\rightarrow
T_{1:l}
\rightarrow
\text{importance}
\rightarrow
T_{l+1:L}(X').
$$

Không cần separate semantic encoder.

## 15.1. Vì sao điều này hấp dẫn?

Một external selector thêm:

$$
T_{\mathrm{selector}}.
$$

Target-derived selector reuse:

$$
h_i^{(l)}
$$

đã được tính.

Do đó marginal signal cost có thể thấp hơn.

## 15.2. Nhưng không phải free

Full source vẫn phải chạy qua:

$$
1,\ldots,l.
$$

Do đó:

$$
T_{\mathrm{selection}}
>
0.
$$

Và GemFilter-style relevance evidence từ retrieval/QA không tự động chứng minh global summarization coverage.

Summarization phải giữ:

$$
\{f_1,f_2,\ldots,f_m\}
$$

trải khắp document, thay vì tìm một needle duy nhất.

### Research implication

Early target features là promising **cheap state feature**, nhưng cần summarization-specific validation.

---

# 16. Raw attention có systematic bias

Nếu sử dụng:

$$
A_t(i)
$$

làm relevance score, ta đang giả định:

$$
A_t(i)\propto R_t(i).
$$

Literature cho thấy assumption này quá mạnh.

## 16.1. Attention sink

StreamingLLM cho thấy initial tokens có thể nhận attention rất lớn dù không mang semantic evidence trực tiếp.

Ta có:

$$
A_t(i)
=
R_t(i)
+
S(i)
+\epsilon.
$$

$S(i)$ là sink effect.

## 16.2. Position bias / Lost in the Middle

Long-context models thường sử dụng evidence tốt hơn khi nó nằm ở đầu/cuối context so với giữa.

Một rough bias:

$$
B(p)
\approx
\text{U-shape}.
$$

Do đó:

$$
A_t(i)
=
R_t(i)
+
B(p_i)
+
S(i)
+
\epsilon_i.
$$

Nếu selector dùng raw attention:

$$
\operatorname{TopK}_iA_t(i),
$$

relevance estimation có thể bị bias.

### Research implication

Một target-attention-based routing method nên xem xét:

$$
\boxed{
\tilde A_t(i)
=
\operatorname{Calibrate}
(
A_t(i),
p_i,
h
).
}
$$

Nhưng không nên “debias mọi positional information” một cách máy móc.

Document position đôi khi là real semantic signal.

CNN/DM chẳng hạn có lead bias.

Do đó cần phân biệt:

$$
\boxed{
\text{useful structural prior}
}
$$

và:

$$
\boxed{
\text{spurious positional preference}.
}
$$

---

# 17. Retrieval information không phân bố đều giữa attention heads

Một missing insight quan trọng của bản báo cáo ban đầu là head heterogeneity.

Một số work về retrieval heads cho thấy chỉ một subset nhỏ attention heads chịu trách nhiệm đáng kể cho long-context retrieval.

Do đó:

$$
\frac1H
\sum_hA_{t,h}
$$

có thể làm dilution retrieval signal.

Nếu chỉ:

$$
5\%
$$

heads mang strong retrieval behavior, average với 95% non-retrieval heads có thể tạo noisy score.

Một estimator hợp lý hơn:

$$
G_t(i)
=
\sum_h
w_{t,h}
A_{t,h}(i).
$$

Trong đó:

$$
w_{t,h}
$$

phản ánh utility của head $h$ cho source retrieval.

---

# 18. Retrieval heads bản thân cũng thay đổi theo decoding timestep

Một static set:

$$
\mathcal H_R
$$

không nhất thiết đủ.

Recent work về dynamic retrieval heads cho thấy retrieval behavior có thể chuyển theo timestep.

Điều này rất phù hợp với summarization.

Khi model đang generate entity:

$$
h_{17},h_{23}
$$

có thể active.

Khi generate discourse connector:

$$
h_{17},h_{23}
$$

không còn đóng vai trò chính.

Do đó:

$$
w_{t,h}
$$

nên potentially dynamic:

$$
w_{t,h}
=
f(
h_t,
y_{<t},
\text{attention history}
).
$$

### Research implication

Grounding không chỉ dynamic theo **source chunk**, mà cả theo **head**.

---

# 19. Source-vs-generated-context attention có thể là grounding signal mạnh

Một token có thể attend tới:

### Original source

$$
A_t^{\mathrm{src}}
$$

hoặc:

### previously generated summary

$$
A_t^{\mathrm{gen}}.
$$

Lookback-style approaches exploit ratio:

$$
G_t
=
\frac{
A_t^{\mathrm{src}}
}{
A_t^{\mathrm{src}}
+
A_t^{\mathrm{gen}}
}.
$$

Interpretation:

### $G_t$ cao

Current generation strongly consults source.

### $G_t$ thấp

Generation dựa nhiều vào its own generated trajectory.

Đây là potentially useful distinction.

## 19.1. Relation với proposer

Hypothesis:

$$
G_t\uparrow
\Rightarrow
U_{\mathrm{source\ proposer}}\uparrow.
$$

Nếu model đang source-grounded:

- source n-gram;
- retrieved source context;
- source-conditioned drafter

có thể hữu ích hơn.

Nếu:

$$
G_t\downarrow,
$$

generic neural proposer có thể tốt hơn.

### Research implication

Đây là feature rất đáng kiểm tra trong oracle study.

**Mức bằng chứng:** Evidence cho grounding/hallucination; open hypothesis cho proposer routing.

---

# 20. Factuality phải được xem là system constraint

Long-document summarization khác code generation ở một điểm rất quan trọng:

$$
\boxed{
\text{output phải trung thành với source}.
}
$$

Nếu acceleration làm summary nhanh hơn nhưng mất số liệu hoặc hallucinate fact, gain có thể không có giá trị.

HalluKV và related work cho thấy aggressive KV compression có thể gây factual degradation/hallucination mặc dù aggregate quality metrics không phản ánh đầy đủ.

Do đó objective của approximate inference nên là:

$$
\min T
$$

subject to:

$$
Q\ge Q_0-\epsilon
$$

và:

$$
\boxed{
F\ge F_0-\delta.
}
$$

Trong đó:

- $Q$: summary quality;
- $F$: factual consistency.

Metrics cần bao gồm:

- entity preservation;
- number accuracy;
- date accuracy;
- negation consistency;
- source entailment;
- hallucination score.

### Một interpretation sâu hơn

Source KV/cache không chỉ là memory state.

Nó đóng vai trò **factual anchor**.

Nếu source evidence bị aggressively dropped:

$$
\text{generation}
\rightarrow
\text{self-conditioned generation}
\rightarrow
\text{possible factual drift}.
$$

Điều này làm source-grounding state có ý nghĩa không chỉ cho speed mà còn cho reliability.

---

# 21. Drafter context có thể lossy nhưng target inference vẫn exact

Đây là một insight rất giá trị trong báo cáo gốc.

Target distribution:

$$
p_T(y_t\mid X,y_{<t}).
$$

Drafter không bắt buộc approximate target bằng cùng context.

Nó có thể dùng:

$$
q_D(y_t\mid C_t,y_{<t}),
$$

với:

$$
C_t\subset X.
$$

Thậm chí:

$$
C_t
=
\operatorname{Transform}(X).
$$

Nếu speculative acceptance/correction được implement đúng và target luôn verify trên:

$$
X_{\mathrm{full}},
$$

proposal distribution có thể khác target mà final target distribution vẫn được preserve.

Điều này tạo một rất quan trọng asymmetry:

$$
\boxed{
\text{Target context must be faithful;}
\quad
\text{draft context only needs to be useful.}
}
$$

## 21.1. Possible drafter context

$$
C_t
=
[
C_{\mathrm{relevant}},
C_{\mathrm{recent}},
C_{\mathrm{local}}
].
$$

Hoặc:

$$
C_t
=
\operatorname{TopKChunks}
(
G_t
).
$$

---

# 22. Drafter-only source reordering là một hypothesis thú vị nhưng cần cẩn thận

Báo cáo đề xuất tận dụng lost-in-the-middle bằng cách đưa relevant chunks về đầu/cuối context của drafter.

Conceptually:

$$
X_D
=
[
C_{\mathrm{important}},
C_{\mathrm{local}},
C_{\mathrm{recent}}
].
$$

Nếu drafter là independent LM, đây là plausible.

Nhưng có caveats.

## 22.1. RoPE / positional semantics

Khi reorder:

$$
p_i^{\mathrm{original}}
\neq
p_i^{\mathrm{draft}}.
$$

Drafter distribution có thể thay đổi mạnh.

## 22.2. Target-feature-conditioned drafter

EAGLE/LongSpec/DFlash-like methods có thể dùng:

- target hidden states;
- target embeddings;
- target positional states.

Reorder input nhưng reuse target state có thể gây misalignment.

## 22.3. Retrieval coherence

Một chunk tách khỏi surrounding context có thể mất:

- coreference;
- section context;
- temporal relations.

### Research question

Không nên claim:

> reorder giải quyết Lost in the Middle.

Nên hỏi:

$$
\boxed{
\text{Does position-aware reordering improve draft utility at fixed draft-context budget?}
}
$$

Và benchmark:

$$
L_{\mathrm{acc}},
\quad
T_{\mathrm{draft}},
\quad
U.
$$

---

# 23. Good LM không đồng nghĩa good drafter

DistillSpec cho một lesson quan trọng.

Training a small LM để minimize:

$$
L_{\mathrm{NTP}}
$$

không giống optimize drafter cho target agreement.

Drafter objective thực sự gần hơn:

$$
D(
q_D
\Vert
p_T
)
\downarrow.
$$

DistillSpec cho thấy:

- on-policy data;
- divergence choice;
- target–draft alignment

đều quan trọng.

### Implication

Một summarization drafter không nên được đánh giá bằng:

- perplexity;
- ROUGE riêng;
- generic benchmark accuracy.

Metric quan trọng là:

$$
\boxed{
\text{agreement per unit latency}.
}
$$

---

# 24. Draft latency phụ thuộc architecture chứ không chỉ parameter count

*Decoding Speculative Decoding* bổ sung lesson còn lại.

Giả sử:

$$
|D_1|=|D_2|
$$

theo parameter count.

Nếu $D_1$ deeper:

$$
L_1>L_2,
$$

AR draft requires nhiều sequential layer operations hơn.

Do đó:

$$
T_{\mathrm{draft}}(D_1)
>
T_{\mathrm{draft}}(D_2)
$$

có thể xảy ra dù số params giống nhau.

Paper cho thấy shallow-wide draft architectures có thể superior cho speculative throughput.

Kết hợp DistillSpec:

$$
\boxed{
\text{Good drafter}
=
\text{high target alignment}
+
\text{low sequential latency}.
}
$$

Hay formal hơn:

$$
U_D
=
\frac{
\operatorname{Agreement}(D,T)
}{
T_D
}.
$$

---

# 25. Diffusion drafter giải quyết latency nhưng tạo suffix-decay problem

Diffusion/block-parallel drafter thay:

$$
y_1\rightarrow y_2\rightarrow\cdots\rightarrow y_k
$$

bằng gần-parallel prediction:

$$
(y_1,\ldots,y_k).
$$

Điều này giảm AR draft depth.

Nhưng token $y_i$ ít được conditioning trên exact previous proposed tokens.

Do đó thường:

$$
P_D(y_i=y_i^T)
$$

giảm theo $i$.

Roughly:

$$
a_1>a_2>\cdots>a_k.
$$

Trong khi SD verification chấp nhận longest prefix.

Nếu token 2 sai:

$$
y_{3:k}
$$

dù đúng cũng vô dụng.

Đây là **suffix decay × prefix acceptance mismatch**.

### Research implication

Diffusion proposer có thể hữu ích ở states mà:

- entropy thấp;
- local continuation predictable;
- block structure stable.

Không nhất thiết dùng ở mọi timestep.

Đây lại hỗ trợ multi-proposer routing.

---

# 26. Long-context speculative decoding có ít nhất ba failure modes riêng biệt

LongSpec cho thấy SD trên long context không phải chỉ short-context SD + longer prompt.

Có ba vấn đề khác nhau.

## 26.1. Draft KV growth

$$
M_{\mathrm{KV}}^D
=
O(L_{\mathrm{ctx}}).
$$

Small drafter có thể mất advantage vì cache memory/bandwidth.

## 26.2. Short-train / long-test mismatch

Drafter thường train ở:

$$
L_{\mathrm{train}}\ll L_{\mathrm{test}}.
$$

Target-draft agreement giảm.

## 26.3. Verification kernel inefficiency

Tree masks không tương thích tự nhiên với highly optimized long-context attention kernels.

Do đó một complete long-context SD system phải tối ưu:

$$
\boxed{
\text{draft memory}
+
\text{draft alignment}
+
\text{verification kernel}.
}
$$

### Research implication

Context routing chỉ giải một phần problem.

Nếu method mới tăng acceptance 20% nhưng verification kernel chậm 30%:

$$
S_{\mathrm{E2E}}<1.
$$

---

# 27. Source không chỉ là input — nó là một computational resource

Đây là một synthesis rất mạnh từ SSSD, RAPID và SpecExtend.

Traditional view:

$$
X
\rightarrow
\text{Target}.
$$

New view:

$$
X
\rightarrow
\begin{cases}
\text{conditioning}\\
\text{retrieval datastore}\\
\text{n-gram proposal source}\\
\text{draft context}\\
\text{grounding signal}
\end{cases}
$$

## 27.1. SSSD

Source/self-output chứa exact continuation.

## 27.2. RAPID

Relevant source subset làm drafter cheap hơn.

Một surprising implication là:

$$
\boxed{
\text{drafter không nhất thiết phải nhỏ hơn target}.
}
$$

Nếu:

$$
|C_D|\ll|X|,
$$

một relatively strong drafter trên short context vẫn có thể cheaper.

## 27.3. SpecExtend

Target itself tells drafter source nào nên giữ.

### Research implication

Source-aware SD không nên chỉ hỏi:

> “How to compress source?”

Nên hỏi:

> “Source có thể cung cấp computation nào cho free hoặc cheap?”

---

# 28. Later transformer layers có redundancy trong prefill

SwiftKV được motivate trực tiếp bởi long-prompt/short-output workloads như summarization.

Observation:

$$
h_i^{(l)}
$$

ở later layers có đủ redundancy để KV của later layers có thể được approximated từ earlier representation sau lightweight adaptation.

Do đó prompt tokens không nhất thiết chạy:

$$
1\rightarrow2\rightarrow\cdots\rightarrow L.
$$

Có thể:

$$
1\rightarrow\cdots\rightarrow l
\rightarrow
\text{KV projection}
$$

cho later layers.

Điều này tạo layer sparsity:

$$
C_{\mathrm{layer}}(l).
$$

---

# 29. Token sparsity × layer sparsity × head sparsity × temporal sparsity

Khi kết hợp literature:

### MInference/GemFilter

$$
\text{token/attention sparsity}.
$$

### SwiftKV

$$
\text{layer sparsity}.
$$

### Retrieval-head literature / DejaVu

$$
\text{head/parameter sparsity}.
$$

### Adaptive SD

$$
\text{temporal/speculation sparsity}.
$$

Một abstraction tổng quát là:

$$
\boxed{
C(i,l,h,t)
}
$$

trong đó $C$ quyết định computation allocated cho:

- token $i$;
- layer $l$;
- head/component $h$;
- decoding time $t$.

Đây là một unifying research vision rất mạnh.

Nhưng scope hiện tại không nên solve toàn bộ tensor này.

Một feasible subproblem là:

$$
\boxed{
C(m,C_t,k_t,t)
}
$$

tức:

- proposer $m$;
- source context $C_t$;
- proposal budget $k_t$;
- timestep $t$.

---

# 30. Prefill–decode trade-off phải được nhìn bằng Amdahl's law

Long-document summarization có:

$$
L_{\mathrm{source}}\gg L_{\mathrm{summary}}.
$$

Giả sử baseline latency:

$$
T
=
T_P+T_D.
$$

Prefill fraction:

$$
f_P
=
\frac{T_P}{T}.
$$

Decode fraction:

$$
f_D
=
\frac{T_D}{T}.
$$

Nếu ta accelerate decode $S_D$ lần:

$$
S_{\mathrm{E2E}}
=
\frac1{
f_P+\frac{f_D}{S_D}
}.
$$

Ví dụ:

$$
f_P=0.7,\quad f_D=0.3.
$$

Ngay cả:

$$
S_D\rightarrow\infty,
$$

ta chỉ có:

$$
S_{\mathrm{E2E}}
\le
\frac1{0.7}
=
1.43\times.
$$

Đây là reason fundamental tại sao decode-only paper có thể có:

$$
3\times\text{ TPOT}
$$

nhưng E2E summarization gain nhỏ.

### Research implication

Mọi result phải report riêng:

$$
\boxed{
TTFT,\quad TPOT,\quad E2E.
}
$$

Không được đặt cạnh nhau:

- 7× TTFT;
- 3× decode;
- 2× E2E

như cùng một quantity.

Literature review tổng hợp của project đã nhấn mạnh distinction này.

---

# 31. Semantic state và system state phải được xem cùng lúc

Từ các insight trên, chỉ nhìn semantic features là chưa đủ.

Ta có thể define:

$$
s_t
=
[
s_t^{\mathrm{semantic}},
s_t^{\mathrm{system}}
].
$$

## 31.1. Semantic state

Potential features:

$$
H_t
$$

target entropy;

$$
G_t
$$

source-grounding ratio;

$$
C_t^{\mathrm{copy}}
$$

source-copyability;

$$
R_t
$$

retrieval-head activity;

$$
A_t^{\mathrm{history}}
$$

recent accepted length;

$$
D_t
$$

draft confidence/divergence.

## 31.2. System state

$$
B
$$

batch/load;

$$
L_{\mathrm{ctx}}
$$

context length;

$$
T_{\mathrm{verify}}
$$

recent measured verification latency;

$$
T_D^m
$$

drafter cost;

$$
M_{\mathrm{KV}}
$$

KV pressure.

---

# 32. Action space tự nhiên

Router có thể chọn:

$$
u_t
=
(m_t,k_t,C_t).
$$

Trong đó:

$$
m_t
\in
\{
\text{no-spec},
\text{n-gram},
\text{neural},
\text{diffusion}
\}.
$$

$$
k_t
\in
\{1,\ldots,k_{\max}\}.
$$

Và:

$$
C_t
\subseteq X.
$$

Objective:

$$
\boxed{
u_t^*
=
\arg\max_u
\frac{
\mathbb E[
\text{committed tokens}
\mid
s_t,u
]
}{
T(u\mid s_t)
}.
}
$$

Đây là formulation chính xác hơn:

$$
\max L_{\mathrm{acc}}
$$

hoặc:

$$
\max\text{acceptance rate}.
$$

---

# 33. Nhưng router overhead có thể phá toàn bộ lợi ích

Một adaptive controller không tự nhiên tốt hơn fixed policy.

Nếu:

$$
T_{\mathrm{router}}
$$

lớn, gain có thể biến mất.

Điều kiện cần:

$$
\boxed{
T_{\mathrm{router}}
<
T_{\mathrm{saved\ compute}}.
}
$$

Đây là lý do target-derived signals hấp dẫn.

Các quantity như:

- entropy;
- attention;
- hidden state;
- accepted history

đã phần lớn được compute.

Mục tiêu nên là:

$$
\text{reuse already-paid-for signals}.
$$

Thay vì chạy thêm:

$$
\text{7B routing LLM}.
$$

Một controller thực tế có thể chỉ là:

- thresholds;
- linear model;
- shallow MLP;
- contextual bandit;
- tiny decision tree.

---

# 34. Research hypothesis trung tâm chưa được literature chứng minh

Sau toàn bộ synthesis, các facts sau đã có evidence khá mạnh:

$$
\boxed{
\begin{aligned}
&\text{token difficulty varies};\\
&\text{accepted length varies by position};\\
&\text{proposer performance varies by workload};\\
&\text{n-gram utility increases with overlap};\\
&\text{source relevance changes during summary decoding};\\
&\text{attention/retrieval behavior is dynamic};\\
&\text{optimal speculation depends on systems regime}.
\end{aligned}}
$$

Nhưng chưa có evidence trực tiếp cho:

$$
\boxed{
\text{grounding state}
\rightarrow
\text{best speculative proposer}.
}
$$

Hay cụ thể:

$$
P(
m_t^*
\mid
G_t,H_t,C_t^{\mathrm{copy}},A_{<t}
)
$$

có sufficiently low entropy hay không.

Đây chính là hypothesis cần kiểm chứng.

---

# 35. Oracle study nên là gate đầu tiên

Không cần build router ngay.

Đầu tiên chạy multiple proposers trên cùng target decoding traces:

$$
M=
\{
\text{n-gram},
\text{EAGLE},
\text{DFlash},
\text{AR}
\}.
$$

Tại mỗi timestep $t$, tính:

$$
U_t(m).
$$

Sau đó oracle:

$$
m_t^*
=
\arg\max_mU_t(m).
$$

Compare:

$$
S_{\mathrm{oracle}}
$$

với:

$$
S_{\mathrm{best-fixed}}.
$$

Define:

$$
\boxed{
\Delta_{\mathrm{oracle}}
=
S_{\mathrm{oracle}}
-
S_{\mathrm{best-fixed}}.
}
$$

## Decision rule

Nếu:

$$
\Delta_{\mathrm{oracle}}\approx0,
$$

multi-proposer routing không đáng tiếp tục.

Nếu:

$$
\Delta_{\mathrm{oracle}}\gg0,
$$

tiếp tục hỏi:

$$
\boxed{
\text{Can cheap features predict }m_t^*?
}
$$

---

# 36. Grounding trace study

Song song oracle proposer trace, log:

$$
H_t
$$

entropy;

$$
G_t
$$

source/generated attention ratio;

$$
O_t
$$

source n-gram overlap;

$$
R_t
$$

retrieval-head activity;

$$
A_{t-1:t-r}
$$

recent accepted history;

$$
p_D
$$

draft confidence.

Sau đó analyze:

$$
P(
m_t^*
\mid
H_t,G_t,O_t,R_t
).
$$

Questions:

### Q1

Có region:

$$
O_t\uparrow
\Rightarrow
m_t^*=\text{n-gram}?
$$

### Q2

Có region:

$$
G_t\uparrow,\ O_t\downarrow
\Rightarrow
m_t^*=\text{neural}?
$$

### Q3

Có region:

$$
H_t\uparrow
\Rightarrow
m_t^*=\text{AR/no-spec}?
$$

### Q4

Diffusion có region riêng không?

Nếu không:

$$
\text{drop diffusion expert}.
$$

---

# 37. Summarization-specific evaluation là bắt buộc

Oracle study nên chạy trên datasets có structure khác nhau.

### CNN/DailyMail

High lexical overlap / lead bias.

Expected:

$$
U_{\mathrm{ngram}}\uparrow.
$$

### XSum

Highly abstractive.

Expected:

$$
U_{\mathrm{ngram}}\downarrow.
$$

### GovReport

Long structured documents.

Tests:

- dynamic source relevance;
- numbers;
- section movement.

### Multi-News

Multiple documents.

Tests:

- coverage;
- redundancy;
- source switching.

### QMSum

Query-focused summarization.

Tests:

- sparse relevance;
- targeted evidence.

Nếu proposer complementarity chỉ xuất hiện trên CNN/DM:

$$
\text{method may merely exploit extractiveness}.
$$

Nếu xuất hiện trên GovReport/Multi-News/XSum:

$$
\text{hypothesis stronger}.
$$

---

# 38. Research thesis tổng hợp

Toàn bộ evidence từ speculative decoding, long-context inference, semantic selection và summarization có thể được cô đọng thành:

$$
\boxed{
\textbf{
Long-document summarization inference is a heterogeneous compute-allocation problem.
}
}
$$

Computation utility thay đổi theo:

$$
\boxed{
\text{semantic state}
\times
\text{source grounding}
\times
\text{proposer}
\times
\text{context}
\times
\text{systems regime}.
}
$$

Fixed policy:

$$
(m,C,k)=\text{constant}
$$

giả định workload homogeneous.

Evidence hiện tại ngày càng cho thấy assumption này không đúng.

Do đó research question tổng quát là:

$$
\boxed{
\textbf{
Can cheap target-derived signals predict which computation is worth paying for next?
}
}
$$

Trong scope speculative decoding cho summarization, câu hỏi trở thành:

$$
\boxed{
\textbf{
Can grounding, copyability and difficulty signals dynamically select proposer, draft context and speculation budget to maximize committed-token throughput?
}
}
$$

---

# 39. Điều gì đã được literature giải quyết và điều gì vẫn còn mở?

| Thành phần | Literature hiện tại | Tình trạng |
|---|---|---|
| Adaptive proposal length | Adaptive SD / oracle studies | Đã có |
| Adaptive proposer | Performance-or-Illusion oracle / routing work | Đã có generic |
| N-gram proposal | PLD / SSSD / UniSpec | Đã có |
| Neural proposal | EAGLE family | Đã có |
| Diffusion proposal | DFlash family | Đã có |
| Long-context drafter | LongSpec | Đã có |
| Target attention → draft context | SpecExtend | Đã có |
| Dynamic source relevance | DYLE | Đã có summarization |
| Attention debiasing | Found in the Middle | Đã có |
| Retrieval-head specialization | Retrieval-head literature | Đã có |
| Source grounding signal | Lookback-style work | Đã có |
| Factuality-sensitive context | HalluKV | Đã có |
| Grounding → proposer selection | — | **Chưa được chứng minh rõ** |
| Joint proposer + context + $k$ | — | **Gap tiềm năng** |
| Summarization-specific adaptive SD | rất hạn chế | **Gap tiềm năng** |
| Grounding + runtime joint routing | chưa rõ direct prior | **Gap mạnh hơn** |

Do đó novelty không nên được đặt ở:

> “chúng tôi dùng attention.”

hoặc:

> “chúng tôi adapt speculation length.”

Mà phải nằm ở interaction:

$$
\boxed{
\text{
summarization grounding
\times
proposer utility
\times
dynamic context
\times
systems-aware budget
}.
}
$$

---

# 40. Kết luận

Các nghiên cứu hiện tại đang chỉ đến một observation thống nhất:

> **LLM inference không cần cùng loại computation ở mọi token, mọi source region, mọi layer và mọi serving condition.**

Trong speculative decoding:

$$
\text{fixed proposer}
+
\text{fixed }k
$$

lãng phí rejected verification.

Trong long-context inference:

$$
\text{full source}
$$

không có utility ngang nhau cho drafter tại mọi timestep.

Trong summarization:

$$
\text{source dependence}
$$

thay đổi khi generation chuyển từ fact này sang fact khác, từ copying sang abstraction và synthesis.

Trong production:

$$
\text{same speculative policy}
$$

có utility khác nhau tùy batch, context length và hardware.

Vì vậy một hướng nghiên cứu tự nhiên là chuyển từ:

$$
\boxed{
\text{better drafter}
}
$$

sang:

$$
\boxed{
\textbf{better allocation of speculative computation}.
}
$$

Và bước nghiên cứu tiếp theo không nên là xây một architecture phức tạp ngay lập tức.

Bước đúng là:

$$
\boxed{
\text{Oracle proposer study}
\rightarrow
\text{Grounding trace analysis}
\rightarrow
\text{Feature predictability}
\rightarrow
\text{Lightweight controller}
\rightarrow
\text{Production evaluation}.
}
$$

Nếu oracle proposer complementarity không tồn tại trên summarization, hướng multi-proposer routing nên được loại bỏ sớm.

Nếu complementarity tồn tại nhưng grounding features không predict được winner, cần tìm state representation khác.

Chỉ khi cả hai điều kiện:

$$
\Delta_{\mathrm{oracle}}\gg0
$$

và:

$$
P(m_t^*\mid s_t)
$$

có predictability đủ mạnh, thì Grounding-Aware Adaptive Speculative Decoding mới có nền tảng empirical đủ chắc để trở thành phương pháp chính.

Đây cũng là cách biến toàn bộ literature hiện tại từ một tập hợp các optimization rời rạc thành một research thesis thống nhất:

$$
\boxed{
\textbf{
Summarization-Specific Adaptive Compute Allocation
}
}
$$

với speculative decoding là nơi đầu tiên để kiểm chứng thesis đó.