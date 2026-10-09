# AMR-DFlash — hồ sơ ý tưởng, paper story và huấn luyện

**Tài liệu chính để kiểm soát ý tưởng và xây dựng paper.**
Ngày tạo/cập nhật: **09/10/2026**. Revision: **2 — xác nhận corpus/cache 50K có trên server**.

Tên đầy đủ: **Adaptive Multi-Resolution Memory for Long-Context Diffusion
Speculative Decoding**. Tên canonical trong repo là **AMR-DFlash**;
“ARM-DFlash” trong trao đổi chỉ cùng ý tưởng.

**Quyết định đã chốt:** giữ nguyên target và toàn bộ pretrained DFlash-5L;
chỉ học global pooling, một indexer theo draft block và một global gate.
Huấn luyện từ target-regenerated responses và target-feature cache 50K hiện có
trên server sau khi xác nhận tương thích, với warm-up ngắn rồi adapter-only training.

**Cập nhật artifact 09/10:** người vận hành xác nhận regenerated train/val
và target-feature cache 50K cho Qwen3-4B đã có trên server. Cache được ghi
trong run MR-DFlash ngày 28/09; audit hidden recompute ngày 05/10 của cache
trong run đó fail 32/32 mẫu train và val. Chưa có report audit mới chứng minh
cache hiện tại đã được sửa hoặc tương thích với AMR. Xem [trạng thái chi tiết](amr_dflash_training_data.md#2-tận-dụng-corpuscache-50k-hiện-có).

**Trạng thái bằng chứng:** đây là đề xuất nghiên cứu và dàn ý paper hoàn chỉnh,
chưa là báo cáo kết quả AMR. Code hiện tại vẫn là V0 preference-selector +
fixed global slots. Trainer, cache importer và teacher collector cho phương
pháp trong tài liệu này chưa triển khai; chưa có acceptance/throughput trên B200.

## Mục lục

1. [Vai trò tài liệu và quyết định cốt lõi](#1-vai-trò-tài-liệu-và-quyết-định-cốt-lõi)
2. [Paper story trong một mạch](#2-paper-story-trong-một-mạch)
3. [Tên paper, abstract và đóng góp dự kiến](#3-tên-paper-abstract-và-đóng-góp-dự-kiến)
4. [Introduction: lập luận theo từng đoạn](#4-introduction-lập-luận-theo-từng-đoạn)
5. [Background, ký hiệu và observation](#5-background-ký-hiệu-và-observation)
6. [Method: kiến trúc memory adapter](#6-method-kiến-trúc-memory-adapter)
7. [Dữ liệu training và tái sử dụng cache 50K](#7-dữ-liệu-training-và-tái-sử-dụng-cache-50k)
8. [Loss và các thành phần được train](#8-loss-và-các-thành-phần-được-train)
9. [Warm-up và training chính](#9-warm-up-và-training-chính)
10. [Inference và target verification](#10-inference-và-target-verification)
11. [Thiết kế thí nghiệm và metric](#11-thiết-kế-thí-nghiệm-và-metric)
12. [Claim, evidence và điều kiện kết luận](#12-claim-evidence-và-điều-kiện-kết-luận)
13. [Figures, tables và cấu trúc bản thảo](#13-figures-tables-và-cấu-trúc-bản-thảo)
14. [Related work và ranh giới novelty](#14-related-work-và-ranh-giới-novelty)
15. [Rủi ro, giới hạn và kết quả không có gain](#15-rủi-ro-giới-hạn-và-kết-quả-không-có-gain)
16. [Readiness và việc cần triển khai](#16-readiness-và-việc-cần-triển-khai)
17. [Sổ quyết định và cách cập nhật ý tưởng](#17-sổ-quyết-định-và-cách-cập-nhật-ý-tưởng)

## 1. Vai trò tài liệu và quyết định cốt lõi

File này là **điểm bắt đầu và nguồn chính cho ý tưởng, paper story, phạm vi
claim và recipe training**. Các tài liệu đi kèm giải thích sâu từng phần:

| Tài liệu | Vai trò |
|---|---|
| [Đặc tả kỹ thuật](superpowers/specs/2026-10-08-amr-dflash-design.md) | Contracts của memory, positions, gradients, evaluation và checkpoint |
| [Dữ liệu/training](amr_dflash_training_data.md) | Regenerate/cache, schema, audit, trạng thái dữ liệu server; phụ lục lệnh V0 |
| [Kế hoạch triển khai](superpowers/plans/2026-10-08-amr-dflash-implementation.md) | Các tasks T1–T12 và tiêu chí nghiệm thu |
| [Baseline guide](baselines/amr_dflash.md) | Phương pháp hiện tại và launcher V0 đã tồn tại |
| [Workboard](project_workboard.md) | Trạng thái điều hành workstream A2 |
| [Proposal 08/10](../src/ARMdflash/AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md) | Lịch sử V0 acceptance-preference/complementary slots |

Khi thay đổi ý tưởng, cập nhật sổ quyết định ở cuối file và đồng bộ tài liệu
đi kèm. Trạng thái code cần đối chiếu source/tests; một quyết định trong docs
không có nghĩa là chức năng đã triển khai.

| Quyết định | Phương pháp chính |
|---|---|
| Backbone | Target, embeddings, output head, DFlash projection/norm và cả năm layers frozen |
| Memory | Global summaries + selected fine raw features + exact local features |
| Tổ chức attention | Một working memory dùng chung năm layers; full live block luôn được giữ |
| Selection | Mean-pooled group keys; learned indexer chọn nhóm một lần/block, gather raw values |
| Compression | Weighted pooling nhỏ trên các nhóm liên tiếp; không có full-rank value projection mới |
| Objective | Weighted draft prediction CE + auxiliary indexer KL |
| Training | Cùng một adapter optimizer; warm-up ngắn, sau đó final-budget training |
| Evaluation | Actual greedy acceptance và committed tokens/s; full-context target verification |
| Extension | LoRA, native CSA/HCA theo layer và prefix loss xét riêng sau baseline |

## 2. Paper story trong một mạch

**Một câu về đề tài:** học cách cung cấp context ở nhiều độ phân giải cho
một pretrained diffusion drafter được giữ nguyên, để giữ target–draft
alignment với chi phí context thấp hơn trong long-context decoding.

**Problem.** DFlash dự đoán một block song song nhưng vẫn phải đọc target-derived
context dài qua các draft layers. Giảm context có thể giảm draft cost, đồng thời
làm acceptance giảm và tăng số lần target verification. Tiết kiệm attention chỉ
có giá trị khi tổng thời gian sinh token đã commit thực sự giảm.

**Observation.** Các probe trong repo cho thấy attention của DFlash phân bổ
qua prompt, committed output và live block; một phần tập trung, phần còn lại
phân tán. Thứ hạng context được ưu tiên còn khác target attention. Những quan
sát này gợi ý lựa chọn nhiều độ phân giải, chưa chứng minh memory nào giữ acceptance.

**Hypothesis.** Exact local memory giữ chi tiết gần; fine retrieval giữ nguyên
features của vùng quan trọng với draft state; global summaries bổ sung lịch sử
xa/phân tán. Một tổ hợp nhỏ, cùng feature space pretrained, có thể thích nghi
mà không cập nhật DFlash backbone.

**Method.** AMR tạo working memory chung gồm ba nhánh, rồi đưa nó vào K/V
projections có sẵn của năm DFlash layers. Indexer chạy một lần cho cả block.
Global pooling và gate học bằng draft prediction; indexer học từ compact
dense-drafter attention supervision trên một subset train nhỏ.

**Training.** Dùng response regenerate và target-feature cache đã có theo
báo cáo; audit fingerprints/parity để xác nhận có thể reuse cho AMR.
Lấy nhiều random anchors/document, train block 1 anchor + 15 proposals.
Warm-up dùng pooling gần mean và memory budget rộng; training chính giảm về
budget deploy. Chỉ cập nhật adapter nhỏ, dùng chung pipeline qua hai giai đoạn.

**Outcome cần chứng minh.** Acceptance giữ đủ tốt để committed throughput
tăng trên long context, tổng adaptation cost phù hợp mục tiêu ít tài nguyên,
short-context routing ổn định và output khớp greedy target.

**Thông điệp paper:** cấu trúc memory và khả năng tận dụng pretrained drafter
là trọng tâm. Tăng acceptance vượt dense là một kết quả đáng quan tâm nếu đo
được; giữ acceptance và giảm tổng latency đã là hướng kiểm chứng chính.

## 3. Tên paper, abstract và đóng góp dự kiến

### 3.1. Tên làm việc

Tên ưu tiên: **AMR-DFlash: Lightweight Multi-Resolution Memory Adaptation
for Frozen Diffusion Drafters**.

Tên mở rộng: **AMR-DFlash: Adaptive Multi-Resolution Memory for
Long-Context Diffusion Speculative Decoding**.

Chưa đưa “acceptance-optimal”, “training-free” hoặc một hệ số speedup vào tên.
Phương pháp có train adapter; objective chưa trực tiếp tối ưu actual acceptance.

### 3.2. Abstract dự thảo ở trạng thái đề xuất

> Pretrained block-parallel drafters có thể giảm chi phí đề xuất token trong
> speculative decoding, nhưng conditioning trên context dài đặt ra sự đánh
> đổi giữa drafting cost và target–draft alignment. Chúng tôi đề xuất
> AMR-DFlash, một memory adapter nhỏ cho pretrained DFlash được giữ nguyên.
> Adapter kết hợp summaries toàn cục, retrieval theo draft state ở độ phân
> giải gốc và exact local memory trong một working memory dùng chung các
> draft layers. Global pooling được học qua draft prediction, còn block
> indexer được học bằng auxiliary attention distillation trên một subset
> training nhỏ. Training tái sử dụng target-regenerated responses và cache
> target features 50K Qwen3-4B đã có trên server, sau khi xác nhận parity,
> với warm-up ngắn trước khi chuyển về memory budget
> triển khai. Đánh giá được thiết kế để đo actual acceptance, committed
> throughput, tính đúng của full-context verification và tổng chi phí
> adaptation. Câu hỏi thực nghiệm là liệu memory adaptation có giữ alignment
> với ít context keys hơn và tạo lợi ích về thời gian sinh trên long context.

Khi có kết quả, thay hai câu cuối bằng số đo có scope: target/draft, phần cứng,
context bins, số documents, throughput speedup và acceptance retention kèm
uncertainty. Hiện không có số AMR để điền vào abstract.

### 3.3. Ba đóng góp dự kiến

1. **Memory adaptation cho frozen block-parallel drafter:** working memory
   chung kết hợp global, fine và local context, giữ nguyên pretrained feature
   space và live-block attention.
2. **Recipe huấn luyện ít thành phần:** response prediction + compact indexer
   distillation; warm-up và training chính dùng cùng adapter optimizer,
   không cần thu preference labels cho mỗi context candidate.
3. **Đánh giá đầy đủ sự đánh đổi:** paired rollout đo acceptance/throughput,
   ablations kiểm tra từng nhánh và báo chi phí cache/teacher/training.

Đây là các đóng góp **cần được xác lập bằng thí nghiệm và related work**.
Attention diagnostic là động cơ; chi tiết pooling hoặc auxiliary KL riêng lẻ
không tự tạo novelty.

## 4. Introduction: lập luận theo từng đoạn

| Đoạn | Nội dung cần viết | Bằng chứng cần đi cùng |
|---|---|---|
| 1 — Bối cảnh | Long-document generation đòi hỏi cả alignment và latency tốt; speculative decoding phụ thuộc draft cost và tokens commit/round | Background và đo dense DFlash trên workload mục tiêu |
| 2 — Khoảng trống | Có pretrained parallel drafter nhưng adaptation cho long context cần tránh chi phí train lại backbone | Phân rã runtime và baseline checkpoint giữ nguyên |
| 3 — Đánh đổi | Chỉ bỏ context dễ làm draft sai; chỉ giữ global summaries có thể mất chi tiết | Local/fine/global interventions và rollout |
| 4 — Insight | Context có thể được phân bổ nhiều độ phân giải tùy draft state | Probe có scope rõ; Figure 1 và ablations tiếp theo |
| 5 — Phương pháp | Working memory chung, raw fine retrieval, global pooling nhỏ, frozen DFlash | Kiến trúc, parameter accounting và gradient paths |
| 6 — Training | Reuse regenerate/cache; warm-up giúp adapter thích nghi với features pretrained trước sparse budget | Cache audit, convergence/retention và cost table |
| 7 — Kết quả/đóng góp | Tóm tắt kết quả đo được và ba đóng góp ở trên | Chỉ viết ở thì đã đạt sau paired holdout |

Không mở đầu bằng kết luận “attention thừa nên bỏ đi không mất gì”. Không
khẳng định verification là bottleneck theo một tỷ lệ cố định khi chưa có
profiling đúng target/GPU/workload. Nếu draft chiếm phần nhỏ trong runtime,
phải trình bày trần lợi ích của việc chỉ giảm draft cost.

## 5. Background, ký hiệu và observation

### 5.1. Thiết lập bài toán

Gọi target là $T$, pretrained drafter là $D_{\theta_0}$. Trong thiết kế
này, cả hai đều frozen. Tại một round, target đã xử lý prefix dài $N$;
anchor $x_a$ đã biết nhưng chưa đưa feature của anchor vào context bank.
Block size $B=16$: một anchor và tối đa 15 proposal positions.

Target features từ các layer được checkpoint chọn được nối và project:

$$
H=\operatorname{hidden\_norm}
  (\operatorname{fc}(\operatorname{concat}(F^{(\ell_1)},\ldots,F^{(\ell_s)})))
  \in\mathbb R^{N\times d}.
$$

Đọc layer IDs từ checkpoint; không mặc định một danh sách cho mọi target.
Đây là **target-derived context features của drafter**, khác raw target KV.

DFlash dùng target features làm context K/V và dự đoán block song song.
Training gốc dùng random response anchors, các proposal được mask và blocks
được cô lập khỏi nhau. AMR kế thừa cách tạo block, nhưng giữ cả draft backbone
frozen trong adaptation. [DFlash, §4](https://arxiv.org/html/2602.06036v1#S4).

Để giải thích economics của phương pháp, xét trung bình các rounds trong
cùng timing scope:

$$
\operatorname{TPS}\approx
\frac{\mathbb E[G_{\rm out}]}
 {\mathbb E[T_{\rm memory}+T_{\rm draft}+T_{\rm verify}+T_{\rm other}]}.
$$

AMR có thể giảm draft cost nhưng tăng memory overhead hoặc làm giảm tokens
commit/round. Vì vậy cần đo cả tử số và mẫu số. Trong mô hình đơn giản giữ
acceptance và các chi phí khác cố định, nếu draft chiếm tỷ lệ f của round time
thì ngay cả loại bỏ toàn bộ draft cost cũng chỉ cho trần speedup 1/(1-f).
Đây là phép phân tích đánh đổi, không phải dự đoán gain thực tế của AMR.

| Ký hiệu | Ý nghĩa trong AMR |
|---|---|
| $H_{<a}$ | Features của prefix được biết trước anchor, chỉ positions nhỏ hơn a |
| $m,K$ | Fine group size và số nhóm tối đa được chọn |
| $c,L$ | Global group size và số local raw tokens |
| $G_\eta$ | Global pooled features với compressor parameters η |
| $I_\phi$ | Block indexer với parameters φ |
| $b_g$ | Scalar additive attention bias cho global keys |
| $M_a$ | Working memory được xây cho anchor a |
| $A_{\rm raw}, A_{\rm committed},G_{\rm out}$ | Matched proposals, accepted proposals được emit, tổng token commit thực tế |

### 5.2. Các quan sát đã có trong repo

Probe ngày 06/10 gồm **10 GovReport documents, 40 generations** ở bốn mức
prompt 3K/5K/8K/16K trên Modal L40S. Tại 16K, mean attention mass là:

| Vùng/đại lượng | Số đo diagnostic | Ý nghĩa giới hạn |
|---|---:|---|
| Prompt | 58,80% total mass | Lịch sử xa vẫn nhận attention |
| Output đã commit | 19,85% total mass | Cần xét cả output context, không chỉ prompt |
| Live draft block | 21,35% total mass | Có động cơ giữ nguyên live-block attention |
| Top-1K mọi keys | 67,29% total mass | Top-K đơn thuần không bao phủ mọi mass |
| Top-1K trong prompt | 46,08% **prompt mass** | Mẫu số khác dòng trên |

Các số là document-averaged diagnostic trên cohort nhỏ và backend
instrumented, không phải kết quả AMR. [Protocol và số đo](experiments/2026-10-06_dflash_attention_10_instances_full_mass.md).

Trong phép đo paired trên cohort đó, overlap Top-1K cache positions của
target và DFlash ở 16K là **19,8%**. Nó cho thấy hai bảng xếp hạng attention
khác nhau trong protocol đã đo, chưa chứng minh dùng drafter teacher tốt hơn
target teacher. [So sánh target–DFlash](experiments/2026-10-06_dflash_target_attention_comparison.md).

**Suy luận để thiết kế thí nghiệm:** cần global branch để thử xử lý thông tin
phân tán, fine branch để giữ chi tiết, và state-conditioned selection để thử
thích nghi từng round. Causal utility phải đo bằng interventions và acceptance.

## 6. Method: kiến trúc memory adapter

### 6.1. Một working memory, ba độ phân giải

~~~mermaid
flowchart LR
    T["Frozen target: prefix features"] --> P["Frozen DFlash projection + norm"]
    P --> G["Global: learned weighted pooling"]
    P --> F["Fine groups: fixed mean keys"]
    P --> L["Local: exact recent features"]
    Q["Anchor embedding + latest/recent features"] --> I["Learned block indexer"]
    F --> I
    I --> R["Top-K groups: gather raw features"]
    G --> M["Shared working memory"]
    R --> M
    L --> M
    M --> D["Frozen DFlash-5L + full live block"]
    D --> B["15 parallel proposals"]
    B --> V["Full-context target verification"]
    V --> C["Commit prefix; update memory"]
~~~

Ba nhánh tồn tại trong cùng attention memory. Global summaries có thể bao
phủ thông tin đã có ở fine/local; chồng lấp này là có chủ đích. Tên “complementary”
của V0 không còn hàm ý một phép chia disjoint context trong phiên bản này.

### 6.2. Global branch: nén mạnh, giá trị cùng feature space

Chia absolute timeline thành nhóm c tokens, căn theo position 0. Chỉ dùng
nhóm hoàn tất trong prefix; tail chưa đủ c tokens được exact local giữ vì
$L\ge c$. Với nhóm $B_b$:

$$
e_i=w_2^\top\operatorname{SiLU}(W_1h_i),\qquad
g_b=\sum_{i\in B_b}\alpha_i h_i,\qquad
\alpha_i=\frac{\exp(e_i)}{\sum_{k\in B_b}\exp(e_k)}.
$$

MLP có kích thước $d\to r_C\to1$, khởi đầu $r_C=32$.
Values vẫn là H; không thêm projection $d\to d$ mới.
Số entries là $\lfloor N/c\rfloor$, không phải 128 slots cố định.
Pooling dùng FP32/log-sum-exp ổn định rồi cast về dtype drafter.

Khởi tạo final pooling weight bằng zero, tầng trước bình thường, để pooling
ban đầu uniform. Global keys nhận finite logit bias $b_g=-4$, có thể học;
không triệt tiêu nhánh bằng hard zero khiến compressor mất gradient.
Ở bước đầu, gradient của tầng W1 có thể bằng zero do final weight bằng zero;
kiểm tra compressor học qua output weight và các bước tiếp theo, tránh coi
mọi tensor đều phải có nonzero gradient ngay tại initialization.

### 6.3. Fine branch: index nhóm nhẹ, đọc raw features

Chia prefix thành nhóm m tokens và lấy mean cố định $u_b$.
Indexer key $k_b=W_k u_b$, với dimension $r_I=64$. Query:

$$
z_a=[E(x_a);h_{a-1};\operatorname{mean}(H_{a-16:a})],\qquad
q_a=\operatorname{MLP}_\phi(z_a),\qquad
s_{a,b}=\frac{q_a^\top k_b}{\sqrt{r_I}}.
$$

Recent window được clip theo prefix thực tế. Query có context ngay trước
layer đầu, thay vì dùng các mask embeddings giống nhau làm query nội dung.
Một query chọn support cho cả block và cả năm layers.

Eligible groups phải hoàn tất trước anchor và nằm hoàn toàn ngoài local
window. Lấy $\min(K,\#\text{eligible})$ nhóm có score cao nhất; **gather
tất cả raw H của nhóm được chọn**, không dùng mean u_b làm attention value.
Union với local, loại trùng raw positions và sort theo absolute positions.

Mean pooling ở fine branch chỉ phục vụ index keys. Learned scorer không nhận
gradient từ phép hard Top-K/gather; auxiliary loss ở §8 giải quyết điểm này.

### 6.4. Local branch và attention pretrained

Giữ tối đa L raw features gần anchor, cùng original positions. Working memory:

$$
M_a=[R_{\rm fine}\cup R_{\rm local};G_\eta].
$$

Mỗi draft layer dùng lại K/V projections và k_norm của checkpoint:

$$
O_\ell=\operatorname{Attn}
 (Q_\ell,[KV_\ell(M_a);KV_\ell(\text{live block})]).
$$

Raw K dùng original RoPE positions. Global K dùng integer midpoint của nhóm,
rotate sau pooling/projection; V không rotate. Global bias chỉ áp cho global
keys. Đây là quy ước approximation cần ablate, không tương đương dense attention.
Toàn bộ live block vẫn bidirectional; các training blocks khác nhau bị cô lập.

### 6.5. Budget và chi phí phải đo

$$
B_{\rm context}\le mK+L+\lfloor N/c\rfloor.
$$

Ví dụ N=16.384, m=4, K=256, L=128, c=64: tối đa **1.408 context keys**,
hoặc **1.424 keys** khi tính thêm live block 16. Đây là key-count bound,
chưa phải hệ số speedup, giảm FLOPs toàn hệ thống hoặc giảm VRAM.

Indexer vẫn score các eligible groups; global attention vẫn tăng theo N/c.
Prefill/build, projections, Top-K, gather, incremental update, target verification
và resident feature/KV banks đều phải vào profiling. Cache nhóm hoàn tất giúp
tránh rebuild toàn bộ mỗi round nhưng vẫn có build cost ban đầu.

### 6.6. Liên hệ HCA/CSA

DeepSeek-V4 kết hợp HCA nén mạnh với CSA chọn compressed entries và bố trí
hai cơ chế theo layers. AMR lấy cảm hứng từ nguyên lý nhiều độ phân giải,
nhưng dùng target-derived features, memory chung và **raw fine retrieval**.
Không mô tả AMR như việc transplant nguyên bản attention V4.
[DeepSeek-V4, §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3).

## 7. Dữ liệu training và tái sử dụng cache 50K

### 7.1. Đầu vào cần có

| Thành phần dữ liệu | Dùng để làm gì | Bắt buộc/điều kiện |
|---|---|---|
| Prompt đầy đủ và final target-regenerated response | Tạo clean sequence và proposal labels | Bắt buộc; giữ lịch sử assistant trước final response |
| Token IDs, response boundary, EOS/valid mask, absolute positions | Lấy anchors và tránh off-by-one/leakage | Bắt buộc, cùng template/tokenizer/truncation |
| Source document/conversation ID và split | Chống train–validation–holdout overlap | Bắt buộc, audit full-history/content |
| Raw target features hoặc projected H cache | Tránh chạy lại target cho mỗi batch | Cache target Qwen3-4B 50K đã có theo xác nhận 09/10; chỉ reuse cho AMR khi fingerprint và recompute audit khớp |
| Compact dense-DFlash group attention teacher | Huấn luyện indexer | Chỉ subset train; anchors/visibility cố định |
| Reference summary gốc | ROUGE khi inference | Tùy dataset; không thay target-regenerated labels |

Theo xác nhận 09/10, regenerated train/val và target-feature cache của run
50K đã có trên B200 với Qwen3-4B. Đường dẫn đã khai báo trong config MR-DFlash:

~~~text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/regenerated_full/{train,val}.jsonl
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/target_features_qwen3_4b_full/{train,val}
~~~

Đây là **artifact đã tồn tại**, không đồng nghĩa nó đã qua contract audit
của AMR. Có thể tái sử dụng response nếu schema, tokenizer/template, split,
boundary và target provenance khớp; không cần autoregressive generate lại
toàn bộ 50K chỉ để bắt đầu trainer mới.

### 7.2. Cache audit trước reuse

Khóa target/draft snapshot, tokenizer/template, input IDs, layer selection,
position/boundary/truncation, dtype policy và projection fingerprint. Recompute
subset từ clean tokens để so sánh từng cache level theo tolerance đã ghi.
Kiểm tra quy ước hidden_states index của layer, không chỉ shape.

Raw target feature cache đi qua frozen fc/norm một lần. Projected H cache
không project lần hai. Cache không khớp cần rebuild bằng **causal teacher forcing
trên clean sequence**, thay vì chạy autoregressive capture từng trajectory.
Người vận hành xác nhận cache 50K từ Qwen3-4B đang có trên server. Tuy vậy,
audit ngày 05/10 của cache run 28/09 fail hidden recompute 32/32 train và
32/32 validation; chưa có report audit mới sau lần fail đó. Vì vậy **cache
tồn tại, tương thích snapshot/AMR chưa được xác nhận**. Importer mới cũng chưa
triển khai. Giữ nguyên nguồn, kiểm fingerprints rồi audit lại trước khi reuse.

### 7.3. Một anchor thành training block như thế nào

Với token sequence x và anchor response position a:

~~~text
Context được phép:       H[0:a]       (positions 0 ... a-1)
Live input:             x[a], MASK, ..., MASK       (16 positions)
Proposal labels:        x[a+1], ..., x[a+15]        (tối đa 15)
Loss mask:              hợp lệ tới EOS/response end
Memory/teacher access:  chỉ prefix trước a; không dùng future features
~~~

Có thể tính features của toàn clean sequence một lần bằng causal target,
nhưng phải slice/mask theo anchor **trước pooling và selection**. Một summary
chứa future token vẫn là leakage dù midpoint của nó nằm trước anchor.

Sample nhiều anchors/document để tận dụng feature bank. Training anchors
đổi theo seed/epoch; validation anchor manifest cố định. Khi pack nhiều blocks,
không cho block này thấy live K/V, anchor hoặc future memory của block khác.
Training adapter lấy anchors có AMR active. Blocks không có valid proposals
hoặc chỉ đi dense bypass, không có đường gradient tới adapter, được skip/log
reason; short-context parity được kiểm tra riêng trong evaluation.

### 7.4. Teacher nhỏ dành cho indexer

Chọn subset **train-only** và anchors cố định. Dense frozen DFlash chạy với
cùng clean prefix, anchor và 15 masks. Aggregate attention thành distribution
trên eligible fine groups rồi lưu group IDs/probabilities và fingerprints.
Không lưu full attention matrices hoặc logits toàn vocabulary cho mọi mẫu 50K.

Indexer warm-up/main phải có teacher-labeled anchors thực tế; điểm bắt đầu
là 25% anchors trong batch khi đủ dữ liệu. Anchors khác chỉ có prediction loss.
Random anchors mới không được gắn teacher của một anchor khác.

### 7.5. Dữ liệu server hiện có: điều đã biết và chưa biết

Theo cập nhật người vận hành ngày 09/10, run 50K trên server có
`regenerated_full/{train,val}.jsonl` và raw target-feature cache
`target_features_qwen3_4b_full/{train,val}` từ Qwen3-4B. Đường dẫn đầy đủ nằm
ở §7.1. Cache tồn tại; audit compatibility được ghi riêng ở §7.2.

Terminal report cùng ngày cho biết validation cleaning giữ 2.399 records,
bỏ 101 content overlaps. V0 prepare chọn 200 train + 50 validation documents
cho run **amr_pilot_20261009T180405Z**:

| Split | ArXiv được chọn | ShareGPT được chọn | Tổng |
|---|---:|---:|---:|
| Train | 100 | 100 | 200 |
| Validation | 44 | 6 | 50 |

Prompt tokens sau truncation nằm trong **4.140–10.234**, mean **6.473,772**.
Cap 16.384 không chứng minh có samples gần 16K. Eligible long ShareGPT rất
ít: 139 train và 6 validation; không mặc định validation long-context cân bằng.

Các số trên xác nhận prompt preparation V0 theo báo cáo terminal. Chúng chưa
xác nhận response/cache contract, labels hoặc trainer mới. Path và chi tiết
audit nằm trong [hướng dẫn dữ liệu](amr_dflash_training_data.md).

## 8. Loss và các thành phần được train

### 8.1. Prediction objective

Cho proposal distribution q ở block b, offset j và valid mask v:

$$
\mathcal L_{\rm draft}=
\frac{\sum_{b,j}v_{b,j}w_j[-\log q_{b,j}(y^*_{b,j})]}
     {\sum_{b,j}v_{b,j}w_j},\qquad
w_j=\exp(-(j-1)/\gamma),\quad j=1,\ldots,15.
$$

Khởi đầu γ=7, nhấn mạnh lỗi sớm trong block. DFlash đã dùng exponentially
weighted prediction CE; đây là objective được kế thừa, không là đóng góp mới.
[DFlash, training và appendix](https://arxiv.org/html/2602.06036v1#A1.SS1).

CE train compressor/gate thông qua frozen drafter. Nó là surrogate cho
acceptance: token prediction loss thấp hơn chưa đảm bảo greedy prefix dài hơn
hoặc throughput cao hơn trên rollout.
Batch có tổng valid weight bằng zero phải skip với reason trước khi chia;
không tạo loss/metric NaN hoặc báo một optimizer step không có supervision.

### 8.2. Indexer objective: vì sao cần một auxiliary loss

Hard Top-K tạo discrete indices; CE không backprop thông thường qua indices
vào scorer. Với attention teacher $A^{\rm dense}$, eligible group set
$\mathcal E_a$, định nghĩa:

$$
u_{a,b}=\sum_{\ell,h,j}v_{a,j}w_j
                 \sum_{i\in B_b}A^{\rm dense}_{\ell,h,j,i},\qquad
t_{a,b}=\frac{u_{a,b}}{\sum_{b'\in\mathcal E_a}u_{a,b'}}.
$$

Aggregation dùng cùng early-position weights, loại local/live/ineligible
groups trước normalization. Empty domain hoặc zero denominator được skip
với reason. Không tạo uniform teacher giả rồi báo có supervision.

$$
\mathcal L_{\rm idx}
=\operatorname{mean}_{a\in\mathcal A_{\rm teacher}}
  D_{\rm KL}\left(\operatorname{stopgrad}(t_a)
  \;\|\;\operatorname{softmax}(s_\phi(a,\mathcal E_a))\right).
$$

Attention teacher là **allocation proxy của dense drafter**, chưa là
acceptance utility. DSA có tiền lệ warm-up indexer bằng attention KL; cách
gộp groups/layers/proposal queries ở đây là lựa chọn riêng cần kiểm chứng.
[DeepSeek-V3.2, §2.1.1](https://arxiv.org/html/2512.02556v1#S2.SS1.SSS1).

### 8.3. Một optimizer, hai đường gradient

$$
\boxed{\mathcal L_{\rm main}=\mathcal L_{\rm draft}+\beta\mathcal L_{\rm idx}}
$$

β=1 là điểm bắt đầu pilot, cần theo dõi magnitude và validation. Nếu một
batch không có teacher hợp lệ, chỉ tính CE và log số supervised anchors bằng zero.

| Thành phần | Train? | Gradient đến từ đâu |
|---|---|---|
| Target + token embeddings + output head | Frozen | Không có weight updates |
| DFlash fc/norm + attention/FFN của năm layers | Frozen | Truyền input gradients, không cập nhật weights |
| Fine mean pooling/raw values | Cố định | Không có parameters mới |
| Global pooling MLP | Có | CE qua global memory và frozen DFlash |
| Global attention bias | Có | CE qua attention logits |
| Indexer key/query projections | Có | Auxiliary KL trực tiếp trên group scores |

“Joint training” ở đây là cùng batch/pipeline/optimizer cập nhật adapter.
Không diễn giải thành mọi module đều nhận gradient từ một actual-acceptance loss.
Indexer thay support mà compressor/drafter đọc, nhưng supervised scorer và
pooling vẫn có gradient paths khác nhau.

Frozen weights không đồng nghĩa toàn forward được no_grad. Target feature
extraction/teacher collection có thể detach; **draft và output-head forward
phải giữ autograd** để gradient về global pooling/gate. Optimizer chỉ chứa
parameters mới; assert frozen hashes giữ nguyên.
Packing/checkpoint lưu contracts của adapter, optimizer/scheduler, RNG/sampler,
phase/step/budget schedule, model/data/cache/teacher fingerprints và eval history.
Không cần save lại frozen model weights; checkpoint V0 có schema khác.

### 8.4. Loss mở rộng, mặc định tắt

Một prefix surrogate có thể thử sau baseline:

$$
\mathcal L_{\rm prefix}=-\frac{1}{J_b}
       \sum_{k=1}^{J_b}\prod_{j=1}^{k}q_{b,j}(y^*_{b,j}),
$$

với J_b là valid proposal length, rồi average qua valid blocks.
Tính qua cumulative log probabilities để ổn định. Product có thể nhỏ làm
gradient yếu; nó không bằng actual greedy acceptance. Chain/prefix reward
đã có trong [Teaching Diffusion to Speculate Left-to-Right, §5.5](https://arxiv.org/html/2606.11552v1#S5.SS5).
Nếu thử, dùng λ=0 làm control và cùng objective cho dense adaptation control.

Dense output/hidden distillation trên subset nhỏ cũng là ablation warm-up,
mặc định tắt vì cần teacher pass thêm. Main recipe chỉ CE + indexer KL.

## 9. Warm-up và training chính

### 9.1. Mục tiêu warm-up

Warm-up giúp **adapter mới** thích nghi với frozen pretrained feature space
và học ranking trước khi memory budget nhỏ. DFlash không bị khởi tạo lại.
Global pooling ban đầu uniform, fine/local values là features gốc, global
gate finite/conservative; indexer nhỏ vẫn cần initialization và supervision.

Budget rộng giảm mức độ thay đổi conditioning ban đầu. Nó chưa đảm bảo
performance bằng dense: pooled entries và global bias vẫn thay đổi attention.
Kiểm tra dense bypass identity riêng, rồi đo retention của AMR forced.

### 9.2. Các giai đoạn và điểm chuyển

| Phase | Dữ liệu và việc làm | Parameters/loss | Điều kiện chuyển |
|---|---|---|---|
| 0 — Audit + dense baseline | Response/cache/splits audit; baseline paired và token parity | Không train | Có artifacts hợp lệ và dense reference |
| 1 — Adapter warm-up | Teacher subset + CE blocks, pooling uniform, budget rộng | Chỉ pooling/indexer/gate; CE + β KL | Gradients/metrics hợp lệ; retention được đo |
| 2 — Main memory training | Cùng dataset/sampler, final budget, tiếp tục teacher quota | Cùng optimizer và loss | Checkpoint chọn trên validation acceptance/cost |
| 3 — Calibration + holdout | Đo dense/AMR crossover; khóa routing/config | Không train | Full paired holdout report |

Warm-up pilot tối đa **200 optimizer steps**, khởi đầu c=32,
K≤1.024; chuyển qua K=512 rồi K=256, c về 64 ở các điểm validation đã ghi.
Schedule cụ thể và checkpoint tại phase boundary phải được lưu để resume.
Không dùng warm-up vô hạn để bù một thiết kế memory không đạt retention.

Main pilot đầu đo **200 steps** trên 2–5K eligible train documents,
validation 50–100 documents khóa riêng. Đây là kiểm tra khả thi, không khẳng
định hội tụ trong 200 steps. Chỉ mở run 50K sau runtime/acceptance/cost checks.

### 9.3. Cấu hình khởi đầu cho pilot

| Nhóm | Giá trị dự kiến, chưa benchmark |
|---|---|
| Deploy memory | m=4, c=64, K=256, L=128; L≥c |
| Small modules | r_I=64, r_C=32; query recent window 16 |
| Initialization | Uniform global pooling; global bias -4; không reset backbone |
| Anchors | 8 response anchors/document mỗi lượt; tăng sau profile |
| Objective | γ=7, β=1; prefix/output distillation mặc định tắt |
| Optimizer | AdamW LR 1e-4, weight decay 0,01; gradient norm clip 1 |
| Teacher quota | 25% anchors có compact teacher khi đủ dữ liệu |
| Prediction validation | Toàn fixed validation anchor manifest mỗi 50 steps |
| Rollout validation | Toàn validation documents mỗi 500 steps và cuối phase |
| Reproducibility | Seed 17; khóa sampler, dataset/cache/teacher/model hashes |

Các giá trị này là **recipe đề xuất**, chưa là flags/YAML được CLI V0 hỗ trợ.
200-step pilot vẫn chạy phase-end rollout dù chưa đến cadence 500.

Gate warm-up đề xuất là mean accepted-proposal retention ≥95% của dense
trong từng nonzero-dense context bin. Nếu dense acceptance bằng zero, báo
absolute difference. Đây là pilot criterion, không phải kết quả hoặc cam kết
đạt 100% performance. Final-budget AMR phải được đánh giá lại đầy đủ.

### 9.4. Sơ đồ training và thuật toán

~~~mermaid
flowchart TD
    A["50K regenerate + cache"] --> B["Audit response, tokens, features, splits"]
    B --> C["Clean sequences + shared feature store"]
    C --> D["Random train anchors / fixed validation anchors"]
    C --> E["Train-only subset: dense attention teacher"]
    D --> W["Warm-up: wide-budget adapter"]
    E --> W
    W --> M["Main: final-budget adapter, same optimizer"]
    E --> M
    M --> V["Prediction validation + actual greedy rollout"]
    V --> K["Select checkpoint; calibrate routing"]
    K --> H["Locked paired holdout + total cost report"]
~~~

Pseudocode mô tả thuật toán, **không phải API/CLI đã hiện thực**:

~~~text
load target and pretrained DFlash; freeze all existing weights
initialize small pooling, block indexer and global gate
audit dataset/cache; prepare train-only compact attention teachers
build one optimizer containing only new adapter parameters
for phase in [warmup, main]:
    for batch in mixed_anchor_sampler:
        read cached prefix features; enforce anchor visibility
        build grouped global + selected raw fine + exact local memory
        predict proposals with frozen DFlash, autograd enabled
        loss = weighted_CE + beta * indexer_KL_on_valid_teacher_anchors
        backward; clip adapter gradients; optimizer_step
        run full fixed-anchor validation at prediction cadence
        run full validation rollout at rollout cadence
        save adapter + optimizer + sampler/RNG + phase/budget/contracts
    run phase-end rollout; apply recorded budget transition
select checkpoint on validation; calibrate and lock routing
evaluate locked holdout with no parameter updates
~~~

### 9.5. Vì sao có thể nhẹ hơn, và chi phí còn lại

Số parameters mới có bậc $4dr_I+r_I^2+dr_C$, cộng pooling output/bias/gate.
Đếm số thực tế khi triển khai. Frozen backbone không có optimizer state hay
weight gradients; cached target features tránh chạy target mỗi training step.
Một lần index/block và shared memory tránh năm indexer/adapter riêng.

Vẫn cần forward/backward qua năm draft layers để train compressor.
Attention teacher collection, cache I/O và vocabulary head có thể chiếm nhiều
thời gian/VRAM. Dùng lazy loader, bounded RAM cache, anchor chunks và CE/logit
chunks; không nhân toàn feature bank hoặc logits cho từng anchor vô hạn.
Ít trainable parameters không tự chứng minh ít GPU-hours hay train nhanh hơn.

## 10. Inference và target verification

1. Target prefill full prompt; tạo pending anchor và processed feature bank.
2. Nếu routing chọn dense, dùng DFlash gốc; nếu chọn AMR, dùng grouped memory.
3. DFlash đưa ra tối đa 15 parallel proposals với full live block.
4. Target full-context verifier xác định matched prefix và correction/bonus.
5. Commit theo EOS/output cap; crop rejected target/draft state đúng contract.
6. Append chỉ target features đã xử lý và commit; cập nhật completed groups,
   index keys và local tail trước round tiếp theo.

Không append features của rejected proposals; pending anchor chỉ trở thành
processed feature khi target thực sự xử lý. Mỗi completed global group được
cache khi weights cố định; load/update adapter yêu cầu rebuild cache.
Training pooling dùng graph mới, không reuse detached inference accumulators.

Short-context gate là **calibration theo cost**, không cần thêm neural gate
training trong recipe chính. Khóa threshold theo GPU/backend/dtype/batch/checkpoint;
report AMR forced và routed AMR riêng để biết gain đến từ memory hay bypass.

Greedy target token parity là yêu cầu correctness. ROUGE khi có reference
là quality/sanity check, không dự kiến tăng chất lượng vượt chính target khi
output được full-context verifier giữ đúng contract.

## 11. Thiết kế thí nghiệm và metric

### 11.1. Research questions

| Câu hỏi | Thí nghiệm trả lời |
|---|---|
| RQ1 — Có tăng tốc thực tế với frozen backbone? | Dense vs final-budget AMR forced, paired full rollout, latency decomposition |
| RQ2 — Ba độ phân giải có bổ trợ? | Local+fine, local+global và full adapter; kiểm soát budget |
| RQ3 — Learning có cần thiết? | Mean vs learned pooling; strong fixed/random vs learned indexer |
| RQ4 — Adaptation có ít tài nguyên? | Params, cache/teacher/train/validation GPU-hours và convergence cùng compute |
| RQ5 — Warm-up có giúp? | Có/không warm-up với tổng training compute tương đương |
| RQ6 — Giữ tính đúng và short-context behavior? | Target greedy parity, dense bypass identity, short-context routed benchmark |

### 11.2. Workload và fairness

Khởi đầu Qwen3-4B + Qwen3-4B-DFlash-b16, một B200, greedy, batch size 1,
block 16. So cùng target/draft snapshots, tokenization/template, generation
cap, dtype/backend và timing scope. Cache identity và memory build cost được ghi.

Split theo source document/conversation; khóa validation và holdout trước
train. Báo context length **thực tế**, source coverage và output lengths.
Đo short bins và long bins tới 16K khi đủ data; không lấp một bin trống bằng
cap config. Generalization ngoài training distribution cần benchmark held-out
long-document summarization và, nếu đủ tài nguyên, target/draft pair thứ hai.

Các subset thuộc một nguồn phải đủ số documents; không coi nhiều anchors
cùng document là nhiều observations độc lập. Config/checkpoint/routing chọn
trên validation, khóa trước holdout.

### 11.3. Metric contract

$$
\operatorname{TPS}_{\rm decode}=
\frac{\sum_d G_{{\rm decode},d}}{\sum_d T_{{\rm decode},d}},\qquad
\operatorname{Speedup}_{\rm AMR/dense}=
\frac{\operatorname{TPS}_{\rm AMR}}{\operatorname{TPS}_{\rm dense}}.
$$

Primary metric dùng tổng token/tổng thời gian, không lấy nghịch đảo mean TPOT.
Numerator/denominator cùng decode scope; khai báo cách xử lý initial token,
EOS, cap và TTFT. Timing chuẩn có GPU synchronization phù hợp, warm runs
và repetitions; instrumented attention collector không làm benchmark backend.

| Metric | Phải phân biệt |
|---|---|
| A_raw | Số proposal match liên tiếp, tối đa 15; không gồm anchor |
| A_committed | Proposal thực sự được emit sau EOS/output-cap clipping |
| G_out | Token commit thực tế, gồm correction/bonus theo contract; không luôn A+1 |
| Round count/commits per round | Cơ chế vì sao latency tăng/giảm |
| Draft/memory/verify latency | Bao gồm build, index, gather, projection, update và verifier |
| TTFT, decode, E2E | Tách prefill/build và toàn request; báo cả ba scopes |
| Resident/peak memory | Target KV, raw bank, per-layer KV, grouped cache, activations |
| Adaptation cost | Audit/recompute cần thiết + teacher + train + validation/calibration GPU-hours |

Prediction prefix-match trên teacher response chỉ là actual greedy acceptance
khi regenerate/target/decoding alignment đã xác minh. Sampled responses cho
teacher-token match proxy; **actual greedy rollout vẫn bắt buộc**.

### 11.4. Baselines và ablations tối thiểu

| Cấu hình | Mục đích và control |
|---|---|
| Target autoregressive | Mốc reference/parity và speedup so target |
| Dense pretrained DFlash | Baseline chính; cùng frozen checkpoint |
| Local + fixed fine selection | Baseline rẻ; có recency/heuristic mạnh, không chỉ random |
| Local + learned fine, bỏ global | Giá trị bổ sung của global branch |
| Local + global, bỏ learned retrieval | Giá trị bổ sung của fine branch |
| Full AMR | Phương pháp chính |
| Full AMR với mean global pooling | Learned compressor có hơn pooling cố định? |
| Fixed/random/learned indexer cùng support budget | Giá trị của learned scorer |
| Raw selected values vs compressed fine values | Tác dụng giữ pretrained feature geometry |
| Warm-up vs direct final-budget training | Hiệu quả initialization/schedule, matched total compute |
| Routed vs forced AMR | Tách crossover gate khỏi intrinsic method gain |

Budget sweep K/c/L chỉ dùng validation để chọn điểm deploy. Báo
**matched-total-key** theo entries thực tế và **matched-compute** riêng,
không đồng nhất hai phép so. Nhánh bị bỏ cần có control tái phân bổ budget,
tránh kết luận bổ trợ chỉ vì full adapter có nhiều keys hơn.

Prefix loss, dense distillation, LoRA hoặc layer-specific selection là
extensions. Nếu mở, thêm baseline cùng objective/trainable budget/compute;
không trộn vào kết quả main frozen-memory recipe.

### 11.5. Validation trước full experiment

- **CPU/static:** causal visibility, groups/positions, packed-block isolation,
  gradient tới đúng adapter, frozen-weight identity, resume/schema accounting.
- **B200 runtime ngắn:** khoảng năm phút real data, loss/gradients hữu hạn,
  data loader và checkpoint/evaluator chạy được; chưa là kết luận khoa học.
- **Pilot:** 2–5K train, locked validation, đo retention và cost ở final budget.
- **Holdout:** paired documents, repeated timing, document bootstrap CI;
  report cả failures/coverage, không chỉ aggregate đẹp.

Evaluator phải có progress riêng, phase start/end, heartbeat ít nhất mỗi
60 giây, counts và elapsed time. Missing/corrupt checkpoint, empty data, NaN,
wrong fingerprint hoặc incomplete evaluation phải báo failure. Partial artifact
không được trình bày thành full-validation summary.

## 12. Claim, evidence và điều kiện kết luận

| Claim dự kiến | Evidence cần có | Trạng thái 09/10 |
|---|---|---|
| C1 — Adapter mới không train lại DFlash | Trainable inventory, optimizer assertions, frozen hashes trước/sau | Thiết kế đã chốt; trainer mới pending |
| C2 — Giữ alignment tại reduced context budget | Paired actual acceptance, per-bin retention, final-budget forced AMR | Chưa đo |
| C3 — Tăng committed throughput | Full decode/E2E timing, memory cost đầy đủ, paired holdout CI | Chưa đo |
| C4 — Global/fine/local bổ trợ | Component ablations với budget/compute controls | Chưa đo |
| C5 — Adaptation ít tài nguyên | Total GPU-hours, wall time, cache/teacher cost và learning curves | Chưa đo |
| C6 — Giữ target correctness/short behavior | Greedy token parity, dense identity và short-context routed cost | Cần kiểm chứng thiết kế mới |
| D0 — Attention có cả tập trung/phân tán | Probe 10 GovReport và paired target comparison | Diagnostic đã có; scope nhỏ |

Pilot goals đề xuất: token parity 100%, accepted-proposal retention ≥95%
trong bins có dense A>0, measured decode speedup >1. Các ngưỡng này dùng
để quyết định tiếp tục nghiên cứu; scientific claim phải kèm uncertainty và
phạm vi samples/hardware. Retention 95% không tự đảm bảo throughput gain.

Chỉ nói “giữ nguyên backbone” khi weight audit qua. Chỉ nói “nhanh hơn” khi
actual rollout có total timing. Chỉ nói “train nhanh/chi phí thấp” khi đã tính
cả preprocessing và teacher collection. Nếu CI speedup chưa đủ rõ, kết luận
chưa đủ bằng chứng thay vì chọn một lần chạy thuận lợi.

## 13. Figures, tables và cấu trúc bản thảo

### 13.1. Figures/tables dự kiến

| Artifact | Nội dung | Vị trí/lập luận |
|---|---|---|
| Figure 1 | Pipeline ba memory branches; freeze/train coloring; verifier | Tổng quan method ngay introduction |
| Figure 2 | Attention diagnostic theo vùng/state/layer, scope và denominator | Motivation; không trình bày thành acceptance |
| Figure 3 | Training anchor mask, teacher path và CE/KL gradient paths | Training/causality dễ kiểm tra |
| Figure 4 | Acceptance–throughput Pareto theo context bin và budgets | Main evidence; đánh dấu dense baseline |
| Figure 5 | Warm-up/main learning curves: CE, KL, actual retention, GPU-hours | Khả thi và adaptation cost |
| Figure 6 | Stacked draft/memory/verify latency + failure cases | Giải thích gain hoặc bottleneck |
| Table 1 | Main holdout: throughput, A_raw/A_committed/G, TTFT/E2E, memory | Kết quả chính |
| Table 2 | Component/learning ablations cùng controls | Kiểm chứng giả thuyết nhiều độ phân giải |
| Table 3 | Params và cost audit/recompute/teacher/train/eval | Evidence cho lightweight adaptation |
| Table 4 | Source/length/split coverage, parity và failures | Tính đầy đủ, robustness |

Chưa tạo plots kết quả AMR khi chưa có artifacts. Không đưa các con số budget
lý thuyết vào bảng “measured speedup”. Figures paper dùng artifact xuất được,
caption ghi sample size, aggregation, timing scope và uncertainty.

### 13.2. Dàn ý bản paper hoàn chỉnh

1. **Abstract:** problem → method → training → kết quả đã đo → scope.
2. **Introduction:** bảy đoạn ở §4, kết thúc bằng contributions.
3. **Background and Motivation:** DFlash/verification, context-cost tradeoff,
   diagnostics và hypothesis có thể bác bỏ.
4. **AMR-DFlash:** global/fine/local; shared pretrained attention; positions,
   memory accounting và streaming lifecycle.
5. **Lightweight Adaptation:** data/cache audit; anchor blocks; CE/KL và
   gradient table; initialization/warm-up/final-budget schedule.
6. **Experimental Setup:** datasets/splits, target/draft/B200/backend,
   baselines, metrics, timing và statistical protocol.
7. **Results and Analysis:** main RQ1; complementary branches RQ2;
   learning/warm-up RQ3–RQ5; correctness/short behavior RQ6; limitations.
8. **Related Work:** pretrained speculative drafters, compressed/sparse
   attention, memory adaptation và acceptance-oriented objectives.
9. **Limitations:** teacher proxy, shared support, position approximation,
   cost/storage và phạm vi generalization.
10. **Conclusion:** chỉ kết luận những C1–C6 thực nghiệm hỗ trợ; nêu boundary
    của gain và hướng extension.
11. **Appendices:** contracts/schema; exact hyperparameters/schedules;
    cache audit; mask/streaming examples; per-source/per-bin/per-document
    results; seed/timing; compute accounting; failed runs và V0 provenance.

Phần Results hiện là kế hoạch ở §11–§12. Khi có artifact, giữ cấu trúc câu
hỏi và thay planned evidence bằng measured results, không viết kết quả giả.

## 14. Related work và ranh giới novelty

| Nhóm/nguồn đã đối chiếu | Quan hệ với AMR | Điều phải chứng minh khác biệt |
|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036), [official code](https://github.com/z-lab/dflash) | Nền pretrained parallel drafter và target-feature conditioning | Frozen-memory adaptation có gain vượt dense hoặc heuristic-memory control |
| [DeepSeek-V4 HCA/CSA](https://arxiv.org/html/2606.19348v1#S2.SS3) | Nguyên lý global compression + sparse detailed context | Shared target-feature memory/raw retrieval cho block drafter; không claim phát minh HCA/CSA |
| [DeepSeek-V3.2 DSA](https://arxiv.org/html/2512.02556v1#S2.SS1.SSS1) | Tiền lệ attention-supervised indexer | Group/block teacher phù hợp frozen drafter và có actual utility |
| [MagicDec](https://arxiv.org/abs/2408.11049) | Sparse draft KV và cost-aware long-context speculation | So khác regime/drafter/memory/training; chỉ giảm draft context chưa là novelty |
| [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/html/2606.11552v1#S5.SS5) | Prefix/chain-oriented objective đã tồn tại | AMR novelty nằm ở memory adaptation; optional loss cần control riêng |
| [Context-Adaptive DFlash trong repo](baselines/context_adaptive_dflash.md) | Hướng training-free chọn/budget/reuse context | Đo learned memory có lợi hơn policy rẻ ở cùng accounting; implementation evidence tách riêng |

MagicDec xét sparse draft cache và drafting strategy theo bottleneck context/
batch. Không dùng số speedup của MagicDec làm bằng chứng AMR trên B200 batch 1.
[Paper MagicDec](https://arxiv.org/abs/2408.11049).

Không tuyên bố “first” khi chưa literature review có hệ thống. Cần mở rộng
related work về learned KV compression, sparse retrieval và frozen-model
memory adaptation trước submission. Các nguồn ở đây là nền đã kiểm tra,
không phải tuyên bố review đã bao quát toàn lĩnh vực.

## 15. Rủi ro, giới hạn và kết quả không có gain

| Rủi ro | Dấu hiệu | Cách xử lý trong nghiên cứu |
|---|---|---|
| Dense drafter cần chi tiết phân tán | Acceptance giảm dù attention coverage cao | Tăng budget trong run mới; report failure bins; kiểm tra simple baseline |
| Global pooling mất cấu trúc/scale | CE/retention kém, gate triệt gần hết global | Mean/learned pooling và bias/position ablations |
| Attention teacher không phản ánh utility | Indexer KL tốt nhưng learned selection không hơn heuristic | Giới hạn claim; kiểm tra can thiệp và error cases |
| Một support chung thiếu nhu cầu layer/query | Một số layers/offsets suy giảm rõ | Báo breakdown; layer-specific extension tách khỏi main |
| Midpoint RoPE không phù hợp summaries | Long-range behavior suy giảm | Position-policy ablation cùng budget |
| Index/gather/build vượt phần tiết kiệm | Draft attention nhanh hơn nhưng total decode không tăng | Profile actual kernels/cache; xác định crossover hoặc no-gain |
| Target verify dominates | Giảm draft cost chỉ tạo gain nhỏ | Báo runtime decomposition; không đổi metric để che bottleneck |
| Existing cache không hợp lệ | Token/feature recompute mismatch | Rebuild causal features; tính cost vào adaptation |
| Long ShareGPT quá ít | Source/length imbalance rõ | Ghi coverage, chọn quota khả thi; không tạo claim cân bằng giả |
| Warm-up không đạt ở final budget | Wide budget ổn, sparse budget fail | Tăng budget/đơn giản hóa trong run mới; không tự mở full backbone |

Nếu acceptance giữ được nhưng throughput không tăng, kết quả là **adapter
khả thi về alignment nhưng chưa có systems gain**. Nếu throughput tăng nhờ
routing mà forced AMR không hơn dense, báo đúng cơ chế crossover. Nếu frozen
adapter không đạt, kết luận giới hạn của memory-only adaptation; LoRA là
phương pháp mở rộng mới với chi phí/claim mới.

Không coi kết quả âm là lỗi cần sửa bằng cách đổi holdout hay bỏ bin khó.
Giữ failed-run artifacts và decision log để paper có ranh giới kết luận rõ.

## 16. Readiness và việc cần triển khai

### 16.1. Trạng thái hiện tại

- [x] Chốt hướng frozen DFlash + shared lightweight memory adapter.
- [x] Có docs về data/cache, loss/gradient, warm-up, evaluation và migration.
- [x] Có V0 code và CPU evidence để tái sử dụng có kiểm tra.
- [ ] Converter giữ final response và audited cache importer cho recipe mới.
- [ ] Grouped pooling/raw group retrieval, masks và streaming implementation.
- [ ] Compact train-only dense-attention teacher collector.
- [ ] Multi-anchor adapter-only trainer, checkpoint/resume, shared evaluator.
- [ ] B200 runtime validation và final-budget acceptance pilot.
- [ ] Paired holdout, ablations và total adaptation cost report.

V0 capture → candidates → label → train-selector → train-compressor là
pipeline lịch sử. Các lệnh đó không thực hiện recipe CE + group-indexer KL
trong file này. Không dùng V0 checkpoint/manifest như artifact mới qua đổi tên.

### 16.2. Thứ tự triển khai

1. Response/token/cache contract và audit: đủ dữ liệu thật trước train.
2. Frozen-model wrapper + grouped memory: positions/causality/gradient tests.
3. Train-only teacher collector và indexer loss: kiểm tra domain/normalization.
4. Unified trainer + fixed-anchor/full-rollout evaluator + checkpoint resume.
5. B200 runtime ngắn, pilot 2–5K, final-budget retention/cost.
6. Mở 50K nếu pilot hỗ trợ; calibration và locked holdout/ablations.

Tasks chi tiết nằm trong [kế hoạch T1–T12](superpowers/plans/2026-10-08-amr-dflash-implementation.md).
Server offline dùng Python 3.12/dependencies đã có; local chỉ dev CPU theo
[CPU workflow](cpu_dev_workflow.md). Tài liệu này không tạo lệnh main training
khi trainer mới chưa tồn tại.

## 17. Sổ quyết định và cách cập nhật ý tưởng

### 17.1. Những quyết định đã chốt

| ID | Quyết định 09/10/2026 | Lý do/điều kiện thay đổi |
|---|---|---|
| D1 | Freeze toàn bộ pretrained DFlash và target | Mục tiêu ít tài nguyên; LoRA chỉ extension có evidence |
| D2 | Shared memory ba branches, không xen kẽ native CSA/HCA | Ít thành phần, giữ interface pretrained; cần ablate support chung |
| D3 | Fine mean keys + learned indexer + raw group values | Index nhẹ, hạn chế thay feature geometry |
| D4 | Global scalar weighted pooling trên completed groups | Giảm params; đủ lịch sử, causal streaming |
| D5 | CE + auxiliary indexer KL, không main preference/RL | Reuse regenerate/cache; Top-K cần đường gradient riêng |
| D6 | Warm-up gần mean/wide budget trước main budget | Adapter mới cần adaptation; không reset backbone |
| D7 | Actual throughput và acceptance là evidence chính | Keys/loss/attention coverage chỉ giải thích cơ chế |
| D8 | Phân biệt idea, code và measured result | Tránh hiểu docs mới là trainer/B200 đã sẵn sàng |

### 17.2. Các câu hỏi còn mở

| Câu hỏi | Quyết định tạm thời | Evidence để chốt |
|---|---|---|
| Final m/c/K/L tối ưu? | 4/64/256/128 | Validation Pareto và kernel profiling |
| Teacher subset cần bao nhiêu? | Subset nhỏ, 25% labeled anchors/batch | Teacher-cost vs indexer/acceptance curve |
| Bao nhiêu main steps/epochs đủ? | Pilot 200 steps rồi chọn budget dài hơn | Convergence và actual validation, không giả định từ 50K |
| Warm-up schedule cụ thể? | ≤200 steps, c32→64, K1024→512→256 | Retention ở phase boundaries và total compute |
| Scalar gate và midpoint đủ tốt? | Finite -4, midpoint completed group | Bias/position/pooling ablations |
| Shared support có giới hạn gì? | Một support/block cho năm layers | Per-layer/offset diagnostics và failure cases |
| Routing crossover bao nhiêu? | Calibrate sau khi có checkpoint | Paired dense/AMR cost theo GPU/backend/bin |
| Cache hiện tại reuse được cho AMR? | 50K Qwen3-4B cache tồn tại theo xác nhận 09/10; audit 05/10 của cache run 28/09 fail 32/32 mỗi split, chưa có report audit mới | Kiểm fingerprints + recompute subset; quyết định reuse hoặc rebuild |
| Prefix loss/LoRA có cần? | Mặc định tắt | Main recipe ổn rồi thử với controls/compute riêng |

### 17.3. Mẫu ghi một thay đổi tiếp theo

~~~text
Ngày / revision:
Decision ID:
Thay đổi và lý do:
Evidence hoặc artifact hỗ trợ:
Ảnh hưởng tới method / loss / data / compute / claims:
Tài liệu và implementation tasks phải đồng bộ:
Trạng thái: đề xuất / đã chốt / đã triển khai / đã đo
~~~

Mỗi kết quả mới phải có run/config/model/data hashes, hardware/backend,
timing scope và local artifact/report link. Cập nhật C1–C6 ở §12 trước khi
đưa số vào abstract/introduction/conclusion. Thay đổi backbone hoặc objective
phải ghi recipe mới để giữ khả năng so sánh với main frozen-memory method.
