# AMR-DFlash — memory adapter nhẹ trên pretrained DFlash

Ngày tạo: **08/10/2026**. Cập nhật thiết kế: **09/10/2026**.

**Quyết định hiện tại:** giữ nguyên target và toàn bộ pretrained DFlash-5L;
chỉ huấn luyện một memory adapter nhỏ dùng chung giữa năm draft layers.
Thiết kế này thay hướng train selector bằng acceptance preferences.
**Trạng thái:** thiết kế đã thống nhất; code hiện có vẫn là V0
preference-selector + global slots. Trainer và compressor mới chưa được triển khai.
Người vận hành xác nhận regenerated data và target-feature cache 50K từ Qwen3-4B
đã có trên server; parity/fingerprint với snapshot và trainer AMR vẫn cần kiểm tra.

Tên canonical trong repository là **AMR-DFlash**; “ARM-DFlash” trong trao đổi
chỉ cùng phương pháp. File giữ ngày tạo trong tên để bảo toàn các liên kết.

[Hồ sơ ý tưởng và dàn ý paper](../../amr_dflash_paper_story.md) là tài liệu
chính để kiểm soát paper story, recipe training và các quyết định nghiên cứu.
File này là **đặc tả kỹ thuật đi kèm** cho kiến trúc và implementation contracts.
Xem [dữ liệu/train](../../amr_dflash_training_data.md),
[kế hoạch triển khai](../plans/2026-10-08-amr-dflash-implementation.md)
và [launcher V0](../../baselines/amr_dflash.md).
[Proposal 08/10](../../../src/ARMdflash/AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md)
và các contracts trong src/ARMdflash là lịch sử V0, chưa đồng bộ thiết kế này.

## 1. Bài toán, giả thuyết và mục tiêu

DFlash dự đoán song song một block từ target hidden features của prefix.
Ở long context, đọc context K/V dài làm tăng chi phí drafting. Phương pháp
phân bổ thông tin thành ba mức: global summaries, selected fine context và
exact local context, rồi dùng chính pretrained drafter để dự đoán block.

**Giả thuyết:** memory adapter học bằng draft prediction có thể giữ phần lớn
acceptance của dense DFlash với context attention rẻ hơn, tạo lợi ích về
committed tokens/s mà không cần cập nhật backbone.

Primary metric là tổng decode committed tokens chia tổng decode seconds.
Actual accepted proposals, committed tokens/round, draft latency, TTFT,
E2E, resident memory và tổng adaptation GPU-hours giải thích sự đánh đổi.
Giảm số keys hay giảm training loss chưa chứng minh tăng tốc.
Tăng acceptance vượt dense là kết quả có thể đạt, chưa phải điều kiện tiên quyết.

- Biến độc lập: thành phần memory, budget, learned/fixed pooling và learned/fixed selection.
- Biến phụ thuộc: acceptance, committed throughput, latency và chi phí adaptation.
- Biến kiểm soát: target/draft snapshots, tokenizer/template, feature layers,
  dtype/backend, block 16, generation cap, splits và protocol đo.
- Phạm vi đầu tiên: greedy, một B200, giữ nguyên năm layers và một anchor +
  15 proposals. Speculative sampling và thay depth/block thuộc nghiên cứu sau.

Các thăm dò attention chỉ tạo động cơ, không chứng minh causal utility.
[Báo cáo 10 GovReport](../../experiments/2026-10-06_dflash_attention_10_instances_full_mass.md)
đo 21,35% full attention mass trên live block tại 16K, trên cohort nhỏ và backend
instrumented. Không dùng attention coverage hoặc timing đó làm acceptance/speedup.

## 2. Lựa chọn kiến trúc và liên hệ HCA/CSA

| Lựa chọn | Quyết định |
|---|---|
| Shared memory adapter, frozen DFlash | Phương pháp chính: ít module, gần đầu vào pretrained, index một lần/block |
| Native HCA/CSA xen kẽ theo layer | Ablation tương lai; thay visibility và query routing sâu hơn |
| Full draft fine-tuning | Không thuộc phương pháp chính; chi phí và claim khác |
| LoRA attention nhỏ | Extension khi memory-only chưa đạt yêu cầu, báo riêng params và GPU-hours |

[DeepSeek-V4](https://arxiv.org/html/2606.19348v1#S2.SS3) dùng
HCA để nén mạnh và CSA để chọn compressed entries, phân bổ theo layer,
kèm local window. AMR kế thừa nguyên lý nhiều độ phân giải, nhưng dùng
target-derived features và memory chung. Nhánh fine chọn **raw features của
nhóm được index**, thay vì attention trên compressed entries như CSA gốc.
Không gọi phiên bản này là transplant nguyên bản HCA/CSA.

~~~mermaid
flowchart LR
    DATA["Regenerated prompt + response"] --> CACHE["Target features đã audit"]
    CACHE --> P["Frozen DFlash fc + hidden_norm"]
    P --> G["Global weighted pooling nhỏ"]
    P --> F["Fine block index: mean pooling cố định"]
    P --> L["Exact local features"]
    Q["Anchor + latest/recent committed features"] --> I["Indexer nhỏ: một lần/block"]
    F --> I
    I --> R["Gather raw features của nhóm đã chọn"]
    G --> M["Working memory chung"]
    R --> M
    L --> M
    M --> D["Frozen DFlash-5L + full live block"]
    D --> LOSS["Weighted draft CE"]
    TEACH["Dense DFlash attention trên subset"] --> IDX["Indexer KL"]
    I --> IDX
~~~

Training dùng response tokens làm labels. Inference giữ full-context target
verification, crop rejected prefix và commit đúng EOS/output cap.

## 3. Tham số được học và ngân sách

Với H = hidden_norm(fc(target_features)) có shape [N,d]:

| Thành phần | Trạng thái |
|---|---|
| Target, token embeddings và output head dùng chung | Frozen |
| DFlash fc, hidden_norm, năm attention/FFN layers, final norm | Frozen |
| Fine-block mean pooling | Cố định |
| Global pooling MLP d → r_C → 1 | Trainable |
| Indexer key projection d → r_I và query MLP 3d → r_I → r_I | Trainable |
| Global attention bias dùng chung | Trainable scalar |

Khởi đầu r_C=32, r_I=64. Global values là weighted mean của H; phương pháp
chính không có full-rank value projection, per-layer adapter hoặc residual
projection mới. Pooling weights ban đầu uniform bằng final pooling weight
bằng zero và tầng trước được khởi tạo bình thường. Indexer nhỏ vẫn cần
khởi tạo và học; không khởi tạo lại DFlash.

Số tham số chính có bậc 4*d*r_I + r_I² + d*r_C, cộng biases/gate nếu có.
Trainer phải ghi số thực tế và assert optimizer không chứa frozen params.
Mục tiêu là module nhỏ; không tự gán tỷ lệ tiết kiệm thời gian theo tỷ lệ params.

| Hyperparameter | Điểm bắt đầu pilot, chưa benchmark |
|---|---:|
| Fine group size m | 4 tokens |
| Global group size c | 64 tokens |
| Selected fine groups K | 256 |
| Exact local window L | 128 tokens |
| Global additive logit bias khởi tạo | -4, finite |
| Query recent window | 16 committed features |
| Anchor blocks/document mỗi lượt train | 8, tăng sau profile |
| Optimizer | AdamW, LR 1e-4, weight decay 0,01, clip norm 1 |
| Seed | 17 |

Các giá trị này chưa phải env/CLI được V0 hỗ trợ. c là số tokens mỗi global
entry; khác với num_slots=128 cố định của V0.

## 4. Xây memory, positions và streaming

### 4.1. Global memory

Chia absolute timeline thành nhóm c tokens, luôn căn theo vị trí 0.
Chỉ xuất global entry khi nhóm đã hoàn tất trong prefix được target xử lý.
Nhóm đang dở được giữ bằng local window; yêu cầu L >= c trong config.

\[
a_i=w_2^\top\operatorname{SiLU}(W_1 h_i),\qquad
g_b=\frac{\sum_{i\in B_b}\exp(a_i)h_i}
          {\sum_{i\in B_b}\exp(a_i)}.
\]

Số global entries là floor(N/c), không cố định theo mọi N. Không truncate
global entries để giữ một trần slots mà bỏ lịch sử xa.
Dùng FP32 và log-sum-exp ổn định khi pooling; cast về dtype drafter khi attention.

Ở inference, nhóm hoàn tất không đổi và được cache; giữ accumulator cho tail,
append chỉ features đã xử lý và commit. Sau load/update weights phải rebuild
memory/index; accumulator cũ không có ý nghĩa với compressor mới.
Training dựng lại pooling với autograd, không tái sử dụng inference accumulator
đã detach để thay thế graph.

Global summaries có thể chứa thông tin trùng selected/local raw features.
Đây là nhiều độ phân giải chồng lấp; không gọi là disjoint complement.

### 4.2. Fine retrieval

Mỗi nhóm m tokens có mean feature cố định u_b; indexer key k_b=W_k u_b.
Query đọc anchor embedding, latest H và mean recent H của prefix đã biết.
Score là dot product chuẩn hóa theo sqrt(r_I), có thể thêm một scalar position bias.

Indexer chọn một support cho cả block, dùng chung năm layers. Query có context
ngay trước layer đầu, tránh dùng 15 mask embeddings giống nhau làm query nội dung.
Eligible groups phải hoàn tất trước anchor và nằm hoàn toàn ngoài exact local
window. Chọn min(K, số eligible groups), gather raw features nguyên bản,
union local, loại trùng và sort theo original absolute positions.

Top-K không truyền gradient qua indices. Key/query scorer học bằng indexer
loss; raw values không được gọi là một differentiable selector.

### 4.3. DFlash attention và accounting

Mỗi layer dùng Q/K/V/O projections, k_norm và residual/FFN pretrained.
Context memory chung gồm selected raw, local raw và global entries.
K/V của toàn bộ live block tiếp tục được thêm như DFlash gốc:

\[
O_\ell=\operatorname{Attn}
(Q_\ell,[KV_\ell(R_{\rm fine}\cup R_{\rm local}),
          KV_\ell(G),KV_\ell(\text{live block})]).
\]

Raw K giữ original RoPE positions. Global K lấy integer midpoint của nhóm
hoàn tất, rotate sau pooling/projection; V không rotate. Global bias chỉ đặt
trên global keys. Quy ước này là lựa chọn cần ablate, không là phép biến đổi
tương đương dense attention. Live block vẫn bidirectional.

Số context keys tối đa:
\[
B_{\rm ctx}\le mK+L+\lfloor N/c\rfloor.
\]
Live block 16 nằm ngoài B_ctx. Với N=16384, m=4, K=256, L=128, c=64:
B_ctx <= 1408, tổng keys <= 1424. So sánh matched-total-budget dùng số entries
thực tế sau loại trùng; selection-only không được cấp ít keys hơn rồi gọi công bằng.

Dense bypass tắt global branch và giữ mọi raw positions để kiểm tra identity.
Short-context routing dùng threshold pilot và calibration theo server/backend
sau cùng; không tự gọi một threshold cố định là cost-optimal.
Full resident raw feature/KV bank và full target KV vẫn có thể tồn tại.
Báo storage toàn hệ thống, không suy giảm VRAM từ số keys attention.

## 5. Dữ liệu và tính nhân quả

Một sample có clean prompt p, target-regenerated response y, token IDs/positions,
source ID, split và cache fingerprint. Giữ response cho prediction labels;
reference ArXiv gốc chỉ phục vụ ROUGE. Tạo random anchors trong response.
Với anchor ở position a, block là token đã biết x_a + tối đa 15 masks,
labels là x_(a+1)...x_(a+15), valid mask kết thúc tại EOS/response end.

Context features chỉ là H[0:a], không gồm feature của anchor chưa được target
xử lý hoặc proposal positions. Whole clean sequence có thể được chạy qua causal
target một lần; slice/mask theo anchor vẫn phải áp dụng **trước pooling và selection**.
Entry có một token ngoài prefix không được dùng, kể cả centroid nằm trong prefix.

Nhiều anchors cùng document dùng chung feature store nhưng không nhìn thấy
live K/V, anchor token hoặc local memory của block khác. Packing không biến
training thành attention causal xuyên các blocks. Padding và post-EOS labels
không vào loss; state không có valid proposals hoặc không có adaptation path
được skip với reason.

Projected cache dùng lại được khi target, draft fc/norm, layers, token IDs,
template, positions, truncation và dtype policy khớp. Nếu chỉ có raw target
features, project bằng frozen fc/norm một lần. Raw target-feature cache 50K
Qwen3-4B có trên server theo xác nhận ngày 09/10. Audit ngày 05/10 của cache
run 28/09 ghi mismatch hidden 32/32 train và validation; chưa có report audit
mới. Vì thế artifact tồn tại nhưng reuse cho thiết kế AMR chưa được xác nhận,
và importer mới chưa triển khai. Không chấp nhận chỉ vì shape đúng. Không chỉnh
prompt bằng head/tail truncation rồi dùng cache/greedy labels của prompt khác.

Split theo source document, kiểm tra IDs/full-history content, giữ validation
và holdout độc lập. Training adapter tập trung long-context anchors; short
context kiểm tra dense routing/parity. Pilot 2–5K train documents đại diện,
validation khóa riêng; mở 50K sau runtime và acceptance/cost checks.

Chi tiết schema, audit cache, teacher subset và dữ liệu server:
[dữ liệu huấn luyện](../../amr_dflash_training_data.md).

## 6. Loss và gradient

### 6.1. Prediction loss chính

\[
\mathcal L_{\rm draft}=
\frac{\sum_{b,j}v_{b,j}w_j[-\log q_{b,j}(y^*_{b,j})]}
     {\sum_{b,j}v_{b,j}w_j},\qquad
w_j=\exp(-(j-1)/\gamma),\quad j=1,\ldots,15.
\]

v là valid mask; gamma=7 là điểm bắt đầu theo DFlash block-16.
[DFlash training](https://arxiv.org/html/2602.06036v1#S4.SS2)
dùng random anchors và block prediction;
[phụ lục](https://arxiv.org/html/2602.06036v1#A1.SS1) hỗ trợ cache offline.
Prediction CE là surrogate cho acceptance; không gọi là trực tiếp tối ưu actual A.

Target/feature extraction được detach. Frozen draft forward và frozen output
head phải giữ graph khi train compressor/gate; không đặt toàn bộ forward
trong no_grad/inference_mode. Freezing giảm optimizer state và weight gradients,
nhưng vẫn cần forward/backward đối với đầu vào qua năm layers.

### 6.2. Indexer supervision nhỏ, không preference dataset

Trên subset **train-only**, chạy frozen dense DFlash cho cùng anchors và
prefix visibility. Thu attention trên raw context, aggregate qua heads/layers
và proposal queries với cùng w_j; giữ eligible nonlocal groups rồi chuẩn hóa:

\[
t_{a,b}=
\frac{\sum_{\ell,h,j}v_{a,j}w_j
      \sum_{i\in B_b} A^{\rm dense}_{\ell,h,j,i}}
     {\sum_{b'\ {\rm eligible}}\sum_{\ell,h,j}v_{a,j}w_j
      \sum_{i\in B_{b'}} A^{\rm dense}_{\ell,h,j,i}}.
\]

Loại local/live keys trước normalization, áp đúng eligibility như indexer.
Teacher denominator bằng zero hoặc không có eligible groups thì skip loss đó
với reason; không tạo distribution/reward giả. Lưu group IDs và t, không lưu
full attention matrices/full-vocabulary logits cho toàn 50K.
Teacher có visibility/backend/fingerprint riêng; collector timing thuộc
adaptation cost, không dùng cho inference benchmark.

\[
\mathcal L_{\rm idx}=
D_{\rm KL}(\operatorname{stopgrad}(t_a)\|
                  \operatorname{softmax}(s_\phi(a,\cdot))).
\]

Teacher là attention allocation proxy của drafter, không là acceptance utility
hay target attention. Khác dense-compressed teacher của native CSA; phải ablate
learned indexer so với fixed/random selection cùng budget.
[DSA V3.2](https://arxiv.org/html/2512.02556v1#S2.SS1.SSS1) là tiền lệ
indexer distillation; không khẳng định cách aggregate mới là recipe DeepSeek-V4.

Một optimizer cập nhật compressor, scorer và gate:
\[
\mathcal L_{\rm main}=\mathcal L_{\rm draft}+\beta\mathcal L_{\rm idx}.
\]
Indexer nhận gradient từ L_idx; compressor/gate từ L_draft.
Sampler main trộn teacher-labeled anchors, điểm bắt đầu 25% batch khi đủ dữ liệu,
để scorer tiếp tục học sau warm-up; anchors khác chỉ có L_draft.
Phải log số anchors có indexer loss. beta=1 là điểm bắt đầu pilot cần validation.

### 6.3. Các loss bổ sung

Warm-up có thể thử dense-DFlash output/hidden distillation trên subset nhỏ,
với teacher cố định, cùng prefix/block và coefficient giảm dần. Mặc định không
đòi thêm loss đó hoặc full-vocabulary teacher store.

Prefix-product loss mặc định tắt, chỉ thêm sau baseline:
\[
\mathcal L_{\rm prefix}
=-\frac{1}{15}\sum_{k=1}^{15}\prod_{j=1}^{k}q_j(y_j^*),
\]
với valid-length normalization cho block ngắn. Nó có gradient yếu khi
probabilities nhỏ, không bằng greedy acceptance và không là novelty riêng.
[Chain reward đã có](https://arxiv.org/html/2606.11552v1#S5.SS5);
so cả dense và adapter với cùng objective khi thử loss này.

## 7. Training hai giai đoạn

| Giai đoạn | Trainable | Cách chạy | Điểm chuyển |
|---|---|---|---|
| Baseline/audit | Không | Dense DFlash, kiểm tra cache và parity | Có baseline ghép cặp, cache contract hợp lệ |
| Warm-up ngắn | Compressor, indexer, gate | Pooling uniform; budget rộng; teacher subset; L_draft + beta L_idx | Loss/gradient/checkpoint hợp lệ, kiểm tra retention |
| Memory training chính | Cùng các module nhỏ | Chuyển budget về cấu hình deploy, prediction blocks từ regenerate | Chọn checkpoint bằng validation acceptance/cost |
| Calibration/holdout | Không | Force dense/AMR, đo crossover, khóa config | Báo actual acceptance và throughput trên holdout |

Warm-up dùng c=32, tối đa K=1024 groups ở pilot; chuyển K qua 512 tới 256
và c tới 64. Các mốc kiểm tra theo optimizer steps, không theo số giờ suy đoán.
Warm-up giới hạn ban đầu 200 optimizer steps, prediction validation mỗi 50
steps và full rollout validation cuối phase. Nếu không đạt retention,
ghi nhận và tăng memory budget trong run mới hoặc đơn giản hóa, không lặp
warm-up vô hạn hoặc tự mở backbone.

Training chính đánh giá prediction trên **toàn validation anchor manifest**
mỗi 50 steps; actual full-validation rollout mỗi 500 steps và cuối phase.
Pilot đầu đo 200 steps trước mở run dài. Teacher-label quota, budget schedule,
phase boundary và metric history nằm trong checkpoint.

Retention 95% mean accepted proposals của dense là **gate pilot đề xuất**,
không phải kết quả đã có hoặc mục tiêu 100% được bảo đảm. Tính theo từng context
bin; nếu dense A bằng zero, không tính ratio, báo absolute difference.
Warm-up budget chưa là deploy budget; cần đánh giá AMR forced ở budget cuối
và bao gồm toàn bộ memory cost trước khi kết luận.
Short-context dense bypass và target greedy parity được kiểm tra riêng.

## 8. Validation, reproducibility và chi phí

Một evaluator core hỗ trợ in-memory/checkpoint, với hai scopes rõ:
prediction blocks và full greedy rollout. Prediction prefix-match chỉ là
actual greedy A nếu regenerate/target/decoding alignment đã được xác minh;
nếu response sampled thì báo teacher-token match, không gắn nhãn verifier acceptance.

Checkpoint mới lưu adapter weights, optimizer/scheduler/RNG/sampler,
phase/step, fingerprints, groups/budgets/query policy, teacher-manifest hash,
dataset/anchor policy và evaluation results. Không save lại frozen model weights.
V0 checkpoint có schema khác; không load như checkpoint adapter mới.

Evaluator log phase start/end, progress bar, số samples, metrics và thời gian.
Missing/corrupt checkpoint, empty validation, non-finite metric, wrong
fingerprint hoặc partially completed evaluation phải fail; không báo success.
Record heartbeat ít nhất mỗi 60 giây; lưu partial artifacts và error reason
nếu evaluator bị gián đoạn, không dùng partial summary như full validation.
Training logs có CE/indexer loss, valid proposals, teacher quota, gradient norm,
trainable params, step time, blocks/s, cache I/O và VRAM.

Review criteria pilot, cần đo trên B200:
~~~yaml
review_criteria:
  metrics:
    - name: target_greedy_token_parity
      direction: "=="
      threshold: 1.0
    - name: accepted_proposal_retention_per_nonzero_dense_bin
      direction: ">="
      threshold: 0.95
    - name: paired_decode_committed_throughput_speedup
      direction: ">"
      threshold: 1.0
  performance:
    - "Báo end-to-end, draft và memory costs; không đặt ngưỡng giờ train chưa đo."
    - "Tính tổng GPU-hours cho audit/recompute cần thiết, teacher, train, validation và calibration."
  observability:
    - "Log theo step, progress riêng cho evaluation, phase-end summary và failure reasons."
  stability:
    - "Loss/gradients hữu hạn; optimizer chỉ cập nhật adapter; frozen weights giữ hash."
  custom:
    - "No future-feature leakage, rejected features không append, original raw positions."
~~~

Scientific claim dùng paired holdout theo document, đủ samples và confidence
interval; các thresholds trên chỉ là tiêu chí pilot, không thay kiểm định.
Đếm A_raw, A_committed và G tách biệt; G không luôn bằng A+1 do correction,
bonus, EOS/cap. Throughput = sum(G_decode)/sum(T_decode), không reciprocal mean TPOT.

L0: static/semantic/gradient tests khi triển khai. L1: assembled training pipeline
trên một B200 với real data, kiểm tra ngắn khoảng năm phút trước pilot lớn.
Local chỉ CPU; không sửa driver. Runtimes offline Python 3.12 và dependencies
có sẵn; không thêm venv từng baseline hoặc installer online.
Docs-only revision này không thực hiện L1 hoặc báo đã train.

## 9. Paper story và ablations

**Problem:** pretrained block-parallel drafter phải đọc target context dài,
nhưng không cần mọi vùng ở cùng độ phân giải mỗi round.

**Observation:** DFlash phân bổ attention qua prompt, committed output và live
block; vùng được ưu tiên thay đổi theo state. Evidence hiện có là diagnostic
trên cohort nhỏ, cần interventions và rollout để xác minh.

**Hypothesis:** coarse global summaries + state-conditioned fine retrieval +
exact local features cho một frozen drafter có thể giữ alignment với chi phí thấp.

**Method:** một adapter dùng chung, learned pooling và một block-level indexer;
fine branch giữ pretrained feature geometry, không thay backbone/depth/block.

**Training:** dùng target-regenerated tokens và audited cached features;
weighted draft prediction huấn luyện pooling/gate, compact attention
distillation huấn luyện scorer. Hai giai đoạn cùng pipeline, không cần
candidate-preference labeling hoặc RL cho phương pháp chính.

**Outcome cần chứng minh:** measured committed throughput tốt hơn dense trên
long context, với acceptance retention và short-context/target parity được báo rõ.
Không hứa tăng chất lượng tóm tắt vượt target; full-context verifier giữ output
theo target nếu implementation đúng.

| Ablation | Câu hỏi |
|---|---|
| Dense pretrained DFlash | Baseline cùng checkpoint, backend, workload |
| Local + fine, không global | Global information có ích ngoài retrieval |
| Local + global, không learned fine | State-conditioned retrieval có ích |
| Full adapter | Ba mức có bổ trợ |
| Mean vs learned global pooling | Pooling học được có đáng chi phí |
| Fixed/random vs learned indexer | Scorer học được có giá trị ngoài budget |
| Raw selected groups vs compressed fine entries | Geometry pretrained có giúp retention |
| Có/không warm-up, cùng training compute | Warm-up có giúp adaptation |
| Có/không prefix loss | Đóng góp objective, kiểm soát dense tương ứng |

Matched-total-key và matched-compute comparisons báo riêng; không đồng nhất
key count với FLOPs. Bao gồm cache build/update, gathers, full-bank storage và
target verification. Bootstrap theo document, không theo anchors tương quan.
Khóa config/checkpoint/crossover trước holdout; no-gain là outcome hợp lệ.

## 10. Code hiện có và tác động lên kế hoạch

| Hiện có V0 | Đối với thiết kế mới |
|---|---|
| Frozen five-layer adapter, raw positions, live block | Tái sử dụng sau parity checks |
| Cached greedy verifier, output A/G, fingerprints | Tái sử dụng và mở rộng contracts |
| Prompt-only prepare-data, trajectory capture | Giữ cho V0; cần converter giữ response và cache importer mới |
| Token selector preference training | Thay bằng block indexer + compact attention teacher |
| Learned fixed global slots với d→d projections | Thay bằng local grouped scalar weighted pooling |
| Candidate-verifier-logit compressor KL | Thay main objective bằng weighted prediction CE |
| Batch-1 state trainer, memory-only checkpoint | Thêm multi-anchor packing, unified adapter optimizer/resume |

### Impact on Plan

- T1–T3: giữ frozen model/verifier, sửa config/accounting sang m/c/K/L và kiểm tra causal groups.
- T4: chuyển capture training sang response/token/cache/anchor contract; V0 capture là legacy.
- T5–T6: thay candidates/preferences bằng compact dense-attention teacher và block indexer.
- T7: grouped pooling không full-rank value projection; thêm incremental completed-group lifecycle.
- T8–T9: weighted CE + indexer KL, joint **adapter-only** trainer, checkpoint/resume và hai scopes evaluation.
- T10–T11: cache/grouped retrieval, profiling và CLI mới có schema tách V0.
- T12: retention/throughput/compute ablations; không yêu cầu full-backbone fine-tune cho phương pháp chính.

Chi tiết tasks nằm trong [kế hoạch đã cập nhật](../plans/2026-10-08-amr-dflash-implementation.md).
Các tài liệu V0 trong src/ARMdflash và [review V0](../../reviews/2026-10-08_amr_dflash_implementation_review.md)
giữ giá trị truy vết; chúng không mô tả trainer mới đã triển khai.
