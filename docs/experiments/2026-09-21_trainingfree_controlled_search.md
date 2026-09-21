# Training-Free controlled empirical search — 2026-09-21

## Mục đích

Đây là mốc đo có kiểm soát sau V3 để quyết định có đáng viết router/head
selection hay không. E41 đo độ tập trung theo head; E42 đo upper bound của
hybrid policy: một số head được giữ global, các head còn lại giữ đúng các
source token có attention lớn nhất. Cả hai đều là oracle trên attention hiện
có, chưa phải physical KV executor và không được diễn giải thành speedup.

Protocol chi tiết nằm ở
[`src/TrainingFree/plans/2026-09-21-controlled-empirical-search.md`](../../src/TrainingFree/plans/2026-09-21-controlled-empirical-search.md).

## Cấu hình và artifact

| Trường | Giá trị |
|---|---|
| Model | Qwen3-0.6B |
| GPU | Modal, NVIDIA A10 |
| Dữ liệu | 20 `gov_report` + 20 `multi_news` từ `data/longbench_100_14k/` |
| Decode | `max_new_tokens=32`, seed 42; greedy `argmax` token selection |
| Dtype / runtime | `bfloat16`; Modal image Python 3.12, `torch==2.11.0`, `transformers==5.12.1` |
| Attention backend | Eager decode attention (`output_attentions=True`); prefill uses SDPA temporarily, then switches back to eager |
| Prompt | `Summarize the following document faithfully and concisely. Return only the summary.\n\nDocument:\n` + Qwen chat template; `enable_thinking=False` |
| Model architecture | Qwen3-0.6B, 28 layers, 16 query heads, 8 KV heads (GQA) |
| Layer | only last layer, id 27 |
| Hierarchy | region 1024, block 128, 4 representatives/block, 4/region |
| Gate | expansion ≤30%, mean missed mass ≤1%, P99 missed mass ≤5% |
| E41 run | `e41-head-concentration-20260921` |
| E42 run | `e42-oracle-hybrid-head-20260921` |
| Raw trace | Modal Volume `fast-infer-text-sum-cache:/outputs/recap_kv_v3/e42-oracle-hybrid-head-20260921/hierarchy_trace.jsonl` |
| Manifest | Modal Volume `fast-infer-text-sum-cache:/outputs/recap_kv_v3/e42-oracle-hybrid-head-20260921/manifest.json` |
| Analyzer | [`scripts/analyze_trainingfree_e41.py`](../../scripts/analyze_trainingfree_e41.py), [`scripts/analyze_trainingfree_e42.py`](../../scripts/analyze_trainingfree_e42.py) |

JSON tóm tắt config và kết luận: [`2026-09-21_trainingfree_controlled_search.json`](2026-09-21_trainingfree_controlled_search.json).

## Đối chiếu với baseline audit

Report này **đã được bổ sung đối chiếu với baseline audit**, nhưng đây là đối
chiếu theo ngữ cảnh chứ chưa phải một bảng speedup/quality apples-to-apples.
Baseline audit đầy đủ nằm ở
[`2026-09-21_modal_trainingfree_reference.md`](2026-09-21_modal_trainingfree_reference.md)
với record `modal-trainingfree-reference-20260921`.

### Kết quả baseline audit hiện có

Các số dưới đây là mean của các record hợp lệ; `E2E` tính bằng ms, `tok/s` là
throughput, ROUGE là F1.

| Dataset | Method | Target/GPU | n | E2E ms | tok/s | ROUGE-2 | ROUGE-L | Coverage |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gov_report` | Vanilla HF | Llama 3.1 8B / A100 80GB | 20 | 6663.0 | 22.26 | 0.0903 | 0.1400 | complete |
| `gov_report` | EAGLE-3 | Llama 3.1 8B / A100 80GB | 20 | 4802.3 | 32.19 | 0.0862 | 0.1242 | complete |
| `gov_report` | DFlash | Llama 3.1 8B / A100 80GB | 20 | 3465.7 | 41.32 | 0.0960 | 0.1441 | complete |
| `multi_news` | Vanilla HF | Llama 3.1 8B / A100 80GB | 20 | 3037.2 | 43.17 | 0.0759 | 0.1526 | complete |
| `multi_news` | EAGLE-3 | Llama 3.1 8B / A100 80GB | 20 | 1872.0 | 73.45 | 0.0573 | 0.1404 | complete |
| `multi_news` | DFlash | Llama 3.1 8B / A100 80GB | 20 | 1488.0 | 95.74 | 0.0791 | 0.1538 | complete |
| `qmsum` | Vanilla HF | Llama 3.1 8B / A100 80GB | 4 | 6643.5 | 20.30 | 0.0915 | 0.2087 | OOM từ mẫu 5 |
| `qmsum` | EAGLE-3 | Llama 3.1 8B / A100 80GB | 5 | 5967.1 | 19.50 | 0.0634 | 0.1835 | OOM từ mẫu 6 |
| `qmsum` | DFlash | Llama 3.1 8B / A100 80GB | 20 | 4593.4 | 29.17 | 0.0829 | 0.2032 | complete |

### Training-Free đặt cạnh baseline như thế nào?

| Nhóm | Target/GPU | Output | Metrics đã có | Trạng thái |
|---|---|---:|---|---|
| Vanilla HF / EAGLE-3 / DFlash | Llama 3.1 8B / A100 80GB | 128 tokens | E2E, throughput, ROUGE, coverage; EAGLE có acceptance | Baseline systems/quality audit |
| Training-Free V3/E41/E42 | Qwen3-0.6B / A10 | 32 tokens | K95 concentration, exact expansion, missed attention mass, attention output error | Oracle audit; chưa có E2E, VRAM, throughput, ROUGE |

Do khác target model, GPU, output budget và loại metric, không được lấy số
`E2E`, `tok/s` hoặc ROUGE của Llama 3.1 8B để suy ra Training-Free nhanh hơn,
chậm hơn hay giữ chất lượng tốt hơn. Tương tự, `exact expansion` và `missed
attention mass` của E41/E42 không thể thay thế trực tiếp cho throughput hoặc
ROUGE của baseline.

Giá trị của bảng đối chiếu hiện tại là xác định khoảng trống đo lường:

- baseline audit cho thấy mốc systems/quality cần vượt: DFlash đạt `1.92x`
  Vanilla trên `gov_report` và `2.04x` trên `multi_news` với coverage đầy đủ;
- E41/E42 mới chỉ trả lời Training-Free có headroom attention đủ để tiếp tục
  hay không. Câu trả lời tinh tế hơn là: **có headroom oracle đáng kể,
  nhưng chưa đạt risk contract dưới gate**, đặc biệt trên `multi_news`;
- để đưa Training-Free vào cùng bảng speedup, cần chạy dense control và
  Training-Free trên cùng target/data/GPU/output budget, đồng thời sinh E2E,
  throughput, peak memory và quality; khi đó mới được thêm cột Training-Free
  vào bảng baseline chính.

## Phạm vi bằng chứng và quy ước đọc kết quả

40 tài liệu này là **development/search data**, không phải final evaluation
hay untouched holdout. Các quyết định V3, E41 và E42 đều dùng cohort này; vì
vậy mọi kết luận về độ tổng quát chỉ là exploratory. `qmsum` có trong baseline
audit nhưng không có trong controlled search này.

E41/E42 đạt mức **R7 decision memo trên một R6 dev-search slice**: launcher,
trace và toàn bộ điều kiện search đã hoàn tất, sau đó có quyết định dừng. Điều
này không đồng nghĩa với paper-ready systems evidence; chưa có physical KV
mutation, E2E/VRAM/throughput, logit fidelity hoặc quality comparison cùng
model.

### Reproducibility còn thiếu

Manifest đã khóa model path, seed, dtype và các tham số hierarchy. Tuy nhiên
record hiện chưa lưu model revision/weight hash và attention implementation như
một trường manifest độc lập; thông tin backend ở trên được truy nguyên từ code.
Các run tiếp theo phải ghi thêm model SHA/revision, package versions, backend,
prompt/template, GQA mapping và generation flags vào manifest để có thể tái
lập ở mức paper.

## Hồ sơ độ dài source của cohort đã đo

`source_tokens` là span document sau chat-template, không phải toàn bộ context
hay tên `longbench_100_14k`. Vì vậy cohort thực tế chưa phủ vùng 16–32K hoặc
trên 32K.

| Dataset | n | Mean | P50 | P90 | Min–Max | Buckets |
|---|---:|---:|---:|---:|---:|---|
| `gov_report` | 20 | 8,874 | 8,854 | 12,026 | 2,696–13,538 | 8 `<8K`, 12 `8–16K` |
| `multi_news` | 20 | 2,796 | 2,559 | 5,240 | 545–6,488 | 20 `<8K` |

Đây là một limitation quan trọng: kết quả hiện tại chưa phải bằng chứng cho
long-context scaling. Giả thuyết source-routing chỉ được kiểm tra ở tối đa
13.5K source token và `multi_news` còn ngắn hơn đáng kể.

## E41 — per-head source concentration

K95 fraction là số source token nhỏ nhất cần giữ để chứa 95% source attention
của từng head, chia cho tổng số source token.

| Dataset | Mean K95 | P95 K95 | P99 K95 | Head fraction K95>0.10 | K95>0.25 | K95>0.50 |
|---|---:|---:|---:|---:|---:|---:|
| `gov_report` | 0.2429 | 0.4924 | 0.6108 | 95.12% | 37.93% | 4.59% |
| `multi_news` | 0.3847 | 0.5854 | 0.6642 | 99.97% | 85.45% | 16.42% |

Kết luận E41: **PASS heterogeneity**. Head có khác biệt đáng kể, vì vậy E42
được mở. Tuy nhiên, nhiều head vẫn diffuse; E41 không phải bằng chứng rằng
head routing sẽ có lợi về hệ thống.

### Stability và persistent global heads

Phân tích bổ sung được tính từ chính E41 trace, theo zero-based head id. Với
mỗi head, `P(K95>0.5)` là tỷ lệ decode steps trên 20 document mà head đó cần
hơn một nửa source để chứa 95% mass. Top-head Jaccard được tính giữa các cặp
document sau khi xếp hạng head theo mean K95 của từng document.

| Dataset | Head 9 mean K95 | `P(K95>0.5)` head 9 | Mean Jaccard top-2 | Mean Jaccard top-4 |
|---|---:|---:|---:|---:|
| `gov_report` | 0.5149 | 55.1% | 0.933 | 0.754 |
| `multi_news` | 0.5989 | 92.7% | 0.586 | 0.507 |

Điều này cho thấy hai kết luận đồng thời:

- có một tín hiệu persistent rõ ở head 9 và đặc biệt mạnh trên `multi_news`;
- tập global head rộng hơn không ổn định như nhau: top-4 Jaccard giảm còn
  0.507 trên `multi_news`.

Đây mới là exploratory stability trên dev cohort, không phải holdout. Nó chưa
chứng minh rằng một tập head cố định sẽ transfer sang dataset/model khác.

## E42 — oracle hybrid-head sweep

### Rule chọn head và leakage status

E42 dùng một **dataset-level static head set**, không phải per-document hay
per-token selection:

1. gom tất cả decode steps của 20 document trong từng dataset;
2. tính mean K95 theo từng `(layer, head)`;
3. chọn `ceil(16 × global_fraction)` head có mean K95 lớn nhất;
4. giữ nguyên tập head này cho toàn bộ rows của dataset.

Do bước 1–3 dùng chính dev trace được đánh giá ở bước 4, đây là một lựa chọn
head có leakage và có lợi cho oracle. Nó không phải policy deployable đã học
trước. Ngược lại, routed heads dùng oracle mạnh hơn nữa: tại từng decode
step/head, chọn đúng top source positions theo attention hiện tại. E42 vì vậy
không đo chất lượng router; nó hỏi xem ngay cả dưới selection oracle, frontier
có đạt contract hay không.

### Định nghĩa expansion và output error

Với source length (S), query-head (h), global indicator (G_h), và routed
fraction (r), metric trong bảng là **source-only exact expansion**:

\[
e_h =
\begin{cases}
1, & h \in \mathcal H_G,\\
\lceil rS\rceil/S, & h \notin \mathcal H_G,
\end{cases}
\qquad
E_{src}=\operatorname{mean}_{document,step,head} e_h.
\]

Vì `ceil` và vì global 10% của 16 head thực tế là 2/16 = 12.5%, nên
global 10% + routed 10% cho khoảng 21.25%, không phải đúng 19%. Metric này
chỉ tính source positions; oracle vẫn giữ các non-source/live positions để
tính context error. Do đó `E_src` là lower bound cho full-QK physical cost,
không phải speedup hay tổng active-QK fraction.

`missed mass` là source attention mass bị bỏ trong routed heads. `attention
output error` là relative L2 error của context sau khi giữ global/non-source
positions, giữ top source positions và renormalize attention:

\[
\frac{\|\hat o-o\|_2}{\|o\|_2}.
\]

Các số dưới đây là `E_src / mean missed mass / P99 missed mass`; `PASS` chỉ khi
cả ba gate cùng đạt; các tỷ lệ hiển thị theo phần trăm.

### `gov_report`

| Global head | Routed 5% | Routed 10% | Routed 20% | Routed 30% |
|---:|---|---|---|---|
| 10% | 16.880 / 1.204 / 13.466 F | 21.254 / 0.709 / 8.751 F | 30.005 / 0.313 / 4.400 F | 38.753 / 0.155 / 2.352 F |
| 20% | 28.755 / 0.804 / 9.475 F | 32.504 / 0.460 / 5.850 F | 40.004 / 0.195 / 2.856 F | 47.503 / 0.094 / 1.566 F |
| 30% | 34.692 / 0.488 / 4.657 F | 38.128 / 0.269 / 2.761 F | 45.004 / 0.108 / 1.236 F | 51.878 / 0.050 / 0.656 F |
| 40% | 46.566 / 0.421 / 4.641 F | 49.378 / 0.230 / 2.700 F | 55.003 / 0.091 / 1.210 F | 60.627 / 0.041 / 0.649 F |

### `multi_news`

| Global head | Routed 5% | Routed 10% | Routed 20% | Routed 30% |
|---:|---|---|---|---|
| 10% | 16.897 / 1.428 / 14.100 F | 21.270 / 0.971 / 9.954 F | 30.010 / 0.530 / 5.814 F | 38.770 / 0.309 / 3.446 F |
| 20% | 28.769 / 1.311 / 14.100 F | 32.517 / 0.890 / 9.954 F | 40.009 / 0.485 / 5.814 F | 47.517 / 0.283 / 3.446 F |
| 30% | 34.705 / 0.936 / 11.240 F | 38.140 / 0.629 / 7.757 F | 45.008 / 0.338 / 4.270 F | 51.891 / 0.196 / 2.542 F |
| 40% | 46.577 / 0.381 / 5.406 F | 49.388 / 0.249 / 3.515 F | 55.007 / 0.129 / 1.866 F | 60.638 / 0.073 / 1.082 F |

### Output error và uncertainty ở frontier

Các metric output dưới đây đã có trong trace nhưng không được đưa vào bảng
matrix compact ở trên. Đây là mean/P95/P99 của relative attention-output error;
chúng vẫn là context-level oracle metrics, chưa phải logit agreement hay NLL.

| Dataset | Policy | `E_src` | Missed mean | Missed P99 | Output error mean / P95 / P99 | Gate |
|---|---|---:|---:|---:|---:|:---:|
| `gov_report` | global 10% + routed 10% | 21.254% | 0.709% | 8.751% | 1.580% / 6.448% / 12.118% | FAIL |
| `gov_report` | global 10% + routed 20% | 30.005% | 0.313% | 4.400% | 0.702% / 3.009% / 6.139% | FAIL |
| `gov_report` | global 10% + routed 30% | 38.753% | 0.155% | 2.352% | 0.371% / 1.602% / 3.395% | FAIL |
| `multi_news` | global 10% + routed 10% | 21.270% | 0.971% | 9.954% | 2.134% / 8.515% / 14.948% | FAIL |
| `multi_news` | global 10% + routed 20% | 30.010% | 0.530% | 5.814% | 1.157% / 4.695% / 8.585% | FAIL |
| `multi_news` | global 10% + routed 30% | 38.770% | 0.309% | 3.446% | 0.687% / 2.793% / 5.244% | FAIL |

Bootstrap document-level, 10,000 resamples, seed `20260921` cho thấy các điểm
near-frontier không nên bị đọc như khác biệt tuyệt đối chính xác:

| Dataset / policy | `E_src` 95% CI | Mean missed mass 95% CI | P99 missed mass (point) |
|---|---:|---:|---:|
| `gov_report`, 10% + 20% | 30.003–30.008% | 0.242–0.400% | 4.400% |
| `multi_news`, 10% + 20% | 30.006–30.018% | 0.478–0.591% | 5.814% |

CI được bootstrap theo document, trong khi matrix chính là pooled decode-step
summary; chênh lệch nhỏ giữa point estimate và CI center là do weighting. P99
ở đây vẫn là descriptive point estimate, chưa phải confidence interval ổn định.

### Diễn giải

- Cấu hình global 10% + routed 10% cho headroom source-only rõ rệt: khoảng
  21.3% expansion. Tuy nhiên P99 missed mass còn 8.75% trên `gov_report` và
  9.95% trên `multi_news`.
- Cấu hình global 10% + routed 20% là **near-frontier**, không phải failed
  hypothesis hoàn toàn: `gov_report` đạt mean/P99 mass gate ở 30.005% expansion;
  `multi_news` vẫn P99 5.814% và expansion 30.010%. Sai lệch expansion 0.005–
  0.010 điểm phần trăm là sát ngưỡng và cần đọc cùng CI, nhưng vẫn fail literal
  theo gate đã khóa.
- Routed 30% giảm missed mass và output error rõ rệt, nhưng `E_src` tăng lên
  khoảng 38.8% trở lên. Đây là trade-off cost–tail-risk, không phải thiếu
  hoàn toàn headroom.
- Không có sweep nào đạt đồng thời ba gate trên cả hai dataset. Vì vậy chưa có
  cơ sở promote thành **deployable** router/physical KV implementation.

### Task dependence: GovReport và Multi-News

Multi-News có source attention kém nén hơn rõ ràng: mean K95 là 38.47% so với
24.29% trên GovReport; tỷ lệ head có K95 >25% là 85.45% so với 37.93%. Đây là
tín hiệu exploratory phù hợp với giả thuyết rằng source-attention
compressibility phụ thuộc task/document structure. Trace hiện chưa có source
boundary annotations hoặc số lượng article con, nên chưa được phép kết luận
nguyên nhân là multi-document structure; cần kiểm tra riêng ở holdout.

## Từ oracle expansion đến systems cost

E42 chưa đo `routing`, `gather` hoặc kernel execution. Effective cost cần được
định nghĩa riêng:

\[
R_{cost}=\frac{F_{router}+F_{exact}+F_{gather}}{F_{dense}}.
\]

Trong run hiện tại, `E_src` chỉ có thể được đọc như lower bound oracle cho
`F_exact`; chưa có số đo cho hai thành phần còn lại và chưa có index-overhead
được phân bổ vào E42. Vì vậy không được đổi trực tiếp `E_src=0.30` thành
`3.3x` hay bất kỳ speedup nào.

Ngay cả khi source attention giảm, E2E còn bị Amdahl giới hạn. Với `$f$` là
tỷ lệ decode time thực sự nằm ở source attention và `$s$` là speedup của riêng
phần đó:

\[
S_{E2E,max}=\frac{1}{(1-f)+f/s}.
\]

Run E41/E42 chưa đo `$f$`, TPOT, HBM hoặc kernel time; đây là evidence gap cần
đóng trước khi quyết định ngưỡng expansion có đủ để cạnh tranh mốc DFlash hay
không.

## Quality bridge còn thiếu

Chuỗi hiện tại mới là:

\[
\text{missed attention mass}
\rightarrow
\text{attention-context error}
\rightarrow ? \rightarrow
\text{logit/NLL/summary quality}.
\]

Output error đã được đo ở mức context như bảng trên, nhưng chưa chạy
generation dưới masked-attention policy. Chưa có `Top1Agreement`, ΔNLL,
ROUGE hoặc factuality. Do đó gate missed mass hiện là safety contract của
search, chưa được calibration với quality.

## Review kết luận

### CONFIRMED

- E41 hoàn tất 40/40 document, trace hợp lệ; layer 27 có head-wise
  heterogeneity và persistent head signal.
- E42 hoàn tất 40/40 document, 16 sweep global/routed trên mỗi dataset; không
  có cấu hình nào đạt đồng thời `E_src≤30%`, mean missed mass `≤1%` và P99
  missed mass `≤5%` trên cả hai dataset.
- E42 vẫn chứng minh có oracle headroom đáng kể: global 10% + routed 20% gần
  frontier trên GovReport, nhưng chưa đạt contract chung do expansion literal
  và tail risk trên Multi-News.

### EXPLORATORY

- Head 9 ổn định hơn các head khác trên cohort hiện tại; top-head stability
  giảm đáng kể trên Multi-News khi tập global mở rộng.
- Multi-News có K95 cao hơn GovReport, gợi ý source-attention compressibility có
  thể phụ thuộc task/document structure.
- Các số context-output error và document bootstrap CI là diagnostic; chúng
  chưa chứng minh logit fidelity hay summary quality.

### FAILED / INCOMPLETE

- Đây là **scientific gate failure của policy/config**, không phải lỗi vận hành:
  E42 đã chạy đủ 40/40, `error_samples=0`.
- Chưa có E43 block-score oracle trong run này. `STOP_BEFORE_E43` là đúng với
  vai trò **promotion gate**; nếu muốn chạy E43, chỉ nên gọi nó là locked
  diagnostic để phân biệt lỗi head sparsity với block granularity, không tune
  đến khi đạt.
- Chưa có all-layer evidence, long-decode 64/128 tokens, untouched holdout,
  E2E/TPOT/VRAM/throughput, effective cost, logit/NLL, ROUGE hoặc factuality.

### HIGHEST VERIFIED RUNG

**Verified through R7 (decision memo over completed R6 dev-search).** Artifact
chứng minh là Modal manifests/traces của `e41-head-concentration-20260921` và
`e42-oracle-hybrid-head-20260921`, cùng report này. Chưa verified bởi final
holdout, paper-scale multi-layer study hoặc systems benchmark cùng target.

### EVIDENCE GAPS

1. Stability mới ở layer 27 và cùng dev cohort; chưa biết head set có transfer
   qua layer, length, dataset hoặc model không.
2. `E_src` là source-only lower bound; chưa có `$R_{cost}$`, router/gather
   overhead hoặc Amdahl fraction.
3. Missed mass/output-context error chưa nối được với token logits, NLL và
   summary quality.
4. Cohort thực tế chưa có source dài hơn 13.5K tokens và output chỉ tối đa 32;
   chưa kiểm tra late-decode drift.
5. Baseline DFlash/EAGLE/Vanilla khác target/GPU/output budget; chưa được phép
   claim direct speedup.

### RECOMMENDED NEXT

Chạy **một locked diagnostic replication** trên untouched 20+20 document với
Qwen3-0.6B, các layer `{2, 7, 14, 21, 27}`, output `{32, 64, 128}`, giữ nguyên
selection rule và gate. Trace phải bổ sung per-layer E41/E42, output error,
document bootstrap CI và source-length buckets. Chỉ sau diagnostic này mới
quyết định giữa:

- dừng source-routing vì failure ổn định qua layer/length; hoặc
- mở một E43 block-score diagnostic đã khóa trước, không phải physical
  implementation/promotion.

Mọi run mới phải ghi run ID, model hash/revision, package/backend, prompt,
GQA mapping, formula `E_src`, raw trace path và trạng thái gate vào workboard.

## Kết luận ngắn

Kết quả hiện tại không nói “source routing hoàn toàn dead”. Nó nói chính xác
hơn: **head heterogeneity và oracle sparsification có thật; oracle hybrid
head chạm frontier chi phí trên GovReport, nhưng simple global/routed-head
decomposition chưa đồng thời thỏa compute contract và tail-risk contract trên
GovReport + Multi-News.** Vì vậy chưa promote router/physical KV, và chưa có
claim systems speedup hay quality.

Verified through R7 (decision memo over completed R6 dev-search). Not yet
verified by untouched holdout, multi-layer/long-decode study, or
apples-to-apples systems benchmark.
