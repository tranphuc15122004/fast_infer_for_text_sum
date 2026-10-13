# Kế hoạch thực nghiệm B — DFlash candidate uncertainty và speculation depth

Ngày cập nhật: **2026-10-13**.

**Trạng thái:** protocol đã được hiện thực hóa thành harness B0/B1/B2/B1-V
trong `speculation_experiments/` và `scripts/run_dflash_speculation_B.py`;
test CPU pass, nhưng **chưa chạy B0/B1/B2/B1-V trên B200**, chưa có kết quả
nghiên cứu hoặc controller mới. B3/B4 chưa implement vì phụ thuộc kết quả.
Các ngưỡng dưới đây là
**đề xuất đăng ký trước**, cần khóa trong campaign manifest trước đo dữ liệu.

**Mục tiêu:** xác định speculation depth hữu ích có biến đổi theo decoding
state, có tín hiệu rẻ dự đoán được biến đổi đó, và việc dùng tín hiệu có giảm
E2E latency sau khi tính đủ overhead hay không.

**Thư mục nghiên cứu:** `src/TrainingFree/`; artifacts ở
`outputs/dflash_speculation_B/<run_id>/`, gitignored.

**Phạm vi kiểm chứng:** inference training-free; CPU kiểm logic, schema,
causality; GPU kiểm checkpoint/backend, cache, greedy parity và latency thật.
Không có training/backward hoặc tiny-overfit trong validation.

**Thiết kế đánh giá:** replay paired trên cùng prefix; calibration và predictive
evaluation tách theo source group; E2E trên documents độc lập sau lock. Mỗi
phase có progress, coverage report, CI và decision memo. Mọi tín hiệu được
gắn thời điểm có sẵn để tránh rò thông tin tương lai.

**Kiến trúc dự kiến:** tái sử dụng DFlash bank/verifier và reference replay
hiện có; bổ sung collector candidate uncertainty cùng evaluator chuyên biệt
cho B. Controller chỉ được triển khai khi opportunity và predictability của
cơ chế tương ứng đã đủ bằng chứng.

## Phần I — Chúng ta cần biết điều gì?

### 1. Ba insight và hai quyết định khác nhau

| Insight | Câu hỏi | Bằng chứng cần có |
|---|---|---|
| Opportunity | Depth tối ưu có biến đổi và best fixed bỏ lỡ lợi ích đáng kể? | Cost/acceptance surfaces từ same-prefix replay; oracle gap vượt timing noise |
| Predictability | Tín hiệu có sẵn tại thời điểm quyết định có dự đoán accepted prefix/cost không? | Predictive evaluation trên source groups không dùng calibration |
| Actionability | Quyết định dựa trên tín hiệu có giảm latency thật? | Paired E2E tốt hơn best fixed, tính extraction/controller/cache overhead, output exact |

Phân biệt rõ hai cơ chế:

| Cơ chế | Thời điểm quyết định | Action | Chi phí có thể giảm |
|---|---|---|---|
| Adaptive draft length | Trước draft forward | Chọn gamma; draft lại đúng shape gamma+1 | Draft và target verification |
| Adaptive verification prefix | Sau draft forward 15 candidates | Chọn k; verify anchor + k candidates đầu | Target verification; draft 15 đã phát sinh |

Entropy của candidate trong round hiện tại chỉ dùng được cho quyết định thứ
hai. Muốn chọn draft length phải dùng tín hiệu trước draft hoặc uncertainty
của các rounds trước và kiểm chứng riêng khả năng dự đoán xuyên rounds.

**Giả thuyết tổng quát:** có states chỉ tận dụng được vài speculative tokens,
có states tận dụng block dài; uncertainty của drafter và/hoặc acceptance
history giúp phân biệt các states này. Đây là giả thuyết cần kiểm chứng,
không phải thuộc tính đã được xác nhận của DFlash.

### 2. Thứ tự thực nghiệm

| Bước | Làm gì? | Để làm gì? | Kết quả nhận được |
|---|---|---|---|
| B0 | Kiểm runtime, cache, instrumentation, AR/native/adapter parity | Loại lỗi verifier hoặc collector trước nghiên cứu | Correctness report và supported shape grid |
| B1 | Gamma sweep 3/7/11/15, draft riêng ở cùng state | Đo opportunity của adaptive draft length | L/T/J theo state/action, gamma oracle, marginal utility |
| B2 | Thu entropy/confidence candidate, parent signals, history | Đo predictability và phần thông tin bổ sung của drafter | Survival calibration, Brier, correlations, cost regret |
| B1-V | Draft 15 một lần; replay verifier prefixes 3/7/11/15 | Đo opportunity riêng của adaptive verification | Verification cost curve, k oracle, gap sau draft cost/overhead |
| B3 | Nếu B2 + B1-V có triển vọng, chạy controller verification prefix | Đo actionability | Paired E2E, exactness, overhead và action distributions |
| B4 | Khi cần, dùng uncertainty các rounds trước dự đoán round sau | Đánh giá khả năng giảm cả draft computation | Heldout cross-round prediction, rồi draft-length E2E nếu đủ bằng chứng |

B1 được phân tích trước. Collector B2 có thể thu ngay trong lượt gamma 15
của B1 để tiết kiệm forward, nhưng không đưa extraction overhead vào cost
baseline B1. B1-V dùng cùng states khi có thể, sau khi verifier-prefix harness
được validate; không cần controller để đo cost surface.

Nếu B1 gap nhỏ, giảm ưu tiên adaptive draft length. Vẫn có thể khảo sát B2
và B1-V vì proposals của draft shape nhỏ khác prefix của block 15. **B1 fail
không tự động chứng minh adaptive verification không có tiềm năng.**

## Phần II — Protocol chung và định nghĩa đo lường

### 3. Cấu hình, dữ liệu, split và quy mô

Primary runtime dự kiến là B200 theo [kế hoạch A](experiment_plan_A_13_10_26.md)
và [hồ sơ server](../../../docs/server_environment.md): Python 3.12 hệ thống,
local model/dataset snapshots, dependency đã mirror. Local CPU dùng profile
local sẵn có, không đo latency GPU. Kế hoạch này không khởi chạy cloud hoặc
cài package mới.

| Thành phần | Giá trị phải khóa |
|---|---|
| Models | Qwen3-4B target + Qwen3-4B-DFlash-b16, frozen weights, snapshot hashes |
| Runtime | BF16, eval/inference mode, actual target/draft backend và versions; cùng GPU signature |
| Workload | Batch 1, một process/card; greedy temperature 0, no thinking |
| Context | Full target KV và full draft context ở mọi thực nghiệm B |
| Gamma / k | 3,7,11,15; block gồm một anchor + candidates |
| Natural output | EOS bật; cap pilot 1024, dev/confirm 2048; báo capped fraction |
| Data | GovReport, Multi-News, QMSum; source grouping và exposure audit |
| Seeds | Cohort/order 42; bootstrap 20261013; lưu schedule cụ thể |

FA4 production path dự kiến theo campaign A; xác nhận bằng probe và forward
thật trên server, không suy backend hoạt động từ import thành công. Diagnostic
eager/SDPA có runtime và evidence riêng; không trộn timings vào primary.

Canonical corpus: `data/longbench_100_14k/{gov_report,multi_news,qmsum}.jsonl`
hoặc server paths tương ứng. Token lengths dùng tokenizer target trên prompt
thực. Bins mô tả: [0,4K), [4K,8K), [8K,12K), [12K,16K); không padding để
tạo bins. QMSum nhiều queries cùng conversation là **một source group**.

| Giai đoạn | Quy mô đề xuất | Vai trò |
|---|---|---|
| B0/pilot | 2 groups/dataset = 6; tối đa 6 prefixes/group | Debug, đo thời gian/cell và xác nhận đủ artifacts |
| B1/B2/B1-V khám phá | 10 GovReport + 10 Multi-News + 4 QMSum = 24 groups; mục tiêu 15 prefixes/group, range 10–20 | Opportunity, grouped predictive validation, chọn signal và cost buckets |
| Dev E2E trước lock | Cùng pool khám phá, đủ các fixed controls và surviving controllers | Chọn best fixed E2E và lock policy |
| Xác nhận | Mục tiêu 18/18/12 groups mới; dùng quota thật sau exposure audit | Confirm frozen protocol, đủ 3 timed repetitions; 5 là default |

Exposure audit ngày 13/10 của campaign A ghi pool sạch còn 37 GovReport,
33 Multi-News và **3 QMSum groups**, xem
[bản ghi exposure](../docs/context_adaptive_dflash/experiment_progress_2026-10-09.md).
Đó là trạng thái tài liệu hiện có, cần audit lại cả exposure mới của A/B trước
lock. Với pool hiện tại có thể chạy confirm 18/18/3 và ghi **partial quota**;
3 QMSum groups không đủ xác nhận mạnh về generalization riêng QMSum. Nếu không
bổ sung nguồn sạch, kết luận chính phải giới hạn ở GovReport/Multi-News, QMSum
là kết quả exploratory. Không nhân queries để tăng số documents độc lập.

Manifest khóa `headline_datasets` trước confirm. Nếu chỉ có 3 clean QMSum
groups, headline confirm macro dùng GovReport/Multi-News; macro cả ba datasets
vẫn được báo như partial/exploratory, không dùng cho gate generalization đủ
ba datasets. Không bỏ QMSum khỏi headline vì kết quả đo của nó bất lợi.

Không tạo split B mâu thuẫn source-group/exposure registry của A. B2 khám phá
dùng 4-fold grouped cross-validation trong 24 groups: toàn bộ states/repeats
của group vào cùng fold; phân tầng dataset khi quota cho phép. Mọi cutpoints,
cost fitting và chọn biến chỉ đọc training folds. Chọn estimator bằng inner
grouped validation trong training folds nếu thử hơn một cấu hình. Các metrics
out-of-fold chỉ là development evidence vì đã dùng để chọn hướng nghiên cứu.

Sau dev, fit tables cuối trên pool khám phá, lock policy/baselines và đánh giá
confirm một lần. Nếu A đã dùng một confirm group để phát triển ý tưởng B,
group đó thành exposed. Có thể chia sẻ cohort confirm A/B đã đăng ký chung
trước khi xem kết quả, nhưng không gọi các analyses thích nghi sau đó là untouched.

### 4. Reference rollout, prefix selection và history

Mỗi document có một reference rollout **full-context gamma=15 cố định** đã
qua greedy parity với target-only AR. Trace chứa tất cả rounds, token IDs,
processed output count, pending anchor, outcomes và uncertainty nếu thu.
Không lấy states từ controller adaptive để làm reference chính.

Chọn tối đa 15 regular states theo ba strata progress [0,1/3), [1/3,2/3),
[2/3,1): tối đa 5 states/stratum, phân bố theo processed output positions,
bao gồm state đầu nếu đủ điều kiện. Chọn nearest reference round tại các
mốc cách đều trong từng stratum, tie chọn round sớm hơn; deduplicate.
Không lấp bằng timing repetitions hoặc chuyển tùy ý state đầu sang stratum
cuối. Trace ngắn có ít hơn 10 states vẫn giữ và báo actual coverage.

Regular state phải còn ít nhất 16 reference output tokens trước EOS/cap
và đủ output budget cho mọi gamma. Điều kiện này dùng future output **chỉ
để chọn cohort chẩn đoán**; báo selection bias và số boundary states bị loại.
EOS/cap states được kiểm correctness riêng; E2E luôn tính đủ mọi boundary.

`progress_offline = processed_output / reference_output_length` chỉ dùng
phân tầng/phân tích. Online features dùng processed output count, round index
và fraction của **configured output cap**; không dùng độ dài summary tương
lai hoặc vị trí EOS tương lai. Với capped trace, “cuối” nghĩa là cuối phần đã
quan sát. Phân tích natural-ended và capped traces riêng.

History trước round t lấy **toàn bộ rounds reference trước t**, kể cả rounds
không được chọn replay. Các biến đăng ký: previous L, mean L/15 ở tối đa
4 rounds trước, EMA L/15 alpha=0.1, previous all-accepted và missing-history
flag. Không dùng L_t của action đang replay để tạo feature cho round t.
First round có history missing, không gán 0 như một observed failure.

State identity dùng dataset/source_group_id, prompt IDs hash, processed prefix
IDs hash, anchor ID/position và round index; **không chứa gamma/action**.
Action key tách riêng, để pairing các gamma đúng cùng state. Không tái sử
dụng action-specific `state_hash` cũ làm group key cho gamma sweep.

Ưu tiên fork live reference cache/bank. Nếu rebuild từ token prefix, kiểm
replayed gamma 15 có proposals/L/anchor nhất quán với reference và so cache,
hidden features trong tolerance đã khóa. Prefill toàn prefix có thể khác
incremental KV về số học; mismatch phải được điều tra, không dùng hai kiểu
state như các observations tương đương.

### 5. Timing và accounting

Ký hiệu tại regular state:

$$L_t(a)=\text{số candidates accept liên tiếp},\quad C_t(a)=1+L_t(a).$$

Anchor đã được chọn nhưng chưa được xử lý; correction/bonus sau verification
là pending anchor kế tiếp. Không tính correction/bonus như thêm một candidate
accepted của round hiện tại. Trong code cũ `mean_accept_length` gồm anchor;
report B luôn tách L và C. EOS trimming/final pending emission có accounting
riêng, không dùng công thức regular round để che boundary.

Cost chính B1: synchronized wall round gồm gather/full-context preparation
thật cần cho action, draft, target verify, matching/accept decision, crop KV
và cập nhật draft bank từ retained target features. Không chạy source-attention
collector của A. Log CUDA events cho draft/gather/verify/bank update và GPU
span; các component lồng nhau không được cộng hai lần.

Clone/rebuild cache, restore scratch, lưu JSONL và uncertainty collector chỉ
phục vụ diagnostic nằm ngoài T_B1; log riêng setup_ms/extraction_ms. State
được restore độc lập cả target KV, bank length và active bank values, collector
scratch/RNG. Không gọi empty_cache/GC trong measured span; không dọn giữa các
timed actions khiến allocator regime khác E2E mà không ghi scope.

Timing B1 và B1-V: warmup tối thiểu 3 lần/shape/context bucket sau kernel JIT,
6 repetitions/state/action; permutation action order đổi theo repetition,
lưu schedule. L/proposals của greedy repeats phải giống nhau. Repeats chỉ
đo timing, không tăng unique predictive samples hoặc CI sample count.

Ba repeats đầu dùng chọn oracle/best fixed; ba repeats sau dùng đánh giá
cost của lựa chọn đã khóa tại state. Báo thêm optimistic oracle dùng cả 6
repeats và paired replicate spread. Nếu 3+3 quá noisy, pilot phải quyết định
tăng repeats trước campaign; không chọn lại oracle trên evaluation repeats.
Đây là kiểm soát min-selection bias, không tạo predictive heldout documents.

Cost extraction B2 có lượt instrumented/uninstrumented cùng proposals.
B1 timing không bao gồm optional extraction; B3/B4 E2E phải gồm extraction
và mọi synchronization thật cần để chọn action.

## Phần III — Thực nghiệm và cách diễn giải

### 6. B0 — Correctness trước nghiên cứu

**Làm gì:** kiểm prefill, draft/verify từng gamma, reject tại candidate đầu,
giữa, all accepted, multi-round, EOS trong accepted prefix, pending EOS,
cap/tail shape, positions và cache rollback; collector bật/tắt.

**Để làm gì:** phân biệt prediction kém với lỗi adapter/kernel hoặc lỗi thu
tín hiệu. Pilot H100 cũ chưa đạt greedy parity, nên không bỏ bước này.

So target-only AR ↔ native DFlash ↔ full adapter trên cùng tokenized prompts,
cùng greedy backend/stopping. Kiểm tất cả gamma 3/7/11/15, không chỉ 15.
Native control phải được cấu hình đúng gamma và EOS/cap accounting.

**Nhận gì:** `parity.json` chứa exact token equality, first divergence,
branch, logits/top-1 margin quanh divergence, target KV/bank lengths và
instrumentation equality. Collector không được thay proposals/L/output.

**Kết luận:** có mismatch/cache violation → `invalid_correctness`, dừng
research/performance gate để điều tra; không kết luận B vô dụng. Diagnostic
logs để debug được lưu riêng. Chỉ mở regular replay và E2E claim sau parity
pass trên production runtime. Nếu một shape unsupported, campaign 4-gamma
là incomplete; reduced grid chỉ là protocol mới được khai báo rõ.

### 7. B1 — Optimal draft length có biến đổi hữu ích không?

**Can thiệp:** ở cùng prefix, anchor và full bank, chạy độc lập
gamma ∈ {3,7,11,15}; mỗi gamma có draft forward riêng và target-cache fork.
DFlash bidirectional attention khiến proposal prefix từ block dài không đại
diện cho draft shape ngắn.

**Thu:** proposals từng gamma, L, C, draft/verify/round T, input/current context
length, processed output, progress stratum, dataset, source group, repetition,
boundary/censor flags và reference-history features.

$$J_t(\gamma)=\frac{T_t(\gamma)}{1+L_t(\gamma)},\qquad
\gamma_t^{oracle}=\arg\min_\gamma J_t(\gamma).$$

Oracle dùng timing selection repeats, cost report dùng evaluation repeats.
Tie chọn gamma nhỏ hơn; báo near-optimal set J ≤ 1.02 × min J từ selection
repeats. Argmin biến đổi trong tập near-tie không đủ chứng minh opportunity.

#### 7.1. Phân tích nhận được

| Phân tích | Cách làm | Insight |
|---|---|---|
| Oracle distribution | Histogram gamma; near-optimal coverage của từng fixed gamma; within-document transitions | Có fixed gamma gần tối ưu phần lớn states hay không? |
| Context/progress | Heatmap gamma oracle và J regret theo dataset/input bins/progress | Heterogeneity nằm ở context hay giai đoạn sinh? |
| Marginal benefit | Paired ΔL và ΔT cho 3→7, 7→11, 11→15; fraction L_long ≤ L_short | Block dài có tạo thêm accepted tokens tương xứng cost? |
| Proposal change | Token-prefix equality/divergence giữa draft shapes | Gain đến từ depth hay shape-dependent proposal quality? |
| Tail survival | P(L≥3/7/11/15); P(L15≤3), P(L15≥11); cùng margins/cost | Có nhóm states accept sâu đáng khai thác? |
| Robustness | Oracle selection/evaluation repeats; replicate spread; near-ties | Heterogeneity thật hay noise của timing? |
| Simple fixed controls | Global fixed; dataset-specific fixed và context/progress rules fit trên train folds | Có cần tín hiệu uncertainty sau khi đã biết regime đơn giản? |

Không mặc định L tăng đơn điệu theo gamma: proposals thay đổi theo draft
shape. Báo cả gamma dài accept ít hơn gamma ngắn và nguyên nhân candidate
prefix đổi. Rare long-accept states chỉ là insight mô tả nếu không tạo gap
cost hữu ích hoặc không nhận diện được trên documents mới.

#### 7.2. Oracle gap, best fixed và giới hạn

Best fixed diagnostic: gamma tối thiểu mean J, tính mean states/document,
mean documents/dataset rồi mean ba datasets. Gọi là **best fixed hindsight
trên diagnostic cohort**, không phải best fixed đã chọn từ dev cho E2E.

Metric chính so cơ hội local là document/dataset-macro phiên bản của:

$$G_J=1-\frac{\sum_t J_t(\gamma_t^{oracle})}
                     {\sum_t J_t(\gamma_{fixed}^{diag})}.$$

Các tổng trong công thức biểu diễn cùng states; khi báo macro áp cùng trọng
số document/dataset ở tử/mẫu. Với oracle chọn từ repeats độc lập, G_J evaluation
có thể âm; đó là thông tin về instability, không được clip thành 0.

Giữ thêm đúng diagnostic người dùng đề xuất:

$$G_T=1-\frac{\sum_t T_t(\gamma_t^{oracle})}
                     {\sum_t T_t(\gamma_{fixed}^{diag})}.$$

G_T so tổng **chi phí round**, dù hai phía commit số tokens khác nhau; có thể
âm khi oracle chọn block dài hiệu quả hơn theo J. Vì vậy G_T không phải primary
cost-per-token gain và không dùng một mình để quyết định dừng. Báo bổ sung
ratio-of-sums T/C như resource diagnostic; mỗi metric có cùng pairing/weights.
Tất cả đều là local replay opportunity, chưa phải E2E speedup hoặc upper bound
chặt cho một adaptive rollout với state occupancy khác.

#### 7.3. Tín hiệu nào dẫn tới kết luận nào?

| Tín hiệu trên evaluation timings, CI theo document | Kết luận / hành động |
|---|---|
| G_J ≥5%, CI95 lower >2%; ≥20% states có best fixed regret >5%, không chỉ near-ties | Draft-length opportunity đáng ưu tiên; tiếp tục predictability của pre-draft signals/B4 |
| CI95 upper G_J ≤2% | Ít opportunity thực tiễn trong grid/workload hiện tại; ưu tiên best fixed, dừng draft-length controller mới |
| Histogram gamma đổi nhiều nhưng G_J nhỏ hoặc repeat oracle không ổn định | Không đủ bằng chứng biến đổi hữu ích; kiểm timing/near-ties trước |
| Một short gamma near-optimal ≥90% states, tail states không đóng góp gap | Có fixed control mạnh; contribution adaptive yếu |
| Phần lớn L15≤3 nhưng có subset L15≥11, subset tạo gap J rõ | Useful depth phân hóa; kiểm tín hiệu nhận diện long-survival states ở B2/B4 |
| Chỉ context/dataset rule gần oracle trên unseen groups | Regime-dependent tuning đã đủ; uncertainty cần chứng minh giá trị bổ sung |
| CI bao cả ≤2% và ≥5%, hoặc coverage thiếu | `inconclusive`; thêm documents trước khi chốt, không dùng p>0.05 để bác bỏ |
| Gap ổn định giữa 2% và 5%, CI đủ hẹp | Opportunity nhỏ hơn mức ưu tiên đã khóa; lưu insight, ưu tiên fixed tuning, không gọi hypothesis sai |

Ngưỡng 5%, 2%, 20%, 90% là mức thực tiễn đề xuất để khóa trước run, không
là quy luật của DFlash. Evidence về opportunity chủ yếu từ gap/regret; entropy
có tương quan không làm B1 tự động pass.

### 8. B2 — Candidate uncertainty có dự đoán accepted prefix không?

**Làm gì:** giữ full context, draft 15 candidates, thu phân phối uncertainty
của chính DFlash trước target verification. Dùng outcomes cùng round làm
labels offline; chưa thay action trong generation.

**Để làm gì:** xác định tín hiệu drafter có dự đoán được future acceptance
và bổ sung thông tin sau khi đã biết context, progress và history hay không.

#### 8.1. Tín hiệu và thời điểm có sẵn

Với raw candidate logits d_i, tính FP32, signal temperature = 1.0:

$$p_i^D=\mathrm{softmax}(d_i),\quad
H_i^D=-\sum_v p_i^D(v)\log p_i^D(v),\quad
h_i^D=H_i^D/\log|V|.$$

$$c_i^D=\max_v p_i^D(v),\qquad
m_i^D=p_{i,(1)}^D-p_{i,(2)}^D.$$

Drafter hiện chọn argmax, nên c_i là xác suất candidate được propose.
Log thêm logit top-1/top-2 margin để phân tích; không tính entropy từ one-hot
sau argmax. Parent entropy dùng raw target logits tạo pending anchor:
prefill last logits hoặc previous verification row L_previous.

| Nhóm | Fields | Có sẵn khi nào? | Được dùng cho |
|---|---|---|---|
| Regime R | Prompt/current length, processed count, round index, cap fraction | Trước draft | Draft length hoặc verification |
| History A | Previous L, mean4/EMA L/15, previous all-accepted, missing flag | Trước draft, reference rounds <t | Cả hai |
| Parent H/C | Normalized entropy, chosen anchor probability, probability/logit margin | Trước draft | Cả hai |
| Drafter D-H | h_1…h_15; mean/max/sum prefix | Sau draft 15, trước verify | Verification của round hiện tại |
| Drafter D-C | c_1…c_15, margins; min/mean confidence prefix | Sau draft 15, trước verify | Verification của round hiện tại |
| Target candidate entropy | Logits sau current verification | Sau action | Explanation offline; cấm làm deployable input của current action |

Không cần target forward bổ sung để lấy parent entropy hoặc candidate draft
logits. Nhưng full-vocabulary softmax/reduction, device transfer và sync
**có overhead**, cần đo và tính ở B3. Chỉ lưu scalar vectors/top-2 summaries,
token IDs; không lưu vocabulary distributions mọi round ra đĩa.

#### 8.2. Labels và ba phân tích cơ chế

Primary labels tại mỗi i=1…15:

$$Y_{t,i}=\mathbf1[L_t(15)\ge i].$$

All-accepted L=15 là right-censored đối với depth >15, nhưng Y_i đều được
quan sát đúng cho i≤15. Không suy extrapolated acceptance sau position 15.

| Phân tích | Cách làm cụ thể | Câu hỏi trả lời |
|---|---|---|
| Candidate rejection risk | Trong states có L≥i−1, phân entropy/confidence theo bins fit train folds; đo P(L=i−1 \| L≥i−1, signal) | Khi candidate thực sự được reach, uncertainty có dự đoán rejection? |
| Prefix survival | Dự đoán Y_k từ mean/max/sum h_1:k, min/mean c_1:k, h_k/c_k, R/A | Uncertainty tổng hợp có phân biệt prefix sống dài? |
| Confident-but-wrong | c_i≥0.9 trong risk set; tỷ lệ reject, so với overall và low-confidence bins | Drafter có tự tin sai, khiến confidence threshold không đáng tin? |

Threshold 0.9 là diagnostic đăng ký trước; báo thêm đường liên tục theo
confidence để tránh diễn giải một bin tùy ý. Candidate mismatch **sau rejection
đầu** không được gán như rejection độc lập ở trajectory thật. Target rows
sau đó đã conditioning trên suffix có thể sai, không là teacher AR labels.

#### 8.3. Predictive controls và estimator nhỏ

Tất cả variants có cùng regime R để đo thông tin **bổ sung** của signal.
Tên signal-only dưới đây nghĩa là chỉ thêm nhóm signal đó bên cạnh R.

| ID mới, không relabel controller V1 | Inputs |
|---|---|
| R | Context + deployable generation progress |
| R-H | R + parent entropy |
| R-A | R + reference acceptance history |
| R-C | R + parent confidence/margin |
| R-HA | R + parent entropy + history |
| R-PreAll | R + tất cả pre-draft signals |
| R-DH | R + current DFlash entropy summaries |
| R-DC | R + current DFlash confidence/margin summaries |
| R-A-DH | R + history + DFlash entropy |
| R-A-DC | R + history + DFlash confidence/margin |
| R-DAll | R + history + parent signals + DFlash entropy/confidence |

So R-DH/R-DC với R-H/R-C; contribution của entropy so R-A-DH với R-A,
và R-DAll với cùng inputs **bỏ D-H**. Nếu confidence đã đủ, không gọi gain
của DAll là bằng chứng entropy riêng.

Estimator đầu tiên là smoothed survival curves theo bins: context 4K-token
buckets, output progress [0,128), [128,512), [512,∞), history EMA <0.5/≥0.5,
signal quartiles fit trên training groups. Để tránh tổ hợp bins rỗng, primary
table giới hạn R + một signal summary; interaction tables chỉ khi đủ support.
Beta(1,1) smoothing cho mỗi survival threshold. Leaf cần ≥8 unique states
từ ≥3 source groups; thiếu support backoff signal → regime → global training
curve, log fallback. Repeats không tăng support. Out-of-support context
được flag riêng; cost table không backoff sang context ngắn hơn.

Primary per-threshold entropy summary là max h_1:i; confidence summary là
min c_1:i. Mean/sum và h_i/c_i là sensitivity analyses, chọn trên inner dev
folds trước confirm. Sửa survival thành nonincreasing theo i bằng cumulative
minimum, clamp [0,1]; báo số lần repair. Cùng smoothing/support/fallback cho
các variants. Nếu table quá thưa, báo limitation; không chuyển sang predictor
phức tạp rồi đánh giá tiếp trên cùng confirm set.

#### 8.4. Metrics và đọc kết quả

$$BS=\frac1{15}\sum_{i=1}^{15}
       \operatorname{mean}_{documents}\operatorname{mean}_{states}
       (\widehat p_{t,i}-Y_{t,i})^2.$$

Pooled headline mean thêm đều theo dataset. Integrated Brier trên 15 positions
là primary; báo riêng k=3/7/11/15, reliability plots, calibration-in-the-large,
bin support/fallback và tail-positive counts. Định nghĩa incremental skill:

$$\Delta BS_{rel}=\frac{BS_{control}-BS_{signal}}{BS_{control}}.$$

Nếu BS_control=0, relative skill là undefined, dùng absolute difference.
Spearman giữa entropy summaries và L, confidence/history và L: báo pooled,
theo dataset/progress/context, within-document khi đủ variation. CI theo
source groups; Spearman chỉ là mô tả discrimination, không thay calibration
hoặc đánh giá decision utility. Tăng context/progress control để kiểm correlation
có chỉ phản ánh stage/length hay không.

Predict **gamma oracle B1** chỉ dùng pre-draft variants R…R-PreAll hoặc lagged
signals B4. Outcome models riêng từng gamma vì proposals shape khác nhau.
Current entropy của gamma15 không phải input deployable cho current gamma.
Predict **k oracle B1-V** được dùng D-H/D-C cùng round. Accuracy chỉ secondary;
primary decision metric là cost regret:

$$r_t=J_t(\widehat a_t)/J_t(a_t^{oracle})-1.$$

Score decisions bằng evaluation timings độc lập, so fixed/regime-only regret;
không lấy gamma15 survival rồi coi L_gamma=min(L15,gamma) trong B1.
Do oracle action được chọn từ timing repeats khác, measured regret đôi khi âm;
báo nguyên giá trị và variability, không clip âm để làm predictor trông tốt hơn.

| Tín hiệu quan sát trên out-of-fold groups | Kết luận / hành động |
|---|---|
| D-H/D-C cải thiện Brier ≥5% tương đối so R, CI95 improvement lower >0, calibration đủ tail support | Drafter uncertainty có predictive value; xét B1-V decision utility |
| Thêm D-H vào R-A hoặc confidence control cải thiện ≥2% tương đối, CI lower >0 | Có thông tin entropy bổ sung sau history/confidence; giữ nhánh combined để đánh giá |
| Parent entropy correlate nhưng không cải thiện so R-A/R | Chưa có contribution parent entropy; loại khỏi controller primary |
| D-H không bổ sung nhưng D-C/history tốt | Giữ confidence/history; bỏ entropy khỏi claim và policy primary |
| Brier tốt hơn nhưng k/gamma regret không giảm | Dự đoán acceptance chưa giúp cost objective; chưa triển khai claim adaptive hữu ích |
| Low entropy/high confidence vẫn reject thường xuyên và curves không tách rõ | Uncertainty đơn giản có thể thiếu target alignment; giữ counterexamples, ưu tiên tín hiệu khác |
| Gain chỉ trên training groups hoặc biến mất trong từng regime | Có thể overfit/confounding; chưa kết luận predictability |
| CI bao zero và meaningful threshold; ít long-survival groups | `inconclusive`; tăng source groups/tail coverage, không coi số rounds lớn là sample size |

5%/2% Brier skill là ngưỡng screening đề xuất, không tự chứng minh actionability.
Nếu CI95 upper improvement thấp hơn ngưỡng đã khóa, kết luận chưa có contribution
thực tiễn được hỗ trợ cho signal đó trong workload này; không khẳng định entropy
vô dụng với mọi model/task. Confirm dùng estimator/contrasts đã khóa trên dev.

### 9. B1-V — Có opportunity riêng cho verification prefix không?

**Làm gì:** ở mỗi state, draft full block 15 một lần, giữ cùng candidate IDs
và uncertainty. Với mỗi k=3/7/11/15, target verify anchor + k candidates đầu
bằng cache fork riêng; cập nhật/cache bank như một round thật.

**Để làm gì:** xác định target verification giảm k có tiết kiệm đủ chi phí
để bù committed tokens bị mất hay không, khi draft cost đã cố định.

Với causal exact verifier, label kỳ vọng L_t^V(k)=min(L_t(15),k). Phải kiểm
thật điều kiện này và target next-anchor/output parity trên production kernel;
BF16/backend shape differences có thể làm top-1 khác. Không mặc định identity
là bằng chứng correctness. Unverified suffix bị bỏ, không đưa vào KV/bank.

Thu T_D(15), T_V(k), matching/crop/bank update T_U(k), extraction T_E và
controller/dispatch T_Q. Cost diagnostic cơ bản:

$$J_t^V(k)=\frac{T_D(15)+T_V(k)+T_U(k)}{1+L_t^V(k)}.$$

Báo thêm overhead-adjusted J có T_E + T_Q đo/profiling cho từng policy, và
break-even overhead: phần tiết kiệm ms/token so best fixed k còn lại sau
extraction. T_D nằm trong tử vì đang tối ưu cost cả round trên expected commits;
policy này là local cost heuristic, không phải lời giải tối ưu dài hạn.

Dùng timing 3-selection/3-evaluation và metric G_J như B1, action thay bằng k.
Oracle cùng proposals là upper reference offline; so với **best fixed k với
draft15** và với best fixed **draft gamma** trong dev E2E, hai controls khác nhau.

| Tín hiệu | Kết luận / hành động |
|---|---|
| Gap overhead-adjusted ≥5%, CI lower >2%, k oracle ổn định ngoài near-ties; B2 giảm regret | Đủ lý do thử B3 |
| T_V(3)≈T_V(15), hoặc fixed short k/global gamma đã gần tối ưu | Verification-prefix có ít tiềm năng ở backend này, dù uncertainty correlate |
| Oracle gain chỉ xuất hiện khi bỏ T_D hoặc bỏ extraction/sync cost | Chưa có opportunity đúng accounting; dừng controller claim |
| B1 gap nhỏ nhưng B1-V gap rõ | Chuyển trọng tâm sang verification control; không gọi đó là giảm draft computation |
| B1-V gap nhỏ nhưng B1 gap rõ | Cân nhắc pre-draft signals/B4; verification-only có thể không đủ |
| CI rộng/unsupported prefix cache | `inconclusive` hoặc `invalid_correctness`; chưa đưa B3 vào full campaign |

### 10. B3 — Controller có giảm E2E latency không?

**Điều kiện mở:** B0 parity, B2 signal có heldout predictive value và B1-V
cho thấy opportunity sau overhead. Chỉ thử curves/tables nhỏ đã khóa.

Sau draft15 và thu features s_t:

$$\widehat C_t(k)=1+\sum_{i=1}^k\widehat P(L_t(15)\ge i\mid s_t),$$

$$\widehat k_t=\arg\min_{k\in\{3,7,11,15\}}
 \frac{\widehat T_D(15)+\widehat T_V(k)+\widehat T_U(k)
       +\widehat T_E+\widehat T_Q}{\widehat C_t(k)}.$$

Extraction thực hiện trong chính round, reused logits; không draft forward
thứ hai, không thêm target forward để lấy entropy. Cost table theo current
context và runtime, frozen sau dev. No-support action được loại; fallback best
fixed k/gamma đúng cơ chế và production shape đã validate. Mọi override logged.

Khi truncate, L≤k; retain anchor + L verified candidates, crop target KV tới
n_t+1+L, append đúng target features của retained tokens. Pending anchor kế
tiếp lấy verifier row L, kể cả all-k-accepted. Discard suffix draft15; không
tái dùng suffix như tokens đã verify. EOS/cap xử lý cùng controls.

#### 10.1. Comparisons E2E bắt buộc

| Variant | Draft | Verify | Vai trò |
|---|---|---|---|
| Target-only AR | Không | AR step | Correctness và latency reference |
| DFlash fixed15 | gamma15 | 15 | Vanilla control |
| Best fixed draft gamma | Chọn trên dev từ 3/7/11/15 | Cùng gamma | Primary speed control; chặn gain chỉ do dùng block ngắn |
| Best fixed verification k | Luôn 15 | k cố định, chọn trên dev | Chặn gain chỉ do truncate bằng constant |
| Regime-only verification | Luôn 15 | R | Chặn gain chỉ do context/progress |
| Entropy-only verification | Luôn 15 | R + D-H | Đo entropy contribution; chỉ chạy nếu B2 giữ signal |
| Confidence-only verification | Luôn 15 | R + D-C | Khi B2 giữ confidence |
| History-only verification | Luôn 15 | R + history | Đo history control |
| Combined verification | Luôn 15 | Inputs B2 đã chứng minh bổ sung | Chỉ chạy khi có lý do từ B2 |

History trong B3 dùng outcomes **trước round từ chính rollout policy**; đây
là causal feature deployable, nhưng khác reference history B2. Báo distribution
shift/support/fallback; capped k chỉ cho biết L_full≥k nếu all accepted, không
coi L=k là full15 rejection. Primary history update dùng acceptance fraction
L/k cùng executed-k và all-accepted/censor flag; estimator phải được dev-calibrate
cho định nghĩa này trước lock. R-A reference L/15 của B2 chỉ screening; không
âm thầm reuse như cùng distribution. Có thể lấy calibration rollouts deployable
trên training groups với policy frozen iteration count=1 trước dev/confirm.
Không chạy extra full15 verifier online để phục hồi reference history.

Nếu history deployment calibration chưa đủ, entropy/confidence-only controller
vẫn có thể được đo độc lập; history/combined cell ghi unavailable. Không dùng
uncensored-history assumption để tạo gain giả.

Trong dev chọn best fixed bằng geometric document/dataset-macro E2E latency,
không dùng best mean acceptance hoặc B1 local J để chọn E2E baseline. Lock
fixed gamma, fixed k, inputs, bins, fallback, history definition và policy trước
confirm. So instrumented/uninstrumented fixed15 để đo extraction distortion;
primary controls tắt collector không cần cho inference của chúng.

Warmup 3 runs/variant trên dev-only shapes, ngoài measured requests. Confirm
5 timed repetitions/request/variant, tối thiểu 3; fresh caches/request-local
history, xáo thứ tự variants trong mỗi repetition block. Fixed outputs greedy
phải giống AR từng token. Natural EOS bật; fixed-output scaling là secondary
campaign riêng, không dùng output sau EOS để claim summarization quality.

#### 10.2. Metrics, success và failure

Primary E2E wall time từ trước target prefill tới khi generation/finalization
kết thúc, CUDA sync ở hai đầu; gồm full bank construction, extraction,
controller, verification, crop/update, fallback và boundary. Model load/JIT,
warmup và serialization sau generation nằm ngoài; scope giống controls.

Thu E2E, TTFT, decode/TPOT, mean L và C, rounds/request, gamma/k frequencies,
controller/extraction/sync cost, wasted proposals, fallback/support, memory,
output IDs/hash/length và ROUGE nếu có references. Breakdown instrumented là
diagnostic; E2E primary giữ production synchronization pattern.

Với speedup S=T_control/T_policy: mean log S qua repetitions của mỗi document,
mean documents/dataset, mean datasets, exponentiate. Báo absolute latency
delta, median/p95 mô tả, dataset-specific effects và document-cluster CI.

| Tín hiệu xác nhận | Kết luận |
|---|---|
| Exactness pass; S≥1.03 và adjusted CI95 lower >1 so **cả best fixed draft gamma và best fixed verification k** | Actionability của adaptive verification được xác nhận trong workload/runtime đã đo |
| Combined còn vượt history-only và strongest signal-only với CI >1 | Có contribution thực tế của phối hợp signals |
| Nhanh hơn fixed15 nhưng không hơn best fixed gamma/k | Gain được giải thích bằng constant shorter block/prefix; chưa chứng minh adaptation |
| Acceptance rate tăng nhưng E2E không giảm | Không có inference gain; phân tích lost commits, rounds và overhead |
| B2 tốt, B1-V tốt nhưng B3 không nhanh | Local predictor/utility chưa chuyển thành E2E; phân tích distribution shift và cost model |
| T_V giảm nhưng tổng E2E không giảm vì extraction/sync/T_D chiếm ưu thế | Bottleneck/overhead chặn gain; giới hạn verification-only đã được đo |
| CI95 upper S<1.03 | Chưa có gain đạt mức thực tiễn đề xuất; báo có/không cải thiện nhỏ theo CI quanh 1 |
| CI chứa cả 1 và 1.03, quota thiếu hoặc missing cells | `inconclusive/partial`; không claim success hoặc no effect |
| Bất kỳ output mismatch/invalid cache | Correctness fail; không claim lossless speedup dù timing tốt |

3% là effect threshold đề xuất thống nhất với G4 cũ. Điểm S≥1.03 với CI lower>1
chứng minh gain dương với effect estimate thực tiễn; nếu muốn claim **gain ít
nhất 3%** thì lower bound phải vượt 1.03. Báo từng contrast, không chia hai
aggregate throughput means để tạo paired speedup mới.

### 11. B4 — Uncertainty round trước có dự đoán draft length round sau?

**Khi nào làm:** B1 có gap hữu ích, B3 bị giới hạn bởi draft cost, hoặc muốn
kiểm riêng nhánh adaptive draft length. Đây là nhánh có điều kiện, không là
việc bắt buộc để hoàn tất B1/B2.

**Làm gì:** thu uncertainty ở **mọi** reference round, không chỉ selected
prefixes. Tại t, inputs chỉ gồm uncertainty của các rounds <t, R, A và parent
signals. Mean/max entropy, min confidence của block trước được phân biệt với
summaries chỉ trên accepted/reached positions; thông tin L_previous đã có trước t.

Labels là L_t(gamma)/gamma oracle từ B1 same-prefix replay ở state t.
So R-A/R-PreAll với thêm lagged D-H/D-C; stratify theo số output tokens đã
advance ở round trước. Hai selected prefixes cách nhau nhiều rounds không
được gán là t−1/t. Grouped validation và cost regret dùng protocol B2.

**Tín hiệu kết luận:** lagged uncertainty giảm Brier/regret ngoài history và
regime trên unseen groups → cân nhắc draft-shape controller. Correlation cùng
round tốt nhưng lagged contribution CI upper thấp hơn threshold → candidate
uncertainty chưa hỗ trợ chọn next draft length; giữ verification-only hoặc
pre-draft history policy. CI rộng → chưa đủ dữ liệu.

Nếu triển khai E2E, gamma_t được chọn **trước** draft forward; logits round t
chỉ cập nhật state cho t+1. Draft lại gamma_t, không cắt proposals15 như B1.
Primary lag calibration reference15 chỉ chứng minh predictability trên fixed
trajectory; adaptive rollout phải log previous gamma, censoring và support,
dev-calibrate cho policy history thật như B3. Missing history/unsupported
lagged shape fallback fixed gamma. So best fixed draft gamma, regime-only,
history-only và lagged combined, áp exactness/E2E threshold như B3. Không suy
B4 actionability chỉ từ cross-round Spearman.

## Phần IV — Phân tích thống kê, bàn giao và triển khai

### 12. CI, weighting và tránh kết luận vượt dữ liệu

- Đơn vị bootstrap là source document/group, stratified theo dataset, 10.000
  draws; mọi states/actions/repetitions của group đi cùng nhau. Headline macro
  trọng số đều documents trong dataset, đều datasets. Báo pooled state-micro
  như secondary và actual counts, tránh documents dài chi phối.
  Với confirm partial, chỉ mean các `headline_datasets` đã khóa tại mục 3;
  exploratory macro cả ba phải ghi partial quota, không thay cohort sau đo.
- B1/B1-V bootstrap phải chọn lại best fixed diagnostic từ selection repeats
  trong mỗi resample và đánh giá bằng evaluation repeats. Confirm fixed controls
  đã khóa không được chọn lại từ confirm bootstrap. Với sample nhỏ, ghi rõ CI
  có độ ổn định hạn chế, đặc biệt QMSum chỉ 3 clean groups.
- B2 bootstrap paired out-of-fold document metrics cho các variants. CI này
  có điều kiện trên fitted folds; không đại diện đầy đủ uncertainty của model
  selection. Confirm dùng frozen estimator; grouped refit sensitivity chỉ trên
  dev. Nhãn oracle gamma/k và prediction accuracy cần báo near-tie ambiguity.
- Các contrasts xác nhận chính được khóa trước: B1 gap, B1-V gap, B2 strongest
  drafter so R-A và thêm entropy so strongest non-entropy, B3 so hai best fixed
  controls. Dùng Holm correction cho p-values trong từng research family;
  với m paired superiority contrasts của B3/entropy ablations, dùng bootstrap
  intervals ở confidence 1−0.05/m cho quyết định family-wise. Báo cả CI95 mô tả
  và adjusted interval dùng gate; một controller cần thắng cả hai controls.
- Phân tích positions 1…15, bins, datasets và nhiều summaries khác là exploratory
  trừ contrasts đã đăng ký. Không chọn một bin thuận lợi rồi gọi pooled hypothesis
  được xác nhận. Báo within-regime và between-regime findings riêng.
- Không độc lập hóa timing repetitions, queries QMSum hoặc thousands of rounds.
  Missing/error/parity failures hiện trong coverage denominator; không drop để
  tính speedup thuận lợi. Incomplete cells khiến affected gate pending.
- Không reject practical null vì p>0.05. Low-potential conclusion cần CI upper
  dưới practical threshold. Kết quả âm hợp lệ vẫn là deliverable nghiên cứu.

### 13. Artifacts và báo cáo cần bàn giao

```text
outputs/dflash_speculation_B/<run_id>/
  campaign.json                 # hypotheses, thresholds, budgets, scope
  runtime.json                  # actual models/code/GPU/backend signatures
  cohort_manifest.json           # source groups, splits, exposure, hashes
  schedule.json                  # state/action/repetition orders
  evidence/parity.json           # all shapes + branches + instrumentation
  traces/reference.jsonl         # all fixed15 reference rounds + IDs
  states/prefix_manifest.jsonl    # unique states + selection/progress metadata
  b1/draft_shape.jsonl           # independent gamma proposals/L/T/components
  b2/signals.jsonl               # current/lagged features with origins
  b2/oof_predictions.jsonl       # fold IDs, survival probabilities, labels
  b1v/verify_prefix.jsonl         # same proposals15; k-specific verifier cost
  calibration/curves.json        # bins, support, backoff, cost tables
  dev/requests.jsonl             # fixed controls + eligible policies
  locked_config.json             # baseline choices + frozen estimator
  confirm/requests.jsonl         # repetitions + exactness + timings + ROUGE
  confirm/rounds.jsonl           # actions, causality, commits/cache/fallback
  reports/coverage.json
  reports/metrics.json
  reports/decision.md
  figures/                       # exported plots, not committed to git
```

Mọi record JSONL qua `io_util.JsonlWriter`, cuối stream có summary. Row schemas
tách reference/state/action/prediction/request, có version và status. Một
row ghi dataset, source group, sample, state identity, action, repetition,
signature/fold và feature origin khi áp dụng. Entropy có raw/normalized, signal
temperature/vocab metadata. Timing có setup/extraction/controller/round riêng;
L, C, proposed candidates, censor/boundary/EOS riêng.

Resume key gồm run + runtime/code/cohort hash + phase + state/action + repetition.
Partial JSONL không coi là completed cell; resumed run không đổi signature.
Model/backend/prompt hashes đổi tạo run mới và kiểm parity lại. Không commit
outputs, candidate arrays/model caches hoặc metric artifacts vào repository.

| Figure/table | Nội dung | Câu hỏi trả lời |
|---|---|---|
| F1 | J cost surface + gamma oracle distribution + near-optimal coverage | B1 opportunity có lớn/ổn định không? |
| F2 | ΔL/ΔT theo gamma, survival curves và long-survival subgroup | Block dài hữu ích tại những states nào? |
| F3 | Candidate rejection risk theo uncertainty trong đúng risk set | Entropy/confidence có liên quan rejection? |
| F4 | Heldout reliability + Brier skills + cost regret | Tín hiệu có generalize và giúp quyết định? |
| F5 | Verification latency theo k + overhead-adjusted oracle gap | B3 có headroom thật không? |
| F6 | Paired E2E/rounds/overhead/action frequency, theo dataset/context | Controller có thắng strong fixed controls? |
| T1 | Counts/exposure/parity/errors/support/fallback/capped output | Kết quả có đủ điều kiện diễn giải? |
| T2 | Decision ledger B0/B1/B2/B1-V/B3/B4, effect/CI/limitations | Insight nào đã xác nhận, cái nào còn pending? |

Decision memo trả lời mỗi experiment bằng **làm gì → quan sát gì → độ chắc
chắn → kết luận → việc tiếp theo**. Ví dụ dưới đây là mẫu diễn giải, không
phải kết quả giả:

| Tổ hợp findings | Kết luận cuối của B |
|---|---|
| B1 gap nhỏ; B1-V gap nhỏ | Depth adaptation ít tiềm năng trên runtime/grid này; fixed tuning đủ |
| Opportunity lớn; signals không dự đoán được trên groups mới | Có headroom hindsight, chưa có deployable predictor |
| Signals dự đoán được; opportunity cost nhỏ | Uncertainty giải thích acceptance, chưa đáng làm adaptive controller |
| B1-V + B2 tốt; B3 không thắng fixed | Có predictable depth, actionability chưa đạt; báo bottleneck/overhead |
| B3 thắng hai best fixed và exactness | Confidence-aware adaptive verification có giá trị thực tế |
| History thắng, entropy không bổ sung | Giữ history-based control; loại entropy contribution |
| Confidence thắng, entropy không bổ sung | Giữ confidence; không gán benefit cho entropy |
| Chỉ B4 lagged signals + draft-length E2E đạt | Có predictive adaptive draft length, phân biệt với verification control |
| Parity fail hoặc CI/quota thiếu | Kết quả invalid hoặc inconclusive; không claim paper speedup |

### 14. Kế hoạch triển khai theo chức năng

Phần này là **hợp đồng triển khai dự kiến**, không là mô tả CLI đã có.
Executor V1 vẫn chỉ nhận draft_shape; việc có Action.length_mode string
verify_prefix chưa tạo đường chạy B3. Đọc kế hoạch task-by-task trước sửa mã.

| Task | Files dự kiến | Làm gì và bằng chứng hoàn tất |
|---|---|---|
| T1 — Protocol/cohort | Mới `src/TrainingFree/speculation_experiments/{config,cohort,schema}.py`; tái sử dụng splits/artifacts helpers | Lock grid/thresholds/exposure, source-group folds, state identity và summaries; CPU fixtures không leakage/duplicate |
| T2 — Candidate collector | Sửa `context_adaptive/{attention,types}.py`; mới `speculation_experiments/signals.py` | Optional FP32 entropy/confidence/margin vectors, không giữ full logits; fixture phân phối uniform/peaked, proposal equality và missing/history origins |
| T3 — Full reference/parity | Sửa hoặc tái sử dụng `context_interventions/execution.py`; mới `speculation_experiments/reference.py` | Fixed15 all-round trace; AR/native/adapter cả 4 shapes, EOS/cap và collector on/off; first-divergence report |
| T4 — B1/B1-V replay | Mới `speculation_experiments/replay.py`; tái sử dụng bank/cache helpers | Isolated forks, draft-shape riêng vs same-proposals prefix verify, warmup/schedule/setup exclusion, full round accounting; branch/cache/replay consistency fixtures |
| T5 — Predictive evaluation | Mới `speculation_experiments/{survival,evaluation}.py` | Train-fold-only bins, monotone curves, support/backoff, Brier/reliability/Spearman/regret; CPU analytic fixtures và leakage checks |
| T6 — Report/decisions | Mới `speculation_experiments/report.py`, `scripts/analyze_dflash_speculation_B.py` | Coverage, cluster CI, practical null/positive/inconclusive ledger, figures; report từ saved artifacts không cần load model |
| T7 — Verification controller, có điều kiện | Sửa `context_adaptive/generation.py` qua proposal/verifier API rõ ràng; mới `speculation_experiments/controller.py` | Verify k+anchor, discard suffix, row-L pending, cost/overhead/history censoring; GPU output exact và strong fixed controls |
| T8 — Launcher/campaign [INTEGRATION] | Mới `scripts/run_dflash_speculation_B.py`, `scripts/run_dflash_speculation_B_b200.sh`, `src/TrainingFree/docs/dflash_speculation_B_b200.md` | Master/runtime helpers, phase gates, shared replay/evaluator, resume/signatures, request/round JSONL và runbook |
| T9 — B4, có điều kiện | Mới `speculation_experiments/lagged.py`; tái sử dụng T5/T7 | Adjacent-round causal features, shape/censor support, draft-shape rollout nếu pass predictive gate |

T7/T9 không bắt buộc triển khai nếu evidence gate không đạt. T8 có đúng một
integration task cho campaign inference; validation áp inference static/CPU
logic + checkpoint/GPU runtime, không áp training-loss/MFU/overfit checks.
Future meaningful fixtures nằm ở `src/TrainingFree/tests/`, gắn từng chức năng,
không import tests/probe vào production inference core. Tái sử dụng API hiện
có khi đúng contract; không import benchmark CLI vào core để lấy helpers.

API phải phân biệt `propose(draft_gamma)` và `verify(proposal, verify_k)`;
DraftProposal metadata gồm draft_gamma, proposed IDs và optional uncertainty.
Verification outcome có verified_candidates, accepted_candidates, retained
features, next anchor và crop lengths. Không overload một gamma field vừa là
draft shape vừa là verification prefix. Giữ hành vi V1 mặc định để tránh làm
sai baseline A; added mode có parity evidence riêng.

#### 14.1. CLI phases hiện có và điều kiện chạy

Harness hiện có mode `prepare`, `preflight`, `validate`, `b1`, `b1v`,
`analyze`, `run`; invocation và artifacts mô tả trong
[`docs/dflash_speculation_B_b200.md`](../docs/dflash_speculation_B_b200.md).
`validate` thực hiện B0 parity và ghi fixed-gamma-15 reference trace;
`analyze` thực hiện offline B1/B2 analysis; `run` chạy B0→B1→B1-V→analysis.
Không có B3/B4, verify-prefix generation controller hoặc E2E controller
benchmark.

| Phase | Inputs chính | Outputs / điều kiện đi tiếp |
|---|---|---|
| prepare | Corpus, reviewed exposure registry, cohort phase và protocol | Cohort/split/campaign manifest; không load model |
| preflight | Local model snapshots và runtime | Runtime/GPU/backend signature; không chạy phase nghiên cứu |
| validate (B0) | Checkpoint thật, cùng prompts, gamma 3/7/11/15 | Exact AR/native/adapter parity; gamma 15 ghi reference trace; dừng nếu mismatch |
| b1 | B0 pass, fixed15 reference states | Independent draft-shape replay, timing split, gamma15 signal collection |
| b1v | B0 pass + paired B1 proposals | Draft15 một lần/state, k=3/7/11/15 verifier replay, acceptance/cache/anchor checks |
| analyze (B2) | B1 action/signal streams; B1-V nếu đã chạy | Grouped OOF survival/Brier, risk diagnostics, oracle/fixed metrics và decision screening |
| run | Cohort/config đầy đủ | Chạy tuần tự validate→b1→b1v→analyze; outputs resumable theo trace/state |

Các phase E2E dev/lock/confirm và B3/B4 vẫn là thiết kế tương lai; CLI không
được hiểu là có controller hoặc phép đo E2E adaptive.
Wrapper dùng `scripts/common/runtime.sh`, config master ngoài repository qua
`config/master.path`, `FAST_INFER_PYTHON` hoặc system `python3` 3.12, offline.
Không tạo venv từng baseline. Một process/card; nếu shard documents qua nhiều
cards, mỗi paired action/variant cùng document giữ cùng GPU, ghi GPU identity,
không gộp H100/B200 timings thành một runtime.

#### 14.2. Observability, budget và failures

Phase-start/phase-end, progress theo document/state/action, completed/total
cells và heartbeat tối đa 60 giây. Chạy report tạm sau mỗi document hoàn tất;
gates chỉ tính khi đủ planned cohort/cells. Pilot 6 groups đo actual GPU time
để dự toán trước ma trận 24 groups × ~15 states × 4 gamma × 6 repeats ≈8.640
timed rounds B1, ngoài warmup/reference. B1-V thêm số verifier actions tương
ứng và draft15/extraction profiling. Không suy wall time từ pilot H100 cũ.

Default harness hiện chạy B0→B1+B2 signals→B1-V→offline analysis; B3/B4 có
điều kiện và chưa được implement. Nếu GPU budget hạn chế, giảm số source groups/phase và ghi
partial; không giảm controls/repetitions rồi giữ nguyên confidence claim.
Không tạo cloud spend từ kế hoạch này; Modal diagnostic là campaign riêng.

| Failure | Hành vi bắt buộc |
|---|---|
| Asset/package/kernel không hợp lệ | Fail preflight với asset_missing/package_missing/kernel_incompatible; không online install hoặc silent backend swap |
| Parity/cache/instrumentation mismatch | Lưu first-divergence rồi dừng affected research gate |
| OOM | Ghi error/state; cleanup ngoài measured span; retry tối đa 1 lần cùng signature; lặp lại thì incomplete cell |
| Missing/non-finite signal hoặc timing | Invalid row có lý do; không impute thành zero/outcome thành rejection |
| Không đủ groups/tail states/cost support | insufficient_data/partial hoặc fixed fallback logged; không fabricated rows |
| Resume signature khác | Từ chối merge; run mới và parity lại |
| CUDA fatal/context hỏng | Dừng worker; giữ artifacts đã flush; resume process mới từ complete cells |
| Aggregation thiếu paired action | Coverage fail và affected gate pending, không complete-case speedup headline |

### 15. Deliverable và phạm vi claim

Nghiên cứu hoàn tất khi có protocol đã khóa, đủ artifacts cho các phases đã
mở, coverage/correctness evidence và câu trả lời B1/B2 cùng opportunity của
cơ chế được giữ. B3/B4 có thể kết thúc bằng quyết định không mở vì không đủ
headroom/predictability; kết quả âm đúng quy trình là một kết luận hữu ích.

Claim cho paper cần đồng thời: production parity, cohort/exposure đủ rõ,
estimators/baselines frozen trước confirm, CI theo source groups và E2E tốt
hơn strong fixed controls sau overhead. Local oracle, Spearman, acceptance
rate tăng hoặc pilot throughput chưa đạt exactness không thay được các bằng
chứng này. B-only full context cũng chưa chứng minh interaction A+B/C.

**Checklist nghiên cứu sau khi triển khai:**

- [ ] B0: full-context greedy parity mọi shape, instrumentation/caches/stopping.
- [ ] B1: independent draft-shape sweep, timing selection/evaluation tách, gap CI.
- [ ] B2: candidate vectors đúng thời điểm, grouped validation, incremental signals.
- [ ] B1-V: same-proposals prefix verification, overhead-adjusted opportunity.
- [ ] B3 nếu mở: dev-chosen strong fixed controls, lock, exact paired confirm E2E.
- [ ] B4 nếu mở: causal adjacent-round signals và real draft-shape evaluation.
- [ ] Coverage/exposure/QMSum limitations và positive/negative/inconclusive ledger.

Code harness B0/B1/B2/B1-V có test CPU; parity GPU, cache-prefix validation,
timings và mọi paper claim vẫn **pending**. B3/B4 và E2E exactness chưa được
implement; chưa có kết quả hoặc speedup từ kế hoạch này.
