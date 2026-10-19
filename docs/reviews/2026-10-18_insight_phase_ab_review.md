# Đánh giá phương pháp và triển khai suy luận Phase A/B

Ngày đánh giá: 18/10/2026.

> Đây là đánh giá trước sửa lỗi. Các lỗi đã xử lý và bằng chứng mới được ghi ở
> [nhật ký sửa logic từng phase](../_INSIGHT/phase_ab_logic_corrections_20261018.md).
> Phạm vi công việc hiện tại là đánh giá riêng A/B; thực nghiệm tổng thể A+B là
> giai đoạn sau, không phải điều kiện hoàn thành screening từng phase.

## 1. Phạm vi và kết luận

Đã đọc năm tài liệu trong `docs/_INSIGHT`, đối chiếu với runtime
`src/TrainingFree/context_adaptive`, `context_interventions`,
`speculation_experiments`, các script benchmark liên quan và artifact hiện có.
Đây là đánh giá tĩnh kèm đọc dữ liệu NPZ/CSV/JSON; không chạy lại inference,
benchmark GPU hoặc bộ test. Các kết luận về mã nguồn áp dụng cho working tree
hiện tại, trong đó có nhiều thay đổi chưa commit. Artifact cũ chưa gắn đủ code
signature để quy mọi kết quả về đúng phiên bản runtime đang đọc.

**Kết luận:** hướng nghiên cứu có giá trị, nhưng bộ tài liệu hiện tại chưa đủ
cơ sở để gọi là “paper-ready”. Phase A có pilot chạy GPU và cơ chế chọn ngữ cảnh
khả thi, nhưng một phần bằng chứng đang đo thuật toán khác với công thức đề xuất.
Phase B có lõi survival đúng và mục tiêu hệ thống hợp lý; mô hình hazard ba thành
phần vẫn là đề xuất, với một số diễn giải lý thuyết cần sửa trước khi triển khai.

| Hạng mục | Đánh giá |
|---|---|
| A: dùng tín hiệu target vòng trước để chọn context cho drafter vòng sau | Hợp lý, giữ được luồng suy luận nhân quả |
| A: max pooling nhiều query | Heuristic đáng thử; chưa có bằng chứng hiện tại tách biệt được hiệu quả của max với mean |
| A: kết luận vượt SnapKV/FastKV/RocketKV nguyên bản | Chưa được hỗ trợ bởi cách dựng baseline hiện tại |
| A: kết quả coverage đa tầng | Có lỗi thay thế layer và lỗi diễn giải số trung bình |
| B: conditional hazard, risk set, tail-sum | Đúng về cấu trúc toán học |
| B: present/past/future là ba cơ chế trực giao đã chứng minh | Không được chứng minh |
| B: chọn prefix theo full-round cost | Hợp lý như policy cục bộ; chưa chứng minh tối ưu throughput toàn chuỗi |
| B: roofline bypass và ngưỡng cải thiện | Hướng thiết kế tốt khi dựa trên cost profile thực đo |
| A+B: hệ thống kết hợp | Cần đánh giá tương tác; không thể nhân hai speedup độc lập |

## 2. Các vấn đề ưu tiên cao của Phase A

### A1 — P0: Ma trận đa tầng chứa cột thay thế bằng Layer 18

Trong `scripts/run_candidate_benchmarks.py:489–524`, khi snapshot không có 36
layer, C3 “Layer 27” và E3 “Band 14–20” đều trả về `t[1]`, giống C1.
C4 “Layer 12” trả về `t[0]`; E2 “18/27/35” dùng trung bình ba hàng sẵn có.
Probe mặc định lấy `{0, middle, last}` tại
`scripts/probe_dflash_target_comparison.py:493–499`.

Đọc trực tiếp thư mục snapshot mà script mặc định sử dụng cho thấy:

- 120/120 snapshot chỉ có 3 hàng target attention.
- 10 document khác nhau, bốn context cap, mỗi context có 30 snapshot.
- Tại context 16384, B=2048, CSV của C1, C3 và E3 đều có đúng
  `mean_cov_candidate_gt = 0.9161762204425974` cho M1.

Vì vậy, báo cáo chưa đo được khả năng tổng quát của M1 qua L27 và band 14–20;
con số 46.42% đang gắn nhãn L12 thực chất đến từ hàng attention tầng đầu trong
bộ snapshot mặc định. Lập luận loại L12 vì “early-layer collapse” cũng cần rút
lại cho tới khi có dữ liệu đúng tầng. “120 snapshots từ 30 sample ở 16k” cần sửa:
artifact hiện tại có 10 document, 30 snapshot ở 16k, 120 snapshot trên bốn cap.

**Cần làm:** lưu `layer_ids` trong từng snapshot, tra cứu bằng ID, báo thiếu tầng
thay vì thay thế; thu đúng các tầng trước khi tính lại ma trận. Không dùng số
trung bình nhiều cột trùng nhau làm bằng chứng độc lập.

### A2 — P0: Con số 91.62% chưa đo Multi-Query Max-Pooling

Probe tạo vector `target` bằng
`probabilities[:, :, :aligned_queries, :].mean(dim=(0, 1, 2))` tại
`scripts/probe_dflash_target_comparison.py:209–211`. Các snapshot cũ chỉ giữ
vector đã gộp, không giữ chiều query.

Benchmark chọn M1 bằng top-B của chính vector đó tại
`scripts/run_candidate_benchmarks.py:134–169`, rồi tính coverage bằng tổng khối
lượng được giữ lại trên vector attention dùng làm tham chiếu.

Đây là **mean-query attention top-B**, khác với
`max_query(mean_heads(attention))` trong đề xuất. Với cùng ngân sách và tập token
bảo vệ, top-B của vector tham chiếu tất nhiên tối đa hóa tổng khối lượng của
vector đó. Việc thắng các bản smoothing trên metric này chưa chứng minh M1
cải thiện acceptance hoặc chọn đúng thông tin ngữ nghĩa.

**Cần làm:** giữ attention theo từng query; so max/mean/parent trên cùng dữ liệu.
Chọn mask từ vòng t và đo acceptance hoặc draft-logit quality ở vòng t+1.
Coverage trên cùng vector vẫn có ích cho chẩn đoán nhưng phải ghi rõ đây là
đánh giá giữ lại attention mass của chính vector chọn lọc.

### A3 — P1: Runtime và công thức paper chưa mô tả cùng một thuật toán

`context_adaptive/generation.py:112` chỉ lấy `accepted + 1` query, gồm anchor và
tiền tố đã được xác nhận; số query thay đổi theo round, có thể chỉ còn một khi
không candidate nào được chấp nhận. Runtime không lấy cố định 15 query như mô tả.
Việc loại query sau điểm gãy có cơ sở vì chúng phụ thuộc tiền tố đã bị từ chối,
nhưng cần ghi đúng và có ablation.

`selection.py:62` xếp hạng bằng tổng điểm chunk; `selection.py:130–138` thêm cả
chunk khi còn đủ ngân sách. `AdaptiveConfig` mặc định chunk 128, source anchors
64 và recent output 256; paper mô tả top token với sink 16 và local 32.
Do đó, pilot runtime của selector M1 đang gần biến thể chọn chunk hơn là công
thức M1 chọn từng token trong báo cáo.

**Cần làm:** khóa định nghĩa riêng cho token selector và chunk selector, gồm
query scope, head aggregation, key scope, phần bảo vệ và cách dùng ngân sách dư.
Đồng nhất định nghĩa đó giữa toán, replay và E2E.

### A4 — P1: Tái dựng attention nhiều query đang dùng sai vị trí nhân quả

`generation.py:113–117` truyền một số nguyên `logical_length + accepted` làm
`query_positions`. `context_adaptive/signals.py:145–147` gán số nguyên đó cho
mọi query. Khi có nhiều query, các hàng sớm được phép nhìn key của những vị trí
sau nó trong block đã commit, khác causal attention thật của target.

Đây là lỗi của tín hiệu selector tái dựng; nó không tự chứng minh verifier đã
phát token sai. Tuy nhiên, nó làm sai trọng số và mẫu số softmax dùng để chọn
context. Hàm tương tự trong `context_interventions/attention.py:80–92` cũng có
nhánh scalar như vậy.

**Cần làm:** truyền vector vị trí riêng cho từng hàng, ví dụ
`logical_length + arange(accepted + 1)`, và giữ đúng causal/sliding-window mask.

### A5 — P1: Các baseline đang là biến thể chuyển thể, không phải tái hiện 1:1

FastKV vendored dùng nhiều query của observation window, pooling, chọn KV theo
nhóm head và token-selective propagation
(`externals/FastKV/baselines/fastkv/utils.py:94–128`). M0 chỉ lấy parent query ở
L18 nên không thể mang nhãn FastKV nguyên bản. arXiv `2410.21337` trong báo cáo
trỏ đến bài về phát hiện prompt injection; bài FastKV có ID
[2502.01068](https://arxiv.org/abs/2502.01068).

[SnapKV](https://arxiv.org/html/2404.14469v2) chọn KV từ attention trong observation
window và theo từng head. Áp một avg-pool k=7 lên vector đã gộp ở L18 chỉ tái hiện
một thành phần của phương pháp. Giả định “các phương pháp AR chỉ nhìn một query”
trong báo cáo cũng không đúng với cơ chế observation window.

RocketKV vendored tổng hợp observation queries, gộp GQA rồi max-pool với kernel
lấy từ config, sau đó còn các bước của pipeline RocketKV
(`externals/RocketKV/gpt-fast/rocket.py:293–329`). Một phép max-pool k=7 trên một
vector không đủ đại diện cho toàn bộ hệ thống gốc.

[DuoAttention](https://arxiv.org/abs/2410.10819) phân biệt retrieval heads và
streaming heads, không phải phép trộn L18/L35 theo tỷ lệ 50/50. M4 nên được gọi
là ablation ensemble do tác giả thiết kế. Nó không phải cận trên lý thuyết của
acceptance hay throughput. Có thể dựng oracle coverage đúng nghĩa bằng top-B
của vector mục tiêu đã định nghĩa; oracle hệ thống phải dựa trên reward/cost
thực đo của các action tại cùng state.

**Cần làm:** đổi tên M0/M2/M3 thành parent-query / avg-pool / max-pool adaptations
khi chỉ so các toán tử trên cùng tín hiệu. Nếu muốn claim vượt công trình gốc,
cần benchmark pipeline gốc và ghi rõ mọi thay đổi khi chuyển sang drafter.

### A6 — P1: Bảng E2E và các metric phần cứng chưa truy nguyên đầy đủ

Artifact `outputs/modal_proposed_selectors_inference_results.json` hiện có một
prompt 8192 token, output cap 64 và năm cấu hình:

| Cấu hình trong JSON | TPS | Speedup so dense DFlash | Mean alpha ghi trong JSON |
|---|---:|---:|---:|
| Dense | 26.42 | 1.000 | 2.033 |
| Parent query | 31.24 | 1.182 | 2.652 |
| Max-query | 31.76 | 1.202 | 2.652 |
| Mean-query | 31.85 | 1.205 | 2.652 |
| Duo | 31.04 | 1.175 | 2.773 |

JSON này không chứa SnapKV, RocketKV, Deep Sink hay M1-Chunk như các cột E2E của
báo cáo. Nó cũng không xác nhận bộ số 31.3/30.4 TPS. Có thể có lần chạy khác,
nhưng cần trỏ rõ artifact, sample IDs và code version cho từng con số.
Pilot này cho mean-query nhỉnh hơn max-query, cùng acceptance; chưa đủ để kết
luận max là lựa chọn tốt nhất. Baseline speedup ở đây là dense DFlash, chưa phải
target-only AR.

`generation.py:1482` định nghĩa `mean_accept_length` bằng trung bình `1 + L`.
Trong JSON của M1, `accepted_total / speculative_rounds = 38/23 ≈ 1.652`, trong
khi `mean_alpha = 2.652`. Paper phải phân biệt L, tỷ lệ L/k và commits 1+L.

Token-match trong script so với dense DFlash và chỉ trên phần độ dài chung
(`scripts/modal_test_cadflash_selectors.py:268–278`). Nó chưa xác nhận exact
sequence parity với target-only AR, kể cả chiều dài và EOS.

“Memory coalescing” được tính bằng mô hình `indices // 16` trong
`scripts/benchmark_shortlist_tradeoff.py:141–149`, không phải hardware counter.
Một token KV gồm head × head_dim phần tử nên không thể mặc nhiên coi 16 token
là một cache line. Chunk continuity cao chưa chứng minh PagedAttention đã được
tích hợp hoặc GPU memory transactions đã tốt hơn.

Microbenchmark trong `scripts/benchmark_selection_timing.py` đo nhiều phần bằng
NumPy/PyTorch CPU, có cả GPU latency ước lượng. Nó không bao phủ QK, softmax,
GQA expansion, truyền CPU, xếp hạng và gather của runtime. “0 forward pass phụ”
là đúng về số lượt gọi model, nhưng không đồng nghĩa “0 overhead”: runtime
đang tính lại QK/softmax và đồng bộ dữ liệu về CPU. Không có bằng chứng overlap
đủ để nói toàn bộ overhead đã được ẩn.

`DraftContextBank` vẫn cấp phát full bank và append KV cho mọi token commit;
chọn ngân sách chủ yếu giảm lượng KV drafter đọc khi attention, chưa tương đương
giảm 87.5% VRAM toàn hệ thống (`context_adaptive/cache.py:43–51, 79–93`).

Ngoài ra, `scripts/generate_paper_matrix_figures.py:39–156` chứa bảng số gán trực
tiếp; mean 91.28% của M1 là mean L18/L35, không phải mean mọi candidate. CSV
cross-candidate hiện có ghi mean bảy candidate là 82.8328% cho M1 và 84.4878%
cho M4; chính bảy candidate ấy vẫn chứa các phép thay thế sai tầng ở A1.

## 3. Đánh giá mô hình đề xuất Phase B

### B1 — Những phần nên giữ

Đặt `p_j = P(Y_j=1 | Y_<j=1, F)` và `S(k)=product(p_j)` là đúng theo chain rule,
không cần giả định độc lập các token. Huấn luyện trên risk set đến token lỗi đầu
tiên tránh gán nhãn sai cho phần đuôi. `E[commits(k)] = 1 + sum(S(1:k))` đúng
cho round thông thường, cần hiệu chỉnh tại EOS/output boundary.

Sigmoid continuation rồi nhân tích cho survival không tăng theo k là cấu trúc
hợp lý. Survival thật có thể bằng 0 hoặc có đoạn phẳng; “giảm nghiêm ngặt” là
đặc tính tham số hóa hữu hạn, không phải định luật về dữ liệu.

Tử số cost cần gồm draft cố định, target verify, gather/cache/bank update và
overhead policy. Profile phần cứng để bypass bước extraction là ý tưởng tốt.
Grouped evaluation, best fixed-k và đo E2E là các lựa chọn thực nghiệm phù hợp.

### B2 — P1: Present/past/future là đặc trưng dự báo, chưa phải cơ chế đã chứng minh

Đẳng thức `m = c(1-exp(-delta_z))` đúng khi không clipping; đưa `max(m, epsilon)`
vào vế trái thì phải sửa cả biểu thức tương ứng ở vế phải.
Nhưng một số m không bảo toàn toàn bộ thông tin của c và delta_z: ví dụ
`(p1,p2)=(0.6,0.2)` và `(0.5,0.1)` cùng margin 0.4 nhưng confidence và logit gap
khác nhau. Margin không phải sufficient statistic đã chứng minh cho acceptance.

`TV(delta_x,Q)=1-Q(x)` là đẳng thức đúng. Tuy nhiên, target softmax P vẫn không
phải Dirac chỉ vì decoder chọn greedy. Dirac mô tả lựa chọn token tất định.
Đẳng thức TV này không tự cung cấp định lý nối tổng `1-c_i` với độ lệch hidden
state, attention value hay xác suất argmax của target.

Phương trình `h_k ∝ sum_v Q_k(v)e_v` tại §3.3 không phải tính chất tổng quát
của Transformer: từ `Q=softmax(W h)` không suy ra phép nghịch đảo đó. Các hướng
embedding không được chứng minh độc lập/ngẫu nhiên, và attention không nhất
thiết chia đều 1/K. Vì vậy, “Mode-Splitting Variance” hiện là giả thuyết trực
giác, không phải định lý; cũng chưa đủ căn cứ loại entropy khỏi mọi mô hình.

Ba miền chỉ số không giao nhau không làm ba đặc trưng trực giao thống kê.
Trong drafter bidirectional, confidence hiện tại đã phụ thuộc các vị trí khác.
Hệ số future dương, kể cả CI không chứa 0, chỉ chứng minh liên hệ dự báo có điều
kiện trong mô hình đã fit, chưa chứng minh nhiễu nhân quả từ tương lai.

**Cần làm:** gọi đây là position-aware hazard với đặc trưng present/prefix/suffix.
Đo giá trị dự báo tăng thêm của từng nhóm bằng held-out log-loss/Brier và policy
regret. Nếu muốn claim cơ chế, cần can thiệp vào mask/hidden states tương lai,
giữ các yếu tố khác cố định và đo tác động lên dự đoán sớm.

### B3 — P1: Công thức history và quá trình fit chưa nhất quán

§4.1 dùng history như EMA độ dài tau trong [0,15]. Trong code,
`AdaptiveController.observe()` lưu EMA của `accepted_prefix / verification_k`
trong [0,1] (`controller.py:180–185`). Cùng tên nhưng khác đại lượng.

Ngay cả nếu tau đúng là E[L], phép đổi
`p_hist=((tau+1)/(K+2))^(1/K)` không được suy ra từ mean accepted length. Nếu
mọi continuation probability bằng p thì E[L] = sum(p^j), không phải một xác suất
sống sót cuối block để lấy căn bậc K. Cộng offset này không bảo đảm khử frailty.

Loss trong §4.2 fit `sigmoid(Phi beta)` nhưng predict lại cộng history offset.
Nếu giữ offset, phải tính nó từ quá khứ theo cùng protocol và đưa vào cả loss.
Khi k thay đổi, L quan sát bị right-censored nếu accept hết k; L/k cũng đổi ý
nghĩa theo action. Cần calibration cho rollout có k biến đổi, hoặc tạm bỏ history.

Chiếu beta về không âm sau năm bước Newton không giải đúng bài toán logistic
có ràng buộc. Hessian dương xác định nhờ ridge là đúng, nhưng không bảo đảm
Newton không damping hội tụ trong năm bước hoặc mọi feature scale đều ổn định.
Claim “nghịch đảo chỉ 64 phép nhân”, “<0.3 ms”, “ổn định tuyệt đối” cần thay bằng
thuật toán solve, tiêu chí hội tụ và thời gian đo thực tế.

Warmup chạy riêng k=15 cho nhãn hazard không thể đồng thời đo đủ cost của
k=3/7/11. Cần tách warmup model, calibration hazard và profiling action trên
prefix/cache fork. Báo cả cold-start cost, warmup tokens và amortized latency;
không loại chi phí thích nghi trên request test khỏi E2E mà không khai báo.

### B4 — P1: Policy cost hợp lý nhưng claim tối ưu toàn cục và overhead còn quá mạnh

Với các cycle tái sinh hoặc quá trình stationary/ergodic phù hợp, tỷ số kỳ vọng
reward/cost có ý nghĩa throughput. Các round trong chuỗi văn bản thay đổi context,
và action thay đổi số round, history, bank state, tín hiệu Phase A kế tiếp.
Tối thiểu hóa tỷ số theo từng state không tự suy ra tối ưu tỷ số dài hạn của
toàn policy. Nên mô tả J(k) là mục tiêu cục bộ có cơ sở hệ thống và kiểm chứng E2E.

“Định lý 6” có đại số hợp lý dưới giả định cost(k) >= cost(3), cùng overhead
không âm và survival hợp lệ. Cost thực đo có thể không đơn điệu vì kernel và
nhiễu; dùng min cost của mọi action được hỗ trợ sẽ chặt chẽ hơn. Bypass dựa
trên bảng cost là cổng theo lợi ích có thể đạt, chưa phải phân tích roofline đầy
đủ nếu không đo FLOPs/bytes hoặc vùng giới hạn phần cứng.

Khi so với DFlash tĩnh không có controller, phải tính cả chi phí extraction mà
controller đã trả trước khi fallback. So J(k) với J(15) cùng thêm overhead chỉ
so hai action sau khi đã bật controller; chưa bảo đảm controller thắng bản tĩnh.
“0.00 ms tuyệt đối” nên đổi thành “bỏ qua uncertainty extraction”; kiểm tra gate,
lookup, nhánh Python và các signal khác vẫn có chi phí.

Fast top-2 có thể giảm tensor trung gian. Tuy nhiên `logsumexp` vẫn đọc toàn bộ
vocabulary và tính exponential trong reduction; top-k cũng phải quét vocabulary.
Không thể suy ra giảm 99.98% tổng memory traffic hoặc chỉ còn 30 phép exp từ
đoạn PyTorch trong §6.1. Cần microbenchmark đúng shape/dtype/GPU và đo E2E.

### B5 — Trạng thái triển khai hiện tại

Tìm trong `src` và `scripts` chưa thấy `R-Hazard-3P`, `fit_hazard_curve`,
`predict_hazard_curve`, `should_bypass_uncertainty` hay `fast_top2_only`.
Runtime B3 vẫn dùng binned survival với cumulative minimum và chọn min J trực
tiếp; chưa có hysteresis được đề xuất (`b_controller.py:28–101`).

Việc gộp năm vector thành một lần `.cpu().tolist()` đã có trong `signals.py`;
đề xuất này không còn là phần chưa triển khai. Full-vocab entropy và finite-check
sync vẫn còn. Callback nhận margin từ generation, nhưng
`b_controller.py:207–210` chỉ đưa entropy/confidence vào model.

Một lỗi schema tiềm ẩn: `_signal_value()` lấy parent entropy/confidence từ
`row['parent_signal']`, trong khi callback B3 chỉ tạo `parent_entropy` ở top level
và không cung cấp parent confidence. Các variant có parent features như R-DAll
chưa thể cắm trực tiếp vào callback này. Luồng B3 hiện khóa chủ yếu R-DH/R-DC nên
không nên suy rộng lỗi này thành “mọi policy B3 đều hỏng”.

Đề xuất bỏ entropy của rejected suffix trong B4 đáng làm ablation, nhưng không
phải kết luận bắt buộc: đặc trưng suffix cũ có thể vẫn dự báo độ khó văn bản.
Chỉ token lỗi đầu tiên được quan sát là lỗi prefix; các vị trí sau nó không thể
đều được coi là target đã reject theo nhãn độc lập. Muốn cắt theo L+1 cần truyền
previous accepted/censoring vào callback và fit lại đúng schema deployment.

## 4. Giá trị nghiên cứu và cách kết nối A/B

Đóng góp có thể bảo vệ là: **tái sử dụng thông tin verifier để giảm context đọc
của drafter, kết hợp hazard được hiệu chuẩn để chọn prefix verify theo lợi ích
và chi phí trên ngữ cảnh dài**. Max pooling đơn lẻ và survival chain rule khó
đủ làm novelty chính.

[LibraSpec](https://arxiv.org/html/2608.08721v1) đã trực tiếp nghiên cứu chọn số
token đáng verify cho diffusion drafter theo gain/cost. Nó còn nêu giả định
confidence được hiệu chuẩn với acceptance; vì vậy confidence không mặc nhiên
là xác suất match. Ví dụ `0.45^6` trong insight chỉ minh họa một proxy dưới giả
định calibration/conditioning phù hợp, chưa phải xác suất thất bại đã biết.
[DSpark](https://arxiv.org/abs/2607.05147) cũng dùng prefix survival và profile
throughput của engine. Phần cần chứng minh mới là hiệu quả của calibration,
context selection và tương tác của hai quyết định trên workload của tác giả.

Tương tác A/B là thực chất: A đổi logits/confidence/acceptance của drafter;
B đổi số query đã verify có thể dùng để chọn context vòng sau. Với runtime lấy
accepted+1 query, k ngắn làm giảm số query tiềm năng cho A. Hazard fit từ full
context không thể mặc nhiên dùng cho context đã nén.

Tài liệu cũng đang dùng “Phase B” cho hai việc khác nhau: roadmap cuối MASTER
report nói tích hợp M1 vào runtime/PagedAttention; `proposed_method_phase_b.md`
nói điều khiển prefix. Cần thống nhất A=context selection, B=verification control;
engine integration là hạng mục triển khai của hệ thống kết hợp.

## 5. Thứ tự hoàn thiện được đề xuất

1. **Sửa bằng chứng trước:** layer IDs, thiếu layer, mean/max query, đơn vị sample,
   mean coverage, metric acceptance và provenance từng bảng. Giữ các kết quả cũ
   với nhãn phù hợp; không gắn lại thành kết quả thuật toán mới.
2. **Khóa thuật toán A:** token/chunk, accepted/all queries, causal reconstruction,
   protection budget và dense-bank cost. So parent, mean và max trên cùng cấu hình.
3. **Triển khai B tối giản:** present-only logistic hazard có intercept theo vị trí
   làm đối chứng; sau đó thêm prefix, suffix, history đã định nghĩa đúng. Fit và
   predict cùng feature/offset; xử lý censoring. Đóng băng model trên confirm set.
4. **Đo opportunity trước E2E lớn:** oracle gap giữa fixed-k và chọn k theo state,
   cộng extraction/controller cost. Nếu gap quá nhỏ sau overhead, giữ B như kết
   quả giới hạn thay vì làm phức tạp controller.
5. **Ma trận tương tác:** dense DFlash; A-only; B-only; A+B; target-only AR và best
   fixed (budget,k) chọn trên dev. Tách so toán tử chuyển thể với so pipeline gốc.
6. **Đánh giá xác nhận:** document held-out trên GovReport, QMSum, MultiNews; nhiều
   context/output lengths; paired repeats và CI bootstrap theo document. Báo
   TTFT, decode TPS, E2E latency, full GPU memory, overhead, E[L], E[1+L], exact
   token/EOS/length parity với AR. ROUGE đánh giá tác vụ, không chứng minh lossless.

Một liên hệ toán học có thể dùng để giới hạn claim coverage: với cùng attention
a và cùng value vectors có norm <= M, bỏ mass 1-C rồi chuẩn hóa lại phần giữ có
thể chặn sai khác attention output bởi `2M(1-C)`. Nhưng để suy ra bảo toàn argmax
cần thêm chặn truyền sai số qua mạng và logit gap; target/drafter ở đây khác
mạng nên không thừa hưởng trực tiếp cận này. Do đó, chưa thể viết “coverage 90%
suy ra hidden state gần hoàn hảo và acceptance cao nhất”.

**Định vị hiện tại:** Phase A là kết quả thăm dò có pilot GPU, cần sửa thiết kế
đánh giá và sự khớp giữa công thức với runtime. Phase B là đề xuất có lõi toán
khả thi, cần hạ các claim cơ chế xuống giả thuyết và triển khai trước khi kết
luận hiệu năng. Chưa đủ bằng chứng để tuyên bố superior SOTA hoặc paper-ready.
