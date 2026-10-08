# DFlash trên context dài: tổng hợp thực nghiệm và cơ sở phân tích cho paper về chọn/nén memory

Ngày tổng hợp: **2026-10-06**. Phiên bản: **1.1**. Phạm vi: các khảo sát DFlash trong phiên nghiên cứu hiện tại, liên hệ với chuỗi thực nghiệm Training-Free E41–E44 và bài *Strong Drafts Need Compact Memories* do nhóm cung cấp. Phiên bản này cập nhật nhận xét về block/thiên lệch đầu–cuối và hướng cơ chế inference tận dụng DFlash đã train; không thêm thực nghiệm.

Tài liệu này là hồ sơ bằng chứng và phân tích để phát triển paper. Các số liệu DFlash lấy từ artifact của những run đã hoàn tất; việc tổng hợp không chạy thêm GPU, training hoặc can thiệp KV. Các thiết kế module và thực nghiệm tiếp theo được ghi là đề xuất, chưa có kết quả.

## 1. Kết luận nghiên cứu hiện tại

Hướng ưu tiên sau làm rõ của người nghiên cứu là **tận dụng backbone DFlash đã train, giữ nguyên block 16 vị trí và cải tiến cơ chế chọn/đọc/cập nhật cache context phục vụ drafter**. Cache này gồm prompt và output đã commit; target verifier tiếp tục dùng full context. Đề xuất cụ thể hiện tại dùng attention của chính DFlash để chọn key, tái sử dụng ngắn hạn và dense refresh; xem [cơ chế inference](2026-10-06_dflash_context_execution_mechanism.md) và mục 8.3. Các nhánh học module trong hồ sơ được giữ như phương án nghiên cứu đã thảo luận, không phải hướng đã chốt. Chi phí retrain là động cơ vận hành, còn tính hữu ích của cơ chế cần được kiểm tra bằng acceptance và chi phí inference.

Các quan sát chính tại prompt 16K:

- **Block 16 vị trí nhận tỷ trọng attention ổn định tương đối:** mean toàn trajectory khoảng **21–23% tổng mass** qua bốn cap; riêng 16K, lượt đầu/giữa/cuối là **24,37% / 20,56% / 20,23%**. Đây không phải hằng số 20% ở mọi layer/round (I1).
- **Attention thường ưu tiên vùng đầu/cuối, đặc biệt cuối context:** tại 16K, hai bin 10% ở hai đầu nhận **58,43% tổng mass**; khi bỏ block, hai đầu cache nhận **47,16% cache mass**. Vùng đầu có thể liên quan attention sink, vùng cuối có thể liên quan recency/output; chưa có can thiệp tách các nguyên nhân (I5/I8).
- Top-1K/4K trên mọi key bao phủ **67,29% / 82,61% tổng attention mass**, cao hơn cửa sổ liên tiếp tốt nhất cùng budget (**49,61% / 63,36%**). Các key có score cao phân bố ở nhiều vị trí.
- Khi bỏ đúng 16 block key khỏi phép đo, Top-1K/4K cache bao phủ **58,58% / 77,96% cache mass**. Nếu luôn giữ block, tập này giữ **67,43% / 82,66% tổng mass gốc**. Phần cache sẽ nén vẫn có mass đáng kể ngoài tập chọn.
- Phân bố có cả nhóm key rất tập trung và phần mass trải rộng. Top-64 mọi key nhận **46,53% mass**, trong khi cần khoảng **7.035 key để đạt 90%**. Context dài đi cùng giảm độ tập trung tương đối; chưa có bằng chứng attention trở thành phân bố đều.
- Tập key được ưu tiên ổn định hơn giữa hai lượt liền kề, nhưng thay đổi nhiều giữa các mốc xa. Tại 16K, Top-1K cache trùng **82,65%** giữa hai lượt liền kề và **31,11%** giữa lượt giữa–cuối. Điều này gợi ý cập nhật memory theo trạng thái sinh, đồng thời để ngỏ khả năng tái sử dụng ngắn hạn.
- Một tập key chung có coverage khác nhau giữa năm draft layer. Phép mean phù hợp cho overview và proxy của memory chung, nhưng không chứng minh mọi layer/head/query có cùng nhu cầu.
- Attention target chưa cho thấy quan hệ đủ ổn định để dùng làm **selector độc lập** cho DFlash. Parent Top-1K ở lượt trước chỉ nhận **29,80% tổng DFlash mass** ở lượt sau, hay **38,01% cache mass**, tại 16K. Khi cộng block được giữ sẵn, coverage gốc là **51,12%** trên phép tổng hợp layer-pair của khảo sát parent.
- **Chưa đo được bonus-to-next-draft:** 4.326 cặp parent được thu đều dùng correction query. Literal bonus query `q=15` có **0 cặp** với lượt DFlash kế tiếp. Vì vậy chưa có kết luận riêng về tín hiệu bonus token.

Những kết quả này tạo động cơ nghiên cứu memory thích ứng và memory tổng hợp. Chúng chưa chứng minh bỏ key ít mass sẽ giữ acceptance, module học tốt hơn selector không train, hoặc giảm draft KV sẽ tạo speedup toàn hệ thống.

## 2. Câu hỏi nghiên cứu và bản đồ thực nghiệm

### 2.1. Những câu hỏi đã được khảo sát

| Câu hỏi | Bằng chứng hiện có | Trạng thái kết luận |
|---|---|---|
| DFlash có tập trung attention vào ít key trên context dài không? | Coverage Top-K, tỷ lệ Top-K theo số key, entropy, key count cho 90%/95% mass | Có nhóm tập trung mạnh, đồng thời có phần mass rộng; mô tả trên cohort hiện tại |
| Một cửa sổ liên tiếp hoặc quy tắc vị trí đầu/cuối có bao phủ tốt không? | So sánh cùng budget với Top-K rải rác, trên mọi key và trên cache bỏ block | Coverage thấp hơn oracle rải rác; chưa đo acceptance sau lược bỏ |
| Tập key quan trọng có cố định qua decoding không? | Overlap và coverage khi tái sử dụng tập chọn giữa các mốc và giữa lượt liền kề | Có locality ngắn hạn và drift dài hạn; chưa khóa lịch refresh |
| Có thể dùng target attention thay cho DFlash attention để chọn context không? | So sánh cùng lượt và parent-to-next-draft, theo target layer thấp/giữa/cao | Chưa đủ cơ sở cho predictor trực tiếp hoặc selector độc lập |
| Attention của literal bonus query có dự đoán được context ở draft sau không? | Không có transition `q=15` sang một lượt draft kế tiếp | Chưa được kiểm tra trong dữ liệu hiện có |
| Chọn/nén cache có giữ acceptance và giảm chi phí thật không? | Chưa có forward dưới cache đã chọn/nén trong chuỗi khảo sát mới | Cần can thiệp và benchmark tiếp theo |

### 2.2. Sổ thực nghiệm

| Mã trong tài liệu | Thực nghiệm | Quy mô và runtime | Kết quả/đầu ra đã có |
|---|---|---|---|
| D0 | Thăm dò DFlash trên một instance | 1 GovReport × 4 cap; 8 draft round/cap; 32 round; 1 Modal L40S | Histogram, heatmap, coverage prompt-conditioned và vector theo layer |
| D1 | DFlash toàn bộ key scope | 10 GovReport × 4 cap; 40 trajectory; 4.366 round; 1 Modal L40S | Prompt/output/anchor/mask mass, full-key vectors và 40 continuation tới EOS |
| D2 | Phân tích selection, concentration, spatial/temporal/layer | CPU, dùng lại toàn bộ vector D1 | Coverage trên 100% mass, oracle Top-K và fixed-position strategies |
| D2-C | Loại block khỏi phép đo; mô phỏng giữ cache + block | CPU, dùng lại chính 4.366 vector D1 | Cache-conditional coverage và tổng mass gốc giữ khi block luôn có mặt |
| D3 | Target và DFlash trên cùng proposal/lượt | Replay 10 × 4; 4.366 round; target layers 0/18/35, draft layers 0–4; 1 L40S | JS, Top-K overlap, coverage chéo, scope mass; 40/40 replay khớp |
| D4 | Parent target query và DFlash lượt kế tiếp | Replay cùng 10 × 4; 4.326 transition; 1 L40S | Ghép `q=k` đúng prefix, parent histograms và layer-pair metrics; 0 bonus transition |
| T0 | Nhật ký train DFlash baseline trên 4 B200 | Nhật ký server: step 2.155/8.430 sau 9h17m05s | Train loss/accuracy và checkpoint được mô tả trong log; chưa là eval acceptance trong khảo sát D1 |
| TF41–42 | Training-Free: head concentration và hybrid oracle | Qwen3-0.6B, target layer 27; 20 GovReport + 20 Multi-News; tối đa 32 token; Modal A10 | Có heterogeneity và headroom oracle; không có candidate qua toàn bộ gate đã khóa |
| TF43 | Temporal support reuse | Cùng model; adaptive pilot 40 doc và fixed-budget targeted pilot 20 doc; Modal A10 | Mean locality có tín hiệu, tail/cost gate không đạt; chưa physicalize KV |
| TF44 | FidelityKV fake quantization | Qwen3-0.6B; 40 doc/1.244 teacher-forced token; Modal A10 | Một số INT8 có logit headroom; các cấu hình nén uniform đều không qua toàn bộ gate |

D2 và D2-C **không thêm sample hoặc trajectory độc lập**. D3/D4 là replay của cùng 10 document, 40 conditions và continuation D1. Không cộng số round của các replay để tăng cỡ mẫu thống kê. Smoke, replaycheck và run bị loại chỉ dùng kiểm tra quy trình, không được gộp vào scientific aggregate.

Mốc “30 giờ trên 4 GPU” là chi phí người nghiên cứu cung cấp, tương đương 120 GPU-giờ nếu cả bốn GPU được dùng trong suốt thời gian đó. Chưa gắn được mốc này với một raw run MR cụ thể. Log T0 ghi một run **DFlash baseline**, dừng ở step 2.155 sau 9h17m05s; train accuracy 23,1% không chứng minh hội tụ acceptance. Các nhận định cũ “thu được >90% năng lực” hoặc “warm-start trong 1 giờ” chưa có kiểm chứng và không dùng làm claim cho paper. Xem [T0](2026-10-05-dflash-4gpu-training-log.md) và [phân tích hướng MR](2026-10-06_mr_dflash_research_directions.md).

### 2.3. Kết quả D0 và lý do chuyển sang toàn bộ key scope

D0 đã đo cả breakdown prompt/generated/block, nhưng hình và chỉ số concentration ban đầu ưu tiên **prompt-conditioned**. Bảng lịch sử dưới đây chỉ dùng cho truy vết; các insight chính của tài liệu dùng full mass và cache + block của D1 trở đi.

| Prompt D0 | Prompt mass / tổng | Top-1K / prompt mass | Best-window 1K / prompt mass |
|---|---:|---:|---:|
| 3K | 60,74% | 90,87% | 65,59% |
| 5K | 64,07% | 80,56% | 49,18% |
| 8K | 68,77% | 73,25% | 31,93% |
| 16K | 67,35% | 51,10% | 20,67% |

Tại 16K, D0 không quan sát 1K chứa gần toàn bộ prompt mass; vị trí ưu tiên đổi trong tám round đầu. D1 sau đó mở rộng sang mười document và toàn continuation, đưa output đã commit cùng block vào key scope. D0 và D1 khác sample/round weighting, nên không lấy chênh lệch hai bảng làm tác động nhân quả của output hoặc số instance.

Một coverage 90% **trong prompt** không có nghĩa 90% tổng mass. Quy đổi cần cộng mass đúng tập key trên mỗi round rồi mới aggregate. Do đó nhận xét cũ Top-1K prompt cao và block khoảng 20% không mâu thuẫn: chúng dùng mẫu số/phạm vi khác nhau.

## 3. Thiết lập, đối tượng đo và mẫu số

### 3.1. Thiết lập chung của D0–D4

Target là **Qwen3-4B**, drafter là **Qwen3-4B-DFlash-b16**, năm draft layer. D1 chọn ngẫu nhiên có seed 42 mười GovReport đủ dài từ `data/longbench_200/gov_report.jsonl`. Mỗi document tạo prefix source đạt prompt length chính xác **3.072 / 5.120 / 8.192 / 16.384 token** sau instruction và chat template; không dùng padding. Decode greedy, no thinking, tới EOS hoặc 512 output token, với guard 512 round.

D1 hoàn tất 40/40 trajectory tại EOS, output **111–435 token**, gồm EOS, tổng 4.366 draft round. Nhãn 3K/5K/8K/16K là **prompt ban đầu**; tổng số key còn tăng khi output được commit. Số key trung bình ở bốn cap khoảng 3.207 / 5.246 / 8.329 / 16.522.

GPU thực tế của các khảo sát DFlash chính là **một NVIDIA L40S trên Modal**, BF16, Python 3.12, torch `2.11.0+cu130`, transformers `5.12.1`. Target giữ SDPA; drafter dùng eager để thu attention. Timing của collector có instrumentation không dùng để claim latency production.

| Thành phần | Snapshot/thiết lập |
|---|---|
| Target weights | `1cfa9a7208912126459214e8b04321603b3df60c` |
| DFlash weights | `b74e3a329c4d963783143b1e970d95b002be72bd` |
| Target feature layer IDs, zero-based | `[1, 9, 17, 25, 33]` |
| Target attention layers trong D3/D4 | `[0, 18, 35]` của 36 layer |
| Draft attention layers | `[0, 1, 2, 3, 4]` |
| Block | 1 anchor + 15 mask/proposal positions |

**Các khảo sát attention dùng checkpoint upstream trong manifest, không dùng checkpoint server T0.** Vì vậy không gán kết quả D1 cho DFlash baseline do nhóm train hoặc cho MR-DFlash.

### 3.2. DFlash nhận gì từ target?

Trong source hiện tại, DFlash nối hidden states từ năm target feature layer, chiếu qua `fc` và normalization, rồi mỗi draft layer dùng `k_proj`/`v_proj` của chính nó để tạo context K/V. Draft queries và K/V của block cũng được tạo từ hidden states của drafter. `cache_target` phục vụ verification và `cache_draft` phục vụ drafting là hai cache riêng.

Vì vậy “KV từ target” trong định hướng này được hiểu chính xác là **context KV của drafter được tạo từ target hidden features**. Không có giả định rằng năm draft layer sử dụng nguyên cùng ma trận K/V của một target attention layer. Nguồn features chung không bắt buộc Q/K projections, query states hoặc attention rankings của các layer giống nhau. Xem [source DFlash](../../externals/dflash/dflash/model.py) và [collector](../../scripts/probe_dflash_attention.py).

### 3.3. Attention được thu ở đâu?

DFlash attention lấy từ attention weights **sau softmax trong self-attention của từng draft Transformer layer**. Vector riêng layer đã mean qua query heads và **15 proposal queries**; anchor query bị loại khỏi phép mean. Overview mean tiếp năm layer. Đây là mean các xác suất attention, không phải mean raw QK scores rồi mới softmax.

Với draft layer ℓ, query head h, proposal query j và key i:

```text
a_t^ℓ(i) = mean_(h, j=1..15) softmax(QKᵀ + mask)_ℓ,h,j,i
p_t(i)   = mean_ℓ a_t^ℓ(i), rồi chuẩn hóa trên mọi key để loại sai số số học nhỏ
```

NPZ D1 giữ năm vector riêng layer trên mọi key; không giữ đầy đủ tensor riêng head/query cho mọi round. **“100%” là toàn bộ key mass của những proposal queries được đo**, không phải thu mọi attention của target hoặc mọi query của hệ thống.

D3/D4 tái dựng target probabilities từ Q/K của cùng forward, áp đúng scaling/mask và softmax float32; phép SDPA tạo output target được giữ nguyên. Không ghi nhận target mass lên verifier key tương lai. Cách thu target khác cơ chế hook eager DFlash, nên replay invariance và tổng mass được kiểm tra riêng.

### 3.4. Phạm vi cần nén và công thức coverage

Ở draft round t, gọi P là prompt ban đầu, O_t là output đã commit trước round, C_t = P ∪ O_t là cache prefix, B_t là block 16 vị trí hiện tại. Toàn bộ key scope là U_t = C_t ∪ B_t.

```text
Coverage toàn bộ: M_t(S) = Σ_(i∈S) p_t(i), với Σ_(i∈U_t) p_t(i) = 1
Block mass:       b_t = M_t(B_t)
Cache mass:       m_t = M_t(C_t) = 1 - b_t
Cache profile:    q_t(i) = p_t(i)/m_t, i∈C_t
Cache coverage:   C_t(S) = Σ_(i∈S) q_t(i)
Giữ block + S:    R_t(S) = b_t + m_t C_t(S)
```

Chỉ số chính cho phương pháp dùng **R_t trên tổng mass gốc**, vì block luôn được giữ. C_t là chẩn đoán phần cache mà module tác động. Công thức được tính từng round rồi mới lấy mean; không nhân các mean `m` và `C` thay cho mean tích.

Top-K mọi key có tổng budget K. Top-K cache + block có budget **K cache key + 16 block key**. Hai loại budget này không hoàn toàn giống nhau: ví dụ 67,29% Top-1K mọi key và 67,43% Top-1K cache + block cùng hợp lệ. Top-K là K vị trí attention cao nhất, có thể rải rác; “Top-K” không có nghĩa K token ở đầu sequence. 1K = 1.024, 4K = 4.096.

Các oracle Top-K dùng attention của chính forward dense đang đánh giá. Chúng là chẩn đoán coverage tối ưu của việc giữ K key gốc, **chưa là một selector có thể chạy trước draft**. Một module tạo memory mới có thể tổng hợp thông tin từ nhiều key; Top-K dense không phải giới hạn lý thuyết của module đó.

### 3.5. Aggregation và cách đọc hình

Tính mean các round trong từng document, rồi mean đều mười document. Với D3/D4, tính metric cho từng cặp target layer × draft layer rồi tổng hợp như mô tả trong artifact. Top-K của vector mean khác mean coverage khi từng layer tự chọn Top-K riêng.

Histogram D1 dùng prompt bin 1K, output bin 128, anchor một key và nhóm mask 15 key. **Chiều cao là tổng mass của bin**, không phải mass trung bình mỗi token; các bin có số token khác nhau. Mốc đầu/giữa/cuối D1 là record đầu, record `len(records)//2`, record cuối; không đảm bảo mốc giữa tương ứng 50% output token. Parent histogram bắt đầu ở DFlash lượt 2 vì lượt đầu chưa có parent từ verification trước đó.

Ở lượt draft đầu, output **đã commit trong cache** bằng 0 nên cột O bằng 0. Anchor đã tồn tại trong block và thường là token đầu tiên target dự đoán sau prefill; do đó cột anchor vẫn có mass. Mask positions là input lúc DFlash forward, không được diễn giải như token dự đoán cuối cùng. Hidden states ở các vị trí này tiếp tục biến đổi qua layer.

Bốn cap sử dụng các prefix source khác nhau và sinh output khác nhau. Xu hướng theo cap chưa cô lập được tác động của độ dài với tác động nội dung. Round trong một trajectory có tương quan; 4.366 round không phải 4.366 mẫu thống kê độc lập. Các range đã có là min–max giữa document, chưa phải confidence interval hoặc kiểm định significance.

## 4. Insight về context mà DFlash sử dụng

### I1. Output đã commit và block nhận mass lớn; chỉ xem prompt sẽ thiếu phần quan trọng

| Prompt ban đầu | Draft round | Prompt / tổng | Output / tổng | 15 mask / tổng | Anchor / tổng | Block = mask + anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3K | 818 | 50,40% | 26,77% | 21,54% | 1,29% | 22,82% |
| 5K | 914 | 53,93% | 24,22% | 20,73% | 1,12% | 21,85% |
| 8K | 1.206 | 55,04% | 22,64% | 21,25% | 1,07% | 22,32% |
| 16K | 1.428 | 58,80% | 19,85% | 20,35% | 1,00% | 21,35% |

Mỗi hàng dùng toàn bộ key mass làm mẫu số. Nhóm mask nhận khoảng 20–21% mean mass qua các cap, nhưng output mass giảm từ 26,77% xuống 19,85%. Vì vậy không coi “20% output và 20% mask” là hằng số theo cap, round hoặc layer.

Tại 16K, mean mười document ở lượt đầu/giữa/cuối là prompt **75,63% / 60,28% / 56,87%**, output **0% / 19,16% / 22,91%**, block **24,37% / 20,56% / 20,23%**. Output mass tăng trong trajectory; ổn định tương đối sau đầu sinh không đồng nghĩa cố định toàn thời gian.

**Khi loại block:** C_t vẫn chứa output đã commit. Các insight cache phía dưới tiếp tục tính cả prompt và output; không chuyển về mẫu số prompt-only. Output cần được append vào memory khi được target commit. Block bị loại chỉ trong phép chẩn đoán rồi được giữ sẵn trong phương pháp.

**Hàm ý:** module nên xử lý prefix động thay vì chỉ chọn một tập source token lúc prefill. Chưa thể suy từ mass lớn rằng nhóm output hoặc mask đóng góp đúng một tỷ lệ tương ứng vào chất lượng logits.

![Hình 1: DFlash attention trên prompt, output đã commit, anchor và mask; mean đều mười document, bốn cap và ba mốc sinh](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/attention_histograms_full_mean.png)

*Hình 1.* Mọi cột trong một panel cộng thành 100% key mass; prompt bin 1K và output bin 128 có độ rộng khác nhau. Lượt đầu chưa có committed output. [PDF vector](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/attention_histograms_full_mean.pdf).

### I2. Key có score cao nằm rải rác; cửa sổ liên tiếp bỏ ngoài nhiều mass

| Prompt | Top-1K / tổng | Best-window 1K / tổng | Top-4K / tổng | Best-window 4K / tổng |
| --- | ---: | ---: | ---: | ---: |
| 3K | 92,38% | 66,27% | 100,00% | 100,00% |
| 5K | 83,92% | 59,61% | 98,89% | 82,89% |
| 8K | 76,96% | 55,62% | 93,59% | 73,70% |
| 16K | 67,29% | 49,61% | 82,61% | 63,36% |

Coverage trong bảng dùng tổng mass trên mọi key. Tại 16K, chọn rải rác tốt hơn best-window **17,68 điểm** với 1K và **19,25 điểm** với 4K. Với budget 1K, key đầu/giữa/cuối nhận **5,83% / 1,99% / 49,61%**; chia 512 đầu + 512 cuối giữ 50,91%, thấp hơn Top-1K 67,29%.

**Khi loại block và giữ block sẵn trong phương pháp:**

**Budget 1K cache key + 16 block key:**

| Prompt | Top-K / cache | Best-window / cache | Top-K + block / tổng | Window + block / tổng |
| --- | ---: | ---: | ---: | ---: |
| 3K | 90,31% | 57,83% | 92,51% | 67,49% |
| 5K | 79,68% | 48,60% | 84,11% | 59,92% |
| 8K | 70,56% | 42,87% | 77,13% | 55,68% |
| 16K | 58,58% | 35,95% | 67,43% | 49,66% |

**Budget 4K cache key + 16 block key:**

| Prompt | Top-K / cache | Best-window / cache | Top-K + block / tổng | Window + block / tổng |
| --- | ---: | ---: | ---: | ---: |
| 3K | 100,00% | 100,00% | 100,00% | 100,00% |
| 5K | 98,61% | 78,35% | 98,92% | 83,18% |
| 8K | 91,81% | 66,12% | 93,64% | 73,77% |
| 16K | 77,96% | 53,42% | 82,66% | 63,40% |

Tại 16K, Top-1K cache giữ 58,58% cache mass so với 35,95% của best-window; khi cộng block, coverage tổng tương ứng **67,43% / 49,66%**. Top-4K cache + block giữ **82,66%**, best-window 4K cache + block giữ **63,40%**. Chênh lệch rải rác–liên tiếp vẫn còn trong phần cache sẽ nén.

Trong full-key analysis, best-window 1K/4K trùng cửa sổ cuối ở 4.366/4.366 round, vì cửa sổ chứa cả output và block. Sau khi loại block, tỷ lệ này là **4.140/4.366** và **4.302/4.366**; vì vậy không dùng phát biểu “cửa sổ cuối luôn tối ưu” cho cache.

**Hàm ý:** cần khảo sát selection phân tán hoặc memory tổng hợp nhiều vùng. Chưa có bằng chứng việc lấy recent-only đủ giữ acceptance; recency vẫn là đối chứng cần có.

![Hình 2: coverage cache sau loại block và tổng mass giữ khi block luôn được giữ](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_selection_coverage.png)

*Hình 2.* Các chiến lược dùng cùng budget cache. Cột cache-conditional và cột tổng + block có mẫu số khác nhau. [PDF vector](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_selection_coverage.pdf).

### I3. Chọn đúng Top-K vẫn bỏ ngoài phần mass đáng kể; budget cố định khó giữ cùng coverage

| Prompt | Top-1K / tổng | Top-2K / tổng | Top-4K / tổng | Top-8K / tổng | Key cho 90% tổng mass | Key cho 95% tổng mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3K | 92,38% | 97,75% | 100,00% | 100,00% | 772 | 1.387 |
| 5K | 83,92% | 92,10% | 98,89% | 100,00% | 1.706 | 2.638 |
| 8K | 76,96% | 84,92% | 93,59% | 99,97% | 3.066 | 4.593 |
| 16K | 67,29% | 74,25% | 82,61% | 92,10% | 7.035 | 10.129 |

Ở 16K, ngoài Top-1K/4K còn **32,71% / 17,39% tổng mass**. Budget tăng bốn lần từ 1K lên 4K chỉ tăng coverage 15,32 điểm. Tại 3K, 4K lớn hơn key count nên coverage 100% là trường hợp giữ toàn bộ, không phải một kết quả nén.

**Khi loại block:** Top-1K/4K cache giữ **58,58% / 77,96% cache mass** tại 16K; ngoài tập còn 41,42% / 22,04% cache mass. Khi giữ block + Top-4K cache, vẫn còn **17,34% tổng mass gốc** ngoài tập. Đạt 90%/95% **cache mass** cần khoảng **8.187 / 10.983 cache key**; đạt 95% **tổng mass khi block giữ sẵn** cần khoảng **10.113 cache key + 16**.

**Hàm ý:** chỉ giữ nguyên một tập nhỏ raw key có giới hạn coverage quan sát được. Memory tổng hợp phần ngoài tập là một giả thuyết đáng khảo sát. Attention tới memory sau nén sẽ thay đổi, nên không đọc 17,34% mass bị bỏ như 17,34% accuracy bị mất, hoặc dùng Top-K dense để giới hạn module tạo đại diện mới.

### I4. Context dài có phần mass trải rộng hơn, đồng thời vẫn có nhóm rất tập trung

| Prompt | Top 10% key / tổng | Tỷ lệ key cho 90% tổng mass | H/logN toàn scope | Top 10% cache / cache mass | Tỷ lệ cache key cho 90% cache mass | H/logN cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3K | 81,16% | 23,99% | 0,722 | 76,25% | 30,66% | 0,749 |
| 5K | 76,43% | 32,51% | 0,737 | 70,25% | 38,97% | 0,775 |
| 8K | 74,81% | 36,80% | 0,737 | 67,80% | 43,85% | 0,785 |
| 16K | 71,95% | 42,58% | 0,745 | 64,45% | 49,60% | 0,803 |

So cùng tỷ lệ key giúp tránh kết luận chỉ từ việc 1K trở thành budget nhỏ hơn tương đối. Top 10% mọi key giảm từ 81,16% xuống 71,95% mass; tỷ lệ key cần cho 90% mass tăng từ 23,99% lên 42,58%. Tại 16K, Top-16 nhận **26,23%**, Top-64 nhận **46,53%**, dù Top-64 chỉ khoảng 0,39% key count.

**Khi loại block:** Top 10% cache vẫn giữ **64,45% cache mass**, Top-16/64 cache nhận **22,49% / 34,31%** tại 16K. Cache entropy chuẩn hóa tăng từ 0,749 lên 0,803; tỷ lệ key cần cho 90% cache mass tăng từ 30,66% lên 49,60%. Nhóm tập trung không chỉ đến từ block.

**Hàm ý:** dữ liệu phù hợp với thiết kế giữ một số chi tiết được ưu tiên và học đại diện cho phần rộng còn lại. Chưa thể kết luận tỷ lệ hai nhánh, số memory slot hoặc nhu cầu giữ toàn bộ phần tail. Câu “attention trở nên trải đều” cần được thay bằng “độ tập trung tương đối giảm và nhiều key hơn cùng chia sẻ phần mass còn lại”; phân bố vẫn khác xa uniform.

![Hình 3: độ tập trung trên cache, không chứa block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_concentration.png)

*Hình 3.* Coverage oracle và entropy mô tả vector attention dense đã aggregate; chưa đo drafting dưới cache nén. [PDF vector](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_concentration.pdf).

### I5. Thiên lệch về cuối vẫn mạnh; hai vùng đầu/cuối chưa bao phủ phần lớn cache một cách ổn định

| Prompt | 10% key đầu / tổng | 10% key cuối / tổng | Hai đầu / tổng | 10% cache đầu / cache | 10% cache cuối / cache | Hai đầu / cache |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3K | 12,63% | 58,91% | 71,54% | 16,49% | 47,01% | 63,50% |
| 5K | 11,13% | 55,46% | 66,60% | 14,36% | 43,12% | 57,48% |
| 8K | 6,35% | 54,79% | 61,14% | 8,24% | 41,83% | 50,07% |
| 16K | 7,50% | 50,93% | 58,43% | 9,55% | 37,61% | 47,16% |

Bin chứa khoảng 10% số key được chia riêng trên U_t hoặc C_t. Ở 16K, 10% key cuối full sequence nhận **50,93% tổng mass**; khi bỏ block, 10% cache cuối vẫn nhận **37,61% cache mass**. Hai vùng đầu/cuối cache gộp nhận 47,16%; **52,84% cache mass còn ở 80% vị trí giữa**.

**Khi loại block:** ưu thế recency giảm nhưng còn rõ. Phần cuối cache chứa cả cuối prompt và output đã commit, nên chưa quy được spike cuối hoàn toàn cho nội dung ở cuối report. So các cap cũng thay prefix nội dung.

**Diễn giải attention sink:** ưu tiên đầu/cuối là quan sát về phân phối; attention sink ở đầu và recency/output ở cuối là các giải thích có thể có. Các token format được attend cao trong I8 bổ sung động cơ khảo sát sink, nhưng chưa xác nhận cơ chế nhân quả. Không gọi mọi spike ở cuối là attention sink hoặc coi high-mass key mặc định là evidence nội dung.

**Hàm ý:** giữ recent context là đối chứng hợp lý; allocation cố định chỉ theo đầu/cuối chưa được chứng minh đủ. Trong toàn scope, spike cuối còn bao gồm block hiện tại. Phát biểu “hai nửa đầu/cuối” không chính xác với phép đo: các bảng này xét hai **bin 10% ở hai đầu**, không phải hai nửa vốn cộng thành toàn sequence.

### I6. Mass theo nhóm khá ổn định, nhưng tập key thay đổi; có locality ngắn hạn để khảo sát reuse

| Prompt | Top-1K full: liền kề | Top-1K cache: liền kề | Top-1K full: giữa–cuối | Top-1K cache: giữa–cuối | Top-4K cache: giữa–cuối |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3K | 88,38% | 88,31% | 54,29% | 53,89% | 100,00% |
| 5K | 86,07% | 85,93% | 49,20% | 48,73% | 88,10% |
| 8K | 83,03% | 82,85% | 37,18% | 36,42% | 68,96% |
| 16K | 82,88% | 82,65% | 31,92% | 31,11% | 49,67% |

Overlap là số key ID chung chia budget tập trước, không phải Jaccard. Full-key temporal analysis matching persistent prefix bằng vị trí tuyệt đối và block bằng slot; một block slot chung qua round không biểu thị cùng token/hidden state. Cache-only analysis loại hoàn toàn block slots.

Ở 3K, Top-4K cache chứa toàn bộ cache lúc chọn, nên overlap giữa–cuối bằng 100%; output key mới xuất hiện sau đó chưa có trong tập cũ và vẫn có thể nhận mass. Không đọc 100% overlap này như coverage 100% tại lượt cuối.

Tại 16K, Top-1K/4K cache trùng **82,65% / 88,87%** giữa hai lượt liền kề, nhưng chỉ **31,11% / 49,67%** giữa lượt giữa–cuối. So sánh coverage **ở lượt cuối**, với block luôn được giữ:

| Budget cache | Cache từ lượt giữa / cache mass | Chọn lại / cache mass | Tập cũ + block / tổng mass | Chọn lại + block / tổng mass |
|---|---:|---:|---:|---:|
| 1K | 25,36% | 57,81% | 40,47% | 66,35% |
| 4K | 43,30% | 77,52% | 54,78% | 82,07% |

**Khi loại block:** thay đổi tập cache vẫn rõ. Tái sử dụng tập của round liền trước mất trung bình **5,99 / 5,67 điểm cache coverage** với 1K/4K; trên tổng mass khi giữ block là **4,72 / 4,46 điểm**. Các mean liền kề chỉ dùng round có lượt trước; không thay bằng mean toàn trajectory.

Chênh lệch dài hạn gồm thay weight trên prefix cũ và key output mới xuất hiện. **Hàm ý:** khảo sát append output mới, refresh ưu tiên và reuse ngắn hạn. Chưa có cơ sở khóa refresh mỗi round hoặc mỗi 4/8 round; lựa chọn cần acceptance, tail-risk và overhead thực tế.

![Hình 4: drift tập cache và coverage tổng khi giữ block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_temporal_selection.png)

*Hình 4.* So tập chọn theo thời gian trên vector dense; không phải trajectory sinh bằng cache cũ. [PDF vector](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_temporal_selection.pdf).

### I7. Năm layer có nhu cầu khác nhau; mean attention là proxy của memory chung

Trong mỗi phạm vi full hoặc cache, chọn Top-K chung bằng vector mean năm layer, rồi đánh giá trên từng layer ở 16K. Hai phạm vi có tập chọn riêng:

| Draft layer | Top-1K chung full / tổng | Top-4K chung full / tổng | Top-1K chung cache / cache | Top-1K cache + block / tổng |
| --- | ---: | ---: | ---: | ---: |
| L0 | 53,19% | 73,83% | 49,76% | 53,36% |
| L1 | 69,48% | 81,63% | 49,38% | 69,56% |
| L2 | 80,58% | 91,45% | 69,64% | 80,70% |
| L3 | 62,55% | 79,05% | 56,32% | 62,70% |
| L4 | 70,65% | 87,11% | 68,15% | 70,83% |
| Vector mean | 67,29% | 82,61% | 58,58% | 67,43% |

**Khi loại block:** một tập Top-1K cache chung giữ khoảng **49,38–69,64% cache mass** theo layer. Layer 1 nhận coverage tổng cao hơn layer 0 một phần nhờ block có nhiều mass hơn; vì vậy cần xem cache coverage bên cạnh tổng coverage.

Nếu từng layer tự chọn Top-1K riêng trên mọi key, mean coverage là **73,87%**, cao hơn **67,29%** khi dùng một tập chung. Đây là các tập khác nhau theo layer, không phải một tập 1K đạt 73,87% cho cả model. Mean layer sau chuẩn hóa cache cũng khác chuẩn hóa vector mean: những layer có cache mass lớn đóng góp nhiều hơn vào cache profile gộp.

**Hàm ý:** module chung cần được đánh giá theo từng layer; layer-specific memory là ablation tiềm năng. Việc tất cả layer nhận features từ target không bảo đảm attention giống nhau. Chưa suy được layer nào quyết định acceptance, hoặc head/query nào có thể bỏ, từ dữ liệu đã aggregate.

### I8. Vị trí attention cao chưa tương đương evidence nội dung hoặc thông tin cần thiết

D0 ghi các token format như `<|im_start|>`, `</think>` và xuống dòng trong nhóm được attend cao. Prompt chứa report, instruction và chat template. Hidden feature tại một vị trí có thể mang thông tin từ context rộng hơn chính token văn bản ở vị trí đó.

**Khi loại block:** các vấn đề này vẫn còn trong cache prompt/output. Tỷ trọng cao của một punctuation hoặc sink key không đủ để gọi nó là evidence nội dung; cũng chưa đủ để gọi nó vô ích. Với một head/query, attention output là `Σ_i a_i V_i`; ảnh hưởng lên logits còn phụ thuộc giá trị V, hướng/norm, redundancy và các layer sau.

**Hàm ý:** attention labels có thể là proxy hoặc supervision phụ. Giá trị của một context key cần được kiểm tra qua can thiệp và ảnh hưởng tới draft logits/acceptance. Memory-distillation loss có thể khảo sát tái tạo attention output hoặc logits, thay vì chỉ tối đa mass giữ.

## 5. Target attention có dự đoán được context của DFlash không?

Hai phép ghép D3 và D4 trả lời hai câu hỏi khác nhau. D3 so sánh model trên cùng proposal đã draft; D4 kiểm tra tín hiệu target có trước lượt DFlash mới. Không gộp hai bảng thành cùng một predictor evaluation.

### 5.1. D3: target và DFlash cùng lượt, cùng proposal được dự đoán

DFlash mask queries 1–15 được ghép với target verifier queries 0–14, vì chúng dự đoán cùng proposal token index. Target layer 0/18/35 và DFlash layer 0–4 được đo riêng. JS dùng log₂, phạm vi [0,1], giá trị 0 khi hai vector giống nhau.

| Prompt | JS toàn scope | JS cache-conditional | Overlap Top-1K | Target mass trên DFlash Top-1K / tổng | DFlash mass trên Target Top-1K / tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3K | 0,404 | 0,478 | 49,05% | 56,81% | 54,80% |
| 5K | 0,427 | 0,503 | 33,62% | 50,13% | 44,92% |
| 8K | 0,441 | 0,520 | 25,96% | 48,81% | 37,93% |
| 16K | 0,459 | 0,539 | 19,81% | 46,07% | 30,79% |

Cross coverage dùng tổng mass của model đang đánh giá, chỉ cộng weight trên Top-1K cache của model kia. Overlap dùng số vị trí chung chia 1.024. Phần block vẫn nằm trong denominator JS toàn scope; target causal proposal queries và DFlash noncausal mask queries có khác biệt cấu trúc.

Tại 16K, mass prompt/output/block trung bình target là **56,6% / 15,9% / 27,5%**, DFlash là **58,8% / 19,9% / 21,4%**. Có tương đồng ở nhóm rộng, trong khi thứ hạng token cụ thể chỉ trùng 19,8% Top-1K.

**Khi loại block:** JS cache-conditional còn 0,539 ở 16K, nên khác biệt không chỉ do block. **Kết luận:** chưa có một ranking KV chung đủ ổn định để dùng target attention như ground truth duy nhất cho DFlash selector. Tổng prompt mass gần nhau không chứng minh cùng dùng một tập key.

Target verifier attention của **cùng round** có sau khi DFlash đã tạo proposals. Dùng nó để chọn cache trước chính lượt draft đó sẽ nhìn thông tin chưa có ở thời điểm triển khai. D3 phục vụ chẩn đoán/nhãn offline; tín hiệu online phải được ghép theo thời gian như D4.

![Hình 5: JS theo target layer và draft layer trong phép ghép cùng lượt](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_js_matrices.png)

*Hình 5.* Ma trận là mean theo document của metric từng layer pair. [PDF vector](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_js_matrices.pdf).

### 5.2. D4: query target tạo anchor ở lượt trước và DFlash ở lượt kế tiếp

Ở verifier round r, gọi k là số proposal khớp liên tiếp. Query target `q=k` dự đoán token được dùng làm anchor cho round r+1, nếu decoding tiếp tục. `k<15` là correction query; `k=15` là literal bonus query. Parent query nhìn thấy prefix dài `prefix_r + k + 1`, phải khớp đúng cache prefix của DFlash r+1; điều kiện này đã kiểm tra trên mọi transition.

Parent target vector được pad mass 0 lên **anchor mới + 15 mask mới** khi so trên toàn U_(r+1), vì các key đó chưa tồn tại trong parent forward. Cache-conditional metric chuẩn hóa riêng phần prefix chung. Lượt DFlash đầu chưa có preceding verifier parent trong protocol này, nên không thuộc D4.

| Prompt | Transition | JS toàn scope | JS cache-conditional | Overlap Top-1K | Parent Top-1K / parent mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3K | 808 | 0,544 | 0,475 | 47,31% | 94,49% |
| 5K | 904 | 0,562 | 0,499 | 32,62% | 92,27% |
| 8K | 1.196 | 0,582 | 0,520 | 25,31% | 90,00% |
| 16K | 1.418 | 0,604 | 0,548 | 19,45% | 87,27% |

| Prompt | DFlash mass trên Parent Top-1K / tổng | DFlash mass trên Parent Top-1K / cache | Parent Top-1K + block giữ sẵn / tổng |
| --- | ---: | ---: | ---: |
| 3K | 52,79% | 68,82% | 75,58% |
| 5K | 43,31% | 55,71% | 65,13% |
| 8K | 36,72% | 47,60% | 59,01% |
| 16K | 29,80% | 38,01% | 51,12% |

Cột cuối được suy trực tiếp từ hai metric cùng transition/layer pair: `parent-selected cache mass của DFlash + DFlash block mass`. Phép cộng tuyến tính khớp mean từng cặp. Đây vẫn là **coverage dense đã quan sát**, chưa chạy DFlash dưới tập cache do parent chọn. Parent Top-K được chọn riêng theo target layer và đánh giá từng draft layer; không xem cột này như coverage của một selector đã hợp nhất ba target layer.

Không xem 51,12% parent-selected cache + block và 67,43% oracle shared-cache + block ở I2 như một so sánh phương pháp paired đã khóa: D4 tính theo layer pair và chỉ trên transition có parent, còn I2 chọn trên vector mean và dùng mọi round. Muốn so selector trực tiếp cần cùng query/layer aggregation, tập round và budget trong can thiệp kế tiếp.

Tại 16K, Parent Top-1K giữ **87,27% mass của chính parent query**, nhưng nhận 29,80% tổng mass của DFlash lượt sau, hay 38,01% cache mass. Chiều ngược lại, DFlash Top-1K cache nhận **70,28% parent mass**. Coverage chéo bất đối xứng: parent attention tập trung không kéo theo tập đó đủ cho DFlash.

**Khi loại block:** JS cache-conditional 0,548 và coverage cache 38,01% vẫn thể hiện khác biệt. Với phương pháp giữ block, coverage dự kiến trên tổng mass gốc là 51,12%; vì vậy cần tránh gọi 29,80% là coverage của toàn bộ tập mà phương pháp thực sự giữ.

**Kết luận của D4:** chưa thấy quan hệ ổn định đủ để dùng trực tiếp parent target attention làm selector độc lập cho DFlash ở context dài. Thực nghiệm chưa kiểm tra một module học ánh xạ từ target attention/features sang nhu cầu DFlash; cũng chưa chạy can thiệp để đo chất lượng một selector parent.

### 5.3. Giới hạn riêng của giả thuyết bonus token

D4 có **4.326 transition**, query index mean **1,12**, max **10**. Tất cả transition là correction-to-next-draft; **literal bonus `q=15` có 0 transition**.

Trong raw log có **7 record cuối với `accepted_proposals=15`**, nhưng mỗi record chỉ commit **3–9 token** vì continuation gặp EOS trong block được chấp nhận. Chỉ số raw k kiểm tra cả block trước khi cắt EOS; những record này không tạo anchor cho một lượt DFlash sau đó. Vì vậy không xem chúng là bảy quan sát bonus-to-next-draft.

Cách viết đúng cho paper: **“Khảo sát parent-query hiện tại chưa thu được cặp literal bonus query với draft kế tiếp; quan hệ bonus-to-next-draft vẫn chưa được đo.”** Kết luận bất lợi cho direct parent ranking trong D4 áp dụng cho các correction-query transition quan sát được; chưa bác bỏ riêng giả thuyết bonus query.

![Hình 6: parent target query ở round r và DFlash ở round r+1; target layer thấp/giữa/cao](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/figures/parent_attention_histograms_fullmass.png)

*Hình 6.* DFlash mean năm layer và ba parent target layer; mean đều mười document ở chuyển tiếp đầu/giữa/cuối. Parent có mass 0 trên block mới theo định nghĩa. [PDF vector](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/figures/parent_attention_histograms_fullmass.pdf).

### 5.4. Target layer ảnh hưởng kết luận; similarity chưa là usefulness

| Target layer tại 16K | D3 JS full | D3 JS cache | D4 parent JS full | D4 parent JS cache |
| --- | ---: | ---: | ---: | ---: |
| T0 | 0,396 | 0,461 | 0,557 | 0,495 |
| T18 | 0,428 | 0,483 | 0,550 | 0,485 |
| T35 | 0,553 | 0,672 | 0,704 | 0,666 |

Layer 35 lệch rõ hơn layer thấp/giữa trong cả hai phép ghép. Tuy nhiên layer nào có JS thấp hơn chưa được chứng minh giữ acceptance tốt hơn khi dùng để chọn key. JS toàn scope của D4 còn chịu việc parent không có block mới; không so trực tiếp JS D3/D4 để tuyên bố predictor tốt/xấu hơn theo một test đã khóa.

Không dùng phát biểu tuyệt đối “không có bất kỳ phần nào giống nhau” hoặc “target attention không chứa thông tin hữu ích”: D3 có group mass tương đối gần nhau, D4 có partial overlap/cross coverage. Phát biểu đã được bằng chứng hỗ trợ là **chưa có cơ sở dùng trực tiếp target attention để dự đoán đủ context DFlash hoặc làm selector duy nhất**. Khả năng học một ánh xạ và ích lợi sau nén chưa được kiểm tra.

## 6. Liên hệ với nghiên cứu Training-Free đã thực hiện

Chuỗi Training-Free trước đó nghiên cứu attention/source KV của **target Qwen3-0.6B**; chuỗi mới khảo sát context phục vụ **DFlash Qwen3-4B**. Các model, layer, dữ liệu chọn mẫu, output budget, GPU và intervention scope khác nhau. Dùng kết quả cũ làm động cơ/kinh nghiệm thiết kế, không gộp chúng thành bằng chứng trên DFlash hoặc bảng speedup chung.

| Thực nghiệm trước | Quan sát quan trọng | Giới hạn và hàm ý cho hướng mới |
|---|---|---|
| E41 | Mean K95 source: GovReport 24,29%, Multi-News 38,47%; nhu cầu giữa head khác nhau | Một số head cần context rộng; chỉ đo layer 27, ≤32 output token, dev cohort |
| E42 | Hybrid global 10% + routed 20% có source expansion khoảng 30,005%/30,010%; P99 missed mass 4,400%/5,814% cho GovReport/Multi-News | Oracle có headroom nhưng không cấu hình nào đạt toàn bộ gate trên hai dataset; expansion chưa là physical cost |
| E43 | Candidate R8/C16/F0.1: source cost 52,55%, mean missed 0,448%, P99 12,89%; không qua gate <50%/≤1%/≤5% | Locality trung bình không đủ kiểm soát tail; refresh/gather/blockification làm working set tăng; cost là mô phỏng analytical |
| E44 | K8V8: Top-1 98,875%, ΔNLL +0,024%, attention-output error mean/P99 2,906%/6,367%; uniform quantization không qua toàn bộ gate | NLL thấp không bảo đảm prediction agreement hoặc distortion; fake quantize/dequantize chưa giảm physical KV bytes |

E41/E42 dùng 40 document; E43 adaptive pilot dùng 40 document, fixed targeted pilot 20 document. E44 dùng 40 document với 1.244 teacher-forced token, so với dense BF16 replay mới. Không lấy 40 doc của mỗi phase làm các cohort độc lập nếu chúng dùng cùng tài liệu. Chưa có E43 masked generation/physical KV executor hoặc E44 packed-cache speedup trong các report này.

Mối nối nghiên cứu hợp lý là: **attention có cấu trúc và locality, nhưng source selection bằng quy tắc/score oracle chưa đủ vượt cost–tail-risk contract ở thiết lập cũ; khảo sát mới mở câu hỏi liệu memory học cho drafter có tổng hợp được phần phân tán với chi phí nhỏ hơn hay không.** Full target verification tạo phạm vi khác: compression có thể ảnh hưởng acceptance/tốc độ draft, còn verifier giữ thông tin đầy đủ. Điều đó chưa tự chứng minh phương pháp hiệu quả hoặc cho phép dùng gate E43 như một ngưỡng acceptance của DFlash.

Nguồn: [E41/E42](2026-09-21_trainingfree_controlled_search.md), [E43](2026-09-23_trainingfree_e43_temporal_support_reuse.md), [E44](2026-09-23_fidelitykv_e44_results.md). Các baseline Llama/A100 trong report cũ là bối cảnh systems riêng, không dùng làm đối chứng paired cho khảo sát Qwen/L40S này.

## 7. So với *Strong Drafts Need Compact Memories*

Nguồn đã đọc là [PDF nhóm cung cấp](../../outputs/Strong%20Drafts%20Need%20Compact%20Memories.pdf): **Tong Yuan, Chengxi Liao, Zeyi Wen — Strong Drafts Need Compact Memories: Long-Context Speculative Decoding with Compressed KV Cache**, metadata trên PDF ghi arXiv `2608.30252v1`, 31-08-2026. Phân tích dưới đây dựa trực tiếp vào §4–5 của PDF; đây là kết quả/thiết kế của công trình khác.

| Khía cạnh | MASW trong PDF | Hướng DFlash đang phát triển |
|---|---|---|
| Ý tưởng chung | Giảm memory phía draft, target giữ full KV, giữ năng lực draft mạnh | Trùng động cơ rộng; không xem riêng ý tưởng “draft memory nén + full verifier” là novelty mới |
| Drafter | Independent autoregressive Llama draft | DFlash block-parallel, context K/V được tạo từ target hidden features |
| Memory | Sink KV + exact local window + learned slots tích lũy cho lịch sử xa | Chưa chọn kiến trúc; dữ liệu gợi ý selection phân tán, tổng hợp phần rộng và cập nhật theo trạng thái |
| Learning | Freeze draft backbone; học mirrored K/V projections cho memory slots | Đề xuất giữ DFlash-5L và học module nhỏ; chưa triển khai/chạy pilot này |
| Cập nhật | Slot định kỳ, slot mới đọc local/sink/slot cũ; append và rollback | Cần thiết kế theo committed output và semantics verification của DFlash |
| Vị trí | Slot chia sẻ RoPE position với raw token kế tiếp; raw positions giữ nguyên | Chưa khóa scheme cho memory tổng hợp; selection raw key cần giữ đúng absolute position |
| Block/rollback | 16-token raw-KV rollback window | 16 vị trí draft hiện tại: 1 anchor + 15 proposal queries/keys; đây là hai đối tượng khác nhau |
| Training cost | PDF ghi pretrain adaptor trên 2B token rồi SFT task data, truncate 8K | Không dùng bài này để suy ra adaptor của nhóm train xong trong 1 giờ |
| Bằng chứng DFlash attention | Không phải khảo sát DFlash trong PDF | D1–D4 là bằng chứng nội bộ mới về allocation; chưa phải chứng minh hiệu quả module |

Insight tương đồng là **giảm lượng memory draft đọc mà giữ khả năng conditioning trên lịch sử xa**, thay cho giảm năng lực backbone. MASW học **đại diện mới**; nó không dựa vào giả thuyết rằng một tập nhỏ raw token chứa hầu hết attention của DFlash. Vì vậy tail mass trong D1 không mâu thuẫn với động cơ learned compression, đồng thời không chứng minh một adapter sẽ xử lý tail thành công.

Một hướng phân biệt cần kiểm chứng là memory/selection thích ứng với trạng thái của block diffusion, dùng target-derived features và đánh giá nhu cầu đa layer. Incremental memory, learned slots, frozen backbone và full target verification đã có tiền lệ trong PDF; chỉ chuyển một trong những ý này sang DFlash chưa đủ để chốt đóng góp. Tài liệu này không xác nhận novelty so với toàn bộ literature; cần survey riêng khi khóa kiến trúc và claim paper.

## 8. Hàm ý cho các phương pháp sẽ phát triển

### 8.1. Các điều kiện mặc định để cô lập tác động của memory

| Thành phần | Điều kiện cho pilot đề xuất |
|---|---|
| Backbone | Giữ cùng pretrained DFlash-5L; ghi checkpoint chính xác |
| Live block | Giữ đủ 16 queries/keys theo semantics gốc; không tính block vào budget nén cache |
| Context nén | Prompt + output đã commit trong cache phục vụ drafter, tạo từ target features |
| Verifier | Full target KV/context, cùng verification/rollback/backend |
| Selector online | Chỉ sử dụng thông tin có trước lượt draft; current DFlash attention là oracle/nhãn offline |
| Vị trí | Raw selected key giữ absolute position/RoPE; learned slot cần scheme riêng và ablation |
| So sánh | Cùng model/template/dtype/output budget/hardware; baseline depth và block size giữ cố định |
| Báo cáo | Coverage tổng + cache, acceptance/commit thực tế, latency từng phần, draft KV và tổng peak VRAM |

Config warm-start MR-2S hiện tại thay cả memory lẫn depth từ DFlash-5L xuống hai stage; converter chỉ chuyển các source layer được chọn. Nó chưa cô lập “chỉ học memory trên backbone giữ nguyên”. Cũng chưa có chế độ freeze backbone riêng trong config đó. Không gọi warm-start hiện tại là adapter-only đã được kiểm chứng.

### 8.2. Nhánh phương pháp và bằng chứng cần bổ sung

| Nhánh đề xuất | Động cơ từ khảo sát | Điều cần chứng minh |
|---|---|---|
| Selection raw key không train | Top-K rải rác hơn cửa sổ; locality ngắn hạn | Selector online giữ acceptance sau gather và có cost/token tốt hơn dense/recency |
| Learned selector nhỏ | Tập ưu tiên thay đổi theo trạng thái, layer và document | Học từ thông tin có sẵn trước draft; hơn heuristic ở cùng budget/cost; generalize holdout |
| Learned memory tổng hợp | Raw Top-K nhỏ còn tail mass lớn | Đại diện mới giữ thông tin cần cho logits/acceptance với ít slot, tính cả build/update cost |
| Hybrid chi tiết + memory | Nhóm tập trung và phần rộng cùng tồn tại | Hơn selection-only và synthesis-only dưới cùng budget và chi phí thích nghi |
| Layer-specific allocation | Shared set coverage lệch giữa năm layer | Gain đủ bù routing/metadata/kernel complexity; không chỉ làm coverage average đẹp hơn |
| Temporal reuse/update | Adjacent overlap cao hơn các mốc xa | Append output và refresh giảm overhead mà kiểm soát degradation/acceptance tails |

Target attention chưa được chọn làm tín hiệu mặc định hoặc label duy nhất. Có thể đưa parent target attention vào một ablation feature sau này, nhưng tính hữu ích của ánh xạ học cần được chứng minh riêng. Target hidden features và target attention rankings là hai nguồn khác nhau; kết quả D4 không bác bỏ dùng hidden features vốn đã là conditioning của DFlash.

Loss tiềm năng gồm target-token prediction, distillation dense draft logits hoặc tái tạo attention output từng layer; attention coverage có thể là metric/phụ trợ. Chưa khóa loss/architecture từ study quan sát. Freeze tham số backbone giảm optimizer/gradient tham số, nhưng có thể vẫn cần backward qua backbone để học module trước nó; không suy thời gian train tỷ lệ với số tham số trainable.

### 8.3. Cập nhật hướng ưu tiên: cơ chế inference trên DFlash đã train

Người nghiên cứu đã làm rõ mục tiêu là phát triển **cơ chế tận dụng trained DFlash cho long context**. Đề xuất scorer học riêng được giữ tại [tài liệu lịch sử](2026-10-06_dflash_learned_context_proposal.md); hướng ưu tiên mới được trình bày trong [cơ chế điều phối context](2026-10-06_dflash_context_execution_mechanism.md).

V1 được đề xuất: lượt đầu DFlash đọc toàn context và tạo ranking cache bằng attention của chính nó; những lượt gần nhau dùng key được ưu tiên rải rác cùng một nhóm bảo vệ nhỏ ở đầu/gần block; thêm output thực sự commit và đọc lại toàn context để refresh. Có thể dùng phản hồi số token commit để điều chỉnh budget hoặc refresh ở lượt sau. Block 16 luôn giữ nguyên, target verifier vẫn full context. Tập chung và ranking riêng layer cần so ở cùng budget mỗi layer.

Hai giới hạn cần giữ trong thiết kế: attention trên cache đã cắt không nhìn thấy key bị loại, nên không coi key không quan sát là zero hoặc tự suy toàn context từ sparse attention; pruning cũng làm softmax chuẩn hóa lại, nên giữ block không bảo đảm block tiếp tục nhận đúng 20% mass. Lịch refresh, budget, lợi ích layer-specific routing và mọi speedup/acceptance sau can thiệp đều **chưa được đo**.

Nhánh dùng layer đầu DFlash đọc toàn context để chọn key cho các layer sau cũng được ghi trong đề xuất. Chưa có kết quả cho thấy ranking layer đầu đại diện đủ cho bốn layer còn lại. Những nội dung này là hàm ý và kế hoạch, không thêm observation vào D0–D4.

## 9. Khoảng trống bằng chứng và thực nghiệm kế tiếp

Các mục trong phần này là **kế hoạch**, chưa thực hiện trong lần tổng hợp.

### 9.1. Can thiệp cache nhỏ để nối coverage với chất lượng drafting

Bắt đầu tại 16K trên cùng 10 document, với các prefix/state đã lưu và budget cache **1K/4K/8K + block 16**. Giữ cùng anchor, output prefix và target features để cô lập tác động cache. So dense với current-DFlash-attention oracle, previous-DFlash selection, previous-parent-target selection theo layer, recency và random. Random là đối chứng phải chạy; các bảng coverage hiện có chưa phải kết quả can thiệp random.

Current-DFlash Top-K phải lấy từ forward dense, nên nhánh này là oracle chất lượng/upper bound coverage và không được đo như một giải pháp tăng tốc triển khai. First pass đo prediction/logit agreement cùng prefix, rồi đo accepted prefix với target verification full context. Tiếp theo mới kiểm tra free-running rollout: các policy có thể tạo draft/acceptance khác, nên trajectory không còn cố định như phép đo coverage.

Primary outcome nên là **số token thực tế commit mỗi round** cùng chi phí/token; báo thêm accepted proposals và phân bố theo vị trí proposal. Raw `accepted_proposals` ở terminal round có thể đếm qua EOS, nên cần cắt theo stop semantics. Với greedy, kiểm tra output cuối so target full-context greedy dưới cùng cấu hình. Các replay 40/40 hiện có so với **run DFlash nguồn**, chưa thay thế kiểm tra exactness của pipeline nén mới với autoregressive target.

### 9.2. Chỉ mở pilot học sau khi biết oracle chất lượng còn bao nhiêu headroom

Nếu oracle cache tại budget nhỏ giữ acceptance đủ tốt, so heuristic với một learned selector nhỏ. Nếu selection-only có giới hạn chất lượng, thử learned memory tổng hợp/hybrid với cùng backbone năm layer. Khởi đầu bằng pilot ngắn, ghi rõ thời gian, GPU-giờ, lượng supervision và chi phí checkpoint nền; giá trị cụ thể cần calibration, không dùng ước lượng 1 giờ như cam kết.

Split theo document để tách train/calibration/holdout. Dữ liệu attention của mười document hiện tại là exploratory/dev data, không dùng lại chúng làm untouched holdout cho module đã thiết kế từ các insight này. Mở rộng GovReport và thêm Multi-News/QMSum sau khi pilot có tín hiệu, rồi mới đánh giá transfer model/context.

Đối chứng tối thiểu cho module học: dense DFlash, selection không train, learned module, fine-tune DFlash thường với cùng data/compute adaptation, và các ablation cần để cô lập synthesis/selection/reuse. Nếu giảm depth trong một variant, cần baseline cùng depth để tránh quy mọi gain cho memory.

### 9.3. Benchmark physical execution và tính cả chi phí dựng memory

Sau khi có quality signal, giảm thật lượng K/V đọc hoặc dùng kernel thực thi phù hợp; dense mask không chứng minh savings hệ thống. Giữ cùng target verifier có KV cache, rollback và scheduler. Reference MR inference full-prefix `use_cache=False` có thể phục vụ correctness, nhưng chưa là đối chứng latency công bằng với production DFlash.

```text
T_e2e = T_prefill + Σ_round (T_draft + T_verify + T_select/build/update + T_other)
tau   = mean token thật sự commit mỗi round, xử lý EOS đúng
cost/token = mean chi phí các round / tau, với cùng cohort và cách weighting
```

Báo TTFT/prefill, decoding/TPOT, E2E, accepted/commit distribution, memory build/update/gather cost, draft KV bytes và tổng peak VRAM. Giảm draft KV không bằng giảm cùng tỷ lệ tổng VRAM vì target giữ full KV. Nếu phần được tối ưu chỉ chiếm tỷ lệ f của thời gian, trần Amdahl khi xóa toàn phần là `1/(1-f)`; f cần được profile, chưa đo trong collector hiện tại.

Khóa decision gate sau pilot và trước holdout, chọn theo mục tiêu speed hoặc memory. Dùng paired comparison theo document, confidence interval và nhiều seed phù hợp; báo ROUGE khi có reference. Chưa có speedup, VRAM reduction hoặc ROUGE sau nén để điền vào paper lúc này.

### 9.4. Nếu tiếp tục kiểm tra giả thuyết literal bonus query

Cần thu các round **k=15 và thực sự tiếp tục sinh** rồi ghép query `q=15` với DFlash round sau; báo correction và bonus subset riêng. Có thể chọn thêm task/instance có continuation đủ dài và điều kiện full-accept thường xảy ra, nhưng vẫn dừng đúng EOS. Không ép decode qua EOS để tạo bonus sample. Parent từ prefill cho lượt draft đầu là một phép ghép bổ sung khác, chưa có trong D4.

## 10. Những claim có thể dùng và những claim chưa đủ bằng chứng

| Nội dung | Cách viết được số liệu hiện tại hỗ trợ | Phát biểu cần tránh |
|---|---|---|
| Spatial coverage | Trên cohort này, scattered Top-K bao phủ nhiều dense attention mass hơn best contiguous window cùng budget | Một cửa sổ không bao giờ đủ cho chất lượng drafting |
| Sparsity | Có nhóm key tập trung mạnh và phần mass trải rộng; fixed small raw-key budget chưa bao phủ hết | DFlash chỉ dùng 1K context, hoặc cần mọi token để dự đoán |
| Length trend | Độ tập trung tương đối giảm khi cap tăng trong các prefix được khảo sát | Context length là nguyên nhân duy nhất; attention trở thành uniform |
| Live block | Block nhận mean khoảng 21–23% full mass; phương pháp dự kiến giữ đủ 16 vị trí | 16 token mang đúng 20% thông tin hữu ích ở mọi layer/round |
| Output | Output đã commit nhận mass đáng kể và cần thuộc scope cache | Chỉ chọn source prompt đã đại diện cho toàn bộ context |
| Temporal | Adjacent support ổn định hơn long-lag support; reuse đáng khảo sát | Có thể reuse cố định 8 round mà không giảm acceptance |
| Layer | Coverage của shared set khác nhau theo draft layer | Shared target features làm năm attention distributions tương đương |
| Target attention | Direct ranking chưa có quan hệ đủ ổn định để chốt selector độc lập | Target attention hoàn toàn vô ích, hoặc đã chứng minh predictor hữu hiệu |
| Bonus token | Không có cặp literal bonus-to-next-draft trong sample; câu hỏi còn mở | Attention bonus đã được so sánh ở 4.326 transition và bị bác bỏ |
| Learned memory | Các quan sát tạo động cơ cho selection/synthesis thích ứng | Module chưa train đã giữ chất lượng, hoặc tail mass được nén losslessly |
| Runtime | Thu attention và replay đã hoàn tất, artifact có kiểm tra | Eager collector timing chứng minh speedup production |
| Training cost | Có log baseline và chi phí người dùng báo; cần giảm adaptation budget | 1,5 epoch đủ >90% năng lực, hoặc adapter chắc chắn train xong trong 1 giờ |
| Novelty | Có khác biệt cần nghiên cứu giữa block DFlash và independent AR memory | “Nén draft KV + full target” tự nó là đóng góp mới |

Các số trên là descriptive result trong cohort, chưa có kiểm định population. Các report cũ dùng nhãn rung R7 cho việc hoàn tất khảo sát/replay; nhãn này không được dùng để gọi phương pháp nén đã được xác nhận hoặc benchmark paper đã hoàn tất.

## 11. Gợi ý lập luận cho phần motivation và empirical analysis của paper

Một đoạn motivation có thể phát triển từ hồ sơ hiện tại:

> Chúng tôi nghiên cứu giảm chi phí conditioning của DFlash trên context dài bằng cách rút gọn memory phục vụ drafter và giữ target verification trên toàn prefix. Khảo sát mười tài liệu GovReport ở bốn độ dài prompt cho thấy attention của drafter có nhóm key tập trung mạnh, đồng thời có phần mass trải rộng và tập key ưu tiên thay đổi trong quá trình sinh. Vì vậy lựa chọn context theo một cửa sổ cố định hoặc một ranking đơn giản cần được kiểm chứng trước khi xây memory nhỏ.

Một đoạn empirical analysis có thể dùng sau khi bổ sung trích dẫn hình/bảng:

> Trên prompt 16K, Top-1K và Top-4K mọi key bao phủ lần lượt 67,29% và 82,61% attention mass, cao hơn best contiguous window cùng budget nhưng vẫn bỏ ngoài một phần mass đáng kể. Kết quả tiếp tục hiện diện trên cache sau khi loại 16 vị trí live block. Tập Top-1K cache trùng 82,65% giữa hai draft round liền kề và 31,11% giữa mốc giữa–cuối, tạo động cơ khảo sát memory được cập nhật theo trạng thái sinh. Parent target Top-1K chỉ bao phủ 38,01% cache attention của DFlash lượt sau ở 16K; các parent transition quan sát được đều là correction query, nên chưa đủ bằng chứng cho selector dựa riêng vào target attention hoặc giả thuyết literal bonus query.

Đóng góp phương pháp, kết quả acceptance/speedup, tỷ lệ nén vật lý và tính tổng quát sẽ cần bổ sung sau triển khai. Những đoạn trên mô tả motivation/quan sát; chưa được dùng như abstract báo cáo một phương pháp đã đạt hiệu quả.

## 12. Giới hạn cần giữ khi đưa vào paper

1. Mười document thuộc một task, một sample seed và một cặp checkpoint upstream; chưa có holdout cho module hoặc transfer sang model/task khác.
2. Prompt cap và nội dung prefix cùng thay đổi. Output dài 111–435 token; chưa khảo sát generated history hàng nghìn token hoặc prompt 32K+.
3. Vector đã mean qua heads/proposal queries; specialization và tail theo từng head/query có thể bị che. Layer vectors còn lưu, đầy đủ head/query tensor không có cho mọi round.
4. Eager DFlash probabilities và target Q/K probabilities được thu trong instrumentation; replay được kiểm tra nhưng timing không là production benchmark.
5. Các phép cache-only là chuẩn hóa lại dense attention, không phải forward bỏ block hoặc forward cache nén. Queries/hidden states đã chịu ảnh hưởng của block trong forward gốc.
6. Top-K dense là oracle theo attention hiện tại; selector sử dụng nó cần dense pass hoặc predictor riêng. Chưa đo chi phí lấy target attention online.
7. Attention weight không đo trực tiếp evidence content, logit contribution hoặc causal necessity. Khi xóa/nén KV, softmax và hidden states các layer sau đổi.
8. D4 chưa có bonus-to-next sample. Terminal raw acceptance cần xử lý EOS trước khi dùng làm acceptance/commit metric.
9. D3/D4 so cùng reference trajectory; chưa là intervention, chưa có thống kê về target-attention selector learned mapping.
10. Chi phí training, baseline systems cũ và attention pilot hiện dùng model/runtime khác; không có cơ sở ghép chúng thành một claim paired speedup.

## 13. Nguồn số liệu, kiểm tra và tái lập

### 13.1. Run IDs và artifact chính

Các đường dẫn dưới đây tính từ repository root; thư mục `outputs/` bị gitignore. Paper cần lưu raw artifacts hoặc archive cùng manifest; tài liệu Markdown không thay thế các vector và trace.

| Mã | Run / nguồn | Summary và audit |
|---|---|---|
| D0 | `dflash-attention-govreport-l40s-20261006` | [summary](../../outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/summary.json), [manifest](../../outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/manifest.json) |
| D1 | `dflash-attention-fullmass-govreport10-l40s-20261006` | [summary](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/summary.json), [artifact audit](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/artifact_audit.json) |
| D2 | Cùng D1, `full_context_analysis/` | [summary](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/summary.json), [audit](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/analysis_audit.json) |
| D2-C | Cùng D1, `context_cache_analysis/` | [summary](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/summary.json), [audit](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/analysis_audit.json) |
| D3 | `dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b` | [summary](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/summary.json), [audit](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/analysis_audit.json) |
| D4 | `dflash-parent-attention-next-draft-govreport10-l40s-20261006b` | [summary](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/summary.json), [audit](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/analysis_audit.json) |

Audit của lần tổng hợp tài liệu: [document audit](../../outputs/research_documents/dflash_context_memory_analysis_20261006/document_audit.json), ghi kiểm tra bảng số liệu, nguồn/hash, link hình và phạm vi bằng chứng. Đây là kiểm tra tài liệu, không thêm inference hoặc test chất lượng phương pháp.

D1 audit đã đối chiếu **4.366 NPZ**, shape `(5, prefix_tokens + 16)`, finite/nonnegative, mean vector khớp JSONL; max sai số tổng mass mean là **4,26×10⁻⁵**. D2 tính lại Top-1K khớp record gốc trong **4,11×10⁻¹⁵**, kiểm tra oracle bounds/budget monotonicity. D2-C kiểm tra loại đúng 16 key, cache + block mass, công thức R và consistency theo layer.

D3 và D4 khớp token output cùng số draft round với D1 ở **40/40 trajectory**. Tổng mass riêng layer lệch tối đa **1,07×10⁻⁴**; target future-key mass bằng 0. D4 audit kiểm tra **4.326/4.326** parent-query/prefix/role/layer-pair alignments, giữ riêng số bonus transition bằng 0. Các code snapshots có trong artifact; collector hiện tại đã phát triển qua nhiều phase, nên dùng bản code của từng run khi cần tái lập chính xác.

Các run bị loại: attempt đổi backend target làm trajectory lệch nguồn; attempt parent reduction nhầm head/query axis dừng tại sanity check. Chỉ run thành công có suffix `20261006b` trong bảng được dùng cho D3/D4. Các sự cố này không được tính thành scientific observation hoặc sample đã nén thất bại.

### 13.2. Danh mục báo cáo chi tiết

- [D0: một instance](2026-10-06_dflash_attention_probe_results.md).
- [D1: mười instance, full mass](2026-10-06_dflash_attention_10_instances_full_mass.md).
- [D2/D2-C: toàn context và cache bỏ block](2026-10-06_dflash_full_context_compression_analysis.md).
- [D3: target cùng lượt](2026-10-06_dflash_target_attention_comparison.md).
- [D4: parent query lượt trước](2026-10-06_dflash_parent_attention_next_draft.md).
- [Protocol attention](2026-10-06_dflash_context_support_design.md) và [hướng phát triển MR](2026-10-06_mr_dflash_research_directions.md).
- [Training-Free E41/E42](2026-09-21_trainingfree_controlled_search.md), [E43](2026-09-23_trainingfree_e43_temporal_support_reuse.md), [E44](2026-09-23_fidelitykv_e44_results.md).
- [Cơ chế inference tận dụng DFlash đã train, hướng ưu tiên hiện tại](2026-10-06_dflash_context_execution_mechanism.md); [selector học riêng, đề xuất cũ được giữ để truy vết](2026-10-06_dflash_learned_context_proposal.md).

### 13.3. Lệnh xử lý artifact đã có

```bash
python3 scripts/analyze_dflash_full_context.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006
python3 scripts/analyze_dflash_context_cache.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006
python3 scripts/analyze_dflash_target_comparison.py \
  --input outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b
python3 scripts/analyze_dflash_target_comparison.py \
  --input outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b
python3 scripts/plot_dflash_parent_attention.py \
  --input outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b
```

Đây là các lệnh xử lý lại số liệu đã có; lần tổng hợp tài liệu không chạy lại inference/training. Manifest D1 giữ dataset SHA-256, source/sample IDs, snapshots và phiên bản runtime. D3/D4 còn ghi hash manifest/summary nguồn và exact replay checks.

### 13.4. Quy tắc bổ sung kết quả sau này

Mỗi thực nghiệm mới cần thêm câu hỏi/claim, scope cache/block/query, sample và split, model/code revision, method có dùng oracle hay không, metric/denominator, decision gate, kết quả và artifact. Khi có can thiệp, cập nhật coverage cùng logit/acceptance; khi có physical executor, thêm latency/memory với dense control. Giữ các negative result và evidence gap trong hồ sơ để tránh thay đổi câu chuyện theo kết quả cuối.
