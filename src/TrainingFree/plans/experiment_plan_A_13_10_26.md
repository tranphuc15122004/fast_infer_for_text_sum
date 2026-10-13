# Kế hoạch thực nghiệm A — Context selection cho DFlash

Ngày cập nhật: **2026-10-13**.

**Trạng thái ngày 13/10/2026:** harness A1–A3 đã được triển khai. B200
preflight/prepare thành công; lần validate đầu dừng do gate cũ đòi exact match
với AR. Trên 6 mẫu, native DFlash và full adapter khớp nhau 6/6, nhưng cả hai
khác AR. Protocol amendment bên dưới đổi gate A sang native/full-adapter
parity; A1/A2/A3 vẫn chưa chạy. B200 là môi trường chính; Modal là diagnostic
tùy chọn, với ngân sách còn 20 USD.

## Phần I — Câu hỏi và yêu cầu nghiên cứu

Tôi đề xuất 3 thực nghiệm theo thứ tự, nhằm trả lời một câu hỏi duy nhất:

> DFlash thực sự cần những phần context nào để duy trì hoặc tăng accepted length, và có thể loại bỏ bao nhiêu context mà không làm giảm hiệu quả drafting?

Trước mắt, cố định $\gamma=15$, tắt adaptive controller, chỉ nghiên cứu A.

## Thực nghiệm A1 — Context nào thực sự quan trọng?

Mục tiêu: Xác định vai trò của từng vùng context bằng cách can thiệp trực tiếp vào draft KV.

Tại cùng một decoding prefix, giữ nguyên target verification và so sánh:

| Cấu hình              | Context DFlash được sử dụng            |
| --------------------- | -------------------------------------- |
| Full                  | Toàn bộ context                        |
| Source-only           | Source + instructions/anchors bắt buộc |
| Recent-only           | Recent output + instructions/anchors   |
| Source + Recent       | Source và recent generated output      |
| Remove low-attention  | Full trừ các KV có attention thấp      |
| Remove high-attention | Full trừ các KV có attention cao       |

Với các cấu hình loại bỏ, cần kiểm soát số KV bị loại giống nhau để so sánh công bằng (thử với các lượng KV phù hợp 1024 , 2048, ...). Sau khi lược bớt các KV thì các phần còn lại cần giữ được vị trí tuyệt đối của chúng trong câu. không nên gộp bằng một cách khác như random, ...

Đo:

$$ \boxed{\Delta L_t(S)=L_t(S)-L_t(\mathrm{Full})} $$

Trong đó $L_t$ là số speculative candidate tokens được accept liên tiếp.

Các thực nghiệm remove context cần được thực hiện trên một số kịch bản sau : 
- Dựa trên attention của các layer [1,9,17,25,33] của target model , hoặc trung bình của chúng (6 trường hợp). Dùng cho toàn bộ cả 5 layer của Dflash.
-  Dựa trên attention của các layer [1,9,17,25,33] của target model dùng cho tương ứng từng layer của Dflash

Insight cần tìm: Acceptance phụ thuộc mạnh vào source, recent output, hay chỉ một tập nhỏ các KV? Việc bỏ high-attention KV có thực sự gây hại hơn bỏ low-attention KV không?

## Thực nghiệm A2 — Attention của mô hình nào chọn context tốt nhất?

Mục tiêu: Kiểm chứng liệu attention có thực sự dự đoán được context hữu ích.

Giữ cùng budget $B$, protected tokens và $\gamma=15$.

| Selector                  | Ý nghĩa                                        |
| ------------------------- | ---------------------------------------------- |
| Random                    | Baseline để đo lợi ích thực sự của selection   |
| Recency                   | Ưu tiên KV gần nhất                            |
| Target-parent attention   | Phương pháp A hiện tại                         |
| Previous DFlash attention | Tái sử dụng attention từ draft rounds trước    |
| Current DFlash attention  | Oracle chẩn đoán, không dùng làm online policy |

Đo $\Delta L_t(S)$ và tỷ lệ rounds mà reduced context giữ nguyên hoặc tăng acceptance.

Ở báo cáo trước, target-parent Top-1K chỉ giữ khoảng 38% DFlash cache-attention mass tại 16K. Vì vậy, tôi đặc biệt muốn kiểm chứng previous DFlash attention có tốt hơn target attention hay không.

Nếu previous DFlash attention tốt hơn random, recency và target-parent, ta có cơ sở chọn hướng draft-centric context selection.

## Thực nghiệm A3 — Có thể giảm context đến mức nào?

Mục tiêu: Tìm quan hệ giữa context budget, accepted length và latency.

Với selector tốt nhất từ A2, thử:

$$ B\in\{512,1024,2048,4096,8192,\mathrm{Full}\} $$

Đo đồng thời:

$$ L(B),\qquad T_{\mathrm{draft}}(B),\qquad J(B)=\frac{T_{\mathrm{round}}(B)}{1+L(B)} $$

Quan trọng là phân biệt hai trường hợp:

- Acceptance improvement: giảm context nhưng $L(B)>L(\mathrm{Full})$.
- Acceptance preservation: giảm context, $L(B)\approx L(\mathrm{Full})$, nhưng latency thấp hơn.

Trường hợp thứ hai cũng hoàn toàn đủ để chứng minh giá trị của A.

Sau đó mới kiểm tra context ranking có thể được tái sử dụng trong bao nhiêu rounds trước khi cần refresh, thay vì mặc định refresh mỗi 4 rounds như hiện tại.

## Protocol chung cho cả ba thực nghiệm

Để kết quả có ý nghĩa nhân quả:

1. Same-prefix paired replay: Mỗi selector/budget được chạy trên cùng target prefix, cùng pending anchor và cùng $\gamma$. Không dùng các trajectories khác nhau để so accepted length trực tiếp.
2. Target luôn dùng full KV: Chỉ thay đổi draft-context selection, không thay đổi verifier.
3. So cùng số KV thực tế: Giữ các protected tokens nhất quán, tránh so các phương pháp chỉ dựa vào nominal budget.
4. Đánh giá hai cấp: Trước tiên đo acceptance tại fixed prefixes; khi có selector tốt mới chạy full generation để đo E2E latency, bao gồm selection/gather/refresh overhead.

Nên bắt đầu với khoảng 20–30 documents thuộc GovReport, Multi-News và QMSum, lấy các prefix ở đầu/giữa/cuối quá trình generation. Sau giai đoạn khám phá, xác nhận trên tập document độc lập và báo confidence intervals theo document.

Điều kiện tiên quyết cho A là full adapter tái tạo native full DFlash trên cùng prompt/runtime. Output DFlash có thể khác target-only AR; ghi nhận first-divergence như diagnostic nhưng không chặn nghiên cứu A. Kết luận A nói về giảm context so với native full DFlash, không gọi đó là lossless speedup so với AR.

## Kết quả cuối cùng chúng ta cần

Ba kết luận cần chứng minh

A1: Vùng context nào có ảnh hưởng trực tiếp tới acceptance?

A2: Có tín hiệu rẻ, sẵn có từ các rounds trước để xác định vùng đó không?

A3: Sử dụng tín hiệu ấy, giảm context đến mức nào thì có lợi về latency mà không làm giảm acceptance đáng kể?

Tôi ưu tiên A1 → A2 → A3, không triển khai dynamic budget ngay.

Nếu A2 không tìm được selector tốt hơn các baseline đơn giản thì không nên phát triển thêm adaptive context controller. Ngược lại, nếu kết quả xác nhận rằng DFlash chỉ cần một working set nhỏ, thay đổi theo generation state, đó sẽ là insight vững chắc để thiết kế lại A thành một phương pháp đơn giản và có cơ sở thực nghiệm.

---

## Phần II — Protocol cụ thể và kế hoạch thực hiện

### 1. Phương án được chọn và phạm vi kết luận

Chọn **toàn bộ quy trình trên B200, bắt đầu bằng pilot nhỏ có kiểm soát**.
Lợi thế là dùng một runtime cho pilot và xác nhận, tái sử dụng một bộ replay
states và attention scores cho nhiều can thiệp, đồng thời dành số document
độc lập lớn hơn cho bước xác nhận.

| Phương án | Chi phí và hạn chế | Quyết định |
|---|---|---|
| B200 pilot → B200 khám phá/xác nhận | Dùng tài nguyên server sẵn có; tránh staging và chuyển runtime Modal | Chọn |
| Modal nhỏ → B200 khám phá/xác nhận | Tùy chọn khi cần diagnostic ngoài server; vẫn cần staging riêng | Không bắt buộc |
| Toàn bộ ma trận và heldout trên Modal | Chưa có parity; dễ tiêu hết credit trước bước xác nhận | Không chọn cho ngân sách hiện tại |
| Chỉ so attention coverage | Rẻ nhưng chưa trả lời ảnh hưởng nhân quả lên acceptance hoặc latency | Chỉ dùng diagnostic bổ sung |

Pilot nhỏ trả lời liệu harness đúng, phép can thiệp có tạo tín hiệu và các
cấu hình có triển vọng. Sáu documents không đủ để khẳng định tổng quát cho ba datasets.
Kết luận xác nhận A1/A2/A3 cần cohort độc lập trên B200; kết quả âm hoặc CI rộng
được báo đúng là không hỗ trợ giả thuyết hoặc chưa đủ bằng chứng.

Campaign này có gamma cố định, budget cố định trong mỗi cell và trọng số frozen.
Training loss, backward, tiny overfit và Validation Pyramid dành cho training
không áp dụng. Kiểm chứng inference gồm verifier, cache, positions, attention
collector, stopping, paired replay và phép đo GPU.

### 2. Bằng chứng nền và tài nguyên hiện có

- [Thiết kế hiện tại](../docs/context_adaptive_dflash/design.md) đã có full
  draft bank, gather theo positions và selector theo source chunks.
- [Báo cáo attention 06/10](../../../docs/experiments/2026-10-06_dflash_parent_attention_next_draft.md)
  ghi parent Top-1K giữ 38,0% DFlash cache-attention mass tại 16K. Đây là phép
  đo phân phối attention, chưa là bằng chứng về acceptance sau loại KV.
- [Pilot GPU 09/10](../docs/context_adaptive_dflash/modal_fa4_h100_results_2026-10-09.md)
  có mismatch giữa full DFlash và AR, và toàn bộ outputs chạm cap. Với protocol
  A đã sửa, AR divergence là diagnostic; cần bảo đảm native/full-adapter parity
  và báo rõ dừng EOS hay chạm output cap trong các phase.
- B200 validate ngày 13/10, run `dflash_context_A_b200_20261013T163438Z`:
  native DFlash = full adapter trên 6/6 mẫu; cả hai khác AR, first mismatch
  lần lượt tại token 103, 544, 89, 44, 69 và 96. Mỗi output dài đúng cap
  1024; các rollout phải phân biệt dừng EOS với dừng do cap.
- Corpus đã chuẩn bị nằm tại
  outputs/context_adaptive_dflash/m0_local_primary_cpu_20261009/inputs/primary/.
  Ba files GovReport, Multi-News, QMSum có 300 records nhưng 220 source groups.
- Split hiện có phân bổ source groups:

| Split | GovReport | Multi-News | QMSum |
|---|---:|---:|---:|
| Calibration | 20 | 20 | 4 |
| Dev | 20 | 20 | 4 |
| Test | 60 | 60 | 12 |

Một meeting QMSum có nhiều query vẫn là một đơn vị độc lập. Không yêu cầu
8–20 meetings QMSum trong dev khi pool chỉ có 4.

CLI Modal local có phiên bản 1.5.5, active profile phuclk30. Kiểm tra chỉ đọc
environment mặc định ngày 13/10 trả danh sách volumes rỗng. Các volumes trong
pilot cũ chưa được xác nhận có thể dùng trong environment này. Cần chuẩn bị
snapshots/models/data riêng trước job; đăng nhập thành công chưa xác nhận assets,
credit còn lại hoặc quyền GPU. Không lưu token đăng nhập trong artifacts.

### 3. Ba thẻ thực nghiệm và quy tắc quyết định

| Thực nghiệm | Giả thuyết có thể bị bác bỏ | Chỉ số chính | Quy tắc kết luận đề xuất |
|---|---|---|---|
| A1 attention | Bỏ high-attention KV gây hại hơn bỏ low-attention KV ở cùng số KV | Mean theo document của L(remove-low) − L(remove-high) | CI 95% nằm trên 0; effect ≥0,25 candidate/round để gọi là có ý nghĩa thực tiễn |
| A2 selector | Previous DFlash attention chọn context tốt hơn Random, Recency và Target-parent | Chênh lệch L(previous) − L(baseline), cùng B | Xác nhận tốt hơn cả ba khi từng contrast có CI điều chỉnh trên 0 và effect ≥0,25 |
| A3 hiệu năng | Reduced context giữ acceptance và giảm E2E latency | Paired geometric macro E2E speedup so full DFlash | Point estimate ≥1,05; CI 95% lower bound >1; non-inferiority acceptance đạt |

A1 về vùng context dùng các contrasts đã đăng ký tại mục 7. Không dùng contrast
attention làm tiêu chí duy nhất cho câu hỏi source/recent.

Acceptance preservation được định nghĩa trước bằng **non-inferiority margin
0,25 candidate/round**: lower bound CI của mean ΔL lớn hơn −0,25. Việc khác
biệt không có ý nghĩa thống kê chưa đủ để kết luận preservation. Acceptance
improvement yêu cầu CI của ΔL nằm trên 0; báo thêm liệu effect đạt 0,25 hay không.

Ngưỡng 0,25 và speedup 1,05 là mức thực tiễn đề xuất, không được chọn lại sau
khi xem heldout. Các guardrails bắt buộc: native full DFlash và full adapter
phải khớp output IDs; AR mismatch được báo cáo nhưng không chặn. A3 báo paired
ROUGE deltas của reduced so với native full DFlash và acceptance non-inferiority.
Do chưa đăng ký biên non-inferiority cho ROUGE, chỉ mô tả điểm số và không tuyên
bố chất lượng tóm tắt được bảo toàn dựa trên ROUGE. Các guardrails khác gồm
target full KV, protected tokens, số KV thực, vị trí gốc, không rò thông tin
tương lai và không bỏ error/missing rows để tạo gain.

Không đòi A1 hoặc A2 phải có gain để coi nghiên cứu hoàn tất. Kết quả âm hợp lệ
cũng trả lời câu hỏi. Nếu previous attention không vượt selector đơn giản,
dừng nhánh controller mới; vẫn có thể đo A3 cho selector đơn giản nếu nó giữ
acceptance và có triển vọng giảm cost.

### 4. Cấu hình chung phải khóa

| Thành phần | Giá trị chính |
|---|---|
| Target / drafter | Qwen3-4B / Qwen3-4B-DFlash-b16, snapshot SHA-256 cố định |
| Inference | BF16, eval, inference mode, batch size 1, temperature 0 |
| Draft shape | gamma = 15, block size = 16 gồm một pending anchor |
| Target | Full prompt + toàn processed output; cùng verifier cho mọi can thiệp |
| Prompt | Cùng tokenizer/template, no thinking; source span map trên prompt thực |
| Selection chính | Theo từng token-position; cùng số keys trên từng draft layer |
| Seed | Split/cohort = 42; Random selector = 17, 42, 123 |
| Recent window | 256 processed output tokens; không tính pending anchor |
| Source anchors | 64 source tokens đầu, hoặc toàn source nếu ngắn hơn |
| Natural output | EOS bật; cap pilot 1024, campaign B200 2048 |
| Boundary | Những round không còn đủ 15 candidates dùng xử lý boundary chung |

Phiên bản Python/Torch/Transformers/FA4 và GPU được ghi thành runtime signature.
Dùng profile đã pin tương thích requirements.txt và đường Qwen/DFlash trong
runner H100 hiện có; mọi thay đổi pin/backend là một runtime mới và phải qua
parity lại. Không tự sửa checkpoint config hoặc hidden-feature routing để
phù hợp layer attention đang khảo sát.

B là số **context positions trên mỗi layer**, bao gồm protected positions và
không bao gồm 16 vị trí block hiện tại. D là số positions bị loại trên mỗi
layer. Báo cả vector counts 5 layers, tổng K/V bytes được gather và logical
context length. Hai mask khác nhau cùng B vẫn là cùng tài nguyên theo layer.

Selection luôn sorted, unique, trong [0,n_t). Gather K/V đã có RoPE tại vị trí
gốc; current anchor/masks dùng positions n_t đến n_t+15. Không renumber context,
không thay RoPE, không đổi target cache theo sparse physical length.

**Bảo vệ theo từng câu hỏi:**

- G0 = instruction/query/template/boundary positions + 64 source anchors.
  Pending anchor được bảo vệ riêng trong current block.
- A1 dùng G0 để có thể can thiệp vào recent output. Bảo vệ recent cố định trong
  A1 sẽ làm Source-only không còn đúng ý nghĩa và che khuất tác động recent.
- A2/A3 dùng G = G0 ∪ last 256 processed output positions, nhất quán trong
  mọi selector của cùng phase.
- Khi B < |G|, cell là infeasible; không nới B âm thầm hoặc bỏ instructions.
  Khi B ≥ n_t, canonicalize thành Full và không đếm thêm một observation.

Random trong A2 chỉ chọn tập vị trí cần giữ. Tập được chọn vẫn giữ thứ tự và
vị trí gốc khi đưa vào drafter.

### 5. Cohort, prefix và chống leakage

Giữ split hiện có, kèm exposure audit của probe 06/10 và hai pilots 09/10.
So source fingerprints và provenance của các pilot cũ với cohort mới trước
lock; group từng được xem phải thuộc khám phá. Nếu audit làm giảm pool test,
báo quota thực tế và bổ sung groups chưa xem hoặc thu hẹp kết luận; không
chuyển exposed group thành heldout bằng cách đổi sample ID.

| Giai đoạn | GovReport | Multi-News | QMSum | Tổng source groups |
|---|---:|---:|---:|---:|
| Pilot B200, subset dev; Modal tùy chọn | 2 | 2 | 2 | 6 |
| B200 khám phá, gồm pilot | 10 | 10 | 4 | 24 |
| B200 xác nhận, từ test chưa xem | 18 | 18 | 12 | 48 |

Mỗi group dùng một query/record chọn theo hash với seed 42. Quota ưu tiên có
documents khoảng 4–8K và 12–16K prompt tokens nếu dataset có đủ; counts được
tính lại bằng target tokenizer. Không kéo dài Multi-News hoặc nhân bản QMSum
để tạo length bins cân bằng giả. Độ dài là stratum mô tả, không nhân số mẫu.
Length-bin files có code-completion; chỉ dùng ba summarization datasets.

Thu một reference trajectory full DFlash đã khớp AR cho mỗi document. Lưu
prefix IDs và metadata đủ rebuild state; không ghi mọi full KV tensors theo
round ra đĩa. Một document được load/rebuild một lần cho các replay tại nó.

Chọn tối đa **6 fixed prefixes/document**: prefill, round thứ hai nếu có,
và các round gần 25%, 50%, 75%, 90% tiến độ generated output. Điểm trùng được
loại; không tạo prefix sau EOS. Chỉ lấy regular rounds có đủ remaining budget
và ít nhất 16 reference output tokens trước EOS/cap cho phép đo chính.
Đánh giá EOS/cap ở correctness cases riêng và hạch toán đủ trong E2E.

Đánh dấu capped trajectory; “cuối” trên trajectory capped là gần cuối phần
đã quan sát, chưa phải gần kết thúc tự nhiên. Nếu reference ngắn, báo số
prefix khả dụng. Prefill không có previous draft attention được báo riêng;
không trộn fallback Full vào hiệu quả riêng của previous-attention selector.

Snapshot/replay state gồm prefix hash, n_t, pending anchor ID/position, full
target KV hoặc recipe rebuild, full draft bank, parent Q origin/row, previous
dense draft scores và origin round. Sau mỗi action phải khôi phục state;
không để commit/rejection hoặc scratch của action trước ảnh hưởng action sau.
Thời gian clone/rebuild phục vụ nghiên cứu nằm ngoài round latency, được log
riêng; prefill và bank construction thật vẫn nằm trong request E2E.

### 6. Gate 0 — Correctness trước ma trận A

Thứ tự kiểm chứng từ rẻ tới đắt:

| Bước | Kiểm gì | Bằng chứng để đi tiếp |
|---|---|---|
| R0 | Protocol, cohort quotas, metrics, runtime và ngân sách đã khóa | Manifest kế hoạch và hashes |
| R1 | Local CPU: config, source mapping, protected sets, exact cardinality, sorted positions, resume keys | Báo cáo logic và schema; chưa chứng minh GPU correctness |
| R2 | Checkpoint thật: một prefill/draft/verify, gamma 15, collector bật/tắt | Shapes hữu hạn, bank/target positions đúng, output không đổi vì instrumentation |
| R3 | So AR ↔ native DFlash ↔ full adapter; sparse masks và reject/commit/EOS | Parity report, first-divergence nếu fail |
| R4 | Launcher B200 thật trên cohort 6 groups, artifact/resume đầy đủ | Đường chạy và kết quả hợp lệ; không coi là research gain |
| R5 | A1/A2/A3 pilot | Ma trận nhỏ đủ coverage; chọn việc cần mở rộng |
| R6 | B200 dev rồi heldout sau lock | Đủ planned cells và cohort độc lập |
| R7 | Phân tích, CI và trả lời từng câu hỏi | Decision memo và bảng coverage/gates |

R2/R3 ở đây là các bước dành cho inference, không có backward hoặc tiny
overfit. Không dùng smoke exit 0 thay thế verifier parity.

R3 phải có:

1. So target AR next-token logits với verifier tại **cùng input prefix,
   absolute positions và full cache**. Chỉ rows trên accepted prefix và first
   reject row có prefix tương ứng để so trực tiếp; rejected suffix phải được
   rebuild đúng prefix nếu muốn chẩn đoán riêng.
2. So native drafter với full bank adapter: K/V sau RoPE, draft logits,
   candidate IDs, accepted length và output IDs. Layer IDs target là
   module indices 0-based; hidden_states có offset embedding +1.
3. Force reject đầu, reject giữa, all-accept, multi-round crop/append, EOS
   ở anchor/accepted prefix và cap còn dưới 16 tokens. Không dừng vì EOS trong
   rejected suffix. Kiểm riêng bonus/correction và parent row L.
4. So sparse reference với gathered K/V giữ positions gốc; target vẫn full
   và không bị action làm thay đổi cache.
5. Bật/tắt collector không thay backend hoặc output. Attention future-key
   mass bằng 0; normalization, GQA, scale và sliding mask đúng.

Full adapter yêu cầu **100% trùng native full DFlash** trên cohort validation.
Greedy output mismatch với AR được lưu cùng vị trí/token divergence nhưng không
chặn A; pilot hiện tại cho thấy native DFlash và full adapter cùng khác AR.
So sánh target verifier logits tại cùng prefix/absolute positions vẫn dùng để
chẩn đoán correctness số học; tolerance BF16 đề xuất atol=1e-2, rtol=1e-2,
lưu max errors và top-1/top-2 margin.

Nếu native full DFlash và full adapter không khớp: dừng A1–A3 và ghi diagnostic
report. AR mismatch đơn lẻ không làm parity gate fail. Nếu dùng Modal, chỉ dành
tối đa 45 phút GPU cho gate này trong lượt pilot; không tiêu hết 20 USD để đuổi lỗi.
Điều tra tiếp trên CPU hoặc B200 bằng harness sau khi được triển khai.
Pilot cũ cần đánh giá theo gate native-DFlash/full-adapter; mismatch với AR được lưu riêng.

### 7. A1 — Vùng context và độ quan trọng của attention

#### 7.1. Can thiệp theo vùng

Gọi S là source positions, Y là toàn processed output và R là last 256
positions của Y. Mọi tập đều union G0 và giữ nguyên current block.

| Cấu hình | Tập context giữ lại |
|---|---|
| Full | G0 ∪ S ∪ Y |
| Source-only | G0 ∪ S |
| Recent-only | G0 ∪ R |
| Source + Recent | G0 ∪ S ∪ R |

Contrasts đăng ký:

- L(Source + Recent) − L(Source-only): tác động bổ sung recent output khi
  source đã có.
- L(Source + Recent) − L(Recent-only): tác động bổ sung source khi recent có.
- L(Full) − L(Source + Recent): tác động output cũ ngoài recent window.
- ΔL của từng reduced context so Full và tỷ lệ ΔL ≥ 0.

Ba contrasts đầu có thể khác số KV vì đúng bản chất thêm/bỏ một vùng.
Để trả lời **vai trò vùng ở cùng lượng KV**, bổ sung matched-deletion:
loại D_region ∈ {64,128} positions khỏi source, recent output hoặc old output,
giữ mọi vùng khác nguyên. Chọn positions trong từng vùng bằng hash/seed 42;
cùng quy tắc và D_region. Chỉ so trên giao các states mà mọi vùng so sánh
đều có đủ D_region keys ngoài G0. Báo effects theo vùng và contrasts giữa vùng.

Primary contrast vùng là source-vs-recent tại D_region=128. Old-output contrast
chỉ kết luận trên late states đủ old-output keys. Pilot thực hiện D_region=128;
B200 bổ sung 64. Nếu không đủ eligible states, báo unavailable thay vì lấp
bằng keys vùng khác. Bốn cấu hình theo vùng và matched-deletion phải có nhãn
phân tích riêng để người đọc biết ảnh hưởng của cardinality.

Contrast vùng có thể dương hoặc âm: CI loại 0 cho bằng chứng có tác động,
và |effect| ≥0,25 candidate/round cho tác động thực tiễn. Với matched deletion,
ΔL(delete-source) < ΔL(delete-recent) nghĩa là mất cùng số source keys gây
hại hơn mất recent keys. Khi R=Y hoặc Y rỗng, một số controls trùng nhau;
canonicalize mask nhưng giữ labels/eligibility, không tạo thêm mẫu độc lập.

#### 7.2. Loại low/high attention với số KV bằng nhau

Thu target-parent attention tại module indices [1,9,17,25,33], kiểm checkpoint
có đúng depth/layer mapping. Năm vector được lấy tại cùng parent query:
prefill dùng last prompt query; sau verify dùng row L tạo pending anchor.
Không dùng last verifier row khi có rejection.

Mean qua heads sau FP32 softmax trên full allowed key scope. Chọn được cả
source và processed output; không giới hạn collector vào source chunks.
Raw normalized attention mỗi target layer dùng để tạo shared mean.

| ID scoring | Tập chọn trên 5 draft layers |
|---|---|
| target_1, target_9, target_17, target_25, target_33 | Một vector target tương ứng, dùng chung cho cả 5 draft layers |
| target_mean5 | Mean đều 5 vector target, dùng chung |
| target_corresponding | target 1→draft 0; 9→1; 17→2; 25→3; 33→4 |

Trong mỗi scoring, loại D keys thấp nhất hoặc cao nhất ngoài G0, tie-break
theo absolute position tăng dần. Hai phía có cùng D trên **từng layer**;
target_corresponding có masks khác nhau giữa layers nhưng counts bằng nhau.

- Modal: D ∈ {1024,2048}; 7 scorings × 2 loại × 2 D = 28 actions/state.
- B200 dev: thêm D=4096; 42 attention actions/state.
- Cùng 4 controls vùng, pilot có 32 actions/state, tối đa 1.152 replays cho
  6 documents × 6 prefixes, chưa tính matched-deletion và trường hợp infeasible.
- D không khả thi được đánh dấu trước; không clip thành D nhỏ hơn rồi giữ
  nhãn cũ. Full observation được tái sử dụng giữa D và scorings của cùng state.

Primary A1-attention là target_mean5 tại D=2048. Các layer riêng, mapping
tương ứng, D khác và interaction theo phase/length là secondary. Chọn
contrast vùng và attention để xác nhận trước khi mở test, không lựa
contrast có p-value nhỏ nhất trên heldout.

Kết luận cần nêu: vùng nào bổ sung thông tin hữu ích, vị trí high-attention
có quan trọng hơn low-attention, và tác động có thay đổi theo generation
phase hay không. Attention mass được giữ chỉ là diagnostic đi kèm ΔL.

### 8. A2 — Chọn context bằng tín hiệu nào

Dùng G nhất quán và chọn **chính xác B keys/layer** bằng token ranking.
Selection theo chunk 128 tokens hiện có chỉ là secondary thực dụng: nếu
chạy, phải so cùng actual counts hoặc công khai phần budget dư; không gộp
vào bảng chính theo nominal B.

| Selector | Score và phạm vi |
|---|---|
| Random | Sampling không hoàn lại trên keys ngoài G; seeds 17/42/123; shared mask |
| Recency | Absolute position giảm dần trên tất cả keys ngoài G; shared mask |
| Target-parent | Layer 18 của phương pháp hiện tại; shared mask, parent row đúng |
| Previous DFlash | Scores của dense draft round trước, riêng từng draft layer |
| Current DFlash | Dense draft tại chính fixed prefix rồi chọn lại; oracle offline |

Recency ở đây bao gồm source và output ngoài G, khác tên recent_only trong
selector cũ vốn fill source chunks gần cuối. Ghi rõ selector ID mới.
Nếu A1 tìm target scoring tốt hơn layer 18, thêm best_target_from_A1 làm
đối chứng mạnh trên dev, rồi khóa scoring trước heldout.

**Attention DFlash:** mean qua heads và 15 proposal queries, không gồm
anchor query. Softmax vẫn có full context + 16 current block keys trong
mẫu số; chỉ sau đó loại current block để xếp context. Ranking chính riêng
5 layers; shared mean là một ablation thêm trên B200. Khi tạo shared mean,
chuẩn hóa cache-conditional từng layer trước mean và ghi convention.

Previous scores chỉ lấy từ round đã diễn ra, với origin prefix/round và age.
Không dùng gold summary, outcome current replay, hoặc attention tại prefix
hiện tại để gắn nhãn previous. Keys mới chưa có previous scores nằm trong
recent protection khi age không vượt 16 rounds với gamma 15; phải kiểm
coverage thực tế, không gán score 0 cho omitted/unseen keys để giả ranking.

A2 chính đo age=1 trên full reference trajectory. Đây là phép kiểm tín hiệu
causal khi dense previous scores có sẵn; **chưa chứng minh tín hiệu miễn phí
trong reduced rollout**. Previous sparse attention không nhìn omitted keys
nên không thay thế dense global ranking. Chi phí bootstrap/refresh và độ
bền ranking được kiểm riêng ở A3.

- Modal: B ∈ {1024,2048}; 7 actions/B gồm 3 Random seeds và 4 selectors còn lại.
  Tối đa 504 replays, giảm tại prefill không có previous signal.
- B200 dev: thêm B=4096; target scoring mạnh và shared/per-layer ablation nếu
  A1 có tín hiệu. Budget chính để xác nhận A2 là B=2048.
- Oracle được ghi đầy đủ chi phí dense diagnostic nhưng không tham gia chọn
  selector để triển khai hoặc bảng speedup.
- Reuse full dense draft và scores current oracle của mỗi state; các reduced
  actions vẫn phải chạy drafter và verifier thật.

Báo mean ΔL, P(ΔL≥0), P(ΔL>0), quantiles ΔL, prefix-survival P(L≥i) và
coverage theo dataset/phase/length. Random seeds được average trong state
trước aggregation theo document; chúng không tạo ba documents độc lập.

Previous attention chỉ được gọi tốt hơn cả ba đối chứng khi cả ba contrasts
đều đạt tiêu chí mục 3. Nếu chỉ vượt một đối chứng, nói rõ đối chứng đó.
Selector để A3 được chọn trên dev từ nhóm có thể triển khai: ưu tiên
preservation, sau đó total round cost có hạch toán; không chọn theo mass.

### 9. A3 — Budget, chi phí và thời gian reuse

#### 9.1. Quét budget

Với selector đã chọn từ A2, quét B ∈ {512,1024,2048,4096,8192,Full}. Đây
là fixed-budget cells; không dùng dynamic budget theo entropy/history.
Giữ Full và Recency B=2048 làm controls, và budget được chọn cho selector chính.

Trên mỗi fixed prefix đo:

- L(B), ΔL(B), các vector actual context counts.
- T_draft, T_verify, T_signal, T_select, T_gather, T_bank_update.
- T_round theo wall-clock đã synchronize phù hợp và J_t=T_round/(1+L_t).
- Chi phí lấy scores phục vụ rounds sau và chi phí dense refresh.

J_t là chỉ số tại round. Chi phí amortized báo bằng sum T_round / sum(1+L)
trên các regular rounds; không gọi mean các ratio là chi phí toàn request.
E2E wall-clock gồm prefill, full bank, bootstrapping, refresh, host launch,
selection, gather, verification và finalization. Model load/cold-start Modal
được báo riêng vì là chi phí vận hành, không cộng vào primary request latency.

Native full DFlash đã qua correctness là baseline E2E chính; full adapter là
đối chứng implementation. Báo cả hai để gain không chỉ xuất phát từ việc
full adapter gather/copy toàn context chậm. AR là control exactness và
latency tham khảo.

Modal quét 6 budgets bằng replay, nhưng chỉ rollout **hai reduced budgets**
có triển vọng trên dev, cùng native Full, adapter Full và AR. Mỗi variant
có 3 repetitions/document; timing repeat không tăng số outcome độc lập.
B200 dev chạy toàn budget feasible; chọn một B toàn campaign trước heldout.
Nếu không có budget đáp ứng preservation, báo trade-off và không claim
acceptance preservation.

#### 9.2. Reuse ranking sau khi đã có budget tốt

Đo age ∈ {1,2,4,8,16} trên replay trajectory: scores tại round t−age chỉ
được chọn trước khi nhìn current draft/verify. So ΔL so fresh ranking,
P(ΔL≥0), coverage và source-position overlap theo age.

Nếu selector thắng là previous attention, budget sweep dùng period=4 như
một cấu hình tham chiếu đã khai báo, rồi mới khảo sát period 2/4/8/16 sau
khi có B tốt. Không mặc định 4 là tối ưu. Age=1 chỉ đo chất lượng tín hiệu;
refresh period=1 là dense-every-round instrumentation control, không phải
một reduced-context policy có lợi miễn phí.

Dense refresh phải là **round sinh thật** được verify và commit, có full
cost. Round 0 bootstrap Full; các rounds còn lại dùng B đã khóa. Đây là
lịch refresh cố định, không phải adaptive budget controller. Nếu coverage
ranking không đủ cho keys ngoài G, refresh thực và ghi lý do/chi phí.

Modal chỉ đo age decay bằng replay và nhiều nhất một lịch refresh thử.
B200 đo rollout periods 2/4/8/16, khóa một period trên dev và xác nhận trên
heldout. Nếu selector tốt nhất là Recency hoặc Target-parent, refresh DFlash
chỉ là diagnostic; không ép thêm dense draft vào policy đó.

### 10. Đo timing, thống kê và lựa chọn trước heldout

**Acceptance:** một outcome/greedy action/fixed state; timing repetitions
không được nhân số observations. Mean các prefixes trong document, rồi
mean documents trong dataset. Pooled chính là mean đều ba dataset means,
công khai thêm mean theo document để thấy ảnh hưởng của quotas.

**Latency:** warmup trước từng nhóm shapes (context length, budget, backend,
block shape) ngoài timed request nhưng vẫn tính vào chi phí Modal. Proposal:
3 warmups/shape, 3 timing repetitions; nếu pilot noise cao thì báo uncertainty
và dành phép đo 5 repetitions cho B200 trước lock. Randomize variant order
trong từng document/repetition, xen Full control; batch 1 và một process/GPU.
Collector diagnostic của A1/A2 được tắt trong native Full E2E baseline.
Selected policy E2E phải bật và trả toàn bộ overhead cần để chạy policy đó.

CUDA events phục vụ GPU components; synchronize tại biên đo và dùng host
wall-clock cho tổng round/request. Không cộng các component overlapping để
giả total. Pilot timing shapes không đủ warmup được gắn diagnostic.
Báo GPU name/UUID/capability/driver, runtime, thermal/utilization nếu sẵn có.

E2E speedup của document là ratio mean repeated latency Full / Variant,
với cùng output IDs, output cap, stopping và prompt. Geometric mean trong
dataset, rồi geometric mean ba dataset ratios. Không lấy ratio throughput
trung bình của các requests có output lengths khác nhau.

CI dùng stratified paired cluster bootstrap, **resample source groups trong
từng dataset**, 10.000 resamples, seed 42; mọi prefixes/repeats/queries của
group đi cùng nhau. Với pilot 2 groups/dataset, CI chỉ diagnostic và phải
công khai ít clusters.

Đăng ký families xác nhận:

- A1-attention: target_mean5, D=2048, CI 95%.
- A1-region: ba additive contrasts và matched source-vs-recent D=128.
  Bốn contrasts dùng simultaneous Bonferroni CI 98,75%; báo eligible
  subcohort riêng từng contrast.
- A2: previous-vs-Random/Recency/Target-parent tại B=2048 dùng simultaneous
  Bonferroni CI 1−0,05/3 (xấp xỉ 98,33%); best_target_from_A1 nếu thêm là
  secondary đã khóa.
- A3: một selector/B/refresh period đã khóa; CI 95% cho E2E và ΔL.
  Claim preservation-and-speedup chỉ khi cả hai guardrails đạt.

Các layer/D/B/length/age khác là exploratory hoặc cần đăng ký family riêng.
Không chọn config bằng test. Dùng document variance từ dev để ước lượng
độ rộng CI dự kiến trước lock; 48 groups là quy mô khả dụng, không bảo đảm
power. Nếu precision dự kiến chưa đủ, mở rộng nguồn document trước test
hoặc chấp nhận kết luận inconclusive; không nhân QMSum queries thành meetings.

Natural-EOS results và capped results có breakdown riêng. ROUGE được tính
qua helpers chung khi có reference; greedy exact output là guardrail chính.
Không dùng ROUGE tốt để hợp thức hóa mismatch. Workload fixed-output nếu
cần là secondary có stopping/cost scope riêng, không gộp vào natural headline.

### 11. Modal tùy chọn: GPU, giới hạn 20 USD và campaign B200

Toàn bộ mục 11.1–11.2 chỉ áp dụng nếu chủ động chọn Modal cho diagnostic.
Đường chạy B200 chính tại mục 14 không phụ thuộc Modal SDK, tài khoản hoặc
volumes. Các dòng ma trận ghi Modal/pilot ở mục 7–9 là quy mô pilot nhỏ
tương đương, cũng được dùng cho pilot B200.

#### 11.1. GPU và vận hành

Chọn **một H100 80 GB, request H100!**, 4 CPU cores, 32 GiB host RAM.
Hopper có đường FA4 đã từng chạy trong repo; kích thước checkpoint 4B,
batch 1 và context ≤16K chưa tạo lý do phải dùng B200 cho pilot.

[Modal GPU docs](https://modal.com/docs/guide/gpu) ghi H100 có thể tự nâng
sang H200; H100! tránh nâng khi benchmark. Luôn log GPU thực tế. Nếu cần
GPU khác thì mở runtime riêng, kiểm parity lại và không gộp latency hai GPU.
L40S là phương án diagnostic SDPA/eager riêng nếu FA4 không hoạt động,
không phải fallback âm thầm cho primary timing.

Build image/pin dependencies và chuẩn bị snapshots bằng CPU job hoặc upload
cache trước khi cấp GPU. Reuse image theo digest. Chỉ upload code cần thiết,
ba dataset files/cohort manifests và model snapshots đã xác minh; không upload
toàn workspace có credentials. GPU job dùng local snapshots và offline flags.
Production server B200 vẫn dùng system python3 3.12 và dependencies đã cài,
không cài online hoặc tạo venv riêng.

Một container GPU load model một lần, chạy các phases tuần tự. Commit
artifacts sau document/batch nhỏ; outputs có run ID riêng. Scale tối đa
một GPU/container và một container hoạt động; không deploy service giữ nóng.
Không tự retry toàn job sau lỗi. Resume theo completed cell IDs và runtime
hashes, tính mọi giờ retry vào trần chung.

#### 11.2. Dự toán và điều kiện dừng

Giá niêm yết được kiểm tra 13/10/2026 tại
[Modal Pricing](https://modal.com/pricing): H100 0,001097 USD/s =
3,9492 USD/GPU-hour; CPU 0,0000131 USD/core/s; RAM 0,00000222 USD/GiB/s.
Với 4 cores và 32 GiB RAM, GPU container dự toán **4,393584 USD/hour**.
Đây là phép tính từ giá công khai, chưa phải hóa đơn workspace.

| Phase | Trần GPU wall time | Dự toán container |
|---|---:|---:|
| Runtime/kernel + parity/stopping | 45 phút | 3,30 USD |
| Reference traces + 6-document launcher smoke | 30 phút | 2,20 USD |
| A1/A2 replay + A3 budget/age replay | 45 phút | 3,30 USD |
| A3 selected rollout/timing | 45 phút | 3,30 USD |
| Dự phòng kỹ thuật, chỉ dùng khi còn quota | 30 phút | 2,20 USD |
| Tổng GPU container tối đa | 3 giờ 15 phút | 14,28 USD |

Giữ tối đa khoảng 3,72 USD trong trần 18 USD cho CPU asset job, build,
storage/egress và sai số dự toán. Hai USD còn lại không được lập lịch dùng.
Credit 20 USD do người dùng cung cấp; không giả định còn thêm monthly credit.
Trước job, đọc mức usage/credit thực và giá hiện hành; nếu phần khả dụng thấp
hơn dự toán thì giảm phases hoặc chuyển B200.

Runner tương lai cần cumulative wall-time/cost ledger, phase timeout và
global GPU limit 11.700 giây, tính cả model load, kernel compile và warmups.
Khi projected total vượt 18 USD hoặc thời gian phase hết, checkpoint và dừng;
phase chưa đủ coverage ghi partial. Không coi trần tính trong runner là
bảo đảm tổng invoice: các workloads khác trong workspace cũng tiêu credit.
[Modal Budgets](https://modal.com/docs/guide/budgets) phân biệt usage budget
trước credit và spend limit sau credit; việc đổi dashboard budget thuộc
chủ workspace, không nằm trong nhiệm vụ lập kế hoạch này.

Ước lượng thời gian được hiệu chỉnh bằng timing của 1–2 documents đầu, trước
khi mở hết pilot. Nếu parity hết 45 phút vẫn fail, giữ ngân sách còn lại và
chuyển diagnostic sang B200. Nếu có deadline/quota chặt, ưu tiên theo thứ tự:
parity → A1 đủ 7 scorings và hai D → A2 hai B → A3 replay → A3 rollout.
Không bỏ một scoring của A1 rồi báo đã trả lời đầy đủ ma trận.

#### 11.3. Campaign B200

Nếu pilot đã chạy trên B200, giữ code, snapshots, manifests và runtime đó
cho campaign; mọi đổi kernel/version đều phải kiểm parity lại. Nếu đã chọn
pilot Modal, chuyển assets/manifests sang server và tạo runtime signature
B200/CUDA/kernel mới rồi chạy lại parity/smoke trước ma trận.
Không chuyển kết quả H100 thành số latency B200 bằng hệ số ước lượng.

Chạy 24 dev source groups, hoàn thành A1/A2/A3 và refresh study, rồi khóa
selector/scoring/B/period/baselines/thresholds. Heldout 48 source groups chạy:

- A1: 4 controls vùng, contrasts matched-deletion khả thi và primary
  target_mean5 low/high D=2048; thêm corresponding mapping đã khóa nếu
  dev cho thấy cần xác nhận.
- A2: năm selectors, Random đủ 3 seeds, B=2048; oracle tiếp tục chỉ diagnostic.
- A3: native Full, adapter Full, selector/B/period đã khóa và Recency control;
  AR exactness kiểm trên từng document bằng reference IDs đã verified.

Một worker batch 1/card. Data-parallel theo document để tăng tốc hoàn tất,
nhưng mọi variants/repetitions của một document chạy cùng card/process;
không chạy nhiều worker cạnh tranh một GPU trong primary latency study.
Manifest ghi shard IDs/GPU IDs; merge chống duplicate, kiểm expected cells
và tất cả error rows. Giới hạn VRAM theo server, không sao chép đồng thời
nhiều full-bank snapshots không cần thiết.

### 12. Công việc triển khai sau khi kế hoạch được duyệt

Danh sách dưới đây là **phạm vi tương lai**, chưa có script/CLI mới để chạy
A1–A3. Lượt lập kế hoạch chỉ sửa tài liệu này.

| Task | Files/phạm vi dự kiến | Đầu ra và điều kiện hoàn tất |
|---|---|---|
| T1 — Khóa cohort và assets | Helpers splits/prompt hiện có; cohort manifests dưới outputs/ | IDs/groups/lengths/quota, exposure audit, model/runtime hashes, cost ledger |
| T2 — Harness verifier | generation.py, bank/attention APIs; harness diagnostic mới | AR/native/adapter và sparse parity, branch/EOS/cap report |
| T3 — Replay và masks | Module mới src/TrainingFree/context_interventions/ | Immutable state, exact token budgets, region/low/high/per-layer masks; CPU logic checks |
| T4 — Scores và causality | Wrapper TargetQueryCapture; collector context scores đủ source+output | 5 target layers, parent row, previous/current draft, origin/age/coverage, collector parity |
| T5 — Evaluation/report core | Evaluator mới dùng io_util/ROUGE helpers chung | Pairing/cardinality validation, document macro/CI, coverage và decision memo |
| T6 — Launchers [INTEGRATION] | scripts/probe_dflash_context_interventions.py, scripts/modal_dflash_context_interventions.py, scripts/run_dflash_context_experiments.sh | Một evaluator/replay core cho Modal và B200; phase gates, timing, budgets, resume, shards |

Các paths mới là đề xuất, không phải files đã triển khai. T6 ghép pipeline
inference, không gọi training validation. T1–T5 hoàn tất logic và contracts
trước khi launcher được cấp GPU; model backend thực được kiểm ở R2–R4.

Contracts CLI dự kiến: phases prepare, preflight, parity, trace, a1, a2, a3,
refresh, report; options run-id, cohort-manifest, target/draft paths,
output-root, runtime/backend, budget/gamma, seed/repetitions, max-cost,
max-gpu-seconds, resume và shard-index/count. Backend option phải được nối
thật vào loader và ghi backend đã chạy; không dựa vào flag chỉ tồn tại trên giấy.

Wrapper B200 dùng scripts/common/runtime.sh, FAST_INFER_PYTHON hoặc python3
hệ thống 3.12 và master config qua config/master.path hoặc --config. Đọc
model/data paths từ config, không hardcode paths Modal. GPU IDs/shards là
launcher options; core không import Modal hoặc validation fixtures.

Kiểm chứng logic dự kiến tập trung các cases có thể làm sai kết luận:
source/output span, protected set theo phase, per-layer actual count,
absolute/RoPE positions, snapshot restore, parent early-reject row, previous
signal age/future leakage, Random pairing, cluster bootstrap và resume/merge.
Đây là yêu cầu cho bước triển khai, chưa có checks được chạy cho harness mới.

### 13. Artifacts, quan sát tiến độ và điều kiện kết thúc

Artifacts ở outputs/dflash_context_interventions/<run_id>/, được gitignore:

~~~text
protocol.json                 # Cấu hình khóa, runtime/data/code hashes
cohort_manifest.json          # Source groups, query và quota/eligibility
budget_ledger.json            # Wall time, phases, cost dự toán/đã dùng
evidence/parity.json           # Cases, tensor errors, token equality, gate
traces/<group_id>/             # Prefix IDs và causal metadata để rebuild
replay/{a1,a2,a3,age}.jsonl     # Actions, outcomes, counts, positions hashes
rollout/<variant>/             # Requests, rounds, output IDs, timing
locked_config.json            # Selector/B/refresh/baselines trước test
reports/{coverage,metrics}.json
reports/decision.md
~~~

Replay rows tối thiểu gồm dataset/group/sample/prefix IDs, split,
logical length, output phase, gamma thực, pending anchor, G/protection hash,
scoring layer/mapping, selector, nominal và actual counts từng layer,
positions hash, origin/age, Random seed, L_full/L_action/ΔL, timing scope,
runtime/code signatures và status/reason.

Mọi JSONL qua io_util.JsonlWriter và kết thúc summary. Summary phân biệt
planned/eligible/executed/success/infeasible/error/missing; error/partial không
biến thành zero gain hoặc successful sample. Resume không append thêm outcome
trùng hoặc summary giữa stream. Báo cohort/phase bắt đầu, progress theo
document/cell, heartbeat tối đa 60 giây, phase-end và budget remaining.

Sai asset/hash, source mapping, NaN/Inf scores/timing, invalid positions,
cache corruption hoặc token mismatch là lỗi gate. OOM/cell timeout được
ghi rõ và dừng/rerun trong quota; không tự thay B, dtype hoặc dataset để qua.

| Câu hỏi | Đầu ra cần có | Điều kiện kết luận |
|---|---|---|
| A1: context nào ảnh hưởng acceptance? | Region contrasts + matched deletion + low/high matrix | Effect và CI trên cùng prefix; counts/positions hợp lệ |
| A2: tín hiệu previous có tốt không? | Paired selector table, P(ΔL≥0), age/coverage | Vượt đối chứng nào được nói đúng đối chứng đó; oracle riêng |
| A3: giảm đến B nào có lợi? | Budget curves, component costs, paired E2E và refresh curve | Preservation/improvement và latency gates; full overhead |
| Có thể claim exact acceleration? | AR/native/adapter equality, EOS/cap và heldout coverage | Correctness đạt, đủ planned cells, output scope khớp |

Decision memo cuối ghi từng giả thuyết confirmed/not_supported/inconclusive,
highest verified rung, giới hạn cohort/GPU, deviations và artifact hashes.
Kết thúc nghiên cứu khi đã trả lời cả ba câu hỏi bằng bằng chứng hoặc nêu
đúng câu chưa đủ bằng chứng. Nếu A2 không vượt baseline đơn giản, ghi rõ
dừng hướng adaptive controller mới; không tiếp tục thêm độ phức tạp để
tìm gain sau khi xem heldout.

### 14. Đường chạy chính trên B200 qua shell launcher

#### 14.1. Trách nhiệm của launcher

Đề xuất scripts/run_dflash_context_experiments.sh làm entrypoint chung.
Shell chịu trách nhiệm config/interpreter, GPUs, thứ tự phases, logging,
gates, resume và merge; Python core thực hiện masks, attention, draft/verify,
metrics và thống kê. Launcher chỉ được triển khai sau giai đoạn lập kế hoạch.

Luồng chính:

~~~text
master config + runtime
  → preflight thư viện/assets/cohort
  → parity + EOS/cap + collector checks
  → reference traces + immutable replay states
  → A1 → A2 → A3 → refresh study
  → dev report + locked configuration
  → heldout replay/rollout → paired CI + decision memo
~~~

| Mode launcher dự kiến | Công việc | Quy mô |
|---|---|---|
| preflight, mặc định | Kiểm paths/config/imports/version/GPU, chưa benchmark | Không chạy generation |
| validate | Kernel/collector forward và AR/native/adapter/sparse parity | 2 groups/dataset, cases biên bắt buộc |
| pilot | Sau validate đạt, chạy ma trận nhỏ và timing chọn lọc | 6 groups, 6 prefixes/group tối đa |
| dev | Ma trận khám phá đầy đủ, budget và refresh selection | 24 groups |
| confirm | Chỉ đọc locked config, xác nhận trên test chưa xem | 48 groups nếu exposure audit giữ đủ quota |
| report | Coverage, paired metrics/CI và memo từ artifacts | CPU |

Options dự kiến gồm --config, --mode, --gpu-ids, --run-id, --output-root,
--resume và --plan-only. --plan-only in expected groups/cells/resources và
không cấp GPU cho generation. --resume phải tiếp tục đúng run signatures;
không tự đổi seed/budget/phase khi gặp lỗi. Confirm từ chối chạy nếu chưa có
parity evidence và locked config hợp lệ.

Ban đầu dùng một card B200 và batch 1. Khi cần giảm wall time, phân documents
cho nhiều cards, mỗi card một worker; giữ mọi variants của một document trên
cùng card. Dừng các workloads khác trên cards benchmark trước primary timing.
Không dùng throughput launcher nhiều process/card để suy single-request latency.

#### 14.2. Paths và assets canonical

Đọc theo [hồ sơ server](../../../docs/server_environment.md):

| Thành phần | Path/config |
|---|---|
| Repository | /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum |
| Master config | /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env |
| Corpus ba datasets | /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k |
| Target được ghi trong hồ sơ | /workspace/storage-shared/models/Qwen3-4B |
| Pretrained DFlash b16 | MODEL_DFLASH_DRAFT trong master; cần xác nhận snapshot đầy đủ |
| Output | outputs/dflash_context_interventions/<run_id>/ dưới repository |
| JIT/cache | FAST_INFER_CACHE_ROOT ở filesystem có quyền ghi |

Launcher dùng Python hệ thống 3.12 qua runtime chung, offline flags và master
config; không yêu cầu clone/download/cài package khi chạy. Đặt interpreter
explicit nếu có Conda/venv đang active để tránh chọn nhầm Python của shell.
Manifest data/split được rebase sang server paths nhưng giữ fingerprints,
IDs và content hashes; không source đường dẫn /home/tuantb từ artifacts local.
Không giả định outputs gitignored đã được đồng bộ lên server.

#### 14.3. Những gì đã có và những gì cần kiểm/bổ sung

Đánh giá dưới đây từ source/hồ sơ trong repo; **chưa kiểm tra trực tiếp máy
server B200 trong lượt lập kế hoạch**.

| Thành phần | Bằng chứng hiện có | Việc còn cần |
|---|---|---|
| Python/Torch/CUDA/Transformers | Hồ sơ dùng system Python 3.12 và stack GPU cài sẵn | Preflight ghi phiên bản thực, CUDA tensor và GPU thực |
| FA4/CUTLASS compatibility | Runner có shim ampere_helpers/nvvm.fmax và probe FA4 | Probe qua helper repo rồi kernel forward thật; khóa backend |
| DFlash implementation | externals/dflash và core bank/gather/verifier hiện có | Validate với pretrained target/draft snapshots trên B200 |
| Model target/draft | Target path ghi trong hồ sơ; draft config qua master | Check config, index, mọi weight shards, tokenizer, hashes và block size 16 |
| Datasets | Canonical LongBench path đã ghi | Check ba files, references, source spans; tạo cohort/split/exposure manifests |
| JSONL/ROUGE/report helpers | Helpers chung có trong repo | Evaluator A1–A3 phải nối vào helpers và bổ sung paired analysis |
| Experimental harness A1–A3 | Chưa có dedicated CLI/masks/replay cho toàn protocol mới | Triển khai T2–T6 trước khi shell launcher chạy được nghiên cứu |
| Dedicated shell launcher | Đã có runner Context-Adaptive DFlash cũ | Viết wrapper phase gates/modes/resume cho campaign mới |

Các packages trực tiếp của campaign chủ yếu là torch, transformers,
accelerate, safetensors, numpy, huggingface_hub, FA4 và các dependencies CUDA
phù hợp; chúng đã được đề cập trong shared runtime/manifest. DFlash được
vendored; ROUGE và bootstrap có helpers/std library hoặc NumPy. Modal,
SGLang/vLLM, SpecForge và libnuma không thuộc đường inference này.
Không đề xuất cài thêm package trước khi có lỗi preflight cụ thể.

Có hai điểm preflight cần xử lý:

1. Hồ sơ server ghi FA4 b15/CUTLASS 4.5.x, requirements.txt pin FA4 b19/
   CUTLASS 4.7.1. Manifest là mục tiêu dependency, không chứng minh máy hiện
   cài phiên bản đó. Lưu versions thực; nếu stack hiện tại qua shim/probe,
   kernel và parity thì có thể khóa stack đó cho campaign. Không tự nâng
   package để làm khớp bản Modal.
2. scripts/check_b200_env.py vẫn kiểm import flash_attn cho baseline dflash;
   kiểm này không chứng minh FA4 cho B200 và chưa có tên campaign mới.
   Dedicated preflight phải dùng _probe_flash_attention_4 của repository,
   CAD_ATTENTION_BACKEND=flash_attention_4 và forward thật. SDPA dùng cho
   diagnostic đã ghi scope; không silent fallback trong primary benchmark.

CLI cũ scripts/infer_context_adaptive_dflash.py có prepare/preflight/smoke/
calibrate/dev/test/lock/report và selected target layer đơn. Nó chưa có
phases A1/A2/A3/parity/trace của campaign mới, collector đầy đủ mọi context
positions, 7 scoring configurations hay exact token-budget replay. Lặp
CLI cũ bằng shell chưa tạo được các can thiệp khoa học đã yêu cầu.

Kết quả preflight tương lai phải phân biệt package_missing,
kernel_incompatible, asset_missing, harness_not_implemented và parity_fail.
Như vậy người chạy biết thiếu artifact, thiếu mã nghiên cứu hay lỗi runtime,
và chỉ bổ sung đúng phần đó trước khi dùng GPU cho ma trận lớn.
