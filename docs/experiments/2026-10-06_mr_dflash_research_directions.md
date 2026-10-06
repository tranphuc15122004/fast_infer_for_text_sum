# Đề xuất phát triển MR-DFlash với ngân sách train hạn chế

Ngày: 2026-10-06. Đây là đề xuất nghiên cứu dựa trên source và tài liệu hiện có;
các pilot dưới đây chưa được chạy. Mốc 30 giờ trên 4 GPU do người dùng cung cấp
tương đương 120 GPU-giờ. Chưa xác nhận mốc này thuộc run MR nào.

## 1. Khuyến nghị

Tạm dừng các run MR từ đầu. Giữ checkpoint DFlash đã có, thử giảm context/memory
ở phía drafter, rồi chỉ học phần memory bổ sung nếu thử nghiệm không train cho
thấy có cơ hội. Mục tiêu là cải thiện chi phí inference long-context của DFlash
đã được huấn luyện, với ngân sách thích nghi nhỏ và verifier chính xác.

Ưu tiên ba hướng theo thứ tự:

1. Chọn context cho drafter không cần train; xác định đường đánh đổi acceptance
   và chi phí theo độ dài input.
2. Adapter memory gắn vào backbone DFlash giữ nguyên độ sâu; chỉ fine-tune ngắn
   sau khi hướng 1 có tín hiệu.
3. Điều chỉnh ngân sách memory và độ dài verify theo chi phí quan sát được.

Nếu không giảm được latency nhưng giảm rõ VRAM, đánh giá như một hướng tiết
kiệm memory. Claim tốc độ và claim memory phải có bằng chứng riêng.

## 2. Bằng chứng trong repo và giới hạn

| Quan sát | Hệ quả cho quyết định |
|---|---|
| [Nhật ký 4 B200](2026-10-05-dflash-4gpu-training-log.md) ghi DFlash-5L dừng ở step 2.155/8.430 sau 9h17m, train accuracy 23,1%. | Đây là run DFlash baseline, không phải bằng chứng MR mất đúng 30h. Loss giảm và train accuracy chưa chứng minh hội tụ acceptance. |
| [Config warm-start](../../src/MR_DFlash/configs/train_qwen3_4b_mr_dflash_fast_warmstart.yaml) dùng MR 2 stage từ DFlash 5 layer. | Vừa thay memory vừa giảm độ sâu; không cô lập được tác động của memory. |
| [Converter checkpoint](../../src/MR_DFlash/checkpoint.py) chọn source layer theo chỉ số stage. | Với 2 stage, chỉ lấy layer 0 và 1; layer 2–4 không được chuyển. Compressor/indexer vẫn khởi tạo mới. Load thành công không đảm bảo giữ năng lực của DFlash-5L. |
| Config warm-start không có chế độ freeze backbone riêng; trainer tối ưu các tham số có `requires_grad=True`. | Warm-start hiện tại không đồng nghĩa với chỉ học compressor/indexer. Adapter-only cần bổ sung cơ chế freeze và kiểm tra gradient. |
| [Reference inference](../../src/MR_DFlash/inference.py) chạy target full-prefix mỗi vòng, `use_cache=False`. | Dùng cho correctness/acceptance; so tốc độ với DFlash production cần cùng verifier có KV cache và rollback tương đương. |
| [Workboard M1](../project_workboard.md) ghi hidden parity fail 32/32 mẫu ở mỗi split, chưa có fingerprint weights lịch sử. | Cần xác định snapshot và sai lệch do backend/dtype hay do weights/token/layer trước khi dùng cache cho run mới. Không tự kết luận các checkpoint cũ vô dụng. |

Không dùng nhận định trong nhật ký rằng 1,5–2 epoch đã thu được hơn 90% năng lực
làm tiêu chí dừng. Cũng không dùng ước tính warm-start 1 giờ như kết quả đo.
Giảm anchors từ 512 xuống 256 giảm số query được giám sát, nhưng chi phí I/O,
memory build, indexer, MLP và evaluation khiến thời gian không nhất thiết giảm
một nửa. `tokens_per_second` của trainer đếm phần tử input, có thể gồm padding;
đó không phải số token được chấp nhận khi inference.

## 3. Các hướng ưu tiên

### A. Chọn context cho DFlash đã train, không cập nhật trọng số

Giả thuyết: ở input dài, drafter có thể dùng ít context hơn mà vẫn giữ phần lớn
độ dài prefix được target chấp nhận. Thử giữ một số token đầu, recent window
và token xa được chọn theo stride hoặc score rẻ; giữ nguyên absolute position
và RoPE. Bước đầu chọn token, tránh đồng thời thêm pooling học được.

- Giữ nguyên DFlash checkpoint, số layer, feature layers và block size 16.
- So dense với recent-only và recent + distant selection. Thử hai tổng budget
  512 và 1.024 token, bao gồm cả recent/sink tokens.
- Có thể bắt đầu recent window 256, sink 32; dùng tập calibration để chọn.
- Selection chỉ được dùng thông tin đã có ở thời điểm draft. Oracle dùng
  target token tương lai chỉ là phép chẩn đoán.
- Đo cả refresh mỗi vòng và refresh mỗi 4 vòng nếu pilot đầu có tín hiệu.
- Không bật xuất attention matrix dày đặc chỉ để lấy score nếu chi phí này
  triệt tiêu lợi ích. Selector, gather và cập nhật memory phải được tính giờ.
- Sparse mask trên tensor dense chỉ giúp thăm dò acceptance. Claim tốc độ cần
  thật sự giảm lượng K/V được đọc hoặc có kernel thực thi sparse phù hợp.
- Target tiếp tục verify với full context; memory chỉ append token đã commit.

Đối chứng tối thiểu là dense, recent-only và recent + distant selection. Nếu
selector không hơn recent-only, chưa có bằng chứng cho việc thêm indexer.

Đây là ứng viên ưu tiên vì không mất train. Tuy nhiên, chọn context cho draft
đã có tiền lệ: [SpecExtend](https://arxiv.org/abs/2505.20776) dùng retrieval
liên model cho draft KV. Áp dụng cho DFlash cần kiểm chứng lại; không mặc định
coi việc chuyển sang block diffusion là đóng góp mới đầy đủ.

### B. Adapter memory trên backbone giữ nguyên độ sâu

Giả thuyết: phần mất acceptance khi chọn/nén memory có thể được khôi phục bằng
một module nhỏ, với ngân sách thích nghi thấp hơn retrain drafter.

Thiết kế đề xuất giữ nguyên DFlash-5L và thêm nhánh memory có gate, khởi tạo để
đường dense ban đầu hoạt động. Sau đó tăng dần mức dùng memory nén. Đây là thiết
kế mới cần triển khai; config MR-2S hiện tại chưa làm điều này.

Pilot hai pha:

1. Freeze backbone, học adapter/compressor và gate; thêm indexer sau nếu selector
   đơn giản không đủ. Thăm dò 100–200 step với 64–128 anchors trên 1K–3K mẫu.
2. Nếu acceptance cải thiện, thử unfreeze layer cuối hoặc LoRA nhỏ với LR thấp
   hơn; mở rộng tối đa 300–500 step trong trần GPU-giờ đã đặt.

Anchors/data/LR là giá trị khởi đầu để calibration, không phải recipe đã xác
nhận. Dùng mẫu ngắn và một phần context dài phù hợp checkpoint; kiểm tra mẫu
dài còn giữ nguyên response được giám sát. Không cắt 16K xuống 4K bằng cách
truncate mù, làm mất summary hoặc toàn bộ supervision.

Freeze giảm optimizer state và gradient tham số, nhưng vẫn có thể cần backward
qua backbone để học adapter đặt trước nó. Không hứa tốc độ train tăng theo tỷ
lệ số tham số bị freeze.

Đối chứng: DFlash giữ nguyên; DFlash fine-tune thường cùng data/anchors/budget;
memory không train; memory + adapter. Báo chi phí checkpoint nền riêng, cùng
chi phí thích nghi mới. Nếu dùng DFlash-2L → MR-2S như pilot rẻ, phải ghi rõ đây
là thí nghiệm trên backbone khác với DFlash-5L.

Một công trình gần là [Strong Drafts Need Compact Memories](https://arxiv.org/abs/2608.30252),
dùng adapter tạo memory KV phía draft và full KV phía target. Cần đọc kỹ trước
khi xác định novelty cho phiên bản DFlash của mình.

### C. Điều chỉnh memory budget và độ dài verify theo chi phí

Giả thuyết: input và giai đoạn sinh summary cần ngân sách khác nhau. Một policy
nhỏ có thể tránh draft/verify dài khi prefix thường bị reject sớm.

Bắt đầu với bảng quy tắc trên calibration: số token commit gần đây, thời gian
draft/verify/update và chiều dài context. So với các cấu hình cố định đã tune
trên cùng calibration, tránh chỉ so với block size 16 mặc định.

Với checkpoint train block 16, bước rẻ là vẫn draft 16 nhưng chỉ verify prefix
4/8/16 theo policy. Điều này chủ yếu giảm verify cost; không giảm chi phí draft
16. Đổi trực tiếp số mask/query thành block 4/8 làm thay conditioning của block
diffusion, cần thí nghiệm riêng. Backend có thể cần compile/bucket độ dài để
policy không gây overhead mới.

Kết hợp budget memory và verify length chỉ sau khi từng knob riêng đã có lợi.
[AdaFlash](https://github.com/ZinYY/AdaFlash) đã có adaptive candidate length và
on-policy training; một policy heuristic đơn lẻ chưa đủ để claim novelty.

### D. Local correction head: phương án dự phòng

[DeLS-Spec](https://arxiv.org/abs/2607.07409) giữ DFlash cố định, học local head
độc lập rồi kết hợp logits. Đây là hướng giảm chi phí học đáng đối chiếu nếu
rejection chủ yếu do quan hệ token trong block.

Ưu tiên thấp hơn A/B: các [thí nghiệm causal screening cũ](../../outputs/dflash_residual/2026-09-07_causal_screening/final_report.md)
trong repo chưa thấy gain đủ cho causal reveal/self-conditioned redraft; đây
là bằng chứng trên context ngắn và runtime cũ, không bác bỏ local head trên
B200. Chỉ thử sau khi phân tích checkpoint hiện tại cho thấy đúng loại lỗi;
chi phí head và bước tuần tự phải vào latency cuối.

## 4. Đo đúng mục tiêu

Với mỗi vòng, dùng:

```text
chi phí/token = (T_draft + T_verify + T_select/update + T_other) / tau
tau = số token thực sự commit trung bình mỗi vòng, gồm token target bổ sung
T_e2e = T_prefill + tổng thời gian các vòng + overhead còn lại
```

Cách đánh đổi draft cost và acceptance được trình bày trong
[DFlash](https://arxiv.org/html/2602.06036v1). Train accuracy trung bình và
`eval_accept_ge_*` trên teacher-forced anchors chỉ là proxy; vẫn cần acceptance
trên trajectory target sinh khi inference.

Nếu draft/update chỉ chiếm 20% tổng latency, loại bỏ hoàn toàn phần đó chỉ cho
tối đa 1,25x khi acceptance và các phần còn lại giữ nguyên. Vì vậy profile
prefill/verify/draft trước: summary ngắn với prompt dài có thể bị prefill chi phối.

Khóa target weights/tokenizer, feature layers, prompt/chat template, dtype,
backend, output limit, batch size và hardware giữa các cặp. So greedy trước;
exactness với target greedy là điều kiện của pilot. Với sampling, cần verifier
đúng phân phối và proposal probability đúng; greedy parity chưa chứng minh điều
đó. Báo ROUGE khi có reference, cùng token output và EOS.

Đo riêng draft memory và tổng peak VRAM: giảm 70% draft KV không đồng nghĩa giảm
70% memory của toàn hệ thống vì target vẫn giữ full KV. Kernel/rollback/cache
target phải cùng semantics khi so các phương pháp.

## 5. Pilot với trần 24 GPU-giờ

Đây là ngân sách đề xuất, bằng 1/5 một run 120 GPU-giờ. Không quy đổi thành cam
kết wall-clock; đo step time và evaluation time trên server trước khi mở rộng.
Chi phí sửa runtime được ghi riêng. Nếu runtime chưa sẵn sàng, pilot chỉ đóng
được acceptance/cost profile, chưa đóng claim speedup production.

| Pha | Việc làm | Trần GPU-giờ | Điều kiện chuyển tiếp |
|---|---|---:|---|
| 0 | Ghi checkpoint/config thực tế; chẩn đoán cache snapshot; đo dense DFlash và checkpoint MR đã có nếu có | 4 | Biết acceptance và nơi chi phối latency; exactness đạt |
| 1 | Hai budget context, recent-only và distant selection; cùng verifier | 8 | Có giảm chi phí/token hoặc giảm memory rõ ràng |
| 2 | Chỉ chạy một adapter pilot có đối chứng fine-tune thường | 8 | Có cải thiện so với memory không train ở cùng budget |
| 3 | Kiểm chứng variant tốt nhất; thử policy nhỏ nếu pha trước có lợi | 4 | Lợi ích lặp lại được trên holdout |

Calibration riêng 24 document; holdout tối thiểu 60 document chia giữa
GovReport, Multi-News và QMSum, phân tầng input 4K/8K/16K. Thêm 32K chỉ nếu
target/draft/backend hỗ trợ và có dữ liệu phù hợp. Mẫu/split phải tách theo
document, loại trùng với train. Dùng cùng prompt/output budget cho các cặp;
tránh ép sinh qua EOS chỉ để cân token.

Gate sàng lọc đề xuất, không phải kết quả đã đạt:

- Hướng tốc độ: E2E nhanh hơn DFlash đã tune ít nhất 10% trên hai workload dài;
  paired confidence interval có tín hiệu dương, exactness 100% trên holdout.
- Hướng memory: draft KV giảm ít nhất 30%, E2E chậm thêm không quá 5%; báo thêm
  tổng VRAM và throughput ở mức concurrency liên quan.
- Nếu acceptance giảm làm chi phí/token tăng, dừng scale training. Nếu prefill
  chi phối và draft-only gần chạm trần lợi ích, chuyển trọng tâm sang prefill.
- Không mở thêm một grid nhiều loss/độ sâu/indexer. Chỉ scale ứng viên vượt gate;
  sau đó mới dùng benchmark lớn và nhiều seed để kết luận.

`scripts/mr_dflash/evaluate_pilot.py --exactness-check` đã có để kiểm tra
checkpoint và acceptance reference. Nó chưa thay thế benchmark production.
Local hiện chỉ dev CPU; các ngân sách trên dành cho server được cấp GPU.

## 6. Danh sách ứng viên và quyết định sàng lọc

| Ứng viên | Quyết định |
|---|---|
| Recent-only draft context | Đối chứng bắt buộc cho A |
| Recent + distant token selection | Ưu tiên A |
| Reuse selection qua nhiều vòng | Ablation A sau khi có gain |
| Pooling cố định không train | Hoãn; cần kiểm tra lệch feature/position |
| Adapter memory giữ DFlash-5L | Ưu tiên B nếu A có tín hiệu |
| DFlash-2L → MR-2S | Pilot phụ; không thay đối chứng 5L |
| LoRA/layer cuối cho memory adaptation | Pha hai của B |
| Adaptive verify prefix | Ưu tiên C theo bottleneck |
| Policy chung memory và block budget | Sau khi từng knob có lợi riêng |
| Local correction head độc lập | Dự phòng D |
| Cây draft rộng | Hoãn; CPU screening cũ không đạt, chưa có bằng chứng B200 |
| Train MR từ đầu, tăng depth hoặc đổi loss liên tiếp | Hoãn cho tới khi có pilot chứng minh cơ hội |

Câu hỏi nghiên cứu nên khóa: với DFlash đã train, có thể giảm lượng memory đọc
khi draft mà vẫn giữ cost/token tốt hơn ở context dài, chỉ cần ít hoặc không
cần thích nghi không? Đóng góp tiềm năng nằm ở cơ chế, vùng có lợi và trade-off
được kiểm chứng; tính mới cần đối chiếu các công trình gần trước khi viết paper.
