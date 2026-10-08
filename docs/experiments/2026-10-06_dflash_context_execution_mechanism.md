# Cơ chế điều phối context khi inference với DFlash đã train

Ngày: **2026-10-06**. Trạng thái: **đề xuất**, chưa triển khai hoặc có kết quả can thiệp cache. Tài liệu này thay hướng ưu tiên của [đề xuất selector học riêng](2026-10-06_dflash_learned_context_proposal.md) sau khi người nghiên cứu làm rõ mục tiêu: tận dụng DFlash đã train để cải tiến inference long context.

## 1. Mục tiêu và cơ chế được đề xuất

**DFlash tự xác định working set của drafter bằng attention của chính checkpoint đã train, tái sử dụng tập key trong ngắn hạn và làm mới từ toàn context khi cần.**

Thiết kế V1 dùng nguyên trọng số DFlash, Q/K/V projections, năm layer và block size 16. Thành phần mới là policy quản lý context đọc trong inference, gồm lựa chọn key, cập nhật working set và điều chỉnh budget. Thiết kế này không cần train predictor riêng; đây là lựa chọn triển khai cho mục tiêu hiện tại, không phải kết luận rằng mọi adaptation sau này đều cần bị loại.

Scope lựa chọn là **prompt + output đã commit trong cache phục vụ drafter**. Target tiếp tục verify với full context/KV. Mục tiêu ban đầu là giảm chi phí drafting trên long context, giữ số token thực tế commit mỗi round; giảm tổng resident VRAM cần một thiết kế storage riêng.

## 2. Working set

```text
Context mỗi draft layer đọc:
  16 vị trí block hiện tại, luôn giữ
  + nhóm bảo vệ nhỏ ở đầu/cache gần
  + các key rải rác được chính DFlash ưu tiên
```

Budget là `K` cache key **mỗi layer**, cộng 16 block key. Nhóm đầu/cuối nằm trong `K`; không thêm ngoài budget. Với variant chọn một tập chung cho mọi layer, cũng dùng `K` mỗi layer để so công bằng.

Một nhóm bảo vệ khởi đầu có thể là 64 key đầu và 128 key gần block nhất, loại trùng và giới hạn theo số token thực có. Đây là hyperparameter pilot, không phải số sink token đã được xác định bởi histogram. Phải so có/không nhóm này ở cùng budget.

Vùng cuối gồm cả prompt và output tùy giai đoạn sinh. Token output mới cần được xét ngay sau khi commit, không chờ tới dense refresh. Toàn bộ block 16 vẫn dùng semantics gốc: 1 anchor + 15 mask/proposal positions.

## 3. V1: tái sử dụng attention với dense refresh

### 3.1. Khởi tạo bằng một lượt draft đầy đủ

Lượt draft đầu đọc toàn context như DFlash gốc. Thu mean post-softmax attention riêng từng draft layer, qua heads và **15 proposal queries**, giữ block trong key scope nhưng loại anchor query khỏi mean.

Trên mỗi layer, lấy ranking của cache key sau khi loại đúng 16 block vị trí. Ranking này được tạo bằng các Q/K đã train của DFlash, không phải attention target hoặc score từ một network mới.

Lượt bootstrap vẫn sinh proposal và được target verify bình thường; không chạy thêm một dense forward rồi bỏ output chỉ để lấy ranking.

### 3.2. Các lượt dùng working set

Tại lượt `t`, gọi `G_t` là nhóm bảo vệ đầu/cuối hiện tại và `rank_l` là ranking cache của layer `l` từ dense refresh gần nhất:

```text
S_l,t = G_t ∪ first(K - |G_t|, rank_l excluding G_t)
```

Chỉ chọn key có position đã commit. Bổ sung đủ key nếu cần và số cache key cho phép. Các token mới chưa có trong ranking cũ được vào nhóm gần block; dense refresh kế tiếp sẽ chấm chúng cùng toàn bộ cache.

Mỗi layer đọc raw K/V của `S_l,t` cộng đầy đủ block hiện tại. Giữ absolute positions/RoPE gốc của selected keys. Một tập chung dựa trên mean năm layer là variant đơn giản để ablate; chọn riêng layer có thể hợp nhu cầu hơn nhưng chưa có bằng chứng chất lượng sau can thiệp.

Attention của block ở lượt trước không được chuyển nguyên thành score cho output vừa commit: output context K/V được tạo lại từ target hidden features, khác K/V mask/anchor tạm của drafter.

### 3.3. Đọc lại toàn context để cập nhật

Thử refresh mỗi 2 hoặc 4 lượt, với dense draft đầu đã được tính vào chi phí. Ở lượt refresh, cả năm layer đọc lại toàn context, sinh proposal thực và thay ranking theo attention mới.

Không suy overlap liền kề khoảng 83% thành lịch refresh đã bảo đảm chất lượng. Tập ưu tiên giữa lượt giữa/cuối thay đổi nhiều; refresh và budget phải được quyết định bằng can thiệp/rollout.

Một policy phản hồi có thể refresh sớm và tăng `K` khi số token thực tế commit giảm trong một cửa sổ ngắn. Chỉ dùng thông tin verification đã có, để quyết định **lượt sau**. Bắt đầu bằng lịch/budget cố định, sau đó mới so policy phản hồi để tách tác động từng thành phần.

Quan trọng: attention sparse chỉ quan sát key đang được giữ và có mẫu số softmax khác dense. Nó không phát hiện được key quan trọng đã bị loại. V1 không cập nhật ranking toàn cache bằng cách coi key không quan sát là zero; global ranking chỉ được thay khi dense refresh.

## 4. Insight và cơ chế tương ứng

| Insight | Cơ chế |
|---|---|
| Block 16 nhận mean khoảng 21–23% tổng mass | Giữ đủ block trong mọi lượt |
| Đầu/cuối thường được ưu tiên | Nhóm bảo vệ nhỏ, có ablation; giữ đúng vị trí |
| Key được ưu tiên nằm rải rác | Ranking DFlash chọn ở mọi vị trí, không chỉ một cửa sổ |
| Output đã commit nhận mass đáng kể | Cho prompt/output cùng nằm trong cache; thêm output mới vào working set gần block |
| Adjacent overlap cao hơn long-lag overlap | Tái sử dụng ngắn hạn, dense refresh định kỳ hoặc sớm |
| Năm layer có nhu cầu khác nhau | Ranking riêng layer; so với variant tập chung |
| Context dài có phần mass trải rộng | Khảo sát budget 1K/4K/8K và feedback; không giả định 1K luôn đủ |
| Parent target ranking chưa đủ cơ sở cho predictor trực tiếp | Dùng attention chính DFlash để chọn, target verification để phản hồi |

Nguồn số liệu: [hồ sơ phân tích đã hoàn tất](2026-10-06_dflash_context_memory_paper_analysis.md). Các tỷ lệ overlap và oracle coverage của hồ sơ chủ yếu đo **tập chung từ mean năm layer**; không tự động coi chúng là kết quả của policy riêng layer này.

## 5. Nhánh tiếp theo: dùng layer đầu của DFlash để chọn trong cùng lượt

Một nhánh khác tận dụng trực tiếp checkpoint là **layer 0 đọc toàn context, dùng attention của nó chọn key cho bốn layer sau**. Layer 0 vẫn thực hiện forward bình thường; việc chọn là một policy giữa các layer, không thêm predictor hoặc layer mới.

Nhánh này có lợi thế nhìn toàn context ở mọi lượt và không dùng current attention của layer chưa chạy. Tuy nhiên, chưa có phép đo chứng minh ranking layer 0 đại diện đủ cho layer 1–4. Việc các layer có nguồn target features chung không bảo đảm ranking giống nhau.

Do đó, V1 ưu tiên reuse + refresh vì có bằng chứng locality trực tiếp. Layer-0 routing là một ablation tiếp theo, trước hết đo cross-layer coverage trên artifact cũ, rồi đo logits/acceptance sau can thiệp. Có thể khảo sát kết hợp ranking hiện tại của layer 0 với lịch sử riêng layer, nhưng chưa khóa policy này.

## 6. Chi phí và hai điểm dễ gây kết luận sai

### 6.1. Chi phí đọc lại toàn context phải được tính

Nếu tỷ lệ lượt dense là `f`, còn các lượt khác đọc `K` key trong `N` cache key, mô hình đếm key lý tưởng có tỷ lệ đọc cache khoảng:

```text
f + (1 - f) × K/N
```

Đây không phải speedup latency. Nó chưa tính block, projection, score extraction, ranking, gather, target verification và phần còn lại của model. Thu attention qua eager hoặc tính lại QK có thể xóa lợi ích; cần production kernel/collector phù hợp sau khi có tín hiệu chất lượng.

Giữ full drafter KV làm ngân hàng để gather giúp tái chọn và refresh, nhưng chưa giảm resident KV. Variant chỉ giữ target-derived features chung rồi tái tạo draft K/V phải tính bytes của features và chi phí projection. Target KV vẫn full trong cả hai variant.

### 6.2. Giữ block không giữ nguyên tỷ trọng attention

Khi bỏ context key, softmax chuẩn hóa lại. Với Q/K giữ nguyên ở một head/query, block mass mới là `b/R`, với `b` là block mass gốc và `R` là mass gốc của toàn tập giữ. Layer sau còn có thể đổi query states.

Không thiết kế policy với yêu cầu block luôn nhận 20%. Đánh giá bằng fidelity logits/attention output, số token thực tế commit và cost/token; coverage dense là chẩn đoán. Ở 16K, oracle shared Top-4K cache + block giữ 82,66% tổng mass gốc, nhưng chưa có acceptance sau bỏ key.

## 7. Thực nghiệm nhỏ để kiểm tra cơ chế

Đề xuất bắt đầu với 3 document 16K trong cohort dev, 4K cache + block 16:

1. Trên trạng thái cố định đầu/giữa/cuối, so dense, current-DFlash oracle, previous-DFlash selection và đầu+cuối. Oracle dùng dense attention nên chỉ là đối chứng chất lượng.
2. Nếu có tín hiệu, chạy rollout cùng checkpoint với dense, đầu+cuối và reuse + refresh mỗi 2/4 lượt. Thêm random làm đối chứng nếu mở quy mô; giữ seed và budget rõ ràng.
3. Báo token thực tế commit tới EOS, logits/accepted prefix, decoding cost/token, chi phí refresh/gather và peak VRAM. So riêng tập chung/tập theo layer khi cần xác định lợi ích routing.

Lượt refresh cũng là lượt draft sản xuất; tính cả bootstrap, refresh và thu attention vào thời gian. Rollout bị đổi state so teacher nên không suy trực tiếp từ oracle coverage offline. Toàn bộ cohort hiện tại đã dùng để phát triển ý tưởng; holdout mới cần cho claim generalization.

Không có GPU run mới hoặc kết quả triển khai trong tài liệu này.

## 8. Liên hệ phương pháp đã có và định hướng paper

Giữ token gần đây cùng heavy hitters đã có trong [H2O](https://arxiv.org/abs/2306.14048). Giữ initial keys để xử lý attention sinks đã có trong [StreamingLLM](https://arxiv.org/abs/2309.17453). Vì vậy “sink + recent + Top-K” tự nó chưa phải novelty.

Phần cần nghiên cứu cho DFlash là policy theo **block proposal**, nhu cầu đa layer của pretrained drafter, refresh để tránh điểm mù do pruning và feedback từ full target verification. Nếu cơ chế cải thiện trade-off acceptance/cost trên long context, kết quả đó có thể làm nền cho paper; không gọi một policy chỉ được mô tả là đóng góp đã được chứng minh.

Định hướng hiện tại: **cải tiến cách DFlash đã train sử dụng context trong inference**, với V1 reuse + refresh; ranking layer đầu trong cùng lượt là nhánh so sánh tiếp theo.
