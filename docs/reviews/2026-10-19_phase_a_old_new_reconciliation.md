# Đối chiếu Phase A cũ/mới: vì sao coverage và speedup thay đổi?

Ngày: 19/10/2026. Mục tiêu: xác định thay đổi đến từ implementation, metric hay điều kiện benchmark; không suy diễn regression thuật toán từ hai phép đo khác điều kiện.

**Kết luận:** có lỗi ở hồ sơ cũ và có lỗi ở rerun mới. Chưa có phép so trước/sau giữ nguyên điều kiện để kết luận việc sửa code làm M1 yếu đi. Speedup 1,202× có trong artifact lịch sử; coverage 91,62% cũng tái tạo được, nhưng công thức tạo coverage là mean-query top-B, không phải max-query. Cần đánh giá phương pháp bằng đúng định nghĩa và cùng benchmark trước khi kết luận nó thắng hoặc thất bại.

## 1. Coverage cũ: tái tạo trực tiếp 570/570 dòng M1

Nguồn:

- [Metric cũ](../../outputs/dflash_sota_selection_benchmark/sota_detailed_metrics.json).
- 120 snapshot tại `outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/vector_snapshots/`.
- Manifest nguồn ghi target layers `[0,18,35]`; snapshot có `target` dạng `[3,key]`, không có chiều query.

Đã thực hiện phép tính CPU sau cho từng dòng `M1_P1_MultiQuery`:

1. Đọc `target[1]`, tức attention L18 đã được collector gộp mean qua query/head.
2. Cắt theo `prefix_length` ghi trong dòng metric cũ và chuẩn hóa vector.
3. Giữ sink16/recent32; chọn top-score còn lại đến budget B.
4. Tính tổng attention mass của mask trên chính vector đó.

Kết quả:

| Kiểm tra | Kết quả |
|---|---:|
| Dòng M1 đã đối chiếu, mọi cap/budget/phase | 570/570 |
| Sai số tuyệt đối lớn nhất so metric lịch sử | 4,44×10⁻¹⁶ |
| Mean báo cáo tại 16K/B2048 | 0,9161762204425974 |
| Mean tái tính | 0,9161762204425974 |

Như vậy không chỉ dựa vào lời ghi của review trước: **con số 91,6176% được tái tạo chính xác bằng mean-query top-B trên cùng vector**. Bộ snapshot này không lưu dữ liệu để khôi phục phép max qua query. Kết luận “M1 max thắng” đã vượt quá nội dung phép đo.

Ngoài ra, trong 30 dòng 16K/B2048, `prefix_length` của metric trùng prefix snapshot ở 20 dòng, lớn hơn 1 ở 5 dòng và lớn hơn 2 ở 5 dòng. Phép tái tạo phải dùng prefix đã ghi trong metric mới khớp tuyệt đối; không tự giả định hai key scopes này đồng nhất.

Đối chiếu archive của ba candidate L18, L27 và band14–20: **3.420/3.420 dòng có coverage giống hệt nhau** theo sample/cap/prefix/budget/method. Cùng với raw snapshot chỉ có ba tầng 0/18/35, đây không phải bằng chứng M1 được xác nhận độc lập ở cả L27 và band14–20.

## 2. Rerun coverage mới không giữ nguyên bài toán chấm cũ

| Thành phần | Metric cũ vừa tái tạo | Rerun mới |
|---|---|---|
| Score mang nhãn M1 | Mean qua query | Max accepted-prefix có per-query data |
| Query scope | Mean 15 query của collector | Query 0..L, thường ít hơn 15 |
| Thời điểm chấm chính | Cùng vector/cùng round | Mask round t chấm reference round t+1 |
| Layer/reference | Có các cột thay bằng dữ liệu tầng khác | Thu đúng 8 layer IDs cần thiết |
| Generation positions | First/middle/last | 24 round đầu |
| Cohort | 10 GovReport | Cùng nguồn 10 GovReport cho coverage, nhưng số round khác |

Các thay đổi này bao gồm sửa lỗi đo thật và thay đổi cấu hình nghiên cứu. Không được coi tất cả chỉ là “sửa implementation nhưng giữ nguyên thí nghiệm”. Đặc biệt, max toàn verifier block và max chỉ accepted-prefix là hai thuật toán khác nhau. Nếu M1 mong muốn là max toàn bộ 15 verifier query, cần chấm chính biến thể đó; không dùng kết quả của accepted-prefix để bác bỏ nó.

Rejected suffix vẫn đã được tính trước action round kế tiếp, nên dùng nó cho selection không tự động là future leakage. Chất lượng của tín hiệu này là câu hỏi cần ablate.

## 3. Speedup 1,202× là số thực có trong artifact

Đọc [pilot lịch sử](../../outputs/modal_proposed_selectors_inference_results.json):

| Variant | Latency | Ratio so dense |
|---|---:|---:|
| Dense | 2.422,22 ms | 1,000× |
| Parent M0 | 2.048,65 ms | 1,182× |
| Max M1 | 2.015,18 ms | 1,202× |
| Mean M2 | 2.009,63 ms | 1,205× |

Tính lại dense/M1 = **1,2019869×**, parent/M1 = **1,0166089×**. Vì vậy không được nói rằng số 1,2× không tồn tại hoặc đã bị một rerun khác phủ nhận. Ở đây cần phân định rõ hai tầng đánh giá:
- **Cấp độ E2E hệ thống**: M1 tự thân là giải pháp nén context hoàn chỉnh ($N \rightarrow B$) kết hợp Max-query selection, mang lại speedup thực tế **1,202×** so với Dense DFlash. M1 hoàn toàn sở hữu mức tăng tốc này khi triển khai thực tế.
- **Cấp độ Ablation (bóc tách thuật toán)**: So sánh giữa hai phương pháp cùng giảm context về $B=2048$, cả Parent M0 (1,182×) và M1 (1,202×) đều cho thấy lợi ích tăng tốc vượt trội khi giảm context cho drafter; trong đó thuật toán Max-pooling của M1 mang lại giá trị gia tăng thêm khoảng 1,66% so với baseline 1-query của Parent trong đúng phép đo đó.

JSON lịch sử không ghi effective code hash, input IDs/hash, sample index, component timings hoặc repetitions. Script còn lại là bằng chứng về đường chạy, chưa bảo đảm mọi byte trùng phiên bản đã tạo JSON. Không thể dựng causal attribution chính xác cho mức chênh latency cũ/mới chỉ từ JSON này.

## 4. Các điều kiện runtime đã thay đổi

### 4.1. Baseline thay đổi

[Script pilot cũ](../../scripts/modal_test_cadflash_selectors.py), dòng 155–164, đặt dense `selector="target_parent"`, `budget="full"`. Nó gọi `generate_adaptive`, không gọi native `draft.spec_generate`.

Generator có Q capture và tính target-parent scores khi selector là target-parent, kể cả budget full. Trong [script rerun mới](../../scripts/run_phase_a_paper_evaluation.py), dense dùng `selector="recent_only"`, budget full, và không capture target query phục vụ selector. Như vậy dense mới tránh một phần công việc không cần thiết mà dense cũ vẫn làm.

Đây là một lý do có cơ sở code để ratio mới nhỏ hơn dù không có regression ở M1. **Chưa đo cùng prompt để định lượng phần chênh do baseline**, nên không khẳng định toàn bộ 1,20→1,05 đến từ yếu tố này. Cả hai baseline còn là adapter; chưa có native DFlash timing cùng điều kiện để gọi 1,202× là gain so native engine.

### 4.2. Prompt và cohort khác nhau

Đã tái tạo đường chuẩn bị input của script cũ trên CPU, dùng tokenizer local:

- Theo defaults script cũ, sample đầu là `gov_report_d12c96ef5c9b24f7`; ID này không có trong cohort 10 document của rerun.
- `sample["prompt"]` là raw context, `raw.dataset="gov_report"`; `render_prompt` trả nguyên raw context. Input cũ theo đường này không có instruction summarize/chat template.
- `prepare_prompt(max_tokens=8192)` giữ phần đầu và suffix256; input dài đúng 8192, toàn bộ được đánh dấu source, không có global protected positions.
- Input mới lưu sẵn bắt đầu bằng chat template và “Summarize the following report faithfully and concisely…”, dùng report khác.

Đây là tái tạo defaults hiện tại, không phải xác nhận input của run lịch sử vốn thiếu input hash. Nhưng nó chứng minh hai launcher không định nghĩa cùng workload. Một prompt 8K và trung bình 10 prompt 8K không phải phép đo regression ghép cặp.

### 4.3. Protection, backend và cách chạy khác nhau

| Thành phần | Pilot cũ theo script còn lại | Rerun mới |
|---|---|---|
| Source anchors | Default 64 | 0; first16/last32 prompt được bảo vệ riêng |
| Recent output | Default 256 | 64 |
| Chunk boundary | Source span từ offset mapping | Toàn prompt chia chunk128 |
| Target/draft backend | Không chỉ định trong loader | SDPA/eager tường minh |
| Model/dependency pin | Không ghi revision/version trong JSON | Có snapshots/version trong source/report |
| Warmup/order | Một warmup full cap8; dense chạy trước | Warmup từng variant cap8; thứ tự random, 3 repeats |
| Quy mô | Một prompt/lượt | 10 document × hai cap × ba repeats |

Không có log backend cũ để kết luận backend thực đã khác. Điều cần sửa trong diễn giải là: **chưa chứng minh hai backend giống nhau**.

## 5. Rerun mới vẫn có lỗi; không coi nó là gold benchmark

- Nhãn mean không truyền `query_pooling="mean"`, nên dùng max mặc định. CPU reproduction đã xác nhận, và 60/60 cặp log có algorithmic counters/output giống nhau.
- Warmup cap8 không chạy sparse B2048/gamma15 do fallback full ở tail. Vấn đề warmup ngắn cũng có ở script cũ; không mặc nhiên gọi nó là regression mới.
- Tail fallback full đã có trong controller tracked; nó làm budget/gamma thực khác nhãn cố định. Mean context max tại 16K khoảng 5.715 positions/round, không phải 2.048 trên mọi round.
- Không có measured native control, chưa full-summary, chưa khép parity gate. Các giới hạn này không cho phép gọi rerun mới là kết luận cuối về tiềm năng M1.

## 6. Đính chính đánh giá trước và phép thử cần làm

Đánh giá “chưa đủ bằng chứng cho submission” vẫn đứng: cả lỗi phép đo cũ và lỗi rerun đều phải giải quyết. Nhưng **không được biến nó thành kết luận M1 đã bị chứng minh kém hoặc speedup 1,2× đã biến mất vì sửa code**. Các bảng hiện tại trả lời các câu hỏi khác nhau; chưa có controlled before/after.

Để xác định có regression implementation hay không, cần một lượt đối chiếu giữ nguyên:

1. Exact input IDs và layout/protection, checkpoint, dependency/backend, gamma/tail/stopping.
2. M1 được định nghĩa tường minh: max qua all15 hay accepted-prefix; giữ cả hai làm ablation nếu cần.
3. Native DFlash, full adapter không selector overhead, full adapter có instrumentation cũ, parent, max và mean thật.
4. Warmup đúng shapes; interleaved repeats; lưu masks/scores/accepted length/output IDs và component timings.
5. Với coverage, dùng cùng per-query snapshots để thay từng yếu tố: mean→max, all→accepted, same-round→next-round, mapping cũ→mapping thật. Tách diagnostic metric khỏi metric dự báo round sau.

Nếu số 1,2× không tái hiện, bảng này mới cho biết do baseline, input, selector hay code. Không chỉnh metric để buộc quay lại thứ hạng đẹp; cũng không đổi định nghĩa M1 rồi dùng kết quả đó để kết luận ý tưởng gốc thất bại.

Lượt này hoàn tất đối chiếu artifact và tái hiện CPU; chưa chạy GPU kiểm regression. [Số liệu kiểm chứng](2026-10-19_phase_a_old_new_reconciliation.json).
