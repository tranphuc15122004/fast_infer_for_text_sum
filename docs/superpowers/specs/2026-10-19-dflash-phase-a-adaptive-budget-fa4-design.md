# Đề xuất Phase A: adaptive context budget để tối ưu speedup FA4

Ngày: 19/10/2026. Trạng thái: **được chọn để triển khai trên FA4**.

## 1. Mục tiêu và phạm vi

Chọn context budget theo thời gian và kết quả của các round gần đây để giảm
thời gian sinh thực tế. Bản đầu chỉ thay budget của Phase A trên FA4 native:
giữ selector M1/L18, chunk 128, sink 16, recent 32 và gamma 15. Target luôn
verify trên full context. vLLM và adaptive verification length nằm ngoài phạm vi.

Mục tiêu cuối là latency của toàn request ở cùng output và chất lượng.
Không mặc định budget lớn làm acceptance tốt hơn: context focusing có thể
khiến budget nhỏ vừa nhanh hơn vừa có acceptance tốt hơn.

## 2. Căn cứ trong repository

- `src/Benchmark/common/phase_a_selector.py`: hiện có selector với budget cố định.
- `externals/dflash/dflash/model.py`: đã có acceptance mỗi round và CUDA events
  cho draft/verify; hiện chỉ tổng hợp timing khi kết thúc generation.
- `src/Benchmark/native_fa4_benchmark.py`: đo wall time đồng bộ ở biên request,
  có đối chứng vanilla và dense DFlash.
- `src/TrainingFree/context_adaptive/controller.py`: đã có nguyên tắc
  cost/expected-commit và chế độ `a_only`. Có thể dùng lại ý tưởng hoặc phần
  thống kê thuần Python; không kéo toàn bộ joint controller/engine vào FA4.
- `src/TrainingFree/policy.py::adaptive_budget`: dùng entropy relevance;
  không phải controller timing mà đề xuất này cần.

## 3. Lựa chọn cách điều khiển

| Cách | Ưu điểm | Hạn chế | Vai trò |
|---|---|---|---|
| Đặt ngưỡng draft ms rồi tăng/giảm budget | Đơn giản | Không tính số lần verify; có thể tối ưu sai mục tiêu | Ablation |
| Chọn budget theo attention coverage/entropy | Có tín hiệu ngay | Coverage không phải acceptance hay speedup | Ablation/khởi tạo sau này |
| Ước lượng ms/token từ timing và acceptance gần đây | Trực tiếp cân bằng chi phí với tiến độ sinh | Cần prior hoặc exploration; observation nhiễu | **Đề xuất V1** |

V1 dùng thống kê nhỏ với cửa sổ suy giảm và thử budget lân cận có giới hạn.
Không cần mô hình neural hay chạy nhiều draft song song để chọn một action.

## 4. Hàm mục tiêu

Trong round t, gọi L_t là số accepted candidates, không gồm parent token.
Tiến độ prefix bình thường là C_t = 1 + L_t. EOS/output cap cần hạch toán riêng.

Với state s_t và budget B, dùng:

```text
J_t(B) = E[T_round | s_t, B] / E[C | s_t, B]        # ms/token
B_next = argmin_B J_t(B)
```

T_round bao gồm controller, signal reconstruction, selection, KV gather,
draft, verify và bookkeeping. T_draft của các round gần đây là thành phần
chính để cập nhật chi phí theo B; T_verify phụ thuộc context length và gamma,
không được giả định target context thu nhỏ theo B.

Ước lượng trực tiếp T_round là tín hiệu chọn action chính. CUDA component
timings dùng để phân rã và chẩn đoán T_draft(B), không cộng thêm một lần vào
T_round. Nếu về sau dùng mô hình cộng component thay thế, phải chứng minh
accounting đủ và không chồng lấn trước khi bật.

Ước lượng trên một cửa sổ dùng tỷ số tổng thời gian / tổng commit có cùng
trọng số. Không lấy trung bình các T_round/C_t của từng round: cách đó không
khớp throughput của cả cửa sổ.

Đây là surrogate online, không phải bảo đảm optimum toàn cục hoặc speedup.
Kết luận cuối phải dùng wall time của toàn generation, tính cả bootstrap,
exploration, tail và controller overhead.

## 5. Không gian budget và selector

Khoảng budget khởi đầu: từ 512 đến 8192 token, bắt đầu ở 2048 và di chuyển
theo các nấc chunk 128 token. Đây là giới hạn đề xuất để pilot trên B200,
không phải budget tối ưu đã được xác nhận.

- Budget là trần context keys, không gồm block parent + 15 draft candidates.
- B >= logical context length được canonicalize thành `full`, loại action trùng.
- Giữ nguyên quy tắc anchors/chunks cho cả fixed và adaptive controls.
- Log cả `requested_budget`, `selected_context_tokens` và `unused_budget`.
  Selector hiện có thể chỉ chọn 1968 token ở cap 2048; không gọi đó là 2048
  token thực đọc. Không sửa packing riêng cho adaptive rồi so với fixed cũ.
- Full action bỏ qua reconstruction/ranking/gather nếu không cần signal ở round đó;
  chi phí quản lý signal cho những lần quay lại sparse vẫn phải được tính.
- Không ép quan hệ đơn điệu giữa budget và acceptance, hoặc giữa budget và latency.

## 6. Thống kê online và khởi tạo

V1 giữ cho mỗi budget tổng trọng số, round-ms, số commit và component draft /
verify để chẩn đoán. Sau mỗi observation hợp lệ, thống kê mọi budget được giảm
trọng số `decay=0.9`; action vừa chạy được cộng thêm một observation. Chi phí
được tính bằng tổng round-ms chia tổng commit, không lấy trung bình các tỷ số
từng round.

Triển khai V1 cold-start ở 2048 và chỉ học trong request hiện tại. Không có
prior đã calibration, context bucket hay chia sẻ observation giữa warmup và
request đo. Round draft đầu sau prefill được ghi là bootstrap nhưng không đưa
vào thống kê: action đầu được chuẩn bị trong prefill, không có cùng boundary
đo với những action tiếp theo. Không suy outcome cho budget chưa chạy. Request
ngắn có thể không đủ round để học hữu ích.

## 7. Quy trình chọn budget

1. Prefill full target; dùng query cuối prompt để tạo signal cho draft đầu.
2. Khởi tạo budget ở 2048 trong V1 cold-start; không dùng prior chưa được calibration.
3. Mỗi round thực thi đúng một draft với budget đã chọn rồi verify bình thường.
4. Ghi observation theo action thực chạy; nhận timing GPU trễ nếu chưa ready.
5. Sau 3 round ở cùng budget, đánh giá chi phí trên token commit.
6. Nếu budget thử thắng incumbent ít nhất 3%, nhận nó làm incumbent và nhân
   đôi bước tìm kiếm tới trần 1024 token.
7. Nếu budget thử không thắng, quay chiều và giảm bước một nửa, tối thiểu
   128 token. Nếu hai phía ở bước tối thiểu đều không thắng, giữ incumbent.
8. Sau 12 round giữ incumbent, thử lại một lân cận để thích ứng khi decode
   tiếp tục và đặc tính các round đổi.
9. Mọi lần thử là round sinh thật và được tính vào latency cuối request.
10. Khi remaining output slots <= gamma+1, ngừng exploration, dùng action
    incumbent; nếu request kết thúc thì không chọn action tiếp theo.

Các số 3/12/3% là giá trị khởi đầu; khóa trên dev trước held-out test. Với
request ngắn, có thể tắt exploration để tránh mất nhiều round cho việc học.
Không thể phân biệt chắc chắn đổi độ khó văn bản với tác động budget chỉ bằng
online samples; dùng repeated trials và same-prefix replay khi đánh giá.

Nếu thiếu query hoặc target key signal, draft dùng full context và observation
budget của round đó bị loại khỏi controller; trace ghi rõ lý do. Cấu hình không
hợp lệ fail-fast khi tạo selector, tránh âm thầm báo một action khác với action
thực thi.

## 8. Đo timing không tạo thêm hàng rào đồng bộ

Đặt boundary wall time tại điểm acceptance scalar đã sẵn có của vòng lặp.
Interval giữa hai boundary bao gồm chọn action, selector, draft và verify của
round sau. Khi tích hợp, chuyển selection cho next draft vào đầu round tương
ứng để budget, signal và timing cùng một action; vẫn dùng Q/K từ verify trước.
Bookkeeping ở biên phải được tính đúng một lần. Bootstrap/tail có nhãn riêng.

Dùng `perf_counter` để đo wall interval tại boundary đang cần cho `.item()`;
không thêm `torch.cuda.synchronize()` mỗi round. Đây là latency quan sát được
của pipeline Python hiện tại, có thể chứa CPU launch gaps/scheduling noise.

Dùng pool CUDA events cho component timings. Chỉ gọi `elapsed_time` khi
end-event `query()` báo hoàn tất; chưa ready thì giữ observation pending và
tiếp tục với thống kê trước đó. Không busy-wait, không gọi event synchronize
chỉ để controller ra quyết định. Timing nhận trễ phải gắn đúng round/action.
Cuối request thu nốt events tại synchronization boundary vốn có.

Theo tài liệu PyTorch, `query()` kiểm tra hoàn tất, `elapsed_time()` trả thời
gian giữa events, còn `synchronize()` chặn CPU chờ hoàn tất:
https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html

Bản đầu không tính toàn bộ entropy vocabulary hay chuyển vector attention
về CPU để điều khiển budget. Controller làm việc trên vài scalar/statistics.

## 9. Điều kiện correctness trước khi học budget

- Prefill dùng last prompt query; verify dùng đúng parent + accepted prefix.
- Reconstructed attention có đúng Q-norm, RoPE, GQA, scale và causal positions.
- Target KV và full draft bank vẫn giữ logical positions để budget có thể tăng lại.
- Không crop vĩnh viễn bank xuống B; budget chỉ đổi tập KV đọc trong draft attention.
- Sửa signal/correctness chung cho cả fixed và adaptive controls trước khi so sánh.
- Giữ nguyên gamma=15 và verifier để quy kết được tác động của budget.

Không dùng kết quả coverage để giả lập counterfactual acceptance của budget khác.
Muốn biết outcome budget khác ở cùng prefix phải thực sự draft+verify action đó
trong replay offline; không tính replay đó là online speedup.

## 10. Quy trình thực nghiệm để biết adaptive có đáng làm không

1. **Profiling FA4:** full và từng fixed budget trên cohort calibration riêng;
   warmup tất cả shapes thật. Đo toàn draft, selector/gather và round wall time.
2. **Sweep fixed budgets:** cùng prompt IDs, gamma, max output/EOS; xác định
   best fixed trên dev. Nếu latency draft gần phẳng theo B hoặc acceptance
   không bù overhead, ghi nhận và ưu tiên tối ưu hot path trước.
3. **Same-prefix replay:** ở prefix giữ cố định, đo các budget để xác định
   optimum có đổi theo round hay chỉ theo document/context length.
4. **Online rollout:** dense, fixed-2048, best-fixed, adaptive-frozen-prior,
   adaptive-timing+acceptance; thêm draft-ms-only làm ablation.
5. **Held-out test:** giữ nguyên bounds/step/hyperparameters đã khóa trên dev; sinh
   summary đủ dài tới EOS hoặc cap thống nhất. Tính cả exploration và bootstrap.

Báo cáo paired request wall speedup so dense, best fixed và vanilla; kèm TPOT,
draft/verify/selector/controller time, C_t, budget trajectory và p95 latency.
Lặp nhiều lần, hoán đổi thứ tự variants và tính khoảng tin cậy theo document,
không coi nhiều round hoặc timing repetitions là document độc lập.

Điều kiện nhận làm mặc định: output correctness/quality đạt gate và adaptive
cải thiện paired E2E latency so **best fixed**, không chỉ fixed-2048. Nếu chỉ
tốt ở một context bin thì giới hạn phạm vi kết luận. Khởi đầu đặt mục tiêu
controller <1% thời gian generation; đây là budget overhead cần đo, không
phải hiệu năng đã được chứng minh.

## 11. Bề mặt triển khai dự kiến

- Thêm controller/statistics/timing adapter nhỏ dưới `src/Benchmark/common/`.
- Cho selector nhận budget của round, giữ default fixed mode để đối chứng.
- Nối choose/observe và timing vào `dflash_generate`; scope patch dưới method
  FA4 adaptive, không tác động vLLM hoặc Phase B.
- Bổ sung runner options cho fixed/adaptive, prior path, grid và round trace.
- Output JSONL qua writer chung, có summary, runtime signature và policy ID;
  trace được buffer trong RAM, không ghi từng round vào file trong hot loop.
- Tests CPU cho ratio-of-sums, action-specific updates, decay theo round age,
  hysteresis, pending events, reset, grid canonicalization và tăng budget trở lại.
- GPU gate cho FA4 dispatch, parity, overhead và full-summary paired latency.

Không triển khai adaptive controller hoặc chạy GPU trong bước viết đề xuất này.

## 12. Quyết định triển khai sau khi chốt budget theo chunk

Thiết kế triển khai giữ `dflash_phase_a` ở fixed mode theo mặc định và bật
controller bằng `DFLASH_PHASE_A_BUDGET_MODE=adaptive`. Search không quét một
grid budget cố định. Nó di chuyển trên budget nguyên theo bước tối thiểu
`chunk_size=128`, bắt đầu từ 2048, giới hạn mặc định 512–8192.

Controller giữ số đo theo budget đã thực sự chạy. Sau mỗi 3 round cùng budget,
nó so sánh tỷ số thời gian round có suy giảm theo thời gian trên số token đã
commit. Khi budget thử mới thắng quá 3%, nó trở thành incumbent và bước tìm
kiếm tăng gấp đôi tới trần 1024 token. Khi không thắng, controller quay chiều
và giảm bước tới tối thiểu 128. Nếu cả hai lân cận ở bước 128 đều không thắng,
controller giữ incumbent; sau 12 round nó thử lại để bắt thay đổi hành vi.

`round_ms` dùng để quyết định gồm thời gian thực hiện draft + verify đo bằng
đồng hồ tường giữa hai mốc round, cộng overhead selector/controller trước đó
đã được quy về budget hiện tại. Draft/verify CUDA event time được log riêng.
Overhead selector/controller được đo và gắn vào quan sát budget kế tiếp. E2E
đồng bộ của cả request vẫn là thước đo cuối cùng, gồm cả bootstrap, exploration
và overhead tìm kiếm.

Per-request state reset ở đầu mỗi generation. Khi request kết thúc, controller
không chọn budget tiếp theo; trace vẫn giữ observation cuối nếu có boundary đo
hợp lệ. Adaptive trace được xuất cùng từng record sample để kiểm tra trajectory
và từng quyết định.
