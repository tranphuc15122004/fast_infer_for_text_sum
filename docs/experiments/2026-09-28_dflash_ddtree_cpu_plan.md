# Kế hoạch thử nghiệm DFlash DDTree trên CPU

> Trạng thái: đã triển khai; hoàn tất sàng lọc khả thi CPU ban đầu; dừng biến thể cây do không đạt ngưỡng độ trễ đặt trước.

**Câu hỏi:** Cây draft có ngân sách cố định, dựng từ một lượt DFlash, có giảm độ trễ giải mã greedy chính xác so với DFlash chain trên prompt tóm tắt chạy bằng CPU local không?

**Giả thuyết:** Với ngân sách xác minh 16 node, các nhánh DFlash có xác suất cao giúp khôi phục những token target đầu tiên mà nhánh top-1 duy nhất bỏ lỡ, giảm đủ số vòng target để vượt cấu hình chain tốt nhất đã thử.

**Baseline:** `Qwen/Qwen3-4B` + `z-lab/Qwen3-4B-DFlash-b16` đã cache; giải mã greedy target-only và DFlash chain với `block_size=4,16`.

**Biến thể:** Giữ nguyên target và draft model; dựng cây tiền tố top-16 từ phân phối DFlash theo từng vị trí, dùng heap để mở rộng theo thứ tự ưu tiên, target attention chỉ nhìn node tổ tiên, đi cây theo target một cách chính xác và thu gọn KV theo đường đi. Không cập nhật tham số.

**Chỉ số chính:** Thời gian sinh end-to-end trên CPU theo cặp, cùng prompt và ngân sách token sinh. Báo cáo mức tăng tốc theo cặp so với từng baseline chain.

**Điều kiện bắt buộc:** Token greedy phải khớp target-only 100%; quy tắc độ dài đầu ra và EOS phải giống nhau; thời gian end-to-end tính cả dựng cây và thu gọn cache; báo cáo số token được chấp nhận mỗi lần gọi target và RSS cực đại.

**Dữ liệu / chia tập:** Dòng đầu tiên xác định cố định của mỗi tập GovReport, MultiNews, QMSum từ `data/longbench_100_14k/`; đầu vào tối đa 512 token, đầu ra tối đa 32 token cho vòng sàng lọc ban đầu. Đây là kiểm tra cơ chế và độ trễ, không phải kết luận về chất lượng hay thống kê. Chỉ mở rộng nếu runtime cho phép.

**Seed:** Giải mã greedy; cố định seed torch bằng 42 để tái lập.

**Ngân sách:** Chỉ CPU local, 20 luồng, vòng sàng lọc ban đầu tối đa 30 phút. Không dùng CUDA hoặc tải dữ liệu qua mạng.

**Thứ tự chạy:** Kiểm tra đơn vị cho phần dựng cây/mask/đi cây, sau đó chạy một mẫu Qwen3-4B thật và so khớp target; cuối cùng đo thời gian theo cặp trên ba dataset.

**Tiêu chí thành công:** Đầu ra chính xác trên mọi dòng và cây nhanh hơn ít nhất 10% so với chain baseline nhanh nhất trên ít nhất hai dataset. Cải thiện nhỏ hơn hoặc chỉ xuất hiện trên một dataset là chưa đủ kết luận; nếu chậm hơn thì dừng.

**Chỉ mang tính thăm dò:** ROUGE trên ba mẫu, trace thứ hạng ứng viên và mọi suy rộng từ CPU sang GPU.

## Tệp dự kiến

- `scripts/dflash_tree_cpu.py`: nạp model CPU, tạo draft DFlash, dựng cây DDTree, xác minh cây, thu gọn KV theo đường đi và xuất kết quả benchmark theo cặp.
- `tests/test_dflash_tree_cpu.py`: kiểm tra dựng cây, mask và đi cây với tensor nhỏ xác định.
- `outputs/dflash_ddtree_cpu/`: JSONL thô, manifest lượt chạy và tổng hợp.
- `docs/experiments/2026-09-28_dflash_ddtree_cpu_report.md`: kết quả đo và giới hạn.

## Trình tự triển khai

1. Thêm kiểm tra hợp đồng cho phần dựng cây và xác minh; chạy để thấy kiểm tra thất bại trước khi cài đặt.
2. Cài đặt hàm tiện ích cho cây và làm cho các kiểm tra tương ứng đạt.
3. Cài runner CPU dùng các snapshot đã cache; chạy smoke một ví dụ và so với target-only greedy.
4. Chạy vòng sàng lọc cố định theo cặp trên ba dataset, tính toàn bộ thời gian vào E2E.
5. Ghi lại độ chính xác, độ trễ, thời gian dựng cây/xác minh, số token được chấp nhận và quyết định dừng.

## Kết quả

Xem [`2026-09-28_dflash_ddtree_cpu_report.md`](2026-09-28_dflash_ddtree_cpu_report.md). Phần triển khai đạt các kiểm tra hợp đồng và smoke test khớp đầu ra. Trong vòng sàng lọc 32 token theo cặp, cây 16 node chậm hơn DFlash chain block size 4 trên cả ba prompt. Dừng tối ưu CPU cho biến thể này; chỉ giữ phần triển khai làm prototype kiểm tra tính đúng đắn.
