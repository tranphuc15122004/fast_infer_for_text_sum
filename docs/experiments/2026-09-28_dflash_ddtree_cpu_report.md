# Cây draft DFlash kiểu DDTree — kết quả khả thi trên CPU

## Quyết định

Dừng hướng này như một phương pháp tăng tốc trên CPU local. Kết quả giải mã greedy khớp tuyệt đối trên các mẫu đã chạy và nhanh hơn DFlash chain với block size 16, nhưng chậm hơn baseline block size 4 trên cả ba dataset. Tiêu chí đặt trước yêu cầu nhanh hơn ít nhất 10% so với chain nhanh nhất trên ít nhất hai dataset; kết quả không đạt.

## Phương pháp và thiết lập

Phần triển khai tạo cây tiền tố theo thứ tự ưu tiên từ một lượt draft DFlash, giới hạn 16 node và giữ 16 ứng viên đầu mỗi vị trí draft. Qwen3-4B xác minh cây bằng attention chỉ nhìn các node tổ tiên; KV và hidden state của đường đi được chấp nhận được thu gọn để dùng ở vòng kế tiếp. Không huấn luyện hay cập nhật tham số.

- Mô hình đích: `Qwen/Qwen3-4B` đã cache; draft model: `z-lab/Qwen3-4B-DFlash-b16` đã cache.
- Runtime: CPU local, PyTorch float32, SDPA, 20 luồng; không dùng CUDA.
- Đầu vào: dòng đầu tiên của GovReport, MultiNews và QMSum trong `data/longbench_100_14k/`, cắt còn tối đa 512 token.
- Giới hạn đầu ra: 32 token. Mỗi dataset chạy một prompt; đây là bước sàng lọc tính khả thi, không phải benchmark đủ mạnh về thống kê.
- Các cấu hình so sánh: target greedy, DFlash chain block size 4 và 16, cây draft 16 node.
- Thời gian end-to-end bao gồm tạo draft, dựng cây, xác minh bằng target và thu gọn cache. Mọi cấu hình đều sinh 32 token và khớp tuyệt đối với target greedy trên từng prompt.
- RAM cư trú cực đại: 32.949 GiB.

Lệnh chạy:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
  .venv/bin/python scripts/dflash_tree_cpu.py \
  --datasets gov_report,multi_news,qmsum --samples-per-dataset 1 \
  --max-input-tokens 512 --max-new-tokens 32 --node-budget 16 --top-k 16 \
  --threads 20 --output outputs/dflash_ddtree_cpu/screen32.jsonl
```

## Kết quả theo cặp

| Dataset | Target greedy (ms) | DFlash chain-4 (ms) | DFlash chain-16 (ms) | Tree-16 (ms) | Tree so với chain-4 |
|---|---:|---:|---:|---:|---:|
| GovReport | 34,991.951 | **26,446.988** | 35,781.248 | 30,824.073 | 16.5% chậm hơn |
| MultiNews | **25,889.054** | **24,906.777** | 33,437.162 | 32,719.396 | 31.4% chậm hơn |
| QMSum | **27,868.080** | 34,143.490 | 52,749.728 | 34,902.982 | 2.2% chậm hơn |

Cây nhanh hơn chain-16 lần lượt 13.9%, 2.1% và 33.8% trên GovReport, MultiNews và QMSum. Tuy vậy, so với cấu hình chain tốt nhất cho từng prompt, cây chậm hơn trên cả ba dataset. Độ trễ trung bình của ba prompt là 32.82 giây với tree-16 và 28.50 giây với chain-4, tức chậm hơn 15.1%.

Cây cần 8, 9 và 11 vòng xác minh cho ba prompt. Riêng việc dựng cây mất 67–140 ms và thu gọn cache mất 482–840 ms; chi phí chính là xác minh 16 nhánh mỗi vòng trên cấu hình CPU/SDPA này. Số token được chấp nhận nhiều hơn trong mỗi vòng không bù được chi phí đó.

## Kiểm tra và giới hạn

- Lệnh `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/test_dflash_tree_cpu.py -q`: cả 3 kiểm tra đều đạt.
- Smoke test với model thật và đầu ra 4 token cũng khớp tuyệt đối với target greedy, nhưng giải mã bằng cây vẫn chậm hơn.
- Kết quả 32 token chỉ có một prompt cho mỗi dataset, nên dùng để quyết định dừng thử nghiệm chứ không đại diện cho ước lượng ổn định trên toàn bộ dữ liệu. Token đầu ra giống hệt đảm bảo giữ nguyên nội dung sinh từ target trong các mẫu này; ROUGE không bổ sung bằng chứng chất lượng ở đây.
- Thử nghiệm chỉ đưa ra kết luận cho phần triển khai CPU và backend SDPA này. Kết quả không dự đoán được hiệu năng GPU, nơi kernel xác minh cây và batching có chi phí khác.

Bản ghi thô và manifest nằm tại `outputs/dflash_ddtree_cpu/screen32.jsonl` và `outputs/dflash_ddtree_cpu/screen32.manifest.json` (artifact benchmark được gitignore).
