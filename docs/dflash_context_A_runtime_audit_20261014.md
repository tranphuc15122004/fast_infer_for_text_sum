# Rà soát runtime campaign DFlash Context A — 14/10/2026

## Phạm vi và giới hạn

Rà soát launcher `start/check/resume`, nguồn dữ liệu và exposure, validate,
replay A1/A2/A3, rollout A3, checkpoint, dev lock, partial confirm và report.
Máy phát triển chỉ chạy CPU; chưa chạy một campaign full bằng Qwen3-4B,
DFlash-b16 và FlashAttention-4 trên B200 sau bản sửa này. Kết quả test dưới
đây không phải bằng chứng về parity, latency hoặc độ ổn định kernel FA4 trên
B200. Bộ dữ liệu local đã đối chiếu khớp hash ba dataset trong audit đi kèm.

## Các lỗi đã tìm và sửa

| Vấn đề | Hậu quả trước bản sửa | Cách xử lý |
| --- | --- | --- |
| Loader gán `DATA_FILE=DATA_INPUT` dù nguồn là thư mục | Preflight báo hai nguồn dữ liệu | Chuẩn hóa nguồn cùng đường dẫn, nhận diện loại file/thư mục; CLI chọn nguồn đầy đủ |
| Master có thể chọn XSum/representative | Campaign A thiếu cả ba dataset bắt buộc | Launcher A truyền corpus LongBench đã audit cho mọi phase, có override riêng `DCE_DATA_DIR` |
| A2/A3 không cô lập exception ở từng cell | Một selector lỗi có thể bỏ cả prefix | Mỗi cell thử tối đa hai lần; lỗi được lưu rồi chạy cell tiếp theo; lỗi cache/GPU nghiêm trọng vẫn dừng |
| Đếm ba prefix lỗi thành lỗi hệ thống | Một sample có nhiều prefix lỗi làm dừng campaign | Đếm sample thất bại hoàn toàn; ngưỡng ba sample liên tiếp vẫn được giữ |
| Chỉ kiểm tra số hữu hạn trong parity, bỏ qua trace round | NaN/Inf làm lỗi lưu JSON ngoài vùng retry | Kiểm tra cả adapter round trước khi lưu reference; sample lỗi được retry/bỏ qua |
| Chữ ký cell thiếu số prefix, warmup, repetition và seed | Resume có thể trộn protocol đo khác nhau | Bổ sung các tham số vào chữ ký; từ chối resume nếu thay đổi protocol |
| So khớp process bằng chuỗi có RUN_ID làm tiền tố | Nhận nhầm worker, cản start/resume; lỗi khi file PID mất | Đọc argv phân cách NUL, so khớp toàn bộ RUN_ID; vẫn chặn worker mồ côi, kể cả thiếu file PID |
| Confirm chưa kiểm tra lại ba hash dev đã khóa | Artifact dev sửa/thiếu sau lock không được phát hiện trước GPU | Kiểm tra hash parity, rollout và cohort dev trước khi mở confirm |
| Retry/resume A3 bỏ warmup ở lượt đo còn thiếu | Timing có thể chứa chi phí kernel chưa warmup, ảnh hưởng gate latency | Warmup đầy đủ trước lượt đo đầu thành công của cấu hình trong tiến trình hiện tại; retry giữ warmup, resume không coi checkpoint cũ là warmup mới |
| Kiểm tra finite attention bằng `.item()` từng token | Tạo số lần đồng bộ host/device tăng theo context | Kiểm tra toàn vector một lần; giữ nguyên công thức attention và cách phát hiện NaN/Inf |

## Kiểm tra thực hiện

- Regression tái hiện lỗi trước khi sửa: exception cell A2/A3, mất warmup,
  RUN_ID làm tiền tố, NaN trace round, protocol signature, artifact dev đổi
  sau lock và nhiều prefix lỗi thuộc một sample.
- Chạy trực tiếp `start` với tiến trình CPU thay cho phase model: kiểm tra đủ
  tám phase, `campaign.status`, EXIT trap, khóa `flock` được process nền giữ,
  từ chối resume trùng/đã hoàn tất, bảo toàn exit code 3 và 130.
- Chạy thật `_prepare`, `_run_phase`, checkpoint/merge, thống kê/report,
  `_lock` và confirm với corpus đã audit; các GPU forward được thay bằng
  kết quả mô phỏng có kiểm soát. Kiểm tra cohort dev 10/10/4, confirm 18/18/3,
  mọi arm/repetition và lần resume không tính lại cell thành công.
- Model nhỏ chạy thật trên CPU: Qwen3 36 layer và draft 5 layer, mapping
  `[1,9,17,25,33]`, attention eager. Dựng replay state, thử Full → mask giảm →
  Full, kiểm tra proposal/acceptance lặp lại và cache/bank prefix bất biến.
- Kiểm tra cú pháp Bash, compile Python, `git diff --check`; phân tích 12
  module với 213 lời gọi hàm trực tiếp, không thấy keyword không được hỗ trợ
  hoặc key literal trùng trong dict tại thời điểm rà soát.

Lệnh kiểm tra local cho campaign A và dependency DFlash:

Lượt chạy cuối: **156 pass**, hai warning về driver/NVML của máy CPU local.
Không có test fail trong tập kiểm tra A/DFlash này.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_dflash_context_campaign.py \
  tests/test_dflash_context_data_paths.py \
  tests/test_dflash_context_flow.py \
  tests/test_dflash_context_launcher.py \
  tests/test_dflash_context_replay_cpu.py \
  tests/test_context_adaptive_dflash_regressions.py \
  tests/test_dflash_attention_probe.py \
  tests/test_dflash_wrapper_contract.py \
  src/TrainingFree/tests/test_context_interventions.py \
  src/TrainingFree/tests/test_context_interventions_evaluation.py \
  src/TrainingFree/tests/test_context_interventions_execution.py \
  src/TrainingFree/tests/test_context_interventions_report.py \
  src/TrainingFree/tests/test_context_interventions_resilience.py
```

## Ba test quy ước chung còn lỗi ngoài luồng A

Lượt chạy mở rộng thêm các test config/runtime chung trả về 182 pass và ba
fail trước các bổ sung test CPU/warmup cuối cùng:

1. `test_main_runners_source_shared_runtime`: kiểm tra nội dung wrapper
   `run_length_bins_benchmark.sh`.
2. `test_main_runners_do_not_bypass_shared_interpreter`: kiểm tra nội dung
   wrapper `run_finetuning_phase1.sh`.
3. `test_scripts_root_keeps_only_primary_entrypoints`: danh sách kỳ vọng chỉ
   có năm entrypoint, trong khi HEAD đã có mười file `run_*.sh`, gồm A và B.

Hai launcher và hai file test này đều đã đối chiếu không thay đổi so với
HEAD. Chưa sửa chúng trong lần rà soát A này. Không diễn giải kết quả riêng
campaign A thành toàn bộ test repository đã qua.

## Cập nhật và chạy trên server

Bản vá thay đổi chữ ký code/checkpoint. Dùng `start` tạo RUN_ID mới sau khi
cập nhật; giữ nguyên code/runtime/model/data khi resume cùng phiên bản.

```bash
tar -xzf dflash_context_A_b200_audit_fix_20261014.tar.gz
sha256sum -c dflash_context_A_b200_audit_fix.sha256
bash scripts/run_dflash_context_A_b200.sh check &&
bash scripts/run_dflash_context_A_b200.sh start
```

Lỗi cục bộ hết retry được ghi lại và loại khỏi thống kê hợp lệ. Parity thật
không khớp, cache/GPU mất trạng thái, artifact/provenance hỏng, không lưu
được kết quả hoặc dev không đạt gate vẫn có thể dừng campaign. Chỉ log
`campaign complete` của lượt B200 thực tế xác nhận lượt đó hoàn tất.
