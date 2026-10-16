# Kế hoạch triển khai thực nghiệm vLLM

**Mục tiêu:** giảm generation lặp và dùng vLLM 0.30 theo requirements server,
giữ semantics A/B và công khai coverage/capability thực tế.

**Thiết kế:** [đặc tả](../specs/2026-10-15-vllm-research-execution-design.md).
**Runtime:** Python 3.12; manifest server hiện hành; không cài thêm online.
**Thực thi:** trong phiên hiện tại; runner và adapter có thể phát triển ở các
file độc lập, tích hợp và review sau. Không sửa thay đổi có sẵn của người dùng
trong `scripts/run_dflash_speculation_B.py`.

## Công việc

- [x] Runner native: thêm measurement latency/throughput, batch submission,
  bounded warmup và diagnostic report-only vào `src/Benchmark/vllm_all_baselines.py`.
  Viết regression CPU trước; giữ CLI cũ hoạt động và schema có summary.
- [x] Adapter research 0.30: xác minh source trong venv, tách compatibility
  và worker hooks trong `src/Benchmark/vllm_research/`; test semantics CPU
  trước. Không khai báo capability dựa vào version string đơn thuần.
- [x] Bộ điều phối: CLI `scripts/infer_vllm_research.py`, wrapper trong
  `scripts/runners/`, dispatcher, plan/capability/coverage và resume;
  dùng chung master/runtime; lỗi từng job có terminal record và campaign tiếp tục.
- [x] Tích hợp A/B: context/replay/proposal hooks theo các capability thực
  sự triển khai; kết nối analysis CPU; không im lặng bỏ cell chưa hỗ trợ.
- [x] Docs/runbook: lệnh full đơn giản, default rõ, giới hạn GPU verification,
  thứ tự đồng bộ source/offline server và output artifact.
- [ ] Kiểm chứng cuối: pytest liên quan, Bash syntax/Python compile,
  `git diff --check` và GPU parity trên server. Các test CPU ban đầu của native,
  adapter, planner và executor đã chạy trước; thay đổi tích hợp cuối chưa được
  chạy lại, GPU chưa xác minh.

## Sổ quyết định

- 15/10: người dùng yêu cầu triển khai sau khi hoàn thiện thiết kế; tiếp tục
  thực hiện, không thêm vòng xin xác nhận.
- 15/10: local `.venv` có source vLLM 0.30.0 nhưng GPU không hoạt động;
  kiểm chứng server được phân biệt với CPU/static tests.
- 15/10: A1/A2/A3 replay, B1 và B1-V giữ trạng thái unsupported cho tới khi
  worker có fork/rollback target+draft KV theo request; B0 full trace, A3 E2E,
  B3 fixed-k và B4 fixed-gamma có worker đường chạy độc lập.
