# Update Latest LongBench Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cập nhật báo cáo benchmark LongBench chính bằng artifact mới nhất trong `outputs/Benchmark_results/ketqua_benchmark/`, phản ánh đúng metric đầy đủ của SpecExtend và trạng thái mới của toàn bộ baseline.

**Architecture:** Dùng `ketqua_benchmark/metrics_summary.json` và `metrics_summary.md` làm nguồn tổng hợp mới nhất; dùng `logs/metrics_audit.log` và các file `specextend_*.metrics.json` để kiểm tra coverage/schema. Thay nội dung báo cáo cũ bằng kết luận phân biệt rõ timing/quality đầy đủ, speedup ghép cặp được, và các baseline failed hoặc khác measurement scope.

**Tech Stack:** Markdown, JSON/JSONL artifacts, Python 3 read-only audit, `rg`, `git diff`.

**Spec:** `outputs/Benchmark_results/ketqua_benchmark/run_manifest.json` và `outputs/Benchmark_results/ketqua_benchmark/metrics_summary.json`.

## Global Constraints

- Không suy diễn speedup khi artifact ghi `valid_speedup_pairs=0`.
- Không gộp số liệu run cũ vào run mới nếu khác trạng thái hoặc measurement scope.
- Ghi rõ run mới dùng 100 mẫu/dataset, 5 dataset, 7 baseline, `max_new_tokens=2048`, seed 42.
- Giữ các metric có trong artifact: timing, throughput, memory, speculative acceptance, semantic/code quality.
- FAFO `e2e_only` không được trình bày như có prefill/decode/TTFT đầy đủ.
- MagicDec failed trong run mới phải được ghi là failed, không dùng số liệu MagicDec từ run cũ.

---

### Task 1: Chốt nguồn dữ liệu và số liệu cập nhật

**Files:**
- Read: `outputs/Benchmark_results/ketqua_benchmark/run_manifest.json`
- Read: `outputs/Benchmark_results/ketqua_benchmark/metrics_summary.json`
- Read: `outputs/Benchmark_results/ketqua_benchmark/metrics_summary.md`
- Read: `outputs/Benchmark_results/ketqua_benchmark/logs/metrics_audit.log`

- [x] Xác nhận manifest có 7 baseline, 5 dataset, 100 mẫu/dataset, output budget 2048 và run ID `b200-full-8gpu-1shards-baseline-first`.
- [x] Xác nhận audit coverage: Vanilla HF/FA, EAGLE3, DFlash, SpecExtend thành công; MagicDec failed; FAFO `e2e_only` thành công.
- [x] Trích xuất các bảng overall và theo dataset từ `metrics_summary.md`, không tự tính lại metric từ log nếu summary đã có.

### Task 2: Cập nhật báo cáo chính

**Files:**
- Modify: `outputs/Benchmark_results/longbench_full_benchmark_analysis.md`

- [x] Đổi tiêu đề, nguồn run và kết luận điều hành sang artifact mới nhất.
- [x] Cập nhật coverage/status theo 35 cells của run mới.
- [x] Thay bảng speed/quality cũ bằng số liệu mới, gồm SpecExtend full E2E + quality và FAFO e2e-only.
- [x] Ghi MagicDec là failed trong run mới và loại số liệu MagicDec cũ khỏi kết luận canonical.
- [x] Ghi rõ speedup chỉ có cho DFlash/EAGLE3 theo summary; SpecExtend chưa có valid speedup pairs trong artifact mới.
- [x] Cập nhật các mục CONFIRMED, EXPLORATORY, FAILED/INCOMPLETE, EVIDENCE GAPS và RECOMMENDED NEXT.

### Task 3: Kiểm tra báo cáo sau chỉnh sửa

**Files:**
- Read: `outputs/Benchmark_results/longbench_full_benchmark_analysis.md`

- [x] Kiểm tra không còn tuyên bố cũ rằng SpecExtend là `decode-only` hoặc thiếu semantic quality.
- [x] Kiểm tra không còn dùng MagicDec speedup cũ như kết quả của run mới.
- [x] Kiểm tra các số tổng hợp chính của SpecExtend khớp `metrics_summary.md`.
- [x] Chạy kiểm tra Markdown/grep và xem `git diff --check`.
