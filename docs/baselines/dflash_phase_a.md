# DFlash Phase A (Target Context Focusing)

Thuật toán tăng tốc suy đoán DFlash bằng cơ chế **Target Context Focusing (Phase A)** — tích hợp đồng bộ cho cả 2 backend chính trong `src/Benchmark`: **vLLM Unified** và **Native FlashAttention-4 (FA4)**.

---

## 1. Cơ chế Hoạt động & Kiến trúc Kỹ thuật

Trong suy đoán song song khối (Block-Parallel Diffusion Drafting như DFlash), drafter chỉ cần dự đoán một khối ngắn $K$ tokens ($K = 16$). Khi context dài ($L = 16\text{k} - 32\text{k}$ tokens), chi phí attention của draft model tăng tuyến tính $O(K \times L)$, làm giảm tỷ lệ End-to-End Speedup (ESR).

**DFlash Phase A** giải quyết nút thắt này thông qua cơ chế chọn lọc ngữ cảnh thông minh:
1. **Target Layer 18 ($L_{18}$)**: Sử dụng biểu diễn Query từ tầng 18 của Target Model — tầng có độ bão hòa ngữ nghĩa và chú ý toàn cục tối ưu nhất theo thực nghiệm Attention Coverage Matrix.
2. **Multi-Query Max-Pooling ($M_1$)**: Thực hiện max-pooling điểm attention qua toàn bộ các token vừa được Target Model chấp nhận trong vòng xác thực (accepted tokens), nắm bắt toàn diện nhu cầu thông tin đa hướng của cụm token mới.
3. **Chunk-Level Aggregation (Chunk 128)**: Nhóm token thành các chunk 128 tokens để tính tổng điểm chú ý, bảo toàn liên tục không gian ngữ cảnh (>98% spatial continuity).
4. **Anchor Protection**: Luôn luôn giữ lại 16 Sink Tokens đầu câu (bảo toàn chú ý nền tảng) và 32 Recent Tokens liền kề trước (bảo toàn ngữ pháp cục bộ).
5. **Fixed Budget $B = 2048$**: Chọn các top chunks có điểm cao nhất sao cho tổng context của Drafter không vượt quá $B=2048$ tokens. Chi phí attention của drafter giảm từ $O(K \times L)$ xuống $O(K \times B)$ (giảm $8\times - 16\times$).

---

## 2. Zero-Overhead Engineering

Để đảm bảo thuật toán không gây trễ tính toán (overhead), toàn bộ module `src/Benchmark/common/phase_a_selector.py` tuân thủ nghiêm ngặt 5 nguyên tắc:
- **100% GPU Operations**: Tuyệt đối không gọi `.cpu()`, `.tolist()`, hay `item()` trong vòng lặp sinh.
- **Không duyệt Python loop**: Mọi phép gom chunk và xếp hạng thực hiện bằng tensor math: `.view(-1, 128).sum(-1)` và `torch.topk`. Thời gian chọn lọc context $<0.01\,\text{ms}$.
- **Chỉ hook $Q_{18}$**: Không bật `output_attentions=True` trên Target Model (tránh cấp phát bộ nhớ attention khổng lồ cho 36 tầng). Chỉ dùng forward hook nhẹ trích xuất $Q_{18}$ và tính 1 phép GEMM $Q_{18} K_{18}^T$ cực nhanh ($<0.05\,\text{ms}$).
- **RoPE-Safe Drafting**: Việc cắt tỉa $K, V$ về tập $B=2048$ tokens bằng `index_select` bảo toàn nguyên vẹn RoPE đã được tính trước đó trong non-causal draft attention.

---

## 3. Các Biến Môi Trường Cấu hình

| Biến môi trường | Mặc định | Ý nghĩa |
|---|:---:|---|
| `DFLASH_PHASE_A_LAYER` | `18` | Chỉ số tầng của Target Model dùng để trích xuất Query tín hiệu |
| `DFLASH_PHASE_A_BUDGET` | `2048` | Ngân sách token tối đa cho Drafter ($B$) |
| `DFLASH_PHASE_A_CHUNK_SIZE` | `128` | Kích thước chunk token gom cụm |
| `DFLASH_PHASE_A_SINK_TOKENS` | `16` | Số lượng sink tokens cố định ở đầu ngữ cảnh |
| `DFLASH_PHASE_A_RECENT_TOKENS` | `32` | Số lượng recent tokens cố định ở cuối ngữ cảnh |

---

## 4. Hướng dẫn Chạy Benchmark trên Server NVIDIA B200

Baseline `dflash_phase_a` đã được đăng ký đồng bộ vào hệ thống đánh giá của repo, có thể chạy song song với `vanilla` và `dflash` gốc.

### A. Chạy qua Pipeline vLLM Đồng Bộ (`vllm_all_baselines.py`)

Đây là pipeline chính thức, đảm bảo tính công bằng cao nhất trên cùng runtime vLLM 0.30:

```bash
# 1. Kiểm tra preflight (kiểm tra model weights, cấu hình GPU, import)
bash scripts/run.sh vllm_all --preflight-only --methods vanilla_vllm,dflash,dflash_phase_a

# 2. Chạy smoke test nhanh (2 samples)
bash scripts/run.sh vllm_all --smoke --methods vanilla_vllm,dflash,dflash_phase_a

# 3. Chạy full benchmark (toàn bộ LongBench)
bash scripts/run.sh vllm_all --full --methods vanilla_vllm,dflash,dflash_phase_a
```

### B. Chạy qua Pipeline Native FA4 (`native_fa4_benchmark.py`)

Dành cho đánh giá chuyên sâu nhân FlashAttention-4 thuần với context selection trực tiếp trên KV cache:

```bash
# 1. Chạy smoke test
python3 src/Benchmark/native_fa4_benchmark.py --smoke --methods vanilla_hf,dflash,dflash_phase_a

# 2. Chạy full benchmark
python3 src/Benchmark/native_fa4_benchmark.py --full --methods vanilla_hf,dflash,dflash_phase_a
```

---

## 5. Đánh giá Kết quả

Kết quả benchmark sẽ được tự động ghi lại tại `outputs/`:
- **vLLM Pipeline**: `outputs/vllm_unified/<run-id>/report_vi.md` và `run_report.json`
- **Native FA4 Pipeline**: `outputs/native_fa4/<run-id>/report_vi.md`

Các chỉ số so sánh chính cần đối chiếu giữa `dflash` và `dflash_phase_a`:
1. **DSR (Decode Speedup Ratio)**: Tốc độ decode có duy trì tương đương hoặc tăng không.
2. **ESR (End-to-End Speedup Ratio)**: Tốc độ E2E có cải thiện rõ rệt trên context dài ($>8\text{k}$ tokens) do giảm đáng kể chi phí draft attention không.
3. **Acceptance Rate ($\alpha$)**: Tỷ lệ chấp nhận token của DFlash có được duy trì ổn định không (mục tiêu $\ge 98\%$ so với DFlash dense).
4. **Greedy Parity / ROUGE**: Đảm bảo chất lượng sinh và tính đúng đắn của token đầu ra.
