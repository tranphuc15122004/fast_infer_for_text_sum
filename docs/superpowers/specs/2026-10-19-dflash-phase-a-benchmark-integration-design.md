# DFlash Phase A Benchmark Integration Spec (vLLM & FA4)

> **Status:** Approved Design  
> **Date:** 2026-10-19  
> **Objective:** Triển khai thuật toán Context Focusing tối ưu của Phase A (Winning Production Recipe) cho cả hai runtime chính trong `src/Benchmark`: **vLLM** (`vllm_all_baselines.py`) và **FA4 Native** (`native_fa4_benchmark.py`), đảm bảo chạy đồng bộ và đối chiếu trực tiếp với các baseline hiện hữu trên server B200.

---

## 1. Bối cảnh & Mục tiêu

Theo báo cáo tổng kết khoa học [`docs/_INSIGHT/PHASE_A_MASTER.md`](file:///home/tuantb/fast_infer_text_sum/docs/_INSIGHT/PHASE_A_MASTER.md), cơ chế **Context Focusing (Phase A)** giải quyết hai nút thắt cốt tử của DFlash khi sinh tóm tắt văn bản dài ($N \ge 8\text{K}, 16\text{K}$):
1. **Giảm gánh nặng tính toán Attention của Drafter**: Cắt giảm kích thước KV cache từ $N$ xuống $B=2048$ (cắt giảm tới $87.5\%$ FLOPs attention).
2. **Loại bỏ nhiễu ngữ cảnh (Context Distraction)**: Giúp drafter tập trung vào phân đoạn văn bản liên quan, tăng số token được chấp nhận trung bình mỗi round ($E[L]$ tăng $+43.3\%$, từ $0.709$ lên $1.016$ tokens/round tại 16K context).

Cấu hình tối ưu tuyệt đối đã được xác lập (**The Winning Production Recipe**):
- **Tầng trích xuất (Target Layer)**: $L_{18}$ đơn lẻ (Middle Retrieval Core - Rank 1 Peak toàn cục 87.26% Core Mean).
- **Toán tử Attention (Query Pooling)**: Multi-Query Max-Pooling ($M_1$) trên toàn bộ `accepted-prefix`.
- **Đơn vị phân bổ vùng nhớ**: Chunk 128 tokens (Spatial Continuity $>98\%$).
- **Neo cố định (Fixed Anchors)**: 16 Sink tokens đầu + 32 Recent tokens cuối.
- **Ngân sách làm việc**: $B = 2048$ tokens (tùy biến qua env `DFLASH_PHASE_A_BUDGET`).

Mục tiêu thiết kế này là mang cấu hình tối ưu này vào cả hai backend chuẩn của repo để có thể chạy benchmark đo đạc trực tiếp trên máy chủ NVIDIA B200:
- **FA4 Backend** (`src/Benchmark/native_fa4_benchmark.py`)
- **vLLM Backend** (`src/Benchmark/vllm_all_baselines.py`)

---

## 2. Kiến trúc & Các thành phần (Zero-Overhead Engineering)

> [!IMPORTANT]
> **5 Nguyên tắc Kỹ thuật Zero-Overhead (Tuyệt đối không để xảy ra nút thắt tính toán):**
> 1. **100% GPU-Resident (Tuyệt đối không GPU-CPU sync)**: Toàn bộ quá trình tính điểm $M_1$, gom nhóm Chunk 128, tìm Top-$K$ chunks, tạo boolean mask và lập chỉ mục KV Cache đều thực thi thuần túy trên GPU bằng PyTorch tensor operations (`torch.amax`, `.view(-1, 128).sum(-1)`, `torch.topk`, `torch.index_select`). Không bao giờ gọi `.cpu()`, `.tolist()` hay loop Python trên token/chunk trong vòng lặp sinh.
> 2. **Không bật `output_attentions=True` trên Target Model**: Việc bật `output_attentions=True` sẽ ép PyTorch lưu trữ ma trận attention của toàn bộ 36 tầng (tốn hàng GB VRAM và chậm hơn 30-50ms mỗi step). Thay vào đó, chỉ đặt forward hook thu giữ duy nhất vector $Q$ tại tầng $L_{18}$ (`target_layer=18`), rồi thực hiện 1 phép nhân ma trận cục bộ cực nhanh $Q_{18} K_{18}^T$ ($K \le 16$, $N \le 16384$, thời gian chỉ $<0.05$ ms trên B200).
> 3. **Chunk 128 Aggregation siêu tốc bằng Tensor View**: Chuyển vector điểm 16K thành tensor `[num_chunks, 128]` và `.sum(dim=-1)` trên GPU trong $2\,\mu\text{s}$, sau đó gọi `torch.topk` trực tiếp trên GPU.
> 4. **Zero Memory Copy trong vLLM PagedAttention**: Không di chuyển dữ liệu KV trong VRAM. Chỉ lọc mảng `block_table` (mỗi chunk 128 tokens tương ứng đúng 8 block vật lý của PagedAttention 16 tokens), truyền 128 block IDs vào `attn_metadata` để kernel PagedAttention của vLLM tự động chỉ đọc đúng $B=2048$ tokens.
> 5. **Coalesced Memory Access trong FA4**: Nhờ cơ chế Chunk 128 tokens, các chỉ mục được chọn gồm các dải liên tục 128 tokens liền kề, tối ưu hóa triệt để GPU memory coalescing khi `index_select` trên KV cache của drafter.

### 2.1. Module cốt lõi: `src/Benchmark/common/phase_a_selector.py`
Tập trung toàn bộ logic toán học và giải thuật chọn ngữ cảnh của Phase A thành một module thuần túy, 100% GPU-resident:
- `PhaseAConfig`: Cấu hình tham số với giá trị mặc định từ Phase A Master, hỗ trợ đọc từ biến môi trường (`DFLASH_PHASE_A_LAYER=18`, `DFLASH_PHASE_A_BUDGET=2048`, `DFLASH_PHASE_A_CHUNK_SIZE=128`, `DFLASH_PHASE_A_SINK=16`, `DFLASH_PHASE_A_RECENT=32`).
- `TargetLayer18QueryCapture`: Hook nhẹ chỉ gắn vào `target.model.layers[18].self_attn.q_proj` để thu giữ query tokens của vòng verify hiện tại ($K \le 16$), không làm chậm bất kỳ tầng nào khác.
- `compute_multi_query_scores(query_18, key_18, scaling, query_pooling="max") -> torch.Tensor`:
  Thực hiện 1 phép GEMM nhỏ trên GPU ($Q_{18} K_{18}^T$), sau đó lấy `torch.amax(dim=-2)` qua các accepted queries. Thời gian $<0.05$ ms.
- `select_focused_context_indices(scores, total_length, config) -> torch.Tensor`:
  - Thực hiện `.view(-1, 128).sum(dim=-1)` trên GPU.
  - Loại trừ chunk đầu (chứa sink) và chunk cuối (chứa recent) bằng cách gán $-\infty$ trên GPU.
  - `torch.topk` chọn các chunk có điểm cao nhất.
  - Sinh tensor chỉ mục `[B]` tăng dần hoàn toàn bằng phép toán tensor trên GPU.
- `select_focused_paged_blocks(scores, total_length, block_size, config) -> torch.Tensor`:
  - Dành riêng cho vLLM PagedAttention: Chọn các chỉ số block vật lý trong `block_table` trên GPU.

### 2.2. Tích hợp FA4 Backend (`src/Benchmark/native_fa4_benchmark.py`)
- Bổ sung `dflash_phase_a` vào `SUPPORTED_METHODS` trong `src/Benchmark/common/flashattn_runtime.py`.
- Bổ sung `dflash_phase_a` vào `MODEL_REPOS` trong `src/Benchmark/native_fa4_benchmark.py` trỏ tới `MODEL_DFLASH_DRAFT`.
- Trong `dflash_generate`: Khi `method == "dflash_phase_a"`, hook để trích xuất ma trận chú ý tầng 18 của Target Model trong mỗi bước verify.
- Thu gọn Draft KV Cache / Key-Value inputs của Drafter về tập indices $\mathcal{I}_{\text{focused}}$ trước khi gọi FlashAttention-4 forward.
- Thu thập và xuất metrics: ghi nhận `phase_a_budget`, `phase_a_selected_tokens`, tỷ lệ chấp nhận $E[L]$ và kiểm tra token parity với greedy vanilla reference.

### 2.3. Tích hợp vLLM Backend (`src/Benchmark/vllm_all_baselines.py`)
- Bổ sung `dflash_phase_a` vào `METHODS` và `DRAFT_ARGUMENTS` trong `src/Benchmark/vllm_all_baselines.py`.
- Trong `src/Benchmark/common/vllm_pilot.py`:
  - Mở rộng `select_pilot_methods` để chấp nhận `dflash_phase_a`.
  - Cài đặt `install_phase_a_vllm_compat()`: Hook vào `DFlashSpeculator._run_model` (hoặc `propose`) trong Model Runner V2 của vLLM.
  - Khi `dflash_phase_a` hoạt động: Tận dụng tín hiệu hidden/attention của target layer 18 để tính chunk score, sau đó lọc `block_tables` của PagedAttention trong `attn_metadata` cho draft forward pass sao cho drafter chỉ đọc các block thuộc $B=2048$ tokens.
- Giữ nguyên các đo lường chuẩn của pipeline: TPOT, DSR, ESR, Acceptance Length ($E[L]$), Token LCS Overlap, Parity check.

---

## 3. Global Constraints & Testing Strategy
- **Không phá vỡ baseline hiện tại**: `dflash` (dense), `eagle3`, `domino`, `dspark`, `vanilla` phải giữ nguyên 100% hành vi cũ.
- **CPU Testability**: Mọi logic trích xuất điểm, chunking, block table mapping phải chạy được và kiểm thử được bằng unit test trên CPU local (`.venv/bin/pytest`) mà không cần GPU cu130.
- **B200 Readiness**: Code tương thích hoàn toàn với GPU B200 (SM100+), FA4 CUTLASS b15 và vLLM 0.30 Model Runner V2.
