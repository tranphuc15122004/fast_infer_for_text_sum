# Ma trận model dùng cho các baseline

Note này chốt model phục vụ đánh giá thực tế trong benchmark long-context
text summarization. Mục tiêu là dùng một target model chung nhiều nhất có thể,
đồng thời giữ đúng cặp target/draft bắt buộc của từng baseline.

`MR_DFlash` không phải một baseline trong ma trận này. `DFlash` ở các bảng dưới
đây là đường inference benchmark với cặp target/draft đã chốt; còn
`src/MR_DFlash` hiện chỉ là workspace train bản sao DFlash để phát triển ý
tưởng mới, chưa có model hoặc kết quả MR-DFlash riêng.

## 1. Bộ model đã chốt

| ID | Model | Vai trò chính |
|---|---|---|
| M1 | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Target LLM chung cho phần lớn baseline |
| M2 | `meta-llama/Llama-3.2-1B-Instruct` | Draft model cho `speculative_prefill` |
| M3 | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` | EAGLE-3 draft cho Llama 3.1 |
| M4 | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` | DFlash draft cho Llama 3.1 |
| M5 | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | Compressor của LLMLingua-2 |
| M6 | `sentence-transformers/all-MiniLM-L6-v2` | Encoder cho semantic selection |
| M7 | `lmsys/vicuna-7b-v1.5-16k` | Target cho LongSpec và SpecExtend classic fallback |
| M8 | `double7/vicuna-68m` | Draft cho SpecExtend classic fallback |
| M9 | `sail/longspec-vicuna-7b-v1.5-16k` | LongSpec draft tương ứng Vicuna 7B |

## 2. Ánh xạ baseline → model

| Baseline | Target / model chính | Draft / model phụ | Ghi chú đánh giá |
|---|---|---|---|
| Semantic Selection (`full`, `random`, `lead`, `tfidf`, `mmr`) | M1 | M6 | M1 sinh summary; M6 tạo embedding cho selector. Các scheme phải dùng cùng target, token budget và dữ liệu. |
| FastKV | M1 | — | Dùng Llama 3.1 cho full; smoke hiện tại trong config vẫn mặc định Mistral. |
| RocketKV | M1 | — | Full LongBench dùng Llama 3.1; smoke kernel không tải model. |
| GemFilter | M1 | — | Llama 3.1 cần `SELECT_LAYER_IDX=13`; smoke hiện tại dùng Phi-3.5 để an toàn VRAM. |
| LLMLingua-2 | M1 | M5 | M5 nén context, sau đó M1 sinh summary. Không dùng M5 làm target LLM. |
| MInference | M1 | — | Llama 3.1 nằm trong danh sách model được hỗ trợ của MInference. |
| `speculative_prefill` | M1 | M2 | Target Llama 3.1 8B, draft Llama 3.2 1B; smoke có thể dùng target nhỏ hơn. |
| EAGLE-3 | M1 | M3 | M3 là checkpoint EAGLE-3 được huấn luyện cho đúng Llama 3.1; không hoán đổi sang base model khác. |
| DFlash | M1 | M4 | M4 là DFlash draft tương ứng Llama 3.1; target và draft phải cùng họ/tokenizer tương thích. |
| SpecExtend + EAGLE-3 | M1 | M3 | Đúng setup ở Figure 1 của paper; headline 3.86x của paper lại dùng DeepSeek-R1-Distill-Llama-8B + EAGLE-3 trên AIME-24. |
| SpecExtend classic fallback | M7 | M8 | Chỉ dùng khi benchmark riêng nhánh classic của repo; không đại diện cho cấu hình EAGLE-3. |
| LongSpec | M7 | M9 | Đây là cặp chính thức `lmsys/vicuna-7b-v1.5-16k` + `sail/longspec-vicuna-7b-v1.5-16k`. |
| MagicDec | M1 | Không có draft riêng ở self-spec | MagicDec self-spec dùng target Llama 3.1 đã convert sang `model.pth`; `double7/vicuna-68m` không phải dependency bắt buộc của cấu hình này. |
| HiGOE | Không có target inference chung cố định | Mô hình phụ trợ | Retriever bắt buộc theo repo là `facebook/contriever`; LLM judge mặc định gọi API, hoặc có thể cấu hình local Llama 3.1. |

## 3. Bộ target dùng chung

M1 là target chung được ưu tiên cho các baseline sau:

```text
Semantic Selection
FastKV
RocketKV
GemFilter
LLMLingua-2
MInference
speculative_prefill
EAGLE-3
DFlash
MagicDec
```

Điều này giúp so sánh latency, throughput và chất lượng summary trên cùng một
LLM. LongSpec và SpecExtend classic giữ cặp Vicuna chính thức; riêng nhánh
SpecExtend + EAGLE-3 dùng M1/M3 theo checkpoint tương thích của paper.

## 4. Phân biệt full evaluation và smoke test

Các model smoke hiện có trong repo chủ yếu nhằm kiểm tra import, kernel,
schema output và đường chạy tối thiểu. Chúng không phải model dùng để báo cáo
kết quả cuối cùng.

| Baseline | Smoke hiện tại | Full evaluation nên dùng |
|---|---|---|
| FastKV | Mistral-7B + SnapKV/SDPA | M1 + FastKV/FlashAttention |
| GemFilter | Phi-3.5-mini + eager | M1 + GemFilter |
| LLMLingua-2 | Target nhỏ nếu cần | M1 + M5 |
| EAGLE-3 | Qwen3-4B + Qwen3-EAGLE hiện có trong config cũ | M1 + M3 |
| DFlash | Qwen3-4B + Qwen3-DFlash hiện có trong config cũ | M1 + M4 |
| MagicDec | TinyLlama checkpoint | M1 đã convert sang MagicDec `model.pth` |
| SpecExtend + EAGLE-3 | M1 + M3, giới hạn input/output ngắn | M1 + M3 |
| LongSpec | Import/kernel smoke | M7 + M9 |
| `speculative_prefill` | TinyLlama fallback | M1 + M2 |
| RocketKV | Không tải model | M1 trong full pipeline |
| HiGOE | Import + Contriever round-trip | Contriever + LLM judge được chỉ định |

## 5. Các điểm cần cập nhật trước khi chạy full

1. Hai model gated M1 và M2 phải được tải sau khi đăng nhập Hugging Face và
   chấp nhận license tương ứng.
2. `externals/Sematic_selection/infer.py` hiện còn mặc định Qwen3-4B; cần
   truyền hoặc đổi default sang M1, còn M6 giữ làm embedding model.
3. Config EAGLE-3 và DFlash hiện còn tham chiếu Qwen3; cần thay target/draft
   sang M1/M3 và M1/M4 tương ứng.
4. MagicDec không nhận trực tiếp checkpoint HF khi chạy benchmark; cần convert
   M1 sang định dạng `model.pth` và đặt đúng model key `llama-3.1-8b`.
5. LongSpec phải chạy với `MODEL_NAME=vicuna7b` để lấy đúng cặp M7/M9.
6. Smoke SpecExtend + EAGLE-3 dùng `SCRIPT=run_eagle.py`,
   `MODEL_NAME=llama3_1_8b` và cặp M1/M3; M7/M8 chỉ là classic fallback.
   Không dùng smoke này để khẳng định đã tái hiện số 3.86x trong paper.
7. Khi so sánh các baseline, phải cố định dataset normalized, số mẫu, prompt
   template, `MAX_NEW_TOKENS`, decoding và cách đo prefill/decode.

## 6. Trạng thái cache tại thời điểm lập note

Đã có trong Hugging Face cache: M3, M4, M5, M6, M7, M8 và M9.

Chưa có: M1 và M2. Hai model này đang bị chặn bởi quyền gated vì phiên
Hugging Face chưa đăng nhập.
