# FAFO

FAFO là draftless fumble decoding kết hợp lossy KV-cache compression cho profile
Llama 3.1 8B Instruct. Adapter
của workspace dựng config GSM8K tạm cho một hoặc nhiều prompt, gọi pipeline
upstream rồi chuẩn hóa log throughput/timing sang schema JSONL chung.

## Nguồn và revision

- Upstream: <https://github.com/Escanord/FAFO>
- Commit đã vendored: `52d9ce549505476a3d56d4f31f29ea9d53aef086`
- Các thư mục này không còn `.git` nested; `.git` của workspace chính vẫn giữ
  nguyên.

## Chạy smoke

```bash
FAFO_MODEL=/workspace/shared_storage/model/Llama3.1-8B-Instruct \
FAFO_DATA_FILE=data/representative_100/xsum_representative.jsonl \
FAFO_MAX_SAMPLES=1 FAFO_MAX_NEW_TOKENS=16 \
FAFO_KV_METHOD=stream-llm \
bash scripts/run.sh fafo --smoke
```

Đổi `FAFO_KV_METHOD=quest` để chạy biến thể Quest. Kết quả chuẩn hóa nằm ở
`outputs/fafo.jsonl`; raw result và `exp.log` upstream nằm trong thư mục
`outputs/fafo_runtime/` tương ứng.

## Điều kiện và giới hạn

- FAFO upstream được phát triển với Python 3.9.20, CUDA 12.1 và GPU NVIDIA
  lớn; cần kiểm tra compatibility trước khi chạy trong shared Python 3.12.
- Pipeline dùng CUDA trực tiếp và không có CPU fallback. `FAFO_USE_FLASH=1`
  chỉ bật khi FlashAttention tương thích đã có sẵn; mặc định smoke dùng
  `use_flash=false` để giảm thêm một điểm phụ thuộc.
- Eval upstream GSM8K thêm prompt few-shot và trả timing/log aggregate; adapter
  chỉ dùng prompt đầu vào của workspace làm một câu hỏi smoke, không sửa file
  dataset/config vendored.
