# SSSD

SSSD là speculative decoding model-free dựa trên retrieval. Adapter của
workspace gọi trực tiếp benchmark offline của fork SGLang vendored tại
`externals/SSSD/`, chạy được profile smoke Llama 3.1 8B Instruct, rồi chuyển
metric aggregate sang schema JSONL chung.

## Nguồn và revision

- Upstream: <https://github.com/huawei-csl/sglang-sssd>
- Commit fork đã vendored: `194f2d70c536ceed77c5c420a64de4de641807b3`
- Submodule `sssd_speculator`: `7a39799c85774a876e0ddacb84f6c3eb96da2818`
- Các thư mục này không còn `.git` nested; `.git` của workspace chính vẫn giữ
  nguyên.

## Chạy smoke

```bash
SSSD_MODEL=/workspace/shared_storage/model/Llama3.1-8B-Instruct \
SSSD_DATA_FILE=data/representative_100/xsum_representative.jsonl \
SSSD_MAX_SAMPLES=1 SSSD_MAX_NEW_TOKENS=16 \
bash scripts/run.sh sssd --smoke
```

Wrapper tạo một JSONL tạm theo format `conversations`, gọi
`python -m sglang.bench_offline_throughput --speculative-algorithm SSSD`, bỏ
warmup và ghi kết quả vào `outputs/sssd.jsonl` (hoặc `OUTPUT_FILE`).

## Điều kiện và giới hạn

- Cần GPU CUDA, fork SGLang tương thích với torch/CUDA/GPU và extension native
  `sssd_speculator` đã được build/cài trong shared runtime.
- Có thể dùng datastore retrieval đã chuẩn bị bằng `SSSD_DATASTORE_PATH`; để
  trống thì adapter vẫn kiểm tra được wiring với datastore rỗng.
- SSSD upstream trả timing aggregate thay vì text theo từng sample, vì vậy
  record hiện tại là aggregate của batch; smoke được khóa ở một sample.
- Máy dev T4 trong workspace chạy CPU nên không thể khẳng định inference SSSD
  thành công ở đây; cần chạy lệnh trên server GPU có model snapshot local.
