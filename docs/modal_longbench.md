# Chạy LongBench representative trên Modal

`run_longbench_200.py` vẫn là runner chuẩn. `scripts/modal_longbench.py` chỉ
đóng gói phần hạ tầng Modal còn thiếu: image Python 3.12, mount source/data,
cache Hugging Face, master-env tạm trong container và Volume lưu output. Khi
container bắt đầu, app tạo/reuse `/mnt/fast-infer/venv` bằng Python 3.12 với
`--system-site-packages`; child runner và mọi baseline đều được gọi bằng
`/mnt/fast-infer/venv/bin/python`.

Đây là venv được dựng lại từ dependency manifest trong image Modal, không phải
bản sao byte-for-byte của venv server B200; Modal không truy cập được các
wheel/path tuyệt đối của server.

## Chuẩn bị một lần

Model Llama cần token Hugging Face. Secret chỉ được truyền vào container khi
đặt tên qua `MODAL_HF_SECRET`; không ghi token vào repository:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

Volume mặc định là `fast-infer-text-sum-cache`. Có thể đổi bằng
`MODAL_VOLUME=...`; Volume này chứa cả model cache, CUDA cache và output nên
những lần chạy sau không phải tải model lại.

Image mặc định cài các dependency public ở `requirements.modal.txt`. Các
kernel CUDA đặc thù không được giả định là đã có. Nếu chọn baseline cần
FlashAttention, build image bằng:

```bash
MODAL_INSTALL_FLASH_ATTN=1 MODAL_GPU=A100-80GB \
  modal run scripts/modal_longbench.py --mode smoke \
  --baselines "vanilla_hf vanilla_fa"
```

Với image hiện tại (`torch 2.11.0+cu130`, Python 3.12), PyPI chỉ cung cấp
source distribution cho `flash-attn==2.8.3.post1`, nên Modal sẽ phải compile
CUDA extension khá lâu. Có thể dùng wheel đã build đúng ABI cho smoke/
benchmark A100 bằng cách truyền URL sau:

```bash
export MODAL_FLASH_ATTN_WHEEL='https://github.com/adithyaxx/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
MODAL_HF_SECRET=huggingface MODAL_INSTALL_FLASH_ATTN=1 \
MODAL_FLASH_ATTN_WHEEL="$MODAL_FLASH_ATTN_WHEEL" MODAL_GPU=A100-80GB \
  modal run scripts/modal_longbench.py --mode representative \
  --baselines "vanilla_hf specextend" --datasets lcc --max-samples 1 \
  --max-new-tokens 32 --max-input-tokens 4096 \
  --run-id specextend-flashattn-smoke
```

Đây là wheel cộng đồng, không phải artifact upstream; chỉ dùng khi đã kiểm
tra `import flash_attn` và version trong runtime. Nếu không muốn dùng wheel
này, bỏ `MODAL_FLASH_ATTN_WHEEL` để build từ source chính thức.

`MODAL_INSTALL_VLLM=1` và `MODAL_INSTALL_FLASHINFER=1` là các cờ opt-in tương
tự cho baseline tương ứng. Không bật chúng nếu baseline không cần, vì image
sẽ lớn và build lâu hơn.

MagicDec cần checkpoint đã convert, không thể dùng trực tiếp HF checkpoint.
Modal runner mặc định tìm file tại:
`/mnt/fast-infer/checkpoints/magicdec/llama-3.1-8b/model.pth`. Người dùng cần
chủ động upload artifact này vào Volume (đặc biệt nếu checkpoint là dữ liệu
riêng tư):

```bash
modal volume put fast-infer-text-sum-cache \
  checkpoints/magicdec/llama-3.1-8b/ \
  checkpoints/magicdec/llama-3.1-8b/
```

Sau đó kiểm tra MagicDec bằng FlashInfer mà không để pip thay Torch/CUDA:

```bash
MODAL_HF_SECRET=huggingface MODAL_INSTALL_FLASHINFER=1 \
MODAL_GPU=A100-80GB modal run scripts/modal_longbench.py \
  --mode representative --baselines magicdec --datasets lcc \
  --max-samples 1 --max-new-tokens 32 --max-input-tokens 4096 \
  --run-id magicdec-flashinfer-smoke
```

## Kiểm tra trước khi tốn GPU/model download

Chạy preflight remote, không load model và không chạy inference:

```bash
MODAL_GPU=A100-80GB \
  modal run scripts/modal_longbench.py \
  --mode representative \
  --baselines vanilla_hf \
  --datasets "gov_report lcc" \
  --max-samples 20 \
  --preflight-only
```

## Smoke rồi representative

Mặc định app chỉ chọn `vanilla_hf` để xác nhận image/cache/data trước; mặc
định này không đại diện cho toàn bộ ma trận baseline. Smoke đầu tiên:

```bash
MODAL_HF_SECRET=huggingface MODAL_GPU=A100-80GB \
  modal run scripts/modal_longbench.py \
  --mode smoke \
  --baselines vanilla_hf \
  --datasets "gov_report lcc"
```

Sau khi smoke đạt, chạy profile representative canonical (`20 mẫu ×
gov_report,lcc`) như sau:

```bash
MODAL_HF_SECRET=huggingface MODAL_GPU=A100-80GB \
  modal run scripts/modal_longbench.py \
  --mode representative \
  --baselines vanilla_hf \
  --datasets "gov_report lcc" \
  --max-samples 20 \
  --run-id modal-representative-01
```

Muốn thêm baseline, truyền cùng một chuỗi vào `--baselines`, ví dụ:

```bash
MODAL_HF_SECRET=huggingface \
MODAL_INSTALL_FLASH_ATTN=1 \
  modal run scripts/modal_longbench.py \
  --mode representative \
  --baselines "vanilla_hf vanilla_fa eagle3 dflash specextend fafo" \
  --datasets "gov_report lcc" \
  --max-samples 20
```

Các model draft mặc định là Hugging Face repo ID. Có thể ghi đè bằng các cờ
`--model`, `--eagle-model` và `--dflash-model`. `magicdec` không nằm trong ví
dụ trên vì checkpoint `.pth` của nó là artifact riêng; cần đưa checkpoint vào
Volume rồi cấu hình biến tương ứng trong runner trước khi đưa baseline đó vào
ma trận.

## Lấy output

Remote function in `run_id`, đường dẫn manifest và trạng thái exit. Artifact
được commit vào Volume sau khi child runner kết thúc:

```bash
modal volume ls fast-infer-text-sum-cache outputs/longbench_100_14k
modal volume get fast-infer-text-sum-cache \
  outputs/longbench_100_14k/modal-representative-01 \
  outputs/modal-representative-01
```

Trong thư mục run cần kiểm tra `run_manifest.json`, `logs/` và
`metrics_summary.{json,csv,md}`. Nếu `vanilla_hf` dùng eager attention bị OOM ở
context dài, chỉ thêm `--max-input-tokens N` khi chấp nhận thay đổi phạm vi
đo; giá trị này sẽ được ghi trong manifest.

## Những gì không được kế thừa từ server

Modal không nhìn thấy các path `/workspace/storage-shared/...` trong
`config/master.path`. App tạo master-env mới trong `/tmp`, bật
`FI_OFFLINE=0`, dùng HF cache trên Volume và mount dataset từ workspace local.
Do đó không dùng trực tiếp server master-env cho job Modal.
