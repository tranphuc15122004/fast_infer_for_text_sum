# Hồ sơ server benchmark

Đây là nơi canonical lưu thông tin môi trường server của project. Khi làm
việc trên server, ưu tiên tài liệu này thay vì tự suy đoán đường dẫn từ máy
local.

## Đường dẫn cố định

```text
Repository:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

Shared data/config:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data

Master config:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env

LongBench canonical:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k

Legacy representative data:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/representative_100
```

Tên dataset đúng là `representative_100`.

## Runtime server

- Server benchmark dùng trực tiếp lệnh `python3` từ `PATH`.
- `python3` trên server là Python 3.12.
- Không tạo hoặc activate virtualenv trên server.
- `.venv` trong workspace local chỉ dùng để mô phỏng/debug trước khi đưa code
  lên server.
- Dependency, CUDA kernel, model và checkpoint phải được cài/mirror sẵn vì
  server không có internet trực tiếp.
- Master config phải đặt `FI_PYTHON=python3`, `FI_DEVICE=cuda` và
  `FI_OFFLINE=1`.

## Venv cô lập trên B200

Nếu không muốn dùng Python hệ thống, tạo venv Python 3.12 mới và cài toàn bộ
manifest bằng pip:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
export FAST_INFER_B200_VENV="$PWD/.venv-b200"
bash scripts/setup_b200_venv.sh
source "$FAST_INFER_B200_VENV/bin/activate"
```

`requirements.txt` không nhúng public index; pip sẽ dùng mirror/index đã được
operator cấu hình trên server. Với server offline hoặc các route public bị giới
hạn, chỉ dùng wheelhouse đã mirror đủ artifact:

```bash
export B200_OFFLINE=1
export B200_WHEELHOUSE=/workspace/storage-shared/nlp/dungdx4/phuc_projects/offline_wheelhouse
bash scripts/setup_b200_venv.sh
```

`flash-attn==2.8.3.post1` hiện là source distribution trong mirror nên được
để ngoài manifest. Vì vậy `pip install -r requirements.txt` có thể chạy trực
tiếp trên server. Bản FA2 này chỉ dùng cho GPU Ampere/Ada/Hopper; không dùng
cho `vanilla_fa` trên B200/Blackwell.

Server hiện đã có `flash-attn==2.8.3.post1`, `flash-attn-4==4.0.0b15` và
shared CUTLASS 4.5.x. FA4 beta b15 còn import
`cutlass.utils.ampere_helpers`, trong khi CUTLASS hiện tại đã bỏ module cũ
này, và b15 dùng cách gọi positional cũ cho `nvvm.fmax`. Code runner tự đăng
ký module tương thích và adapter `nvvm.fmax` trong đúng process benchmark;
không sửa `site-packages`, không uninstall FA2, không cài lại package. FA2
vẫn được giữ cho các baseline/backend cần nó.

Không kiểm tra FA4 bằng lệnh import trực tiếp, vì lệnh đó bỏ qua shim của
repository. Dùng probe của runner:

```bash
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
"$FAST_INFER_PYTHON" -c '
from common.vanilla_inference import _probe_flash_attention_4
ok, reason = _probe_flash_attention_4()
if not ok:
    raise SystemExit(reason)
print("FA4 import: OK")
'
```

Launcher sẽ áp dụng cùng shim trong preflight và child runner, tự chọn FA4 khi
import probe thành công, đồng thời chặn trước benchmark nếu dependency thực sự
bị hỏng. Khi cần các baseline vanilla_fa
trên GPU được FA2 hỗ trợ, DFlash hoặc SpecExtend full, chạy thêm helper; helper
patch bản source tạm sang C++20 để khớp header Torch 2.14 và build với
`MAX_JOBS=8`. Có thể đổi số job bằng `FLASH_ATTENTION_MAX_JOBS`.
`flashinfer-jit-cache` không có trong mirror nên không nằm trong manifest;
FlashInfer sẽ JIT vào `FLASHINFER_WORKSPACE_BASE`.

Script không xoá target đã tồn tại. Cài system packages `libnuma1` và
`libnuma-dev` bằng image/OS package manager trước khi chạy SGLang; chúng không
thuộc `requirements.txt`.

## MR-DFlash cache smoke tối giản

Để kiểm tra riêng chain MR-DFlash trên B200, mặc định dùng backend
`--cache-backend hf --cache-attention-backend sdpa`. Nhánh này dùng
PyTorch + Transformers và không import SGLang/SpecForge/FlashInfer trong
phase hidden-cache; vì vậy không cần cài thêm `flash-attn`, `sglang-kernel`
hay `specforge` cho smoke cache.

Khi muốn smoke đúng đường tăng tốc SGLang, truyền thêm
`--cache-backend specforge_sglang`. Với runner Phase 1, giữ
`--cache-attention-backend sdpa` hoặc `auto`; cache worker sẽ map giá trị này
sang `flashinfer` cho registry của SpecForge/SGLang.

Runner Phase 1 cũng đã truyền đầy đủ adaptive controls của hidden-cache
worker. Trên B200 dùng `--cache-auto-batch` để bắt đầu từ batch 1, dự đoán
batch kế tiếp theo token capacity của static pool, tự backoff khi OOM và retry
đúng buffer CPU đang dang dở. `--cache-auto-batch-max-size` chỉ là trần khẩn
cấp, không phải batch cố định.

Các package Python trực tiếp cần cho chain này đã nằm trong
`requirements.txt`: `torch`, `transformers`, `numpy`, `PyYAML`,
`safetensors` và `tqdm` (cùng các dependency của Transformers). Trên server
chỉ cài từ mirror/offline wheel của manifest này; không cài `requirements.local.txt`
và không dùng wheel CUDA khác với torch/CUDA của image B200.

Trước khi chạy, đặt JIT/cache runtime vào filesystem có quyền ghi. Lệnh smoke
phase 1 đã khóa backend HF/SDPA và chạy đủ
`regenerate → validate → tokenize → hidden cache`:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
export PYTHONPATH="$PWD/src"
export FAST_INFER_CACHE_ROOT=/tmp/fast_infer_cache
export FI_OFFLINE=1

python3 scripts/mr_dflash/run_phase1_smoke.py \
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/pilot_prompts.jsonl \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_smoke_20 \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --num-samples 20 --max-length 8192 --max-new-tokens 256 \
  --device cuda --local-files-only --resume
```

Backend `specforge_sglang` vẫn được giữ cho throughput benchmark và smoke
accelerated riêng. Để kiểm tra đúng phase hidden-cache còn lại trên B200,
chạy thêm smoke sau khi HF smoke tối giản đã pass:

```bash
# SpecForge's offline capture must resolve the installed/pinned SGLang wheel.
# Do not prepend the vendored SSSD SGLang source here: its scheduler API is a
# different contract and would shadow the cu130 wheel.
export PYTHONPATH="$PWD/src:$PWD/externals/SpecForge"
export FLASHINFER_WORKSPACE_BASE=/tmp/fast_infer_cache/flashinfer
export TRITON_CACHE_DIR=/tmp/fast_infer_cache/triton
export TORCH_EXTENSIONS_DIR=/tmp/fast_infer_cache/torch_extensions

python3 scripts/mr_dflash/run_phase1_smoke.py \
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/pilot_prompts.jsonl \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_sglang_smoke_20 \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --num-samples 20 --max-length 8192 --max-new-tokens 256 \
  --device cuda --local-files-only --resume \
  --cache-backend specforge_sglang --cache-attention-backend sdpa \
  --cache-auto-batch --cache-auto-batch-start-size 1 \
  --cache-auto-batch-safety-fraction 0.95 \
  --cache-auto-batch-target-vram-gb 160 \
  --cache-auto-batch-hard-vram-gb 163 \
  --cache-auto-batch-max-size 128
```

Đường accelerated này cần SGLang/SpecForge, `flashinfer-python`,
`flashinfer-cubin`, `sglang-kernel` và các binary khớp torch/CUDA. Với manifest
hiện tại, `sglang-kernel` phải là wheel cu130 trong offline wheelhouse; không
dùng wheel generic khác CUDA. `setup_server_env.py --all` chỉ là import
preflight, còn lệnh trên mới xác nhận model load, static pool, hidden capture,
writer và resume contract.

`setup_server_env.py --all` là preflight cho toàn bộ benchmark, nên vẫn kiểm
tra cả các module tùy chọn của baseline khác. Nếu mục tiêu hiện tại chỉ là
MR-DFlash smoke HF/SDPA, có thể kiểm tra trực tiếp các package tối giản và
chạy lệnh phase 1 ở trên; thiếu `flash-attn` hoặc `sgl_kernel` không phải lỗi
của đường cache HF.

## Khởi tạo/kiểm tra

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

# Tạo thư mục/link/master nếu còn thiếu; không chạy preflight.
python3 scripts/setup_server_env.py --init

# Kiểm tra Python 3.12, package, master config và LongBench.
python3 scripts/setup_server_env.py --check

# Hoặc init + check trong một lần. `--all` cũng kiểm tra các dependency và
# module MR-DFlash dùng cho train/cache/inference theo chế độ import-only;
# không load Qwen3-4B và không chiếm GPU cho runtime smoke.
python3 scripts/setup_server_env.py --all
```

`--all` kiểm tra các package runtime bắt buộc, bao gồm `torch`,
`transformers`, `PyYAML`, `safetensors`, `tqdm` và các module trong
`src/MR_DFlash`. Đây là preflight thư viện/source, không phải kiểm tra model,
cache hay checkpoint cụ thể. Khi cần xác nhận GPU B200 và asset của baseline,
dùng thêm `scripts/check_b200_env.py`; khi cần kiểm tra model thật, chạy smoke
riêng của pipeline.

Script không ghi đè phần cấu hình operator-owned và không xoá dataset đã
checkout. Nếu master config đã được tạo bởi script này, `--init` sẽ refresh
block managed defaults (bao gồm `longbench_100_14k`); các đường dẫn model,
draft model, MagicDec `.pth` và SSSD datastore vẫn do operator giữ nguyên.

### Preflight vLLM/SGLang cho MR-DFlash

Server runtime hiện hành phải resolve đúng vLLM `0.24.0`, SGLang `0.5.14` và
adapter/patch SpecForge từ môi trường đã cài sẵn. Worker regenerate vLLM gọi
OpenAI-compatible `/v1/completions`; worker không tự cài hoặc tự tải model.

```bash
python3 - <<'PY'
import importlib.metadata as metadata
for name in ("vllm", "sglang"):
    print(name, metadata.version(name))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
PY

python3 scripts/mr_dflash/vllm_regenerate.py --help >/dev/null
python3 scripts/mr_dflash/parallel_stage.py --help >/dev/null
```

Không đưa source SGLang vendored khác vào `PYTHONPATH`; chỉ thêm
`externals/SpecForge` theo hướng dẫn khi adapter của SpecForge yêu cầu. Mỗi
vLLM server phải chạy trên GPU riêng bằng `CUDA_VISIBLE_DEVICES`; dừng server
farm trước khi dùng chính các GPU đó cho SGLang hidden-state cache.

## Chạy benchmark LongBench

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

python3 scripts/run_longbench_200.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
  --mode smoke \
  --run-id b200-smoke-all
```

Sau khi smoke pass, dùng `--mode representative` hoặc `--mode full`. Kết quả
được ghi dưới `outputs/longbench_100_14k/<run-id>/`; xem `run_manifest.json` và
`logs/` để kiểm tra từng cell.

Các lỗi import/tương thích đã được xử lý trong source vendored và adapter nên
không cần cài thêm `dflash`, `MagicDec`, `termcolor` hoặc `fastchat` từ
internet. Sau khi đồng bộ code lên server, chạy lại:

```bash
python3 -m py_compile \
  scripts/infer_dflash.py scripts/infer_magicdec.py scripts/infer_longspec.py \
  scripts/common/model_compat.py
python3 scripts/setup_server_env.py --check
```

SSSD là ngoại lệ về dữ liệu: datastore `.idx` là artifact riêng, không nằm
trong repository. Để smoke không bị bỏ qua, có thể để trống
`SSSD_DATASTORE_PATH`; để benchmark retrieval công bằng, phải điền đường dẫn
`.idx` được build cho Llama 3.1 và tokenizer đang dùng.

## Ghi chú bảo mật

Không commit `HF_TOKEN`, thông tin đăng nhập, hoặc đường dẫn chứa secret vào
repository. Chỉ commit tài liệu path/runtime ổn định và các file cấu hình mẫu.
