# B200 GPU Readiness Design

## Mục tiêu

Làm cho toàn bộ quy trình benchmark của repository có một đường chạy rõ ràng
trên server GPU B200 dùng `python3` từ PATH; đồng thời cung cấp
kiểm tra trước khi chạy và smoke test một sample để phát hiện sớm lỗi
dependency, model/checkpoint, CUDA, kernel hoặc output schema.

`.venv` trong workspace hiện tại chỉ là môi trường mô phỏng server để kiểm tra
trước; máy Tesla T4 vẫn chỉ được dùng cho code path không cần CUDA, contract và
lỗi launcher. Không coi CPU fallback là bằng chứng EAGLE3, DFlash hoặc một CUDA
kernel chạy được trên B200.

## Phạm vi

- 14 baseline đang được nối vào `scripts/run.sh`.
- Các launcher, config và shared runtime liên quan đến Python `.venv`.
- Preflight cho GPU/server, model cache, checkpoint, package import và cache
  JIT writable.
- Smoke một sample, output JSONL/summary và báo cáo trạng thái từng baseline.

Không thay đổi thuật toán của baseline vendored, không port GPU-only baseline
sang CPU, không tải dependency/model qua internet và không đánh dấu PASS khi
chỉ chạy được bằng mock CUDA.

## Tiêu chí hoàn thành

1. Trên B200, preflight, launcher và child process dùng cùng interpreter
   `python3` được resolve từ PATH; trong mô phỏng local, cùng contract đó được
   chạy bằng `.venv/bin/python`. Cả hai phải là Python 3.12.
2. Có profile B200 riêng, không làm hỏng config T4/CPU hiện có.
3. Preflight phân biệt được:
   - host không có CUDA hoặc GPU không phải target;
   - thiếu package/wheel/compiled extension;
   - thiếu model/checkpoint/dataset;
   - lỗi import/kernel/runtime thực tế.
4. Có một lệnh chạy smoke toàn bộ baseline, tiếp tục chạy baseline sau khi một
   baseline lỗi, lưu log/output riêng và sinh summary machine-readable.
5. Mỗi baseline được chạy tối đa một sample trong B200 smoke; các phép lặp nội
   bộ của kernel benchmark phải được ghi rõ là repetition, không phải sample.
6. GPU-only baseline giữ nguyên CUDA guard; không có shortcut biến lỗi CUDA
   thành PASS.
7. Khi chạy trên B200 thật, preflight và smoke phải kiểm tra được CUDA path
   thực tế, dtype/device placement, custom kernel import/forward, output hữu hạn
   và schema cuối.

## Thiết kế

### 1. Profile server B200

Thêm `config/b200.env` làm lớp cấu hình server, gồm:

- `FAST_INFER_PYTHON="python3"` hoặc đường dẫn tuyệt đối tương ứng trên server;
  không yêu cầu activate virtualenv;
- `CUDA_VISIBLE_DEVICES` và device mặc định;
- `HF_HOME`, `TRANSFORMERS_CACHE`, `TRITON_CACHE_DIR`,
  `FLASHINFER_WORKSPACE_BASE`, `TORCH_EXTENSIONS_DIR` trỏ tới nơi writable;
- model/checkpoint canonical M1–M9 đã có trong server cache;
- giới hạn smoke `MAX_SAMPLES=1`, input/output token ngắn;
- danh sách baseline và dataset được phép chạy.

Profile này được truyền vào runner bằng `--config` hoặc biến môi trường; các
config baseline gốc vẫn giữ default của chúng để tương thích dev/T4. Khi mô
phỏng local, caller override rõ ràng
`FAST_INFER_PYTHON="$PWD/.venv/bin/python"`; không thay đổi profile production
thành `.venv`.

### 2. GPU preflight

Thêm một preflight dùng interpreter đã resolve (`python3` trên server hoặc
`.venv/bin/python` khi mô phỏng) để kiểm tra theo thứ tự:

1. Python 3.12 và executable được resolve từ shared runtime.
2. `torch.version.cuda`, `torch.cuda.is_available()`, số GPU, tên GPU,
   compute capability và VRAM.
3. B200 target check theo đặc tính runtime được báo bởi PyTorch/NVIDIA, nhưng
   không suy đoán rằng T4 là B200 khi chạy mô phỏng.
4. Import các package chung và package/extension riêng của từng baseline được
   chọn.
5. Tồn tại và cấu hình tương thích của target model, draft model, EAGLE/DFlash
   checkpoint và dữ liệu.
6. Khả năng ghi các thư mục cache JIT; không để lỗi cache read-only bị nhầm là
   lỗi kernel.
7. Một probe tensor nhỏ trên CUDA cho các backend có thể probe an toàn.

Kết quả gồm text để người dùng đọc và JSON summary để runner dùng. Exit code
khác 0 khi host chưa đủ điều kiện; preflight không tự bỏ qua lỗi bắt buộc.

### 3. B200 smoke runner

Thêm runner điều phối theo từng baseline:

- resolve một interpreter duy nhất một lần (`python3` trên server, `.venv/bin/python`
  khi mô phỏng);
- chạy preflight chung trước;
- tạo output/log path riêng có timestamp hoặc run id;
- gọi lại launcher chuẩn, không gọi trực tiếp implementation vendored;
- ép một sample và generation ngắn;
- bắt exit code, timeout và summary cuối của từng baseline;
- tiếp tục baseline kế tiếp;
- sinh `b200_smoke_summary.json` với trạng thái `PASS`, `BLOCKED` hoặc
  `FAIL`, reason và artifact path.

`BLOCKED` dành cho thiếu phần cứng/dependency/model đã xác định trước; `FAIL`
dành cho code đã vào runtime nhưng lỗi bất ngờ; `PASS` chỉ khi output/schema
và các check runtime của baseline đều thành công.

### 4. Launcher và contract

Rà soát từng wrapper để:

- không hard-code đường dẫn Python khác `.venv`;
- không hard-code device ngoài trường hợp baseline bắt buộc single-GPU và có
  comment rõ ràng;
- export đúng `PYTHONPATH` cho repo vendored;
- truyền đủ model/checkpoint/data/token limits từ profile;
- kiểm tra file local trước khi nạp nhiều GB weights;
- ghi lỗi có thể hành động thay vì traceback thiếu context.

EAGLE3 và DFlash vẫn là GPU-only. B200 smoke phải chạy đúng CUDA path của chúng
và không thêm CPU fallback giả. Các baseline thiếu compiled dependency phải bị
preflight chặn với tên package và lệnh kiểm tra cụ thể.

### 5. Kiểm thử

- Contract tests cho profile, preflight output/exit semantics, launcher command
  construction, one-sample limit và output summary.
- Regression tests giữ các fix hiện có cho RocketKV, DFlash PYTHONPATH và
  semantic-selection fixture.
- Static checks: `bash -n`, `.venv/bin/python -m compileall`, `pytest`,
  `git diff --check`.
- Trên máy T4: chạy preflight ở chế độ “hardware unavailable” để xác nhận
  chẩn đoán chính xác; chạy CPU-safe smoke hiện có.
- Trên B200 thật: chạy preflight, sau đó B200 smoke 14 baseline. Chỉ lượt này
  mới được ghi là xác nhận CUDA/kernel compatibility.

## Luồng vận hành

```text
config/b200.env
        |
        v
shared .venv + GPU preflight ---- fail --> BLOCKED report
        |
        v
one-sample launcher matrix
        |
        +--> per-baseline logs/output/exit code
        |
        v
b200_smoke_summary.json ---- PASS only after real CUDA + schema checks
```

## Rủi ro và xử lý

- Không có B200 trong workspace hiện tại: không tuyên bố kernel pass; chỉ tạo
  và chạy được validation contract/hardware-unavailable path.
- Wheel CUDA/compiled extension khác nhau theo server: preflight báo exact
  import/version thay vì tự cài online.
- Model gated hoặc chưa mirror: profile cho phép local snapshot override và
  báo missing asset trước khi load.
- Baseline upstream có CLI/ABI riêng: wrapper là boundary duy nhất; không đưa
  logic thích nghi rải vào runner.
