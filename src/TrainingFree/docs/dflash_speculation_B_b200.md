# Thực nghiệm B — DFlash uncertainty và verification depth trên B200

Ngày cập nhật: **2026-10-14**.

Harness có CLI cho B0, B1, B2, B1-V và các phase E2E có điều kiện B3/B4. Logic
được kiểm tra trên CPU; chưa chạy cache-prefix validation hay timing GPU trên
B200. B0 phải hoàn thành các lượt chạy và ghi chẩn đoán AR/native/adapter; khác
token với AR không chặn phase sau. Lỗi runtime hoặc bất biến verifier/cache
vẫn chặn phase bị ảnh hưởng.

## Bố cục

| Mode | Làm gì | Kết quả |
|---|---|---|
| `prepare` | Đọc corpus, exposure manifests và tạo source-group cohort/split | `campaign.json`, `cohort_manifest.json`, `split_manifest.json` |
| `preflight` | Nạp target/DFlash từ local snapshot, kiểm runtime/checkpoint/backend | `runtime.json` |
| `validate` | So target-only AR, native DFlash và adapter trên gamma 3/7/11/15; gamma 15 tạo reference rollout | `evidence/parity.json`, `traces/*.json` |
| `b1` | Replay từng gamma bằng draft forward riêng trên cùng prefix/cache/bank; gamma 15 thu signal ở lượt riêng | Bảng B1, vector DFlash/parent signals |
| `b1v` | Draft gamma 15 rồi verify prefix k=3/7/11/15 trên cache fork riêng; kiểm tra L và pending anchor | Bảng B1-V |
| `analyze` | Tính oracle/fixed gaps, bootstrap theo source group, grouped OOF survival/Brier và quyết định screening | `reports/*`, không nạp model |
| `run` | Chạy prepare, B0, B1, B1-V rồi analyze tuần tự | Toàn bộ artifacts trên |
| `all` | Chạy B0/B1/B2/B1-V, B3/B4 đủ điều kiện rồi tự chuẩn bị và chạy confirm trên cùng runtime | `all_status.json`, artifacts discovery và confirm |
| `b3-dev` | Chỉ mở khi B1-V và B2 qua gate; chạy AR, fixed gamma/k, regime/history controls và policy signal trên dev; khóa model/cost/fallback | `b3/dev_requests.jsonl`, `b3/locked_config.json` |
| `b3-confirm` | Chạy lock B3 đã chọn trên cohort confirm riêng; paired E2E, ROUGE và tỉ lệ khớp token AR dạng chẩn đoán | `b3/confirm_requests.jsonl`, `b3/confirm_summary.json` |
| `b4` | Ghép uncertainty của đúng round liền trước với state B1 hiện tại; grouped CV chọn gamma trên evaluation timing | `b4/analysis.json`, OOF và lock nếu gate đạt |
| `b4-dev` | E2E draft-length rollout với regime/history controls và lagged policy đủ điều kiện; chọn best fixed gamma/policy trên dev | `b4/dev_requests.jsonl`, `b4/locked_config.json` |
| `b4-confirm` | Chạy B4 lock trên cohort confirm riêng; paired E2E, ROUGE và tỉ lệ khớp token AR dạng chẩn đoán | `b4/confirm_requests.jsonl`, `b4/confirm_summary.json` |

B3 tự đóng gate nếu B1-V chưa có gap ít nhất 5% với CI95 lower >2%, hoặc nếu
`R-DH`/`R-DC` không có Brier skill tối thiểu 5% với CI improvement >0 trên
source groups held-out. Sau warmup dev, B3 tính lại opportunity của B1-V khi
cộng overhead extraction/controller lớn nhất đã đo trong các policy eligible;
nếu gap điều chỉnh không giữ ngưỡng 5% / CI lower >2%, E2E measured dừng.
`R-A` chạy như history-only diagnostic; do history ở
B3 được cập nhật từ rollout có k biến đổi, nó không được chọn làm policy khóa
khi chưa có calibration riêng cho history deployable. Controller đã khóa dùng
cost table B1-V, cộng extraction và callback overhead được đo ở warmup dev.

B4 chỉ ghép round `t-1` nếu trace gamma15 có cả `t-1` và `t` cùng request.
Cost model dùng selection repetitions; evaluation repetitions chỉ chấm OOF
cost/commit theo gamma. Trong rollout, nếu gamma của round trước khác 15 thì
policy fallback về gamma cố định đã khóa vì chưa có dữ liệu calibration cho
shape trước khác. Đây là giới hạn support được ghi vào từng round.

## Chạy trên server

Để treo toàn bộ phần B, cập nhật `scripts/run_dflash_speculation_B.py` có mode
`all`, rồi chạy từ thư mục repo hiện tại:

```bash
B_RUN_ID="B_full_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p outputs/dflash_speculation_B
nohup env PYTHONUNBUFFERED=1 bash scripts/run_dflash_speculation_B_b200.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
  --gpu-ids 0 --mode all --run-id "$B_RUN_ID" \
  --data-dir /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  > "outputs/dflash_speculation_B/$B_RUN_ID.log" 2>&1 < /dev/null &
echo "PID=$!; log=outputs/dflash_speculation_B/$B_RUN_ID.log"
```

`all` dùng cohort discovery 24 nhóm và, khi có policy lock hợp lệ, cohort
confirm 48 nhóm với cùng split manifest. Model chỉ nạp một lần. Gate B3/B4
không đạt được ghi là `not_applicable` và tiến trình đi tiếp; lỗi cục bộ hoặc
thiếu coverage được ghi là `partial`, nhánh độc lập vẫn được thu thập. Lỗi
GPU, filesystem, dependency hoặc chữ ký/verifier/cache nghiêm trọng vẫn dừng.
`all_status.json` phân biệt tiến trình đã kết thúc với dữ liệu có đầy đủ hay
không; không coi cell lỗi là kết quả hợp lệ. Confirm B0 thiếu coverage sẽ chặn
nhánh confirm liên quan nhưng không làm mất dữ liệu đã lưu.

Log nằm cạnh thư mục run; artifacts discovery nằm trong `$B_RUN_ID`, confirm
nằm trong `${B_RUN_ID}_confirm`. Xem tiến độ bằng:

```bash
tail -f "outputs/dflash_speculation_B/$B_RUN_ID.log"
```

Ctrl+C thoát `tail` trong terminal vẫn giữ tiến trình `nohup` đang chạy.
Để tiếp tục đúng cả workflow sau khi hạ tầng bị ngắt, dùng lại cùng lệnh
`--mode all`, cùng run ID/data/exposure manifest và thêm `--resume`.

Đảm bảo master config trỏ tới Python 3.12, local target/DFlash snapshots, FA4
và corpus. Mặc định corpus là
`/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k`;
có thể truyền `--data-dir` hoặc `--data-file`. Exposure manifest phải được rà soát
chung với campaign A/B. Manifest đã có trong repo để bắt đầu là
`data/dflash_context_A_audit/combined_exposure_manifest.json`; nếu có audit
mới hơn, dùng manifest mới đó.

`--data-dir`/`--data-file` tường minh thay thế các đường dẫn dữ liệu cũ trong
master env; chỉ truyền một trong hai flag. Điều này tránh lỗi khi master đang
đồng thời export `DATA_INPUT`/`LONG_BENCH_DATA_DIR` và `DATA_FILE`/`LONG_BENCH_DATA_FILE`.

Lệnh pilot một lượt:

```bash
bash scripts/run_dflash_speculation_B_b200.sh \
  --config /path/to/master.env \
  --gpu-ids 0 \
  --mode run \
  --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --max-new-tokens 1024 \
  --validation-tokens 128 \
  --max-prefixes 6 \
  --warmups 3 \
  --repetitions 6 \
  --selection-repetitions 3
```

`pilot` mặc định chọn 2 source groups/dataset. Lệnh in `run_id` và thư mục
artifact ngay khi tạo cohort. Để chạy từng phase, dùng cùng `--run-id`, corpus,
exposure manifests, `--cohort-phase` và protocol options:

Khi một run đã có `campaign.json` và cần chạy tiếp sau khi bị ngắt, dùng lệnh
ngắn dưới đây. Runner nạp cohort và protocol đã lưu, xác nhận corpus vẫn đúng,
rồi resume B0/B1/B1-V/analyze; không cần truyền lại exposure manifest:

```bash
bash scripts/run_dflash_speculation_B_b200.sh \
  --config /path/to/master.env \
  --gpu-ids 0 \
  --mode run \
  --run-id B_main_02 \
  --resume
```

Nếu corpus không được khai báo trong master config, thêm đúng `--data-dir` hoặc
`--data-file` đã dùng cho run ban đầu.

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode prepare --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode preflight --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode validate --run-id B_pilot_01 --cohort-phase pilot \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b1 --run-id B_pilot_01 --cohort-phase pilot --resume \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b1v --run-id B_pilot_01 --cohort-phase pilot --resume \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --mode analyze --run-id B_pilot_01
```

`--resume` chỉ tái dùng artifact khớp campaign/code/runtime signature. Thay model,
backend, cohort, gamma grid, prompt, protocol hoặc source code liên quan phép đo
thì trace/cell cũ được lưu dưới `stale/` rồi tính lại; code signature cũ không
được gắn lại cho cell mới vì không chứng minh được thay đổi chỉ nằm ở orchestration.
Lưới gamma hiện khóa ở 3/7/11/15; CLI từ chối lưới rút gọn để tránh ngầm đổi
protocol. B3/B4 tự tái dùng E2E cell theo sample khi chữ ký khớp, không cần
`--resume`; cell cũ khác chữ ký được archive và tính lại. B3 dev lưu warmup
overhead profile để giữ nguyên policy cost khi tiếp tục. Các phase khóa policy
vẫn từ chối dùng runtime/model/protocol sai chữ ký thay vì ghép artifacts khác cấu hình.

Trong B0/B1/B1-V, lỗi chỉ ảnh hưởng một sample/state được ghi lại rồi runner
dọn bộ nhớ tạm và tiếp tục các cell độc lập còn lại; phép tính được retry tối đa
hai lần trước khi đánh dấu lỗi. File JSON hỏng được chuyển vào `stale/<phase>/`
và tính lại. B0 ghi lỗi vào
`evidence/parity.json`; B1/B1-V ghi state lỗi thành cell `status=failed`.
`run_status.json` cho biết lượt thu thập là `complete` hay `partial`, số sample/state
đạt và lỗi; cell lỗi sẽ được thử lại khi chạy tiếp với `--resume`. Cell hoàn tất
cùng code/runtime/protocol được giữ lại nếu dữ liệu trong cell đủ số gamma/k và
repetition. Nếu thu tín hiệu B2 lỗi sau hai lần thử, timing B1 vẫn được lưu;
B2 coverage bị giảm và trạng thái tổng hợp chuyển thành `partial`. Lỗi hỏng CUDA context, không còn reference
trajectory hợp lệ, sai runtime/model/corpus/protocol hoặc không thể ghi artifact
vẫn dừng lượt chạy để tránh báo cáo kết quả không đáng tin. Lỗi phân tích hoặc
ghi file trạng thái cuối không làm mất các phép đo đã lưu; raw artifacts được giữ
để phân tích lại; lỗi phân tích được ghi ở `reports/analysis_status.json`.

### Dev / confirm cho B3

Chạy discovery trên đúng cohort `dev`; `run` đã bao gồm B0, B1, B1-V và
analysis. B3 tự từ chối nếu gate B1-V/B2 chưa đạt. `b3-dev` cần ít nhất ba
E2E repetitions (mặc định 5) và lưu policy lock. Output token IDs/hash, tỉ lệ
khớp AR và ROUGE được lưu để đánh giá; khác token với AR không làm phase thất bại:

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode run --run-id B_dev_01 --cohort-phase dev \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b3-dev --run-id B_dev_01 --cohort-phase dev \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --e2e-repetitions 5 --e2e-warmups 3
```

Confirm dùng `run_id` mới và cohort `confirm`. Chạy B0 trước, sau đó đưa đúng
lock từ dev vào `--policy-lock`:

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode prepare --run-id B_confirm_01 --cohort-phase confirm \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode validate --run-id B_confirm_01 --cohort-phase confirm \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b3-confirm --run-id B_confirm_01 --cohort-phase confirm \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --policy-lock outputs/dflash_speculation_B/B_dev_01/b3/locked_config.json \
  --e2e-repetitions 5 --e2e-warmups 3
```

### Dev / confirm cho B4

Trace B0 gamma15 đã bật collector entropy/confidence cho mọi round. B4 ghép
tín hiệu B2 của hai round liền kề chỉ khi tiến triển token output khớp
accepted prefix cộng pending anchor. Khác token với AR không chặn phân tích.
Chạy grouped cost/regret analysis
trên dev sau `run`, rồi chỉ mở E2E nếu ít nhất một lagged model có cost skill
>=2% so regime+history và CI95 lower >0:

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b4 --run-id B_dev_01 --cohort-phase dev \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json

bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b4-dev --run-id B_dev_01 --cohort-phase dev \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --e2e-repetitions 5 --e2e-warmups 3
```

Sau đó chuẩn bị cohort confirm mới, chạy `validate` như B3, rồi:

```bash
bash scripts/run_dflash_speculation_B_b200.sh --config /path/to/master.env \
  --gpu-ids 0 --mode b4-confirm --run-id B_confirm_01 --cohort-phase confirm \
  --exposure-manifest data/dflash_context_A_audit/combined_exposure_manifest.json \
  --policy-lock outputs/dflash_speculation_B/B_dev_01/b4/locked_config.json \
  --e2e-repetitions 5 --e2e-warmups 3
```

B3/B4 requests lưu output token IDs/hash, tỉ lệ khớp AR dạng chẩn đoán và ROUGE
khi có reference; rounds lưu gamma, k, tín hiệu, fallback, accepted/committed
tokens và overhead. Summary gộp ROUGE theo variant. Lock xác nhận
code/runtime signature, corpus/exposure hash và cấu hình E2E (token limit, gamma,
backend, seed, số repetition/warmup); confirm từ chối cấu hình lệch hoặc source
group trùng dev. E2E không tự động gộp giữa nhiều GPU; mỗi paired campaign cần
cùng GPU/runtime identity.

## Đọc artifacts và kết luận

- `evidence/parity.json`: AR/native/adapter token comparisons theo document/gamma.
  AR-token mismatch là diagnostic; lỗi cục bộ trong một replay được ghi riêng,
  còn lỗi CUDA làm hỏng context hoặc sai integrity toàn campaign mới chặn B1/B1-V.
  Prefix B1/B1-V lấy token từ `adapter_ids`, cùng luồng với `adapter_rounds`;
  anchor so với greedy token khi dựng lại full prefill được ghi diagnostic, không
  chặn run. Native DFlash và adapter vẫn phải cùng trajectory.
- `run_status.json`: trạng thái đầy đủ/thiếu một phần của lượt thu thập. Cell
  `status=failed` trong `b1/cells/` hoặc `b1v/cells/` có loại lỗi và traceback;
  cell đó không được đưa vào bảng số liệu.
- `b1/draft_shape.jsonl`: cùng `state_id` có bốn gamma, proposals riêng từng
  shape, accepted candidates và round/component timing. `round_ms` gồm draft,
  full-context gather, verification, cache crop và draft-bank append; cache
  fork/rebuild nằm ngoài span và được ghi riêng.
- `b2/signals.jsonl`: entropy thô/normalized, confidence, probability/logit
  margin ở từng candidate; parent-target signals; history chỉ từ reference
  gamma 15 trước round. Candidate sau rejection chỉ tham gia survival label,
  không được tính là rejection độc lập ở vị trí không reach.
- `b1v/verify_prefix.jsonl`: cùng proposals gamma 15 cho cả k, verify riêng
  prefix và cache fork. Mỗi cell phải thỏa `L_k=min(L_15,k)` và cache crop
  length; pending-anchor comparison với output tham chiếu được lưu diagnostic.
- `reports/b1_metrics.json`: oracle gamma chọn bằng 3 selection repetitions,
  gap/regret đo trên 3 evaluation repetitions, near-optimal fixed coverage,
  tail acceptance, proposal-prefix equality và CI theo source group.
- `reports/b2_metrics.json`: Brier/survival out-of-fold theo document group,
  fallback/support counts, candidate risk set và tỷ lệ confident-but-wrong.
  So entropy với regime R, rồi so `R-A-DH` với `R-A` để kiểm tra entropy có
  thêm thông tin sau history hay không.
- `reports/b1v_metrics.json`: oracle k-gap dùng cùng cách tách selection/evaluation
  như B1. Đây là local cost opportunity, chưa tính chi phí chọn k ở controller.
- `reports/decision.md`: screening theo ngưỡng đã khai báo trong campaign;
  CI rộng hoặc cohort thiếu sẽ cho `inconclusive`, không tự biến thành kết quả âm.
- `b3/dev_requests.jsonl`, `b3/confirm_requests.jsonl`: request paired AR/fixed/adaptive,
  AR-token match diagnostic, ROUGE, E2E wall, generation breakdown, counters và output IDs. `b3/locked_config.json`
  chứa model B2 đã fit trên dev, B1-V cost table, best fixed gamma/k và signature.
- `b4/lagged_training_rows.jsonl`, `b4/oof_predictions.jsonl`, `b4/analysis.json`:
  chỉ ghép round liền kề, đo B1 gamma-specific cost/commit trên groups held-out;
  E2E B4 khóa model trên dev; confirm ghi quality metrics và AR-token match diagnostic.

Ngưỡng B1 đề xuất: gap J ít nhất 5% và CI lower trên 2% để mở nhánh
opportunity; CI upper không quá 2% là ít headroom trong grid/workload hiện tại.
B2: entropy cần Brier skill tối thiểu 5% so R với CI improvement dương; để nói
entropy bổ sung sau history cần tối thiểu 2% so `R-A` và CI dương. Bootstrap
đơn vị source group/document, không phải round hay timing repetition.

Pilot 6 documents chỉ dùng để debug runtime/ước lượng chi phí; grouped folds
và CI từ pilot rất thô. Không dùng pilot làm paper claim. Khi B0 báo token
divergence, đọc first-mismatch fields như chẩn đoán và báo ROUGE trên reference;
lỗi runtime, instrumentation hoặc bất biến cache/verifier vẫn cần được xử lý
trước khi dùng kết quả phase bị ảnh hưởng. CPU test và CLI không chứng minh GPU
correctness/performance; chưa có kết quả chạy trên B200. Outputs B3/B4 có thể
khác AR; report ROUGE theo reference và tỉ lệ khớp AR thay vì giả định chất lượng
giống hệt AR.
