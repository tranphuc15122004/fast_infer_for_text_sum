# Thiết kế thực thi thực nghiệm bằng vLLM

Ngày: 15/10/2026. Phạm vi ưu tiên: DFlash A1–A3, B0–B4 và các baseline
speculative đã có trong runner vLLM. Người dùng đã yêu cầu triển khai theo
`requirements.txt`; tài liệu mô tả đích triển khai, không xác nhận GPU parity.

## 1. Quyết định kiến trúc

Chọn bộ điều phối offline trong Python và adapter theo phiên bản vLLM.
Target forward, quản lý request, paged KV, verification và sampling do vLLM
thực hiện. Phương pháp nghiên cứu phải chạy trong worker, không gửi một HTTP
request mới cho từng token/round. Một engine được giữ sống qua nhiều mẫu.

Ba phương án đã cân nhắc:

| Phương án | Đánh đổi | Quyết định |
|---|---|---|
| Chỉ dùng API generation chuẩn | Nhanh triển khai; không biểu diễn can thiệp KV và B3/B4 | Dùng cho baseline chuẩn |
| Một bộ điều phối + adapter worker vLLM | Tái sử dụng scheduler/kernel, giữ semantics nghiên cứu; cần kiểm chứng adapter | Chọn |
| Viết lại engine hoặc thay tất cả bằng DFlash chuẩn | Chi phí lớn hoặc thay đổi câu hỏi thí nghiệm | Không chọn |

Không coi FlashAttention backend của vLLM là FA4 của executor HF. Mọi artifact
ghi backend thực tế. So sánh chính diễn ra trong cùng engine, dtype, cấu hình
attention, prompt token IDs và chế độ đo. So sánh HF/vLLM là bảng riêng.

## 2. Runtime và tương thích

- Python 3.12, `python3` từ PATH trên server; lựa chọn interpreter theo
  `scripts/common/runtime.sh` và master ngoài repo qua `config/master.path`.
- Manifest hiện khai báo vLLM wheel **0.30.0**, torch **2.13.0**,
  transformers **5.12.1**, flash-attn-4 **4.0.0b19**. Không đổi manifest để
  hợp thức hóa một adapter viết cho phiên bản khác.
- Đọc metadata/version và chữ ký API thực tế. Không import binary CUDA để
  thực hiện kiểm tra static CPU. `docs/server_environment.md` còn nhắc 0.24;
  đó không phải bằng chứng server hiện đang chạy 0.30.
- Baseline dùng API công khai. Adapter nội bộ chỉ bật trên phiên bản đã có
  kiểm thử; không thử patch tương tự một cách im lặng trên bản khác.
- Không cài online, không tạo venv từng baseline, không tải model/dataset.
- Plugin phải được nạp cả trong worker subprocess. Không sửa trực tiếp
  `site-packages`; bootstrap nằm trong repo, có cấu hình opt-in và idempotent.
- Đường custom ban đầu nhắm Qwen3 target + DFlash Qwen3, TP=1. Model/backend
  khác được báo khả năng cụ thể, không giả định cùng cache layout.

## 3. Ma trận thí nghiệm và semantics

| Phần | Công việc bắt buộc | Vị trí thực thi |
|---|---|---|
| AR, EAGLE3, DFlash, Domino, DSpark | Generation native và metrics | Engine chuẩn đã có |
| A1 | Region controls, matched deletion, target attention high/low theo layer | Snapshot + draft attention adapter |
| A2 | Random/recency/target-parent/previous-draft/current-draft oracle, cùng budget | Snapshot + selector |
| A3 replay | Budget và tuổi ranking trên cùng prefix | Snapshot + selector |
| A3 E2E | Full và reduced context, dense refresh period 2/4/8/16 | Proposer + draft attention adapter |
| B0 | Native full reference, token/round trace và smoke adapter | Engine + trace collector |
| B1 | Sweep gamma 3/7/11/15 từ cùng state; mỗi gamma có draft riêng | Snapshot + proposer |
| B1-V | Draft 15 một lần, verify các prefix 3/7/11/15 | Snapshot + verifier adapter |
| B2 | Survival/prediction, source-group folds, cost analysis | CPU, tái dùng analysis hiện có |
| B3 | Draft 15; chọn k **sau draft**, trước target verify | Proposer/runner với độ dài proposal hiệu lực |
| B4 | Chọn gamma **trước draft** bằng tín hiệu round trước | Proposer, state riêng request |

Không thay B3 bằng draft-k và không thay B4 bằng quy tắc theo batch size.
Không pad rồi verify đủ 15 và báo là đã tiết kiệm verification. Ghi riêng
`drafted_candidates`, `verified_candidates`, `accepted_candidates`,
`committed_tokens`; phân biệt anchor/bonus với accepted draft candidates.

Các grid A lấy từ `context_interventions/matrix.py`; B lấy từ
`speculation_experiments/protocol.py`. `full` giữ toàn bộ sample/grid đã chọn,
không tự giảm context/output horizon hoặc bỏ các arm nghiên cứu.

## 4. Context adapter

Target luôn đọc full KV. Chỉ tập key mà drafter đọc thay đổi. Full bank vẫn
được cập nhật từ hidden states của token đã commit; không nhét KV của mask
token bị reject vào bank. Target-layer mapping lấy từ checkpoint, kiểm tra
quy ước layer-output +1 ở vLLM, không hardcode một checkpoint bất kỳ.

Mỗi request có absolute positions, committed length, ranking theo layer,
round index và history riêng. Budget gồm protected keys, loại trùng; current
anchor/mask block được cộng theo semantics checkpoint. Gather KV giữ RoPE
gốc, không đánh số lại key đã chọn. Không dùng physical paged slot làm
logical position hoặc request identity.

`previous_dflash`: bootstrap/dense refresh thực sự đọc full context và sinh
proposal được verify; ranking lấy từ chính forward ấy. Sparse round dùng
ranking dense gần nhất cộng protected/newly committed keys. Không chạy một
dense forward phụ rồi bỏ output để tính ranking miễn phí. Score attention
phải giữ đúng mẫu số gồm block và các mask causal/noncausal của checkpoint.

Full adapter là control riêng để tách chi phí adapter. Full no-op phải dùng
đường native khi không thu tín hiệu. Collector Q/K/attention chỉ hoạt động
khi phương pháp yêu cầu; tính lại QK, top-k, gather, history và controller đều
nằm trong timing E2E. Bộ đếm bytes phân biệt full resident bank và KV đọc;
giảm working set không tự suy thành giảm resident VRAM.

## 5. Replay cùng state

Generation E2E và replay là hai job độc lập; lỗi replay không ngăn E2E.
Replay worker batch-1 giữ một reference trajectory sống, pause tại prefix
được chọn, chạy các action từ snapshot bất biến rồi tiếp tục reference.
Mọi action nhận cùng target KV, draft bank, anchor, positions, history.

Không clone toàn context cho từng action nếu có thể chia sẻ phần prefix
chỉ đọc. Các write vào tail có vùng scratch riêng; page cuối chưa đầy phải
được copy trước khi ghi. Metadata và bank length được rollback sau action.
Nếu backend chưa có primitive fork an toàn, adapter copy active prefix một
lần vào snapshot, tái dùng scratch cho các action; ghi snapshot/setup cost
riêng, không gộp vào online speedup. Không giữ snapshot của toàn corpus.

Prefix schedule dùng first/second round và quantile progress theo reference
output; có thể cần một trace pass riêng sau E2E. Pass thu snapshot không
được tái dùng timing làm baseline sạch. Nếu trace pass khác trajectory E2E,
ghi hai hash và dùng prefix/anchor thực tế của snapshot cho mọi action.
Không lấy accepted count ở trajectory khác làm chuẩn local bắt buộc.

Trong mỗi prefix, Full local là reference; lặp Full cuối sweep để phát hiện
mutation. Mismatch với trace cũ là diagnostic. Mutation/cache corruption làm
vô hiệu prefix và restart worker; không dùng state hỏng cho action kế tiếp.
Previous ranking được lấy từ round thực trước đó, không suy từ một prefix
đã lấy mẫu cách xa nhiều round. B4 cũng chỉ dùng cặp round liền kề.

## 6. Hai phép đo

`latency`: max_num_seqs=1, một engine trên GPU, không chạy đồng thời worker
khác trên GPU ấy. Dùng để báo E2E/TTFT/TPOT và speculative speedup batch-1.

`throughput`: đưa nhiều prompt vào engine, vLLM continuous batching; cấu hình
khởi đầu max_num_seqs=16, token budget có thể override. Báo tổng output
tokens / wall time của tập request, QPS, latency p50/p95, concurrency thực
quan sát. Không chia wall time của batch cho số request rồi gọi là latency.

Profile đầy đủ có thể chạy cả hai theo các stage riêng; artifact không gộp
hai chế độ. Khi chỉ có một GPU, các stage chạy nối tiếp. Có nhiều GPU thì
shard sample theo source group vào các engine độc lập (TP=1), cùng một
measurement/method plan; không hardcode GPU 0. Một reference timing chỉ được
tái dùng cho cùng GPU, engine configuration, measurement, repetition và
thời đoạn đo đã ghi. Không dùng baseline idle so với variant đang tranh GPU.

VRAM mặc định tối đa 0.85 cho một engine/GPU; tính cả workspace, draft bank,
scratch và graph buffers. Trần này là ngân sách cấp phát, không phải mục
tiêu phải dùng hết. Nếu OOM: giải phóng/restart engine, giảm concurrency
throughput theo 16→8→4→2→1 và ghi attempt; không giảm context/horizon.
Giảm concurrency đổi execution profile: đo lại reference tương ứng trước
khi tính speedup. Latency batch-1 OOM được báo lỗi, không đổi thí nghiệm.

## 7. Bỏ công việc lặp

- Một lượt đo E2E mặc định; số repetitions là lựa chọn operator, không áp
  điều kiện bắt buộc 3/5 lượt để được chạy.
- Warmup sau load theo execution shape, mặc định tối đa ba prompt đại diện,
  sinh 64 token; không warmup ba bài đầy đủ trên mỗi sample/config.
- AR, native Full và full-adapter không chạy lại theo từng refresh period
  khi execution configuration thực sự giống nhau; liên kết bằng reference ID.
- Tái dùng analysis CPU, prompt tokenization và các action hoàn toàn trùng
  semantics. Không alias fixed-gamma với fixed-k nếu forward khác nhau.
- So với A3 dev cũ: 24×4×4×(3+3)=2.304 full generations. Với một budget,
  một lượt đo: 24×(3 controls+4 selected)=168 measured generations mỗi
  measurement, chưa tính warmup/trace. Đây là đếm công việc, không phải
  cam kết speedup GPU. Grid nhiều budget được đếm thêm công khai.

## 8. Gate, lỗi và kết luận

Speedup/CI/ROUGE/Brier/oracle-gap không quyết định có chạy phase sau hay
không. B3/B4 chạy mọi policy fit được, kể cả không có lợi. Dev chọn policy
bằng quy tắc đã ghi trước; heldout dùng bản frozen, không bắt policy phải
thắng mới được đánh giá. Thiếu dữ liệu fit ghi `dependency_missing` cho
policy ấy; các baseline/fixed arm vẫn chạy.

Parity token là diagnostic, không dừng campaign. Ghi first divergence,
ROUGE, valid/invalid count và báo speedup quan sát với correctness status;
không gắn nhãn lossless khi chưa có bằng chứng. Cell lỗi hoặc metric NaN
không được biến thành 0 hoặc đưa vào mẫu số kết quả hợp lệ.

Lỗi request/cell được lưu và tiếp tục; lỗi CUDA nghiêm trọng làm restart
worker, retry tối đa một lần. Lỗi global như thiếu target/data/interpreter,
không ghi được output hoặc cấu hình không biểu diễn được thì dừng job liên
quan và vẫn finalize report. Ctrl-C/SIGTERM bảo toàn partial và exit code.
Không có ngưỡng ba sample lỗi làm dừng toàn campaign. Không retry vô hạn.

Report luôn có planned/attempted/succeeded/failed/unsupported/dependency_missing,
coverage theo dataset/method/config và lỗi cụ thể. `execution_status=complete`
chỉ khi mọi cell có kết quả thành công; mọi planned cell có terminal record
chỉ có nghĩa `campaign_status=finished`. Chế độ report-only cho scientific
diagnostics không biến execution failure thành thành công.

## 9. Artifact và resume

Output dưới `outputs/vllm_research/<run-id>/`. `plan.json` liệt kê toàn bộ
cell trước GPU, model/data hashes, sampling, split, capability và estimate.
`runtime.json` chứa version, API/backend thực tế, GPU, code/adaptor revision.
`results.jsonl` và các shard dùng `io_util.JsonlWriter`, có summary cuối;
generation gọi `rouge.add_rouge()`/`rouge.aggregate_rouge()` khi có reference.
Trace/replay tách file, không ghi vocabulary logits hoặc full KV ra JSON.

`cell_id` từ sample/prompt hash, variant/params, measurement, repetition,
runtime identity và sampling. Atomic checkpoint per request/cell, một writer
per shard; resume skip success và retry failed có giới hạn. Đổi method code,
model, data, dtype, prompt, context/output hoặc profile tạo namespace mới;
không xóa kết quả cũ và không ép hash code cũ khớp bằng một cờ relaxed.
Có `events.jsonl`, `progress.json`, console log và report tiếng Việt.

## 10. Giao diện dự kiến

Lệnh dưới đây là giao diện đích, chỉ dùng khi entrypoint đã được triển khai:

```bash
bash scripts/run.sh vllm_research --full --experiments e2e --measurement throughput
bash scripts/run.sh vllm_research --full --experiments A,B --measurement latency
bash scripts/run.sh vllm_research --plan-only --experiments A,B
bash scripts/run.sh vllm_research --resume RUN_ID
```

Master giữ model/data path; CLI override một lần. Không tạo config master
thứ hai. Runner `vllm_all` hiện tại tiếp tục là đường baseline chuẩn; bổ sung
batching, warmup hữu hạn và diagnostics report-only trước khi nối vào bộ
điều phối nghiên cứu. Capability thiếu không được thay bằng variant chuẩn
cùng tên; terminal record phải ghi rõ thí nghiệm chưa thực hiện được.

## 11. Kiểm chứng và thứ tự thực hiện

1. CPU: lập plan/deduplicate, semantic selectors, cost/gamma/k, resume,
   schema/quality report, lỗi request, batching và plugin propagation bằng
   engine giả có kiểm soát; đối chiếu source vLLM 0.30 trong venv local.
2. B200: một lệnh full tự chạy smoke rất ngắn cho **tính năng đang dùng**,
   rồi tiếp tục collection; không chạy một campaign pilot đầy đủ trước.
3. Kiểm tra Full/no-op, source positions/RoPE, target full KV bất biến,
   mixed request lengths, EOS/tail, rejection/rollback và thứ tự request đổi.
4. B3 phải có drafted=15/verified=k; B4 phải draft gamma thực tế; timer
   gồm collector/controller/gather. Unsupported backend được ghi đúng.
5. B200 đo A3 cùng 24 mẫu, ghi wall/job count, decode tokens/s và VRAM;
   so cả batch-1 và throughput riêng. Không công bố tốc độ dự kiến như số đo.

Máy local CPU không xác nhận CUDA graph, kernels, GPU parity hoặc B200
speedup. Các khả năng ấy chỉ được đánh dấu verified sau artifact server.

## 12. Nguồn đối chiếu

- Code repo: `scripts/run_dflash_context_experiments.py`,
  `src/TrainingFree/context_interventions/{matrix,execution,replay}.py`,
  `scripts/run_dflash_speculation_B.py`,
  `src/TrainingFree/speculation_experiments/{protocol,b_controller,lagged}.py`.
- [vLLM 0.30 DFlash proposer](https://docs.vllm.ai/en/v0.30.0/api/vllm/v1/spec_decode/dflash/):
  source adapter phải khớp đường quản lý input/context của phiên bản này.
- [Dynamic speculative decoding](https://docs.vllm.ai/en/v0.30.0/features/speculative_decoding/dynamic_speculative_decoding/):
  policy theo concurrency không thay thế policy theo tín hiệu nghiên cứu.
- [Per-request acceptance metrics](https://docs.vllm.ai/en/v0.30.0/features/speculative_decoding/acceptance_metrics/):
  dùng metric per request, giữ dữ liệu thô vì schema experimental.
