# Review triển khai AMR-DFlash — 08/10/2026

Phạm vi: working tree hiện hành của `src/AMR_DFlash/`, CLI/launcher AMR,
config chung, output và các contract tests. Review này không sửa mã nguồn.

**Kết luận:** adapter, memory và greedy verifier chạy được trên tiny Qwen3;
pipeline có đường chạy xuyên suốt. Tuy nhiên còn lỗi về tính toàn vẹn artifact,
split, reproducibility và benchmark. Chưa nên dùng bản hiện tại cho benchmark
B200 chính thức hoặc kết luận khoa học.

P1: ưu tiên sửa trước khi chạy pilot tạo bằng chứng. P2: sửa để artifact có thể
tái lập, chuyển môi trường và dùng trong đánh giá có ghép cặp.

## R1 — P1: Prefill tính LM-head logits cho toàn bộ prompt

Vị trí: [inference.py](../../src/AMR_DFlash/inference.py), dòng 90–97 và
132–140; [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 673–679;
[evaluation.py](../../src/AMR_DFlash/evaluation.py), dòng 40–46.

Các target prefix forward không truyền `logits_to_keep=1`. Engine chỉ dùng
logits cuối để sinh first token; label/evaluator chỉ cần prefix cache. Hiện
LM head vẫn nhân toàn bộ N hidden states với vocabulary, tạo tensor `[1,N,V]`.
Baseline vendored DFlash đã dùng `logits_to_keep=1` cho prefill.

Đã tái hiện bằng forward hook: prompt 12 token đưa **12 rows** qua LM head;
cùng target với `logits_to_keep=1` chỉ đưa **1 row**. Với long context, đây là
chi phí compute và VRAM lớn không cần thiết, ảnh hưởng TTFT/E2E và adaptation
cost. Ví dụ tensor logits BF16 với N=16384, V=151936 chiếm khoảng 4.64 GiB,
chưa tính hidden states và cache.

Cách sửa: giới hạn logits tại các prefix forward; giữ đầy đủ logits ở bước
verify 16 vị trí. Test cần kiểm tra cả hidden-feature parity và số rows qua
LM head, không chỉ final-token parity.

## R2 — P1: Feature bundle cũ được chấp nhận khi thay weights drafter

Vị trí: [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 287–295 và
420–425; [model.py](../../src/AMR_DFlash/model.py), dòng 73–74.

`projected_context` được tạo bởi `draft.fc` và `draft.hidden_norm`, nhưng
bundle chỉ lưu fingerprint target và feature-layer IDs. Validator không kiểm
tra fingerprint draft; label/train/evaluate không đối chiếu draft trong
capture manifest trước khi đọc projected features.

Đã thay weights `fc` trong một snapshot draft thật được tạo local, giữ nguyên
target và layer IDs. Validator vẫn chấp nhận bundle cũ; projected features
tính lại khác features lưu với max absolute delta **5.87569**. Pipeline có thể
train/evaluate trong không gian feature của drafter khác rồi ghi provenance
của drafter đang load, khiến artifact sai mà không báo lỗi.

Cách sửa: lưu và kiểm tra fingerprint của draft/projection cùng target,
feature-layer IDs và precision/backend contract; kiểm tra tính nhất quán của
capture, labels, teacher tensors và checkpoint trước mỗi phase.

## R3 — P1: Một document có thể xuất hiện ở cả train và holdout

Vị trí: [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 199–201,
258–265 và 297–307.

Capture chưa kiểm tra unique document IDs hoặc overlap theo nội dung. `completed`
chỉ chứa IDs từ lần capture trước; records trùng trong lượt capture hiện tại
không bị từ chối. Cùng ID và prompt tạo cùng bundle/state IDs, dù explicit
split khác nhau. Các dict theo state ID và preferences có thể trộn records
giữa split.

Đã capture JSONL gồm hai records cùng `id="same-doc"`, cùng prompt, split
`train` và `holdout`: sinh **2 state records nhưng chỉ 1 unique state ID**, và
cả hai split đều được giữ. Đây là lỗi data-integrity và rò rỉ holdout, độc lập
với correctness của target verifier.

Cách sửa: khóa identity theo source document, kiểm tra IDs/content hashes và
reject overlap trước khi load model; validate unique state IDs và split nhất
quán khi replay/train. Không chỉ âm thầm bỏ record trùng.

## R4 — P2: Seed training được đặt sau khởi tạo selector

Vị trí: [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 944–945 và
960–961; [runtime.py](../../src/AMR_DFlash/runtime.py), dòng 48.

`load_runtime()` khởi tạo ngẫu nhiên AMRMemory trước `torch.manual_seed()`.
Config seed điều khiển sampling/optimization về sau nhưng không điều khiển
weights ban đầu; weights compressor được lưu cùng checkpoint selector cũng
bị ảnh hưởng.

Đã chạy `train_selector` hai lần với cùng config seed=17, data và 2 steps,
chỉ đổi ambient Torch RNG trước lời gọi: selector weights khác nhau, max
absolute delta **0.77790**. Bài kiểm tra này dùng synthetic preference pairs
để cô lập optimizer/seed, không là bằng chứng về chất lượng acceptance.

Cách sửa: áp dụng seed cho Python/Torch/CUDA trước khi tạo module/load runtime;
ghi seed và chính sách determinism vào artifact. Test so hai lượt train có
weights cuối giống nhau trong cùng runtime CPU.

## R5 — P2: Resume không khóa capture configuration

Vị trí: [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 209–218 và
375–401.

Resume kiểm tra input hash, models, layer IDs, block và memory config, nhưng
bỏ qua output/input caps, state sampling, dtype/backend và split fractions.
Nó giữ state cũ và ghi đè manifest bằng config mới. Nếu có documents chưa
capture, run còn trộn trajectories tạo theo hai cấu hình.

Đã resume run capture `max_new_tokens=32` với `max_new_tokens=4`: không bị từ
chối, manifest ghi **4** trong khi state hiện hữu có
`remaining_output_budget=31`. `max_states_per_document` cũng có thể đổi mà
không bị phát hiện.

Cách sửa: so capture contract đầy đủ trước khi skip/append; giữ manifest gốc
cho phần dữ liệu đã capture. Cho phép mở rộng số documents bằng cơ chế rõ ràng
mà không đổi generation/split contract.

## R6 — P2: Truncate state ID làm ghi đè candidate/teacher tensors

Vị trí: [pipeline.py](../../src/AMR_DFlash/pipeline.py), dòng 572–578 và
805; [artifacts.py](../../src/AMR_DFlash/artifacts.py), dòng 96–98.

`safe_id()` truncate còn 120 ký tự. State ID đã ghép document ID, prompt hash
và suffix state index; khi document ID dài, suffix phân biệt states bị cắt.
Các state khác nhau cùng ghi một file candidate positions hoặc teacher logits.

Đã tạo 2 state với document ID dài 120 ký tự, context lengths 12 và 13:
generator chỉ sinh **1 đường dẫn tensor**; 6 candidate IDs từ state đầu không
còn trong file bị ghi đè. Labeling sau đó sẽ gặp `KeyError` hoặc đọc nhầm dữ liệu.

Cách sửa: tên artifact gồm prefix dễ đọc và digest của **toàn bộ** state ID;
reject collision trước khi ghi. Áp dụng cùng quy tắc cho bundle/candidate/teacher.

## R7 — P2: Chuyển model sang mount path khác làm checkpoint không load

Vị trí: [checkpoint.py](../../src/AMR_DFlash/checkpoint.py), dòng 59–63 và
97–101.

Loader so toàn bộ fingerprint dict, gồm `resolved_path`. Snapshot giống hệt
về bytes/hash nhưng nằm ở local cache hoặc mount khác bị xem như đổi model.

Đã copy nguyên snapshot: SHA-256 giống nhau, nhưng checkpoint load vẫn báo
`fingerprint mismatch for target_fingerprint`. Điều này cản trở chuyển artifact
giữa môi trường hoặc thay mount/cache path trên server.

Cách sửa: so model identity bằng content hashes và contract; giữ đường dẫn
trong provenance, tách khỏi điều kiện tương thích checkpoint.

## R8 — P2: Rollout JSONL thiếu identity để ghép cặp theo document

Vị trí: [cli.py](../../scripts/amr_dflash/cli.py), dòng 304–352.

CLI in sample ID ra stdout nhưng không ghi `sample_id`/`document_id`, split
hoặc tokenized-prompt hash trong generation record. Target/draft fingerprints
cũng không xác định checkpoint selector/compressor được dùng. Schema validator
cơ bản không kiểm tra các trường này, nên tests vẫn pass.

Đã chạy inference qua CLI ở dense/selection/compressor/amr: cả bốn JSONL đều
thiếu ID. Không thể join artifact theo document một cách tự kiểm chứng để
kiểm tra exactness, paired throughput hoặc document-level bootstrap khi khác
thứ tự/shard/giới hạn mẫu. DFlash baseline hiện đã ghi `sample_id`, `prompt_hash`
và `run_config_hash`.

Cách sửa: ghi document/sample ID, split, tokenized-prompt/input manifest hash,
run/config hash và AMR checkpoint digest vào từng generation record. Phép join
phải reject missing/duplicate/mismatched identities.

## R9 — P2: Timing fields không tuân thủ metric contract

Vị trí: [inference.py](../../src/AMR_DFlash/inference.py), dòng 137–149,
237–238 và 322–346; [cli.py](../../scripts/amr_dflash/cli.py), dòng 315–316.

- `ttft_ms` luôn bằng target prefill time, dù first-token argmax và feature
  extraction/projection diễn ra sau mốc đó trước khi tạo output đầu tiên.
- `tpot_ms` dùng `e2e_ms / output_count`, bao gồm prefill; DFlash baseline dùng
  decode time và protocol AMR phân biệt prefill/decode/E2E.
- `memory_build_latency_ms` chỉ cộng cửa sổ memory build, bỏ ngoài cửa sổ đó
  incremental selector-key và compressor updates sau verification, dù runbook
  mô tả field gồm các updates này.

Đây là phát hiện từ các cửa sổ đo trong source; chưa đo magnitude trên CUDA.
E2E hiện vẫn chứa các thao tác này, nên lỗi ở cách đặt scope/attribution chứ
không có bằng chứng overhead bị bỏ khỏi E2E.

Cách sửa: đo tới emit token đầu; định nghĩa TPOT theo policy first-token/EOS
đã khóa; đo update bằng event hoặc field riêng. Khi gộp cohort, báo thêm tổng
committed tokens / tổng decode seconds, bên cạnh mean per-document rate.

## Bằng chứng kiểm tra

Lệnh regression:

```bash
.venv/bin/python -m pytest -q \
  tests/test_amr_dflash_contracts.py tests/test_amr_dflash_launcher.py \
  tests/test_dflash_wrapper_contract.py tests/test_master_config_contract.py
```

Kết quả **30 passed, 1 failed**. Cả 20 tests AMR passed. Failure còn lại ở
`test_scripts_root_keeps_only_primary_entrypoints`: test chưa tính entrypoint
`scripts/run_fa4_benchmark.sh` đã có trong repo; không do AMR thêm root `run_*`.
Cảnh báo CUDA driver/NVML đúng với máy local được quy định chỉ dev CPU.

`py_compile` trên modules/entrypoints AMR, `bash -n` trên launcher/config/dispatcher
và `git diff --check` đều exit 0.

Probe CPU bổ sung dùng snapshot tiny Qwen3 có weights thật, tokenizer local và
5 documents explicit train/validation/holdout: capture 30 states, tạo 180
candidate labels; verifier rewards của fixture này đều tie nên không có
preferences tự nhiên. Selector optimizer smoke dùng 3 synthetic preferences;
compressor dùng teacher logits đã label thật. Hai phase chạy 2 steps,
fixed-state evaluation chạy 2 validation states, CLI rollout chạy cả 4 modes.
Output token IDs của cả 4 modes giống nhau trong fixture này.

Probe riêng với `dtype=bfloat16` cũng chạy capture/label, selector và compressor
mỗi phase 1 step, rồi AMR rollout 8 output tokens trên CPU. Loss/gradients
finite; tensor output có shape `[1,20]` cho prompt 12 tokens. Probe này xác nhận
đường chuyển dtype và backward, không xác nhận kernel/performance B200.

Scripts/probe artifacts tạm nằm ở `/tmp/amr_review_check.py`,
`/tmp/amr_training_review.py`, `/tmp/amr_review_gcihn9xb/` và
`/tmp/amr_training_review.log`, `/tmp/amr_bf16_review.log`; không đưa model/checkpoint vào Git. Các số
tiny-model/CPU trong báo cáo chỉ là bằng chứng runtime/correctness, không là
benchmark hoặc chứng minh scientific gain.

## Những gate vẫn chưa được kiểm chứng

Chưa chạy Qwen3-4B/DFlash checkpoint production hoặc CUDA/B200. Chưa có artifact
chứng minh acceptance/ROUGE/speedup trên corpus thật. Các thiếu hụt đã được V0
ghi rõ gồm validation cadence, optimizer/RNG resume, calibrated cost gate,
resident-memory profiling, matched-budget/statistical experiment runner và
một số cấu trúc compressor trong thiết kế đầy đủ. Những mục này cần được
đóng riêng; passing contract tests không đóng các gate đó.

## Tình trạng sửa lỗi

Ngày 08/10/2026, sau yêu cầu sửa cả chín finding, implementation đã được cập
nhật:

| Finding | Sửa đổi |
|---|---|
| R1 | Prefix chỉ tạo logits cuối; feature extraction vẫn dùng đầy đủ hidden states, verifier giữ cả 16 logits. |
| R2 | Feature/capture contract khóa target, drafter, layers, dtype/backend; downstream kiểm tra bundle, state, labels, teacher và checkpoint. |
| R3 | Kiểm tra toàn bộ manifest trước slicing; từ chối ID/source/content trùng split và kiểm tra lại split/state links khi đọc artifact. |
| R4 | Seed Python/Torch/CUDA trước khi khởi tạo model và memory; checkpoint ghi seed cùng cờ deterministic algorithms. |
| R5 | Resume khóa input/model/config/caps/split/sampling; cho phép tăng sample cap, giữ manifest ban đầu và từ chối run thiếu index. |
| R6 | Artifact filename giữ prefix dễ đọc cộng digest toàn identity cho feature, candidate và teacher. |
| R7 | Checkpoint so fingerprint theo content/files; resolved path chỉ là provenance, không ảnh hưởng portability. |
| R8 | Rollout và summary ghi ID/split, prompt/input/config/checkpoint hashes, run/workload identity và timing policy. |
| R9 | Sửa TTFT/TPOT; tách prefill, projection, memory build/update, draft, verification; CUDA dùng events thay cho sync từng stage. |

Thêm regression tests trên target/drafter tiny local thật, gồm capture/label,
selector/compressor training và fixed-state/rollout ở FP32/BF16. Lần chạy
suite cuối ghi **104 passed, 2 failed**: cả hai failure cùng chỉ ra launcher
FA4 thiếu status check sau khi source shared runtime. Đã thêm status check và
kiểm tra state index theo một lượt tuyến tính; không chạy lại suite sau hai
điều chỉnh cuối này. Kết quả local không chứng minh throughput hoặc độ chính
xác trên B200; các research gates ở trên vẫn mở.

## Kiểm tra lại mức sẵn sàng train — 09/10/2026

Đã chạy lại suite sau các sửa đổi cuối, bao gồm source/content split checks,
state index và launcher FA4:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  tests/test_amr_dflash_regressions.py tests/test_amr_dflash_contracts.py \
  tests/test_amr_dflash_launcher.py tests/test_dflash_wrapper_contract.py \
  tests/test_master_config_contract.py tests/test_paired_reference.py \
  tests/test_paired_output_capture.py tests/test_acceptance_metrics.py \
  tests/test_b200_launcher_contract.py tests/test_b200_preflight_contract.py \
  tests/test_shared_runtime_contract.py --tb=short
```

Kết quả: **106 passed, 1 warning, exit 0**, 87.52 giây. Warning liên quan
driver CUDA local đã được ghi trong AGENTS.md; workload kiểm tra chạy trên CPU.
Log tại `/tmp/amr_readiness_2026_10_09.log`. Kết quả này thay thế tình trạng
104 passed / 2 failed trước khi sửa launcher. `py_compile` cho modules/CLI
AMR, `bash -n` cho launcher và `git diff --check` cũng exit 0.

Pipeline tiny với safetensors/tokenizer local thật đã chạy capture, candidate
labeling, hai bước selector và compressor training, fixed-state evaluation
và rollout bốn modes ở cả FP32/BF16. Selector training dùng preference fixture
được kiểm soát; compressor dùng logits teacher từ verifier thật. Chưa có
bằng chứng preference signal tự nhiên trên corpus production.

**Mức sẵn sàng:** code V0 đủ cho train pilot. Trước khi mở run dài trên B200,
cần preflight đúng target/drafter/data, capture/label với hợp đồng mới, xác
nhận có train preferences không tie/censor và train teacher hợp lệ, rồi chạy
20 bước mỗi phase để kiểm tra loss/gradient, VRAM và checkpoint trên server.
Optimizer/RNG resume và validation định kỳ vẫn chưa có; các scientific gates
giữ trạng thái mở.
