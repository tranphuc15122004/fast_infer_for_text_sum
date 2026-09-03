# GroundSync hypothesis runner

Pipeline này kiểm chứng H1–H5 bằng hai lớp dữ liệu:

- `target_traces.jsonl`: target Qwen3 greedy, attention query cuối sau cached
  forward; chỉ lưu vector source-attention đã gom chunk.
- `speculative_traces.jsonl`: draft Qwen3 nhỏ sinh greedy tại các prefix của
  target, rồi so với canonical target continuation để tính accepted prefix.

Chỉ dùng các model được phép `Qwen3-4B`, `Qwen3-1.7B`, `Qwen3-0.6B`. Loader
dùng `local_files_only=True`, vì vậy thiếu snapshot/cache sẽ thành
`UNAVAILABLE`, không tải mạng và không thay bằng model khác.

## CPU evaluator smoke

Lệnh sau tạo fixture nhân tạo và toàn bộ artifact evaluator. Fixture chỉ kiểm
tra đường đi của metric, không phải bằng chứng cho model Qwen:

```bash
.venv/bin/python -m src.analyze.groundsync.run_experiment \
  --phase synthetic --run-id synthetic-20260829 --max-k 4
```

Kết quả nằm tại `src/analyze/groundsync/results/<run-id>/` gồm raw JSONL,
`metrics.json`, `metrics.csv`, hai PNG và `hypothesis_report.md`.

## Model-backed local-only run

Trên server có snapshot Qwen local và CUDA tương thích:

```bash
python3 -m src.analyze.groundsync.run_experiment \
  --phase all \
  --run-id qwen3-1p7b-gov-smoke \
  --model /path/to/Qwen3-1.7B \
  --draft-model /path/to/Qwen3-0.6B \
  --input data/representative_100/gov_report.jsonl \
  --device cuda:0 --dtype float16 --prefill-chunk-size 512 \
  --max-samples 2 --max-new-tokens 64
```

`--phase target` và `--phase speculative` có thể chạy trong một run `all`; để
phân tích lại raw traces đã có, dùng `--phase analyze --run-dir
src/analyze/groundsync/results/<run-id>`. Khi cần tách acceptance khỏi timing,
đặt verifier output tại `speculative_timing_traces.jsonl`; analyzer dùng file
này cho H4/H5 policy và dùng `speculative_traces.jsonl` cho acceptance đầy đủ.
Orchestrator không ghi đè một directory run không rỗng.

Speculative runner hỗ trợ `--start-offset` để tránh chỉ lấy position 0 (nơi
drift đầu block chưa định nghĩa), `--sample-offset` và `--sample-ids` để tạo
subset train/dev/test tái lập. Output speculative có manifest ghi model,
coverage, kmax, start selection và timing basis.

## Run model-backed đã có trên teslaT4

Cache canonical Qwen3-4B hiện ở
`/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
Qwen3-0.6B đã được tải local ở `/home/tuantb/models/Qwen3-0.6B` và validate
offline. Vì máy local không có CUDA tương thích, discovery chạy CPU:

- `results/qwen3-4b-06b-actual-smoke-20260829/`: 1 mẫu, kiểm tra end-to-end
  target/draft/verifier và timing block.
- `results/qwen3-4b-cnn10-target-20260829/`: 10 mẫu CNN/DailyMail, 80 target
  steps và 20 controlled proposals; H1/H2 lần lượt `FAIL`, H3–H5
  `UNAVAILABLE` theo report.

Run GPU chính `results/qwen3-4b-gov25-gpu-all-20260829/` dùng Python miniconda
ngoài venv với `cuda:0`, FP16, 25 GovReport, 25 target traces và 50 controlled
proposals có timing verifier. Kết quả H1/H2/H4 là `FAIL`, H3/H5 là
`UNAVAILABLE`; xem `hypothesis_report.md` trong run để biết metric và coverage.

T4 `sm75` không được native Flash SDP của torch cu124 hỗ trợ cho context dài,
nên phải dùng `--prefill-chunk-size 512` để tránh eager/math prefill tạo ma trận
attention đầy đủ. Run mở rộng mới nhất:

- `results/qwen3-4b-gov100-gpu-protocol-20260830/`: 100 GovReport target,
  99 ok do một OOM; 99 draft-only proposals; 10 timing rows phủ đủ `k=8`.
- `results/qwen3-4b-cnn100-gpu-protocol-20260830/`: 100/100 CNN/DailyMail
  target, 100 controlled proposals và 12 timing rows; cross-regime H1–H5.
- `results/e0-position-relocation-qwen3-4b-20260830/`: fixture Qwen3-4B với
  cùng evidence đặt ở đầu/giữa/cuối source; raw/no-sink mass cho thấy
  positional confounder còn tồn tại sau sink control.
- `results/qwen3-4b-gov100-multistart-20260830/` và
  `results/qwen3-4b-cnn100-multistart-20260830/`: mỗi document có bốn
  controlled draft proposals tại start `1,6,11,16`; draft-only, dùng để
  kiểm tra lag-drift/history và within-document controls, không dùng để claim
  throughput.

Kết quả và giới hạn chi tiết nằm trong
[`verification_report_2026-08-29.md`](verification_report_2026-08-29.md).

## Ý nghĩa kết quả

H1 đo persistence của source-utilization state bằng `1 - JS`, so với null
shuffle theo document. H2 đo first rejection trong controlled draft–target
acceptance bằng hazard theo vị trí và drift coefficient với 2.000 document
bootstrap; median split chỉ là mô tả phụ. H3/H5 fit predictor nhỏ với split theo document; các
token trong cùng tài liệu không được coi là mẫu độc lập. H4 chỉ kết luận speed
khi có cả `draft_time_ms` và `verification_time_ms`; acceptance-only luôn được
gắn nhãn, không gọi là speedup. Attention là tín hiệu quan sát, không phải
ground-truth attribution.

Các trạng thái `PASS`, `FAIL`, `INCONCLUSIVE`, `UNAVAILABLE` đều xuất hiện
trong `metrics.json`; thiếu model, thiếu class, thiếu timing, thiếu oracle
headroom hoặc coverage thấp không bị chuyển thành pass. H1 composite chỉ PASS
khi calibrated estimator cũng đạt gate; H5 chỉ tính oracle-gain recovery khi
oracle nhanh hơn fixed.

## P0 decision experiments — 2026-09-02

Bộ kiểm định corrected H2/H4, oracle ladder, admission và burstiness được triển
khai trong `p0_decision.py`. Tất cả raw trace, metrics, CSV, biểu đồ và báo cáo
của P0 nằm dưới `results/`; không dùng venv khi chạy GPU T4.

Analyzer chốt có thể tái lập bằng:

```bash
python3 -m src.analyze.groundsync.p0_decision \
  --config src/analyze/groundsync/p0_decision_config_20260902.json \
  --output src/analyze/groundsync/results/p0-decision-final9-20260902 \
  --max-k 16 --candidate-ks 0,2,4,8,16 \
  --horizon-threshold 0.2 --bootstrap-samples 2000
```

Timing P0 dùng AR cached one-token cho `k=0`, draft decode incremental và
cached target verification cho `k>0`. Ladder chỉ giữ các row có đủ timing ở
tất cả `k=0,2,4,8,16`, để mọi policy dùng cùng population. Grounding threshold
được chọn bằng utility trên train/dev theo document rồi freeze trên test; P0-5
dùng 9 start/document (`1,4,7,10,13,16,19,22,25`, `max_k=4`) để đo đủ
round offset `1,2,4,8`. Đây là controlled/discovery evidence, chưa phải throughput EAGLE,
vLLM hay production serving.

## P1/P2 — predictor, strong drafter và E2E — 2026-09-02

P1 cheap predictor nằm trong `p1_predictor.py`. Nó chỉ đọc bốn tín hiệu tại
entry, split theo document và chọn policy trước test:

```bash
python3 -m src.analyze.groundsync.p1_predictor \
  --gov-timing src/analyze/groundsync/results/p0-gov50-k16-timing-cached-20260902/speculative_traces.jsonl \
  --gov-timing src/analyze/groundsync/results/p0-gov10-k16-timing-cached-cont-20260902/speculative_traces.jsonl \
  --cnn-timing src/analyze/groundsync/results/p0-cnn50-k16-timing-cached-20260902/speculative_traces.jsonl \
  --multinews-timing src/analyze/groundsync/results/p1p0-multinews10-timing-20260902/speculative_traces.jsonl \
  --output-dir src/analyze/groundsync/results/p1-cheap-admission-20260902
```

Strong-drafter/P2 direct E2E dùng `p1_strong_drafter.py`, Python miniconda
ngoài venv và CUDA T4; ví dụ:

```bash
/home/tuantb/miniconda3/bin/python3 -m src.analyze.groundsync.p1_strong_drafter \
  --base-model /path/to/Qwen3-4B \
  --eagle-model /path/to/Qwen3-4B_eagle3 \
  --dataset data/representative_100/cnn_dailymail_representative.jsonl \
  --output-dir src/analyze/groundsync/results/p1p2-eagle3-cnn50-20260902 \
  --max-samples 50 --max-new-tokens 32 --max-input-tokens 1024 \
  --total-token 16 --depth 4 --top-k 4 --include-naive
```

Kết quả chính và quyết định cuối được tổng hợp trong
[`final_decision_report_2026-09-02.md`](final_decision_report_2026-09-02.md).
Serving preflight nằm tại `results/p2-serving-preflight-20260902/`; nếu không có
canonical server mount/runtime thì status là `UNAVAILABLE`, không thay bằng
direct E2E.

## Target-KV Conditioned Block Drafting — 2026-09-03

Nhánh Target-KV độc lập với GroundSync/P0 cũ được triển khai bằng:

- `target_kv_experiments.py`: schema, bucket, quy đổi DFlash fallback, bootstrap
  document-level và E0 gate.
- `e0_dflash_failure_map.py`: runner DFlash chính thức với target prefill
  chunked, selective hidden capture và K=`{4,8,16}`.
- `target_kv_e1.py` và `e1_representation_probe.py`: interface representation,
  document-disjoint split, matched-budget probe và negative controls.
- `e0_report.py`, `e1_report.py`: báo cáo Markdown theo từng run.

Mọi model-backed run dùng Python CUDA ngoài `.venv`, local-only cache, FP16,
batch 1 và `prefill_chunk_size=128` trên T4. E0 pilot dùng `max_new_tokens=8`
để mở rộng coverage; mini-confirmation dùng 32 token. E1 giữ feature trung gian
ở CPU và bọc toàn bộ forward bằng `torch.inference_mode()` để không dựng graph
autograd trên GPU.

Các artifact đã chạy:

- E0: `results/tkv-e0-pilot-gov-20260903/`,
  `results/tkv-e0-pilot-multinews-20260903/`,
  `results/tkv-e0-pilot-cnn-20260903/`;
- E0 32-token: `results/tkv-e0-confirm32-gov-20260903/` và
  `results/tkv-e0-confirm32-multinews-20260903/`;
- E1: `results/tkv-e1-gov-4k-20260903/` và
  `results/tkv-e1-multinews-20260903/`;
- đối chiếu runner chính thức: `results/tkv-dflash-official-crosscheck-20260903.jsonl`;
- tổng hợp quyết định: `target_kv_decision_report_2026-09-03.md`.

E0 pilot cho zero accepted draft token trên cả ba dataset; long-context
context-drop gate là `INCONCLUSIVE` vì dataset được chạy FP16 an toàn trên T4
không có bucket dài tự nhiên. E1 Multi-News probe không cho thấy KV vượt
token-wise hidden. Do đó E2/E3 không được chạy và không được suy diễn thành
speedup online; xem report master để biết coverage, numerical guardrail và
giới hạn kết luận.
