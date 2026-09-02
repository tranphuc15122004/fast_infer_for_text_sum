# Findings — GroundSync experiment

## Repo hiện tại

- Git worktree sạch trên nhánh `main`.
- `src/analyze` hiện chỉ có `full_infer/profile_qwen3_long_summary.py`, README
  và kết quả profiler Qwen3 cũ; chưa có pipeline GroundSync/hypothesis analysis.
- Repo đã có các helper chung ở `scripts/common`, profiler Qwen3 ở
  `src/analyze/full_infer`, và wrapper `scripts/run_qwen3_long_profile.sh`.
- Dữ liệu plug-and-play đã có: `data/representative_100/` gồm GovReport,
  Multi-News, CNN/DailyMail, XSum; `data/longbench_200/` có GovReport,
  Multi-News, QMSum và các task khác.

## Proposal cần triển khai

- H1/E1: source-utilization state có temporal persistence trên canonical target
  AR trace.
- H2/E2: source-state drift/transition dự báo first speculative rejection.
- H3/E3: tín hiệu còn predictive sau entropy, draft confidence, acceptance
  history, position, sentence boundary và copyability controls.
- H4/E4: oracle source-state horizon tạo utility về committed tokens/sec hoặc
  cost trên mỗi committed token.
- H5/E5: horizon có thể dự đoán online từ tín hiệu hiện tại/quá khứ.
- Attention sink/position bias phải là confounder được đo và kiểm soát, không
  được gọi raw attention là ground truth.

## Tín hiệu từ repo

- `docs/qwen3_long_profile.md` mô tả profiler Qwen3-4B target-only và cấu hình
  local-files-only; artifact cũ nằm ở `src/analyze/full_infer/results`.
- `docs/model_baseline_matrix.md` ghi Qwen3-4B là cấu hình hiện có cho EAGLE-3
  và DFlash; `docs/baselines/eagle3.md` mô tả chi phí/Qwen3 pairing.
- `scripts/eagle3_infer_qwen3.py` và `scripts/run_eagle3_qwen3.sh` là ứng viên
  để kiểm tra speculative trace nếu model/drafter cache tồn tại.

## Điểm đã chốt trước khi code

- Experiment directory là `src/analyze/groundsync`, đúng yêu cầu artifact.
- Protocol ưu tiên GovReport/representative và có synthetic CPU smoke để kiểm
  tra evaluator trước khi đụng model.
- H4 chỉ được kết luận speed khi có timing đo theo từng `k`; nếu thiếu thì chỉ
  giữ acceptance-only.

## Khảo sát bổ sung

- Profiler hiện tại dùng Transformers target-only, greedy, `output_attentions`/
  `output_hidden_states` chưa được triển khai; cần một pipeline trace riêng để
  giữ attention/hidden-state theo output position.
- EAGLE script hiện tại là benchmark decode-only và yêu cầu CUDA; không phù hợp
  làm nguồn duy nhất cho H2 trên máy CPU. Có thể dùng một controlled speculative
  trace bằng target-vs-draft greedy logits trước, sau đó chạy E2E nếu có GPU và
  checkpoint draft tương thích.
- `src/analyze/full_infer/results` là artifact cũ, không nên ghi đè; experiment
  mới cần thư mục riêng.
- Repo có cấu hình EAGLE Qwen3-4B nhưng chưa có bằng chứng trong khảo sát rằng
  Qwen3-1.7B/0.6B hoặc drafter tương ứng đã được cache.

## Runtime hiện tại

- Local `.venv` là Python 3.12.13 với torch `2.11.0+cu130`, Transformers
  `5.12.1`; trong sandbox CUDA không được expose. Khi tắt venv và chạy trực
  tiếp `/home/tuantb/miniconda3/bin/python3` (Python 3.13, torch `2.6.0+cu124`),
  host thấy Tesla T4, driver `550.163.01`, CUDA `12.4` và
  `torch.cuda.is_available()=True`.
- Cache hợp lệ Qwen3-4B đã được tìm thấy tại
  `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`;
  config là `Qwen3ForCausalLM`, 36 layers và 3 shard safetensors.
  `/home/tuantb/models/Qwen3-4B_eagle3` chỉ là EAGLE head
  `Eagle3LlamaForCausalLM`, không được dùng làm canonical target.
- Qwen3-0.6B ban đầu chưa có nên đã tải vào `/home/tuantb/models/Qwen3-0.6B`;
  config là `Qwen3ForCausalLM`, 28 layers và safetensors đã được mở/kiểm tra
  thành công. Qwen3-1.7B chưa cần tải.
- `torch.cuda.is_available()` vẫn false vì driver local không tương thích;
  discovery model-backed dưới đây chạy CPU. H4/E2E throughput trên server CUDA
  vẫn phải tách khỏi controlled acceptance.

## Kết quả thực thi 2026-08-29

- `synthetic-20260829-final-v2`: evaluator chạy đầy đủ 12 target fixture và 120
  controlled rows, sinh JSONL/CSV/4 PNG/Markdown. Đây là test đường ống, không
  phải bằng chứng Qwen.
- `qwen3-local-smoke-final`: đã thử Qwen3-1.7B target + Qwen3-0.6B draft với
  `local_files_only=True` trên CPU; snapshot không có, nên H1–H5 đều
  `UNAVAILABLE`. Không dùng `/home/tuantb/models/Qwen3-4B_eagle3` vì đó là
  EAGLE head một layer, không phải canonical Qwen3 target.
- Final validation fresh: `34 passed`, `compileall` pass và `git diff --check`
  pass.

## Kết quả model-backed mới nhất

- `qwen3-4b-06b-actual-smoke-20260829`: 1 tài liệu, target Qwen3-4B và draft
  Qwen3-0.6B đều chạy; verifier target đo được timing theo block `k`. Đây là
  pipeline smoke, chưa đủ coverage để kết luận H1–H5.
- `qwen3-4b-cnn10-target-20260829`: 10 tài liệu CNN/DailyMail, 80 target token
  steps và 20 controlled proposals (`max_k=2`, hai prefix mỗi tài liệu). Report:
  - H1 `FAIL` theo gate định trước: persistence excess `0.00306`, bootstrap
    95% CI `[0.00175, 0.00442]`, thấp hơn ngưỡng `0.02` dù adjacent similarity
    nhỉnh hơn shuffle null.
  - H2 `FAIL`: high-drift rejection rate `0.20` thấp hơn low-drift `0.40`,
    chênh lệch `-0.20`.
  - H3 `UNAVAILABLE`: document-split/predictor chưa đủ coverage và class
    variation hữu ích.
  - H4 `UNAVAILABLE`: discovery dùng draft-only, không có verifier timing cho
    mọi row; smoke riêng chỉ xác nhận đường đo timing hoạt động.
  - H5 `UNAVAILABLE`: grounding horizon không tạo nhãn dương ở threshold hiện
    tại, nên chưa đánh giá được predictor.
- Kết luận hiện tại: hypothesis chưa được kiểm chứng thành công trên run
  model-backed này; H1/H2 có bằng chứng không ủng hộ claim dưới gate hiện hành,
  H3–H5 cần coverage/điều kiện đo bổ sung. Synthetic PASS không phải bằng chứng
  Qwen.

## Kết quả GPU T4 mới nhất

- Run `qwen3-4b-gov25-gpu-all-20260829` dùng Python miniconda ngoài venv,
  `cuda:0`, FP16, Qwen3-4B target và Qwen3-0.6B draft trên 25 GovReport.
  Target đạt `25/25`, speculative đạt `50/50`; context sau tokenizer là
  `2.609–22.630` token. Tất cả 50 row có draft/verifier timing.
- H1 `FAIL`: adjacent similarity `0.98997`, shuffle null `0.97113`, excess
  `0.01884`, bootstrap 95% CI `[0.01598, 0.02188]`; CI không vượt gate
  `0.02`.
- H2 `FAIL`: rejection low-drift `1.00`, high-drift `0.8333`, chênh lệch
  high-minus-low `-0.1667`.
- H3 `UNAVAILABLE`: document split đã có 15/5/5 documents nhưng test labels
  chỉ có một class, nên AUROC/gain không hợp lệ.
- H4 `FAIL`: timing thật đã đo; fixed policy đạt `0.0003616` committed-token/ms,
  oracle `0.0002885`, speed gain `-0.2021`, không đạt mục tiêu `>= 0.08`.
- H5 `UNAVAILABLE`: có 375 horizon rows nhưng grounding horizon không tạo
  class variation dương ở threshold `0.2`.
- Run đầu tiên với eager prefill bị OOM vì T4 `sm75` không được native Flash
  SDP của torch cu124 hỗ trợ. Adapter đã được sửa và test hồi quy thêm
  chunked causal prefill; run GPU chính sau đó không OOM. Đây là sửa hạ tầng đo,
  không phải thay đổi hypothesis.

## Mở rộng protocol 2026-08-30

- TDD regression suite cuối: **50 passed**, compileall pass. Bổ sung
  calibration positional prior 32 bins, chunk/sink sensitivity, first-reject
  hazard, position-adjusted hazard coefficient với 2.000 document-bootstrap
  resamples, H3 controls, adaptive entropy/history, true-cost policy và H5
  threshold selection train/dev.
- `qwen3-4b-gov100-gpu-protocol-20260830`: 100 GovReport target requests,
  99 ok/1 OOM; 99 draft-only proposals start=1/kmax=8; timing test-ID subset
  11 ok/1 OOM, 10 rows phủ đủ k=8.
- `qwen3-4b-cnn100-gpu-protocol-20260830`: CNN/DailyMail target 100/100 ok,
  100 controlled proposals và 12 timing rows phủ đủ k=8; dùng kiểm tra
  cross-regime H1–H5.
- GovReport H1 no-sink excess 0,023892 (CI lower 0,021996) nhưng calibrated
  CI lower 0,018257; CNN CI lower 0,010580. Composite H1 FAIL.
- H2 GovReport 99 proposals: drift high-minus-low rejection -0,022857; hazard
  coefficient `-0,0657`, CI document-bootstrap `[-0,0664; -0,0535]`, FAIL.
  CNN/DailyMail 100 proposals: coefficient point `0,00054`, CI `[0,0181;
  0,0301]`, PASS ở regime này. Hai regime khác chiều nên chưa có claim tổng
  quát.
- H3 99 rows, split 59/19/21 documents: baseline/full cùng AUROC 1, gain 0;
  full cải thiện log-loss/Brier nhưng không đạt incremental AUROC gate.
- H4 10 rows timing: grounding horizon oracle 0,0002649 token/ms so với fixed
  k=8 0,0004758, speed gain -0,4433; true-cost hindsight oracle 0,0005333.
- H5 chọn threshold 0,05 từ train/dev, predictor test AUROC 0,35; predicted
  policy 0,0002969 thấp hơn fixed k=4 0,0004111. Grounding oracle chậm hơn
  fixed nên oracle-gain recovery để None, decision UNAVAILABLE.
- CNN/DailyMail H3 no-sink AUROC gain `-0,0267` (calibrated sensitivity
  `+0,0267`), H4 speed gain `-0,4386`, H5 predictor AUROC `0,6875` nhưng
  oracle horizon vẫn chậm hơn fixed; H5 UNAVAILABLE.
- E0 position-relocation Qwen3-4B fixture gồm 3 case cùng evidence ở đầu/
  giữa/cuối. Raw mass lần lượt `0,5029/0,1170/0,1929`, no-sink lần lượt
  `0,5185/0,2297/0,2296`; no-sink giảm nhưng không loại bỏ position
  confounder. Artifact ở `results/e0-position-relocation-qwen3-4b-20260830`.
- Multi-start controlled run với `start=1,6,11,16` tạo 396 GovReport và 400
  CNN/DailyMail proposal rows. GovReport H2 FAIL (CI `[-0,0901;-0,0841]`),
  H3 AUROC gain `-0,0053`; CNN H2 PASS (CI `[0,0061;0,0106]`), H3 gain
  `+0,0086`. Không có verifier timing nên H4 unavailable và H5 inconclusive
  ở hai run.
