# Progress log — GroundSync

## 2026-08-29

- Đọc proposal GroundSync từ shared conversation và xác định mạch H1–H5/E0–E5.
- Đọc `AGENTS.md`; giữ nguyên quy ước Python/runtime/data của repo.
- Khảo sát repo: worktree sạch; xác nhận `src/analyze` mới có profiler Qwen3
  target-only, chưa có implementation GroundSync.
- Xác nhận các dataset representative/LongBench và các script Qwen3/EAGLE hiện có.
- Chưa sửa code; đang chờ chốt design experiment theo ML brainstorming gate.
- Design và implementation plan đã ghi dưới `src/analyze/groundsync/plans/`.
- Subtask 1: viết core metric tests (RED) và implement `core.py`; fresh result
  `12 passed in 0.07s`.
- Subtask 2: viết target adapter tests (RED) và implement `trace_target.py`;
  fresh result combined core/target `15 passed`, compile pass.
- Subtask 3: thêm controlled draft/target acceptance, drift, grounding horizon
  và timing arrays theo `k`; fresh result toàn bộ tests `31 passed`.
- Subtask 4: thêm document-split predictor, bootstrap CI, H1–H5 aggregation,
  CSV/PNG/Markdown artifacts; report tests và integration tests nằm trong tổng
  `31 passed`.
- Subtask 5: thêm `run_experiment.py`, synthetic fixture và README. Synthetic
  run `synthetic-20260829-v2` tạo raw JSONL, metrics, CSV, PNG và report.
- Model-backed local-only smoke `qwen3-local-smoke-20260829` đã thử với
  Qwen3-1.7B/0.6B trên CPU; snapshot không có trong cache, nên target/draft và
  H1–H5 đều ghi `UNAVAILABLE`. Không dùng Qwen2.5-VL hay EAGLE head thay thế.
- Final validation: `34 passed`, `compileall` và `git diff --check` đều pass.
  Run `synthetic-20260829-audit` có đủ 4 PNG + JSON/CSV/Markdown; run
  `qwen3-local-smoke-audit` tái hiện rõ 5/5 hypothesis `UNAVAILABLE` vì thiếu
  Qwen3 snapshot local.
- Audit continuation: verifier H4 đã được sửa để đo block `k` trong một
  forward; full test/compile lại pass và run
  `synthetic-20260829-blocked-audit` chạy thành công. Đã kiểm tra thêm các
  mount/path hệ thống, không có full Qwen3 snapshot; blocker cần external model
  state hoặc user cung cấp đường dẫn checkpoint.
- Cache audit mới: xác nhận full Qwen3-4B canonical target trong Hugging Face
  cache; Qwen3-4B EAGLE head bị loại vì sai architecture. Qwen3-0.6B đã được
  tải local và validate bằng config/tokenizer/safetensors.
- Model smoke mới `qwen3-4b-06b-actual-smoke-20260829` chạy end-to-end trên
  CPU với target Qwen3-4B + draft/verifier Qwen3-0.6B, gồm timing verifier.
- Discovery mới `qwen3-4b-cnn10-target-20260829` chạy 10 mẫu CNN/DailyMail:
  target 10/10, controlled proposal 20/20, report mới cho H1 `FAIL`, H2
  `FAIL`, H3/H4/H5 `UNAVAILABLE` theo coverage/điều kiện đo; không overclaim
  hypothesis từ run này.
- Theo yêu cầu dùng môi trường máy, đã xác minh trực tiếp ngoài sandbox:
  `/home/tuantb/miniconda3/bin/python3` thấy Tesla T4 và CUDA 12.4; không dùng
  `.venv` cho thực nghiệm GPU.
- Discovery GovReport đầu tiên lộ OOM do T4 `sm75` không có native Flash SDP
  phù hợp với torch cu124. Đã thêm chunked causal prefill (mask bottom-right)
  cho target, draft và verifier, kèm regression tests; fresh suite sau patch
  đạt `38 passed`.
- Run GPU chính `qwen3-4b-gov25-gpu-all-20260829` chạy 25/25 target và 50/50
  speculative rows, mọi row có timing. Report mới: H1/H2/H4 `FAIL`, H3/H5
  `UNAVAILABLE`; không có OOM sau patch.
- Đã viết báo cáo diễn giải toàn bộ quy trình, runtime, cache audit, các lần
  OOM và sửa chunked prefill, lệnh tái lập, kết quả H1–H5 và giới hạn tại
  `src/analyze/groundsync/verification_report_2026-08-29.md`.
- Đã audit lần hai theo yêu cầu: bổ sung protocol-versus-implementation
  deviations, gate versioning, inventory 18 run directories, cache download
  event, artifact/log gaps và các metric H1 lag/segment vào báo cáo; không đổi
  raw result hay status H1–H5.

## Bổ sung protocol và kết quả 2026-08-30

- Theo TDD, thêm test/implementation cho positional calibration 32-bin,
  sensitivity chunk 64/128/256, sink 4/8/16, first-rejection hazard theo
  relative draft position, position-adjusted hazard coefficient với 2.000
  document-bootstrap resamples, fixed/adaptive/true-cost policy sweep, H3
  negative controls và H5 threshold selection train/dev. Fresh suite đạt
  **50 passed**;
  `compileall` pass.
- Thêm `start-offset` và `sample-ids` cho speculative runner để không lấy duy
  nhất position 0 và để tạo timing subset đúng document test; mỗi speculative
  output có manifest coverage/timing basis.
- `qwen3-4b-gov100-gpu-protocol-20260830`: 100 GovReport target request,
  99 `ok`, 1 OOM; 99 draft-only proposals tại start=1, kmax=8; 12 test-ID
  timing requests, 11 `ok`, lọc còn 10 rows phủ đủ k=8.
- `qwen3-4b-cnn100-gpu-protocol-20260830`: 100/100 CNN/DailyMail target
  `ok`, 100 controlled proposals và 12 timing rows phủ đủ k=8; dùng làm
  cross-regime H1–H5.
- Kết quả run mở rộng: H1 composite `FAIL` (no-sink CI lower 0,021996 nhưng
  calibrated 0,018257; CNN CI lower 0,010580); H2 GovReport `FAIL` với
  coefficient -0,0657 và CI [-0,0664; -0,0535], CNN/DailyMail `PASS` với CI
  [0,0181; 0,0301]; hai regime không cùng chiều. H3 primary `FAIL` (Gov
  AUROC gain 0; CNN no-sink gain -0,0267 dù calibrated sensitivity +0,0267);
  H4 `FAIL` (grounding oracle speed gain -0,4433 Gov/ -0,4386 CNN); H5
  `UNAVAILABLE` (predictor có metric nhưng grounding oracle chậm hơn fixed,
  nên oracle-gain recovery không được định nghĩa ở cả hai regime).
- Báo cáo chính đã cập nhật thành bản mở rộng, giữ run 25 mẫu là lịch sử và
  link artifact mới; đã ghi thêm hazard coefficient/bootstrap và kết quả
  cross-regime H2–H5. Chưa claim EAGLE/vLLM production throughput hoặc causal
  attribution.
- Đã bổ sung E0 model-backed position-relocation fixture bằng Qwen3-4B trên
  T4: cùng evidence ở đầu/giữa/cuối, 3/3 case `ok`; raw mass
  `0,5029/0,1170/0,1929`, no-sink mass `0,5185/0,2297/0,2296`. No-sink giảm
  nhưng không loại bỏ positional confounder.
- Đã bổ sung lag-drift history và chạy multi-start draft-only trên 100 target
  traces mỗi regime, bốn start `1,6,11,16`: GovReport 396 rows, CNN/DailyMail
  400 rows. H2 giữ cùng pattern theo regime; H3 no-sink gains lần lượt
  `-0,0053` và `+0,0086`, đều dưới gate `0,02`; thiếu verifier timing nên H4
  `UNAVAILABLE`, H5 `INCONCLUSIVE` cho các run bổ sung.
