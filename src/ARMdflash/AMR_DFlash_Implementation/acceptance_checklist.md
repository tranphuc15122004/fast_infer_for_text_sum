# Checklist nghiệm thu AMR-DFlash

Ngày: **08/10/2026**. V0 code, launcher và CPU synthetic contracts đã có; model thật/B200 chưa chạy. Checkbox bên dưới là acceptance gates đầy đủ, không chỉ xác nhận file/code tồn tại.

Đọc [kế hoạch](../../../docs/superpowers/plans/2026-10-08-amr-dflash-implementation.md) và [protocol](experiment_protocol.md). Khi đóng checkbox, ghi command/exit code/run ID/artifact.

## Bằng chứng đã có

- `.venv/bin/python -m pytest -q tests/test_amr_dflash_contracts.py tests/test_amr_dflash_launcher.py`: CPU synthetic contract tests pass; gồm dense DFlash identity, target greedy parity qua reject/EOS/cap cho dense/hybrid/compressor, fixed-state verifier parity, selector/query, candidate budgets, streaming compressor, gradient qua frozen DFlash, checkpoint mismatch và dispatcher/preflight với model assets giả lập.
- `.venv/bin/python -m py_compile src/AMR_DFlash/*.py scripts/amr_dflash/cli.py scripts/infer_amr_dflash.py scripts/check_amr_dflash_b200.py`: syntax check đã pass ở revision trước; cần chạy lại sau thay đổi cuối.
- Chưa có bằng chứng từ Qwen3-4B/DFlash checkpoint thật, CUDA/B200, runtime latency, fitted AMR checkpoint hoặc scientific gain.

Cập nhật **09/10/2026**: suite AMR/regression/launcher/runtime/paired metrics
đã chạy lại sau sửa cuối, **106 passed**, exit 0. Pipeline tiny FP32/BF16 có
capture/label, hai bước mỗi phase train và fixed-state/rollout evaluation;
selector dùng preference fixture được kiểm soát, teacher logits được label
bằng verifier thật. Syntax checks hiện hành cũng pass. Chi tiết command và
phạm vi bằng chứng tại [review](../../../docs/reviews/2026-10-08_amr_dflash_implementation_review.md).

Test synthetic không thay thế checklist B200 bên dưới; các mục giữ unchecked tới khi có artifact của đúng model/data/config.

## M0 — Interface và correctness

- [ ] Strict load/fingerprint target/tokenizer/draft depth/feature IDs/block/mask/EOS.
- [ ] Dense bypass/all-context selection khớp original logits/argmax.
- [ ] Absolute positions/RoPE đúng, compact length không thay logical cursor.
- [ ] Live 16 positions và block attention semantics giữ nguyên.
- [ ] Greedy verify khớp AR qua reject đầu/giữa/full acceptance.
- [ ] Pending anchor/EOS/cap không double count; sum G bằng output length.
- [ ] Rejected proposals/features không vào bank/index/slots.
- [ ] V0 từ chối sampling/batch chưa hỗ trợ.

**Được kết luận:** correctness trong phạm vi đã test; chưa có speedup/generalization.

## M1 — Dữ liệu và nhãn

- [ ] ID/content hashes tách train/validation/calibration/holdout.
- [ ] D0–D4 cohort chỉ exploratory/dev.
- [ ] Feature/state fingerprints, atomic COMPLETE và lazy prefix slices.
- [ ] Candidate guards/positions/cardinality/dedup/seeds hợp lệ.
- [ ] Candidate order không đổi labels; target state reset sạch.
- [ ] Actual acceptance rewards, tie/censoring/no-headroom report.
- [ ] Current/future teacher information chỉ là label.
- [ ] Capture/label GPU-giờ tính vào adaptation cost.

**Được kết luận:** labels/replay theo contract; chưa chứng minh learned policy tốt.

## M2 — Selector

- [ ] Query pre-draft, compact index/valid masks.
- [ ] Hard Top-K budget/guards/unique positions đúng.
- [ ] Preference gradient tới selector, frozen weights grad None.
- [ ] Hard selected-set utility được đo, ngoài pairwise accuracy.
- [ ] Learned vs heuristic ở cùng budget.

**Được kết luận:** selector quality signal trên validation; holdout claim mở.

## M3 — Compressor và train integration

- [ ] Full/streaming build tương đương FP32 tolerance 1e-5.
- [ ] Empty slots mask, midpoint không dùng future bucket span.
- [ ] Raw+slots accounting rõ; global overlap không gọi disjoint residual.
- [ ] Compressor/adapters grad nonzero qua frozen draft.
- [ ] Candidate-prefix row alignment đúng; surrogate không bị gọi exact acceptance loss.
- [ ] Phase-specific optimizer/accumulation/finite losses/grad norms.
- [ ] Full validation mỗi 50 steps/cuối phase, logs/progress/error status.
- [ ] Checkpoint/in-memory eval đồng nhất; atomic save/resume/RNG/sampler test.
- [ ] Capture+labels+train+eval compute cap được thực thi.

**Được kết luận:** pipeline hoạt động; compression chưa cần thiết chỉ vì loss giảm.

## M4 — Physical inference và cost

- [ ] Dense/AMR cùng cached verifier/backend/dtype/hardware/prompt/output.
- [ ] Genuine compact K/V reads, không dense masked N-key attention.
- [ ] Full logical target prefix và compact draft dimensions tách rõ.
- [ ] Timing gồm bootstrap/index/slots/gather/update/switch/terminal.
- [ ] CPU/device/wall scopes không bỏ sót hoặc double count.
- [ ] Resident bank/index/features/slots/temporaries và total VRAM được báo.
- [ ] Oracle/reference timing không vào deployable speedup.
- [ ] JsonlWriter/base/spec schema/summary/ROUGE đúng.
- [ ] Gate calibration khóa; short-context/lazy-build cost được đo.

**Được kết luận:** performance cấu hình đã đo; retained entries không đồng nghĩa resident memory savings.

## M5 — Scientific result

- [ ] Hypothesis/variant/gates khóa trước holdout.
- [ ] A/G/survival gain đo thật, không suy từ attention mass.
- [ ] Throughput/E2E paired document statistics, 95% CI.
- [ ] Holdout không tune; bootstrap document clusters.
- [ ] Selection-only/compression-only/hybrid matched budget/cost.
- [ ] Acceptance vs attention labels so cùng adaptation economics.
- [ ] Ordinary DFlash fine-tune control nếu claim train efficiency.
- [ ] Greedy parity 100%; sampling claim cần verifier/tests riêng.
- [ ] Prefill/decode/E2E và batch/model/domain limitations rõ.
- [ ] Bibliography/novelty kiểm chứng riêng, không claim first từ proposal.
- [ ] Negative result và go/simplify/no-go decision lưu đầy đủ.

## Mẫu ghi nhận milestone

~~~text
Milestone: M0/M1/M2/M3/M4/M5
Ngày:
Code revision / diff hash:
Config / split fingerprint:
Command / exit code:
Run ID / artifact:
Document/state success và failed counts:
Metric / CI / compute:
Gate đạt:
Gate chưa đạt và nguyên nhân:
Quyết định: tiếp tục / đơn giản hóa / dừng
Claim được phép:
~~~

Mẫu để điền lúc thực thi; không là bằng chứng milestone đã đạt.
