# Mẫu hồ sơ và báo cáo thực nghiệm

Ngày: **2026-10-09**. Sao chép mẫu vào `<run_root>/evidence/experiment_report.md`, điền từ artifacts thật. `chưa đo`/`pending` là trạng thái hợp lệ; không điền số minh họa như kết quả. Tiêu chí ở [experiment_matrix](experiment_matrix.md), schema ở [measurement_contract](measurement_contract.md).

## 1. Danh tính và scope

| Field | Giá trị |
|---|---|
| Run ID / ngày UTC / người vận hành | chưa điền |
| Phase / natural hay fixed tokens | chưa điền |
| Code commit / dirty / implementation SHA256 | chưa điền |
| Target / draft / tokenizer path và SHA256 | chưa điền |
| Python / torch / CUDA / transformers / kernel versions | chưa điền |
| GPU model / UUID / driver / backend target và draft / dtype | chưa điền |
| Master config path; các effective nonsecret settings | chưa điền |
| Split / calibration / lock / report paths và SHA256 | chưa điền |
| Greedy temperature / thinking / input cap / output cap / EOS | chưa điền |
| Selector / target layer / chunk / anchors / recent / refresh | chưa điền |
| B×gamma grid / fixed pair / best full gamma | chưa điền |
| Priors / min support / decay / margin / fallback | chưa điền |
| Statistics mode / cost mode / timing mode | chưa điền |
| Repetitions / seeds / warmups / block schedule | chưa điền |

Preregistration và sai lệch protocol: ghi file/hash, điều gì đổi, vì sao, khi nào đổi, đã xem dev/test gì lúc đó. Dataset/code/runtime mới cần run hoặc calibration tương ứng; ghi quan hệ với campaign cũ.

## 2. Dữ liệu, exposure và coverage

| Dataset | Cal groups/records | Dev groups/records | Test groups/records | Actual input token min/median/p95 | Actual output min/median/p95 |
|---|---|---|---|---|---|
| GovReport | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| QMSum | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |
| Multi-News | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

- Probe exposure registry: số IDs, matched_dev, absent_verified, unresolved; file/hash và căn cứ đối soát.
- Group overlap calibration/dev/test: số violations; kiểm duplicate contexts/provenance/truncation/query variants.
- QMSum CI theo source conversation; số queries không dùng như số independent documents.
- Expected cells từ preregistration/lock: liệt kê toàn bộ variants/modes; missing cả variant vẫn là missing cell.
- Expected requests/cell = records trong split × repetitions; đối chiếu IDs/groups với manifest, không chỉ so counts.

| Cell (variant/mode) | Expected | Success | Error | Missing | Source groups | Token mismatches | Summary complete? |
|---|---:|---:|---:|---:|---:|---:|---|
| chưa điền | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | pending |

## 3. GPU validation và calibration

| Hạng mục | Kết quả | Evidence path/hash |
|---|---|---|
| Full adapter ↔ vendored tensor/logit/rollout parity | pending | chưa điền |
| Adapter ↔ AR greedy IDs | pending | chưa điền |
| Sparse positions/GQA/sliding masks | pending | chưa điền |
| Cache rollback/committed features/parent row | pending | chưa điền |
| EOS/cap/remaining/protection/fallback cases | pending | chưa điền |
| Collector kernel / instrumentation control | pending | chưa điền |
| Sampling audit nếu có claim sampling | pending hoặc not_applicable_greedy | chưa điền |

Calibration cohort: groups/records, actual prefix checkpoints, EOS-short trajectories, repetitions/action/state. Unique-state support và independent source groups được báo riêng; repeated timings không tăng outcome support.

| Action / context bucket / refresh | Unique states | Source groups | Timing reps | Cost median/dispersion | Supported? | Backoff/fallback reason |
|---|---:|---:|---:|---|---|---|
| chưa điền | chưa đo | chưa đo | chưa đo | chưa đo | pending | chưa điền |

Báo controller-overhead profile riêng theo policy; unsupported/unseen buckets và tỷ lệ rollout fallback. Liệt kê action nào không có dữ liệu và lý do; không gán null cost thành 0.

## 4. Tuning chỉ trên dev

Lưu toàn bộ 16 fixed-pair và bốn full-gamma results, selector study, output scope và cách chọn. Ghi best fixed `(B*,gamma*)`, best full `gamma_full*`, selected selector/policy, statistic, uncertainty và tie-break. Final dev được chạy lại với controls/actions cuối: file/hash.

| Setting | Dev macro speedup so AR | CI95 | Coverage/exact hợp lệ? | Quyết định |
|---|---|---|---|---|
| chưa điền | chưa đo | chưa đo | pending | chưa điền |

Không dùng test để sửa selector, priors/cutpoints, margin, action grid hoặc threshold gates.

## 5. Bảng kết quả chính

Speedup là `exp(mean_dataset(mean_pair(log(T_baseline/T_candidate))))`; bootstrap 10.000 lần theo source clusters/allocation strata, seed 42, CI95. Latency baseline/candidate và exact coverage phải đi cùng speedup. Báo từng dataset và macro; dùng cùng scope/statistics mode cho một bảng.

| Dataset / scope / mode | Candidate so baseline | Paired requests/groups | E2E baseline/candidate (ms) | Geometric speedup | CI95 | Exact / coverage / provenance | Claim hợp lệ? |
|---|---|---|---|---|---|---|---|
| chưa điền | joint so AR | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | A so full fixed | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | B-history so best full gamma | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | B-entropy so best full gamma | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | B-entropy so B-history | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | joint so best fixed pair | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |
| chưa điền | joint so independent | chưa đo | chưa đo | chưa đo | chưa đo | pending | pending |

`headline_valid` của reporter: ghi giá trị thật; reviewer kiểm thêm full planned matrix, G0–G6 và ngưỡng gain. Reporter không tự tạo hết per-dataset/signal/component comparisons; phân tích bổ sung phải giữ cùng pairing/bootstrap, không chia hai aggregate CI.

## 6. Ablation và interaction

| Mode / comparison | Speedup CI95 | Prediction/regret evidence | Kết luận được phép |
|---|---|---|---|
| Online joint so independent | chưa đo | chưa điền | pending |
| Frozen joint so independent | chưa đo | chưa điền | pending |
| Online/frozen joint so best fixed | chưa đo | chưa điền | pending |
| Joint so no entropy | chưa đo | chưa điền | pending |
| Joint so no source relevance | chưa đo | chưa điền | pending |

Action surface: theo cùng immutable prefix, plot B×gamma cho accepted prefix, round cost và ms/commit; ghi source/state counts, uncertainty và setup exclusions. Budget nào đổi argmin gamma vượt timing noise? Local oracle gap là prefix-replay diagnostic; không trình bày như E2E oracle rollout.

Nếu entropy chỉ correlate mà không cải thiện paired rollout, chưa giữ contribution entropy. Nếu gain chỉ online, ghi hiệu quả của policy cùng online statistics; chưa quy riêng cho interaction.

## 7. Latency, acceptance, memory và chất lượng

| Variant / dataset / mode | TTFT / TPOT / E2E mean, median, p95 | Accept length / rate | Effective emitted acceptance | Refresh / fallback / action usage | Peak allocated/reserved GiB / bank bytes | ROUGE-1/2/L natural |
|---|---|---|---|---|---|---|
| chưa điền | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo | chưa đo |

Acceptance dùng `L/gamma` và shared avg length `mean(1+L)` trên speculative rounds; effective metrics trim EOS riêng. AR không có speculative denominator. Đối soát `processed_commits + final_pending_emitted_count - trimmed_commits = output_tokens`.

Component breakdown: prefill, initial bank projection, signal/controller/selection/gather/bank/draft/verify, wasted work. Ghi host/GPU timing scope; không ép component sum bằng E2E. Báo request preparation/deployment total riêng nếu đo. Ghi model load/hash/JIT/warmup ngoài timing cùng wall duration cả campaign.

ROUGE lấy reference thật theo helper chung; missing reference là unavailable, không 0. Greedy IDs trùng AR thì quality phải nhất quán trong cùng scope. Fixed-token chạy sau EOS thuộc performance-only table riêng; không dùng cho natural quality. V1 vẫn giữ full bank, nên claim giảm memory cần measurement trực tiếp.

## 8. Verdict từng gate và bàn giao

| Gate | Ngưỡng / điều kiện | Status | Evidence/hash | Việc còn lại |
|---|---|---|---|---|
| G0 | Asset/runtime/source mapping/exposure hợp lệ | pending | chưa điền | chưa điền |
| G1 | Full/vendored/AR exact và boundaries đúng | pending | chưa điền | chưa điền |
| G2 | Sparse/cache/parent causal đúng | pending | chưa điền | chưa điền |
| G3 | A ≥1.05× full, CI lower>1 | pending | chưa điền | chưa điền |
| G4 | B ≥1.03× best full, CI lower>1; entropy contribution riêng | pending | chưa điền | chưa điền |
| G5 | Joint ≥1.03× fixed và independent, CI lower>1; interaction/frozen evidence | pending | chưa điền | chưa điền |
| G6 | Full locked matrix complete/exact/provenance, CI và quality báo đủ | pending | chưa điền | chưa điền |

Status dùng `pass`, `fail`, `pending`; check optional có thể `not_applicable` kèm lý do. Không cho unsupported/missing required case thành pass. Candidate lock mang trạng thái `candidate_locked_pending_heldout`, không tự là verdict scientific gates.

Kết luận điền ngắn:

- **Đã chứng minh:** kết quả, phạm vi model/workload/runtime và evidence.
- **Chưa chứng minh:** gate/hypothesis còn pending hoặc fail; limitations.
- **Quyết định:** mở bước tiếp theo / chạy lại cell lỗi / thu hẹp claim / dừng vì kết quả âm.
- **Bàn giao:** server paths, run ID, immutable hashes, completed/pending cells, lệnh resume cùng config, log/artifact vị trí và resource requirements.

Lưu logs/exit codes và error rows; resume cùng signature, không xóa bằng chứng mismatch. Đổi method sau nhìn heldout cần protocol/cohort mới. Kết luận cuối dựa vào measured evidence, không vào việc tài liệu hoặc implementation đã đầy đủ.
