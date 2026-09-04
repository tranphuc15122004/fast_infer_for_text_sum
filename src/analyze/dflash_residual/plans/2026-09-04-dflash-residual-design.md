# Thiết kế thực nghiệm residual headroom của DFlash/DFlash2

## Câu hỏi trung tâm

Khi context dài lên, DFlash thất bại chủ yếu vì token đúng biến mất khỏi
candidate set, hay vì DFlash2 không chọn được token đúng dù token đó vẫn còn
trong candidate set?

Phạm vi này thay thế hướng GroundSync/Target-KV trong phase hiện tại. Không
mở thêm KV adapter và không lặp lại claim cũ chỉ rằng Recall@16 cao hơn
Recall@1.

## Giả thuyết và gate

- **H0 — alignment sanity:** official runner và custom runner trên cùng
  canonical samples phải có acceptance dương và tái hiện gần nhau. Nếu không,
  dừng phân tích khoa học để sửa checkpoint, tokenizer, prompt, thinking mode,
  feature layer, mask/block hoặc runner.
- **H1 — summarization degradation:** sau khi H0 pass, MAT, survival
  `S(j)` và candidate recall của summarization giảm so với canonical task.
- **H2 — candidate coverage:** `Recall@M(j, L)` giảm khi context length hoặc
  draft depth tăng, với `M ∈ {1, 4, 8, 16}`.
- **H3 — residual selection gap:** trên candidate lattice cố định, DFlash2
  thu hồi ngày càng ít phần oracle Top-16 headroom khi `L` tăng.
- **H4 — context-induced suffix decay:** hệ số tương tác của
  `log(L) × j` trong logistic model của Recall@16 là âm và bootstrap CI không
  vượt qua 0.
- **H5** chưa triển khai trong phase này; chỉ mở sau khi H3 pass và sẽ dùng
  error taxonomy source/global.

Gate định lượng mặc định chỉ là ngưỡng admission, không phải kết luận thay
   cho confidence interval:

- positive acceptance: `MAT > 0` và có tối thiểu 5 blocks hợp lệ;
- candidate-generation signal: relative short-to-long Recall@16 drop ít
  nhất 15% hoặc interaction CI âm;
- selection signal: `rho_D2 < 0.5` khi denominator oracle headroom dương;
- stop signal: `rho_D2 ≥ 0.7` và Recall@16 cao ở regime long.

Mọi gate đều trả `PASS`, `FAIL`, `INCONCLUSIVE` hoặc `UNAVAILABLE`; thiếu
  class/context/depth/document không được tự động tính là FAIL.

## Thiết kế dữ liệu và schema

Mỗi dòng trace là một draft position `j` (đánh số 1-based) trong một
speculative block. Các field bắt buộc:

```text
schema_version = dflash_residual.trace.v1
status, run_id, sample_id, document_id, dataset, task_regime
context_length, context_bin, round_index, draft_position, max_depth
target_token_id, candidate_token_ids
dflash_selected_token_id, target_token_source
```

Field block-level được lặp lại có chủ ý để JSONL có thể lọc độc lập:

```text
accepted_draft_len, committed_tokens, block_size, native_block_size
```

`candidate_token_ids` luôn giữ thứ tự score giảm dần và có thể ngắn hơn 16
ở vocab/runner lỗi. `dflash2_selected_token_id` là optional; nó chỉ được
điền bởi selection trace đã join theo `(run_id, sample_id, round_index,
draft_position)`. `target_token_source` phải là `verifier_posterior` hoặc
`canonical_continuation`; không trộn hai semantics trong cùng một run.

Collector DFlash dùng target posterior tại vị trí verifier (`posterior[:, :-1]`)
để phân tích đúng state serving. Target-only canonical continuation có thể
được nạp riêng cho đối chiếu P1, nhưng không được dùng thay thế lặng lẽ.

## Các pha

### P0 — Canonical reproduction

Input là output JSONL của official runner và custom/instrumented runner. Bộ
chuẩn hóa hỗ trợ cả per-block trace và output benchmark có
`acceptance_lengths`. P0 kiểm tra cùng sample IDs, protocol fingerprint,
native block size, `MAT`, `S(1/2/4/8)` và positive acceptance. Chênh lệch được
báo cáo theo sample và theo regime.

### P1 — Task-regime comparison

Trace được gắn `task_regime` trong `{canonical, cnn_dm, govreport,
multi_news, other}`. Analyzer tính MAT, `S(1/2/4/8)`, Recall@1/4/8/16 theo
regime và context bin. Context bin không được suy ra từ tên file nếu field
đã có trong trace.

### P2 — Candidate Coverage Anatomy

Analyzer tính hit indicator cho mọi `M` không vượt quá số candidate có sẵn,
theo `(dataset, context_bin, draft_position)`, sinh bảng JSON/CSV và heatmap
Recall@16 `(L, j)`. Collector hỗ trợ lặp cùng sample ở các cap input
`1K, 2K, 4K, 8K, 16K`; cap, truncate side và block size được ghi trong
manifest.

### P3 — Residual-headroom decomposition

Với mỗi block, tính:

```text
MAT_D   = mean(longest prefix dflash_selected == target_token)
MAT_D2  = mean(longest prefix dflash2_selected == target_token)
MAT_O16 = mean(longest prefix target_token ∈ candidate Top-16)
G_sel   = MAT_O16 - MAT_D
rho_D2  = (MAT_D2 - MAT_D) / G_sel
```

Các block thiếu DFlash2 selection hoặc có oracle denominator không dương được
đưa vào count unavailable; không bị gán 0. Báo cáo tách `candidate_miss`
và `selection_error`.

### P4 — Context × depth interaction

Fit logistic regression bằng IRLS thuần NumPy nếu có, với fallback
dependency-light:

```text
hit ~ 1 + log1p(context_length) + draft_position
          + log1p(context_length) * draft_position
```

Bootstrap theo document, không bootstrap độc lập từng token. Gate H4 chỉ PASS
khi upper 95% CI của hệ số interaction < 0 và có tối thiểu 5 documents ở cả
short/long và tối thiểu 2 draft positions.

## Kiến trúc code

- `schema.py`: validation/normalization và composite trace keys.
- `metrics.py`: acceptance, survival, Recall@M, oracle-prefix, MAT,
  decomposition, logistic interaction và bootstrap.
- `io.py`: JSONL input, official-output adapter, DFlash2 selection join,
  CSV/JSON output.
- `plotting.py`: heatmap/line plot bằng matplotlib tùy chọn; thiếu backend
  chỉ làm plot `UNAVAILABLE`, không làm mất metrics.
- `alignment.py`: H0 normalization và report.
- `trace_dflash.py`: collector GPU tùy chọn, mirror semantics của official
  DFlash Transformers runner, không thay đổi production runner.
- `run.py`: CLI cho `p0`, `p1`, `p2`, `p3`, `p4`, `all`.

## Validation scope

- TDD unit tests cho schema, empty/partial rows, exact prefix semantics,
  candidate miss, rho denominator, document bootstrap và CLI.
- CPU synthetic end-to-end chạy `all`, sinh metrics/CSV/Markdown và kiểm tra
  report không overclaim khi DFlash2 thiếu.
- `py_compile`, `pytest` và `git diff --check` là validation bắt buộc.
- GPU collector chỉ được coi là executable handoff; host dev T4 không có
  CUDA nên không claim GPU result trong local validation.

## Quyết định không thuộc scope

Không train selector, không build source-aware selector và không mở H5 trong
phase này. Các kết luận chỉ nói về bottleneck được quan sát trên trace; không
suy ra DFlash/DFlash2 vô dụng về bản chất ngoài scope model/task/config đã ghi.
