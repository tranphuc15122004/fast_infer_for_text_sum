# Progress Log

## Session: 2026-08-28

### Phase 1: Requirements & Discovery
- **Status:** complete
- **Started:** 2026-08-28
- Actions taken:
  - Đọc AGENTS.md và các skill bắt buộc.
  - Khảo sát tree, schema dữ liệu hiện tại, loader, runner, collector và test contract.
  - Đọc tài liệu đề xuất LongBench 5 task × 200 mẫu.
  - Xác định xung đột semantics giữa summarization runner hiện tại và LCC/RepoBench-P code completion.
  - Xác nhận source LongBench 5 task và tokenizer Llama 3.1 đã có trong cache local.
  - Hoàn tất design spec sau khi người dùng yêu cầu triển khai.
  - Hoàn tất implementation plan tại `docs/superpowers/plans/2026-08-28-longbench-200.md`.
  - Viết helper/config/builder/validator/analyzer và test TDD.
  - Chạy focused test: `6 passed`.
  - Build small-scale offline tại `/tmp/fast_infer_longbench_small`: 5 × 20 = 100 record.
  - Validator pass, schema/duplicate/bin/spot-check pass; ghi lại thống kê token và prompt.
- Ở Phase 4, build chính thức offline tại `data/longbench_200`: 5 × 200 = 1.000 record.
- Validator chính thức pass; hai dataset code có đúng 5 length bins × 40 record.
- Rebuild full-scale độc lập tại `/tmp/fast_infer_longbench_full_repeat_v2` byte-identical với output chính thức, gồm cả manifest.
- Cập nhật loader, collector, metrics, README và tài liệu benchmark để route code-completion không qua ROUGE.
- Files created/modified:
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/superpowers/specs/2026-08-28-longbench-200-design.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Repo/source discovery | `rg --files`, `find data` | Xác định source LongBench local | Chưa có source LongBench trong repo | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-08-28 | Không tìm thấy source LongBench local | 1 | Chưa chạy build; yêu cầu source-root/cache explicit trong thiết kế |
| 2026-08-28 | `.git/index.lock`: Read-only file system khi commit spec | 1 | Không commit trong sandbox; tiếp tục với file spec đã tạo |
| 2026-08-28 | Test duplicate ID dừng ở thiếu manifest fixture | 1 | Bổ sung manifest tối thiểu vào fixture để kiểm tra đúng nhánh duplicate |
| 2026-08-28 | Lệnh rebuild dự phòng dừng ở RepoBench-P trước khi ghi output | 1 | Không lặp build đồng thời; giữ output small-scale đầu tiên đã pass và chuyển determinism sang kiểm tra helper/4 file đã ghi |
| 2026-08-28 | Test collector route giả định metric aggregate nested | 1 | Sửa expectation theo interface flat hiện tại của `aggregate_semantic`/`aggregate_code_completion` |

### Small-scale checkpoint
- **Status:** complete; user đã phê duyệt full-scale
- **Output:** `/tmp/fast_infer_longbench_small`
- **Result:** 100 records, 20/task, validator PASS, 0 duplicate ID, code bins 4/4/4/4/4.
- **Full-scale result:** `data/longbench_200`, 1.000 records, validator PASS, 0 duplicate ID, LCC/RepoBench-P bins 40/40/40/40/40.

### Final verification
- **Status:** complete
- Focused dataset tests và `py_compile` pass.
- `pytest -q` toàn repo không collection được vì test vendored yêu cầu optional packages (`seaborn`, `flex_prefill`, `llmlingua`, `minference`, `fire`, `yunchang`, `sglang`); đây là giới hạn môi trường local, không liên quan builder canonical.

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: Verification & Delivery — complete |
| Where am I going? | Bàn giao; các run benchmark tiếp theo dùng `data/longbench_200` |
| What's the goal? | Canonical LongBench 1.000 mẫu, schema chung, manifest reproducible, runner/evaluator tương thích |
| What have I learned? | Runner/collector hard-code representative_100; LCC/RepoBench-P cần task-specific prompt/metric |
| What have I done? | Đã tạo/validate dataset chính thức, cập nhật evaluator và ghi nhận giới hạn test vendored |

## Phase 6: LongBench inference matrix

- **Status:** implementation complete; B200 execution pending on the target server.
- Added the shared `LONG_BENCH_*` master-config namespace and
  `scripts/run_longbench_200.sh`; all new launchers use the shared Python 3.12
  runtime and do not create per-baseline virtual environments.
- Added `scripts/run_longbench_200.py` with `smoke`, `representative` and `full`
  profiles, deterministic subsets, unique run directories, child timeout,
  logs, source-manifest hash and `run_manifest.json`.
- Added registry/converters/preflight in
  `scripts/common/longbench_adapter.py`; EAGLE3 and SpecExtend preserve source
  IDs. MagicDec now has a canonical prompt branch using its SnapKV engine and
  converted `.pth` checkpoint. SSSD/FAFO remain aggregate scope when the
  upstream runner cannot return per-sample timings.
- Added synchronized vanilla HF eager and vanilla FA FlashAttention-2
  inference, including model-load, prefill/TTFT/decode/E2E, TPOT, throughput,
  QPS, peak-memory and runtime metadata where the backend exposes them.
- Collector now reads nested run output, tracks coverage/status separately and
  routes summarization to ROUGE/BLEU versus LCC/RepoBench-P to exact/edit code
  completion metrics.

### Verification evidence

| Check | Result |
|---|---|
| `pytest -q tests` | **108 passed** |
| `.venv` interpreter | Python **3.12.13** |
| Canonical CPU smoke | **45/45 cells**, all `unsupported_cpu`, zero timing values |
| Collector on CPU smoke | `success=0`, `unsupported_cpu=45`, no speed keys |
| Full profile on CPU without override | correctly exits with CUDA-required error |
| `.venv` shared-env preflight | torch/transformers/vLLM/triton/dflash/LLMLingua imports pass; local flash-attn missing and flashinfer cache is read-only |

CPU/T4 cannot produce valid GPU benchmark numbers in this environment. On B200,
copy the master example to the external master path, verify local model and
draft/checkpoint paths, then run the same launcher with `--mode smoke` before
`representative`/`full`.

### Server setup bootstrap
- **Status:** complete.
- **Requested paths:** repository under `/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum`; stable data/config under `/workspace/storage-shared/nlp/dungdx4/phuc_projects/data`.
- **Actual artifact:** `scripts/setup_server_env.py` with idempotent checks,
  optional initialization, shared-data symlinks and direct `python3` system
  runtime validation; server-side venv creation has been removed.
- **Verification:** `pytest -q tests` → **111 passed**; `py_compile` and
  `git diff --check` pass.

### Canonical server profile
- **Status:** complete.
- Đã ghi repository, shared data, master config, runtime `python3` hệ thống
  Python 3.12 và lệnh setup/benchmark tại `docs/server_environment.md`.
- Đã cập nhật `AGENTS.md`, README, tài liệu benchmark active và
  `config/master.path` theo server path do người dùng cung cấp.

### Runtime compatibility remediation
- **Status:** complete locally; B200 smoke rerun pending.
- Đã sửa toàn bộ traceback đã cung cấp cho các adapter DFlash, MagicDec,
  LongSpec, EAGLE3, SpecExtend và FAFO; bổ sung test hồi quy riêng tại
  `tests/test_runtime_compat_fixes.py`.
- Đã cho phép SSSD smoke chạy với datastore rỗng theo contract upstream;
  benchmark retrieval vẫn yêu cầu `.idx` đúng model/tokenizer.
- Verification: `pytest -q tests` → **119 passed**, `git diff --check`,
  `py_compile` và preflight canonical 45 cell đều pass theo giới hạn CPU/T4.
