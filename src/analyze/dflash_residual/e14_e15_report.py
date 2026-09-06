"""Build the consolidated Vietnamese E14/E14b/E15 experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(root: Path) -> str:
    e14 = _load(str(root / "e14/metrics.json"))
    e14b = _load(str(root / "e14b/metrics.json"))
    e15 = _load(str(root / "e15/metrics.json"))
    e15_cnn = _load(str(root / "e15_in_domain_cnn_dm/metrics.json"))
    e15_gov = _load(str(root / "e15_in_domain_govreport/metrics.json"))
    train_log = _load_jsonl(root / "e15_adapt_cnn_gov_100step/metrics.jsonl")
    lines: list[str] = [
        "# Báo cáo tổng hợp E14–E15: task-induced candidate-lattice degradation",
        "",
        "Ngày chạy: 2026-09-06. Phạm vi gồm E14 (marginal–joint decomposition), E14b (target-entropy matched control) và E15 (minimal summarization adaptation). Báo cáo này ghi nhận kết quả đã chạy, không giả định trước proposal.",
        "",
        "## Kết luận điều hành",
        "",
        "1. E14 cho thấy phần suy giảm `MAT_O16` giữa canonical và summarization được giải thích chủ yếu bởi marginal Top-16 candidate quality. Khi giữ normalized coherence của canonical và thay bằng marginal recall của summarization, MAT phản thực tế còn thấp hơn MAT quan sát được ở cả ba dataset; joint component vì vậy âm (−14.5 đến −16.7% của tổng degradation). Đây không phải bằng chứng summarization có joint coherence tuyệt đối tốt hơn, mà là bằng chứng joint coherence chuẩn hóa không phải nguồn suy giảm chính trong phép so sánh này.",
        "2. E14b vẫn thấy khoảng cách sau khi ghép target entropy tại đúng draft position: standardized MAT của summarization là 5.21–6.68 so với canonical 9.38. Vì vậy target entropy đơn thuần không giải thích hết task gap, nhưng control này bị giới hạn bởi canonical chỉ có 8 documents.",
        "3. E15 task adaptation tối thiểu (100 bước, 100 mẫu train CNN/DM + GovReport, đúng DFlash loss) không cải thiện oracle lattice trên 50 Multi-News held-out: MAT_O16 giảm 0.0258, oracle recovery −0.46%, CI bootstrap 95% [−4.87%, +4.51%]. Hai kiểm tra in-domain 20 mẫu cũng không cho gain oracle nhất quán.",
        "",
        "**Quyết định:** đóng nhánh `joint-prefix/joint-lattice training` ở mức hypothesis hiện tại; chưa có evidence để coi joint coherence là bottleneck chính hoặc để mở proposal mới. Kết quả hiện chỉ ủng hộ một kết luận thận trọng: summarization gây task-specific candidate-generation mismatch, còn nguyên nhân đủ để sửa bằng adaptation đơn giản vẫn chưa được xác định.",
        "",
        "## 1. Thiết lập và tính tái lập",
        "",
        "| Thành phần | Cấu hình thực tế |",
        "|---|---|",
        "| GPU/runtime | Tesla T4; `/home/tuantb/miniconda3/envs/myenv/bin/python3.11`; không dùng môi trường B200 |",
        "| Target | Qwen3-4B local snapshot `1cfa9a7208912126459214e8b04321603b3df60c` |",
        "| DFlash base | `z-lab/Qwen3-4B-DFlash-b16`, local snapshot `61ab4992e5b5ec5913c7f8a9618367b4309533a3` |",
        "| Precision/attention | bfloat16 + SDPA; float16 bị loại vì non-finite logits trên T4 |",
        "| Decoding trace | native block size 16, Top-M=16, context cap 1024, max new tokens 16 |",
        "| Prefix positions | 15 positions hữu ích trong block 16; position 0 là anchor, nên các tổng MAT dùng 1–15 |",
        "| Dữ liệu E14 | canonical GSM8K 8 documents; CNN/DM, GovReport, Multi-News mỗi dataset 100 documents; trace E11 không lỗi |",
        "| Dữ liệu entropy E14b | canonical 8 documents + 50 documents/dataset summarization; tất cả entropy finite, không lỗi schema |",
        "| Dữ liệu E15 | train 50 CNN/DM + 50 GovReport; held-out 50 Multi-News; in-domain kiểm tra 20 CNN/DM + 20 GovReport |",
        "",
        "Tất cả trace E14b đều qua validation: canonical 555 rows, CNN/DM 5,415 rows, GovReport 5,490 rows và Multi-News 6,135 rows; error rows = 0.",
        "",
        "## 2. E14 — Marginal versus joint decomposition",
        "",
        "Với `H_j = 1[target ∈ Top16_j]`, đo `R_j=P(H_j=1)`, `J_j=P(H_1=…=H_j=1)` và `C_j=J_j/∏R_r`. Phản thực tế giữ `C_j` của canonical nhưng dùng marginal recall của summarization. Khi đó `MAT_marginal_cf` là oracle MAT nếu chỉ thay candidate availability, còn coherence giữ như canonical.",
        "",
        "### 2.1 Kết quả theo workload",
        "",
        "| Dataset | Blocks | Docs | R16 at position 15 | J16@4 | J16@8 | J16@15 | C16@8 | C16@15 | MAT_O16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in e14["groups"].items():
        lines.append(
            f"| {name} | {item['blocks']} | {item['documents']} | {_fmt(item['marginal_recall'].get('15'))} "
            f"| {_fmt(item['joint_survival'].get('4'))} | {_fmt(item['joint_survival'].get('8'))} "
            f"| {_fmt(item['joint_survival'].get('15'))} | {_fmt(item['coherence'].get('8'))} "
            f"| {_fmt(item['coherence'].get('15'))} | {_fmt(item['mat_o16'])} |"
        )
    lines.extend([
        "",
        "Lưu ý: cột `R16 at position 15` là marginal Top-16 recall tại vị trí draft cuối được quan sát; `MAT_O16` là tổng survival positions 1–15, không phải trung bình marginal recall.",
        "",
        "### 2.2 Phân rã phản thực tế",
        "",
        "| Dataset | MAT canonical | MAT summary | MAT marginal-CF | Marginal component | Joint component | Marginal fraction | Joint fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in e14["decomposition"].items():
        lines.append(
            f"| {name} | {_fmt(item['mat_o16_canonical'])} | {_fmt(item['mat_o16_summary'])} | {_fmt(item['mat_o16_marginal_counterfactual'])} "
            f"| {_fmt(item['marginal_component'])} | {_fmt(item['joint_component'])} "
            f"| {_fmt(item['marginal_fraction'])} | {_fmt(item['joint_fraction'])} |"
        )
    lines.extend([
        "",
        "Tổng identity được giữ: `MAT_can − MAT_sum = (MAT_can − MAT_marginal-CF) + (MAT_marginal-CF − MAT_sum)`. Fraction >1 và joint fraction âm là hợp lệ trong decomposition counterfactual này; nó xảy ra vì summarization có normalized coherence cao hơn canonical baseline, nên coherence bù lại một phần marginal loss.",
        "",
        "### 2.3 Diễn giải E14",
        "",
        "E14 là **Outcome A — marginal dominates**, không phải Outcome B. Candidate coverage giảm theo task là thành phần chính. Tuy nhiên, canonical chỉ có 8 documents và khác task; vì vậy đây là decomposition mô tả, không phải causal proof rằng summarization “có coherence tốt”. Kết luận an toàn là chưa có evidence rằng normalized joint coherence bị suy giảm thêm sau khi đã điều chỉnh marginal recall.",
        "",
        "## 3. E14b — Control theo target entropy",
        "",
        "E14b ghi `target_entropy = −Σ p_T log p_T` từ target posterior tại mỗi verifier position, sau đó ghép theo 5 quantile bins chung của từng cặp canonical–summary và standardize prefix survival về phân bố entropy-bin của canonical. Đây là control mô tả; không phải matched-pairs causal estimate.",
        "",
        "| Dataset | Actual MAT_O16 | Entropy-standardized MAT | Canonical standardized MAT | Gap sau control | Gap còn lại / canonical |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in e14b["datasets"].items():
        if name == "canonical":
            continue
        gap = float(item["entropy_standardized_gap"])
        ref = float(item["mat_reference_entropy_standardized"])
        lines.append(
            f"| {name} | {_fmt(item['actual_mat_o16'])} | {_fmt(item['mat_entropy_standardized'])} | {_fmt(ref)} | {_fmt(gap)} | {_fmt(gap / ref if ref else None)} |"
        )
    lines.extend([
        "",
        "Kết quả cho thấy entropy matching không xóa gap: còn khoảng 2.69–4.17 MAT units tùy dataset. GovReport có standardized MAT cao hơn CNN/DM/Multi-News, nhưng vẫn thấp hơn canonical. Do canonical chỉ có 8 docs, các bin và survival estimate ở control cần được xem là exploratory; E14b đủ để loại explanation ‘chỉ do entropy’ ở mức hiện tại, chưa đủ cho CI mạnh.",
        "",
        "## 4. E15 — Minimal summarization adaptation",
        "",
        "### 4.1 Quy trình",
        "",
        "- Chuẩn bị 100 conversation teacher-forcing: 50 CNN/DM + 50 GovReport; source giữ tối đa 760 token, reference tối đa 180 token, tổng train sequence ≤1024 và 100/100 mẫu có assistant span liên tiếp.",
        "- Giữ nguyên DFlash 5-layer, target layer IDs `[1,9,17,25,33]`, block size 16, bfloat16, SDPA, warm-start từ checkpoint gốc và loss `dflash`.",
        "- Train 100 steps, batch size 1, 32 anchors, learning rate 1e-4, positional decay gamma 7; không thêm source signal, selector, pairwise/listwise/prefix-utility loss.",
        "- Export checkpoint MR-DFlash về HF format, sau đó chạy collector chuẩn trên trace mới. Smoke 1 step đã pass forward/backward/checkpoint trước run chính.",
        "",
        "### 4.2 Held-out Multi-News — gate chính",
        "",
        "| Metric | Original | Adapted | Delta |",
        "|---|---:|---:|---:|",
    ])
    for key, label in (("mat_d", "MAT_D"), ("recall_at_1", "Recall@1"), ("recall_at_16", "Recall@16"), ("mat_o16", "MAT_O16")):
        lines.append(f"| {label} | {_fmt(e15['baseline'][key])} | {_fmt(e15['adapted'][key])} | {_fmt(e15['adapted'][key] - e15['baseline'][key])} |")
    boot = e15["bootstrap"]["ci95"]
    lines.extend([
        "",
        f"Oracle recovery = `{(e15['adapted']['mat_o16'] - e15['baseline']['mat_o16']) / (e15['canonical_mat_o16'] - e15['baseline']['mat_o16']):.2%}`; gate promising yêu cầu >10%, strong yêu cầu >30%.",
        f"Paired document bootstrap 500 samples: delta MAT_O16 mean `{_fmt(boot['delta_mat_o16']['mean'])}`, 95% CI `{boot['delta_mat_o16']['ci']}`; delta Recall@16 mean `{_fmt(boot['delta_recall_at_16']['mean'])}`, CI `{boot['delta_recall_at_16']['ci']}`; oracle recovery mean `{_fmt(boot['adapted_oracle_recovery']['mean'])}`, CI `{boot['adapted_oracle_recovery']['ci']}`.",
        "",
        "### 4.3 Kiểm tra in-domain",
        "",
        "| Evaluation | Delta MAT_D | Delta Recall@16 | Delta MAT_O16 | Oracle recovery |",
        "|---|---:|---:|---:|---:|",
        f"| 20 CNN/DM | {_fmt(e15_cnn['adapted']['mat_d'] - e15_cnn['baseline']['mat_d'])} | {_fmt(e15_cnn['adapted']['recall_at_16'] - e15_cnn['baseline']['recall_at_16'])} | {_fmt(e15_cnn['adapted']['mat_o16'] - e15_cnn['baseline']['mat_o16'])} | {_fmt((e15_cnn['adapted']['mat_o16'] - e15_cnn['baseline']['mat_o16']) / (e15_cnn['canonical_mat_o16'] - e15_cnn['baseline']['mat_o16']))} |",
        f"| 20 GovReport | {_fmt(e15_gov['adapted']['mat_d'] - e15_gov['baseline']['mat_d'])} | {_fmt(e15_gov['adapted']['recall_at_16'] - e15_gov['baseline']['recall_at_16'])} | {_fmt(e15_gov['adapted']['mat_o16'] - e15_gov['baseline']['mat_o16'])} | {_fmt((e15_gov['adapted']['mat_o16'] - e15_gov['baseline']['mat_o16']) / (e15_gov['canonical_mat_o16'] - e15_gov['baseline']['mat_o16']))} |",
        "",
        "E15 không cho tín hiệu adaptation sửa lattice: held-out MAT_O16 giảm nhẹ và CI chứa 0; in-domain CNN/DM còn giảm rõ hơn, GovReport gần như không đổi. Đây là adaptation chẩn đoán ngắn (100 steps), nên không được dùng để phủ nhận mọi khả năng fine-tuning dài hơn; nó đủ để bác bỏ claim rằng task-matched data đơn giản đã sửa được bottleneck trong setup T4 này.",
        "",
        "## 5. Các lỗi triển khai được phát hiện và xác minh",
        "",
        "- Warm-start helper dùng sai API `load_state_dict` của PyTorch; đã sửa và smoke pass.",
        "- MR-DFlash dùng `Path.walk()` không có trong Python 3.11 của Conda T4; đã thay bằng `Path.rglob()`.",
        "- Capture trước đây mặc định lấy 1 target layer dù draft checkpoint có 5 layer; đã thêm `--target-layer-ids` và chạy đúng `[1,9,17,25,33]`.",
        "- E14b ban đầu dùng cumulative entropy; đã sửa sang entropy tại cùng draft position và canonical-weighted shared bins để tránh đưa prefix history vào control.",
        "",
        "## 6. Verification và artifact",
        "",
        "- Unit tests mới cho E14/E14b và trace entropy: 13 tests pass trong targeted suite.",
        "- Trace E14b: tất cả rows schema-valid, entropy finite, error rows = 0.",
        "- E15 smoke: 1 step pass; main checkpoint export strict 58/58 keys; adapted traces không có collector error.",
        "- Artifact chính: `e14/metrics.json`, `e14b/metrics.json`, `e15/metrics.json`, `e15_in_domain_cnn_dm/metrics.json`, `e15_in_domain_govreport/metrics.json`, cùng các `report.md` tương ứng trong folder này.",
        "",
        "## 7. Quyết định khoa học sau E14–E15",
        "",
        "Hiện chưa nên chuyển sang joint-prefix/lattice training proposal: E14 không cho thấy normalized joint coherence là nguồn degradation chính, E14b chỉ ra entropy không giải thích hết gap, còn E15 adaptation nguyên bản không recover oracle lattice. Claim được hỗ trợ tốt nhất hiện nay là: **summarization có task-specific marginal candidate-generation mismatch trong DFlash, nhưng mechanism/trainable fix vẫn chưa được xác định**. Các nhánh source, KV, selector/ranking-only và generic prefix reranker vẫn đóng.",
        "",
        "Nếu tiếp tục nghiên cứu, cần mở rộng canonical và adaptation budget trước khi thiết kế objective mới; không nên suy diễn từ E15 ngắn rằng một phương pháp train dài hơn chắc chắn thất bại.",
        "",
    ])

    # Detailed audit tables: keep the generated report useful without making
    # the primary result table unreadably wide.
    lines.extend([
        "## Phụ lục A — E14 per-position audit",
        "",
        "Các bảng dưới đây ghi toàn bộ `R16(j)`, `J16(j)`, independence baseline `∏R16` và `C16(j)` cho từng vị trí. `J16` là quantity quyết định oracle prefix; `R16` chỉ là marginal diagnostic.",
        "",
    ])
    for name, item in e14["groups"].items():
        lines.extend([
            f"### {name}",
            "",
            "| j | R16(j) | J16(j) | Independence | C16(j) |",
            "|---:|---:|---:|---:|---:|",
        ])
        for position in range(1, int(item["max_position"]) + 1):
            key = str(position)
            lines.append(
                f"| {position} | {_fmt(item['marginal_recall'].get(key))} | {_fmt(item['joint_survival'].get(key))} "
                f"| {_fmt(item['independent_survival'].get(key))} | {_fmt(item['coherence'].get(key))} |"
            )
        lines.append("")

    lines.extend([
        "## Phụ lục B — E14b entropy-control audit",
        "",
        "Entropy bins được xây riêng cho từng cặp canonical–dataset bằng pooled quantile edges. Ở mỗi position, chỉ các bins xuất hiện ở cả hai regime được dùng; trọng số được lấy theo phân bố canonical và renormalize trên bins chung. Đây là lý do standardized MAT có thể khác MAT raw.",
        "",
        "| Dataset | Records | Entropy edges | Shared bins min–max | Raw MAT (same 50-row trace) | Standardized MAT | Reference MAT |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for name, item in e14b["datasets"].items():
        shared = list(item.get("shared_bins_by_position", {}).values())
        if name == "canonical":
            shared_text = f"{min(shared)}–{max(shared)}" if shared else "—"
            lines.append(
                f"| {name} | {item.get('current_records')} | `{item.get('entropy_edges')}` | {shared_text} | {_fmt(item.get('actual_mat_o16'))} | {_fmt(item.get('mat_entropy_standardized'))} | {_fmt(item.get('mat_reference_entropy_standardized'))} |"
            )
            continue
        shared_text = f"{min(shared)}–{max(shared)}" if shared else "—"
        lines.append(
            f"| {name} | {item.get('current_records')} | `{item.get('entropy_edges')}` | {shared_text} | {_fmt(item.get('actual_mat_o16'))} | {_fmt(item.get('mat_entropy_standardized'))} | {_fmt(item.get('mat_reference_entropy_standardized'))} |"
        )
    lines.extend([
        "",
        "### E14b per-position standardized survival",
        "",
        "| j | CNN/DM current | CNN/DM canonical | GovReport current | GovReport canonical | Multi-News current | Multi-News canonical |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for position in range(1, 16):
        key = str(position)
        values: list[str] = [str(position)]
        for name in ("cnn_dm", "govreport", "multi_news"):
            item = e14b["datasets"][name]
            values.extend([_fmt(item["current_prefix_survival"].get(key)), _fmt(item["reference_prefix_survival"].get(key))])
        lines.append(f"| {' | '.join(values)} |")
    lines.extend([
        "",
        "## Phụ lục C — E15 training and evaluation audit",
        "",
        "### C.1 Training trace",
        "",
        "Loss/accuracy dưới đây chỉ chứng minh pipeline có gradient và checkpoint; chúng không phải primary acceptance metric. Training dùng 100 steps, warmup 4%, nên learning rate về 0 ở step 100.",
        "",
        "| Step | Loss | Accuracy | Learning rate | Grad norm | Elapsed (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    selected_steps = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
    for record in train_log:
        if int(record.get("global_step", -1)) in selected_steps:
            lines.append(
                f"| {record.get('global_step')} | {_fmt(record.get('loss'))} | {_fmt(record.get('acc'))} "
                f"| {_fmt(record.get('lr'), 7)} | {_fmt(record.get('grad_norm'))} | {_fmt(record.get('elapsed_s'))} |"
            )
    lines.extend([
        "",
        "Training loss dao động `8.4975 → 6.8916` ở các mốc đầu/cuối được log, còn accuracy batch dao động mạnh. Không có dấu hiệu loss collapse tới near-zero; do đó E15 chỉ là short pilot, không phải full convergence study.",
        "",
        "### C.2 Acceptance-prefix outcome chi tiết",
        "",
        "| Evaluation | Blocks original/adapted | Docs | MAT_D original/adapted | MAT_O16 original/adapted | R16 original/adapted |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Multi-News held-out | {e15['baseline']['blocks']} / {e15['adapted']['blocks']} | {e15['baseline']['documents']} / {e15['adapted']['documents']} | {_fmt(e15['baseline']['mat_d'])} / {_fmt(e15['adapted']['mat_d'])} | {_fmt(e15['baseline']['mat_o16'])} / {_fmt(e15['adapted']['mat_o16'])} | {_fmt(e15['baseline']['recall_at_16'])} / {_fmt(e15['adapted']['recall_at_16'])} |",
        f"| CNN/DM in-domain | {e15_cnn['baseline']['blocks']} / {e15_cnn['adapted']['blocks']} | {e15_cnn['baseline']['documents']} / {e15_cnn['adapted']['documents']} | {_fmt(e15_cnn['baseline']['mat_d'])} / {_fmt(e15_cnn['adapted']['mat_d'])} | {_fmt(e15_cnn['baseline']['mat_o16'])} / {_fmt(e15_cnn['adapted']['mat_o16'])} | {_fmt(e15_cnn['baseline']['recall_at_16'])} / {_fmt(e15_cnn['adapted']['recall_at_16'])} |",
        f"| GovReport in-domain | {e15_gov['baseline']['blocks']} / {e15_gov['adapted']['blocks']} | {e15_gov['baseline']['documents']} / {e15_gov['adapted']['documents']} | {_fmt(e15_gov['baseline']['mat_d'])} / {_fmt(e15_gov['adapted']['mat_d'])} | {_fmt(e15_gov['baseline']['mat_o16'])} / {_fmt(e15_gov['adapted']['mat_o16'])} | {_fmt(e15_gov['baseline']['recall_at_16'])} / {_fmt(e15_gov['adapted']['recall_at_16'])} |",
        "",
        "Số block có thể thay đổi giữa baseline và adapted vì accepted prefix thay đổi, làm round tiếp theo bắt đầu ở vị trí khác. Vì vậy so sánh chính được ghép theo document, không ghép row-index.",
        "",
        "### C.3 Bootstrap definition",
        "",
        "Bootstrap E15 lấy 50 document Multi-News chung, sample document có hoàn lại 500 lần, giữ toàn bộ block/rows của document, rồi tính lại MAT_D, Recall@16 và MAT_O16. Đây là document bootstrap, không phải iid row bootstrap; nó tránh đánh giá quá tự tin do các rows trong cùng document tương quan.",
        "",
        "## Phụ lục D — Protocol, commands và data flow",
        "",
        "### D.1 Data flow",
        "",
        "```text\nJSONL document\n    → Qwen3 chat-template, context cap 1024\n    → target prefill: hidden states at [1, 9, 17, 25, 33]\n    → DFlash non-causal block draft, native block=16\n    → target verification and greedy posterior\n    → one JSONL row per draft position\n    → offline R16 / J16 / C16 / MAT_O16 analysis\n```",
        "",
        "E14b thêm một nhánh target posterior entropy ở verifier position, nhưng không thay đổi candidate IDs hoặc acceptance logic. E15 chỉ thay draft checkpoint sau training; target model/tokenizer/collector vẫn giữ nguyên.",
        "",
        "### D.2 Representative commands",
        "",
        "```bash\n# E14/E14b offline analysis\npython3 -m src.analyze.dflash_residual.joint_lattice_run --canonical-trace outputs/dflash_residual/2026-09-05_prefix_utility/e11_canonical.jsonl --summary-trace outputs/dflash_residual/2026-09-05_prefix_utility/e11_cnn_dm.jsonl --summary-trace outputs/dflash_residual/2026-09-05_prefix_utility/e11_govreport.jsonl --summary-trace outputs/dflash_residual/2026-09-05_prefix_utility/e11_multi_news.jsonl --entropy-trace outputs/dflash_residual/2026-09-06_joint_lattice/entropy/e14b_canonical.jsonl --entropy-trace outputs/dflash_residual/2026-09-06_joint_lattice/entropy/e14b_cnn_dm.jsonl --entropy-trace outputs/dflash_residual/2026-09-06_joint_lattice/entropy/e14b_govreport.jsonl --entropy-trace outputs/dflash_residual/2026-09-06_joint_lattice/entropy/e14b_multi_news.jsonl --output outputs/dflash_residual/2026-09-06_joint_lattice --max-position 15 --bootstrap-samples 0\n```",
        "",
        "E15 dùng `PYTHONPATH=src /home/tuantb/miniconda3/envs/myenv/bin/python3.11`, không dùng Python server/B200. Checkpoint adaptation được kiểm tra strict 58/58 keys khi export HF; collector adapted có error rows = 0.",
        "",
        "## Phụ lục E — Review theo evidence ladder",
        "",
        "### CONFIRMED",
        "",
        "- **Confirmed:** E14 decomposition đã hoàn tất trên 37 canonical blocks và 716/751/847 summary blocks; marginal counterfactual MAT thấp hơn observed summary MAT ở cả CNN/DM, GovReport và Multi-News.",
        "- **Confirmed:** E14b entropy trace hoàn tất trên 555/5,415/5,490/6,135 rows, tất cả finite và schema-valid; entropy matching vẫn để lại gap.",
        "- **Confirmed:** E15 implementation path chạy ổn định qua smoke, train checkpoint, export strict và held-out collector; adaptation không đạt gate oracle recovery.",
        "",
        "### EXPLORATORY",
        "",
        "- **Exploratory:** diễn giải ‘marginal candidate-generation mismatch’ là cách đọc phù hợp nhất với E14, nhưng canonical chỉ có 8 docs và khác task nên chưa phải causal proof.",
        "- **Exploratory:** E14b loại explanation entropy-only trong control hiện tại, nhưng chưa có document-level CI mạnh do canonical nhỏ.",
        "- **Exploratory:** in-domain E15 trên 20 docs/dataset chỉ là sanity check phụ, không phải full in-domain study.",
        "",
        "### FAILED / INCOMPLETE",
        "",
        "- **Failed scientific gate:** E15 không đạt oracle recovery >10% trên held-out Multi-News; điểm ước lượng là −0.46%.",
        "- **Incomplete:** E15 chưa phải full-budget adaptation study; mới 100 steps, một seed, không đủ để kết luận mọi fine-tuning dài hơn sẽ thất bại.",
        "- **Incomplete:** E14b chưa có canonical replication đủ lớn để báo cáo CI đáng tin cậy.",
        "",
        "### HIGHEST VERIFIED RUNG",
        "",
        "E14/E14b đạt **R7 cho bounded diagnostic study**: đã có decision memo và artifact metrics/report hoàn chỉnh. E15 adaptation mới đạt **R5 short pilot**: real data, real GPU, checkpoint và held-out metric đã chạy, nhưng chưa phải R6 full-budget study. Passing software tests chỉ xác nhận code path, không xác nhận adaptation có hiệu quả.",
        "",
        "### EVIDENCE GAPS",
        "",
        "- Canonical control mới có 8 documents, trong khi summary có 100 documents.",
        "- E14b dùng 50 summary documents/dataset và 5 entropy bins; chưa có nhiều seed hoặc paired task-matched samples.",
        "- E15 mới 100 steps và một seed; chưa biết longer adaptation có cải thiện hay chỉ overfit.",
        "- Chưa có quality/ROUGE hoặc end-to-end tokens/s benchmark cho adapted checkpoint; E15 chỉ khóa candidate-lattice metrics.",
        "- Không được suy ra từ E14 rằng marginal quality là nguyên nhân duy nhất; decomposition chỉ so với canonical coherence counterfactual.",
        "",
        "### RECOMMENDED NEXT",
        "",
        "Thực nghiệm duy nhất nên làm tiếp trước khi thiết kế objective mới là **E14c canonical-expansion replication**: mở canonical lên ít nhất 50–100 documents/task-regime, giữ nguyên context 1K, block 16, Top-16 và entropy protocol; sau đó bootstrap theo document. Mục tiêu là nâng E14/E14b từ bounded diagnostic lên R6 evidence đủ mạnh. Chỉ nếu marginal/joint decomposition vẫn ổn định sau E14c mới xem xét adaptation budget lớn hơn.",
        "",
        "> Verified through R7 for the bounded E14/E14b diagnostic and R5 for the E15 short pilot. Not yet verified by a full-budget adaptation study or a causal training objective.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(Path(args.root))
    Path(args.output).write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
