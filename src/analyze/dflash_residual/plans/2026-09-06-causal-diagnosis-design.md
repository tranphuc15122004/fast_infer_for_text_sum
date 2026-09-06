# Thiết kế E16–E18: Causal diagnosis cho prefix-critical candidate valley

## Câu hỏi nghiên cứu

Sau E14–E15, cần phân biệt hai mechanism còn lại giải thích suy giảm Top-16 oracle của DFlash trên summarization:

1. **State-distribution mismatch:** trạng thái dùng khi triển khai khác trạng thái mà drafter đã học.
2. **Training-utility mismatch:** objective hiện tại phân bổ năng lực học khác với giá trị của candidate ở các vị trí prefix-critical.

Không mở lại source, KV, selector, context sweep hay joint-coherence proposal.

## Thiết kế bounded

### E16 — Replication gate

Mở canonical GSM8K từ 8 lên 50–100 documents nếu nguồn mẫu và checkpoint local cho phép. Giữ nguyên context cap 1024, native block 16, Top-16, greedy target verification và max depth 15. Tính `R16(j)`, `J16(j)`, `MAT_O16`, bootstrap theo document. So sánh vùng vị trí 3–8 với CNN/DM, GovReport và Multi-News.

**Gate:** prefix-critical valley phải tái lập trên canonical mở rộng và ít nhất hai workload summarization. Nếu không, dừng E17/E18.

### E17-A — State-distribution audit

Trên cùng document và cùng target/draft checkpoint, tạo hai chế độ state:

- **reference/teacher-forced:** prefix output được lấy từ reference summary khi xây verifier state;
- **on-policy/deployment:** prefix được lấy từ target-greedy committed output như collector hiện tại.

Giữ candidate-generation/verification protocol giống nhau. So sánh `R16(j)`, `J16(j)`, `MAT_O16` và first-rejection hazard ở vị trí 3–8. Kết quả chỉ được gọi là state evidence nếu protocol khác nhau duy nhất ở prefix state và effect lặp lại ở ít nhất hai dataset.

### E17-B — Training-utility audit

Không sửa objective trước khi đo. Audit:

- theoretical DFlash positional weight;
- anchor/supervision exposure theo block position;
- per-position loss và accuracy trên batch mẫu;
- verifier utility `U_j = P(MAT_O16 >= j)` từ trace.

Hiện code dùng `loss_decay_gamma=7`, với weight `exp(-(k-1)/7)` sau khi bỏ anchor position. Audit phải báo cáo effective weight sau `loss_mask`, không chỉ công thức.

### E18 — Minimal causal intervention

Chỉ chạy nhánh có evidence từ E17:

- state thắng: task-matched on-policy data, original DFlash loss;
- utility thắng: cùng data, positional weighting lấy từ measured utility;
- cả hai thắng: đúng ba variant `on-policy`, `utility`, `on-policy+utility`.

Giữ nguyên target, kiến trúc, block size, Top-16 và inference collector. Primary là held-out `MAT_O16`; secondary là `MAT_D`, `R16(3:8)` và runtime overhead.

## Gates

- `<10%` tăng `MAT_O16`: đóng intervention branch.
- `10–20%`: promising, cần replicate.
- `>20%`: mở proposal/literature novelty check.
- Không được gọi là proposal nếu chỉ tăng Recall@1/R16 mà `MAT_O16` không tăng.

## Validation scope

- R0–R2: static/schema/unit checks và một-batch smoke.
- E16/E17: bounded diagnostic, document bootstrap; không yêu cầu full training.
- E18: T4 short pilot, held-out Multi-News, một seed; chỉ là R5 cho đến khi có full-budget replication.

## Runtime và artifact

- Runtime: `/home/tuantb/miniconda3/envs/myenv/bin/python3.11`, Tesla T4.
- Không dùng môi trường B200.
- Artifact root: `outputs/dflash_residual/2026-09-06_causal_diagnosis/`.
- Report cuối: `causal_diagnosis_report.md`.
