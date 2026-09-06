# Thiết kế E14–E15: Task-Induced Joint Candidate-Lattice Degradation

## Câu hỏi nghiên cứu

Vì sao ở cùng context 1K, DFlash trên summarization có Top-16 oracle MAT thấp hơn canonical? Cần tách candidate quality theo từng position khỏi khả năng duy trì candidate-hit liên tục trong một block.

## Giả thuyết

- **H14:** suy giảm oracle lattice gồm hai phần: marginal Top-16 recall và joint prefix coherence.
- **H14b:** nếu gap joint còn sau khi match target entropy, đây là task-specific structural/training mismatch chứ không chỉ intrinsic target difficulty.
- **H15:** adaptation tối thiểu trên trajectory summarization, giữ nguyên kiến trúc và DFlash CE objective, sẽ cải thiện candidate lattice nếu distribution mismatch là nguyên nhân.

## Biến và metric

Giữ Top-16, native block 16, context cap 1024, greedy target verification. Dùng `R_j`, `J_j`, `C_j=J_j/prod(R_1..R_j)`, MAT_O16 và counterfactual decomposition. E14b dùng target full-vocabulary entropy và entropy-standardized `J_j`/MAT. E15 dùng R@1/R@16, J, C, MAT_D và MAT_O16.

## Cấu hình validation

E14 offline trên trace E11 hợp lệ; E14b thu thêm 8 canonical và 50 mẫu mỗi summarization dataset bằng GPU T4 external conda env, dtype bfloat16. E15 chỉ chạy sau khi E14b còn gap; giữ baseline DFlash, same block/target/checkpoint, document-disjoint held-out dataset. Không mở source/KV/selector branch.

## Gate

- Marginal component >70%: ưu tiên task-distribution mismatch.
- Joint component ≥30% và entropy-standardized gap còn rõ: mở candidate-generation/training investigation.
- E15 oracle recovery <10%: đóng adaptation branch; 10–30% là evidence distribution shift; >30% mới đáng phát triển.

