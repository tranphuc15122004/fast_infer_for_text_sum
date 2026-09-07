# E19–E21 gain-first causal screening

## Mục tiêu

Kiểm tra ba bottleneck có headroom lớn nhất sau E16–E17 mà chưa huấn luyện
method mới:

1. H19: deployment-state mismatch;
2. H20: target miss chỉ nằm ngay ngoài Top-16;
3. H21: conditioning/mask topology khi train khác topology cần cho block
   speculative inference.

Tất cả screening chạy trên Qwen3-4B + Qwen3-4B-DFlash-b16, Top-K tối đa 128,
block native 16, target greedy, GPU T4 bằng Conda ngoài. B200 không được dùng.

## Thiết kế được khóa

### E19 — Fixed-state paired audit

Mỗi document tạo cùng một bank các prefix lengths, mặc định `0,4,8`. Với
mỗi prefix length, dựng đúng hai input IDs có cùng source và cùng độ dài prefix:

- `on_policy`: prefix token từ target greedy trajectory;
- `reference`: prefix token từ reference summary.

Mỗi cặp chạy đúng một DFlash block. So sánh `R16`, `J16`, `MAT_O16`, entropy
và bootstrap theo document 500 lần. Không dùng rollout nên số block giữa hai
condition được giữ cố định.

Gate H19: reference-minus-on-policy `MAT_O16` relative >20% và bootstrap CI
document dương trên ít nhất hai dataset; hoặc chênh lệch R16 positions 3–8
trên 10 percentage points ở ít nhất hai dataset.

### E20 — Full-vocabulary target rank

Không lưu full vocabulary logits. Tại mỗi row tính trực tiếp:

$$
r_t=1+|\{v:\ell_D(v)>\ell_D(y_t^T)\}|.
$$

Lưu một scalar `draft_target_rank`. Offline tính `R@K` và `MAT_O_K` cho
`K={16,32,64,128}` trên các fixed on-policy states với `reveal_count=0`.

Gate H20: `MAT_O32/MAT_O16 - 1 >20%` trên ít nhất hai dataset. Kết quả này
chỉ là diagnostic; không tự động mở tree/wider-candidate method.

### E21-A — Training topology audit

Đọc loss mask thật của training samples và mô phỏng đúng sampler anchor:

- valid anchor: `loss_mask[t] & loss_mask[t+1]`;
- số anchor và phân bố vị trí anchor;
- xác suất có nhãn liên tiếp tại block offset `j`;
- độ sâu nhãn liên tiếp sau anchor;
- số target token tương lai được reveal trong input block train.

Audit này không gọi là gradient attribution. Nó kiểm tra topology, không chỉ
tổng exposure.

### E21-B — Causal-reveal oracle

Trên cùng fixed on-policy state, chạy các condition `r={0,1,2,4}`. `r` token
đầu của candidate block được clamp bằng target-greedy tokens; chúng không được
tính vào conditional accepted-prefix gain. Tính conditional `R16` và
`cMAT_O16` bắt đầu từ position `r+1`.

Gate H21: một trong `r=1,2` tạo relative conditional `cMAT_O16` gain ≥30%
trên ít nhất hai dataset. Chỉ khi gate này đạt mới cân nhắc E22 training.

## Biến và controls

- Independent: state mode, candidate depth K, reveal count r.
- Dependent: `R@K`, `J@K`, `MAT_O_K`, conditional `cMAT_O16`.
- Controls: target/drafter/tokenizer, source document, prefix length, block
  size, dtype, attention backend, seed, GPU.
- Primary comparison: document-level paired/bootstrap, không dùng chênh lệch
  số rollout blocks.

## Validation scope

- R0/R1: deterministic unit tests, schema validation, rank and suffix oracle
  fixtures.
- R5: real T4 fixed-state traces trên 30 docs mỗi dataset nếu dữ liệu hợp lệ.
- Không chạy E22 nếu H19/H21 chỉ pass một dataset hoặc CI chứa zero.

## Expected decisions

- H19 pass: chỉ mở state-alignment intervention.
- H20 pass nhưng H19/H21 fail: ghi nhận shallow-tail diagnostic, chưa train.
- H21 pass: ưu tiên topology-aligned training, không thêm inference head.
- Không gate nào pass: đóng candidate-generation proposal branch trong scope
  Qwen3-4B/T4.
