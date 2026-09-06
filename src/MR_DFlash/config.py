"""Run-config cho quy trình train DFlash (self-contained, offline).

Đây là bản rút gọn, tự đóng gói của typed run config trong SpecForge
(``specforge/config/schema.py``), chỉ giữ các trường phục vụ train DFlash
offline. Các giá trị mặc định khớp legacy DFlash cũ:

- ``learning_rate = 6e-4``, ``num_epochs = 6``, ``warmup_ratio = 0.04``,
  ``max_grad_norm = 1.0``, ``max_length = 3072`` (xem
  ``examples/configs/qwen3-8b-dflash-disaggregated.yaml`` của SpecForge).
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import List, Optional

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
VALID_LAYER_TYPES = {FULL_ATTENTION, SLIDING_ATTENTION}
VALID_LOSS_TYPES = {"dflash", "dpace", "dpace-cumulative-confidence-only",
                    "dpace-continuation-value-only"}


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> List[int]:
    """Phân bố id các layer của target được dùng làm ``target_layer_ids``.

    Port trực tiếp từ ``specforge/modeling/draft/dflash.py``: với 1 draft
    layer lấy layer giữa của target; với nhiều layer trải đều trong khoảng
    ``[1, num_target_layers - 3]``.
    """
    if num_target_layers < 1:
        raise ValueError(f"num_target_layers phải >= 1, got {num_target_layers}")
    if num_draft_layers < 1:
        raise ValueError(f"num_draft_layers phải >= 1, got {num_draft_layers}")
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    if span < num_draft_layers - 1:
        raise ValueError(
            f"num_target_layers={num_target_layers} quá nhỏ cho "
            f"{num_draft_layers} draft layers"
        )
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def resolve_feature_layer_ids(model_config: object) -> Optional[List[int]]:
    """Resolve feature layers, ưu tiên field mới và giữ alias legacy."""
    feature = getattr(model_config, "feature_layer_ids", None)
    legacy = getattr(model_config, "target_layer_ids", None)
    if feature is not None and legacy is not None and list(feature) != list(legacy):
        raise ValueError(
            "target_layer_ids và feature_layer_ids phải giống nhau; "
            "dùng feature_layer_ids cho config mới"
        )
    resolved = feature if feature is not None else legacy
    return None if resolved is None else [int(value) for value in resolved]


def resolve_draft_init_layer_ids(
    model_config: object,
    *,
    num_target_layers: int,
) -> List[int]:
    """Resolve layer copy cho draft, độc lập với feature layer layout."""
    explicit = getattr(model_config, "draft_init_layer_ids", None)
    num_draft_layers = int(getattr(model_config, "draft_num_hidden_layers"))
    values = (
        [int(value) for value in explicit]
        if explicit is not None
        else build_target_layer_ids(num_target_layers, num_draft_layers)
    )
    if len(values) != num_draft_layers:
        raise ValueError(
            "draft_init_layer_ids phải có đúng số draft layer: "
            f"{len(values)} != {num_draft_layers}"
        )
    if any(value < 0 or value >= num_target_layers for value in values):
        raise ValueError(
            "draft_init_layer_ids chứa layer ngoài target: "
            f"values={values}, num_target_layers={num_target_layers}"
        )
    return values


def resolve_dflash_attention_layout(
    layer_types: List[str],
    num_hidden_layers: int,
    sliding_window: Optional[int],
) -> None:
    """Validate cấu hình attention layout (full/sliding) cho từng draft layer."""
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "layer_types phải có đúng num_hidden_layers phần tử: "
            f"{len(layer_types)} != {num_hidden_layers}"
        )
    invalid = set(layer_types) - VALID_LAYER_TYPES
    if invalid:
        raise ValueError(
            "layer_types chỉ hỗ trợ full_attention/sliding_attention, "
            f"got {sorted(invalid)}"
        )
    if SLIDING_ATTENTION in layer_types:
        if sliding_window is None or sliding_window <= 0:
            raise ValueError(
                "sliding_attention yêu cầu sliding_window dương"
            )


@dataclass
class ModelConfig:
    """Cấu hình model target + draft DFlash/MR-DFlash."""

    #: Target model (HF path) — nguồn feature/embedding/lm_head/labels.
    target_model_path: str = "Qwen/Qwen3-8B"
    #: dflash giữ baseline; mr_dflash bật memory đa phân giải.
    architecture: str = "dflash"
    #: Số layer của draft model.
    draft_num_hidden_layers: int = 1
    #: Số layer của target (tự nạp từ target config nếu để None).
    num_target_layers: Optional[int] = None
    #: Legacy alias cho feature_layer_ids; giữ để đọc config cũ.
    target_layer_ids: Optional[List[int]] = None
    #: Các layer target được capture làm context feature (concat).
    feature_layer_ids: Optional[List[int]] = None
    #: Các layer target dùng để copy weight vào draft khi init.
    #: None = tự sinh layout DFlash theo số draft layer.
    draft_init_layer_ids: Optional[List[int]] = None
    #: Độ dài 1 block dự đoán song song.
    block_size: int = 16
    #: Loại attention từng draft layer: full_attention | sliding_attention.
    layer_types: Optional[List[str]] = None
    #: Cửa sổ trượt khi có layer sliding_attention.
    sliding_window: Optional[int] = None
    #: Token id dùng làm "noise" lấp đầy block (None → tự resolve).
    mask_token_id: Optional[int] = None
    #: Warm-start weights-only từ checkpoint draft (không khôi phục optimizer).
    draft_checkpoint_path: Optional[str] = None
    #: [MR hook] Khởi tạo weight draft layer thứ i từ target layer
    #: target_layer_ids[i] (DFlash gốc thường copy rồi fine-tune). SpecForge
    #: mặc định random init trừ khi warm-start.
    init_draft_from_target: bool = False
    #: dtype huấn luyện.
    torch_dtype: str = "bfloat16"
    #: Số stage target attention của MR-DFlash (HCA rồi CSA).
    mr_num_stages: int = 2
    #: Tỉ lệ nén token cho memory HCA.
    hca_compression_ratio: int = 128
    #: Tỉ lệ nén token cho memory CSA.
    csa_compression_ratio: int = 4
    #: Số token target raw luôn giữ ở local memory.
    memory_local_window: int = 128
    #: Số CSA slot tối đa mỗi draft query.
    csa_top_k: int = 64
    #: Chiều projection Q/K indexer; null = hidden_size.
    indexer_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.architecture not in {"dflash", "mr_dflash"}:
            raise ValueError(
                f"architecture chỉ hỗ trợ dflash|mr_dflash, got {self.architecture}"
            )
        if self.block_size < 2:
            raise ValueError(f"block_size phải >= 2, got {self.block_size}")
        if self.draft_num_hidden_layers < 1:
            raise ValueError(
                f"draft_num_hidden_layers phải >= 1, got {self.draft_num_hidden_layers}"
            )
        if self.torch_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"torch_dtype không hợp lệ: {self.torch_dtype}")
        if (
            self.target_layer_ids is not None
            and self.feature_layer_ids is not None
            and list(self.target_layer_ids) != list(self.feature_layer_ids)
        ):
            raise ValueError(
                "target_layer_ids và feature_layer_ids phải giống nhau; "
                "dùng feature_layer_ids cho config mới"
            )
        for name in ("target_layer_ids", "feature_layer_ids", "draft_init_layer_ids"):
            values = getattr(self, name)
            if values is not None and (
                not values or any(int(value) < 0 for value in values)
            ):
                raise ValueError(f"{name} phải là list layer id không âm và không rỗng")
        if self.mr_num_stages < 2:
            raise ValueError("mr_num_stages phải >= 2 (HCA + CSA)")
        for name in (
            "hca_compression_ratio",
            "csa_compression_ratio",
            "memory_local_window",
            "csa_top_k",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} phải >= 1")
        if self.indexer_dim is not None and self.indexer_dim < 1:
            raise ValueError("indexer_dim phải dương hoặc null")


@dataclass
class DataConfig:
    """Dữ liệu huấn luyện: jsonl hội thoại → capture → feature offline."""

    #: jsonl huấn luyện (conversation hoặc pre-formatted text).
    train_data_path: str = ""
    #: jsonl eval (optional).
    eval_data_path: Optional[str] = ""
    #: Thư mục feature offline đã capture (đầu ra của capture / đầu vào train).
    hidden_states_path: Optional[str] = None
    #: Giới hạn mẫu dùng (None = tất cả).
    num_samples: Optional[int] = None
    #: Độ dài chuỗi tối đa sau truncate (legacy DFlash = 3072).
    max_length: int = 3072
    #: Chat template dùng để render + xác định span assistant cho loss_mask.
    chat_template: str = "qwen"
    #: True nếu mỗi dòng jsonl đã là text được template sẵn.
    is_preformatted: bool = False
    cache_dir: str = "./cache"
    #: Số worker dùng khi build dataset.
    build_dataset_num_proc: int = 8
    #: DataLoader worker/prefetch khi chuyển feature lên GPU.
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError("data.num_workers phải >= 0")
        if self.prefetch_factor < 1:
            raise ValueError("data.prefetch_factor phải >= 1")
    #: Cache feature validation set; None = tự sinh dưới output_dir.
    eval_hidden_states_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Các siêu tham số + thuật toán DFlash (legacy defaults)."""

    strategy: str = "dflash"
    num_epochs: int = 6
    max_steps: Optional[int] = 10000
    batch_size: int = 4
    accumulation_steps: int = 1
    learning_rate: float = 6e-4
    warmup_ratio: float = 0.04
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    num_anchors: int = 512
    #: None = không dùng positional decay (SpecForge: loss_decay_gamma).
    loss_decay_gamma: Optional[float] = 7.0
    objective_chunk_blocks: int = 128
    #: sdpa (CPU-safe) | flex (cần flex_attention).
    attention_backend: str = "sdpa"
    #: dflash | dpace | dpace-cumulative-confidence-only | ...
    loss_type: str = "dflash"
    save_interval: int = 1000
    log_interval: int = 10
    #: 0 = evaluate cuối run; >0 evaluate định kỳ theo optimizer step.
    eval_interval: int = 0
    seed: int = 42
    #: FSDP không bắt buộc; nếu >1 dùng DDP đơn giản qua torch.distributed.
    dp_world_size: int = 1

    def __post_init__(self) -> None:
        if self.strategy not in {"dflash", "mr_dflash"}:
            raise ValueError(
                f"strategy chỉ hỗ trợ dflash|mr_dflash, got {self.strategy}"
            )
        if self.attention_backend not in {"sdpa", "flex"}:
            raise ValueError(
                f"attention_backend chỉ hỗ trợ sdpa|flex, got {self.attention_backend}"
            )
        if self.loss_type not in VALID_LOSS_TYPES:
            raise ValueError(f"loss_type không hợp lệ: {self.loss_type}")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio phải thuộc [0,1], got {self.warmup_ratio}")
        if self.eval_interval < 0:
            raise ValueError("eval_interval phải >= 0")


@dataclass
class RunConfig:
    """Config gộp toàn bộ một run train DFlash offline."""

    run_id: str = "mr-dflash"
    output_dir: str = "outputs/mr-dflash"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def dump_yaml(self) -> str:
        """Xuất config ra YAML text để lưu provenance cùng checkpoint."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("cần pyyaml để dump config: pip install pyyaml") from exc

        def _plain(value):
            if is_dataclass(value) and not isinstance(value, type):
                return {k: _plain(v) for k, v in value.__dict__.items()}
            if isinstance(value, dict):
                return {str(k): _plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_plain(v) for v in value]
            return value

        return yaml.safe_dump(_plain(self), sort_keys=False, allow_unicode=True)


__all__ = [
    "FULL_ATTENTION",
    "SLIDING_ATTENTION",
    "VALID_LAYER_TYPES",
    "VALID_LOSS_TYPES",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "RunConfig",
    "build_target_layer_ids",
    "resolve_feature_layer_ids",
    "resolve_draft_init_layer_ids",
    "resolve_dflash_attention_layout",
]
