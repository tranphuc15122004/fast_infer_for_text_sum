"""MR-DFlash — bản tự đóng gói (self-contained) của quy trình train DFlash.

Port từ SpecForge (``externals/SpecForge``), dùng làm gốc cho thí nghiệm
MR-DFlash: DFlash-compatible training plus HCA/CSA target memory, draft model
và reference speculative inference. Toàn bộ chạy độc lập với torch
(+ transformers khi nạp target).
"""

__version__ = "0.1.0"
