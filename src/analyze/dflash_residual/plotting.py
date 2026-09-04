"""Optional matplotlib plots for P2/P4 artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _matplotlib():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/dflash_residual_matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_coverage_heatmap(metrics: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Render Recall@K context×depth heatmap when matplotlib is available."""

    plt = _matplotlib()
    if plt is None:
        return {"status": "unavailable", "reason": "matplotlib_not_available"}
    contexts = list(metrics.get("contexts", []))
    positions = [int(value) for value in metrics.get("positions", [])]
    heatmap = metrics.get("heatmap", {})
    if not contexts or not positions:
        return {"status": "unavailable", "reason": "empty_heatmap"}
    values = [[heatmap.get(context, {}).get(str(position)) for position in positions] for context in contexts]
    if any(value is None for row in values for value in row):
        return {"status": "unavailable", "reason": "sparse_heatmap"}
    figure, axis = plt.subplots(figsize=(max(5.0, len(positions) * 0.45), max(3.0, len(contexts) * 0.45)))
    image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(positions)), [str(position) for position in positions])
    axis.set_yticks(range(len(contexts)), contexts)
    axis.set_xlabel("Vị trí draft j")
    axis.set_ylabel("Context bin")
    axis.set_title(f"Recall@{metrics.get('recall_k', 16)}")
    figure.colorbar(image, ax=axis, label="Recall")
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {"status": "ok", "path": str(output)}


def plot_recovery_by_context(metrics: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Render DFlash2 oracle-headroom recovery as context grows."""

    plt = _matplotlib()
    values = metrics.get("rho_by_context_bin", {})
    if plt is None:
        return {"status": "unavailable", "reason": "matplotlib_not_available"}
    if len(values) < 2:
        return {"status": "unavailable", "reason": "insufficient_context_bins_with_rho"}
    contexts = list(values)
    figure, axis = plt.subplots(figsize=(max(5.0, len(contexts) * 0.8), 3.5))
    axis.plot(contexts, [values[context] for context in contexts], marker="o", color="#0072B2")
    axis.set_xlabel("Context bin")
    axis.set_ylabel("rho_D2")
    axis.set_ylim(bottom=min(0.0, min(values.values()) - 0.05))
    axis.set_title("DFlash2 recovery of Top-16 headroom")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {"status": "ok", "path": str(output)}
