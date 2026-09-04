"""Compatibility helpers for reading RoPE parameters from Transformers 5 configs.

Transformers 5 no longer exposes ``config.rope_theta`` as a plain attribute:
Llama/Qwen RoPE values live under ``rope_parameters`` (mirrored via
``rope_scaling``), and legacy configs keep ``rope_theta`` only in their raw
JSON.  The vendored LongSpec modeling code was written against the legacy
attribute; these helpers read both layouts and fall back to ``to_dict()``.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping


def _find_numeric(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, Real) and not isinstance(value, bool):
            return float(value)
    for value in mapping.values():
        if isinstance(value, Mapping):
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    return None


def config_rope_theta(config: Any, default: float = 10000.0) -> float:
    """Read ``rope_theta`` from a legacy or a Transformers 5 config layout."""

    value = getattr(config, "rope_theta", None)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)

    for attr in ("rope_parameters", "rope_scaling"):
        value = getattr(config, attr, None)
        if isinstance(value, Mapping):
            found = _find_numeric(value, ("rope_theta", "base"))
            if found is not None:
                return found

    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
        if isinstance(raw, Mapping):
            found = _find_numeric(raw, ("rope_theta", "base"))
            if found is not None:
                return found

    return float(default)
