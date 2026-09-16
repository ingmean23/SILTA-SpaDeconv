from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


REQUIRED_PREFIXES = (
    "branch1.",
    "branch2_spatial.",
    "h_projector.",
    "g_projector.",
    "cooperative_decon.",
)


def load_state_dict(path: Path | str) -> Mapping[str, torch.Tensor]:
    path = Path(path)
    try:
        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("SILTA checkpoint must be a non-empty state dictionary")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("SILTA checkpoint contains a non-string parameter key")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError("SILTA checkpoint must contain tensors only")
    missing = [
        prefix for prefix in REQUIRED_PREFIXES
        if not any(key.startswith(prefix) for key in payload)
    ]
    if missing:
        raise ValueError(f"Checkpoint is missing SILTA parameter groups: {missing}")
    return payload


def inspect(path: Path | str) -> dict[str, Any]:
    state = load_state_dict(path)
    non_finite = [
        name for name, value in state.items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(f"Checkpoint contains non-finite tensors: {non_finite[:5]}")
    return {
        "tensor_count": len(state),
        "state_element_count": int(sum(value.numel() for value in state.values())),
        "floating_state_element_count": int(sum(
            value.numel() for value in state.values() if torch.is_floating_point(value)
        )),
        "required_prefixes": list(REQUIRED_PREFIXES),
        "non_finite_tensor_count": 0,
    }
