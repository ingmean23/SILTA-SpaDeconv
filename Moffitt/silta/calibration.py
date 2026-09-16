from __future__ import annotations

import torch
from torch import nn


class TypeLogitCalibrator(nn.Module):
    """Frozen per-type and broad-group logit calibration used at inference."""

    def __init__(self, num_cell_types: int, group_index: torch.Tensor, mode: str) -> None:
        super().__init__()
        self.num_cell_types = int(num_cell_types)
        self.mode = str(mode).lower()
        self.hierarchical = self.mode.startswith("hierarchical")
        self.uses_group_temperature = self.mode.endswith("group_temperature")
        group_index = torch.as_tensor(group_index, dtype=torch.long)
        if group_index.shape != (self.num_cell_types,):
            raise ValueError("group_index must contain one entry per cell type")
        self.num_groups = int(group_index.max().item()) + 1
        self.register_buffer("group_index", group_index)
        self.type_bias = nn.Parameter(torch.zeros(self.num_cell_types))
        if self.hierarchical:
            self.group_bias = nn.Parameter(torch.zeros(self.num_groups))
        else:
            self.register_parameter("group_bias", None)
        if self.uses_group_temperature:
            self.group_temperature_raw = nn.Parameter(torch.zeros(self.num_groups))
        else:
            self.register_parameter("group_temperature_raw", None)

    def effective_bias(self) -> torch.Tensor:
        if not self.hierarchical:
            return self.type_bias
        centered = self.type_bias.clone()
        for group in range(self.num_groups):
            mask = self.group_index == group
            centered[mask] = centered[mask] - centered[mask].mean()
        return self.group_bias[self.group_index] + centered

    def effective_temperature(self) -> torch.Tensor:
        if not self.uses_group_temperature:
            return torch.ones_like(self.type_bias)
        values = torch.exp(0.5 * torch.tanh(self.group_temperature_raw))
        return values[self.group_index]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.effective_temperature() + self.effective_bias()


class ScaledCalibrator(nn.Module):
    """Bounded interpolation between raw and calibrated logits."""

    def __init__(self, base: nn.Module, strength: float) -> None:
        super().__init__()
        self.base = base
        self.strength = float(strength)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        corrected = self.base(logits)
        return logits + self.strength * (corrected - logits)


def build_calibrator(metadata: dict) -> nn.Module:
    base = TypeLogitCalibrator(
        int(metadata["num_cell_types"]),
        torch.tensor(metadata["group_index"], dtype=torch.long),
        str(metadata["mode"]),
    )
    if metadata["class"] == "ScaledCalibrator":
        calibrator: nn.Module = ScaledCalibrator(base, float(metadata["strength"]))
    elif metadata["class"] == "TypeLogitCalibrator":
        calibrator = base
    else:
        raise ValueError(f"Unsupported calibrator class: {metadata['class']}")
    calibrator.load_state_dict(metadata["state_dict"], strict=True)
    return calibrator.eval()
