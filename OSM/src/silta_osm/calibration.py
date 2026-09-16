from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TypeLogitCalibrator(nn.Module):
    """C01 broad-group temperature and per-type bias calibrator."""

    def __init__(self, count: int, group_index: torch.Tensor, mode: str) -> None:
        super().__init__()
        if mode != "group_temperature":
            raise ValueError(f"Unsupported C01 mode in this release: {mode}")
        group_index = torch.as_tensor(group_index, dtype=torch.long)
        if group_index.shape != (int(count),):
            raise ValueError("group_index must contain one entry per cell type")
        self.register_buffer("group_index", group_index)
        self.type_bias = nn.Parameter(torch.zeros(int(count)))
        self.group_temperature_raw = nn.Parameter(
            torch.zeros(int(group_index.max().item()) + 1)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        group_temperature = torch.exp(0.5 * torch.tanh(self.group_temperature_raw))
        temperature = group_temperature[self.group_index]
        return logits / temperature + self.type_bias


class GroupCenteredShapeCalibrator(nn.Module):
    """C04 bounded within-group type bias and temperature calibrator."""

    def __init__(
        self,
        count: int,
        group_index: torch.Tensor,
        bias_cap: float,
        temperature_cap: float,
    ) -> None:
        super().__init__()
        group_index = torch.as_tensor(group_index, dtype=torch.long)
        if group_index.shape != (int(count),):
            raise ValueError("group_index must contain one entry per cell type")
        self.register_buffer("group_index", group_index)
        self.bias_cap = float(bias_cap)
        self.temperature_cap = float(temperature_cap)
        self.raw_bias = nn.Parameter(torch.zeros(int(count)))
        self.raw_log_temperature = nn.Parameter(torch.zeros(int(count)))

    def _center(self, value: torch.Tensor) -> torch.Tensor:
        means = torch.stack([
            value[self.group_index == group].mean()
            for group in range(int(self.group_index.max().item()) + 1)
        ])
        return value - means[self.group_index]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        bias = self.bias_cap * self._center(torch.tanh(self.raw_bias))
        log_temperature = self.temperature_cap * self._center(
            torch.tanh(self.raw_log_temperature)
        )
        return (logits + bias) / torch.exp(log_temperature).clamp(0.5, 2.0)


def load_bundle(path: Path, device: str = "cpu") -> tuple[list[str], nn.Module, nn.Module]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "silta_osm_c01_c04_calibration_state"
    ):
        raise ValueError("Unsupported SILTA calibration bundle schema")
    cell_types = list(map(str, payload["cell_types"]))
    group_index = torch.as_tensor(payload["group_index"], dtype=torch.long)
    c01_block: Mapping[str, Any] = payload["c01"]
    c04_block: Mapping[str, Any] = payload["c04"]
    c01_spec: Mapping[str, Any] = c01_block["spec"]
    c04_spec: Mapping[str, Any] = c04_block["spec"]
    c01 = TypeLogitCalibrator(
        len(cell_types), group_index, str(c01_spec["mode"])
    ).to(device)
    c01.load_state_dict(c01_block["states"]["transfer"], strict=True)
    c04 = GroupCenteredShapeCalibrator(
        len(cell_types),
        group_index,
        float(c04_spec["bias_cap"]),
        float(c04_spec["temperature_cap"]),
    ).to(device)
    c04.load_state_dict(c04_block["states"]["transfer"], strict=True)
    return cell_types, c01.eval(), c04.eval()


def apply_bundle(logits: pd.DataFrame, bundle_path: Path) -> pd.DataFrame:
    cell_types, c01, c04 = load_bundle(bundle_path)
    if list(logits.columns.astype(str)) != cell_types:
        raise ValueError("Logit cell-type order does not match the C04 bundle")
    tensor = torch.as_tensor(logits.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        fractions = torch.softmax(c04(c01(tensor)), dim=-1).cpu().numpy()
    if not np.isfinite(fractions).all():
        raise ValueError("Calibration produced non-finite fractions")
    return pd.DataFrame(fractions, index=logits.index, columns=cell_types)


def apply_c01(logits: pd.DataFrame, bundle_path: Path) -> pd.DataFrame:
    """Apply only C01 and return the explicitly pre-C04 fractions."""
    cell_types, c01, _ = load_bundle(bundle_path)
    if list(logits.columns.astype(str)) != cell_types:
        raise ValueError("Logit cell-type order does not match the C04 bundle")
    tensor = torch.as_tensor(logits.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        fractions = torch.softmax(c01(tensor), dim=-1).cpu().numpy()
    if not np.isfinite(fractions).all():
        raise ValueError("C01 produced non-finite fractions")
    return pd.DataFrame(fractions, index=logits.index, columns=cell_types)
