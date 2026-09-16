"""Optional DACG readout adaptation and cell-type logit calibration."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn


CALIBRATION_MODES = {
    "bias",
    "group_temperature",
    "hierarchical_bias",
    "hierarchical_group_temperature",
}


class BoundedReadoutAdapter(nn.Module):
    """A zero-initialized, g-dependent correction to existing logits."""

    def __init__(
        self,
        fusion_dim: int,
        num_cell_types: int,
        alpha: float,
        interaction_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0 or num_cell_types <= 0:
            raise ValueError("fusion_dim and num_cell_types must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.fusion_dim = int(fusion_dim)
        self.num_cell_types = int(num_cell_types)
        self.alpha = float(alpha)
        self.interaction_weight = float(interaction_weight)

        self.h_norm = nn.LayerNorm(self.fusion_dim, elementwise_affine=False)
        self.g_projection = nn.Linear(
            self.fusion_dim, self.fusion_dim, bias=False)
        self.g_norm = nn.LayerNorm(
            self.fusion_dim, elementwise_affine=False)
        self.interaction_projection = nn.Linear(
            self.fusion_dim, self.fusion_dim, bias=False)
        self.interaction_norm = nn.LayerNorm(
            self.fusion_dim, elementwise_affine=False)
        self.output_projection = nn.Linear(
            self.fusion_dim, self.num_cell_types, bias=False)

        nn.init.eye_(self.g_projection.weight)
        nn.init.xavier_uniform_(self.interaction_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        base_logits: torch.Tensor,
        h0: torch.Tensor,
        g0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if h0.shape != g0.shape:
            raise ValueError(
                f"h0 and g0 must have identical shapes, got {h0.shape} and {g0.shape}")
        if h0.shape[-1] != self.fusion_dim:
            raise ValueError(
                f"Expected cooperative feature dim {self.fusion_dim}, got {h0.shape[-1]}")
        if base_logits.shape[:-1] != h0.shape[:-1]:
            raise ValueError("base logits and cooperative features must share batch/spot axes")
        if base_logits.shape[-1] != self.num_cell_types:
            raise ValueError(
                f"Expected {self.num_cell_types} output types, got {base_logits.shape[-1]}")

        g_value = self.g_norm(self.g_projection(g0))
        interaction = self.interaction_norm(
            self.interaction_projection(self.h_norm(h0) * g_value))
        combined = g_value + self.interaction_weight * interaction
        delta_raw = self.output_projection(combined)
        delta = self.alpha * torch.tanh(delta_raw)
        return {
            "logits": base_logits + delta,
            "delta": delta,
            "delta_raw": delta_raw,
            "g_value": g_value,
            "interaction": interaction,
        }

    @torch.no_grad()
    def diagnostics(
        self,
        base_logits: torch.Tensor,
        output: Mapping[str, torch.Tensor],
    ) -> Dict[str, float]:
        eps = torch.finfo(base_logits.dtype).eps
        delta = output["delta"]
        g_value = output["g_value"]
        interaction = output["interaction"]
        return {
            "adapter_alpha": self.alpha,
            "adapter_delta_abs_mean": float(delta.abs().mean().cpu()),
            "adapter_delta_max_abs": float(delta.abs().max().cpu()),
            "adapter_delta_to_base_std": float(
                (delta.std(unbiased=False) /
                 (base_logits.std(unbiased=False) + eps)).cpu()),
            "adapter_interaction_to_g_std": float(
                (interaction.std(unbiased=False) /
                 (g_value.std(unbiased=False) + eps)).cpu()),
        }


class TypeLogitCalibrator(nn.Module):
    """A bounded identity-initialized per-type and broad-group calibrator."""

    def __init__(
        self,
        num_cell_types: int,
        group_index: torch.Tensor,
        mode: str,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"Unknown calibration mode: {mode}")
        group_index = torch.as_tensor(group_index, dtype=torch.long)
        if group_index.ndim != 1 or len(group_index) != int(num_cell_types):
            raise ValueError("group_index must contain one group id per cell type")
        if len(group_index) == 0 or int(group_index.min()) < 0:
            raise ValueError("group indices must be non-negative")

        self.num_cell_types = int(num_cell_types)
        self.num_groups = int(group_index.max().item()) + 1
        self.mode = mode
        self.hierarchical = mode.startswith("hierarchical")
        self.uses_group_temperature = mode.endswith("group_temperature")
        self.register_buffer("group_index", group_index)

        self.type_bias = nn.Parameter(torch.zeros(self.num_cell_types))
        if self.hierarchical:
            self.group_bias = nn.Parameter(torch.zeros(self.num_groups))
        else:
            self.register_parameter("group_bias", None)
        if self.uses_group_temperature:
            self.group_temperature_raw = nn.Parameter(
                torch.zeros(self.num_groups))
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
        group_temperature = torch.exp(
            0.5 * torch.tanh(self.group_temperature_raw))
        return group_temperature[self.group_index]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return (
            logits / self.effective_temperature()
            + self.effective_bias()
        )

    def identity_loss(self) -> torch.Tensor:
        loss = self.type_bias.pow(2).mean()
        if self.group_bias is not None:
            loss = loss + self.group_bias.pow(2).mean()
        if self.group_temperature_raw is not None:
            loss = loss + self.group_temperature_raw.pow(2).mean()
        return loss

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        bias = self.effective_bias()
        temperature = self.effective_temperature()
        raw_boundary = 0.0
        if self.group_temperature_raw is not None:
            raw_boundary = float(
                torch.tanh(self.group_temperature_raw).abs().max().cpu())
        return {
            "calibration_bias_abs_mean": float(bias.abs().mean().cpu()),
            "calibration_bias_max_abs": float(bias.abs().max().cpu()),
            "calibration_temperature_mean": float(temperature.mean().cpu()),
            "calibration_temperature_min": float(temperature.min().cpu()),
            "calibration_temperature_max": float(temperature.max().cpu()),
            "calibration_temperature_boundary": raw_boundary,
        }


def adapted_logits(
    output: Mapping[str, torch.Tensor],
    adapter: Optional[BoundedReadoutAdapter] = None,
    calibrator: Optional[TypeLogitCalibrator] = None,
) -> Dict[str, torch.Tensor]:
    """Apply optional components to a normal DACG forward output."""
    base_logits = output["decon_logits"]
    if adapter is None:
        adapter_output = {
            "logits": base_logits,
            "delta": torch.zeros_like(base_logits),
            "delta_raw": torch.zeros_like(base_logits),
            "g_value": output["coop_g0"],
            "interaction": torch.zeros_like(output["coop_g0"]),
        }
    else:
        adapter_output = adapter(
            base_logits, output["coop_h0"], output["coop_g0"])
    logits = adapter_output["logits"]
    if calibrator is not None:
        logits = calibrator(logits)
    return {**adapter_output, "logits": logits, "base_logits": base_logits}
