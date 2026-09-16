"""Isolated E1/E2 dense distance-attention variants for seqFISH B v3."""

from __future__ import annotations

import json
import math
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MODES = {"hard", "dense", "fixed", "learned", "legacy_hybrid"}
_SETTINGS: Dict[str, Any] = {}


def _coordinates(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    numeric = frame.select_dtypes(include=[np.number])
    if len(numeric) < 2 or numeric.shape[1] < 2:
        raise ValueError(f"Locations need >=2 rows and numeric columns: {path}")
    values = numeric.iloc[:, :2].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Locations contain non-finite values")
    return values


def normalized_physical_distance(path: Path) -> torch.Tensor:
    coordinates = _coordinates(path)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    positive = np.where(distance > 0, distance, np.inf)
    nearest = positive.min(axis=1)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not nearest.size:
        raise ValueError("Locations do not contain distinct spots")
    scale = float(np.median(nearest))
    return torch.tensor(distance / scale, dtype=torch.float32)


def _sigmas(heads: int) -> tuple[float, ...]:
    base = (1.0, 2.0, 4.0, 8.0)
    if heads == len(base):
        return base
    if heads < len(base):
        return base[:heads]
    return tuple(np.geomspace(1.0, 8.0, num=heads).tolist())


class DenseDistanceAttention(nn.Module):
    """Multi-head attention with optional Gaussian bias in attention logits."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        distance_kind: str,
        learnable_sigma: bool,
        physical_distance: torch.Tensor | None = None,
        gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        if distance_kind not in {"none", "expression", "physical"}:
            raise ValueError(f"Unknown distance kind: {distance_kind}")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.output_dim // self.num_heads
        self.distance_kind = str(distance_kind)
        self.learnable_sigma = bool(learnable_sigma)
        self.scale = self.head_dim ** -0.5
        self.sigma_min = 0.35
        self.sigma_max = 8.0

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.query = nn.Linear(self.input_dim, self.output_dim)
        self.key = nn.Linear(self.input_dim, self.output_dim)
        self.value = nn.Linear(self.input_dim, self.output_dim)
        self.message_projection = nn.Linear(self.output_dim, self.output_dim)
        self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        values = torch.tensor(_sigmas(self.num_heads), dtype=torch.float32)
        if self.learnable_sigma:
            ratio = ((values - self.sigma_min)
                     / (self.sigma_max - self.sigma_min)).clamp(1e-4, 1 - 1e-4)
            self.sigma_logits = nn.Parameter(torch.logit(ratio))
            self.register_buffer("fixed_sigmas", torch.empty(0))
        else:
            self.register_parameter("sigma_logits", None)
            self.register_buffer("fixed_sigmas", values)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.message_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        if physical_distance is None:
            physical_distance = torch.empty(0)
        self.register_buffer(
            "physical_distance", physical_distance.float(), persistent=False)
        self.last_attention_entropy = float("nan")

    def effective_sigmas(self) -> torch.Tensor:
        if not self.learnable_sigma:
            return self.fixed_sigmas
        return self.sigma_min + (
            self.sigma_max - self.sigma_min) * torch.sigmoid(self.sigma_logits)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        return values.view(
            batch, spots, self.num_heads, self.head_dim).transpose(1, 2)

    def _expression_distance(self, normalized: torch.Tensor) -> torch.Tensor:
        unit = F.normalize(normalized, p=2, dim=-1, eps=1e-8)
        distance = (1.0 - torch.matmul(unit, unit.transpose(-2, -1))).clamp_min(0)
        spots = distance.shape[-1]
        eye = torch.eye(spots, device=distance.device, dtype=torch.bool)
        positive = distance.masked_fill(eye.unsqueeze(0), float("inf"))
        nearest = positive.min(dim=-1).values
        finite = nearest[torch.isfinite(nearest) & (nearest > 1e-8)]
        scale = finite.detach().median() if finite.numel() else distance.new_tensor(1.0)
        return distance / scale.clamp_min(1e-6)

    def _distance(self, normalized: torch.Tensor) -> torch.Tensor | None:
        if self.distance_kind == "none":
            return None
        if self.distance_kind == "expression":
            return self._expression_distance(normalized).unsqueeze(1)
        spots = normalized.shape[-2]
        if self.physical_distance.shape != (spots, spots):
            # Pseudo graphs have no physical geometry and use variable node
            # counts. Their g path is frozen/zeroed, so dense fallback is the
            # only geometry-neutral behavior. Real ST evaluation injects the
            # matching distance matrix before forward.
            return None
        return self.physical_distance.to(
            device=normalized.device, dtype=normalized.dtype).view(
                1, 1, spots, spots)

    def forward(self, h: torch.Tensor, adjacency=None, ex_adj=None) -> torch.Tensor:
        del adjacency, ex_adj
        normalized = self.input_norm(h)
        query = self._heads(self.query(normalized))
        key = self._heads(self.key(normalized))
        value = self._heads(self.value(normalized))
        score = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        distance = self._distance(normalized)
        if distance is not None:
            sigma = self.effective_sigmas().to(
                device=score.device, dtype=score.dtype).view(
                    1, self.num_heads, 1, 1)
            score = score - 0.5 * (distance / sigma).square()
        attention = torch.nan_to_num(F.softmax(score, dim=-1), nan=0.0)
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1)
        self.last_attention_entropy = float(entropy.mean().detach().cpu())
        message = torch.matmul(attention, value).transpose(1, 2).contiguous()
        message = message.view(h.shape[0], h.shape[1], self.output_dim)
        gate = torch.sigmoid(self.message_gate_logit)
        output = self.output_norm(
            self.residual_projection(h) + gate * self.message_projection(message))
        return F.gelu(output)


def _dimensions(block: nn.Module) -> tuple[int, int, int]:
    if all(hasattr(block, key) for key in ("input_dim", "output_dim", "num_heads")):
        return int(block.input_dim), int(block.output_dim), int(block.num_heads)
    input_shape = block.input_norm.normalized_shape
    return int(input_shape[0]), int(block.output_dim), int(block.num_heads)


def _new_block(
    old: nn.Module,
    mode: str,
    distance_kind: str,
    physical: torch.Tensor | None,
) -> DenseDistanceAttention:
    input_dim, output_dim, heads = _dimensions(old)
    return DenseDistanceAttention(
        input_dim, output_dim, heads,
        distance_kind=("none" if mode == "dense" else distance_kind),
        learnable_sigma=mode == "learned",
        physical_distance=physical)


def configure_attention(
    model: nn.Module,
    e1_mode: str,
    e2_mode: str,
    locations_path: Path,
) -> None:
    e1_mode = str(e1_mode).lower()
    e2_mode = str(e2_mode).lower()
    if e1_mode not in ATTENTION_MODES or e2_mode not in ATTENTION_MODES:
        raise ValueError(f"Unsupported attention modes: E1={e1_mode}, E2={e2_mode}")
    device = next(model.parameters()).device
    if e1_mode not in {"hard", "legacy_hybrid"}:
        model.branch1 = nn.ModuleList([
            _new_block(block, e1_mode, "expression", None).to(device)
            for block in model.branch1
        ])
    if e2_mode not in {"hard", "legacy_hybrid"}:
        physical = None
        if e2_mode in {"fixed", "learned"}:
            physical = normalized_physical_distance(locations_path)
        model.branch2_spatial = nn.ModuleList([
            _new_block(block, e2_mode, "physical", physical).to(device)
            for block in model.branch2_spatial
        ])
    model.seqfish_e1_attention = e1_mode
    model.seqfish_e2_attention = e2_mode


def _settings_from_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    datasets = list(config.get("datasets", []))
    if len(datasets) != 1:
        raise ValueError("seqFISH attention runtime expects one dataset")
    return {
        "e1": str(config.get("seqfish_e1_attention", "hard")),
        "e2": str(config.get("seqfish_e2_attention", "legacy_hybrid")),
        "locations": Path(str(datasets[0]["real_location_path"])),
    }


def install_model_patch(settings: Mapping[str, Any]) -> None:
    """Patch only DACGModel instances created in this Python process."""
    global _SETTINGS
    _SETTINGS = dict(settings)
    from models.DACG_model import DACGModel

    if getattr(DACGModel, "_seqfish_v3_attention_patch", False):
        return
    original_init = DACGModel.__init__
    original_rebuild = DACGModel.rebuild_io_layers

    def apply(model: nn.Module) -> None:
        configure_attention(
            model, _SETTINGS["e1"], _SETTINGS["e2"],
            Path(_SETTINGS["locations"]))

    @wraps(original_init)
    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        apply(self)

    @wraps(original_rebuild)
    def rebuild(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_rebuild(self, *args, **kwargs)
        apply(self)
        return result

    DACGModel.__init__ = init
    DACGModel.rebuild_io_layers = rebuild
    DACGModel._seqfish_v3_attention_patch = True


def install_from_cli() -> None:
    if "-i" not in sys.argv:
        raise RuntimeError("seqFISH attention runtime requires -i CONFIG")
    config_path = Path(sys.argv[sys.argv.index("-i") + 1])
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    install_model_patch(_settings_from_config(config))


def diagnostics(model: nn.Module) -> Dict[str, Any]:
    rows = []
    for branch, blocks in (("E1", model.branch1), ("E2", model.branch2_spatial)):
        for index, block in enumerate(blocks):
            if isinstance(block, DenseDistanceAttention):
                rows.append({
                    "branch": branch,
                    "block": index,
                    "distance_kind": block.distance_kind,
                    "sigmas": [float(value) for value in
                               block.effective_sigmas().detach().cpu()],
                    "message_gate": float(torch.sigmoid(
                        block.message_gate_logit).detach().cpu()),
                    "attention_entropy": float(block.last_attention_entropy),
                })
    return {
        "e1": str(getattr(model, "seqfish_e1_attention", "hard")),
        "e2": str(getattr(model, "seqfish_e2_attention", "legacy_hybrid")),
        "blocks": rows,
    }
