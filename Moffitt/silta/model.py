from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class GraphAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.dim_per_head = self.output_dim // self.num_heads
        self.Q = nn.Linear(input_dim, output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        query = self.Q(values).view(batch, spots, self.num_heads, self.dim_per_head)
        key = self.K(values).view(batch, spots, self.num_heads, self.dim_per_head)
        value = self.V(values).view(batch, spots, self.num_heads, self.dim_per_head)
        mask = adjacency == 0
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        heads = []
        for head in range(self.num_heads):
            score = torch.matmul(
                query[:, :, head], key[:, :, head].transpose(1, 2)
            ) / math.sqrt(self.dim_per_head)
            score = score.masked_fill(mask, float("-inf"))
            attention = torch.nan_to_num(F.softmax(score, dim=2), nan=0.0)
            heads.append(torch.matmul(attention, value[:, :, head]))
        return torch.cat(heads, dim=2)


class GraphTransformerBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.attention = GraphAttentionBlock(input_dim, output_dim, num_heads)
        # These layers are retained for state-dict parity; the locked forward does not use them.
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(values, adjacency))


class DistanceDecayAttentionBlock(nn.Module):
    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__()
        input_dim = int(metadata["input_dim"])
        output_dim = int(metadata["output_dim"])
        self.output_dim = output_dim
        self.num_heads = int(metadata["num_heads"])
        self.head_dim = output_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.learnable_sigma = bool(metadata["learnable_sigma"])
        self.use_distance_bias = bool(metadata["use_distance_bias"])
        self.sigma_min = float(metadata["sigma_min"])
        self.sigma_max = float(metadata["sigma_max"])
        self.input_norm = nn.LayerNorm(input_dim)
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.message_projection = nn.Linear(output_dim, output_dim)
        self.residual_projection = nn.Linear(input_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        sigmas = torch.tensor(metadata["effective_sigmas"], dtype=torch.float32)
        if self.learnable_sigma:
            ratio = (sigmas - self.sigma_min) / (self.sigma_max - self.sigma_min)
            self.sigma_logits = nn.Parameter(torch.logit(ratio.clamp(1e-6, 1 - 1e-6)))
            self.register_buffer("fixed_sigmas", torch.empty(0))
        else:
            self.register_parameter("sigma_logits", None)
            self.register_buffer("fixed_sigmas", sigmas)
        self.message_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("distance_matrix", torch.empty(0), persistent=False)

    def set_distance_matrix(self, distance: torch.Tensor) -> None:
        if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
            raise ValueError("distance matrix must be square")
        self.distance_matrix = distance.float()

    def effective_sigmas(self) -> torch.Tensor:
        if not self.learnable_sigma:
            return self.fixed_sigmas
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(
            self.sigma_logits
        )

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        return values.view(batch, spots, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, values: torch.Tensor, spatial_adjacency: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        if self.distance_matrix.shape != (spots, spots):
            raise ValueError("physical distance matrix was not installed")
        if spatial_adjacency.shape[-2:] != (spots, spots):
            raise ValueError("spatial adjacency shape mismatch")
        normalized = self.input_norm(values)
        query = self._heads(self.query(normalized))
        key = self._heads(self.key(normalized))
        value = self._heads(self.value(normalized))
        score = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        distance = self.distance_matrix.to(device=values.device, dtype=score.dtype)
        sigmas = self.effective_sigmas().to(device=values.device, dtype=score.dtype)
        score = score - 0.5 * (
            distance.view(1, 1, spots, spots)
            / sigmas.view(1, self.num_heads, 1, 1)
        ).square()
        attention = torch.nan_to_num(F.softmax(score, dim=-1), nan=0.0)
        message = torch.matmul(attention, value).transpose(1, 2).contiguous()
        message = message.view(batch, spots, self.output_dim)
        gate = torch.sigmoid(self.message_gate_logit)
        output = self.output_norm(
            self.residual_projection(values) + gate * self.message_projection(message)
        )
        return F.gelu(output)


class CooperativeFusionDecon(nn.Module):
    def __init__(self, h_dim: int, g_dim: int, m_dim: int, num_types: int,
                 fusion_dim: int, aux_dim: int, lambda_g: float,
                 lambda_x: float, lambda_m: float) -> None:
        super().__init__()
        self.lambda_g = float(lambda_g)
        self.lambda_x = float(lambda_x)
        self.lambda_m = float(lambda_m)
        self.h_projection = nn.Linear(h_dim, fusion_dim)
        self.g_projection = nn.Linear(g_dim, fusion_dim)
        self.m_projection = nn.Linear(m_dim, fusion_dim)
        self.h_norm = nn.LayerNorm(fusion_dim)
        self.g_norm = nn.LayerNorm(fusion_dim)
        self.m_norm = nn.LayerNorm(fusion_dim)
        self.interaction_projection = nn.Linear(fusion_dim, fusion_dim)
        self.fused_norm = nn.LayerNorm(fusion_dim)
        self.mu_layer = nn.Linear(fusion_dim, fusion_dim)
        self.logvar_layer = nn.Linear(fusion_dim, fusion_dim)
        self.z_norm = nn.LayerNorm(fusion_dim)
        self.aux_projection = nn.Linear(fusion_dim, aux_dim)
        self.decon_head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, num_types),
        )

    def forward(self, h: torch.Tensor, g: torch.Tensor, m: torch.Tensor,
                architecture: str) -> torch.Tensor:
        h0 = self.h_norm(self.h_projection(h))
        g0 = self.g_norm(self.g_projection(g))
        interaction = self.lambda_x * self.interaction_projection(h0 * g0)
        m0 = self.m_norm(self.m_projection(m))
        fused = self.fused_norm(
            h0 + self.lambda_g * g0 + interaction + self.lambda_m * m0
        )
        if architecture == "coop_gaussian":
            representation = self.z_norm(self.mu_layer(fused))
        elif architecture == "coop_direct":
            representation = fused
        else:
            raise ValueError(f"Unsupported cooperative architecture: {architecture}")
        return self.decon_head(representation)


class SILTAInferenceModel(nn.Module):
    """Inference-only SILTA model for the released Moffitt checkpoint."""

    ACTIVE_PREFIXES = (
        "branch1.", "branch2_spatial.", "e1_norm.", "e2_norm.",
        "h_projector.", "g_projector.", "mode_embedding",
        "mode_prior_head.", "cooperative_decon.",
    )

    def __init__(self, checkpoint: Mapping[str, Any]) -> None:
        super().__init__()
        config = checkpoint["model_config"]
        widths = list(map(int, config["encoder_out_channels"]))
        heads = int(config["num_heads"])
        genes = int(config["num_genes"])
        types = int(config["num_cell_types"])
        self.structured_mode_tau = float(config["structured_mode_tau"])
        self.decon_architecture = str(config["decon_architecture"])
        self.branch1 = nn.ModuleList()
        input_dim = genes
        for output_dim in widths:
            self.branch1.append(GraphTransformerBlock(input_dim, output_dim, heads))
            input_dim = output_dim
        layers = checkpoint["attention_config"]["layers"]
        self.branch2_spatial = nn.ModuleList(
            [DistanceDecayAttentionBlock(layer) for layer in layers]
        )
        embed_dim = widths[-1]
        h_dim = int(config["structured_h_dim"])
        g_dim = int(config["structured_g_dim"])
        m_dim = int(config["structured_mode_dim"])
        self.e1_norm = nn.LayerNorm(embed_dim)
        self.e2_norm = nn.LayerNorm(embed_dim)
        self.h_projector = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, h_dim), nn.ReLU()
        )
        self.g_projector = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, g_dim), nn.ReLU()
        )
        self.mode_embedding = nn.Parameter(torch.empty(types, m_dim))
        self.mode_prior_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, types)
        )
        self.cooperative_decon = CooperativeFusionDecon(
            h_dim, g_dim, m_dim, types,
            int(config["coop_fusion_dim"]), 512,
            float(config["lambda_g"]), float(config["lambda_x"]),
            float(config["lambda_m"]),
        )

    def install_distance(self, distance: torch.Tensor) -> None:
        for layer in self.branch2_spatial:
            layer.set_distance_matrix(distance)

    def load_active_state(self, full_state: Mapping[str, torch.Tensor]) -> None:
        active = {
            key: value for key, value in full_state.items()
            if key == "mode_embedding" or key.startswith(self.ACTIVE_PREFIXES)
        }
        self.load_state_dict(active, strict=True)

    def forward(self, x: torch.Tensor, ex_adj: torch.Tensor,
                sp_adj: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        h = x
        for layer in self.branch1:
            h = layer(h, ex_adj)
        e1 = self.e1_norm(h)
        g = x
        for layer in self.branch2_spatial:
            g = layer(g, sp_adj)
        e2 = self.e2_norm(g)
        mode_logits = self.mode_prior_head(e1)
        mode_probs = F.softmax(mode_logits / max(self.structured_mode_tau, 1e-6), dim=-1)
        mode = torch.matmul(mode_probs, self.mode_embedding)
        return self.cooperative_decon(
            self.h_projector(e1), self.g_projector(e2), mode,
            self.decon_architecture,
        )
