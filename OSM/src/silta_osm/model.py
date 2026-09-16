from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_state_dict


class DenseAttentionBlock(nn.Module):
    """Vanilla dense multi-head attention used by the released E1 expert."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("output_dim must be divisible by heads")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(heads)
        self.head_dim = self.output_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.query = nn.Linear(self.input_dim, self.output_dim)
        self.key = nn.Linear(self.input_dim, self.output_dim)
        self.value = nn.Linear(self.input_dim, self.output_dim)
        self.message_projection = nn.Linear(self.output_dim, self.output_dim)
        self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        self.output_norm = nn.LayerNorm(self.output_dim)
        raw_beta = torch.full((self.num_heads,), math.log(math.expm1(1.0)))
        self.register_buffer("raw_beta", raw_beta)
        self.message_gate_logit = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32)
        )

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        return values.view(
            batch, spots, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(values)
        query = self._heads(self.query(normalized))
        key = self._heads(self.key(normalized))
        value = self._heads(self.value(normalized))
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * self.scale,
            dim=-1,
        )
        attention = torch.nan_to_num(attention, nan=0.0)
        message = torch.matmul(attention, value).transpose(1, 2).contiguous()
        message = message.view(values.shape[0], values.shape[1], self.output_dim)
        gate = torch.sigmoid(self.message_gate_logit)
        output = self.output_norm(
            self.residual_projection(values)
            + gate * self.message_projection(message)
        )
        return F.gelu(output)


class GraphAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_heads = int(heads)
        self.dim_per_head = self.output_dim // self.num_heads
        self.Q = nn.Linear(input_dim, output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, spots, _ = values.shape
        query = self.Q(values).view(
            batch, spots, self.num_heads, self.dim_per_head)
        key = self.K(values).view(
            batch, spots, self.num_heads, self.dim_per_head)
        value = self.V(values).view(
            batch, spots, self.num_heads, self.dim_per_head)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        heads = []
        for index in range(self.num_heads):
            score = torch.matmul(
                query[:, :, index], key[:, :, index].transpose(1, 2)
            ) / math.sqrt(self.dim_per_head)
            score = score.masked_fill(adjacency == 0, float("-inf"))
            weights = torch.nan_to_num(torch.softmax(score, dim=2), nan=0.0)
            heads.append(torch.matmul(weights, value[:, :, index]))
        return torch.cat(heads, dim=2)


class GraphTransformerBlock(nn.Module):
    """State-compatible inactive normal-graph branch."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_heads = int(heads)
        self.attention = GraphAttentionBlock(input_dim, output_dim, heads)
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(values, adjacency))


class DistanceHypergraphAttentionBlock(nn.Module):
    """Fixed-Gaussian sparse node-hyperedge-node attention used by E2."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 2) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("output_dim must be divisible by heads")
        self.output_dim = int(output_dim)
        self.num_heads = int(heads)
        self.head_dim = self.output_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = 1e-8
        self.input_norm = nn.LayerNorm(input_dim)
        self.node_to_edge_q = nn.Linear(input_dim, output_dim)
        self.node_to_edge_k = nn.Linear(input_dim, output_dim)
        self.node_to_edge_v = nn.Linear(input_dim, output_dim)
        self.edge_center_proj = nn.Linear(input_dim, output_dim)
        self.edge_out_proj = nn.Linear(output_dim, output_dim)
        self.edge_norm = nn.LayerNorm(output_dim)
        self.edge_to_node_q = nn.Linear(input_dim, output_dim)
        self.edge_to_node_k = nn.Linear(output_dim, output_dim)
        self.edge_to_node_v = nn.Linear(output_dim, output_dim)
        self.node_out_proj = nn.Linear(output_dim, output_dim)
        self.res_proj = nn.Linear(input_dim, output_dim)
        self.node_norm = nn.LayerNorm(output_dim)
        self.message_gate_logit = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32)
        )
        self.register_buffer(
            "raw_beta",
            torch.full((self.num_heads,), math.log(math.expm1(1.0))),
        )

    def _as_heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.view(*values.shape[:-1], self.num_heads, self.head_dim)

    def _incidence(
        self, values: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.shape[-2:] != (values.shape[1], values.shape[1]):
            raise ValueError("spatial adjacency does not match the spot count")
        if adjacency.shape[0] == 1 and values.shape[0] > 1:
            adjacency = adjacency.expand(values.shape[0], -1, -1)
        eye = torch.eye(
            values.shape[1], dtype=torch.bool, device=values.device
        ).unsqueeze(0)
        return ((adjacency > 0).to(values.device) | eye).transpose(1, 2)

    def _segment_softmax(
        self, scores: torch.Tensor, group: torch.Tensor, groups: int
    ) -> torch.Tensor:
        pairs, heads = scores.shape
        head_index = torch.arange(heads, device=scores.device).view(1, -1)
        flat_group = (
            group.view(-1, 1) * heads + head_index
        ).expand(pairs, -1).reshape(-1)
        flat_scores = scores.reshape(-1)
        maxima = flat_scores.new_full((int(groups) * heads,), -torch.inf)
        maxima.scatter_reduce_(
            0, flat_group, flat_scores, reduce="amax", include_self=True
        )
        weights = (flat_scores - maxima[flat_group].detach()).exp()
        denominator = flat_scores.new_zeros(int(groups) * heads)
        denominator.scatter_add_(0, flat_group, weights)
        return (weights / denominator[flat_group].clamp_min(self.eps)).view(
            pairs, heads
        )

    @staticmethod
    def _segment_sum(
        values: torch.Tensor, group: torch.Tensor, groups: int
    ) -> torch.Tensor:
        output = values.new_zeros(int(groups), values.shape[1], values.shape[2])
        output.index_add_(0, group, values)
        return output

    def _penalize(
        self, score: torch.Tensor, distance: torch.Tensor
    ) -> torch.Tensor:
        beta = F.softplus(self.raw_beta).to(score).view(1, self.num_heads)
        return score - beta * distance.to(score).view(-1, 1).square()

    def forward(
        self,
        values: torch.Tensor,
        adjacency: torch.Tensor,
        normalized_distance: torch.Tensor,
    ) -> torch.Tensor:
        batch, spots, _ = values.shape
        incidence = self._incidence(values, adjacency)
        batch_index, node_index, edge_index = incidence.nonzero(as_tuple=True)
        normalized = self.input_norm(values)
        distance = normalized_distance.to(values)[node_index, edge_index]

        edge_query = self._as_heads(self.node_to_edge_q(normalized))
        node_key = self._as_heads(self.node_to_edge_k(normalized))
        node_value = self._as_heads(self.node_to_edge_v(normalized))
        edge_group = batch_index * spots + edge_index
        score = (
            edge_query[batch_index, edge_index]
            * node_key[batch_index, node_index]
        ).sum(dim=-1) * self.scale
        score = self._penalize(score, distance)
        weight = self._segment_softmax(score, edge_group, batch * spots)
        edge_message = self._segment_sum(
            weight.unsqueeze(-1) * node_value[batch_index, node_index],
            edge_group,
            batch * spots,
        ).reshape(batch, spots, self.output_dim)
        edge_feature = F.gelu(self.edge_norm(
            self.edge_center_proj(values) + self.edge_out_proj(edge_message)
        ))

        node_query = self._as_heads(self.edge_to_node_q(normalized))
        edge_key = self._as_heads(self.edge_to_node_k(edge_feature))
        edge_value = self._as_heads(self.edge_to_node_v(edge_feature))
        node_group = batch_index * spots + node_index
        score = (
            node_query[batch_index, node_index]
            * edge_key[batch_index, edge_index]
        ).sum(dim=-1) * self.scale
        score = self._penalize(score, distance)
        weight = self._segment_softmax(score, node_group, batch * spots)
        node_message = self._segment_sum(
            weight.unsqueeze(-1) * edge_value[batch_index, edge_index],
            node_group,
            batch * spots,
        ).reshape(batch, spots, self.output_dim)
        gate = torch.sigmoid(self.message_gate_logit)
        return F.gelu(self.node_norm(
            self.res_proj(values) + gate * self.node_out_proj(node_message)
        ))


class AttentionResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())
        self.residual = nn.Linear(input_dim, output_dim)


class GeneAutoEncoder(nn.Module):
    def __init__(self, genes: int, latent: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(genes, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, genes),
        )


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self.dim = int(dimension)
        self.proj = nn.Linear(dimension, dimension)


class DenoiseMLP(nn.Module):
    def __init__(self, latent: int = 64, condition: int = 512) -> None:
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(128)
        self.net = nn.Sequential(
            nn.Linear(latent + condition + 128, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, latent),
        )


class CooperativeFusionDecon(nn.Module):
    def __init__(self, types: int) -> None:
        super().__init__()
        self.fusion_dim = 512
        self.lambda_g = 0.5
        self.lambda_x = 0.0
        self.lambda_m = 0.0
        self.h_projection = nn.Linear(192, 512)
        self.g_projection = nn.Linear(192, 512)
        self.m_projection = nn.Linear(64, 512)
        self.h_norm = nn.LayerNorm(512)
        self.g_norm = nn.LayerNorm(512)
        self.m_norm = nn.LayerNorm(512)
        self.interaction_projection = nn.Linear(512, 512)
        self.fused_norm = nn.LayerNorm(512)
        self.mu_layer = nn.Linear(512, 512)
        self.logvar_layer = nn.Linear(512, 512)
        self.z_norm = nn.LayerNorm(512)
        self.aux_projection = nn.Linear(512, 512)
        self.decon_head = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, types),
        )

    def forward(
        self, h_value: torch.Tensor, g_value: torch.Tensor, mode_value: torch.Tensor
    ) -> torch.Tensor:
        h_projected = self.h_norm(self.h_projection(h_value))
        g_projected = self.g_norm(self.g_projection(g_value))
        interaction = self.lambda_x * self.interaction_projection(
            h_projected * g_projected
        )
        mode_projected = self.m_norm(self.m_projection(mode_value))
        fused = self.fused_norm(
            h_projected
            + self.lambda_g * g_projected
            + interaction
            + self.lambda_m * mode_projected
        )
        return self.decon_head(fused)


class SpatialResidualDecon(nn.Module):
    def __init__(self, types: int) -> None:
        super().__init__()
        self.direct_h_head = self._head(types)
        self.dual_g_head = self._head(types)
        self.delta_head = self._head(types)

    @staticmethod
    def _head(types: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(192), nn.Linear(192, 128), nn.ReLU(),
            nn.Linear(128, types),
        )


class SILTAModel(nn.Module):
    """Inference-only OSM endpoint with checkpoint-compatible module names."""

    def __init__(self, genes: int = 33, types: int = 31) -> None:
        super().__init__()
        self.num_genes = int(genes)
        self.num_cell_types = int(types)
        self.structured_mode_tau = 1.5

        self.branch2_beta = nn.Parameter(torch.tensor(1.5))
        self.mode_embedding = nn.Parameter(torch.empty(types, 64))
        self.m_gate = nn.Parameter(torch.tensor(0.0))
        self.decon_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        dimensions = ((genes, 256), (256, 256), (256, 512))
        self.branch1 = nn.ModuleList([
            DenseAttentionBlock(source, target, 4)
            for source, target in dimensions
        ])
        self.branch2_spatial = nn.ModuleList([
            DistanceHypergraphAttentionBlock(source, target, 2)
            for source, target in dimensions
        ])
        self.branch2_normal = nn.ModuleList([
            GraphTransformerBlock(source, target, 4)
            for source, target in dimensions
        ])

        self.h_projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 192), nn.ReLU())
        self.h_logvar_projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 192))
        self.g_projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 192), nn.ReLU())
        self.g_logvar_projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 192))
        self.m_residual_projector = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 64), nn.Tanh())
        self.fusion = AttentionResidualMLP(448, 256, 512)
        self.e1_norm = nn.LayerNorm(512)
        self.e2_norm = nn.LayerNorm(512)
        self.fused_norm = nn.LayerNorm(512)
        self.z_pre_gate_norm = nn.LayerNorm(512)
        self.mu_layer = nn.Linear(512, 512)
        self.logvar_layer = nn.Linear(512, 512)
        self.zig_logit_layer = nn.Linear(512, 512)
        self.gene_ae = GeneAutoEncoder(genes, 64)
        self.denoise_mlp = DenoiseMLP(64, 512)
        self.decon_head = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, types),
        )
        self.cond_predictor = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, types))
        self.mode_head = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, types))
        self.mode_prior_head = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, types))
        self.m_cls_head = nn.Linear(64, types)
        self.domain_classifier = nn.Module()
        self.domain_classifier.net = nn.Sequential(
            nn.Identity(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
        self.cooperative_decon = CooperativeFusionDecon(types)
        self.spatial_decon = SpatialResidualDecon(types)

    def forward(
        self,
        expression: torch.Tensor,
        expression_adjacency: torch.Tensor,
        spatial_adjacency: torch.Tensor,
        normalized_distance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if expression.dim() == 2:
            expression = expression.unsqueeze(0)
        if expression.shape[0] != 1:
            raise ValueError("The released OSM endpoint requires batch size one")
        spots = expression.shape[1]
        for name, matrix in (
            ("expression_adjacency", expression_adjacency),
            ("spatial_adjacency", spatial_adjacency),
            ("normalized_distance", normalized_distance),
        ):
            if matrix.shape[-2:] != (spots, spots):
                raise ValueError(f"{name} does not match the spot count")

        content = expression
        for block in self.branch1:
            content = block(content)
        content = self.e1_norm(content)

        mode_logits = self.mode_prior_head(content)
        mode_probs = torch.softmax(
            mode_logits / self.structured_mode_tau, dim=-1)

        spatial = expression
        for block in self.branch2_spatial:
            spatial = block(spatial, spatial_adjacency, normalized_distance)
        spatial = self.e2_norm(spatial)

        h_mu = self.h_projector(content)
        g_mu = self.g_projector(spatial)
        mode_latent = torch.matmul(mode_probs, self.mode_embedding)
        logits = self.cooperative_decon(h_mu, g_mu, mode_latent)
        temperature = self.decon_temp.clamp(min=1.0, max=5.0)
        fractions = torch.softmax(logits / temperature, dim=-1)
        return {
            "logits": logits,
            "fractions": fractions,
            "content": content,
            "spatial": spatial,
            "h_mu": h_mu,
            "g_mu": g_mu,
            "mode_probs": mode_probs,
        }


def load_model(
    checkpoint: Path | str,
    device: torch.device | str = "cpu",
) -> SILTAModel:
    state = load_state_dict(checkpoint)
    model = SILTAModel()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint loading failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device).eval()


def state_compatibility(checkpoint: Path | str) -> dict[str, Any]:
    model = load_model(checkpoint, "cpu")
    return {
        "strict": True,
        "model_state_keys": len(model.state_dict()),
        "checkpoint_state_keys": len(load_state_dict(checkpoint)),
    }
