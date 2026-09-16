"""Inference-only SILTA architecture used by the released DLPFC checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dim_per_head = output_dim // num_heads
        self.Q = nn.Linear(input_dim, output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = h.shape
        queries = self.Q(h).view(batch, nodes, self.num_heads, self.dim_per_head)
        keys = self.K(h).view(batch, nodes, self.num_heads, self.dim_per_head)
        values = self.V(h).view(batch, nodes, self.num_heads, self.dim_per_head)
        mask = adjacency == 0
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        outputs = []
        for head in range(self.num_heads):
            scores = torch.matmul(
                queries[:, :, head], keys[:, :, head].transpose(1, 2)
            ) / math.sqrt(self.dim_per_head)
            scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.nan_to_num(F.softmax(scores, dim=2), nan=0.0)
            outputs.append(torch.matmul(weights, values[:, :, head]))
        return torch.cat(outputs, dim=2)


class GraphTransformerBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.attention = GraphAttentionBlock(input_dim, output_dim, num_heads)
        # Retained for exact checkpoint compatibility; the published forward
        # path, like the training code, does not apply these modules.
        self.projection_layer = nn.Linear(output_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.relu(self.attention(h, adjacency))


class SpatialHypergraphAttentionBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        eps: float = 1e-8,
        message_init: float = 0.1,
    ):
        super().__init__()
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = float(eps)
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
        gate_logit = math.log(message_init / (1.0 - message_init))
        self.message_gate_logit = nn.Parameter(torch.tensor(gate_logit))

    def _as_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(*value.shape[:-1], self.num_heads, self.head_dim)

    def _incidence(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.shape[-2:] != (h.shape[1], h.shape[1]):
            raise ValueError("spatial adjacency and feature shapes differ")
        if adjacency.shape[0] == 1 and h.shape[0] > 1:
            adjacency = adjacency.expand(h.shape[0], -1, -1)
        membership = adjacency.to(h.device) > 0
        eye = torch.eye(h.shape[1], dtype=torch.bool, device=h.device).unsqueeze(0)
        return (membership | eye).transpose(1, 2)

    def _segment_softmax(
        self, scores: torch.Tensor, group: torch.Tensor, groups: int
    ) -> torch.Tensor:
        pairs, heads = scores.shape
        head_index = torch.arange(heads, device=scores.device).view(1, -1)
        flat_group = (group.view(-1, 1) * heads + head_index).expand(
            pairs, -1
        ).reshape(-1)
        flat_scores = scores.reshape(-1)
        total_groups = int(groups) * heads
        maxima = flat_scores.new_full((total_groups,), -torch.inf)
        maxima.scatter_reduce_(
            0, flat_group, flat_scores, reduce="amax", include_self=True
        )
        weights = (flat_scores - maxima[flat_group].detach()).exp()
        denominator = flat_scores.new_zeros(total_groups)
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

    def forward(
        self, h: torch.Tensor, adjacency: torch.Tensor, expression_adjacency=None
    ) -> torch.Tensor:
        del expression_adjacency
        batch, nodes, _ = h.shape
        incidence = self._incidence(h, adjacency)
        batch_idx, node_idx, edge_idx = incidence.nonzero(as_tuple=True)
        normalized = self.input_norm(h)

        edge_query = self._as_heads(self.node_to_edge_q(normalized))
        node_key = self._as_heads(self.node_to_edge_k(normalized))
        node_value = self._as_heads(self.node_to_edge_v(normalized))
        edge_group = batch_idx * nodes + edge_idx
        scores = (
            edge_query[batch_idx, edge_idx] * node_key[batch_idx, node_idx]
        ).sum(dim=-1) * self.scale
        attention = self._segment_softmax(scores, edge_group, batch * nodes)
        message = self._segment_sum(
            attention.unsqueeze(-1) * node_value[batch_idx, node_idx],
            edge_group,
            batch * nodes,
        ).reshape(batch, nodes, self.output_dim)
        edge_feature = F.gelu(
            self.edge_norm(self.edge_center_proj(h) + self.edge_out_proj(message))
        )

        node_query = self._as_heads(self.edge_to_node_q(normalized))
        edge_key = self._as_heads(self.edge_to_node_k(edge_feature))
        edge_value = self._as_heads(self.edge_to_node_v(edge_feature))
        node_group = batch_idx * nodes + node_idx
        scores = (
            node_query[batch_idx, node_idx] * edge_key[batch_idx, edge_idx]
        ).sum(dim=-1) * self.scale
        attention = self._segment_softmax(scores, node_group, batch * nodes)
        message = self._segment_sum(
            attention.unsqueeze(-1) * edge_value[batch_idx, edge_idx],
            node_group,
            batch * nodes,
        ).reshape(batch, nodes, self.output_dim)
        gate = torch.sigmoid(self.message_gate_logit)
        return F.gelu(
            self.node_norm(self.res_proj(h) + gate * self.node_out_proj(message))
        )


class CooperativeFusionDecon(nn.Module):
    def __init__(
        self,
        h_dim: int,
        g_dim: int,
        m_dim: int,
        num_cell_types: int,
        fusion_dim: int = 256,
        aux_dim: int = 512,
        lambda_g: float = 1.0,
        lambda_x: float = 0.25,
        lambda_m: float = 0.02,
        dropout: float = 0.1,
    ):
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
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, num_cell_types),
        )

    def forward(
        self, h_mu: torch.Tensor, g_mu: torch.Tensor, m_latent: torch.Tensor
    ) -> torch.Tensor:
        h0 = self.h_norm(self.h_projection(h_mu))
        g0 = self.g_norm(self.g_projection(g_mu))
        interaction = self.lambda_x * self.interaction_projection(h0 * g0)
        m0 = self.m_norm(self.m_projection(m_latent))
        fused = self.fused_norm(
            h0 + self.lambda_g * g0 + interaction + self.lambda_m * m0
        )
        return self.decon_head(fused)


class SILTAInferenceModel(nn.Module):
    """Inference-only architecture for the locked DLPFC SILTA endpoint."""

    def __init__(
        self,
        num_genes: int,
        num_cell_types: int,
        encoder_out_channels=(256, 256, 512),
        num_heads: int = 4,
        h_dim: int = 192,
        g_dim: int = 192,
        mode_dim: int = 64,
        mode_num_classes: int = 7,
        mode_tau: float = 1.5,
        fusion_dim: int = 256,
        lambda_g: float = 1.0,
        lambda_x: float = 0.25,
        lambda_m: float = 0.02,
    ):
        super().__init__()
        self.mode_tau = float(mode_tau)
        self.branch1 = nn.ModuleList()
        self.branch2_spatial = nn.ModuleList()
        input_dim = int(num_genes)
        for output_dim in encoder_out_channels:
            self.branch1.append(
                GraphTransformerBlock(input_dim, output_dim, num_heads)
            )
            self.branch2_spatial.append(
                SpatialHypergraphAttentionBlock(input_dim, output_dim, num_heads)
            )
            input_dim = output_dim
        embed_dim = int(encoder_out_channels[-1])
        self.e1_norm = nn.LayerNorm(embed_dim)
        self.e2_norm = nn.LayerNorm(embed_dim)
        self.h_projector = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, h_dim), nn.ReLU()
        )
        self.g_projector = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, g_dim), nn.ReLU()
        )
        self.mode_prior_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, mode_num_classes)
        )
        self.mode_embedding = nn.Parameter(torch.empty(mode_num_classes, mode_dim))
        self.cooperative_decon = CooperativeFusionDecon(
            h_dim,
            g_dim,
            mode_dim,
            num_cell_types,
            fusion_dim=fusion_dim,
            aux_dim=512,
            lambda_g=lambda_g,
            lambda_x=lambda_x,
            lambda_m=lambda_m,
        )
        self.decon_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def _encode(self, blocks, expression, adjacency):
        hidden = expression
        for block in blocks:
            hidden = block(hidden, adjacency)
        return hidden

    def forward(
        self,
        expression: torch.Tensor,
        expression_adjacency: torch.Tensor,
        spatial_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.e1_norm(
            self._encode(self.branch1, expression, expression_adjacency)
        )
        e2 = self.e2_norm(
            self._encode(self.branch2_spatial, expression, spatial_adjacency)
        )
        mode_probs = F.softmax(self.mode_prior_head(e1) / self.mode_tau, dim=-1)
        m_latent = torch.matmul(mode_probs, self.mode_embedding)
        logits = self.cooperative_decon(
            self.h_projector(e1), self.g_projector(e2), m_latent
        )
        temperature = self.decon_temp.clamp(min=1.0, max=5.0)
        return F.softmax(logits / temperature, dim=-1)


def load_inference_state(path, model: nn.Module, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise TypeError("checkpoint must be a tensor-only state dictionary")
    model.load_state_dict(state, strict=True)
    return model
