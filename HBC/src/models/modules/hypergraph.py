import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialHypergraphBlock(nn.Module):
    """Spatial hypergraph propagation block.

    Each spot-centered spatial neighborhood is treated as one hyperedge. The
    input adjacency therefore acts as the incidence pattern, so this block can
    reuse the existing ST dataloader output without changing main_zacae.py.
    """

    def __init__(self, input_dim, output_dim, use_bias=True, eps=1e-8):
        super().__init__()
        self.eps = float(eps)
        self.hyper_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.res_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=use_bias),
            nn.Sigmoid(),
        )

    def forward(self, h, sp_adj, ex_adj=None):
        if sp_adj.dim() == 2:
            sp_adj = sp_adj.unsqueeze(0)
        if sp_adj.shape[0] == 1 and h.shape[0] > 1:
            sp_adj = sp_adj.expand(h.shape[0], -1, -1)

        H = (sp_adj > 0).to(dtype=h.dtype, device=h.device)
        n = H.shape[-1]
        eye = torch.eye(n, dtype=h.dtype, device=h.device).unsqueeze(0)
        H = torch.maximum(H, eye)

        # H[node, hyperedge] marks whether a node belongs to a neighborhood
        # hyperedge. For symmetric spatial graphs, transpose is equivalent but
        # keeping it explicit makes directed precomputed graphs predictable.
        H = H.transpose(1, 2)
        edge_degree = H.sum(dim=1).clamp_min(self.eps)
        node_degree = H.sum(dim=2).clamp_min(self.eps)

        edge_feat = torch.bmm(H.transpose(1, 2), h) / edge_degree.unsqueeze(-1)
        node_feat = torch.bmm(H, edge_feat) / node_degree.unsqueeze(-1)

        hyper_out = self.hyper_proj(node_feat)
        residual = self.res_proj(h)
        gate = self.gate(h)
        return F.relu(gate * hyper_out + (1.0 - gate) * residual)


class SpatialHypergraphAttentionBlock(nn.Module):
    """Center-conditioned node-hyperedge-node attention.

    A spatial neighborhood centered on spot j is treated as hyperedge j. Both
    attention passes operate only on non-zero incidence entries, avoiding dense
    Q/K/V attention tensors for sparse spatial graphs.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=4,
        use_bias=True,
        eps=1e-8,
        message_init=0.1,
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(
                f"output_dim={output_dim} must be divisible by num_heads={num_heads}"
            )
        if not 0.0 < message_init < 1.0:
            raise ValueError("message_init must be strictly between 0 and 1")

        self.output_dim = int(output_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.output_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = float(eps)

        self.input_norm = nn.LayerNorm(input_dim)

        self.node_to_edge_q = nn.Linear(
            input_dim, output_dim, bias=use_bias)
        self.node_to_edge_k = nn.Linear(
            input_dim, output_dim, bias=use_bias)
        self.node_to_edge_v = nn.Linear(
            input_dim, output_dim, bias=use_bias)
        self.edge_center_proj = nn.Linear(
            input_dim, output_dim, bias=use_bias)
        self.edge_out_proj = nn.Linear(
            output_dim, output_dim, bias=use_bias)
        self.edge_norm = nn.LayerNorm(output_dim)

        self.edge_to_node_q = nn.Linear(
            input_dim, output_dim, bias=use_bias)
        self.edge_to_node_k = nn.Linear(
            output_dim, output_dim, bias=use_bias)
        self.edge_to_node_v = nn.Linear(
            output_dim, output_dim, bias=use_bias)
        self.node_out_proj = nn.Linear(
            output_dim, output_dim, bias=use_bias)
        self.res_proj = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.node_norm = nn.LayerNorm(output_dim)

        gate_logit = math.log(message_init / (1.0 - message_init))
        self.message_gate_logit = nn.Parameter(
            torch.tensor(gate_logit, dtype=torch.float32)
        )

    def _as_heads(self, x):
        return x.view(
            *x.shape[:-1], self.num_heads, self.head_dim)

    def _prepare_incidence(self, h, sp_adj):
        if sp_adj is None:
            raise ValueError("sp_adj is required for hypergraph attention")
        if sp_adj.dim() == 2:
            sp_adj = sp_adj.unsqueeze(0)
        if sp_adj.dim() != 3:
            raise ValueError(
                f"sp_adj must have 2 or 3 dimensions, got {sp_adj.dim()}")
        if sp_adj.shape[-2:] != (h.shape[1], h.shape[1]):
            raise ValueError(
                f"sp_adj shape {tuple(sp_adj.shape)} is incompatible with "
                f"{h.shape[1]} nodes"
            )
        if sp_adj.shape[0] == 1 and h.shape[0] > 1:
            sp_adj = sp_adj.expand(h.shape[0], -1, -1)
        elif sp_adj.shape[0] != h.shape[0]:
            raise ValueError(
                f"sp_adj batch {sp_adj.shape[0]} != feature batch {h.shape[0]}"
            )

        membership = sp_adj > 0
        eye = torch.eye(
            h.shape[1], dtype=torch.bool, device=h.device).unsqueeze(0)
        membership = membership.to(device=h.device) | eye

        # incidence[node, edge] follows the existing mean hypergraph block.
        return membership.transpose(1, 2)

    def _segment_softmax(self, scores, group, num_groups):
        """Softmax over pair scores grouped independently for every head."""
        pair_count, num_heads = scores.shape
        head_index = torch.arange(
            num_heads, device=scores.device).view(1, -1)
        flat_group = (
            group.view(-1, 1) * num_heads + head_index
        ).expand(pair_count, -1).reshape(-1)
        flat_scores = scores.reshape(-1)
        total_groups = int(num_groups) * num_heads

        maxima = flat_scores.new_full(
            (total_groups,), -torch.inf)
        maxima.scatter_reduce_(
            0, flat_group, flat_scores, reduce="amax", include_self=True)
        shifted = flat_scores - maxima[flat_group].detach()
        weights = shifted.exp()
        denominator = flat_scores.new_zeros(total_groups)
        denominator.scatter_add_(0, flat_group, weights)
        weights = weights / denominator[flat_group].clamp_min(self.eps)
        return weights.view(pair_count, num_heads)

    @staticmethod
    def _segment_sum(values, group, num_groups):
        output = values.new_zeros(
            int(num_groups), values.shape[1], values.shape[2])
        output.index_add_(0, group, values)
        return output

    def forward(self, h, sp_adj, ex_adj=None):
        del ex_adj
        batch_size, num_nodes, _ = h.shape
        incidence = self._prepare_incidence(h, sp_adj)
        batch_idx, node_idx, edge_idx = incidence.nonzero(as_tuple=True)
        normalized = self.input_norm(h)

        edge_query = self._as_heads(
            self.node_to_edge_q(normalized))
        node_key = self._as_heads(
            self.node_to_edge_k(normalized))
        node_value = self._as_heads(
            self.node_to_edge_v(normalized))

        edge_group = batch_idx * num_nodes + edge_idx
        node_to_edge_score = (
            edge_query[batch_idx, edge_idx]
            * node_key[batch_idx, node_idx]
        ).sum(dim=-1) * self.scale
        node_to_edge_attn = self._segment_softmax(
            node_to_edge_score, edge_group, batch_size * num_nodes)
        edge_message = self._segment_sum(
            node_to_edge_attn.unsqueeze(-1)
            * node_value[batch_idx, node_idx],
            edge_group,
            batch_size * num_nodes,
        )
        edge_message = edge_message.reshape(
            batch_size, num_nodes, self.output_dim)
        edge_feature = self.edge_norm(
            self.edge_center_proj(h)
            + self.edge_out_proj(edge_message)
        )
        edge_feature = F.gelu(edge_feature)

        node_query = self._as_heads(
            self.edge_to_node_q(normalized))
        edge_key = self._as_heads(
            self.edge_to_node_k(edge_feature))
        edge_value = self._as_heads(
            self.edge_to_node_v(edge_feature))

        node_group = batch_idx * num_nodes + node_idx
        edge_to_node_score = (
            node_query[batch_idx, node_idx]
            * edge_key[batch_idx, edge_idx]
        ).sum(dim=-1) * self.scale
        edge_to_node_attn = self._segment_softmax(
            edge_to_node_score, node_group, batch_size * num_nodes)
        node_message = self._segment_sum(
            edge_to_node_attn.unsqueeze(-1)
            * edge_value[batch_idx, edge_idx],
            node_group,
            batch_size * num_nodes,
        )
        node_message = node_message.reshape(
            batch_size, num_nodes, self.output_dim)

        message_gate = torch.sigmoid(self.message_gate_logit)
        output = self.node_norm(
            self.res_proj(h)
            + message_gate * self.node_out_proj(node_message)
        )
        return F.gelu(output)
