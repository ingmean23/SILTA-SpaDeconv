import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DECON_ARCHITECTURES = {
    "concat",
    "content_only",
    "coop_direct",
    "coop_gaussian",
    "direct_h",
    "spatial_residual",
    "dual_head",
}


def residual_alpha(epoch, alpha_max, delay_epochs, warmup_epochs,
                   schedule="cosine"):
    epoch = int(epoch)
    alpha_max = max(0.0, float(alpha_max))
    delay_epochs = max(0, int(delay_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    schedule = str(schedule or "cosine").lower()
    if alpha_max == 0.0 or epoch < delay_epochs:
        return 0.0
    if warmup_epochs == 0:
        return alpha_max
    progress = min(
        max(float(epoch - delay_epochs + 1) / float(warmup_epochs), 0.0),
        1.0,
    )
    if schedule == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    elif schedule == "linear":
        factor = progress
    else:
        raise ValueError(
            "spatial_residual_schedule must be 'cosine' or 'linear'")
    return alpha_max * factor


class SpatialResidualDecon(nn.Module):
    def __init__(self, h_dim, g_dim, num_cell_types, hidden_dim=128):
        super().__init__()
        self.direct_h_head = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cell_types),
        )
        self.dual_g_head = nn.Sequential(
            nn.LayerNorm(g_dim),
            nn.Linear(g_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cell_types),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(g_dim),
            nn.Linear(g_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cell_types),
        )

    def direct_h_logits(self, h_mu):
        return self.direct_h_head(h_mu)

    def dual_logits(self, h_mu, g_mu):
        return self.direct_h_head(h_mu), self.dual_g_head(g_mu)

    def delta_logits(self, g_mu):
        return self.delta_head(g_mu)


def _zero_loss(reference):
    return reference.sum() * 0.0


def spatial_graph_reconstruction_loss(
    g_mu,
    adjacency,
    temperature=0.5,
    negative_ratio=5,
    generator=None,
):
    if g_mu is None or adjacency is None:
        raise ValueError("g_mu and adjacency are required")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("g_graph_temperature must be positive")
    negative_ratio = max(1, int(negative_ratio))

    if g_mu.dim() == 2:
        g_mu = g_mu.unsqueeze(0)
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)
    if adjacency.shape[0] == 1 and g_mu.shape[0] > 1:
        adjacency = adjacency.expand(g_mu.shape[0], -1, -1)
    if adjacency.shape[:2] != g_mu.shape[:2]:
        raise ValueError(
            "adjacency and g_mu must have matching batch/node dimensions")

    losses = []
    positive_scores = []
    negative_scores = []
    for batch_index in range(g_mu.shape[0]):
        node_count = g_mu.shape[1]
        upper = torch.triu(
            torch.ones(
                node_count, node_count, dtype=torch.bool, device=g_mu.device),
            diagonal=1,
        )
        edge_mask = adjacency[batch_index].to(g_mu.device) > 0
        positive_pairs = torch.nonzero(
            edge_mask & upper, as_tuple=False)
        negative_pairs = torch.nonzero(
            (~edge_mask) & upper, as_tuple=False)
        if positive_pairs.numel() == 0 or negative_pairs.numel() == 0:
            continue

        negative_count = min(
            negative_pairs.shape[0],
            positive_pairs.shape[0] * negative_ratio,
        )
        order = torch.randperm(
            negative_pairs.shape[0],
            device=negative_pairs.device,
            generator=generator,
        )[:negative_count]
        negative_pairs = negative_pairs[order]

        g_unit = F.normalize(g_mu[batch_index], p=2, dim=-1, eps=1e-8)
        pos = (
            g_unit[positive_pairs[:, 0]]
            * g_unit[positive_pairs[:, 1]]
        ).sum(dim=-1) / temperature
        neg = (
            g_unit[negative_pairs[:, 0]]
            * g_unit[negative_pairs[:, 1]]
        ).sum(dim=-1) / temperature
        losses.append(F.softplus(-pos).mean() + F.softplus(neg).mean())
        positive_scores.append(pos.detach())
        negative_scores.append(neg.detach())

    if not losses:
        zero = _zero_loss(g_mu)
        return zero, {
            "positive_score_mean": 0.0,
            "negative_score_mean": 0.0,
            "positive_edges": 0,
            "negative_edges": 0,
        }

    pos_all = torch.cat(positive_scores)
    neg_all = torch.cat(negative_scores)
    return torch.stack(losses).mean(), {
        "positive_score_mean": float(pos_all.mean().cpu().item()),
        "negative_score_mean": float(neg_all.mean().cpu().item()),
        "positive_edges": int(pos_all.numel()),
        "negative_edges": int(neg_all.numel()),
    }
