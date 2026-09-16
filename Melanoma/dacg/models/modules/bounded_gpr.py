"""Bounded generalized PageRank residual for spatial latent features."""

import torch
import torch.nn as nn


class BoundedGPRResidual(nn.Module):
    """Add a learnable, bounded multi-hop spatial residual to a base feature.

    The residual uses signed, L1-normalized GPR coefficients:

        H^(0) = LayerNorm(g_base)
        H^(k) = A_rw H^(k-1)
        r = beta * tanh(W sum_k gamma_k H^(k))
        g_out = g_base + r

    ``W`` is zero-initialized, so construction is an exact identity while
    retaining a trainable path from the first optimization step.
    """

    SUPPORTED_INITS = frozenset(("ppr", "identity", "uniform"))

    def __init__(
        self,
        dim,
        hops=3,
        residual_cap=0.05,
        teleport=0.1,
        init="ppr",
        gate_init=0.0,
        add_self_loops=True,
        eps=1e-8,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hops = int(hops)
        self.residual_cap = float(residual_cap)
        self.teleport = float(teleport)
        self.init = str(init).lower()
        self.add_self_loops = bool(add_self_loops)
        self.eps = float(eps)

        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.hops < 0:
            raise ValueError("hops must be nonnegative")
        if self.residual_cap < 0.0:
            raise ValueError("residual_cap must be nonnegative")
        if not 0.0 < self.teleport <= 1.0:
            raise ValueError("teleport must be in (0, 1]")
        if self.init not in self.SUPPORTED_INITS:
            raise ValueError(
                "init must be one of: " + ", ".join(sorted(self.SUPPORTED_INITS))
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        self.input_norm = nn.LayerNorm(self.dim)
        self.gamma_raw = nn.Parameter(self._initial_gamma())
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        # Preserve the caller's RNG stream: this optional identity path must
        # not change stochastic training simply by being constructed.
        with torch.random.fork_rng(devices=[]):
            self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.reset_identity()

    def _initial_gamma(self):
        count = self.hops + 1
        if self.init == "identity":
            values = torch.zeros(count)
            values[0] = 1.0
            return values
        if self.init == "uniform":
            return torch.full((count,), 1.0 / count)

        values = [
            self.teleport * ((1.0 - self.teleport) ** k)
            for k in range(self.hops)
        ]
        values.append((1.0 - self.teleport) ** self.hops)
        return torch.tensor(values, dtype=torch.float32)

    def reset_identity(self):
        """Reset only the output projection so the residual is exactly zero."""
        nn.init.zeros_(self.out_proj.weight)

    def normalized_gamma(self):
        """Return signed coefficients with bounded total absolute mass."""
        denominator = self.gamma_raw.abs().sum().clamp_min(self.eps)
        return self.gamma_raw / denominator

    def beta(self):
        cap = self.gate_logit.new_tensor(self.residual_cap)
        return cap * torch.sigmoid(self.gate_logit)

    def _normalize_adjacency(self, sp_adj, batch_size, num_nodes, dtype, device):
        if sp_adj.is_sparse:
            sp_adj = sp_adj.to_dense()
        if sp_adj.dim() == 2:
            sp_adj = sp_adj.unsqueeze(0)
        if sp_adj.dim() != 3:
            raise ValueError("sp_adj must have shape [N,N] or [B,N,N]")
        if sp_adj.shape[-2:] != (num_nodes, num_nodes):
            raise ValueError(
                "sp_adj node dimensions do not match g_base: "
                f"{tuple(sp_adj.shape[-2:])} vs {(num_nodes, num_nodes)}"
            )
        if sp_adj.shape[0] == 1 and batch_size > 1:
            sp_adj = sp_adj.expand(batch_size, -1, -1)
        elif sp_adj.shape[0] != batch_size:
            raise ValueError(
                f"sp_adj batch {sp_adj.shape[0]} does not match {batch_size}"
            )

        weights = sp_adj.to(device=device, dtype=dtype).clamp_min(0.0)
        eye = torch.eye(num_nodes, device=device, dtype=dtype).unsqueeze(0)
        if self.add_self_loops:
            diagonal = torch.diagonal(weights, dim1=-2, dim2=-1)
            missing = (diagonal <= self.eps).to(dtype)
            weights = weights + torch.diag_embed(missing)

        degree = weights.sum(dim=-1, keepdim=True)
        isolated = degree <= self.eps
        if isolated.any():
            weights = weights + eye * isolated.to(dtype)
            degree = weights.sum(dim=-1, keepdim=True)
        return weights / degree.clamp_min(self.eps)

    def forward(self, g_base, sp_adj):
        squeeze_batch = False
        if g_base.dim() == 2:
            g_base = g_base.unsqueeze(0)
            squeeze_batch = True
        if g_base.dim() != 3 or g_base.shape[-1] != self.dim:
            raise ValueError(
                f"g_base must have shape [B,N,{self.dim}] or [N,{self.dim}]"
            )

        gamma = self.normalized_gamma()
        beta = self.beta()
        if sp_adj is None:
            zero = torch.zeros_like(g_base)
            diagnostics = {
                "active": beta.new_zeros(()),
                "beta": beta,
                "gamma": gamma,
                "mixed": zero,
                "residual": zero,
                "last_hop": zero,
            }
            return (
                g_base.squeeze(0) if squeeze_batch else g_base,
                diagnostics,
            )

        batch_size, num_nodes, _ = g_base.shape
        adjacency = self._normalize_adjacency(
            sp_adj,
            batch_size=batch_size,
            num_nodes=num_nodes,
            dtype=g_base.dtype,
            device=g_base.device,
        )

        state = self.input_norm(g_base)
        states = [state]
        for _ in range(self.hops):
            state = torch.bmm(adjacency, state)
            states.append(state)

        mixed = torch.zeros_like(states[0])
        for coefficient, hop_state in zip(gamma, states):
            mixed = mixed + coefficient * hop_state
        residual = beta * torch.tanh(self.out_proj(mixed))
        output = g_base + residual
        diagnostics = {
            "active": beta.new_ones(()),
            "beta": beta,
            "gamma": gamma,
            "mixed": mixed,
            "residual": residual,
            "last_hop": states[-1],
        }
        if squeeze_batch:
            output = output.squeeze(0)
            for key in ("mixed", "residual", "last_hop"):
                diagnostics[key] = diagnostics[key].squeeze(0)
        return output, diagnostics

    def extra_repr(self):
        return (
            f"dim={self.dim}, hops={self.hops}, "
            f"residual_cap={self.residual_cap:g}, init={self.init}, "
            f"teleport={self.teleport:g}, "
            f"add_self_loops={self.add_self_loops}"
        )
