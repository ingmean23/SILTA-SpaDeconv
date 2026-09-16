import torch
import torch.nn as nn


COOPERATIVE_ARCHITECTURES = {"coop_direct", "coop_gaussian"}


class CooperativeFusionDecon(nn.Module):
    def __init__(
        self,
        h_dim,
        g_dim,
        m_dim,
        num_cell_types,
        fusion_dim=256,
        aux_dim=512,
        lambda_g=0.5,
        lambda_x=0.25,
        lambda_m=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.lambda_g = float(lambda_g)
        self.lambda_x = float(lambda_x)
        self.lambda_m = float(lambda_m)

        self.h_projection = nn.Linear(h_dim, self.fusion_dim)
        self.g_projection = nn.Linear(g_dim, self.fusion_dim)
        self.m_projection = nn.Linear(m_dim, self.fusion_dim)
        self.h_norm = nn.LayerNorm(self.fusion_dim)
        self.g_norm = nn.LayerNorm(self.fusion_dim)
        self.m_norm = nn.LayerNorm(self.fusion_dim)
        self.interaction_projection = nn.Linear(
            self.fusion_dim, self.fusion_dim)
        self.fused_norm = nn.LayerNorm(self.fusion_dim)

        self.mu_layer = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.logvar_layer = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.z_norm = nn.LayerNorm(self.fusion_dim)
        self.aux_projection = nn.Linear(self.fusion_dim, aux_dim)
        self.decon_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, num_cell_types),
        )

    @staticmethod
    def _spot_std(value):
        if value.dim() == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        if value.dim() < 2 or value.shape[0] <= 1:
            return value.new_zeros(())
        return value.std(dim=0, unbiased=False).mean()

    def forward(
        self,
        h_mu,
        g_mu,
        m_latent,
        architecture,
        g_override="normal",
    ):
        architecture = str(architecture).lower()
        if architecture not in COOPERATIVE_ARCHITECTURES:
            raise ValueError(
                "cooperative architecture must be coop_direct or "
                "coop_gaussian")

        h0 = self.h_norm(self.h_projection(h_mu))
        g0 = self.g_norm(self.g_projection(g_mu))
        override = str(g_override or "normal").lower()
        if override == "shuffle" and g0.shape[-2] > 1:
            g0 = torch.roll(g0, shifts=1, dims=-2)
        elif override == "zero":
            g0 = torch.zeros_like(g0)
        elif override != "normal":
            raise ValueError(
                "cooperative g override must be normal, shuffle, or zero")

        if override == "zero":
            additive_g = torch.zeros_like(g0)
            interaction_raw = torch.zeros_like(g0)
            interaction = torch.zeros_like(g0)
        else:
            additive_g = self.lambda_g * g0
            interaction_raw = self.interaction_projection(h0 * g0)
            interaction = self.lambda_x * interaction_raw

        m0 = self.m_norm(self.m_projection(m_latent))
        m_residual = self.lambda_m * m0
        fused = self.fused_norm(
            h0 + additive_g + interaction + m_residual)

        if architecture == "coop_gaussian":
            mu = self.mu_layer(fused)
            logvar = self.logvar_layer(fused).clamp(min=-10.0, max=10.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                representation = mu + torch.randn_like(std) * std
            else:
                representation = mu
            representation = self.z_norm(representation)
        else:
            representation = fused
            mu = torch.zeros_like(fused)
            logvar = torch.zeros_like(fused)

        aux_z = self.aux_projection(representation)
        logits = self.decon_head(representation)
        eps = torch.finfo(fused.dtype).eps
        h_std = self._spot_std(h0)
        additive_g_std = self._spot_std(additive_g)
        interaction_std = self._spot_std(interaction)
        m_residual_std = self._spot_std(m_residual)

        return {
            "h0": h0,
            "g0": g0,
            "additive_g": additive_g,
            "interaction_raw": interaction_raw,
            "interaction": interaction,
            "m0": m0,
            "m_residual": m_residual,
            "fused": fused,
            "representation": representation,
            "mu": mu,
            "logvar": logvar,
            "aux_z": aux_z,
            "logits": logits,
            "g_to_h_ratio": additive_g_std / (h_std + eps),
            "interaction_to_g_ratio": (
                interaction_std / (additive_g_std + eps)),
            "m_to_h_ratio": m_residual_std / (h_std + eps),
        }
