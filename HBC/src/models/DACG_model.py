"""
DACG: Diffusion-Augmented Conditional Graph Model
用 Diffusion 去噪替换 ZINB 重构头，保留双分支 GAT 编码器 + 域对抗
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from models.modules.layers import (
    GraphTransformerBlock,
    GumbelSoftmax,
    Scaler,
    GRL,
)
from models.modules.hypergraph import (
    SpatialHypergraphAttentionBlock,
    SpatialHypergraphBlock,
)
from models.modules.bounded_gpr import BoundedGPRResidual
from models.modules.cooperative_fusion import (
    COOPERATIVE_ARCHITECTURES,
    CooperativeFusionDecon,
)
from models.modules.spatial_residual import (
    DECON_ARCHITECTURES,
    SpatialResidualDecon,
    spatial_graph_reconstruction_loss,
)
from models.modules.diffusion import NoiseScheduler, DenoiseMLP, GeneAutoEncoder, compute_bio_constraints
from utils.lossFunctions import LossFunctions


class SpatialAttentionGAT(nn.Module):
    """空间注意力GAT（同 ZACAE）"""
    def __init__(self, input_dim, output_dim, num_heads=4, use_bias=True,
                 spatial_gat_mode="mixed",
                 spatial_expr_bias_weight=0.0,
                 spatial_type_bias_weight=0.0,
                 spatial_attention_eps=1e-8):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dim_per_head = output_dim // num_heads
        self.spatial_gat_mode = str(spatial_gat_mode or "mixed").lower()
        self.spatial_expr_bias_weight = float(spatial_expr_bias_weight)
        self.spatial_type_bias_weight = float(spatial_type_bias_weight)
        self.spatial_attention_eps = float(spatial_attention_eps)
        self.Q = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.K = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.V = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _expand_prior(prior, batch_size):
        if prior is None:
            return None
        if prior.dim() == 2:
            prior = prior.unsqueeze(0)
        if prior.shape[0] == 1 and batch_size > 1:
            prior = prior.expand(batch_size, -1, -1)
        return prior

    def forward(self, h, sp_adj, expr_similarity=None, type_probs=None,
                expr_bias_weight=None, type_bias_weight=None):
        B, N, D = h.shape
        pure_mode = self.spatial_gat_mode in (
            "pure", "spatial", "spatial_only")
        expr_weight = (
            self.spatial_expr_bias_weight
            if expr_bias_weight is None else float(expr_bias_weight))
        type_weight = (
            self.spatial_type_bias_weight
            if type_bias_weight is None else float(type_bias_weight))
        if pure_mode:
            expr_weight = 0.0
            type_weight = 0.0

        sp_adj = self._expand_prior(sp_adj, B)
        expr_similarity = self._expand_prior(expr_similarity, B)
        type_prior = None
        if type_probs is not None:
            if type_probs.dim() == 2:
                type_probs = type_probs.unsqueeze(0)
            if type_probs.shape[0] == 1 and B > 1:
                type_probs = type_probs.expand(B, -1, -1)
            type_prior = torch.matmul(
                type_probs, type_probs.transpose(1, 2))

        Q_h = self.Q(h).view(B, N, self.num_heads, self.dim_per_head)
        K_h = self.K(h).view(B, N, self.num_heads, self.dim_per_head)
        V_h = self.V(h).view(B, N, self.num_heads, self.dim_per_head)
        mask = (sp_adj == 0)
        eps = self.spatial_attention_eps
        heads = []
        for i in range(self.num_heads):
            q, k, v = Q_h[:, :, i], K_h[:, :, i], V_h[:, :, i]
            score = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.dim_per_head)
            if expr_weight != 0.0:
                if expr_similarity is None:
                    raise ValueError(
                        "expr_similarity is required when expression bias is enabled")
                alpha = torch.sigmoid(self.alpha)
                expr_prior = (
                    (1.0 - alpha)
                    + alpha * expr_similarity.to(dtype=score.dtype)
                )
                score = score + expr_weight * torch.log(
                    expr_prior.clamp_min(eps))
            if type_weight != 0.0:
                if type_prior is None:
                    raise ValueError(
                        "type_probs is required when type bias is enabled")
                score = score + type_weight * torch.log(
                    type_prior.to(dtype=score.dtype).clamp_min(eps))
            score = score.masked_fill(mask, float('-inf'))
            attn = F.softmax(score, dim=2)
            attn = torch.nan_to_num(attn, nan=0.0)
            heads.append(torch.matmul(attn, v))
        return F.relu(torch.cat(heads, dim=2))

# === PLACEHOLDER_REST ===


class AttentionResidualMLP(nn.Module):
    """注意力残差MLP（同 ZACAE）"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.Sigmoid(),
        )
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        g = self.gate(x)
        return g * self.mlp(x) + (1 - g) * self.residual(x)


class DomainClassifier(nn.Module):
    """GRL + 域判别器（同 ZACAE）"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            GRL(), nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z):
        return self.net(z)


# === PLACEHOLDER_DACG_MODEL ===


class DACGModel(nn.Module):
    """DACG: Diffusion-Augmented Conditional Graph Model
    双分支编码器 + Gaussian 潜空间 + Diffusion 去噪重构 + 解卷积 + 域对抗

    相比 ZACAE 的改动:
    - ZINB latent → Gaussian (mu, logvar → z)
    - ZINB decoder → DenoiseMLP (diffusion 去噪重构基因表达)
    - 删除 ZINB 相关的 KL 散度，改用标准 Gaussian KL
    - 新增 diffusion denoising loss 替代 ZINB reconstruction loss
    """
    @staticmethod
    def _is_hypergraph_spatial_mode(spatial_gat_mode):
        mode = str(spatial_gat_mode or "mixed").lower()
        return mode in ("hypergraph", "hypergraph_spatial", "hyper", "hg")

    def _make_branch2_spatial_block(self, in_dim, out_dim, num_heads, use_bias):
        if self.spatial_gat_mode in (
                "hypergraph_attention", "hypergraph_attn", "hyper_attn",
                "hgat"):
            return SpatialHypergraphAttentionBlock(
                in_dim, out_dim, num_heads=num_heads, use_bias=use_bias)
        if self._is_hypergraph_spatial_mode(self.spatial_gat_mode):
            return SpatialHypergraphBlock(in_dim, out_dim, use_bias=use_bias)
        return SpatialAttentionGAT(
            in_dim, out_dim, num_heads, use_bias,
            spatial_gat_mode=self.spatial_gat_mode,
            spatial_expr_bias_weight=self.spatial_expr_bias_weight,
            spatial_type_bias_weight=self.spatial_type_bias_weight,
            spatial_attention_eps=self.spatial_attention_eps)

    def __init__(
        self, num_genes, num_cell_types, latent_dim=512,
        encoder_out_channels=[256, 256, 512], num_heads=4,
        use_bias=True, fusion_hidden_dim=256, disc_hidden_dim=256,
        diffusion_T=100, diffusion_hidden=1024,
        use_structured_latent=False, structured_use_c_context=False,
        structured_h_dim=192, structured_g_dim=192,
        structured_mode_dim=64, structured_mode_tau=1.5,
        use_h_stochastic=False, h_logvar_min=-6.0, h_logvar_max=2.0,
        h_inference_use_mu=True,
        use_g_stochastic=False, g_logvar_min=-6.0, g_logvar_max=2.0,
        g_inference_use_mu=True,
        use_zig_latent=False, zig_tau=1.0, zig_pi_init=0.05,
        zig_pi_min=0.01, zig_pi_max=0.4, zig_inference_use_expected=True,
        zig_scale_min=0.5, zig_scale_max=1.5, zig_residual_alpha=0.1,
        use_m_residual=False, m_residual_scale=0.05, m_gate_init=0.0,
        use_mode_prior_head=False, m_fusion_scale=1.0,
        mode_num_classes=None, spatial_gat_mode="mixed",
        spatial_expr_bias_weight=0.0, spatial_type_bias_weight=0.0,
        spatial_attention_eps=1e-8,
        decon_architecture="concat",
        spatial_residual_hidden_dim=128,
        coop_fusion_dim=256, coop_lambda_g=0.5, coop_lambda_x=0.25,
        coop_lambda_m=0.0,
        bounded_gpr=None,
        kl_soft_cap=100.0, h_kl_soft_cap=100.0, g_kl_soft_cap=100.0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.latent_dim = latent_dim
        self.out_channels = encoder_out_channels
        self.use_structured_latent = bool(use_structured_latent)
        self.structured_use_c_context = bool(structured_use_c_context)
        self.structured_h_dim = int(structured_h_dim)
        self.structured_g_dim = int(structured_g_dim)
        self.structured_mode_dim = int(structured_mode_dim)
        self.structured_mode_tau = float(structured_mode_tau)
        self.use_h_stochastic = bool(use_h_stochastic)
        self.h_logvar_min = float(h_logvar_min)
        self.h_logvar_max = float(h_logvar_max)
        self.h_inference_use_mu = bool(h_inference_use_mu)
        self.use_g_stochastic = bool(use_g_stochastic)
        self.g_logvar_min = float(g_logvar_min)
        self.g_logvar_max = float(g_logvar_max)
        self.g_inference_use_mu = bool(g_inference_use_mu)
        self.use_zig_latent = bool(use_zig_latent)
        self.zig_tau = float(zig_tau)
        self.zig_pi_init = float(zig_pi_init)
        self.zig_pi_min = float(zig_pi_min)
        self.zig_pi_max = float(zig_pi_max)
        self.zig_inference_use_expected = bool(zig_inference_use_expected)
        self.zig_scale_min = float(zig_scale_min)
        self.zig_scale_max = float(zig_scale_max)
        self.zig_residual_alpha = float(zig_residual_alpha)
        self.use_m_residual = bool(use_m_residual)
        self.m_residual_scale = float(m_residual_scale)
        self.use_mode_prior_head = bool(use_mode_prior_head)
        self.m_fusion_scale = float(m_fusion_scale)
        self.mode_num_classes = int(mode_num_classes or num_cell_types)
        self.spatial_gat_mode = str(spatial_gat_mode or "mixed").lower()
        self.spatial_expr_bias_weight = float(spatial_expr_bias_weight)
        self.spatial_type_bias_weight = float(spatial_type_bias_weight)
        self.spatial_type_bias_scale = 1.0
        self.spatial_attention_eps = float(spatial_attention_eps)
        self.decon_architecture = str(
            decon_architecture or "concat").lower()
        if self.decon_architecture not in DECON_ARCHITECTURES:
            raise ValueError(
                "decon_architecture must be one of: "
                + ", ".join(sorted(DECON_ARCHITECTURES)))
        if (self.decon_architecture != "concat"
                and not self.use_structured_latent):
            raise ValueError(
                "non-concat decon architectures require "
                "use_structured_latent=True")
        self.spatial_residual_alpha = 0.0
        self.spatial_residual_hidden_dim = int(
            spatial_residual_hidden_dim)
        self.coop_fusion_dim = int(coop_fusion_dim)
        self.coop_lambda_g = float(coop_lambda_g)
        self.coop_lambda_x = float(coop_lambda_x)
        self.coop_lambda_m = float(coop_lambda_m)
        self.cooperative_g_override = "normal"
        self.bounded_gpr_config = {}
        self.use_bounded_gpr = False
        self.bounded_gpr = None
        if self.coop_fusion_dim <= 0:
            raise ValueError("coop_fusion_dim must be positive")
        if (self.coop_lambda_g < 0.0 or self.coop_lambda_x < 0.0
                or self.coop_lambda_m < 0.0):
            raise ValueError(
                "cooperative fusion coefficients must be nonnegative")
        if self.decon_architecture in COOPERATIVE_ARCHITECTURES:
            if self.use_h_stochastic or self.use_g_stochastic:
                raise ValueError(
                    "cooperative architectures require deterministic h and g")
            if self.use_zig_latent:
                raise ValueError(
                    "cooperative architectures require use_zig_latent=False")
            if self.m_fusion_scale != 0.0:
                raise ValueError(
                    "cooperative architectures require m_fusion_scale=0")
            if self.coop_lambda_m > 0.0 and not self.use_mode_prior_head:
                raise ValueError(
                    "coop_lambda_m > 0 requires use_mode_prior_head=True")
        if self.spatial_type_bias_weight != 0.0 and not self.use_mode_prior_head:
            raise ValueError(
                "spatial_type_bias_weight requires use_mode_prior_head=True")
        self.kl_soft_cap = float(kl_soft_cap)
        self.h_kl_soft_cap = float(h_kl_soft_cap)
        self.g_kl_soft_cap = float(g_kl_soft_cap)
        embed_dim = encoder_out_channels[-1]

        # === Branch 1: 表达相似性编码 (GAT) ===
        self.branch1 = nn.ModuleList()
        in_dim = num_genes
        for out_dim in encoder_out_channels:
            self.branch1.append(GraphTransformerBlock(in_dim, out_dim, num_heads, use_bias))
            in_dim = out_dim

        # === Branch 2: 双模式 ===
        self.branch2_spatial = nn.ModuleList()
        self.branch2_normal = nn.ModuleList()
        in_dim = num_genes
        for out_dim in encoder_out_channels:
            self.branch2_spatial.append(
                self._make_branch2_spatial_block(
                    in_dim, out_dim, num_heads, use_bias))
            self.branch2_normal.append(GraphTransformerBlock(in_dim, out_dim, num_heads, use_bias))
            in_dim = out_dim
        self.branch2_beta = nn.Parameter(torch.tensor(1.5))

        # === 融合模块 ===
        self.h_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.structured_h_dim),
            nn.ReLU(),
        )
        self.h_logvar_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.structured_h_dim),
        )
        self.g_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.structured_g_dim),
            nn.ReLU(),
        )
        self.g_logvar_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.structured_g_dim),
        )
        self.mode_embedding = nn.Parameter(
            torch.empty(self.mode_num_classes, self.structured_mode_dim)
        )
        self.register_buffer(
            "mode_index",
            torch.arange(num_cell_types, dtype=torch.long),
            persistent=False,
        )
        self.m_residual_projector = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.structured_mode_dim),
            nn.Tanh(),
        )
        self.m_gate = nn.Parameter(torch.tensor(float(m_gate_init)))

        if self.use_structured_latent:
            fusion_input_dim = (
                self.structured_h_dim + self.structured_g_dim
                + self.structured_mode_dim
            )
            if self.structured_use_c_context:
                fusion_input_dim += num_cell_types
        else:
            fusion_input_dim = 2 * embed_dim + num_cell_types
        self.fusion = AttentionResidualMLP(fusion_input_dim, fusion_hidden_dim, latent_dim)
        self.e1_norm = nn.LayerNorm(embed_dim)
        self.e2_norm = nn.LayerNorm(embed_dim)
        self.fused_norm = nn.LayerNorm(latent_dim)
        self.z_pre_gate_norm = nn.LayerNorm(latent_dim)

        # === Gaussian 潜空间 ===
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
        self.zig_logit_layer = nn.Linear(latent_dim, latent_dim)

        # === GeneAutoEncoder: 基因空间 ↔ 低维隐空间 ===
        self.latent_ae_dim = 64
        self.gene_ae = GeneAutoEncoder(num_genes, self.latent_ae_dim)

        # === Latent Diffusion 去噪头（在 AE 隐空间操作）===
        self.noise_scheduler = NoiseScheduler(T=diffusion_T)
        self.denoise_mlp = DenoiseMLP(
            latent_ae_dim=self.latent_ae_dim, cond_dim=latent_dim,
            time_dim=128, hidden_dim=512,
        )
        self.diffusion_T = diffusion_T

        # === 图重构（保留）===
        # 点积重构，无参数

        # === 解卷积头：输出 logits，不加 Softmax ===
        self.decon_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_cell_types),
        )
        self.cond_predictor = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, num_cell_types),
        )
        self.mode_head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, self.mode_num_classes),
        )
        self.mode_prior_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.mode_num_classes),
        )
        self.m_cls_head = nn.Linear(
            self.structured_mode_dim, self.mode_num_classes)
        self.decon_temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # === 域判别器（保留）===
        self.domain_classifier = DomainClassifier(latent_dim, disc_hidden_dim)
        self.cooperative_decon = CooperativeFusionDecon(
            self.structured_h_dim,
            self.structured_g_dim,
            self.structured_mode_dim,
            num_cell_types,
            fusion_dim=self.coop_fusion_dim,
            aux_dim=latent_dim,
            lambda_g=self.coop_lambda_g,
            lambda_x=self.coop_lambda_x,
            lambda_m=self.coop_lambda_m,
        )

        self.loss_fn = LossFunctions()
        self._init_weights()
        self.spatial_decon = SpatialResidualDecon(
            self.structured_h_dim,
            self.structured_g_dim,
            num_cell_types,
            hidden_dim=self.spatial_residual_hidden_dim,
        )
        for module in self.spatial_decon.modules():
            if isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
        self._init_zig_gate_bias()
        self.configure_bounded_gpr(bounded_gpr)

    # === PLACEHOLDER_METHODS ===

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        init.normal_(self.mode_embedding, mean=0.0, std=0.02)

    def configure_bounded_gpr(self, config=None):
        """Configure the optional GPR path without touching existing modules."""
        self.bounded_gpr_config = dict(config or {})
        self.use_bounded_gpr = bool(
            self.bounded_gpr_config.get("enabled", False))
        if not self.use_bounded_gpr:
            self.bounded_gpr = None
            return

        device = next(self.parameters()).device
        self.bounded_gpr = BoundedGPRResidual(
            dim=self.structured_g_dim,
            hops=int(self.bounded_gpr_config.get("hops", 3)),
            residual_cap=float(self.bounded_gpr_config.get(
                "residual_cap", 0.05)),
            teleport=float(self.bounded_gpr_config.get(
                "teleport", 0.1)),
            init=str(self.bounded_gpr_config.get("init", "ppr")),
            gate_init=float(self.bounded_gpr_config.get(
                "gate_init", 0.0)),
            add_self_loops=bool(self.bounded_gpr_config.get(
                "add_self_loops", True)),
            eps=float(self.bounded_gpr_config.get("eps", 1e-8)),
        ).to(device)

    def _init_zig_gate_bias(self):
        """Initialize residual ZIG scale to the identity transform."""
        if not hasattr(self, "zig_logit_layer"):
            return
        init.constant_(self.zig_logit_layer.weight, 0.0)
        if self.zig_logit_layer.bias is not None:
            init.constant_(self.zig_logit_layer.bias, 0.0)

    def set_mode_index(self, mode_index, mode_num_classes=None):
        """Map full decon cell types to mode classes."""
        device = next(self.parameters()).device
        idx = torch.as_tensor(mode_index, dtype=torch.long, device=device)
        if idx.numel() != self.num_cell_types:
            raise ValueError(
                f"mode_index length {idx.numel()} != num_cell_types {self.num_cell_types}")
        num_classes = int(mode_num_classes or (int(idx.max().item()) + 1))
        if num_classes != self.mode_num_classes:
            raise ValueError(
                f"mode_num_classes {num_classes} != model.mode_num_classes {self.mode_num_classes}")
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= self.mode_num_classes:
            raise ValueError("mode_index contains class ids outside mode_num_classes")
        self.mode_index = idx

    def aggregate_mode_target(self, c_true):
        """Aggregate full cell-type proportions into mode classes."""
        if c_true is None:
            return None
        if c_true.shape[-1] == self.mode_num_classes:
            return c_true
        if c_true.shape[-1] != self.num_cell_types:
            raise ValueError(
                f"Expected last dim {self.num_cell_types} or {self.mode_num_classes}, "
                f"got {c_true.shape[-1]}")
        idx = self.mode_index.to(device=c_true.device)
        idx = idx.view(*([1] * (c_true.dim() - 1)), idx.numel())
        idx = idx.expand(*c_true.shape[:-1], idx.shape[-1])
        out = torch.zeros(
            *c_true.shape[:-1], self.mode_num_classes,
            device=c_true.device, dtype=c_true.dtype)
        return out.scatter_add(-1, idx, c_true)

    def _soft_cap_loss(self, value, cap):
        cap_value = torch.as_tensor(
            float(cap), device=value.device, dtype=value.dtype)
        return torch.where(
            value <= cap_value,
            value,
            cap_value + torch.log1p(value - cap_value),
        )

    def _encode_branch1(self, x, ex_adj):
        h = x
        for block in self.branch1:
            h = block(h, ex_adj)
        return h

    def _encode_branch2(self, x, sp_adj=None, ex_adj=None,
                        expr_similarity=None, type_probs=None):
        h = x
        if sp_adj is not None:
            for block in self.branch2_spatial:
                if isinstance(block, SpatialAttentionGAT):
                    h = block(
                        h, sp_adj, expr_similarity, type_probs,
                        expr_bias_weight=self.spatial_expr_bias_weight,
                        type_bias_weight=(
                            self.spatial_type_bias_weight
                            * float(self.spatial_type_bias_scale)),
                    )
                else:
                    h = block(h, sp_adj, ex_adj)
        else:
            beta = torch.sigmoid(self.branch2_beta)
            N = ex_adj.shape[-1]
            I = torch.eye(N, device=ex_adj.device)
            regularized_adj = beta * ex_adj + (1 - beta) * I
            for block in self.branch2_normal:
                h = block(h, regularized_adj)
        return h

    def _expression_cosine_similarity(self, x):
        x_clean = torch.nan_to_num(
            x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = F.normalize(
            x_clean, p=2, dim=-1, eps=self.spatial_attention_eps)
        similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))
        return similarity.clamp(min=0.0, max=1.0).to(dtype=x.dtype)

    def _reparameterize(self, mu, logvar):
        """Gaussian 重参数化"""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _reparameterize_h(self, h_mu, h_logvar):
        if not self.use_h_stochastic:
            return h_mu
        if not self.training and self.h_inference_use_mu:
            return h_mu
        std = torch.exp(0.5 * h_logvar)
        eps = torch.randn_like(std)
        return h_mu + eps * std

    def _reparameterize_g(self, g_mu, g_logvar):
        if not self.use_g_stochastic:
            return g_mu
        if not self.training and self.g_inference_use_mu:
            return g_mu
        std = torch.exp(0.5 * g_logvar)
        eps = torch.randn_like(std)
        return g_mu + eps * std

    def _zero_inflated_gaussian(self, z_pre_gate, fused):
        if not self.use_zig_latent:
            zeros = torch.zeros_like(z_pre_gate)
            ones = torch.ones_like(z_pre_gate)
            return z_pre_gate, zeros, ones, ones, z_pre_gate

        logits = self.zig_logit_layer(fused)
        tau = max(float(self.zig_tau), 1e-6)
        alpha = min(max(float(self.zig_residual_alpha), 0.0), 1.0)
        scale = 1.0 + alpha * torch.tanh(logits / tau)
        scale = scale.clamp(
            min=float(self.zig_scale_min),
            max=float(self.zig_scale_max))
        z = scale * z_pre_gate
        return z, logits, scale, scale, z_pre_gate

    def _reconstruct_graph(self, z):
        z_sq = torch.squeeze(z)
        return torch.sigmoid(torch.mm(z_sq, z_sq.t()))

    def forward(self, x, adj, c=None, mode="sc"):
        """
        Args:
            x: (1, N, num_genes)
            adj: ex_adj (mode=sc) 或 [ex_adj, sp_adj] (mode=st)
            c: (1, N, num_cell_types) 条件，None 则自动预测
            mode: "sc" 或 "st"
        """
        mode_logits = None
        mode_probs = None
        sp_adj = None
        if mode == "sc":
            ex_adj = adj
            E1 = self.e1_norm(self._encode_branch1(x, ex_adj))
            E2 = self.e2_norm(self._encode_branch2(x, sp_adj=None, ex_adj=ex_adj))
            cond_input = torch.cat([E1, E2], dim=-1)
            if c is None:
                c_logits = self.cond_predictor(cond_input)
                c = F.softmax(c_logits / self.decon_temp.clamp(min=1.0, max=5.0), dim=-1)
            else:
                c_logits = None
        else:
            if len(adj) == 3:
                ex_adj_for_branch1, sp_adj, ex_adj_for_spatial = adj
            else:
                ex_adj_for_branch1, sp_adj = adj[0], adj[1]
                ex_adj_for_spatial = ex_adj_for_branch1
            E1 = self.e1_norm(self._encode_branch1(x, ex_adj_for_branch1))
            if self.use_mode_prior_head:
                mode_logits = self.mode_prior_head(E1)
                mode_tau = max(self.structured_mode_tau, 1e-6)
                mode_probs = F.softmax(mode_logits / mode_tau, dim=-1)
                spatial_type_probs = mode_probs.detach()
            else:
                spatial_type_probs = None
            expr_similarity = (
                self._expression_cosine_similarity(x)
                if self.spatial_expr_bias_weight != 0.0 else None)
            E2 = self.e2_norm(self._encode_branch2(
                x, sp_adj=sp_adj, ex_adj=ex_adj_for_spatial,
                expr_similarity=expr_similarity,
                type_probs=spatial_type_probs))
            cond_input = torch.cat([E1, E2], dim=-1)
            c_logits = self.cond_predictor(cond_input)
            c = F.softmax(c_logits / self.decon_temp.clamp(min=1.0, max=5.0), dim=-1).detach()

        if mode_logits is None:
            if self.use_mode_prior_head:
                mode_logits = self.mode_prior_head(E1)
            else:
                mode_logits = self.mode_head(cond_input)
        if mode_probs is None:
            mode_tau = max(self.structured_mode_tau, 1e-6)
            mode_probs = F.softmax(mode_logits / mode_tau, dim=-1)

        # 融合
        h_mu = self.h_projector(E1)
        h_logvar = torch.clamp(
            self.h_logvar_projector(E1),
            min=self.h_logvar_min,
            max=self.h_logvar_max,
        )
        h_latent = self._reparameterize_h(h_mu, h_logvar)
        g_mu_base = self.g_projector(E2)
        if self.bounded_gpr is not None:
            g_mu, gpr_diag = self.bounded_gpr(
                g_mu_base, sp_adj if mode == "st" else None)
        else:
            g_mu = g_mu_base
            gpr_zero = torch.zeros_like(g_mu_base)
            gpr_diag = {
                "active": g_mu_base.new_zeros(()),
                "beta": g_mu_base.new_zeros(()),
                "gamma": g_mu_base.new_empty((0,)),
                "mixed": gpr_zero,
                "residual": gpr_zero,
                "last_hop": gpr_zero,
            }
        g_logvar = torch.clamp(
            self.g_logvar_projector(E2),
            min=self.g_logvar_min,
            max=self.g_logvar_max,
        )
        g_latent = self._reparameterize_g(g_mu, g_logvar)
        m_from_mode = torch.matmul(mode_probs, self.mode_embedding)
        if self.use_m_residual:
            m_from_expr = self.m_residual_scale * self.m_residual_projector(E1)
            m_gate = torch.sigmoid(self.m_gate)
            m_latent = m_gate * m_from_mode + (1.0 - m_gate) * m_from_expr
        else:
            m_from_expr = torch.zeros_like(m_from_mode)
            m_gate = torch.ones((), device=m_from_mode.device,
                                dtype=m_from_mode.dtype)
            m_latent = m_from_mode
        m_fusion_scale = torch.as_tensor(
            float(self.m_fusion_scale),
            device=m_latent.device,
            dtype=m_latent.dtype)
        m_for_fusion = m_fusion_scale * m_latent
        m_logits = self.m_cls_head(m_latent)
        cooperative_output = None
        if self.decon_architecture in COOPERATIVE_ARCHITECTURES:
            self.cooperative_decon.lambda_g = float(self.coop_lambda_g)
            self.cooperative_decon.lambda_x = float(self.coop_lambda_x)
            self.cooperative_decon.lambda_m = float(self.coop_lambda_m)
            cooperative_output = self.cooperative_decon(
                h_mu,
                g_mu,
                m_latent,
                self.decon_architecture,
                g_override=self.cooperative_g_override,
            )
            fused = cooperative_output["fused"]
            mu = cooperative_output["mu"]
            logvar = cooperative_output["logvar"]
            z = cooperative_output["aux_z"]
            z_pre_gate = z
            zig_logits = torch.zeros_like(z)
            zig_scale = torch.ones_like(z)
            zig_residual_scale = torch.ones_like(z)
            zig_gate = zig_residual_scale
            zig_keep = zig_residual_scale
        else:
            if self.use_structured_latent:
                if self.decon_architecture in (
                        "content_only", "direct_h", "spatial_residual"):
                    g_for_fusion = torch.zeros_like(g_latent)
                else:
                    g_for_fusion = g_latent
                fusion_parts = [h_latent, g_for_fusion, m_for_fusion]
                if self.structured_use_c_context:
                    fusion_parts.append(c)
                fused = self.fused_norm(
                    self.fusion(torch.cat(fusion_parts, dim=-1)))
            else:
                fused = self.fused_norm(
                    self.fusion(torch.cat([E1, E2, c], dim=-1)))

            # Gaussian latent, optionally passed through a zero-inflated gate.
            mu = self.mu_layer(fused)
            logvar = self.logvar_layer(fused)
            logvar = torch.clamp(logvar, min=-10, max=10)
            z_pre_gate = self.z_pre_gate_norm(
                self._reparameterize(mu, logvar))
            z, zig_logits, zig_scale, zig_residual_scale, z_pre_gate = (
                self._zero_inflated_gaussian(z_pre_gate, fused))
            zig_gate = zig_residual_scale
            zig_keep = zig_residual_scale

        # === Latent Diffusion: 在 AE 隐空间操作 ===
        device = x.device
        self.noise_scheduler.to(device)
        B = x.shape[0]
        t = torch.randint(0, self.diffusion_T, (B,), device=device)

        # 1. 用 AE encoder 把基因表达压缩到隐空间
        x_sq = x.squeeze(0) if x.dim() == 3 and x.shape[0] == 1 else x
        x_latent = self.gene_ae.encode(x_sq)  # (N, latent_ae_dim=64)

        # 2. 在隐空间加噪
        x_latent_t, noise = self.noise_scheduler.add_noise(x_latent, t)

        # 3. 用 graph encoder 的 z 作为条件，预测隐空间噪声
        if x_latent_t.dim() == 2:
            x_latent_t = x_latent_t.unsqueeze(0)
        z_cond = z  # (1, N, latent_dim=512)
        eps_pred = self.denoise_mlp(x_latent_t, z_cond, t)

        # 4. AE 重构（用于 AE 预训练损失）
        x_ae_recon = self.gene_ae.decode(x_latent)

        # 其他 decoder heads
        recon_graph = self._reconstruct_graph(z)
        temperature = self.decon_temp.clamp(min=1.0, max=5.0)
        fused_decon_logits = self.decon_head(z)
        base_decon_logits = fused_decon_logits
        spatial_delta_logits = torch.zeros_like(fused_decon_logits)
        spatial_delta_applied = torch.zeros_like(fused_decon_logits)
        dual_h_logits = torch.zeros_like(fused_decon_logits)
        dual_g_logits = torch.zeros_like(fused_decon_logits)
        if cooperative_output is not None:
            fused_decon_logits = cooperative_output["logits"]
            base_decon_logits = fused_decon_logits
            spatial_delta_logits = torch.zeros_like(fused_decon_logits)
            spatial_delta_applied = torch.zeros_like(fused_decon_logits)
            dual_h_logits = torch.zeros_like(fused_decon_logits)
            dual_g_logits = torch.zeros_like(fused_decon_logits)
            decon_logits = fused_decon_logits
            decon = F.softmax(decon_logits / temperature, dim=-1)
        elif self.decon_architecture == "direct_h":
            base_decon_logits = self.spatial_decon.direct_h_logits(h_mu)
            decon_logits = base_decon_logits
            decon = F.softmax(decon_logits / temperature, dim=-1)
        elif self.decon_architecture == "spatial_residual":
            spatial_delta_logits = self.spatial_decon.delta_logits(g_mu)
            spatial_delta_applied = (
                float(self.spatial_residual_alpha)
                * torch.tanh(spatial_delta_logits)
            )
            decon_logits = base_decon_logits + spatial_delta_applied
            decon = F.softmax(decon_logits / temperature, dim=-1)
        elif self.decon_architecture == "dual_head":
            dual_h_logits, dual_g_logits = self.spatial_decon.dual_logits(
                h_mu, g_mu)
            decon = 0.5 * (
                F.softmax(dual_h_logits / temperature, dim=-1)
                + F.softmax(dual_g_logits / temperature, dim=-1)
            )
            decon_logits = temperature * torch.log(decon.clamp_min(1e-8))
            base_decon_logits = dual_h_logits
        else:
            decon_logits = fused_decon_logits
            decon = F.softmax(decon_logits / temperature, dim=-1)
        domain_logits = self.domain_classifier(z)

        result = {
            "z": z, "mu": mu, "logvar": logvar,
            "z_pre_gate": z_pre_gate,
            "zig_logits": zig_logits, "zig_pi": zig_scale,
            "zig_scale": zig_scale,
            "zig_residual_scale": zig_residual_scale,
            "zig_gate": zig_gate, "zig_keep": zig_keep,
            "E1": E1, "E2": E2, "fused": fused,
            "h_latent": h_latent, "h_mu": h_mu, "h_logvar": h_logvar,
            "g_latent": g_latent, "g_mu": g_mu, "g_logvar": g_logvar,
            "g_mu_base": g_mu_base,
            "gpr_active": gpr_diag["active"],
            "gpr_beta": gpr_diag["beta"],
            "gpr_gamma": gpr_diag["gamma"],
            "gpr_mixed": gpr_diag["mixed"],
            "gpr_residual": gpr_diag["residual"],
            "gpr_last_hop": gpr_diag["last_hop"],
            "m_latent": m_latent,
            "m_latent_before_scale": m_latent,
            "m_for_fusion": m_for_fusion,
            "m_from_mode": m_from_mode, "m_from_expr": m_from_expr,
            "m_gate": m_gate,
            "m_fusion_scale": m_fusion_scale,
            "spatial_expr_bias_weight": torch.as_tensor(
                self.spatial_expr_bias_weight, device=x.device),
            "spatial_type_bias_weight": torch.as_tensor(
                self.spatial_type_bias_weight * self.spatial_type_bias_scale,
                device=x.device),
            "m_logits": m_logits,
            "c": c, "c_logits": c_logits,
            "mode_logits": mode_logits, "mode_probs": mode_probs,
            "x_latent": x_latent, "x_latent_t": x_latent_t,
            "eps_pred": eps_pred, "eps_true": noise,
            "x_ae_recon": x_ae_recon, "x_original": x_sq,
            "reconstruct_graph": recon_graph,
            "decon": decon, "decon_logits": decon_logits,
            "base_decon_logits": base_decon_logits,
            "spatial_delta_logits": spatial_delta_logits,
            "spatial_delta_applied": spatial_delta_applied,
            "spatial_residual_alpha": torch.as_tensor(
                float(self.spatial_residual_alpha),
                device=x.device, dtype=x.dtype),
            "dual_h_logits": dual_h_logits,
            "dual_g_logits": dual_g_logits,
            "domain_logits": domain_logits,
        }
        if cooperative_output is not None:
            result.update({
                "coop_h0": cooperative_output["h0"],
                "coop_g0": cooperative_output["g0"],
                "coop_additive_g": cooperative_output["additive_g"],
                "coop_interaction_raw": cooperative_output["interaction_raw"],
                "coop_interaction": cooperative_output["interaction"],
                "coop_m0": cooperative_output["m0"],
                "coop_m_residual": cooperative_output["m_residual"],
                "coop_fused": cooperative_output["fused"],
                "coop_repr": cooperative_output["representation"],
                "coop_aux_z": cooperative_output["aux_z"],
                "coop_g_to_h_ratio": cooperative_output["g_to_h_ratio"],
                "coop_interaction_to_g_ratio": cooperative_output[
                    "interaction_to_g_ratio"],
                "coop_m_to_h_ratio": cooperative_output["m_to_h_ratio"],
            })
        return result

    def forward_mode_prior(self, x, ex_adj):
        """Lightweight E1-only forward for pseudo mode-prior pretraining."""
        E1 = self.e1_norm(self._encode_branch1(x, ex_adj))
        mode_logits = self.mode_prior_head(E1)
        mode_tau = max(self.structured_mode_tau, 1e-6)
        mode_probs = F.softmax(mode_logits / mode_tau, dim=-1)
        m_latent = torch.matmul(mode_probs, self.mode_embedding)
        m_logits = self.m_cls_head(m_latent)
        return {
            "E1": E1,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "m_latent": m_latent,
            "m_latent_before_scale": m_latent,
            "m_from_mode": m_latent,
            "m_logits": m_logits,
        }

    @torch.no_grad()
    def reconstruct_genes(self, z, num_steps=20):
        """推理时: 隐空间 DDIM 去噪 → AE decode → 基因表达"""
        device = z.device
        self.noise_scheduler.to(device)
        B, N, _ = z.shape
        # 在 AE 隐空间初始化噪声（64 维，不是 gene_dim）
        x_t = torch.randn(B, N, self.latent_ae_dim, device=device)

        step_size = max(self.diffusion_T // num_steps, 1)
        timesteps = list(range(self.diffusion_T - 1, -1, -step_size))

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            eps_pred = self.denoise_mlp(x_t, z, t)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            x_t = self.noise_scheduler.ddim_step(x_t, eps_pred, t, t_prev)

        # AE decode: 隐空间 → 基因空间
        x_gene = self.gene_ae.decode(x_t)
        return F.relu(x_gene)

        return F.relu(x_t)  # 基因表达非负

    def compute_loss(self, x, adj, output, c_true=None, mode="sc",
                     w_diff=1.0, w_graph=1.0, w_kl=0.1, w_decon=1000.0,
                     w_adv=1.0, w_entropy=0.0, p_global=None, w_sample=1.0,
                     w_ae=1.0, w_bio=0.1, w_cond_pred=0.0, w_mode_ce=0.0,
                     w_mode_soft_kl=0.0,
                     w_m_ce=0.0, w_m_kl_st=0.0,
                     w_mode_global_kl=0.0,
                     w_mode_consistency=0.0,
                     w_h_kl=0.0, w_g_kl=0.0, w_proto=0.0, w_zig_kl=0.0,
                     w_g_graph=0.0, g_graph_target="spatial",
                     g_graph_temperature=0.5, g_graph_negative_ratio=5,
                     g_graph_seed=0):
        """计算所有损失（含 AE 重构 + 生物约束）"""
        # 1. Diffusion denoising loss（在 AE 隐空间）
        diff_loss = F.mse_loss(output["eps_pred"].squeeze(), output["eps_true"].squeeze())

        # 2. 图重构损失
        ex_adj = adj if mode == "sc" else (adj[2] if len(adj) == 3 else adj[0])
        ex_adj = adj if mode == "sc" else (adj[2] if len(adj) == 3 else adj[0])
        graph_recon = self.loss_fn.reconstruction_graph_loss(ex_adj, output["reconstruct_graph"])
        g_graph_loss = torch.tensor(0.0, device=x.device)
        g_graph_diag = {
            "positive_score_mean": 0.0,
            "negative_score_mean": 0.0,
            "positive_edges": 0,
            "negative_edges": 0,
        }
        if mode == "st" and float(w_g_graph) > 0.0:
            graph_target = str(g_graph_target or "spatial").lower()
            if graph_target == "spatial":
                target_adj = adj[1]
            elif graph_target == "expression":
                target_adj = ex_adj
            else:
                raise ValueError(
                    "g_graph_target must be 'spatial' or 'expression'")
            graph_generator = torch.Generator(
                device=output["g_mu"].device)
            graph_generator.manual_seed(int(g_graph_seed))
            g_graph_loss, g_graph_diag = spatial_graph_reconstruction_loss(
                output["g_mu"],
                target_adj,
                temperature=g_graph_temperature,
                negative_ratio=g_graph_negative_ratio,
                generator=graph_generator,
            )

        # 3. Gaussian KL 散度
        mu, logvar = output["mu"], output["logvar"]
        kl_raw = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = self._soft_cap_loss(kl_raw, self.kl_soft_cap)

        h_kl_raw = torch.tensor(0.0, device=x.device)
        h_kl_loss = torch.tensor(0.0, device=x.device)
        if self.use_h_stochastic and "h_mu" in output and "h_logvar" in output:
            h_mu = output["h_mu"]
            h_logvar = output["h_logvar"]
            h_kl_raw = -0.5 * torch.mean(
                1 + h_logvar - h_mu.pow(2) - h_logvar.exp())
            h_kl_loss = self._soft_cap_loss(h_kl_raw, self.h_kl_soft_cap)

        g_kl_raw = torch.tensor(0.0, device=x.device)
        g_kl_loss = torch.tensor(0.0, device=x.device)
        if self.use_g_stochastic and "g_mu" in output and "g_logvar" in output:
            g_mu = output["g_mu"]
            g_logvar = output["g_logvar"]
            g_kl_raw = -0.5 * torch.mean(
                1 + g_logvar - g_mu.pow(2) - g_logvar.exp())
            g_kl_loss = self._soft_cap_loss(g_kl_raw, self.g_kl_soft_cap)

        zig_kl_loss = torch.tensor(0.0, device=x.device)
        if self.use_zig_latent and "zig_residual_scale" in output:
            residual_scale = output["zig_residual_scale"]
            zig_kl_loss = ((residual_scale - 1.0) ** 2).mean()
            zig_kl_loss = zig_kl_loss.clamp(max=500.0)

        # 4. 解卷积损失
        decon_loss = torch.tensor(0.0, device=x.device)
        decon_loss = torch.tensor(0.0, device=x.device)
        cond_pred_loss = torch.tensor(0.0, device=x.device)
        mode_ce_loss = torch.tensor(0.0, device=x.device)
        mode_soft_kl_loss = torch.tensor(0.0, device=x.device)
        m_ce_loss = torch.tensor(0.0, device=x.device)
        m_kl_st_loss = torch.tensor(0.0, device=x.device)
        mode_global_kl_loss = torch.tensor(0.0, device=x.device)
        mode_consistency_loss = torch.tensor(0.0, device=x.device)
        proto_loss = torch.tensor(0.0, device=x.device)
        if c_true is not None:
            decon_logits = output["decon_logits"].reshape(-1, output["decon_logits"].shape[-1])
            c_true_sq = c_true.reshape(-1, c_true.shape[-1])
            log_pred = F.log_softmax(decon_logits / self.decon_temp.clamp(min=1.0, max=5.0), dim=-1)
            decon_loss = F.kl_div(log_pred, c_true_sq, reduction='batchmean')
            mode_target_soft = self.aggregate_mode_target(c_true).reshape(
                -1, self.mode_num_classes)
            mode_target_soft = mode_target_soft.clamp(min=0.0)
            mode_target_soft = mode_target_soft / mode_target_soft.sum(
                dim=-1, keepdim=True).clamp(min=1e-8)
            mode_target = torch.argmax(mode_target_soft, dim=-1)
            c_logits = output.get("c_logits")
            if c_logits is not None:
                cond_logits = c_logits.reshape(-1, c_logits.shape[-1])
                log_cond = F.log_softmax(cond_logits / self.decon_temp.clamp(min=1.0, max=5.0), dim=-1)
                cond_pred_loss = F.kl_div(log_cond, c_true_sq, reduction='batchmean')
            mode_logits = output.get("mode_logits")
            if mode_logits is not None:
                mode_logits_sq = mode_logits.reshape(-1, mode_logits.shape[-1])
                mode_ce_loss = F.cross_entropy(mode_logits_sq, mode_target)
                log_mode = F.log_softmax(
                    mode_logits_sq / max(self.structured_mode_tau, 1e-6),
                    dim=-1)
                mode_soft_kl_loss = F.kl_div(
                    log_mode, mode_target_soft, reduction='batchmean')
                m_latent = output.get("m_latent")
                if m_latent is not None:
                    m_sq = m_latent.reshape(-1, m_latent.shape[-1])
                    target_proto = self.mode_embedding[mode_target]
                    proto_loss = (
                        1.0 - F.cosine_similarity(m_sq, target_proto, dim=-1)
                    ).mean()
            m_logits = output.get("m_logits")
            if m_logits is not None:
                m_logits_sq = m_logits.reshape(-1, m_logits.shape[-1])
                m_ce_loss = F.cross_entropy(m_logits_sq, mode_target)
        else:
            m_logits = output.get("m_logits")
            if m_logits is not None and p_global is not None:
                m_logits_sq = m_logits.reshape(-1, m_logits.shape[-1])
                target = self.aggregate_mode_target(
                    p_global.to(device=x.device, dtype=m_logits_sq.dtype)
                ).flatten()
                if target.numel() == m_logits_sq.shape[-1] and torch.sum(target) > 0:
                    target = target.clamp(min=1e-8)
                    target = target / target.sum().clamp(min=1e-8)
                    target = target.unsqueeze(0).expand_as(m_logits_sq)
                    log_m = F.log_softmax(
                        m_logits_sq / max(self.structured_mode_tau, 1e-6),
                        dim=-1)
                    m_kl_st_loss = F.kl_div(
                        log_m, target, reduction='batchmean')
            mode_probs = output.get("mode_probs")
            if mode_probs is not None and p_global is not None:
                mode_sq = mode_probs.reshape(-1, mode_probs.shape[-1])
                target = self.aggregate_mode_target(
                    p_global.to(device=x.device, dtype=mode_sq.dtype)
                ).flatten()
                if target.numel() == mode_sq.shape[-1] and torch.sum(target) > 0:
                    target = target.clamp(min=1e-8)
                    target = target / target.sum().clamp(min=1e-8)
                    mode_mean = mode_sq.mean(dim=0).clamp(min=1e-8)
                    mode_mean = mode_mean / mode_mean.sum().clamp(min=1e-8)
                    mode_global_kl_loss = F.kl_div(
                        mode_mean.log().unsqueeze(0),
                        target.unsqueeze(0),
                        reduction='batchmean')
            if w_mode_consistency > 0:
                mode_probs = output.get("mode_probs")
                decon = output.get("decon")
                if mode_probs is not None and decon is not None:
                    decon_mode = self.aggregate_mode_target(decon)
                    decon_mode = decon_mode.clamp(min=1e-8)
                    decon_mode = decon_mode / decon_mode.sum(
                        dim=-1, keepdim=True).clamp(min=1e-8)
                    target_mode = mode_probs.detach().clamp(min=1e-8)
                    target_mode = target_mode / target_mode.sum(
                        dim=-1, keepdim=True).clamp(min=1e-8)
                    mode_consistency_loss = F.kl_div(
                        decon_mode.reshape(-1, decon_mode.shape[-1]).log(),
                        target_mode.reshape(-1, target_mode.shape[-1]),
                        reduction='batchmean')

        # 5. 域对抗损失
        domain_logits = output["domain_logits"].squeeze()
        domain_logits = output["domain_logits"].squeeze()
        domain_label = 0 if mode == "sc" else 1
        domain_target = torch.full((domain_logits.shape[0],), domain_label,
                                   dtype=torch.long, device=x.device)
        domain_loss = F.cross_entropy(domain_logits, domain_target)
        domain_loss = domain_loss.clamp(max=50.0)

        # 6. 熵正则化
        entropy_loss = self.loss_fn.self_entropy(output["decon"])

        # 7. Sample-level 全局对齐
        sample_loss = torch.tensor(0.0, device=x.device)
        if p_global is not None:
            decon_sq = output["decon"].squeeze(0)
            if decon_sq.dim() == 1:
                decon_sq = decon_sq.unsqueeze(0)
            sample_loss = ((decon_sq.mean(dim=0) - p_global) ** 2).sum()

        # 8. AE 重构损失（让 autoencoder 学好基因空间压缩）
        ae_loss = F.mse_loss(output["x_ae_recon"].squeeze(), output["x_original"].squeeze())

        # 9. 生物约束损失（UMI + 零表达 + 分布）
        loss_umi, loss_zero, loss_rec = compute_bio_constraints(
            output["x_ae_recon"].squeeze(), output["x_original"].squeeze())
        ae_loss = F.mse_loss(output["x_ae_recon"].squeeze(), output["x_original"].squeeze())
        loss_umi, loss_zero, loss_rec = compute_bio_constraints(
            output["x_ae_recon"].squeeze(), output["x_original"].squeeze())
        bio_loss = loss_umi + loss_zero

        total = (w_diff * diff_loss + w_graph * graph_recon + w_kl * kl_loss
                 + w_decon * decon_loss + w_adv * domain_loss
                 + w_entropy * entropy_loss + w_sample * sample_loss
                 + w_ae * ae_loss + w_bio * bio_loss
                 + w_cond_pred * cond_pred_loss + w_mode_ce * mode_ce_loss
                 + w_mode_soft_kl * mode_soft_kl_loss
                 + w_m_ce * m_ce_loss + w_m_kl_st * m_kl_st_loss
                 + w_mode_global_kl * mode_global_kl_loss
                 + w_mode_consistency * mode_consistency_loss
                 + w_h_kl * h_kl_loss + w_g_kl * g_kl_loss
                 + w_proto * proto_loss + w_zig_kl * zig_kl_loss
                 + w_g_graph * g_graph_loss)

        return {
            "total": total,
            "diffusion": diff_loss,
            "graph_recon": graph_recon,
            "kl": kl_loss,
            "kl_raw": kl_raw.detach(),
            "kl_soft": kl_loss.detach(),
            "decon": decon_loss,
            "domain": domain_loss,
            "entropy": entropy_loss,
            "sample": sample_loss,
            "ae_recon": ae_loss,
            "bio": bio_loss,
            "cond_pred": cond_pred_loss,
            "mode_ce": mode_ce_loss,
            "mode_soft_kl": mode_soft_kl_loss,
            "m_ce": m_ce_loss,
            "m_kl_st": m_kl_st_loss,
            "mode_global_kl": mode_global_kl_loss,
            "mode_consistency": mode_consistency_loss,
            "proto": proto_loss,
            "h_kl": h_kl_loss,
            "h_kl_raw": h_kl_raw.detach(),
            "h_kl_soft": h_kl_loss.detach(),
            "g_kl": g_kl_loss,
            "g_kl_raw": g_kl_raw.detach(),
            "g_kl_soft": g_kl_loss.detach(),
            "g_graph": g_graph_loss,
            "g_graph_positive_score": torch.as_tensor(
                g_graph_diag["positive_score_mean"], device=x.device),
            "g_graph_negative_score": torch.as_tensor(
                g_graph_diag["negative_score_mean"], device=x.device),
            "g_graph_positive_edges": torch.as_tensor(
                g_graph_diag["positive_edges"], device=x.device),
            "g_graph_negative_edges": torch.as_tensor(
                g_graph_diag["negative_edges"], device=x.device),
            "zig_kl": zig_kl_loss,
        }

    def rebuild_io_layers(self, num_genes, num_cell_types,
                          mode_num_classes=None, spatial_gat_mode=None,
                          bounded_gpr=None):
        """顺序迁移训练：重建数据集相关层"""
        device = next(self.parameters()).device
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.mode_num_classes = int(mode_num_classes or num_cell_types)
        if spatial_gat_mode is not None:
            self.spatial_gat_mode = str(spatial_gat_mode or "mixed").lower()
        self.mode_index = torch.arange(
            num_cell_types, dtype=torch.long, device=device)
        embed_dim = self.out_channels[-1]
        first_out = self.out_channels[0]
        num_heads = 4

        # 重建 encoder 第一层
        self.branch1[0] = GraphTransformerBlock(num_genes, first_out, num_heads, True).to(device)
        self.branch2_spatial[0] = self._make_branch2_spatial_block(
            num_genes, first_out, num_heads, True).to(device)
        self.branch2_normal[0] = GraphTransformerBlock(num_genes, first_out, num_heads, True).to(device)

        # 重建 fusion
        self.mode_embedding = nn.Parameter(
            torch.empty(self.mode_num_classes, self.structured_mode_dim, device=device)
        )
        if self.use_structured_latent:
            fusion_input_dim = (
                self.structured_h_dim + self.structured_g_dim
                + self.structured_mode_dim
            )
            if self.structured_use_c_context:
                fusion_input_dim += num_cell_types
        else:
            fusion_input_dim = 2 * embed_dim + num_cell_types
        self.fusion = AttentionResidualMLP(fusion_input_dim, 256, self.latent_dim).to(device)

        # 重建 GeneAutoEncoder（基因维度变了）
        self.gene_ae = GeneAutoEncoder(num_genes, self.latent_ae_dim).to(device)

        # 重建 latent diffusion decoder（隐空间维度不变，不需要重建）
        # self.denoise_mlp 的输入是 latent_ae_dim + latent_dim，都不依赖 num_genes

        # 重建 decon_head 和 cond_predictor
        self.decon_head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_cell_types),
        ).to(device)
        self.spatial_decon = SpatialResidualDecon(
            self.structured_h_dim,
            self.structured_g_dim,
            num_cell_types,
            hidden_dim=self.spatial_residual_hidden_dim,
        ).to(device)
        self.cooperative_decon = CooperativeFusionDecon(
            self.structured_h_dim,
            self.structured_g_dim,
            self.structured_mode_dim,
            num_cell_types,
            fusion_dim=self.coop_fusion_dim,
            aux_dim=self.latent_dim,
            lambda_g=self.coop_lambda_g,
            lambda_x=self.coop_lambda_x,
            lambda_m=self.coop_lambda_m,
        ).to(device)
        self.cond_predictor = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, self.mode_num_classes),
        ).to(device)
        self.mode_head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, num_cell_types),
        ).to(device)
        self.mode_prior_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.mode_num_classes),
        ).to(device)
        self.m_cls_head = nn.Linear(
            self.structured_mode_dim, self.mode_num_classes).to(device)
        init.normal_(self.mode_embedding, mean=0.0, std=0.02)

        # Xavier 初始化新层
        for m in [self.branch1[0], self.branch2_spatial[0], self.branch2_normal[0],
                  self.fusion, self.gene_ae, self.decon_head, self.cond_predictor,
                  self.mode_head, self.mode_prior_head, self.m_cls_head,
                  self.m_residual_projector, self.spatial_decon,
                  self.cooperative_decon]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    init.xavier_normal_(sub.weight)
                    if sub.bias is not None:
                        init.constant_(sub.bias, 0)
        if bounded_gpr is not None:
            self.configure_bounded_gpr(bounded_gpr)

        print(f"  [rebuild_io_layers] num_genes={num_genes}, "
              f"num_cell_types={num_cell_types}, "
              f"mode_num_classes={self.mode_num_classes}, "
              f"spatial_gat_mode={self.spatial_gat_mode}, "
              f"bounded_gpr={self.use_bounded_gpr}")


if __name__ == "__main__":
    num_genes, num_cells, N = 33, 8, 168
    model = DACGModel(num_genes=num_genes, num_cell_types=num_cells)
    x = torch.rand(1, N, num_genes)
    ex_adj = torch.randint(0, 2, (N, N)).float().fill_diagonal_(1.)
    sp_adj = torch.randint(0, 2, (N, N)).float().fill_diagonal_(1.)
    c = torch.rand(1, N, num_cells)

    out_sc = model(x, ex_adj, c=c, mode="sc")
    print("sc mode - z:", out_sc["z"].shape, "decon:", out_sc["decon"].shape,
          "eps_pred:", out_sc["eps_pred"].shape)

    out_st = model(x, [ex_adj, sp_adj], c=None, mode="st")
    print("st mode - z:", out_st["z"].shape, "decon:", out_st["decon"].shape)

    loss_sc = model.compute_loss(x, ex_adj, out_sc, c_true=c, mode="sc")
    print("sc loss:", {k: v.item() for k, v in loss_sc.items()})

    loss_st = model.compute_loss(x, [ex_adj, sp_adj], out_st, c_true=None, mode="st")
    print("st loss:", {k: v.item() for k, v in loss_st.items()})

    recon = model.reconstruct_genes(out_st["z"], num_steps=10)
    print("reconstructed genes:", recon.shape)
