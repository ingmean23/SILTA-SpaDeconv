"""
Diffusion 核心组件 v2：
- GeneAutoEncoder: 基因空间 → 隐空间压缩
- Latent Diffusion: 在低维隐空间做扩散（稳定、快速）
- CFG (Classifier-Free Guidance): 条件引导生成
- 生物约束损失: UMI/零表达/分布约束
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.proj(emb)


class GeneAutoEncoder(nn.Module):
    """基因空间 ↔ 低维隐空间的自编码器，用于 Latent Diffusion"""
    def __init__(self, gene_dim, latent_ae_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, latent_ae_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_ae_dim, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, gene_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


# === PLACEHOLDER_NOISE_SCHEDULER ===


class NoiseScheduler:
    """线性噪声调度器"""
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        return self

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bars[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t]
        while sqrt_ab.dim() < x_0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)
        x_t = sqrt_ab * x_0 + sqrt_1_ab * noise
        return x_t, noise

    def denoise_step(self, x_t, eps_pred, t):
        beta = self.betas[t]
        sqrt_alpha = self.sqrt_alphas[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t]
        while beta.dim() < x_t.dim():
            beta = beta.unsqueeze(-1)
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)
        mean = (x_t - beta / sqrt_1_ab * eps_pred) / sqrt_alpha
        if t.min() > 0:
            return mean + torch.sqrt(beta) * torch.randn_like(x_t)
        return mean

    def ddim_step(self, x_t, eps_pred, t, t_prev):
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >= 0 else torch.ones_like(alpha_bar_t)
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        return torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_pred


class DenoiseMLP(nn.Module):
    """条件去噪网络（DACG 模型内部用）
    在隐空间操作: latent_ae_dim 而非 gene_dim
    """
    def __init__(self, latent_ae_dim=64, cond_dim=512, time_dim=128, hidden_dim=512):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        input_dim = latent_ae_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_ae_dim),
        )

    def forward(self, x_t, z, t):
        t_emb = self.time_emb(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x_t.shape[1], -1)
        h = torch.cat([x_t, z, t_emb], dim=-1)
        return self.net(h)


class CondDenoiseMLP(nn.Module):
    """条件去噪网络（伪点生成器用）+ CFG 支持
    在隐空间操作: latent_ae_dim 而非 gene_dim
    训练时 30% 概率清空条件 → 支持 CFG 引导生成
    """
    def __init__(self, latent_ae_dim=64, cond_dim=8, time_dim=128, hidden_dim=512,
                 cfg_drop_prob=0.3):
        super().__init__()
        self.cfg_drop_prob = cfg_drop_prob
        self.cond_dim = cond_dim
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        input_dim = latent_ae_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_ae_dim),
        )

    def forward(self, x_t, c, t):
        # CFG: 训练时随机丢弃条件
        if self.training and torch.rand(1).item() < self.cfg_drop_prob:
            c = torch.zeros_like(c)
        t_emb = self.time_emb(t)
        h = torch.cat([x_t, c, t_emb], dim=-1)
        return self.net(h)

    def forward_cfg(self, x_t, c, t, guidance_scale=1.8):
        """CFG 引导推理：eps = eps_uncond + scale * (eps_cond - eps_uncond)"""
        t_emb = self.time_emb(t)
        # 有条件预测
        h_cond = torch.cat([x_t, c, t_emb], dim=-1)
        eps_cond = self.net(h_cond)
        # 无条件预测
        h_uncond = torch.cat([x_t, torch.zeros_like(c), t_emb], dim=-1)
        eps_uncond = self.net(h_uncond)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def compute_bio_constraints(x_gen, x_real):
    """计算生物约束损失（不需要标签）
    x_gen: 生成的基因表达, x_real: 真实 ST 基因表达
    """
    # 1. 总 UMI 归一化约束
    gen_total = x_gen.sum(dim=-1).mean()
    real_total = x_real.sum(dim=-1).mean()
    loss_umi = F.mse_loss(gen_total, real_total)

    # 2. 零表达比例约束
    zero_gen = (x_gen.abs() < 0.01).float().mean()
    zero_real = (x_real.abs() < 0.01).float().mean()
    loss_zero = F.mse_loss(zero_gen, zero_real)

    # 3. 表达分布重构
    loss_rec = F.mse_loss(x_gen, x_real)

    return loss_umi, loss_zero, loss_rec
