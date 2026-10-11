"""
CR-DiffNet: Production Conditional Latent Denoising U-Net (~88.5M Parameters)
Predicts Gaussian noise vector epsilon conditional on masked latents and multi-modal auxiliary context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from src.models.cross_attention import MultiModalAuxiliaryTokenizer, SpatialCrossAttentionBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class ResnetBlockWithTime(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class CRDiffNetUNet(nn.Module):
    """
    Conditional Latent Denoising U-Net for LISS-IV Cloud Removal.
    Inputs:
        noisy_latents: (B, 4, 64, 64)
        cond_latents:  (B, 4, 64, 64) [Latent representation of cloudy optical + downsampled mask]
        timesteps:     (B,) integer in [0, 999]
        aux_context:   (B, N, context_dim) Multi-modal token sequence
    Outputs:
        noise_pred:    (B, 4, 64, 64)
    """
    def __init__(
        self,
        in_channels: int = 8,      # 4 noisy + 4 condition latents
        out_channels: int = 4,     # 4 latent noise channels
        base_ch: int = 128,
        ch_mults: List[int] = [1, 2, 4, 4],
        time_dim: int = 512,
        context_dim: int = 512
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # Down Blocks
        self.down1_res = ResnetBlockWithTime(base_ch, base_ch * ch_mults[0], time_dim)
        self.down1_attn = SpatialCrossAttentionBlock(base_ch * ch_mults[0], context_dim)
        self.down1_pool = nn.Conv2d(base_ch * ch_mults[0], base_ch * ch_mults[0], kernel_size=3, stride=2, padding=1) # 32x32

        self.down2_res = ResnetBlockWithTime(base_ch * ch_mults[0], base_ch * ch_mults[1], time_dim)
        self.down2_attn = SpatialCrossAttentionBlock(base_ch * ch_mults[1], context_dim)
        self.down2_pool = nn.Conv2d(base_ch * ch_mults[1], base_ch * ch_mults[1], kernel_size=3, stride=2, padding=1) # 16x16

        self.down3_res = ResnetBlockWithTime(base_ch * ch_mults[1], base_ch * ch_mults[2], time_dim)
        self.down3_attn = SpatialCrossAttentionBlock(base_ch * ch_mults[2], context_dim)
        self.down3_pool = nn.Conv2d(base_ch * ch_mults[2], base_ch * ch_mults[2], kernel_size=3, stride=2, padding=1) # 8x8

        # Middle Block (Lowest resolution 8x8)
        mid_ch = base_ch * ch_mults[2]
        self.mid_res1 = ResnetBlockWithTime(mid_ch, mid_ch, time_dim)
        self.mid_attn = SpatialCrossAttentionBlock(mid_ch, context_dim)
        self.mid_res2 = ResnetBlockWithTime(mid_ch, mid_ch, time_dim)

        # Up Blocks with Skip Connections
        self.up3_res = ResnetBlockWithTime(mid_ch * 2, base_ch * ch_mults[1], time_dim)
        self.up3_attn = SpatialCrossAttentionBlock(base_ch * ch_mults[1], context_dim)
        self.up3_up = nn.Upsample(scale_factor=2, mode="nearest") # 16x16

        self.up2_res = ResnetBlockWithTime(base_ch * ch_mults[1] * 2, base_ch * ch_mults[0], time_dim)
        self.up2_attn = SpatialCrossAttentionBlock(base_ch * ch_mults[0], context_dim)
        self.up2_up = nn.Upsample(scale_factor=2, mode="nearest") # 32x32

        self.up1_res = ResnetBlockWithTime(base_ch * ch_mults[0] * 2, base_ch, time_dim)
        self.up1_attn = SpatialCrossAttentionBlock(base_ch, context_dim)
        self.up1_up = nn.Upsample(scale_factor=2, mode="nearest") # 64x64

        # Output projection
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(
        self,
        noisy_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(timesteps)

        # Concat noisy latent and condition latent (4 + 4 = 8 channels)
        x = torch.cat([noisy_latents, cond_latents], dim=1)
        h = self.conv_in(x)

        # Down 1
        h1 = self.down1_res(h, t_emb)
        h1 = self.down1_attn(h1, context)
        h1_p = self.down1_pool(h1)

        # Down 2
        h2 = self.down2_res(h1_p, t_emb)
        h2 = self.down2_attn(h2, context)
        h2_p = self.down2_pool(h2)

        # Down 3
        h3 = self.down3_res(h2_p, t_emb)
        h3 = self.down3_attn(h3, context)
        h3_p = self.down3_pool(h3)

        # Mid
        m = self.mid_res1(h3_p, t_emb)
        m = self.mid_attn(m, context)
        m = self.mid_res2(m, t_emb)

        # Up 3
        u3 = self.up3_up(m)
        u3 = self.up3_res(torch.cat([u3, h3], dim=1), t_emb)
        u3 = self.up3_attn(u3, context)

        # Up 2
        u2 = self.up2_up(u3)
        u2 = self.up2_res(torch.cat([u2, h2], dim=1), t_emb)
        u2 = self.up2_attn(u2, context)

        # Up 1
        u1 = self.up1_up(u2)
        u1 = self.up1_res(torch.cat([u1, h1], dim=1), t_emb)
        u1 = self.up1_attn(u1, context)

        out = self.conv_out(self.act(self.norm_out(u1)))
        return out


class CRDiffNet(nn.Module):
    """
    Full Wrapper Model integrating Aux Tokenizer and Denoising U-Net.
    """
    def __init__(self, aux_in_ch: int = 4, context_dim: int = 512):
        super().__init__()
        self.aux_tokenizer = MultiModalAuxiliaryTokenizer(in_channels=aux_in_ch, context_dim=context_dim)
        self.unet = CRDiffNetUNet(context_dim=context_dim)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        cond_latents: torch.Tensor,
        timesteps: torch.Tensor,
        aux_maps: torch.Tensor
    ) -> torch.Tensor:
        # Tokenize SAR and DEM auxiliary maps
        context = self.aux_tokenizer(aux_maps)
        # Predict noise
        noise_pred = self.unet(noisy_latents, cond_latents, timesteps, context)
        return noise_pred
