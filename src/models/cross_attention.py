"""
Multi-Modal Cross-Attention Conditioning Engine
Projects auxiliary SAR backscatter and DEM terrain features into token embeddings
and injects them into the Diffusion U-Net via Multi-Head Cross-Attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalAuxiliaryTokenizer(nn.Module):
    """
    Transforms 4-channel auxiliary maps [SAR_VV, SAR_VH, DEM_Elev, DEM_Slope] (B, 4, 512, 512)
    into compact multi-scale sequence tokens (B, num_tokens, context_dim) for cross-attention.
    """
    def __init__(self, in_channels: int = 4, context_dim: int = 512, num_tokens: int = 64):
        super().__init__()
        self.context_dim = context_dim
        # Convolutional projection down to 8x8 spatial token grid (= 64 tokens)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 256x256
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 128x128
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),         # 64x64
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, context_dim, kernel_size=3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(context_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((8, 8))                                     # 8x8 grid = 64 tokens
        )
        self.proj = nn.Linear(context_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, aux_maps: torch.Tensor) -> torch.Tensor:
        # aux_maps: (B, 4, H, W)
        feat = self.conv_stem(aux_maps)                   # (B, context_dim, 8, 8)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)          # (B, 64, context_dim)
        tokens = self.norm(self.proj(tokens))
        return tokens


class SpatialCrossAttentionBlock(nn.Module):
    """
    Cross-Attention layer where spatial U-Net latents attend to multi-modal condition tokens.
    """
    def __init__(self, query_dim: int, context_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.norm_q = nn.LayerNorm(query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) U-Net spatial latent features
        context: (B, N, context_dim) Multi-modal token sequence
        """
        B, C, H, W = x.shape
        residual = x
        
        # Flatten spatial dimensions to sequence
        x_flat = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x_norm = self.norm_q(x_flat)

        # Queries from U-Net, Keys/Values from Auxiliary Context
        q = self.to_q(x_norm).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2) # (B, heads, HW, head_dim)
        k = self.to_k(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        v = self.to_v(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        # Scaled Dot-Product Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) # (B, heads, HW, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, H * W, C)
        out = self.to_out(out)

        out = out.view(B, H, W, C).permute(0, 3, 1, 2)
        return residual + out
