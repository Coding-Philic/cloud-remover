"""
Spatial Compression Autoencoder (VAE)
Compresses 512x512x3 optical imagery by 8x into 64x64x4 latent space, reducing memory by 64x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.shortcut(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_ch: int = 64):
        super().__init__()
        # 512x512 -> 256x256 -> 128x128 -> 64x64 (3 downsamplings = 8x compression)
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        
        self.down1 = nn.Sequential(
            ResnetBlock2D(base_ch, base_ch),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1) # 256x256
        )
        self.down2 = nn.Sequential(
            ResnetBlock2D(base_ch * 2, base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1) # 128x128
        )
        self.down3 = nn.Sequential(
            ResnetBlock2D(base_ch * 4, base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1) # 64x64
        )
        self.mid = ResnetBlock2D(base_ch * 4, base_ch * 4)
        
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=base_ch * 4)
        self.conv_out = nn.Conv2d(base_ch * 4, latent_channels * 2, kernel_size=3, padding=1) # Mean + LogVar
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.mid(h)
        h = self.act(self.norm_out(h))
        moments = self.conv_out(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    def __init__(self, out_channels: int = 3, latent_channels: int = 4, base_ch: int = 64):
        super().__init__()
        # 64x64 -> 128x128 -> 256x256 -> 512x512
        self.conv_in = nn.Conv2d(latent_channels, base_ch * 4, kernel_size=3, padding=1)
        self.mid = ResnetBlock2D(base_ch * 4, base_ch * 4)
        
        self.up1 = nn.Sequential(
            ResnetBlock2D(base_ch * 4, base_ch * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1) # 128x128
        )
        self.up2 = nn.Sequential(
            ResnetBlock2D(base_ch * 4, base_ch * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1) # 256x256
        )
        self.up3 = nn.Sequential(
            ResnetBlock2D(base_ch * 2, base_ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1) # 512x512
        )
        self.norm_out = nn.GroupNorm(num_groups=min(32, base_ch), num_channels=base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.act(self.norm_out(h))
        return self.conv_out(h)


class SpatialCompressionVAE(nn.Module):
    """
    Complete Variational Autoencoder with scale factor 0.18215.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4, scale_factor: float = 0.18215):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = VAEEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = VAEDecoder(out_channels=in_channels, latent_channels=latent_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Scale to [-1, 1]
        x_norm = x * 2.0 - 1.0
        mean, logvar = self.encoder(x_norm)
        # Sample latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_unscaled = z / self.scale_factor
        x_rec = self.decoder(z_unscaled)
        # Scale back to [0, 1]
        return torch.clamp((x_rec + 1.0) / 2.0, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        rec = self.decode(z)
        return rec, z
