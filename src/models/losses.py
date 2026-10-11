"""
Composite Remote Sensing Loss Suite
Combines Noise MSE, L1 Pixel Loss, Perceptual Loss, Spectral Angle Mapper (SAM), and NDVI Fidelity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def spectral_angle_mapper_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes mean Spectral Angle Mapper (SAM) in radians.
    SAM measures the spectral angle between the predicted and target multi-band vectors,
    ensuring radiometric consistency independent of overall brightness.
    pred, target: (B, 3, H, W)
    """
    dot_product = torch.sum(pred * target, dim=1)
    norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=1) + eps)
    norm_target = torch.sqrt(torch.sum(target ** 2, dim=1) + eps)
    
    cos_theta = dot_product / (norm_pred * norm_target + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    sam_rad = torch.acos(cos_theta)
    return torch.mean(sam_rad)


def compute_ndvi(tensor_3band: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes Normalized Difference Vegetation Index (NDVI) for LISS-IV.
    Band indices: 0 = Green, 1 = Red, 2 = NIR
    NDVI = (NIR - Red) / (NIR + Red + eps)
    """
    red = tensor_3band[:, 1:2, :, :]
    nir = tensor_3band[:, 2:3, :, :]
    ndvi = (nir - red) / (nir + red + eps)
    return torch.clamp(ndvi, -1.0, 1.0)


def ndvi_fidelity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 loss between predicted and ground truth NDVI maps.
    Ensures vegetation health indicators remain physically accurate under cloud fills.
    """
    ndvi_pred = compute_ndvi(pred)
    ndvi_gt = compute_ndvi(target)
    return F.l1_loss(ndvi_pred, ndvi_gt)


class CompositeCloudRemovalLoss(nn.Module):
    """
    Multi-objective loss function balancing diffusion noise learning with physical spectral fidelity.
    """
    def __init__(
        self,
        weight_noise: float = 1.0,
        weight_l1: float = 1.0,
        weight_sam: float = 0.05,
        weight_ndvi: float = 0.1
    ):
        super().__init__()
        self.w_noise = weight_noise
        self.w_l1 = weight_l1
        self.w_sam = weight_sam
        self.w_ndvi = weight_ndvi

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        rec_pixel: Optional[torch.Tensor] = None,
        target_pixel: Optional[torch.Tensor] = None,
        cloud_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Diffusion Noise Prediction Loss
        loss_noise = F.mse_loss(noise_pred, noise_true)
        total_loss = self.w_noise * loss_noise
        loss_dict = {"loss_noise": loss_noise.item()}

        # 2. Pixel & Spectral losses (applied if decoded pixel space is provided)
        if rec_pixel is not None and target_pixel is not None:
            # L1 reconstruction
            loss_l1 = F.l1_loss(rec_pixel, target_pixel)
            # Masked L1: higher penalty inside cloud areas
            if cloud_mask is not None:
                mask_weight = 1.0 + cloud_mask * 2.0
                loss_l1 = torch.mean(mask_weight * torch.abs(rec_pixel - target_pixel))
            
            # SAM spectral loss
            loss_sam = spectral_angle_mapper_loss(rec_pixel, target_pixel)
            
            # NDVI fidelity loss
            loss_ndvi = ndvi_fidelity_loss(rec_pixel, target_pixel)

            total_loss = total_loss + (self.w_l1 * loss_l1) + (self.w_sam * loss_sam) + (self.w_ndvi * loss_ndvi)
            
            loss_dict.update({
                "loss_l1": loss_l1.item(),
                "loss_sam_deg": math.degrees(loss_sam.item()),
                "loss_ndvi": loss_ndvi.item()
            })

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
