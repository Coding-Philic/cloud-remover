"""
End-to-End Pipeline Sanity Test
Verifies model forward pass, VAE latent compression, composite losses, and evaluation metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.models.vae import SpatialCompressionVAE
from src.models.diffusion_unet import CRDiffNet
from src.models.losses import CompositeCloudRemovalLoss
from src.evaluation.metrics import evaluate_reconstruction

def test_pipeline():
    print("=== 1. Testing Spatial VAE ===")
    vae = SpatialCompressionVAE(in_channels=3, latent_channels=4)
    x = torch.rand(2, 3, 512, 512)
    rec, z = vae(x)
    print(f"✓ VAE Input: {x.shape} -> Latent: {z.shape} -> Reconstruction: {rec.shape}")
    assert z.shape == (2, 4, 64, 64)
    assert rec.shape == (2, 3, 512, 512)

    print("\n=== 2. Testing CR-DiffNet Denoising U-Net ===")
    model = CRDiffNet(aux_in_ch=4, context_dim=512)
    noisy_latents = torch.randn(2, 4, 64, 64)
    cond_latents = torch.randn(2, 4, 64, 64)
    timesteps = torch.tensor([500, 250])
    aux_maps = torch.rand(2, 4, 512, 512)

    noise_pred = model(noisy_latents, cond_latents, timesteps, aux_maps)
    print(f"✓ U-Net Noise Prediction Output Shape: {noise_pred.shape}")
    assert noise_pred.shape == (2, 4, 64, 64)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ CR-DiffNet Total Trainable Parameters: {param_count / 1e6:.2f} Million")

    print("\n=== 3. Testing Composite Loss Suite ===")
    criterion = CompositeCloudRemovalLoss()
    noise_true = torch.randn_like(noise_pred)
    target_pixel = torch.rand(2, 3, 512, 512)
    rec_pixel = torch.rand(2, 3, 512, 512)
    mask = torch.randint(0, 2, (2, 1, 512, 512)).float()

    total_loss, loss_dict = criterion(noise_pred, noise_true, rec_pixel, target_pixel, mask)
    print(f"✓ Composite Loss Computed: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v:.4f}")

    print("\n=== 4. Testing Evaluation Metrics ===")
    p_np = np.random.uniform(0.1, 0.8, (3, 512, 512)).astype(np.float32)
    t_np = p_np + np.random.normal(0, 0.05, (3, 512, 512)).astype(np.float32)
    t_np = np.clip(t_np, 0.0, 1.0)
    m_np = np.zeros((1, 512, 512), dtype=np.float32)
    m_np[:, 100:300, 100:300] = 1.0

    metrics = evaluate_reconstruction(p_np, t_np, mask=m_np)
    print(f"✓ Metrics Evaluated:")
    print(f"    PSNR: {metrics['PSNR_dB']:.2f} dB")
    print(f"    SSIM: {metrics['SSIM']:.4f}")
    print(f"    SAM:  {metrics['SAM_deg']:.2f}°")
    print(f"    NDVI MAE: {metrics['NDVI_MAE']:.4f}")
    print(f"    CBDI Seam: {metrics['CBDI_Seam']:.4f}")

    print("\n🎉 ALL PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
