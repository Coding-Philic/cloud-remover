"""
DDIM Fast Sampler for Latent Diffusion
Executes accelerated deterministic reverse diffusion sampling in 15–20 steps instead of 1000 DDPM steps.
"""

import math
import torch
import numpy as np


class DDIMFastSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) deterministic sampler.
    Allows 50x inference acceleration with negligible visual quality loss.
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine"
    ):
        self.num_train_timesteps = num_train_timesteps
        if beta_schedule == "cosine":
            # Improved cosine schedule (Nichol & Dhariwal)
            s = 0.008
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.999)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        else:
            self.betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_sampling_timesteps(self, num_inference_steps: int = 15, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Creates uniform sub-sequence of timesteps from [T-1, ..., 0].
        """
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        timesteps = torch.flip(timesteps, dims=[0]).to(device)
        return timesteps

    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        vae,
        cloudy_optical: torch.Tensor,
        mask: torch.Tensor,
        aux_maps: torch.Tensor,
        num_inference_steps: int = 15,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Executes DDIM reverse sampling loop.
        cloudy_optical: (B, 3, 512, 512) in [0, 1]
        mask:           (B, 1, 512, 512) in [0, 1]
        aux_maps:       (B, 4, 512, 512) [SAR_VV, SAR_VH, DEM_Elev, DEM_Slope]
        Returns:
            reconstructed_image: (B, 3, 512, 512) in [0, 1]
        """
        device = cloudy_optical.device
        B = cloudy_optical.shape[0]
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        # 1. Encode cloudy optical image into latent space (64 x 64)
        z_cloudy = vae.encode(cloudy_optical)  # (B, 4, 64, 64)
        z_mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode="nearest")
        # Condition latents: concatenated cloudy latent and mask
        cond_latents = z_cloudy * (1.0 - z_mask)

        # 2. Start from pure Gaussian noise in latent space
        z_t = torch.randn_like(z_cloudy)

        # 3. Step sequence
        timesteps = self.get_sampling_timesteps(num_inference_steps, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)
            
            # Predict noise epsilon
            noise_pred = model(z_t, cond_latents, t_batch, aux_maps)

            # Get alpha_t and alpha_prev
            alpha_t = self.alphas_cumprod[t]
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            alpha_prev = self.alphas_cumprod[prev_t] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # Predict clean latent z_0
            pred_z0 = (z_t - torch.sqrt(1.0 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to z_t
            dir_zt = torch.sqrt(1.0 - alpha_prev - (eta ** 2) * (1.0 - alpha_prev) / (1.0 - alpha_t + 1e-6)) * noise_pred

            # Compute next sample z_{t-1}
            z_t = torch.sqrt(alpha_prev) * pred_z0 + dir_zt

        # 4. Decode generated latent to 512x512 image
        reconstructed_full = vae.decode(z_t)

        # 5. Native ground preservation: Keep 100% of clear ground original
        final_output = cloudy_optical * (1.0 - mask) + reconstructed_full * mask
        return torch.clamp(final_output, 0.0, 1.0)
