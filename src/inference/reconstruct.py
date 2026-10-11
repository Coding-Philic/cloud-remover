"""
End-to-End Scene Reconstruction Pipeline
Orchestrates overlapping sliding-window inference, DDIM sampling, feathering, and boundary histogram matching.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from src.inference.sampler import DDIMFastSampler
from src.inference.feathering import SeamlessMosaicAccumulator
from src.data.cloud_masker import AutomatedCloudMasker


def match_boundary_histograms(
    native_clear: np.ndarray,
    reconstructed: np.ndarray,
    cloud_mask: np.ndarray
) -> np.ndarray:
    """
    Harmonizes reconstructed pixels with surrounding native clear ground using linear gain-offset matching.
    Prevents brightness and hue steps across the cloud boundary.
    """
    output = reconstructed.copy()
    clear_pixels = (cloud_mask[0] == 0)
    cloud_pixels = (cloud_mask[0] == 1)

    if np.sum(clear_pixels) < 100 or np.sum(cloud_pixels) < 100:
        return output

    for c in range(native_clear.shape[0]):
        native_band = native_clear[c][clear_pixels]
        recon_band = reconstructed[c][cloud_pixels]

        mean_native, std_native = np.mean(native_band), np.std(native_band)
        mean_recon, std_recon = np.mean(recon_band), np.std(recon_band)

        if std_recon > 1e-4:
            # Linear moment matching
            matched = (reconstructed[c] - mean_recon) * (std_native / std_recon) + mean_native
            output[c][cloud_pixels] = matched[cloud_pixels]

    return np.clip(output, 0.0, 1.0)


class FullSceneReconstructor:
    """
    Reconstructs entire multi-kilometer LISS-IV scenes from raw arrays.
    """
    def __init__(
        self,
        model,
        vae,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tile_size: int = 512,
        stride: int = 448
    ):
        self.model = model.to(device).eval()
        self.vae = vae.to(device).eval()
        self.device = torch.device(device)
        self.tile_size = tile_size
        self.stride = stride
        self.sampler = DDIMFastSampler()
        self.masker = AutomatedCloudMasker()

    @torch.no_grad()
    def reconstruct_scene(
        self,
        cloudy_optical: np.ndarray,
        sar_data: np.ndarray,
        dem_data: np.ndarray,
        cloud_mask: Optional[np.ndarray] = None,
        ddim_steps: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        cloudy_optical: (3, H, W) in [0, 1] [Green, Red, NIR]
        sar_data:       (2, H, W) in [0, 1] [VV, VH]
        dem_data:       (2, H, W) in [0, 1] [Elevation, Slope]
        cloud_mask:     (1, H, W) in [0, 1] (optional, detected automatically if None)
        Returns:
            cloud_free_scene: (3, H, W) in [0, 1]
            final_mask:       (1, H, W) in [0, 1]
        """
        _, H, W = cloudy_optical.shape

        # 1. Detect clouds and shadows if not provided
        if cloud_mask is None:
            detected_mask = self.masker.generate_mask(cloudy_optical)
            cloud_mask = detected_mask[np.newaxis, :, :]

        accumulator = SeamlessMosaicAccumulator(H, W, channels=3, tile_size=self.tile_size)

        # 2. Iterate sliding windows
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)

                # Extract tile cubes
                opt_tile = cloudy_optical[:, y_start:y_end, x_start:x_end]
                sar_tile = sar_data[:, y_start:y_end, x_start:x_end]
                dem_tile = dem_data[:, y_start:y_end, x_start:x_end]
                mask_tile = cloud_mask[:, y_start:y_end, x_start:x_end]

                # Check if tile has any cloud to reconstruct
                if np.sum(mask_tile) == 0:
                    accumulator.add_tile(opt_tile, y_start, x_start)
                    continue

                # Prepare torch tensors
                t_opt = torch.from_numpy(opt_tile).unsqueeze(0).to(self.device)
                t_mask = torch.from_numpy(mask_tile).unsqueeze(0).to(self.device)
                t_aux = torch.from_numpy(np.concatenate([sar_tile, dem_tile], axis=0)).unsqueeze(0).to(self.device)

                # Execute DDIM Fast Sampling on GPU
                recon_tile = self.sampler.sample_ddim(
                    self.model,
                    self.vae,
                    t_opt,
                    t_mask,
                    t_aux,
                    num_inference_steps=ddim_steps
                )
                recon_np = recon_tile.squeeze(0).cpu().numpy()

                accumulator.add_tile(recon_np, y_start, x_start)

        # 3. Finalize feathered mosaic
        full_reconstructed = accumulator.finalize()

        # 4. Harmonize boundaries with surrounding unclouded land
        harmonized_scene = match_boundary_histograms(cloudy_optical, full_reconstructed, cloud_mask)

        # 5. Native Pixel Guarantee: Clear ground pixels remain 100% original
        final_cloud_free = cloudy_optical * (1.0 - cloud_mask) + harmonized_scene * cloud_mask

        return np.clip(final_cloud_free, 0.0, 1.0).astype(np.float32), cloud_mask
