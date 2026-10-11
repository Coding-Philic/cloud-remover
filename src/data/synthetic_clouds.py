"""
Physics-Based Synthetic Cloud & Cloud-Shadow Generator
Generates realistic multi-fractal Perlin noise clouds and directional shadows for self-supervised training.
"""

import math
import cv2
import numpy as np
from typing import Tuple


def generate_fractal_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int] = (8, 8),
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0
) -> np.ndarray:
    """
    Generates multi-octave 2D fractal noise simulating natural atmospheric turbulence and cloud density.
    """
    h, w = shape
    noise = np.zeros(shape, dtype=np.float32)
    freq = 1.0
    amp = 1.0

    for _ in range(octaves):
        grid_h = max(2, int(res[0] * freq))
        grid_w = max(2, int(res[1] * freq))
        
        # Low frequency random grid
        small_noise = np.random.rand(grid_h, grid_w).astype(np.float32)
        # Bicubic upsampling to smooth the gradient
        octave_noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        
        noise += amp * octave_noise
        freq *= lacunarity
        amp *= persistence

    # Normalize to [0.0, 1.0]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return noise


class SyntheticCloudInjector:
    """
    Injects realistic thick clouds, thin haze, and directional shadows into clear LISS-IV scenes.
    """
    def __init__(
        self,
        cloud_color_rgb: Tuple[float, float, float] = (0.85, 0.88, 0.90),
        shadow_darkening: float = 0.45
    ):
        self.cloud_color = np.array(cloud_color_rgb, dtype=np.float32).reshape(3, 1, 1)
        self.shadow_darkening = shadow_darkening

    def generate_cloud_pattern(
        self,
        shape: Tuple[int, int] = (512, 512),
        coverage: float = 0.35,
        sun_azimuth_deg: float = 135.0,
        sun_elevation_deg: float = 45.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates cloud opacity map, shadow map, and combined binary mask.
        Returns:
            alpha_cloud: Opacity map [0, 1] (continuous for thin haze and thick cumulus)
            alpha_shadow: Shadow attenuation map [0, 1]
            binary_mask: Binary mask (1.0 = corrupted by cloud or shadow, 0.0 = clear)
        """
        h, w = shape
        raw_noise = generate_fractal_noise_2d((h, w), res=(4, 4), octaves=6)

        # Thresholding to achieve target coverage
        cutoff = np.percentile(raw_noise, (1.0 - coverage) * 100.0)
        cloud_density = np.clip((raw_noise - cutoff) / (1.0 - cutoff + 1e-6), 0.0, 1.0)
        # Non-linear opacity curve
        alpha_cloud = np.power(cloud_density, 0.8).astype(np.float32)

        # Compute shadow offset vector based on solar position
        # Cloud height assumption ~ 1000m -> 170 pixels at 5.8m resolution
        shadow_dist_px = int(np.clip(25.0 / math.tan(math.radians(max(20.0, sun_elevation_deg))), 15, 60))
        azimuth_rad = math.radians(sun_azimuth_deg)
        dx = int(-shadow_dist_px * math.sin(azimuth_rad))
        dy = int(shadow_dist_px * math.cos(azimuth_rad))

        # Shift cloud mask to create directional shadow
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_cloud = cv2.warpAffine(alpha_cloud, translation_matrix, (w, h), borderValue=0)
        
        # Shadows exist only on clear ground (not under the cloud itself)
        alpha_shadow = np.clip(shifted_cloud - alpha_cloud, 0.0, 1.0) * self.shadow_darkening

        # Binary ground truth mask: anywhere cloud > 0.15 or shadow > 0.15
        binary_mask = ((alpha_cloud > 0.15) | (alpha_shadow > 0.10)).astype(np.float32)

        # Dilate mask slightly to prevent boundary bleed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary_mask = cv2.dilate(binary_mask, kernel).astype(np.float32)

        return alpha_cloud, alpha_shadow, binary_mask

    def apply_clouds(
        self,
        clean_image: np.ndarray,
        coverage: float = 0.35,
        sun_azimuth_deg: float = 135.0,
        sun_elevation_deg: float = 45.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies synthetic clouds and shadows to clean 3-band LISS-IV image (3, H, W) in [0, 1].
        Returns:
            cloudy_image: (3, H, W) in [0, 1]
            binary_mask: (1, H, W) in [0, 1]
        """
        _, h, w = clean_image.shape
        alpha_c, alpha_s, b_mask = self.generate_cloud_pattern(
            (h, w), coverage, sun_azimuth_deg, sun_elevation_deg
        )

        alpha_c_3d = np.repeat(alpha_c[np.newaxis, :, :], 3, axis=0)
        alpha_s_3d = np.repeat(alpha_s[np.newaxis, :, :], 3, axis=0)

        # 1. Apply Shadow Darkening to ground
        shadowed_ground = clean_image * (1.0 - alpha_s_3d)

        # 2. Blend Clouds on top
        cloudy_image = shadowed_ground * (1.0 - alpha_c_3d) + self.cloud_color * alpha_c_3d
        cloudy_image = np.clip(cloudy_image, 0.0, 1.0)

        return cloudy_image.astype(np.float32), b_mask[np.newaxis, :, :].astype(np.float32)
