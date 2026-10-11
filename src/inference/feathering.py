"""
2D Gaussian & Cosine Apodization Feathering Engine
Eliminates tile boundary seams and grid line artifacts during large-scale satellite mosaicing.
"""

import numpy as np


def compute_2d_cosine_window(tile_size: int = 512, power: float = 2.0) -> np.ndarray:
    """
    Computes a 2D separable sine/cosine apodization window.
    Weights approach 0.0 at the tile borders and 1.0 at the center.
    W(x, y) = (sin(pi * x / N) * sin(pi * y / N))^power
    """
    x = np.sin(np.linspace(0.01, np.pi - 0.01, tile_size))
    w2d = np.outer(x, x) ** power
    return np.clip(w2d, 1e-4, 1.0).astype(np.float32)


class SeamlessMosaicAccumulator:
    """
    Accumulates overlapping reconstructed tiles and normalizes by the aggregated kernel weights.
    """
    def __init__(self, full_height: int, full_width: int, channels: int = 3, tile_size: int = 512):
        self.height = full_height
        self.width = full_width
        self.channels = channels
        self.tile_size = tile_size

        self.image_buffer = np.zeros((channels, full_height, full_width), dtype=np.float32)
        self.weight_buffer = np.zeros((full_height, full_width), dtype=np.float32)
        self.kernel = compute_2d_cosine_window(tile_size=tile_size)

    def add_tile(self, tile_data: np.ndarray, y_offset: int, x_offset: int):
        """
        tile_data: (C, H_t, W_t) in [0, 1]
        y_offset, x_offset: top-left coordinates in full scene
        """
        th, tw = tile_data.shape[1], tile_data.shape[2]
        w_sub = self.kernel[:th, :tw]

        for c in range(self.channels):
            self.image_buffer[c, y_offset:y_offset + th, x_offset:x_offset + tw] += tile_data[c] * w_sub

        self.weight_buffer[y_offset:y_offset + th, x_offset:x_offset + tw] += w_sub

    def finalize(self) -> np.ndarray:
        """
        Normalizes accumulated pixel intensities by total overlapping weights.
        Returns: (C, H, W) float32 in [0, 1]
        """
        safe_weights = np.maximum(self.weight_buffer, 1e-5)
        for c in range(self.channels):
            self.image_buffer[c] /= safe_weights

        return np.clip(self.image_buffer, 0.0, 1.0)
