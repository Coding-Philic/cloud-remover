"""
Sentinel-1 SAR Radar Preprocessor
Implements Lee Speckle Filtering, Decibel (dB) Conversion, and Min-Max Normalization.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def apply_lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Applies the calibrated Lee Speckle Filter to SAR intensity imagery.
    Reduces multiplicative radar speckle while preserving sharp structural boundaries.
    
    Formula:
    R_hat = mean + W * (pixel - mean)
    where W = var / (var + sigma^2_noise)
    """
    img = img.astype(np.float32)
    mean = uniform_filter(img, size=window_size)
    sqr_mean = uniform_filter(img ** 2, size=window_size)
    var = np.maximum(0.0, sqr_mean - mean ** 2)

    # Estimate noise variance from local statistics
    overall_mean = np.mean(img)
    overall_var = np.var(img)
    noise_var = overall_var / (overall_mean ** 2 + 1e-6)

    # Weights
    weights = var / (var + noise_var * (mean ** 2) + 1e-6)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = mean + weights * (img - mean)
    return np.maximum(0.0, filtered)


def linear_to_db(intensity: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Converts linear radar backscatter power to decibels (dB).
    sigma0_dB = 10 * log10(intensity + eps)
    """
    return 10.0 * np.log10(np.maximum(eps, intensity))


def normalize_sar_band(db_arr: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    """
    Normalizes decibel radar values to [0.0, 1.0] for neural network ingestion.
    """
    clipped = np.clip(db_arr, min_db, max_db)
    return ((clipped - min_db) / (max_db - min_db)).astype(np.float32)


class Sentinel1SARPreprocessor:
    """
    Transforms raw Sentinel-1 VV and VH channels into clean 2-channel normalized radar arrays.
    """
    def __init__(
        self,
        window_size: int = 7,
        vv_range: tuple = (-25.0, 0.0),
        vh_range: tuple = (-32.0, -5.0)
    ):
        self.window_size = window_size
        self.vv_range = vv_range
        self.vh_range = vh_range

    def process(self, vv_raw: np.ndarray, vh_raw: np.ndarray) -> np.ndarray:
        """
        Processes VV and VH arrays:
        1. Lee speckle filtering
        2. dB conversion
        3. Normalization to [0, 1]
        Returns: (2, H, W) float32 tensor
        """
        # Lee filter
        vv_filtered = apply_lee_filter(vv_raw, self.window_size)
        vh_filtered = apply_lee_filter(vh_raw, self.window_size)

        # dB conversion
        vv_db = linear_to_db(vv_filtered)
        vh_db = linear_to_db(vh_filtered)

        # Normalize
        vv_norm = normalize_sar_band(vv_db, self.vv_range[0], self.vv_range[1])
        vh_norm = normalize_sar_band(vh_db, self.vh_range[0], self.vh_range[1])

        return np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)
