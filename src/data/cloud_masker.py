"""
Automated Cloud and Cloud-Shadow Detection Engine
Implements spectral tests (Brightness, Whiteness, Shadow dip) and morphological dilation.
"""

import cv2
import numpy as np


class AutomatedCloudMasker:
    """
    Detects thick clouds, thin haze, and cloud shadows in 3-band LISS-IV imagery (B2=Green, B3=Red, B4=NIR).
    """
    def __init__(
        self,
        cloud_brightness_threshold: float = 0.28,
        whiteness_threshold: float = 0.12,
        shadow_nir_threshold: float = 0.12,
        dilation_kernel_radius: int = 15
    ):
        self.cloud_brightness_threshold = cloud_brightness_threshold
        self.whiteness_threshold = whiteness_threshold
        self.shadow_nir_threshold = shadow_nir_threshold
        self.dilation_kernel_radius = dilation_kernel_radius

    def detect_clouds(self, toa_3band: np.ndarray) -> np.ndarray:
        """
        Detects cloud pixels based on visible brightness and whiteness.
        toa_3band: (3, H, W) in [0.0, 1.0] -> [Green, Red, NIR]
        """
        green = toa_3band[0]
        red = toa_3band[1]
        nir = toa_3band[2]

        # Visible Brightness
        vis_brightness = (green + red) / 2.0

        # Whiteness: difference between channels (clouds reflect equally across visible spectrum)
        mean_vis = vis_brightness + 1e-6
        whiteness = (np.abs(green - mean_vis) + np.abs(red - mean_vis)) / mean_vis

        # Cloud decision rule
        cloud_mask = (vis_brightness > self.cloud_brightness_threshold) & (whiteness < self.whiteness_threshold)
        return cloud_mask

    def detect_shadows(self, toa_3band: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
        """
        Detects cloud shadow pixels: sharp drop in NIR and visible bands in non-cloud regions.
        """
        green = toa_3band[0]
        red = toa_3band[1]
        nir = toa_3band[2]

        total_brightness = (green + red + nir) / 3.0
        # Shadows have very low reflectance across all bands
        shadow_mask = (total_brightness < self.shadow_nir_threshold) & (~cloud_mask)
        return shadow_mask

    def generate_mask(self, toa_3band: np.ndarray) -> np.ndarray:
        """
        Generates dilated binary cloud + shadow mask (H, W).
        1.0 = Missing / Corrupted (Cloud or Shadow)
        0.0 = Valid Ground
        """
        cloud_raw = self.detect_clouds(toa_3band)
        shadow_raw = self.detect_shadows(toa_3band, cloud_raw)

        corrupted_raw = (cloud_raw | shadow_raw).astype(np.uint8)

        # Apply morphological dilation with circular structuring element
        # Ensures diffuse cloud penumbra and boundary transitions are fully captured
        r = self.dilation_kernel_radius
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        dilated_mask = cv2.dilate(corrupted_raw, kernel)

        return dilated_mask.astype(np.float32)
