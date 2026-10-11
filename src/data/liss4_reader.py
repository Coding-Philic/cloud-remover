"""
LISS-IV Satellite Data Reader & Radiometric Calibration Engine
Converts raw Digital Numbers (DN) to physically calibrated Top-Of-Atmosphere (TOA) Reflectance.
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Exo-atmospheric Solar Spectral Irradiance (ESUN) in W/(m^2 * um) for Resourcesat-2/2A LISS-IV
ESUN_LISS4 = {
    "B2": 1845.0,  # Green (0.52 - 0.59 um)
    "B3": 1575.0,  # Red   (0.62 - 0.68 um)
    "B4": 1090.0,  # NIR   (0.77 - 0.86 um)
}

# Standard Linear Calibration Gain and Bias factors for LISS-IV Level-2 products
DEFAULT_CALIBRATION = {
    "B2": {"gain": 0.150, "bias": 0.0},
    "B3": {"gain": 0.145, "bias": 0.0},
    "B4": {"gain": 0.130, "bias": 0.0},
}


def calculate_earth_sun_distance(day_of_year: int) -> float:
    """
    Computes Earth-Sun distance (d) in Astronomical Units (AU) on a given Julian day.
    Formula: d = 1.0 - 0.01672 * cos(2 * pi * (day_of_year - 4) / 365.25)
    """
    theta = 2.0 * math.pi * (day_of_year - 4) / 365.25
    return 1.0 - 0.01672 * math.cos(theta)


def dn_to_radiance(
    dn: np.ndarray,
    gain: float,
    bias: float
) -> np.ndarray:
    """
    Converts 10-bit raw Digital Numbers to Spectral Radiance (W / (m^2 * sr * um)).
    L_lambda = DN * Gain + Bias
    """
    return np.maximum(0.0, dn.astype(np.float32) * gain + bias)


def radiance_to_toa_reflectance(
    radiance: np.ndarray,
    band_name: str,
    sun_elevation_deg: float,
    day_of_year: int = 170
) -> np.ndarray:
    """
    Converts Spectral Radiance to Top-Of-Atmosphere (TOA) Reflectance [0.0, 1.0].
    rho_lambda = (pi * L_lambda * d^2) / (ESUN_lambda * cos(theta_sz))
    where theta_sz = 90 - sun_elevation_deg
    """
    d_au = calculate_earth_sun_distance(day_of_year)
    zenith_rad = math.radians(max(5.0, 90.0 - sun_elevation_deg))
    esun = ESUN_LISS4.get(band_name, 1500.0)

    numerator = math.pi * radiance * (d_au ** 2)
    denominator = esun * math.cos(zenith_rad)

    toa = numerator / max(1e-5, denominator)
    return np.clip(toa, 0.0, 1.0).astype(np.float32)


class LISS4SceneReader:
    """
    Reads multi-band LISS-IV scenes and converts them into calibrated 3-channel reflectance arrays.
    """
    def __init__(self, scene_dir: Optional[str] = None):
        self.scene_dir = Path(scene_dir) if scene_dir else None

    def read_scene(
        self,
        b2_path: str,
        b3_path: str,
        b4_path: str,
        sun_elevation_deg: float = 48.5,
        day_of_year: int = 165
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reads Band 2 (Green), Band 3 (Red), and Band 4 (NIR), calibrates to TOA,
        and returns a stacked numpy array (3, H, W) in [0.0, 1.0].
        """
        try:
            import rasterio
            with rasterio.open(b2_path) as ds2, rasterio.open(b3_path) as ds3, rasterio.open(b4_path) as ds4:
                dn2 = ds2.read(1)
                dn3 = ds3.read(1)
                dn4 = ds4.read(1)
                meta = ds2.meta.copy()
        except ImportError:
            # Fallback if rasterio not present
            from PIL import Image
            dn2 = np.array(Image.open(b2_path))
            dn3 = np.array(Image.open(b3_path))
            dn4 = np.array(Image.open(b4_path))
            meta = {"width": dn2.shape[1], "height": dn2.shape[0]}

        rad2 = dn_to_radiance(dn2, DEFAULT_CALIBRATION["B2"]["gain"], DEFAULT_CALIBRATION["B2"]["bias"])
        rad3 = dn_to_radiance(dn3, DEFAULT_CALIBRATION["B3"]["gain"], DEFAULT_CALIBRATION["B3"]["bias"])
        rad4 = dn_to_radiance(dn4, DEFAULT_CALIBRATION["B4"]["gain"], DEFAULT_CALIBRATION["B4"]["bias"])

        toa2 = radiance_to_toa_reflectance(rad2, "B2", sun_elevation_deg, day_of_year)
        toa3 = radiance_to_toa_reflectance(rad3, "B3", sun_elevation_deg, day_of_year)
        toa4 = radiance_to_toa_reflectance(rad4, "B4", sun_elevation_deg, day_of_year)

        stacked_toa = np.stack([toa2, toa3, toa4], axis=0)  # Shape: (3, H, W)
        return stacked_toa, meta
