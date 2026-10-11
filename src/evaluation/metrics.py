"""
Quantitative Remote Sensing Evaluation Suite
Computes PSNR, SSIM, Spectral Angle Mapper (SAM), Band Correlation (CC), NDVI MAE, and CBDI seam scores.
"""

import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict, Optional


def compute_sam_degrees(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees between prediction and target.
    pred, target: (3, H, W)
    mask: (1, H, W) binary mask (evaluates only inside reconstructed zone if provided)
    """
    eps = 1e-7
    if mask is not None:
        idx = (mask[0] == 1)
        if np.sum(idx) == 0:
            return 0.0
        p = pred[:, idx]    # (3, N)
        t = target[:, idx]  # (3, N)
    else:
        p = pred.reshape(3, -1)
        t = target.reshape(3, -1)

    dot = np.sum(p * t, axis=0)
    norm_p = np.sqrt(np.sum(p ** 2, axis=0) + eps)
    norm_t = np.sqrt(np.sum(t ** 2, axis=0) + eps)

    cos_theta = np.clip(dot / (norm_p * norm_t + eps), -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    return float(np.mean(np.degrees(angles_rad)))


def compute_band_correlation(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes Pearson Correlation Coefficient (CC) per band.
    """
    band_names = ["B2_Green", "B3_Red", "B4_NIR"]
    ccs = {}
    for i, b in enumerate(band_names):
        if mask is not None:
            idx = (mask[0] == 1)
            p = pred[i][idx]
            t = target[i][idx]
        else:
            p = pred[i].flatten()
            t = target[i].flatten()

        if len(p) > 1 and np.std(p) > 1e-5 and np.std(t) > 1e-5:
            cc = np.corrcoef(p, t)[0, 1]
        else:
            cc = 1.0
        ccs[f"CC_{b}"] = float(cc)

    ccs["CC_Mean"] = float(np.mean(list(ccs.values())))
    return ccs


def compute_ndvi_mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Computes Mean Absolute Error in NDVI: (NIR - Red) / (NIR + Red)
    """
    eps = 1e-6
    ndvi_pred = (pred[2] - pred[1]) / (pred[2] + pred[1] + eps)
    ndvi_target = (target[2] - target[1]) / (target[2] + target[1] + eps)

    if mask is not None:
        idx = (mask[0] == 1)
        if np.sum(idx) == 0:
            return 0.0
        return float(np.mean(np.abs(ndvi_pred[idx] - ndvi_target[idx])))

    return float(np.mean(np.abs(ndvi_pred - ndvi_target)))


def compute_cbdi_seam_score(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:
    """
    Cloud Boundary Discontinuity Index (CBDI):
    Measures mean gradient difference across the 3-pixel boundary between native and reconstructed areas.
    Score < 0.05 indicates seamless blending.
    """
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask[0].astype(np.uint8), kernel)
    boundary = (dilated - mask[0]).astype(bool)

    if np.sum(boundary) == 0:
        return 0.0

    diff = np.abs(original[:, boundary] - reconstructed[:, boundary])
    return float(np.mean(diff))


def evaluate_reconstruction(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluates complete benchmark scorecard.
    pred, target: (3, H, W) in [0, 1]
    """
    # PSNR & SSIM
    # Transpose to (H, W, 3) for skimage
    p_hwc = np.transpose(pred, (1, 2, 0))
    t_hwc = np.transpose(target, (1, 2, 0))

    psnr_val = peak_signal_noise_ratio(t_hwc, p_hwc, data_range=1.0)
    ssim_val = structural_similarity(t_hwc, p_hwc, channel_axis=2, data_range=1.0)

    # SAM
    sam_deg = compute_sam_degrees(pred, target, mask=mask)

    # CC
    cc_dict = compute_band_correlation(pred, target, mask=mask)

    # NDVI MAE
    ndvi_mae = compute_ndvi_mae(pred, target, mask=mask)

    # CBDI seam score
    cbdi_val = compute_cbdi_seam_score(target, pred, mask) if mask is not None else 0.0

    report = {
        "PSNR_dB": float(psnr_val),
        "SSIM": float(ssim_val),
        "SAM_deg": float(sam_deg),
        "NDVI_MAE": float(ndvi_mae),
        "CBDI_Seam": float(cbdi_val),
        **cc_dict
    }
    return report
