"""
Synthetic Multi-Modal LISS-IV Scene Generator
Generates realistic terrain (river, agricultural fields, forests, urban zones),
co-registered Sentinel-1 SAR, DEM, and synthetic clouds for offline testing and webapp demo.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.data.synthetic_clouds import SyntheticCloudInjector


def generate_landscape_scene(h: int = 512, w: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates realistic 3-band LISS-IV landscape [Green, Red, NIR], Sentinel-1 SAR [VV, VH], and DEM [Elev, Slope].
    """
    np.random.seed(42)

    # 1. Base Agricultural / Forest grid
    green = np.zeros((h, w), dtype=np.float32)
    red = np.zeros((h, w), dtype=np.float32)
    nir = np.zeros((h, w), dtype=np.float32)

    # Grid of agricultural crop parcels
    cell_size = 32
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            crop_type = np.random.rand()
            if crop_type < 0.4:
                # Dense healthy vegetation (Low red, High NIR)
                g_val = np.random.uniform(0.12, 0.18)
                r_val = np.random.uniform(0.06, 0.10)
                n_val = np.random.uniform(0.55, 0.75)
            elif crop_type < 0.7:
                # Fallow / Harvested soil
                g_val = np.random.uniform(0.18, 0.25)
                r_val = np.random.uniform(0.22, 0.32)
                n_val = np.random.uniform(0.25, 0.35)
            else:
                # Moderate vegetation / pasture
                g_val = np.random.uniform(0.14, 0.20)
                r_val = np.random.uniform(0.10, 0.16)
                n_val = np.random.uniform(0.38, 0.50)

            green[y:y+cell_size, x:x+cell_size] = g_val
            red[y:y+cell_size, x:x+cell_size] = r_val
            nir[y:y+cell_size, x:x+cell_size] = n_val

    # 2. Add Meandering River (Water: Low NIR, moderate Green)
    river_mask = np.zeros((h, w), dtype=np.uint8)
    curve_points = []
    for y in range(0, h, 10):
        x = int(w * 0.5 + 80.0 * np.sin(y / 45.0) + 20.0 * np.cos(y / 15.0))
        curve_points.append([x, y])
    curve_pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(river_mask, [curve_pts], isClosed=False, color=255, thickness=28)
    
    is_water = (river_mask > 0)
    green[is_water] = 0.10
    red[is_water] = 0.05
    nir[is_water] = 0.02

    # 3. Add Roads and Settlement
    road_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(road_mask, (0, int(h * 0.3)), (w, int(h * 0.4)), 255, 6)
    cv2.line(road_mask, (int(w * 0.3), 0), (int(w * 0.2), h), 255, 6)
    is_road = (road_mask > 0)
    green[is_road] = 0.22
    red[is_road] = 0.24
    nir[is_road] = 0.20

    # Smooth textures slightly
    green = cv2.GaussianBlur(green, (5, 5), 0.8)
    red = cv2.GaussianBlur(red, (5, 5), 0.8)
    nir = cv2.GaussianBlur(nir, (5, 5), 0.8)

    clean_optical = np.stack([green, red, nir], axis=0) # (3, H, W)

    # 4. Generate Physically Consistent Sentinel-1 SAR Backscatter
    # Water: specular reflection away from radar -> very low backscatter (dark)
    # Dense vegetation / Forest: volume scattering -> medium-high backscatter
    # Urban / Roads: double bounce -> very high backscatter
    sar_vv = np.random.uniform(0.35, 0.65, (h, w)).astype(np.float32)
    sar_vh = sar_vv * 0.6 + np.random.uniform(0.05, 0.15, (h, w)).astype(np.float32)
    sar_vv[is_water] = np.random.uniform(0.02, 0.08, np.sum(is_water))
    sar_vh[is_water] = np.random.uniform(0.01, 0.05, np.sum(is_water))
    sar_vv[is_road] = np.random.uniform(0.80, 0.95, np.sum(is_road))
    sar_vh[is_road] = np.random.uniform(0.65, 0.85, np.sum(is_road))
    sar_stack = np.stack([sar_vv, sar_vh], axis=0) # (2, H, W)

    # 5. Generate DEM Elevation & Slope
    dem_x, dem_y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    elevation = (0.2 + 0.5 * dem_y + 0.1 * np.sin(dem_x * 4.0)).astype(np.float32)
    slope = (np.abs(np.gradient(elevation, axis=0)) + np.abs(np.gradient(elevation, axis=1))).astype(np.float32)
    slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
    dem_stack = np.stack([elevation, slope], axis=0) # (2, H, W)

    return clean_optical, sar_stack, dem_stack


def main():
    out_dir = Path("data/sample_scenes")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating realistic multi-modal test sample in {out_dir}...")

    clean_opt, sar, dem = generate_landscape_scene(512, 512)

    # Inject realistic clouds and shadows
    injector = SyntheticCloudInjector()
    cloudy_opt, mask = injector.apply_clouds(clean_opt, coverage=0.40, sun_azimuth_deg=135.0, sun_elevation_deg=45.0)

    # Save as .npz archive
    npz_path = out_dir / "sample_tile_ner_512.npz"
    np.savez_compressed(
        npz_path,
        clean_optical=clean_opt,
        cloudy_optical=cloudy_opt,
        sar=sar,
        dem=dem,
        mask=mask
    )
    print(f"✓ Created sample multi-modal package at: {npz_path}")
    print(f"  Optical clean shape: {clean_opt.shape}")
    print(f"  Cloudy optical shape: {cloudy_opt.shape}")
    print(f"  Cloud mask coverage: {np.mean(mask) * 100:.1f}%")
    print(f"  SAR channels: {sar.shape}")
    print(f"  DEM channels: {dem.shape}")

if __name__ == "__main__":
    main()
