"""
Multi-Modal 8-Channel PyTorch Dataset & DataLoader
Stacks [LISS-IV(3), SAR(2), Cloud Mask(1), DEM/Slope(2)] into input cubes for the Generative Model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Optional, Tuple, Dict
from src.data.synthetic_clouds import SyntheticCloudInjector


class LISS4CloudRemovalDataset(Dataset):
    """
    Constructs multi-modal 8-channel input cubes and 3-channel cloud-free target pairs.
    """
    def __init__(
        self,
        clean_optical_tiles: np.ndarray,
        sar_tiles: np.ndarray,
        dem_tiles: np.ndarray,
        is_training: bool = True,
        synthetic_cloud_coverage: Tuple[float, float] = (0.15, 0.65)
    ):
        """
        clean_optical_tiles: (N, 3, 512, 512) float32 in [0, 1]
        sar_tiles:           (N, 2, 512, 512) float32 in [0, 1] [VV, VH]
        dem_tiles:           (N, 2, 512, 512) float32 in [0, 1] [Elevation, Slope]
        """
        self.clean_optical = clean_optical_tiles
        self.sar = sar_tiles
        self.dem = dem_tiles
        self.is_training = is_training
        self.coverage_range = synthetic_cloud_coverage
        self.cloud_injector = SyntheticCloudInjector()

    def __len__(self) -> int:
        return len(self.clean_optical)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean_img = self.clean_optical[idx]  # (3, 512, 512)
        sar_img = self.sar[idx]              # (2, 512, 512)
        dem_img = self.dem[idx]              # (2, 512, 512)

        # 1. Generate on-the-fly synthetic cloud and shadow
        coverage = random.uniform(*self.coverage_range)
        sun_azimuth = random.uniform(110.0, 160.0)
        sun_elevation = random.uniform(35.0, 65.0)

        cloudy_img, mask = self.cloud_injector.apply_clouds(
            clean_img,
            coverage=coverage,
            sun_azimuth_deg=sun_azimuth,
            sun_elevation_deg=sun_elevation
        )  # cloudy_img: (3, 512, 512), mask: (1, 512, 512)

        # 2. Geometric Data Augmentation (consistent across all layers)
        if self.is_training:
            # Horizontal Flip
            if random.random() > 0.5:
                clean_img = np.flip(clean_img, axis=2).copy()
                cloudy_img = np.flip(cloudy_img, axis=2).copy()
                sar_img = np.flip(sar_img, axis=2).copy()
                dem_img = np.flip(dem_img, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()

            # Vertical Flip
            if random.random() > 0.5:
                clean_img = np.flip(clean_img, axis=1).copy()
                cloudy_img = np.flip(cloudy_img, axis=1).copy()
                sar_img = np.flip(sar_img, axis=1).copy()
                dem_img = np.flip(dem_img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

            # Random 90 deg rotation
            k = random.randint(0, 3)
            if k > 0:
                clean_img = np.rot90(clean_img, k, (1, 2)).copy()
                cloudy_img = np.rot90(cloudy_img, k, (1, 2)).copy()
                sar_img = np.rot90(sar_img, k, (1, 2)).copy()
                dem_img = np.rot90(dem_img, k, (1, 2)).copy()
                mask = np.rot90(mask, k, (1, 2)).copy()

        # 3. Stack into 8-channel input tensor
        # Channels: [Cloudy_G, Cloudy_R, Cloudy_NIR, SAR_VV, SAR_VH, Mask, DEM_Elevation, DEM_Slope]
        input_cube = np.concatenate([cloudy_img, sar_img, mask, dem_img], axis=0).astype(np.float32)

        return {
            "input_cube": torch.from_numpy(input_cube),        # (8, 512, 512)
            "target_clear": torch.from_numpy(clean_img),       # (3, 512, 512)
            "mask": torch.from_numpy(mask),                    # (1, 512, 512)
            "cloudy_optical": torch.from_numpy(cloudy_img)     # (3, 512, 512)
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
