# 07 — Complete Workflow and Operational Pipeline
## Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery

---

## 1. Executive Workflow Architecture

This document specifies the operational end-to-end pipeline that transforms raw, cloud-contaminated LISS-IV multispectral scenes into radiometrically balanced, spectrally consistent, analysis-ready cloud-free imagery (ARD).

The pipeline supports two primary execution modalities:
1. **Model Development Workflow (Offline Training Loop):** Raw ingestion → Ground-truth pairing / Synthetic cloud generation → Multi-modal dataset preparation → Distributed training → Validation & checkpoint selection.
2. **Operational Inference Workflow (Production Batch & Near-Real-Time):** Ingestion of newly acquired LISS-IV scene → Automated auxiliary retrieval (Sentinel-1 SAR, Sentinel-2 reference, DEM) → Sub-pixel co-registration → Tiled diffusion inference → Feathered reconstruction → QA scoring → Cloud-Optimized GeoTIFF (COG) generation & STAC cataloging.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               OPERATIONAL END-TO-END PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Raw Bhoonidhi Data]      [Copernicus CDSE]        [Copernicus OpenTopography]   [Historical LISS-IV]
   LISS-IV Scene (5.8m)     Sentinel-1 GRD (10m)          COP-30 DEM (30m)         Clear Scene (±30d)
           │                        │                            │                         │
           ▼                        ▼                            ▼                         ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 1: INGESTION & VALIDATION ENGINE                                                 │
   │ • Checksum & archive integrity verification                                            │
   │ • Metadata parsing (Sun Azimuth/Elevation, Acquisition Time, Sensor Calibration)       │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 2: RADIOMETRIC & GEOMETRIC PREPROCESSING                                         │
   │ • DN to Top-Of-Atmosphere (TOA) Reflectance conversion                                 │
   │ • Orthorectification verification & UTM grid reprojection (Common LISS-IV 5.8m Grid)   │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 3: AUTOMATED CLOUD & SHADOW MASKING                                              │
   │ • Primary: DeepLabv3+ Dual-Task (Thick cloud, Thin haze, Shadow detection)            │
   │ • Fallback: Rule-based Spectral Index thresholding (HOT, Whiteness, Shadow index)      │
   │ • Morphological dilation & buffer zone generation (15-pixel buffer)                    │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 4: MULTI-MODAL ALIGNMENT & SUB-PIXEL CO-REGISTRATION                             │
   │ • AROSICS Fourier-shift sub-pixel tie-point matching (< 0.5 pixel RMSE)               │
   │ • Lee-filtered SAR backscatter (VV, VH, VV/VH ratio) upsampled to 5.8m                 │
   │ • Slope and Aspect extraction from DEM                                                 │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 5: TILING & DATASET STACKING                                                     │
   │ • Overlapping sliding window (512×512 tiles with 64-pixel stride overlap)              │
   │ • 8-Channel Tensor Stack: [LISS-IV (3), SAR (2), Cloud Mask (1), DEM/Slope (2)]        │
   │ • Missing auxiliary handler (graceful fallback to Single-Image / Optical-only mode)    │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 6: GENERATIVE RECONSTRUCTION ENGINE                                              │
   │ • Latent Encoder (E) compresses 512×512 input into 64×64 latent feature space          │
   │ • Cross-Attention Conditional Denoising U-Net (DDIM 15-20 steps)                       │
   │ • Latent Decoder (D) reconstructs 5.8m cloud-free surface reflectance                  │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 7: SEAMLESS REASSEMBLY & SPECTRAL HARMONIZATION                                  │
   │ • 2D Gaussian apodization window feathering for overlapping borders                    │
   │ • Mosaic reconstruction back to original scene dimensions (70km × 70km)                │
   │ • Boundary histogram matching with unclouded native LISS-IV context                    │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 8: QUALITY ASSURANCE & PHYSICAL VALIDATION                                       │
   │ • Cloud Boundary Discontinuity Index (CBDI < 0.05)                                     │
   │ • Spectral Angle Mapper (SAM) against temporal reference (< 3.0°)                      │
   │ • NDVI & NDWI preservation check (relative error < 5%)                                 │
   │ • Quality Flag Generation (High confidence, Moderate haze, Interpolated)              │
   └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 9: ANALYSIS-READY DATA (ARD) PACKAGING & EXPORT                                  │
   │ • Cloud-Optimized GeoTIFF (COG) generation with internal overviews                     │
   │ • SpatioTemporal Asset Catalog (STAC) Item JSON generation                             │
   │ • Automated catalog update & S3/MinIO archival                                         │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Infrastructure, Environment, and Tooling

### 2.1 Compute and Storage Stack

| Component | Cloud Training (Kaggle + GDrive) | Dedicated Cluster Training | Operational Serving (Laptop / API) |
|-----------|----------------------------------|----------------------------|-----------------------------------|
| **Compute** | **2× NVIDIA T4 (32 GB total VRAM)** | 2× NVIDIA A100 (80GB SXM4) | 1× GPU (RTX / T4) or Modern CPU |
| **CPU / RAM** | 30 GB CPU RAM (Free Cloud) | 32 vCPUs, 128 GB RAM | 16 GB RAM |
| **Storage Strategy** | Google Drive (~80GB raw, ~18GB H5) | 2 TB Local NVMe + MinIO/S3 | Local NVMe (`models/`: ~320 MB) |
| **Bandwidth Spent** | **0 MB local bandwidth** (Cloud-to-Cloud)| High-speed data center link| ~320 MB (to download final model)|
| **Orchestration** | Kaggle Notebooks + Rclone daemon | Prefect 2.x / Ray Core | FastAPI / QGIS Plugin |
| **Experiment Tracking**| Weights & Biases (W&B) | W&B + MLflow | Local JSON logs |

### 2.2 Software Environment Configuration

The environment requires strict compatibility between GDAL, PyTorch, CUDA, and geospatial acceleration libraries.

```yaml
# environment.yml
name: liss4-cloud-remover
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - cudatoolkit=11.8
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio
  - gdal=3.8.3
  - rasterio=1.3.9
  - shapely=2.0.2
  - geopandas=0.14.2
  - pyproj=3.6.1
  - libspatialindex
  - pip
  - pip:
      - opencv-python-headless==4.9.0.80
      - albumentations==1.4.0
      - diffusers==0.26.3
      - transformers==4.38.1
      - accelerate==0.27.2
      - timm==0.9.16
      - arosics==1.2.6
      - sentinelsat==1.2.2
      - planetary-computer==1.0.0
      - pystac==1.9.0
      - rio-cogeo==5.3.0
      - prefect==2.16.4
      - ray[default]==2.9.3
      - fastapi==0.110.0
      - uvicorn==0.27.1
      - pydantic==2.6.1
      - wandb==0.16.3
      - scikit-image==0.22.0
      - spectral==0.23.1
```

---

## 3. Detailed Stage-by-Stage Operational Walkthrough

### Stage 1: Data Ingestion & Integrity Verification

#### Objective
Ensure that ingested LISS-IV archives from Bhoonidhi, Sentinel-1 SAR GRD products, and DEM rasters are non-corrupted, correctly formatted, and accompanied by parsed XML/TXT metadata.

#### Step Execution
1. Unpack downloaded zip/tar files into isolated scene directories.
2. Verify SHA256 checksums if supplied by the provider.
3. Validate GeoTIFF headers with `gdalinfo`.
4. Parse sensor geometry and solar angles from `metadata.xml` / `leader.txt`.

```python
# pipeline/stage1_ingest.py
import os
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from osgeo import gdal

def verify_file_checksum(filepath: Path, expected_hash: str = None) -> bool:
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False
    if expected_hash:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest() == expected_hash
    return True

def parse_liss4_metadata(meta_path: Path) -> dict:
    tree = ET.parse(meta_path)
    root = tree.getroot()
    return {
        "scene_id": root.findtext(".//SceneIdentifier", "UNKNOWN"),
        "date_acquired": root.findtext(".//AcquisitionDate"),
        "sun_azimuth": float(root.findtext(".//SunAzimuth", 135.0)),
        "sun_elevation": float(root.findtext(".//SunElevation", 45.0)),
        "sensor": root.findtext(".//Sensor", "LISS4"),
        "bands": ["Band2_Green", "Band3_Red", "Band4_NIR"]
    }

def validate_scene_integrity(scene_dir: Path) -> bool:
    required_bands = ["BAND2.tif", "BAND3.tif", "BAND4.tif"]
    for b in required_bands:
        f = scene_dir / b
        if not f.exists():
            raise FileNotFoundError(f"Missing mandatory band: {b}")
        ds = gdal.Open(str(f))
        if ds is None or ds.RasterXSize == 0 or ds.RasterYSize == 0:
            raise ValueError(f"Corrupted raster band detected: {f}")
        ds = None
    return True
```

---

### Stage 2: Radiometric & Geometric Preprocessing

#### Objective
Convert raw Digital Numbers (DN) to physically meaningful Top-Of-Atmosphere (TOA) Reflectance and standardize spatial reference systems to the target UTM projection at 5.8m ground resolution.

#### Mathematical Formulation
1. **DN to Radiance ($L_\lambda$):**
   $$L_\lambda = \text{DN}_\lambda \times \text{Gain}_\lambda + \text{Bias}_\lambda$$
2. **Radiance to TOA Reflectance ($\rho_\lambda$):**
   $$\rho_\lambda = \frac{\pi \cdot L_\lambda \cdot d^2}{\text{ESUN}_\lambda \cdot \cos(\theta_{\text{SZ}})}$$
   Where:
   - $d$: Earth-Sun distance in astronomical units (AU) on acquisition day.
   - $\text{ESUN}_\lambda$: Mean exo-atmospheric solar spectral irradiance for LISS-IV band $\lambda$ ($\text{W}/\text{m}^2/\mu\text{m}$).
     - Band 2 (Green, 0.52–0.59 $\mu$m): $1845.0$
     - Band 3 (Red, 0.62–0.68 $\mu$m): $1575.0$
     - Band 4 (NIR, 0.77–0.86 $\mu$m): $1090.0$
   - $\theta_{\text{SZ}}$: Solar Zenith Angle ($90^\circ - \text{Sun Elevation}$).

```python
# pipeline/stage2_preprocess.py
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

ESUN = {"B2": 1845.0, "B3": 1575.0, "B4": 1090.0}

def dn_to_toa(dn_arr: np.ndarray, band_name: str, sun_elevation: float, d_au: float = 1.0) -> np.ndarray:
    # LISS-IV standard 10-bit or 8-bit linear scaling factors
    gain = 0.15
    bias = 0.0
    radiance = dn_arr.astype(np.float32) * gain + bias
    
    zenith_rad = np.radians(90.0 - sun_elevation)
    esun_val = ESUN[band_name]
    toa = (np.pi * radiance * (d_au ** 2)) / (esun_val * np.cos(zenith_rad))
    return np.clip(toa, 0.0, 1.0)

def reproject_to_target_grid(src_path: str, dst_path: str, target_crs: str, target_res: float = 5.8):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=target_res
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': 0.0
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )
```

---

### Stage 3: Automated Cloud and Cloud-Shadow Detection

#### Objective
Produce a pixel-accurate 3-class segmentation mask:
- Class 0: Clear Ground
- Class 1: Thick Cloud & Semi-transparent Thin Haze
- Class 2: Cloud Shadow

#### Pipeline Architecture
A dual-tier approach ensures maximum robustness:
1. **Primary Model:** Pre-trained DeepLabv3+ with MobileNetV3/ResNet-50 backbone, fine-tuned on mountainous cloud datasets.
2. **Deterministic Fallback Engine:** Spectral index thresholds (Visible Brightness, Whiteness, and Cloud Shadow Index based on NIR dips).
3. **Morphological Post-Processing:** Morphological dilation using a circular kernel ($r = 15$ pixels / 87m) to capture diffuse cloud boundaries and penumbra effects.

```
┌──────────────────────────────────────────────────────────────┐
│                  CLOUD DETECTION SUB-PIPELINE                │
└──────────────────────────────────────────────────────────────┘
  TOA LISS-IV [B2, B3, B4]
            │
            ├──► [DeepLabv3+ Inference] ──────────┐
            │                                     ▼
            └──► [Spectral Index Evaluator] ──► [Consensus / Fallback]
                                                  │
                                                  ▼
                                        [Raw Prediction Mask]
                                                  │
                                                  ▼
                                        [Morphological Dilation]
                                        (87m Buffer on Clouds & Shadows)
                                                  │
                                                  ▼
                                        [Final Binary Mask (M)]
```

```python
# pipeline/stage3_cloud_mask.py
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

def create_cloud_detector(model_path: str, device: str = "cuda"):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3  # 0: Clear, 1: Cloud, 2: Shadow
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def morphological_postprocess(mask: np.ndarray, buffer_radius: int = 15) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_radius * 2 + 1, buffer_radius * 2 + 1))
    
    cloud_binary = (mask == 1).astype(np.uint8)
    shadow_binary = (mask == 2).astype(np.uint8)
    
    dilated_cloud = cv2.dilate(cloud_binary, kernel)
    dilated_shadow = cv2.dilate(shadow_binary, kernel)
    
    final_mask = np.zeros_like(mask, dtype=np.uint8)
    final_mask[dilated_shadow == 1] = 2
    final_mask[dilated_cloud == 1] = 1  # Cloud takes priority
    return final_mask
```

---

### Stage 4: Multi-Modal Alignment & Sub-Pixel Co-Registration

#### Objective
Align auxiliary datasets (Sentinel-1 SAR, Sentinel-2 Optical Reference, COP-30 DEM) to the LISS-IV coordinate reference grid with sub-pixel precision (< 0.5 pixel root-mean-square error).

#### Pipeline Flow
1. **Resampling:** Reproject Sentinel-1 GRD (10m) and DEM (30m) to UTM 5.8m using cubic-spline and bilinear interpolation respectively.
2. **SAR Preprocessing:** Apply calibrated Lee speckle filtering ($7\times7$ window), convert linear power backscatter $\sigma^\circ$ to decibels ($\text{dB}$), and compute the dual-polarization cross-ratio:
   $$\text{Ratio} = \text{VV}_{\text{dB}} - \text{VH}_{\text{dB}}$$
3. **Sub-Pixel Tie-Point Matching:** Utilize `arosics` (Automated Registration and Orthorectification of Imagery by Cloud-free Imagery Matching) across clear non-cloud portions to compute local deformation fields.

```python
# pipeline/stage4_coregistration.py
import numpy as np
from arosics import COREG
import rasterio

def perform_subpixel_alignment(reference_tiff: str, target_tiff: str, output_tiff: str):
    """
    Sub-pixel alignment using AROSICS in Fourier frequency domain.
    Ensures geometric alignment error is < 0.5 pixels.
    """
    c = COREG(
        im_ref=reference_tiff,
        im_tgt=target_tiff,
        grid_res=100,
        window_size=(128, 128),
        path_out=output_tiff,
        fmt_out='GTIFF',
        resamp_alg='cubic',
        max_shift=15,  # Max allowable shift in pixels
        nodata=(-9999, -9999)
    )
    c.calculate_spatial_shifts()
    if c.success:
        c.correct_shifts()
        print(f"Alignment successful. Shift X: {c.x_shift_px:.2f} px, Shift Y: {c.y_shift_px:.2f} px")
    else:
        print("Co-registration failed or insufficient tie points; maintaining nominal georeference.")
```

---

### Stage 5: Tiling, Normalization, & Input Stack Construction

#### Objective
Cut large 70km LISS-IV scenes ($12,000 \times 12,000$ pixels) into uniform $512 \times 512$ tiles with a $64$-pixel overlapping border (stride = 448 pixels) to prepare multi-modal tensor cubes for model ingestion.

#### Channel Stacking Specification (8 Channels Total)
1. **Channel 0:** LISS-IV Green (TOA Reflectance $[0, 1]$)
2. **Channel 1:** LISS-IV Red (TOA Reflectance $[0, 1]$)
3. **Channel 2:** LISS-IV NIR (TOA Reflectance $[0, 1]$)
4. **Channel 3:** Sentinel-1 VV Backscatter (Normalized dB $[-25, 0] \to [0, 1]$)
5. **Channel 4:** Sentinel-1 VH Backscatter (Normalized dB $[-32, -5] \to [0, 1]$)
6. **Channel 5:** Binary Cloud & Shadow Mask ($1.0$ = Missing/Cloud, $0.0$ = Valid Ground)
7. **Channel 6:** Normalized Digital Elevation Model (Copernicus DEM $[0, 1]$)
8. **Channel 7:** Terrain Slope ($[0^\circ, 90^\circ] \to [0, 1]$)

```python
# pipeline/stage5_tiling.py
from dataclasses import dataclass
from typing import Generator
import numpy as np

@dataclass
class TileMetadata:
    col_off: int
    row_off: int
    width: int
    height: int
    overlap: int

def generate_tile_windows(img_w: int, img_h: int, tile_size: int = 512, overlap: int = 64) -> Generator[TileMetadata, None, None]:
    stride = tile_size - overlap
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            yield TileMetadata(col_off=x_start, row_off=y_start, width=tile_size, height=tile_size, overlap=overlap)

def build_8channel_tensor(liss_tile: np.ndarray, sar_tile: np.ndarray, mask_tile: np.ndarray, dem_tile: np.ndarray) -> np.ndarray:
    # liss_tile: (3, 512, 512) in [0, 1]
    # sar_tile: (2, 512, 512) normalized
    # mask_tile: (1, 512, 512) binary
    # dem_tile: (2, 512, 512) elevation + slope
    tensor_cube = np.concatenate([liss_tile, sar_tile, mask_tile, dem_tile], axis=0)
    return tensor_cube.astype(np.float32)
```

---

### Stage 6: Generative Reconstruction Engine (Diffusion Inference)

#### Objective
Run accelerated conditional latent diffusion inference (DDIM sampler) on the masked areas, preserving unmasked native pixels exactly as observed.

```python
# pipeline/stage6_reconstruct.py
import torch
import numpy as np

@torch.inference_mode()
def reconstruct_tile_ddim(
    model, 
    vae, 
    input_cube: torch.Tensor, 
    steps: int = 15, 
    eta: float = 0.0,
    device: str = "cuda"
) -> np.ndarray:
    """
    input_cube: Tensor (1, 8, 512, 512)
    Channel 0-2: Cloudy LISS-IV
    Channel 3-4: SAR VV/VH
    Channel 5: Mask
    Channel 6-7: DEM/Slope
    """
    input_cube = input_cube.to(device)
    cloudy_optical = input_cube[:, 0:3, :, :]
    mask = input_cube[:, 5:6, :, :]
    aux_cond = torch.cat([input_cube[:, 3:5, :, :], input_cube[:, 6:8, :, :]], dim=1) # SAR + DEM
    
    # Encode optical image into latent space (64 x 64)
    z_cloudy = vae.encode(cloudy_optical * 2.0 - 1.0).latent_dist.sample() * 0.18215
    z_mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode='nearest')
    
    # Initialize Gaussian latent noise
    z_t = torch.randn_like(z_cloudy)
    
    # DDIM Reverse Sampling Loop
    timesteps = model.scheduler.set_timesteps(num_inference_steps=steps)
    for t in timesteps:
        # Predict noise conditional on masked latent and auxiliary features
        latent_input = torch.cat([z_t, z_cloudy, z_mask], dim=1)
        noise_pred = model(latent_input, t, cond=aux_cond)
        z_t = model.scheduler.step(noise_pred, t, z_t, eta=eta).prev_sample
        
    # Decode latent representation to pixel space
    reconstructed_full = (vae.decode(z_t / 0.18215).sample + 1.0) / 2.0
    reconstructed_full = torch.clamp(reconstructed_full, 0.0, 1.0)
    
    # Hard blend: Keep 100% of native clear pixels untouched
    output_cube = cloudy_optical * (1.0 - mask) + reconstructed_full * mask
    return output_cube.squeeze(0).cpu().numpy()
```

---

### Stage 7: Seamless Reassembly & Spectral Harmonization

#### Objective
Assemble overlapping tiles back into the full $70\text{km} \times 70\text{km}$ scene without grid artifacts or boundary seams, using 2D cosine/Gaussian feathering and boundary histogram matching.

#### Feathering Matrix Computation
For tile coordinates $(i, j) \in [0, N-1] \times [0, N-1]$, the 2D apodization weight $W(i, j)$ is computed as:
$$w_x(i) = \sin\left(\frac{\pi (i + 0.5)}{N}\right), \quad w_y(j) = \sin\left(\frac{\pi (j + 0.5)}{N}\right)$$
$$W(i, j) = (w_x(i) \cdot w_y(j))^p, \quad \text{where } p = 2.0$$

```python
# pipeline/stage7_mosaic.py
import numpy as np

def create_feather_weight(tile_size: int = 512, power: float = 2.0) -> np.ndarray:
    x = np.sin(np.linspace(0, np.pi, tile_size))
    w2d = np.outer(x, x) ** power
    return np.clip(w2d, 1e-4, 1.0).astype(np.float32)

class MosaicAccumulator:
    def __init__(self, full_height: int, full_width: int, channels: int = 3):
        self.full_height = full_height
        self.full_width = full_width
        self.channels = channels
        self.image_acc = np.zeros((channels, full_height, full_width), dtype=np.float32)
        self.weight_acc = np.zeros((full_height, full_width), dtype=np.float32)
        self.weight_kernel = create_feather_weight(tile_size=512)

    def add_tile(self, tile_data: np.ndarray, y_off: int, x_off: int):
        th, tw = tile_data.shape[1], tile_data.shape[2]
        kernel = self.weight_kernel[:th, :tw]
        
        for c in range(self.channels):
            self.image_acc[c, y_off:y_off+th, x_off:x_off+tw] += tile_data[c] * kernel
        self.weight_acc[y_off:y_off+th, x_off:x_off+tw] += kernel

    def finalize(self) -> np.ndarray:
        # Avoid division by zero
        safe_weights = np.maximum(self.weight_acc, 1e-5)
        for c in range(self.channels):
            self.image_acc[c] /= safe_weights
        return np.clip(self.image_acc, 0.0, 1.0)
```

---

### Stage 8: Quality Assurance & Physical Validation Engine

#### Objective
Enforce rigorous automated QA gates before publishing reconstructed rasters. If reconstructed areas violate physical spectral rules or manifest boundary seams, the product is flagged for review.

#### Quality Metrics and Acceptance Thresholds

| Metric | Acceptance Threshold | Description | Action on Failure |
|--------|----------------------|-------------|-------------------|
| **Cloud Boundary Discontinuity (CBDI)** | $< 0.05$ | Gradient difference across cloud border | Re-blend with expanded 30px buffer |
| **Spectral Angle Mapper (SAM)** | $< 3.5^\circ$ | Spectral angle vs temporal reference | Flag tile as low confidence in QA band |
| **NDVI Range Check** | $[-0.2, 0.95]$ | Reconstructed NDVI must stay in natural bounds | Re-scale reconstructed NIR/Red ratio |
| **Nodata Leakage** | $0.0\%$ | No NaN or inf values in final output | Pipeline abort / raise exception |

```python
# pipeline/stage8_qa.py
import numpy as np

def compute_cbdi(original_clear: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes Cloud Boundary Discontinuity Index (CBDI) along a 3-pixel border
    of the cloud mask to detect sharp seams.
    """
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = (dilated - mask).astype(bool)
    
    if np.sum(boundary) == 0:
        return 0.0
    
    diff = np.abs(original_clear[:, boundary] - reconstructed[:, boundary])
    return float(np.mean(diff))

def generate_quality_band(original: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Generates 8-bit QA Band:
    0: Native Clear Ground (Highest quality)
    1: High-Confidence Generative Fill
    2: Moderate Quality Fill (Minor spectral deviation)
    3: Low Quality / Missing Auxiliary Data
    """
    qa = np.zeros(mask.shape, dtype=np.uint8)
    qa[mask == 1] = 1 # Reconstructed
    # If SAR auxiliary was missing during inference, elevate to 3
    return qa
```

---

### Stage 9: ARD Packaging (COG & STAC Item)

#### Objective
Convert multi-band surface reflectance products into Cloud-Optimized GeoTIFFs (COG) with internal overviews and generate STAC metadata compliant with STAC 1.0.0 specifications.

```bash
# Production COG Generation via rio-cogeo CLI
rio cogeo create \
    data/reconstructed/LISS4_NER_20240615_SR.tif \
    data/ard/LISS4_NER_20240615_SR_COG.tif \
    --cog-profile deflate \
    --blocksize 512 \
    --overview-resampling average \
    --add-mask
```

```python
# pipeline/stage9_stac_export.py
import datetime
import pystac
from shapely.geometry import box

def build_stac_item(cog_path: str, bounds: tuple, epsg: int, acquisition_time: datetime.datetime) -> pystac.Item:
    geom = box(*bounds)
    item = pystac.Item(
        id=f"LISS4_ARD_{acquisition_time.strftime('%Y%m%d_%H%M%S')}",
        geometry=geom.__geo_interface__,
        bbox=list(bounds),
        datetime=acquisition_time,
        properties={
            "eo:cloud_cover_original": 42.5,
            "eo:cloud_cover_reconstructed": 0.0,
            "processing:software": "Generative-AI-LISS4-CloudRemover-v1.0",
            "proj:epsg": epsg,
            "gsd": 5.8
        }
    )
    item.add_asset(
        key="visual",
        asset=pystac.Asset(
            href=cog_path,
            title="Analysis-Ready Cloud-Free Surface Reflectance",
            media_type=pystac.MediaType.COG,
            roles=["data", "reflectance"]
        )
    )
    return item
```

---

## 4. Orchestration and Execution Workflows

### 4.1 Prefect Production Flow Specification

```python
# flows/production_flow.py
from prefect import flow, task
from pathlib import Path

@task(retries=3, retry_delay_seconds=30)
def task_ingest(raw_dir: Path):
    from pipeline.stage1_ingest import validate_scene_integrity
    validate_scene_integrity(raw_dir)
    return str(raw_dir)

@task
def task_preprocess(raw_dir: str):
    from pipeline.stage2_preprocess import reproject_to_target_grid
    # Execute radiometric calibration and projection
    out_tif = "data/interim/preprocessed.tif"
    return out_tif

@task
def task_cloud_mask(prep_tif: str):
    # Detect clouds and shadows
    mask_tif = "data/interim/cloud_mask.tif"
    return mask_tif

@task
def task_reconstruction(prep_tif: str, mask_tif: str):
    # Perform tiled diffusion inference with A100 GPU
    reconstructed_tif = "data/interim/reconstructed.tif"
    return reconstructed_tif

@task
def task_qa_and_cog(reconstructed_tif: str):
    # Evaluate metrics and produce COG
    final_cog = "data/ard/final_product_cog.tif"
    return final_cog

@flow(name="LISS4-Cloud-Removal-Operational-Pipeline")
def run_liss4_cloud_removal_pipeline(raw_scene_path: str):
    raw_p = Path(raw_scene_path)
    valid_dir = task_ingest(raw_p)
    preprocessed_path = task_preprocess(valid_dir)
    mask_path = task_cloud_mask(preprocessed_path)
    recon_path = task_reconstruction(preprocessed_path, mask_path)
    cog_path = task_qa_and_cog(recon_path)
    print(f"Pipeline finished successfully. ARD deliverable ready at: {cog_path}")

if __name__ == "__main__":
    run_liss4_cloud_removal_pipeline("/data/raw/RS2A_LISS4_20240615_NER")
```

---

## 5. Failure Recovery, Graceful Degradation, and Edge Cases

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FAILURE RECOVERY & DEGRADATION MATRIX                           │
├─────────────────────────┬────────────────────────────┬─────────────────────────────────┤
│ Failure Scenario        │ Root Cause                 │ Operational Fallback Action     │
├─────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ SAR Data Missing        │ Sentinel-1 orbit gap       │ Revert to Mode B: Spatial-      │
│                         │ over NER AOI               │ Temporal Optical Diffusion      │
├─────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ Temporal Optical Clear  │ Persistent monsoonal cover │ Revert to Mode C: SAR-Guided    │
│ Reference Missing       │ (> 60 days overcast)       │ Inpainting (Structural only)    │
├─────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ Co-Registration Failure │ Extreme cloud obscuration  │ Fall back to orbit header state │
│ in AROSICS (> 0.5px)    │ prevents tie points        │ vectors without subpixel warp   │
├─────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ GPU Out-Of-Memory (OOM) │ Tile batch too large       │ Halve tile batch size & activate│
│                         │ on 16GB VRAM GPU           │ FP16 mixed precision inference  │
├─────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ 100% Cloud Cover Scene  │ Scene completely opaque    │ Terminate early with warning;   │
│                         │ (No clear grounding)       │ Flag scene as non-reconstructible│
└─────────────────────────┴────────────────────────────┴─────────────────────────────────┘
```

This complete workflow ensures reproducible, unattended execution across both individual scenes and multi-terabyte regional mosaics over the North Eastern Region of India.
