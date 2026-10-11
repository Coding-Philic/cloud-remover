# 04 — Project Architecture
## Generative AI-Based Cloud Removal for LISS-IV Imagery

---

## 1. End-to-End Architecture Overview

The complete system architecture spans four major planes: **Data Plane**, **Training Plane**, **Inference Plane**, and **Serving Plane**.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                          END-TO-END ARCHITECTURE                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐    ║
║  │                      DATA PLANE                                  │    ║
║  │                                                                  │    ║
║  │   Bhoonidhi ──┐                                                  │    ║
║  │   Copernicus ─┤──▶ Download ──▶ Preprocess ──▶ Co-register ──┐  │    ║
║  │   USGS DEM ───┘                                               │  │    ║
║  │                                               ┌───────────────▼──┤    ║
║  │                                               │  Tile Store      │    ║
║  │                                               │  (512×512 NPZ)   │    ║
║  │                                               │  + Split CSVs    │    ║
║  │                                               └───────┬──────────┤    ║
║  └───────────────────────────────────────────────────────┼──────────┘    ║
║                                                           │              ║
║  ┌───────────────────────────────────────────────────────▼──────────┐    ║
║  │                     TRAINING PLANE                                │    ║
║  │                                                                   │    ║
║  │   DataModule ──▶ Model (Diffusion/GAN/Transformer) ──▶ Trainer   │    ║
║  │                                                           │       │    ║
║  │   Losses ◀──────────────────────────────────────────────┘       │    ║
║  │        │                                                          │    ║
║  │        ▼                                                          │    ║
║  │   Checkpoints + Metrics Log (TensorBoard / W&B)                  │    ║
║  └───────────────────────────────────────────┬──────────────────────┘    ║
║                                               │                          ║
║  ┌───────────────────────────────────────────▼──────────────────────┐    ║
║  │                    INFERENCE PLANE                                │    ║
║  │                                                                   │    ║
║  │   Input Scene ──▶ Cloud Mask ──▶ Tiler ──▶ Model ──▶ Blender    │    ║
║  │                                                          │        │    ║
║  │                                                          ▼        │    ║
║  │                                              Cloud-Free GeoTIFF   │    ║
║  └───────────────────────────────────────────┬──────────────────────┘    ║
║                                               │                          ║
║  ┌───────────────────────────────────────────▼──────────────────────┐    ║
║  │                     SERVING PLANE                                 │    ║
║  │                                                                   │    ║
║  │   FastAPI ──▶ Queue ──▶ GPU Worker ──▶ Storage ──▶ QGIS/Web     │    ║
║  └──────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Data Architecture

### 2.1 Data Sources and Their Relationships

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE MAP                               │
│                                                                      │
│   PRIMARY DATA                                                       │
│   ┌─────────────────────────────────────────┐                       │
│   │  LISS-IV (Bhoonidhi)                    │                       │
│   │  • 5.8 m resolution                     │                       │
│   │  • 3 bands: Green, Red, NIR             │                       │
│   │  • 70 km swath                          │                       │
│   │  • Level-2 (radiometrically corrected)  │                       │
│   └─────────────────────────────────────────┘                       │
│                                                                      │
│   AUXILIARY DATA                                                     │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐   │
│   │ Sentinel-1 SAR   │ │ Sentinel-2 L2A   │ │ DEM (SRTM/COP)  │   │
│   │ • 10 m IW GRD    │ │ • 10 m optical   │ │ • 30 m elevation │   │
│   │ • VV/VH polar.   │ │ • 13 bands       │ │ • Cloud-free     │   │
│   │ • Cloud-free!    │ │ • More bands     │ │ • Static         │   │
│   │ • Texture info   │ │ • Spectral ref   │ │ • Terrain info   │   │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘   │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Temporal Reference (Previous cloud-free LISS-IV / S2)        │  │
│   │ • Same area, different date (within ±30 days)                │  │
│   │ • Provides surface context under clouds                      │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Processing Pipeline

```
STAGE 1: RAW DATA ACQUISITION
────────────────────────────
  LISS-IV scenes ──▶ data/raw/liss4/{scene_id}/
  Sentinel-1 GRD  ──▶ data/raw/sentinel1/{product_id}/
  Sentinel-2 L2A  ──▶ data/raw/sentinel2/{product_id}/
  SRTM/COP DEM    ──▶ data/raw/dem/

STAGE 2: PREPROCESSING
─────────────────────
  Each source independently:
  ┌───────────────────────────────────────────────────────────────┐
  │ LISS-IV:    Band merge → Calibrate → Reproject UTM → Clip AOI │
  │ Sentinel-1: Apply orbit → Calibrate → Terrain correct →      │
  │             Speckle filter → Convert dB → Reproject           │
  │ Sentinel-2: Band select (B2,B3,B4,B8) → Reproject →         │
  │             Resample 10m→5.8m                                 │
  │ DEM:        Reproject → Resample → Normalize [0,1]           │
  └───────────────────────────────────────────────────────────────┘

STAGE 3: CO-REGISTRATION & STACKING
───────────────────────────────────
  All sources aligned to LISS-IV's grid:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Reference: LISS-IV (master CRS + resolution)            │
  │ 2. Warp each auxiliary to LISS-IV grid (gdal.Warp)         │
  │ 3. Sub-pixel registration (phase correlation / AROSICS)    │
  │ 4. Stack into 14-channel array                             │
  │ 5. Generate cloud mask                                     │
  └─────────────────────────────────────────────────────────────┘

STAGE 4: TILING
──────────────
  ┌─────────────────────────────────────────────────────────────┐
  │ Scene (14ch, H, W) → Tiles (14ch, 512, 512) with overlap   │
  │ Stride = 512 - 64 = 448 pixels                             │
  │ Edge tiles: zero-padded to 512×512                          │
  │ Output: data/processed/tiles/{scene_id}/{tile_id}.npz       │
  └─────────────────────────────────────────────────────────────┘
```

### 2.3 Directory Layout on Disk

```
data/
├── raw/
│   ├── liss4/
│   │   ├── LISS4_NER_20240615_001/
│   │   │   ├── BAND2_GREEN.tif        (original download)
│   │   │   ├── BAND3_RED.tif
│   │   │   ├── BAND4_NIR.tif
│   │   │   └── metadata.xml
│   │   └── LISS4_NER_20240620_002/
│   │       └── ...
│   │
│   ├── sentinel1/
│   │   └── S1A_IW_GRDH_20240615_NER/
│   │       └── measurement/
│   │           ├── s1a-iw-grd-vv-*.tiff
│   │           └── s1a-iw-grd-vh-*.tiff
│   │
│   ├── sentinel2/
│   │   └── S2A_MSIL2A_20240614_T46QEL/
│   │       └── GRANULE/.../IMG_DATA/R10m/
│   │           ├── T46QEL_20240614_B02_10m.jp2
│   │           ├── T46QEL_20240614_B03_10m.jp2
│   │           ├── T46QEL_20240614_B04_10m.jp2
│   │           └── T46QEL_20240614_B08_10m.jp2
│   │
│   └── dem/
│       └── srtm_46_07.tif
│
├── processed/
│   ├── scenes/                    (full co-registered scenes)
│   │   ├── scene_001_stack.tif    (14-band GeoTIFF)
│   │   └── scene_001_mask.tif     (cloud mask)
│   │
│   └── tiles/                     (512×512 tiles)
│       ├── scene_001/
│       │   ├── tile_r00_c00.npz   (keys: "stack", "gt", "mask")
│       │   ├── tile_r00_c01.npz
│       │   └── ...
│       └── scene_002/
│           └── ...
│
└── splits/
    ├── train.csv     (tile_id, scene_id, cloud_pct, has_sar, ...)
    ├── val.csv
    └── test.csv
```

---

## 3. Model Architecture — Detailed Diagrams

### 3.1 Primary: Conditional Latent Diffusion Model

```
INPUT PREPARATION
─────────────────
Cloudy LISS-IV (3ch) ─┐
Cloud Mask (1ch) ──────┤
SAR VV/VH (2ch) ───────┤──▶ STACK (14ch, 512, 512)
Sentinel-2 (4ch) ──────┤
DEM (1ch) ─────────────┤
Temporal Ref (3ch) ────┘

TRAINING FORWARD PASS
─────────────────────

Step 1: Encode GT to latent
  GT (3, 512, 512) ──▶ [VAE Encoder (frozen)] ──▶ z₀ (4, 64, 64)

Step 2: Add noise
  z₀ + ε (random noise) at timestep t ──▶ zₜ (4, 64, 64)

Step 3: Encode condition
  Stack (14, 512, 512) ──▶ [Fusion Encoder] ──▶ cond_features (256, 64, 64)

Step 4: Predict noise
  zₜ + t + cond_features ──▶ [Conditional U-Net] ──▶ ε_pred (4, 64, 64)

Step 5: Loss
  L = MSE(ε_pred, ε) + λ₁·L1 + λ₂·(1-SSIM) + λ₃·LPIPS + λ₄·SAM

INFERENCE (DDIM Sampling)
─────────────────────────

  z_T ~ N(0, I)  ←── Start from random noise
      │
      ▼
  for t = T, T-Δ, T-2Δ, ..., 0:     (20 steps)
      ε_pred = UNet(zₜ, t, cond)
      zₜ₋₁ = DDIM_step(zₜ, ε_pred, t)
      │
      ▼
  z₀_pred (4, 64, 64)
      │
      ▼
  [VAE Decoder] ──▶ Reconstructed (3, 512, 512)
```

### 3.2 U-Net Denoiser Architecture

```
                     INPUT: zₜ (4, 64, 64) + t (timestep embedding)
                              │
                    ┌─────────▼──────────┐
                    │   Time Embedding    │
                    │   MLP: t → 512-dim  │
                    └─────────┬──────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │              ENCODER                   │
          │                                        │
          │  Down Block 1: 4→128ch, 64×64          │ ──── skip₁
          │  Down Block 2: 128→256ch, 32×32        │ ──── skip₂
          │  Down Block 3: 256→512ch, 16×16 + Attn │ ──── skip₃
          │  Down Block 4: 512→512ch, 8×8 + Attn   │ ──── skip₄
          │                                        │
          └───────────────────┬───────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │            BOTTLENECK                   │
          │  512ch, 8×8 + Self-Attention            │
          │  + Cross-Attention (cond_features)      │
          └───────────────────┬───────────────────┘
                              │
          ┌───────────────────▼───────────────────┐
          │              DECODER                   │
          │                                        │
          │  Up Block 4: 512+512→512ch, 16×16      │ ◀── skip₄
          │  Up Block 3: 512+256→256ch, 32×32      │ ◀── skip₃
          │  Up Block 2: 256+128→128ch, 64×64      │ ◀── skip₂
          │  Up Block 1: 128+4→4ch, 64×64          │ ◀── skip₁
          │                                        │
          └───────────────────┬───────────────────┘
                              │
                    OUTPUT: ε_pred (4, 64, 64)
```

### 3.3 GAN Architecture (Baseline)

```
GENERATOR (U-Net + SPADE)
─────────────────────────
  Input: 14ch stack ──▶ Encoder (Conv↓) ──▶ Bottleneck ──▶ Decoder (ConvTranspose↑ + SPADE) ──▶ Output: 3ch

DISCRIMINATOR (PatchGAN)
────────────────────────
  Input: 3ch (real or fake) + 14ch (condition) = 17ch
      │
      ▼
  Conv(17→64) → Conv(64→128) → Conv(128→256) → Conv(256→512) → Conv(512→1)
      │
      ▼
  Patch-level real/fake prediction map (30×30 for 512 input)
```

### 3.4 Transformer Architecture (Baseline)

```
INPUT: 14ch stack (14, 512, 512)
    │
    ▼
┌────────────────────────────────┐
│  Patch Embedding (4×4 patches) │
│  14ch → 96-dim tokens          │
│  128×128 tokens                │
└────────────┬───────────────────┘
             │
    ┌────────▼────────┐
    │ Swin Stage 1    │  128×128, 96-dim, window=8
    │ (6 Swin Blocks) │
    ├─────────────────┤
    │ Patch Merging    │  64×64, 192-dim
    ├─────────────────┤
    │ Swin Stage 2    │  64×64, 192-dim
    │ (6 Swin Blocks) │
    ├─────────────────┤
    │ Patch Merging    │  32×32, 384-dim
    ├─────────────────┤
    │ Swin Stage 3    │  32×32, 384-dim
    │ (6 Swin Blocks) │
    ├─────────────────┤
    │ Patch Merging    │  16×16, 768-dim
    ├─────────────────┤
    │ Swin Stage 4    │  16×16, 768-dim
    │ (6 Swin Blocks) │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Pixel Shuffle   │  Upsample back to 512×512
    │  Decoder         │  768→3 channels
    └────────┬────────┘
             │
    OUTPUT: Reconstructed (3, 512, 512)
```

---

## 4. Training Infrastructure

### 4.1 Training Pipeline Flow

```
┌─────────────┐
│ Config YAML  │
│ (Hydra)     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│  DataModule      │     │  Model            │
│  • Load splits   │     │  • Diffusion/GAN/ │
│  • Create loaders│     │    Transformer    │
│  • Augmentations │     │  • Loss functions │
└──────┬───────────┘     └──────┬───────────┘
       │                        │
       └────────┬───────────────┘
                │
       ┌────────▼────────┐
       │  PyTorch         │
       │  Lightning       │
       │  Trainer         │
       │                  │
       │  • Mixed Prec.   │
       │  • Grad Clip     │
       │  • Checkpointing │
       │  • Early Stop    │
       └────────┬────────┘
                │
       ┌────────▼────────┐      ┌──────────────────┐
       │  Callbacks       │      │  Logging          │
       │  • ModelCkpt    │──────▶│  • TensorBoard   │
       │  • LRMonitor    │      │  • W&B (optional) │
       │  • EarlyStopping│      │  • CSV Logger     │
       └────────┬────────┘      └──────────────────┘
                │
       ┌────────▼────────┐
       │  Best Checkpoint │
       │  (*.ckpt)        │
       └─────────────────┘
```

### 4.2 Hardware Requirements

| Configuration | GPU | VRAM | RAM | Storage | Training Time (est.) |
|--------------|-----|------|-----|---------|---------------------|
| **Minimum** | 1× RTX 3060 | 12 GB | 32 GB | 500 GB SSD | ~7 days (batch_size=2, grad_accum=8) |
| **Recommended** | 1× RTX 4090 / A100 | 24–40 GB | 64 GB | 1 TB SSD | ~3 days (batch_size=8) |
| **Optimal** | 4× A100 | 40 GB each | 128 GB | 2 TB NVMe | ~18 hours (DDP, batch_size=32) |

> **Memory-saving strategies:** Gradient checkpointing, mixed precision (FP16), gradient accumulation, frozen VAE

---

## 5. Inference Architecture

### 5.1 Single-Scene Inference Pipeline

```
INPUT: Cloudy LISS-IV Scene (GeoTIFF)
    │
    ├── Read with Rasterio (preserve CRS, transform, metadata)
    │
    ▼
CLOUD MASK GENERATION
    │
    ├── Rule-based pre-filter (bright pixel threshold)
    ├── DL segmentation (U-Net) → probability map
    ├── Threshold → binary mask
    ├── Morphological cleanup (dilate cloud boundary by 5px)
    │
    ▼
AUXILIARY DATA LOADING (if available)
    │
    ├── Find matching Sentinel-1 scene (±5 days)
    ├── Find matching Sentinel-2 scene (±10 days)
    ├── Load DEM tile
    ├── Find temporal reference (±30 days, cloud-free)
    ├── Co-register all to LISS-IV grid
    │
    ▼
BUILD INPUT STACK (14 channels)
    │
    ├── [cloudy_liss4(3), mask(1), sar(2), s2(4), dem(1), temporal(3)]
    ├── Zero-fill missing modalities
    │
    ▼
TILE SCENE → 512×512 PATCHES
    │
    ├── Stride = 448 (overlap = 64)
    ├── Pad edges to 512
    ├── N tiles ≈ ceil(H/448) × ceil(W/448)
    │
    ▼
BATCHED MODEL INFERENCE
    │
    ├── Process tiles in batches of 8
    ├── Each tile: model.generate(tile, steps=20) → reconstructed tile
    ├── Apply mask compositing: output = pred * mask + original * (1-mask)
    │
    ▼
MERGE & BLEND
    │
    ├── Gaussian weight map at overlaps
    ├── Weighted average in overlap regions
    ├── Crop padding from edges
    │
    ▼
POST-PROCESS
    │
    ├── Histogram matching at seam boundaries (optional)
    ├── Clip values to valid range [0, 1]
    ├── Generate QA mask (0=original, 1=reconstructed)
    │
    ▼
EXPORT
    │
    ├── Write GeoTIFF with original CRS, transform
    ├── Write QA mask GeoTIFF
    ├── Write metrics JSON (if GT available)
    │
    ▼
OUTPUT: Cloud-Free GeoTIFF + QA Mask + Metrics
```

### 5.2 Optimization Strategies for Inference

| Strategy | Speedup | Trade-off |
|----------|---------|-----------|
| **DDIM (fewer steps)** | 5–10× vs DDPM | Minimal quality drop (< 0.5 dB PSNR) at 20 steps |
| **Mixed Precision (FP16)** | ~2× throughput | Negligible quality impact |
| **ONNX Runtime** | ~2–3× vs PyTorch eager | One-time export cost; static input shapes |
| **TensorRT** | ~3–5× vs PyTorch eager | NVIDIA-only; compilation time |
| **Batched tile processing** | Linear scaling | Limited by GPU memory |
| **Tile caching** | Avoids redundant computation | Extra storage |

---

## 6. Serving Architecture (Prototype)

```
┌──────────────────────────────────────────────────┐
│                   CLIENT LAYER                    │
│                                                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐   │
│  │ QGIS Plugin │  │ Python CLI │  │ cURL/API │   │
│  └──────┬─────┘  └──────┬─────┘  └────┬─────┘   │
│         │               │              │          │
└─────────┼───────────────┼──────────────┼──────────┘
          │               │              │
          └───────────────┼──────────────┘
                          │
                ┌─────────▼──────────┐
                │    FastAPI Server    │
                │    (Port 8000)      │
                │                     │
                │  /reconstruct       │
                │  /reconstruct/sync  │
                │  /mask/detect       │
                │  /health            │
                └─────────┬──────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
     ┌────────▼───┐  ┌───▼────┐  ┌──▼──────────┐
     │ Preprocessor│  │ Cloud  │  │ Reconstruction│
     │ Worker     │  │ Masker │  │ GPU Worker    │
     └────────┬───┘  └───┬────┘  └──┬──────────┘
              │           │          │
              └───────────┼──────────┘
                          │
                ┌─────────▼──────────┐
                │   Output Storage    │
                │   (Local / S3)     │
                └────────────────────┘
```

---

## 7. Reproducibility Architecture

| Aspect | Tool | How |
|--------|------|-----|
| **Code versioning** | Git | All source code, configs, docs tracked |
| **Data versioning** | DVC (optional) | Large data files tracked via `.dvc` pointer files |
| **Config management** | YAML files | All hyperparameters in versioned config files |
| **Random seeds** | `seed_everything(42)` | PyTorch, NumPy, Python random all seeded |
| **Environment** | Docker + `requirements.txt` | Exact Python package versions pinned |
| **Experiment tracking** | TensorBoard / W&B | Metrics, hyperparameters, artifacts logged per run |
| **Checkpoints** | PyTorch Lightning | Top-K checkpoints saved by validation metric |

---

## 8. CRS and Geospatial Integrity

A critical architectural concern is maintaining **geospatial integrity** throughout the pipeline:

```
INPUT CRS (from LISS-IV)
    │
    ├── Read and store: CRS, Affine Transform, nodata, dtype
    │
    ▼
PROCESSING (internally)
    │
    ├── All operations in UTM projection (metric units)
    ├── Pixel coordinates tracked via affine transform
    ├── No CRS conversion during model inference
    │
    ▼
OUTPUT
    │
    ├── Write with EXACT same CRS, transform, resolution
    ├── Pixel-for-pixel spatial correspondence with input
    ├── Band order preserved: Green, Red, NIR
    └── Nodata value: 0 (same as input)
```

**Key Rules:**
1. Never modify the CRS during inference — input and output must match exactly
2. Tile coordinates are tracked via row/col indices relative to the scene origin
3. Auxiliary data is always warped TO the LISS-IV grid, never the reverse
4. Float32 internally for processing, UInt16 for final export (matching LISS-IV native)

---

## 9. Technology Stack Summary

```
LAYER               TECHNOLOGY                          PURPOSE
─────               ──────────                          ───────
Language            Python 3.10+                        Core language
Deep Learning       PyTorch 2.x                         Model development
                    PyTorch Lightning 2.x               Training framework
                    HuggingFace Diffusers               Diffusion model utilities
Geospatial          GDAL 3.x                            Low-level raster I/O
                    Rasterio 1.3+                       Pythonic raster I/O
                    PyProj                              CRS transformations
                    Geopandas                           Vector data handling
Computer Vision     OpenCV 4.x                          Image processing
                    scikit-image                        Image metrics, filters
                    Albumentations                      Data augmentation
Data Management     NumPy                               Array operations
                    Pandas                              Tabular data (splits, metadata)
                    SQLite                              Tile/experiment tracking DB
Visualization       Matplotlib                          Plotting
                    TensorBoard                         Training monitoring
Serving             FastAPI                             REST API
                    ONNX Runtime (optional)             Optimized inference
Containerization    Docker                              Reproducible environments
Testing             pytest                              Unit/integration tests
```

---

> **Next →** Read [05-data-sources-and-downloading.md](./05-data-sources-and-downloading.md) for exact data download commands.
