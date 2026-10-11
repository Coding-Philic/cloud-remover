# 02 — High-Level Design (HLD)
## Generative AI-Based Cloud Removal for LISS-IV Imagery

---

## 1. Purpose

This document defines the **system-level architecture** of the cloud removal framework — the major subsystems, how they interact, what data flows between them, and the deployment strategy. Detailed class-level and API-level design is covered in [LLD (Doc 03)](./03-low-level-design-LLD.md).

---

## 2. System Context Diagram

The framework operates within a broader ecosystem of data providers, compute infrastructure, and end users.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SYSTEMS                            │
│                                                                      │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│  │  Bhoonidhi  │  │ Copernicus Data  │  │   USGS / NASA    │        │
│  │  (LISS-IV)  │  │ Space (S1 / S2)  │  │  (SRTM DEM)     │        │
│  └──────┬──────┘  └────────┬─────────┘  └────────┬─────────┘        │
│         │                  │                      │                  │
└─────────┼──────────────────┼──────────────────────┼──────────────────┘
          │                  │                      │
          ▼                  ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  CLOUD REMOVAL FRAMEWORK                             │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │  Data     │  │  Cloud   │  │ Generative│  │  Post-Processing │    │
│  │ Ingestion │→ │  Mask    │→ │  Model   │→ │  & Export        │    │
│  │ & Preproc │  │ Generator│  │  Engine  │  │                  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          END USERS                                   │
│                                                                      │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ Geospatial │  │  Disaster Mgmt   │  │  Research / Academic │    │
│  │  Analysts  │  │  Agencies        │  │  Community           │    │
│  └────────────┘  └──────────────────┘  └──────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Functional Requirements

| FR-ID | Requirement | Priority | Description |
|-------|-------------|----------|-------------|
| FR-01 | Ingest LISS-IV imagery | P0 (Must) | Accept LISS-IV Level-2 GeoTIFF files (3 bands: G, R, NIR at 5.8 m) |
| FR-02 | Auto-detect cloud and cloud-shadow regions | P0 | Generate binary/probabilistic cloud mask without manual intervention |
| FR-03 | Ingest and preprocess auxiliary data | P0 | Handle Sentinel-1 SAR (VV/VH), Sentinel-2 L2A, DEM, and temporal references |
| FR-04 | Co-register multi-source data | P0 | Align all inputs to a common grid (LISS-IV's CRS and 5.8 m resolution) |
| FR-05 | Reconstruct cloud-covered regions | P0 | Use Generative AI to fill masked regions with realistic surface imagery |
| FR-06 | Preserve spectral consistency | P0 | Ensure band-to-band relationships in reconstructed regions match natural imagery |
| FR-07 | Export cloud-free GeoTIFF | P0 | Output georeferenced, analysis-ready imagery with metadata |
| FR-08 | Tile-based processing | P1 (Should) | Process large scenes via overlapping tiles with seam blending |
| FR-09 | Batch processing mode | P1 | Process multiple scenes sequentially or in parallel |
| FR-10 | Quantitative evaluation metrics | P1 | Compute PSNR, SSIM, SAM, LPIPS, FID on test data |
| FR-11 | Visual QA output | P2 (Nice) | Generate side-by-side comparison images and error maps |
| FR-12 | Model comparison framework | P2 | Support swapping different model architectures (GAN, Diffusion, Transformer) |

---

## 4. Non-Functional Requirements

| NFR-ID | Requirement | Target |
|--------|-------------|--------|
| NFR-01 | **Inference Speed** | < 5 seconds per 512×512 tile on NVIDIA T4 GPU |
| NFR-02 | **Reconstruction Quality** | PSNR > 30 dB, SSIM > 0.90 on NER test set |
| NFR-03 | **Spectral Fidelity** | SAM < 3.0°, per-band CC > 0.93 |
| NFR-04 | **Scalability** | Handle scenes up to 12,000 × 12,000 pixels via tiling |
| NFR-05 | **Reproducibility** | Fixed random seeds, versioned configs, containerized environment |
| NFR-06 | **Modularity** | Model engine swappable without changing data pipeline or post-processing |
| NFR-07 | **Georeferencing Integrity** | Output CRS and geotransform exactly match input |
| NFR-08 | **Memory Efficiency** | Training fits on single 24 GB GPU (with gradient checkpointing) |

---

## 5. Subsystem Decomposition

The framework is divided into **five major subsystems**:

### 5.1 Subsystem Overview

```
┌───────────────────────────────────────────────────────────────┐
│                    CLOUD REMOVAL FRAMEWORK                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SUBSYSTEM 1: DATA INGESTION & PREPROCESSING            │  │
│  │  - Download/load raw imagery (LISS-IV, S1, S2, DEM)     │  │
│  │  - Radiometric calibration, atmospheric correction       │  │
│  │  - Reprojection, resampling to common grid               │  │
│  │  - Co-registration of multi-source data                  │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SUBSYSTEM 2: CLOUD MASK GENERATION                     │  │
│  │  - Rule-based detection (thresholding, spectral tests)  │  │
│  │  - Deep learning segmentation (U-Net / DeepLabv3+)      │  │
│  │  - Cloud shadow estimation (geometric + spectral)       │  │
│  │  - Mask refinement and morphological post-processing    │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SUBSYSTEM 3: GENERATIVE RECONSTRUCTION ENGINE          │  │
│  │  - Model architectures: GAN / Diffusion / Transformer   │  │
│  │  - Multi-modal fusion (SAR + optical + DEM + temporal)  │  │
│  │  - Conditional generation: mask + context → cloud-free  │  │
│  │  - Tiled inference with overlap                         │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SUBSYSTEM 4: POST-PROCESSING & EXPORT                  │  │
│  │  - Seam blending at tile boundaries                     │  │
│  │  - Histogram matching / color harmonization             │  │
│  │  - GeoTIFF export with CRS and metadata                 │  │
│  │  - QA mask generation (reconstructed vs. original)      │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  SUBSYSTEM 5: EVALUATION & COMPARISON                   │  │
│  │  - Quantitative metrics (PSNR, SSIM, SAM, LPIPS, FID)  │  │
│  │  - Visual comparison outputs                            │  │
│  │  - Architecture benchmarking                            │  │
│  │  - Downstream task validation (NDVI accuracy)           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Subsystem Details

| Subsystem | Key Technologies | Inputs | Outputs |
|-----------|-----------------|--------|---------|
| **Data Ingestion** | GDAL, Rasterio, sentinelsat, cdsetool, gdown | Raw LISS-IV TIFFs, S1 GRD, S2 L2A, DEM | Co-registered, normalized multi-source stack |
| **Cloud Masking** | PyTorch, segmentation-models-pytorch, scikit-image | Preprocessed LISS-IV image | Binary cloud mask (0 = clear, 1 = cloud/shadow) |
| **Reconstruction** | PyTorch, Diffusers, torchvision, xformers | Cloudy image + mask + auxiliary data | Reconstructed cloud-free image (3 bands) |
| **Post-Processing** | OpenCV, NumPy, Rasterio, GDAL | Reconstructed tiles + original scene | Full-scene cloud-free GeoTIFF + QA mask |
| **Evaluation** | torchmetrics, scikit-image, matplotlib | Predictions + ground truth | Metrics JSON, comparison plots, benchmark report |

---

## 6. Data Flow — End to End

```
                    ┌──────────────────┐
                    │   RAW LISS-IV    │
                    │   (Cloudy Scene) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   PREPROCESS     │
                    │ • Calibrate      │
                    │ • Reproject UTM   │
                    │ • Normalize [0,1] │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼───────┐     │    ┌─────────▼────────┐
    │  CLOUD MASKING   │     │    │ FETCH AUXILIARY   │
    │ (Generate Mask)  │     │    │ S1, S2, DEM,     │
    │                  │     │    │ Temporal Ref      │
    └─────────┬───────┘     │    └─────────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  CO-REGISTRATION  │
                    │  & STACKING       │
                    │ Build 14-channel  │
                    │ input tensor      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    TILING         │
                    │ 512×512 patches   │
                    │ with 64px overlap │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │   GENERATIVE     │
                    │   MODEL          │
                    │ (Inference per   │
                    │  tile)           │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  MOSAIC &        │
                    │  SEAM BLENDING   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  EXPORT          │
                    │ • Cloud-free     │
                    │   GeoTIFF        │
                    │ • QA Mask        │
                    │ • Metrics JSON   │
                    └──────────────────┘
```

### Channel Composition of Input Stack

When all auxiliary data is available, the model receives a **14-channel** input:

| Channels | Source | Description |
|----------|--------|-------------|
| Ch 1–3 | LISS-IV | Cloudy image (Green, Red, NIR) — cloud pixels present |
| Ch 4 | Cloud Mask | Binary mask (1 = cloud/shadow, 0 = clear) |
| Ch 5–6 | Sentinel-1 SAR | VV and VH polarization backscatter (dB, normalized) |
| Ch 7–10 | Sentinel-2 | Blue, Green, Red, NIR bands (10 m resampled to 5.8 m) |
| Ch 11 | DEM | Elevation (normalized) |
| Ch 12–14 | Temporal Reference | Previous cloud-free LISS-IV (G, R, NIR) |

> **Graceful Degradation:** When auxiliary data is unavailable, those channels are zero-filled and the model adapts via learned channel attention.

---

## 7. Model Strategy (HLD Level)

At the HLD level, we select the model family and comparison strategy. Detailed architecture internals are in [Doc 06](./06-model-training-and-algorithms.md).

### 7.1 Primary Architecture: Conditional Diffusion Model

**Why Diffusion?**
- Produces highest-quality texture and detail among generative models
- Conditional generation (on mask + auxiliary inputs) is well-supported
- More stable training than GANs (no mode collapse)
- State-of-the-art results on image inpainting benchmarks

### 7.2 Baseline Architectures for Comparison

| Architecture | Type | Strength | Weakness |
|-------------|------|----------|----------|
| **Pix2Pix / SpaGAN** | GAN (Conditional) | Fast inference (~50 ms/tile), proven for satellite image translation | Mode collapse risk, less texture detail, training instability |
| **Diffusion (DDPM/DDIM)** | Denoising Diffusion | Best texture/spectral quality, stable training | Slow inference (multiple denoising steps), high VRAM for training |
| **Transformer (SwinIR / Restormer)** | Vision Transformer | Excellent long-range context modeling, no iterative sampling | High memory, may need more training data for generalization |

### 7.3 Inference Modes

The framework supports three inference modes based on data availability:

| Mode | Available Inputs | Expected Quality |
|------|-----------------|-----------------|
| **Mode A — Full Fusion** | LISS-IV + Mask + SAR + S2 + DEM + Temporal | Best (SAR provides cloud-free structural information) |
| **Mode B — Temporal Only** | LISS-IV + Mask + Temporal Reference | Good (temporal reference provides surface context) |
| **Mode C — Single Image** | LISS-IV + Mask only | Acceptable (relies purely on spatial context from surrounding clear pixels) |

---

## 8. Deployment Strategy (HLD Level)

### 8.1 Development & Training Environment

```
┌──────────────────────────────────────┐
│        TRAINING ENVIRONMENT          │
│                                      │
│  • GPU Machine (A100/V100/RTX 4090)  │
│  • PyTorch + Lightning               │
│  • Data stored on local SSD / NFS    │
│  • W&B / TensorBoard for tracking    │
│  • DVC for data versioning           │
│                                      │
└──────────────────────────────────────┘
```

### 8.2 Inference / Prototype Deployment

```
┌──────────────────────────────────────┐
│       INFERENCE ENVIRONMENT          │
│                                      │
│  • Docker container with PyTorch     │
│  • FastAPI REST endpoint             │
│  • ONNX Runtime (optional speedup)   │
│  • Input: LISS-IV GeoTIFF            │
│  • Output: Cloud-free GeoTIFF        │
│                                      │
└──────────────────────────────────────┘
```

### 8.3 Scaling (Future)

- **Horizontal:** Multiple GPU workers behind a load balancer (Redis/Celery queue)
- **Orchestration:** Kubernetes (K8s) for auto-scaling based on queue depth
- **Storage:** S3-compatible object storage for input/output imagery

---

## 9. External Integration Points

| External System | Data | Protocol | Authentication |
|----------------|------|----------|----------------|
| **Bhoonidhi (ISRO)** | LISS-IV Level-2 imagery | HTTPS (manual order + download) | Institutional login (Indian credentials) |
| **Copernicus Data Space** | Sentinel-1 GRD, Sentinel-2 L2A | OData REST API / S3 | OAuth2 (free registration) |
| **USGS EarthExplorer** | SRTM DEM | HTTPS / FTP | Free account |
| **OpenTopography / Copernicus** | Copernicus DEM (COP-30) | HTTPS API | Free / API key |
| **Google Earth Engine** | Any EO dataset (optional) | Python API | Google account + GEE project |

---

## 10. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Deep Learning Framework** | PyTorch (not TensorFlow) | Better ecosystem for Diffusion models, Transformers, research flexibility |
| **Primary Model** | Conditional Latent Diffusion | Best quality-stability trade-off; latent space reduces memory 4–8× |
| **Input Resolution** | 512 × 512 tiles | Balances GPU memory, context window size, and reconstruction quality |
| **Tile Overlap** | 64 pixels | Prevents seam artifacts at tile boundaries |
| **CRS Strategy** | Process in UTM zone of input | Preserves metric distances for 5.8 m GSD |
| **Data Format** | GeoTIFF (Float32 internally, UInt16 export) | Universal geospatial compatibility |
| **Configuration** | YAML config files | Human-readable, version-controllable, easy to modify |

---

## 11. Risk Assessment (HLD Level)

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient LISS-IV training data (Bhoonidhi access) | High | High | Augment with Sentinel-2 pre-training + synthetic cloud simulation |
| Mis-registration between LISS-IV and SAR/S2 | Medium | High | Sub-pixel co-registration (AROSICS/phase correlation), DEM-assisted orthorectification |
| Diffusion model too slow for operational use | Medium | Medium | Use DDIM (10-step sampling), ONNX export, model distillation |
| Spectral inconsistency in reconstructed regions | Medium | High | SAM loss in training, histogram matching in post-processing, per-band normalization |
| Large cloud cover (>80% of tile) — insufficient context | Low | High | Rely heavily on SAR + temporal; flag tiles with QA=uncertain |

---

> **Next →** Read [03-low-level-design-LLD.md](./03-low-level-design-LLD.md) for module-level design, class structures, and API specifications.
