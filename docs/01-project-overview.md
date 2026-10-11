# 01 — Project Overview
## Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery

---

## 1. Executive Summary

Optical remote sensing imagery from the **LISS-IV** (Linear Imaging Self-Scanning Sensor – IV) instrument aboard India's **Resourcesat-2/2A** satellites is among the highest-resolution multispectral data available from an Indian Earth Observation (EO) programme. With a **5.8 m Ground Sample Distance (GSD)** across three spectral bands (Green, Red, Near-Infrared), LISS-IV imagery serves critical national applications — from land-use/land-cover (LULC) mapping and crop monitoring to disaster response and urban planning.

However, **persistent cloud cover** — particularly over the **North Eastern Region (NER)** of India, the Western Ghats, and other tropical/mountainous terrains — renders a significant fraction of acquired scenes unusable. During the monsoon season (June–September), cloud contamination can exceed **70–80%** of the scene area, reducing the number of usable observations to as few as 3–4 per year for a given location.

This project develops a **Generative AI-based framework** that:
- Automatically **detects** cloud and cloud-shadow regions in LISS-IV imagery
- **Reconstructs** the obscured surface using learned spatial, spectral, and temporal priors
- Optionally **fuses** auxiliary data (Sentinel-1 SAR, Sentinel-2 optical, DEM, temporal references) to improve reconstruction fidelity
- Produces **analysis-ready, cloud-free imagery** suitable for downstream geospatial applications

---

## 2. Problem Statement — In Depth

### 2.1 The Cloud Problem in Optical Remote Sensing

Clouds are opaque or semi-transparent atmospheric features that block the sensor's view of the Earth's surface. In optical imagery:

| Issue | Impact |
|-------|--------|
| **Opaque (thick) clouds** | Complete signal loss — underlying surface reflectance is replaced by bright white pixels |
| **Thin/cirrus clouds** | Partial signal attenuation — surface features are visible but spectrally distorted |
| **Cloud shadows** | Dark regions adjacent to clouds where direct sunlight is blocked — causes under-exposure and spectral shift |
| **Haze** | Diffuse atmospheric scattering that reduces contrast and shifts spectral values |

### 2.2 Why Traditional Approaches Fall Short

| Approach | Limitation |
|----------|-----------|
| **Cloud masking (FMask, Sen2Cor, QA bands)** | Only *identifies* clouds — does not recover underlying surface. Creates data gaps (holes) in the imagery |
| **Temporal compositing / mosaicking** | Requires multiple cloud-free observations of the same area. In NER, this can take months, and phenological changes make composites inconsistent |
| **Simple spatial interpolation (bilinear, nearest-neighbor)** | Ignores spatial context — produces blurry, unrealistic results for large cloud patches |
| **Classical inpainting (Navier-Stokes, Telea)** | Works only for small gaps. Cannot reconstruct complex textures, edges, or spectral correlations over large areas |

### 2.3 Why Generative AI?

Generative AI models (GANs, Diffusion Models, Transformers) can learn the **statistical distribution** of cloud-free surface imagery and generate plausible reconstructions conditioned on:
- The **surrounding cloud-free context** within the same scene
- **SAR (Synthetic Aperture Radar)** imagery that "sees through" clouds via microwave sensing
- **Temporal references** — previous or subsequent cloud-free observations of the same area
- **Spectral and textural priors** learned from thousands of training examples

This makes Generative AI the most promising approach for producing high-fidelity, spectrally consistent cloud-free imagery.

### 2.4 The LISS-IV Specific Challenge

LISS-IV presents unique characteristics that make cloud removal both important and challenging:

- **High spatial resolution (5.8 m):** Fine-scale features (individual buildings, field boundaries, narrow rivers) must be preserved — blurry reconstructions are unacceptable
- **3 spectral bands only (G, R, NIR):** Less spectral redundancy than sensors like Sentinel-2 (13 bands), making spectral consistency harder to maintain
- **Limited free data availability:** LISS-IV data is distributed via ISRO's **Bhoonidhi** portal and is not as freely accessible as Copernicus data, limiting training set size
- **NER terrain complexity:** Dense forests, steep mountains, river valleys, and diverse LULC classes make reconstruction inherently difficult

---

## 3. Objectives

| # | Objective | Success Criteria |
|---|-----------|-----------------|
| **O1** | Develop automated cloud and shadow detection for LISS-IV imagery | Detection accuracy (IoU) > 0.90 against manually annotated masks |
| **O2** | Build a Generative AI model for cloud removal and surface reconstruction | PSNR > 30 dB, SSIM > 0.90, SAM < 3.0° on held-out test tiles |
| **O3** | Preserve spatial structures (edges, textures, fine features) in reconstructed regions | LPIPS < 0.15, visual quality confirmed via human evaluation |
| **O4** | Maintain spectral consistency across reconstructed and non-cloud regions | Spectral correlation coefficient > 0.93 per band; NDVI error < 0.03 MAE |
| **O5** | Integrate multi-modal auxiliary data (SAR, temporal, DEM) for improved reconstruction | Measurable improvement (>1 dB PSNR) when auxiliary data is available |
| **O6** | Generate analysis-ready cloud-free products (GeoTIFF with proper CRS, metadata) | Output readable by QGIS/ArcGIS with intact georeferencing |
| **O7** | Evaluate and compare multiple Generative AI architectures | Comparative report covering GAN, Diffusion, Transformer approaches |
| **O8** | Develop a scalable, reproducible workflow | Documented pipeline runnable end-to-end with a single command |

---

## 4. Scope

### 4.1 In Scope

- LISS-IV Level-2 (radiometrically corrected) satellite imagery processing
- Cloud and cloud-shadow detection (automated mask generation)
- Generative AI model development (training, validation, testing)
- Multi-modal data fusion: Sentinel-1 SAR (C-band, VV/VH), Sentinel-2 L2A (10 m optical), SRTM/Copernicus DEM (30 m), temporal LISS-IV references
- Quantitative evaluation: PSNR, SSIM, SAM, LPIPS, FID, CC
- Qualitative evaluation: visual comparison, error maps, NDVI consistency
- Architecture comparison: GAN vs. Diffusion Model vs. Transformer-based approaches
- Prototype deployment: inference script, batch processing, API endpoint
- Complete documentation and reproducibility

### 4.2 Out of Scope (Phase 1)

- Real-time onboard satellite processing
- Hyperspectral or thermal band reconstruction
- Operational deployment at ISRO/NRSC production scale
- Processing of non-LISS-IV Indian EO data (LISS-III, Cartosat, HySIS)
- Radar-only product generation (SAR-to-optical translation without any LISS-IV input)

---

## 5. Expected Outcomes

1. **Automated Cloud-Free Reconstruction Pipeline**
   - End-to-end workflow: input cloudy LISS-IV scene → output cloud-free GeoTIFF
   - Processing time target: < 5 minutes per 512×512 tile on a single GPU (T4/V100)

2. **Improved Temporal Availability of Usable Imagery**
   - Scenes previously discarded due to cloud cover become usable
   - Effective increase in usable observation frequency from ~4/year to ~12–15/year over NER

3. **Spectrally and Spatially Consistent Outputs**
   - Reconstructed regions are indistinguishable (or near-indistinguishable) from real cloud-free imagery in standard geospatial analysis workflows
   - Vegetation indices (NDVI, NDWI) computed from reconstructed imagery closely match ground truth

4. **Comparative Architecture Study**
   - Documented comparison of at least 3 Generative AI approaches (GAN, Diffusion, Transformer)
   - Analysis of trade-offs: quality vs. speed vs. training cost vs. data requirements

5. **Prototype Framework**
   - Python-based, modular codebase with clear separation of concerns
   - Configurable for different sensors, regions, and model architectures
   - Dockerized for reproducibility

6. **Analysis-Ready Data Products**
   - Cloud-free GeoTIFFs with intact coordinate reference system (CRS) and spatial metadata
   - Quality assessment (QA) masks indicating reconstructed vs. original pixels

---

## 6. Key Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| **ISRO / NRSC** | Data provider (Bhoonidhi), potential operational user | Improved LISS-IV product utilization |
| **State Remote Sensing Centres (NER-DIC, NER-SAC)** | Regional data users | Cloud-free imagery for NER development planning |
| **Researchers (RS & GIS)** | Academic users | LULC, agriculture, forestry, hydrology analysis |
| **Disaster Management Agencies (NDMA, SDMA)** | Operational users | Timely cloud-free imagery during flood/landslide events |
| **Geospatial Startups / Industry** | Commercial users | Analytics-ready satellite data products |

---

## 7. Key Terminology (Glossary)

| Term | Full Form / Meaning |
|------|-------------------|
| **LISS-IV** | Linear Imaging Self-Scanning Sensor – IV (5.8 m, 3 bands: G/R/NIR) |
| **NER** | North Eastern Region of India |
| **SAR** | Synthetic Aperture Radar — active microwave sensor, cloud-penetrating |
| **GAN** | Generative Adversarial Network |
| **Diffusion Model** | Denoising Diffusion Probabilistic Model (DDPM) / Score-based generative model |
| **PSNR** | Peak Signal-to-Noise Ratio (dB) — pixel-level reconstruction quality |
| **SSIM** | Structural Similarity Index — perceptual structural quality |
| **SAM** | Spectral Angle Mapper — spectral distortion metric (degrees, lower = better) |
| **LPIPS** | Learned Perceptual Image Patch Similarity — deep feature-based perceptual quality |
| **FID** | Fréchet Inception Distance — distributional quality of generated images |
| **CC** | Correlation Coefficient — per-band linear correlation between GT and prediction |
| **ARD** | Analysis Ready Data — preprocessed, calibrated, georeferenced imagery |
| **COG** | Cloud Optimized GeoTIFF — GeoTIFF format optimized for cloud/web access |
| **STAC** | SpatioTemporal Asset Catalog — metadata standard for EO data |
| **NDVI** | Normalized Difference Vegetation Index = (NIR − Red) / (NIR + Red) |
| **NDWI** | Normalized Difference Water Index = (Green − NIR) / (Green + NIR) |
| **DEM** | Digital Elevation Model |
| **Bhoonidhi** | ISRO's geospatial data portal for Indian satellite imagery |

---

## 8. Assumptions and Constraints

### 8.1 Assumptions

1. LISS-IV Level-2 imagery (radiometrically corrected) is available via Bhoonidhi for the study area
2. Cloud-free reference imagery exists (within ±30 days) for at least 60% of training tiles — needed to create ground truth pairs
3. Sentinel-1 and Sentinel-2 imagery with overlapping spatial and temporal coverage is freely available via Copernicus Data Space
4. Co-registration between LISS-IV and auxiliary datasets can achieve sub-pixel accuracy (< 1 pixel RMSE)
5. A GPU with ≥ 16 GB VRAM (e.g., NVIDIA T4, V100, A100, or RTX 3090/4090) is available for training

### 8.2 Constraints

1. **Data Access:** Bhoonidhi requires Indian institutional credentials and manual ordering — no bulk API exists for LISS-IV. This limits the scale of training data collection
2. **Resolution Mismatch:** Sentinel-2 (10 m) and SRTM DEM (30 m) are coarser than LISS-IV (5.8 m) — resampling introduces interpolation artifacts
3. **Spectral Bands:** LISS-IV has only 3 bands (G, R, NIR) vs. Sentinel-2's 13 — less spectral information for reconstruction
4. **Compute:** Diffusion model training at 512×512 resolution requires significant GPU memory (24–40 GB) and training time (days to weeks)
5. **Ground Truth Limitation:** True "ground truth" under clouds does not exist — we rely on synthetic cloud augmentation over cloud-free imagery for supervised training

---

## 9. Document Index

This documentation suite consists of the following files:

| # | Document | Purpose |
|---|----------|---------|
| 01 | **Project Overview** *(this file)* | Problem statement, objectives, scope, outcomes |
| 02 | [High-Level Design (HLD)](./02-high-level-design-HLD.md) | System-level architecture, subsystems, data flow |
| 03 | [Low-Level Design (LLD)](./03-low-level-design-LLD.md) | Module design, class structures, API contracts, database schema |
| 04 | [Project Architecture](./04-project-architecture.md) | End-to-end architecture diagrams, infrastructure, deployment |
| 05 | [Data Sources & Downloading Guide](./05-data-sources-and-downloading.md) | Every dataset, how to access it, exact download commands |
| 06 | [Model Training Guide & Algorithms](./06-model-training-and-algorithms.md) | Algorithms used, why each was chosen, training procedure, hyperparameters |
| 07 | [Complete Workflow & Pipeline](./07-complete-workflow-and-pipeline.md) | Step-by-step operational pipeline from raw data to cloud-free output |
| 08 | [Detailed Implementation Plan](./08-detailed-implementation-plan.md) | Phased roadmap, milestones, timeline, deliverables, risk mitigation |
| 09 | [Kaggle & Google Drive Cloud Training Guide](./09-kaggle-gdrive-cloud-training-guide.md) | Zero-bandwidth direct cloud ingestion, GDrive sync & 2× T4 GPU training |

---

> **Next →** Read [02-high-level-design-HLD.md](./02-high-level-design-HLD.md) for the system-level architecture.
