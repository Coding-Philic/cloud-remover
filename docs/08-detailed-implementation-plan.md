# 08 — Detailed Implementation Plan and Engineering Roadmap
## Generative AI-Based Cloud Removal for LISS-IV Satellite Imagery

---

## 1. Executive Plan Overview

This document provides the master 24-week (6-month) engineering roadmap, resource allocation model, risk management framework, and milestone verification criteria for delivering the complete Generative AI cloud removal system for ISRO LISS-IV imagery.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               24-WEEK IMPLEMENTATION ROADMAP                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

 Month 1 (W1-W4)         Month 2 (W5-W8)         Month 3 (W9-W12)
 ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
 │ Phase 1: Setup  │ ──► │ Phase 2: Data   │ ──► │ Phase 3: Base   │
 │ & Data Harvest  │     │ & Mask Engine   │     │ Generative Model│
 └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
 Month 6 (W21-W24)       Month 5 (W17-W20)       Month 4 (W13-W16)
 ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
 │ Phase 6: Deploy │ ◄── │ Phase 5: Mosaic │ ◄── │ Phase 4: Diff-  │
 │ & NER Pilot     │     │ & ARD Engine    │     │ usion Fusion    │
 └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Engineering Targets

| Metric | Target Specification |
|--------|----------------------|
| **Spatial Resolution Preserved** | Exactly 5.8 meters GSD (no spatial blurring) |
| **Spectral Fidelity (SAM)** | $< 3.0^\circ$ spectral angle deviation |
| **Structural Similarity (SSIM)** | $> 0.92$ on reconstructed cloud masks |
| **Peak Signal-to-Noise Ratio (PSNR)** | $> 32.5 \text{ dB}$ on synthetic test benchmarks |
| **End-to-End Scene Latency** | $< 15 \text{ minutes}$ for full $70\text{km} \times 70\text{km}$ scene on $1\times \text{A100}$ |
| **Deployment Footprint** | Dockerized FastAPI Microservice + QGIS 3.x Analysis Plugin |

---

## 2. Phase Breakdown and Timeline

### Phase 1: Environment Setup, Tooling & Data Harvesting (Weeks 1–3)
- **Objective:** Establish reproducible software environments, cloud storage buckets, Bhoonidhi harvesting pipelines, and initial 50-scene training corpus over the North Eastern Region (NER) of India.
- **Key Deliverables:**
  - Standardized Conda/Docker environment with CUDA 11.8 / PyTorch 2.1 / GDAL 3.8.
  - GeoJSON AOI definitions covering Assam, Meghalaya, Arunachal Pradesh, and Tripura.
  - 50 LISS-IV cloudy/clear scene pairs ingested and validated.
  - Ingestion pipeline scripts for Sentinel-1 GRD, Sentinel-2 L2A, and COP-30 DEM.

### Phase 2: Preprocessing, Co-Registration & Cloud-Masking Engine (Weeks 4–6)
- **Objective:** Build the automated radiometric calibration, sub-pixel co-registration (AROSICS), and multi-class cloud/shadow segmentation pipeline.
- **Key Deliverables:**
  - Automated DN-to-TOA reflectance calibration module.
  - DeepLabv3+ cloud and shadow detector achieving $\text{IoU} \ge 0.92$.
  - Sub-pixel AROSICS registration achieving $< 0.5$ pixel RMSE across multi-modal pairs.
  - Overlapping tile generator ($512\times512$ with $64\text{px}$ stride) yielding $25,000+$ training patches.

### Phase 3: Baseline Generative Models & Evaluation Suite (Weeks 7–10)
- **Objective:** Implement and benchmark classical and initial generative AI baselines (Spatial Interpolation, Pix2PixHD, and Spatial Attention GAN - SpaGAN).
- **Key Deliverables:**
  - Baseline benchmark testbed implementing PSNR, SSIM, SAM, CC, LPIPS, and NDVI MAE.
  - Pix2PixHD baseline model trained for 100 epochs.
  - SpaGAN attention-guided baseline trained for 100 epochs.
  - Automated benchmark reports confirming baseline performance floor ($\sim 28\text{--}30 \text{ dB}$ PSNR).

### Phase 4: Production Diffusion Model Architecture & Cross-Modal Fusion (Weeks 11–15)
- **Objective:** Develop, optimize, and train the primary Conditional Latent Diffusion Model (LDM) with multi-modal cross-attention fusion (LISS-IV + Sentinel-1 SAR + DEM + Temporal Reference).
- **Key Deliverables:**
  - Multi-sensor feature encoder embedding SAR polarizations (VV/VH) and terrain slope into latent tokens.
  - Latent Diffusion U-Net with DDIM accelerated sampling (15–20 steps).
  - Multi-GPU distributed training run (DDP on 2× A100 GPUs) for 200 epochs.
  - Checkpoint selection achieving $> 32.5 \text{ dB}$ PSNR and $< 2.8^\circ$ SAM.

### Phase 5: Scalable Tiled Inference, Seamless Mosaicing & ARD Packaging (Weeks 16–18)
- **Objective:** Build memory-efficient sliding-window inference, apodization window feathering, boundary histogram matching, and Cloud-Optimized GeoTIFF (COG) generation.
- **Key Deliverables:**
  - Vectorized tile orchestrator with PyTorch AMP FP16 inference engine.
  - 2D Gaussian feathering mosaic module eliminating edge and grid artifacts.
  - Boundary histogram matching engine preserving radiometric consistency with surrounding unclouded ground.
  - Automated COG exporter with internal overviews and STAC Item metadata generator.

### Phase 6: Operational API, QGIS Plugin, Pilot Demonstration & Documentation (Weeks 19–24)
- **Objective:** Package the trained models and pipeline into an operational microservice, build an intuitive QGIS user plugin, run a 10-scene pilot over the NER, and produce final technical reports.
- **Key Deliverables:**
  - Production-ready Docker container with FastAPI endpoints (`/reconstruct`, `/health`, `/status`).
  - Native QGIS 3.x Python Plugin allowing point-and-click cloud removal directly on the GIS canvas.
  - Operational pilot validation report across 10 real-world monsoonal LISS-IV scenes.
  - Comprehensive technical documentation, open-source code repository, and research paper draft.

---

## 3. Sprint-by-Sprint Plan (12 Two-Week Sprints)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              12 AGILE SPRINTS (2 WEEKS EACH)                           │
├────────┬─────────────────┬──────────────────────────────────────┬──────────────────────┤
│ Sprint │ Focus Area      │ Core Engineering Deliverables        │ Primary Lead         │
├────────┼─────────────────┼──────────────────────────────────────┼──────────────────────┤
│ S01    │ Env & Data Ingest│ Conda/Docker, Bhoonidhi scraper, AOI │ Geospatial Engineer  │
│ S02    │ Auxiliary Stacks│ Sentinel-1 & DEM downloaders, S3     │ MLOps / Data Eng     │
│ S03    │ Calibration     │ DN-to-TOA converter, UTM reproject   │ Geospatial Engineer  │
│ S04    │ Masking Engine  │ DeepLabv3+ cloud/shadow detector     │ ML Engineer          │
│ S05    │ Co-registration │ AROSICS Fourier subpixel registration│ Geospatial Engineer  │
│ S06    │ Tiling & DVC    │ 512×512 overlapping dataset, DVC     │ MLOps / Data Eng     │
│ S07    │ GAN Baselines   │ Pix2PixHD & SpaGAN training & eval   │ ML Engineer          │
│ S08    │ Transformer Base│ Restormer/SwinIR cross-comparison   │ ML Engineer          │
│ S09    │ Diffusion Core  │ Latent Diffusion Model & VAE training│ ML Lead              │
│ S10    │ Multi-Modal Fuse│ SAR + DEM Cross-Attention integration│ ML Lead              │
│ S11    │ Inference Engine│ Sliding window, feathering & COG     │ MLOps / Software Eng │
│ S12    │ API & Pilot     │ FastAPI, QGIS Plugin, NER validation │ Full Team            │
└────────┴─────────────────┴──────────────────────────────────────┴──────────────────────┘
```

### Detailed Sprint Specifications

#### Sprint 01: Environment, Tooling & AOI Harvesting
- **User Story:** As an ML Engineer, I need a deterministic development environment and initial LISS-IV sample scenes so that pipeline components can be developed without dependency conflicts.
- **Tasks:**
  1. Construct `environment.yml` and `Dockerfile.dev` with GDAL 3.8, PyTorch 2.1, and CUDA 11.8.
  2. Define GeoJSON polygons for target NER areas: Guwahati urban, Brahmaputra basin, Shillong plateau, and agricultural valleys.
  3. Manually order and harvest 10 test LISS-IV scenes from Bhoonidhi portal.
  4. Build `validate_scene.py` to assert raster integrity, dimensions, and band count.
- **Definition of Done (DoD):** Docker container builds cleanly; 10 LISS-IV scenes downloaded and pass validation tests.

#### Sprint 02: Auxiliary Data Harvesting Pipelines
- **User Story:** As a Data Engineer, I need automated downloaders for Sentinel-1 GRD, Sentinel-2 L2A, and COP-30 DEM so that auxiliary inputs can be fetched on demand.
- **Tasks:**
  1. Implement Copernicus Data Space Ecosystem (CDSE) OData downloader for Sentinel-1 GRD IW.
  2. Implement COP-30 DEM tile downloader via AWS Open Data / OpenTopography.
  3. Build temporal scene search to identify nearest clear-sky LISS-IV observation ($\le 10\%$ cloud) within $\pm 30$ days.
  4. Write integration unit test verifying automated retrieval for an arbitrary AOI bounding box.
- **Definition of Done (DoD):** Automated scripts successfully download matching Sentinel-1 and DEM rasters for all 10 test scenes.

#### Sprint 03: Radiometric Calibration and Orthorectification
- **User Story:** As a Remote Sensing Specialist, I need DNs converted to TOA Reflectance and rasters projected to a common UTM grid so that pixel values represent physical reflectance.
- **Tasks:**
  1. Implement band-specific radiometric calibration for LISS-IV Green (B2), Red (B3), and NIR (B4).
  2. Compute Earth-Sun astronomical distance ($d$) and solar zenith angle corrections from metadata.
  3. Validate reprojected rasters in QGIS to ensure zero spatial drift against Sentinel-2 basemaps.
- **Definition of Done (DoD):** TOA reflectance arrays fall strictly in $[0.0, 1.0]$; reprojection RMSE $< 0.2$ pixels against reference orthophoto.

#### Sprint 04: Cloud & Shadow Masking Engine
- **User Story:** As an ML Engineer, I need an automated cloud and shadow segmentation model so that corrupted pixels can be masked during reconstruction.
- **Tasks:**
  1. Assemble ground-truth mask set (200 diverse LISS-IV / Sentinel-2 tiles manually annotated or cross-calibrated).
  2. Train DeepLabv3+ with MobileNetV3 backbone for 3-class segmentation (Clear, Cloud, Shadow).
  3. Implement deterministic spectral fallback rules (HOT haze index + NIR shadow dip).
  4. Implement morphological dilation post-processing (15-pixel buffer kernel).
- **Definition of Done (DoD):** Cloud/shadow detector achieves $\text{IoU} \ge 0.92$ on independent holdout validation set.

#### Sprint 05: Sub-Pixel Co-Registration Engine
- **User Story:** As a Geospatial Engineer, I need multi-sensor inputs aligned with sub-pixel precision so that spatial details from SAR and temporal optical references match perfectly.
- **Tasks:**
  1. Integrate `arosics` cross-correlation library for automated tie-point detection.
  2. Implement cubic-spline warping of Sentinel-1 backscatter to 5.8m LISS-IV grid.
  3. Handle cloudy regions by masking out cloud areas during tie-point detection.
  4. Create QA diagnostic plot displaying displacement vector fields and residual RMSE.
- **Definition of Done (DoD):** Co-registration RMSE consistently $< 0.5$ pixels across clear ground control points.

#### Sprint 06: Overlapping Tiling & Dataset Versioning
- **User Story:** As an MLOps Engineer, I need large scenes sliced into standardized multi-modal tensor cubes tracked by DVC so that model training is fully reproducible.
- **Tasks:**
  1. Implement $512\times512$ sliding window tiler with $64\text{px}$ overlap.
  2. Stack 8 channels: $[ \text{LISS-IV (3)}, \text{SAR (2)}, \text{Cloud Mask (1)}, \text{DEM/Slope (2)} ]$.
  3. Filter out invalid/nodata border tiles ($> 50\%$ nodata).
  4. Version dataset ($25,000$ tiles) using Data Version Control (DVC) synced to MinIO/S3.
- **Definition of Done (DoD):** Dataset generated, split 70% train / 15% val / 15% test, and tracked in DVC remote.

#### Sprint 07: GAN Baseline Implementations (Pix2PixHD & SpaGAN)
- **User Story:** As an ML Engineer, I need baseline GAN models trained so that we have an empirical performance benchmark for the diffusion architecture.
- **Tasks:**
  1. Implement Pix2PixHD conditional GAN with multi-scale discriminators.
  2. Implement Spatial Attention GAN (SpaGAN) attending specifically to cloud mask boundaries.
  3. Train both models for 100 epochs on training tiles.
  4. Log training curves, perceptual losses, and sample predictions to Weights & Biases (W&B).
- **Definition of Done (DoD):** Baseline models fully trained; baseline PSNR reaches $29.2 \text{ dB}$ (Pix2PixHD) and $30.4 \text{ dB}$ (SpaGAN).

#### Sprint 08: Transformer Baseline (Restormer / SwinIR)
- **User Story:** As an ML Engineer, I need a transformer-based image restoration baseline to evaluate large cloud coverage handling.
- **Tasks:**
  1. Implement Restormer architecture adapted for 8-channel multi-modal input.
  2. Train model for 100 epochs using AdamW optimizer and Cosine Annealing learning rate.
  3. Evaluate reconstruction quality on small ($< 20\%$), medium ($20\text{--}50\%$), and large ($> 50\%$) cloud occlusions.
- **Definition of Done (DoD):** Benchmark report comparing GAN vs Transformer across all metrics generated and archived.

#### Sprint 09: Latent Diffusion Model Core Engine
- **User Story:** As an ML Lead, I need a conditional latent diffusion model (LDM) trained so that high-fidelity textures can be synthesized without adversarial instability.
- **Tasks:**
  1. Train or adapt spatial autoencoder (VAE) to compress $512\times512\times3$ LISS-IV tiles into $64\times64\times4$ latents.
  2. Construct diffusion U-Net with cross-attention condition blocks.
  3. Implement DDPM training objective ($\epsilon$-prediction) with Cosine noise schedule.
  4. Train core diffusion backbone for 100 epochs across 2× A100 GPUs using PyTorch DDP.
- **Definition of Done (DoD):** Diffusion model trains stably without gradient explosion; reconstruction loss converges monotonically.

#### Sprint 10: Multi-Modal Cross-Attention Fusion & DDIM Acceleration
- **User Story:** As an ML Lead, I need auxiliary SAR and DEM features fused into the diffusion process and DDIM fast sampling configured so that reconstruction is physically accurate and rapid.
- **Tasks:**
  1. Implement cross-attention injection: project SAR backscatter and terrain tokens into keys/values of U-Net cross-attention layers.
  2. Implement DDIM reverse sampling with customizable step count (10, 15, 20, 50 steps).
  3. Run hyperparameter ablation comparing sampling steps vs PSNR/SSIM.
  4. Perform full 200-epoch production training run with Exponential Moving Average (EMA) weight consolidation.
- **Definition of Done (DoD):** Final model checkpoint achieves $\text{PSNR} \ge 32.5 \text{ dB}$, $\text{SSIM} \ge 0.92$, and $\text{SAM} \le 2.8^\circ$ at 15 DDIM steps.

#### Sprint 11: Seamless Mosaicing & ARD Packaging
- **User Story:** As a Software Engineer, I need reconstructed tiles reassembled into the complete $70\text{km}$ scene and exported as COGs with STAC metadata so that outputs are analysis-ready.
- **Tasks:**
  1. Implement 2D Gaussian apodization feathering accumulator for overlapping tile regions.
  2. Implement boundary histogram matching to ensure native and reconstructed pixels match spectrally.
  3. Convert output multi-band rasters to Cloud-Optimized GeoTIFFs (COG) with internal pyramid overviews.
  4. Generate STAC Item GeoJSON files with spatio-temporal metadata and quality flags.
- **Definition of Done (DoD):** Full-scene reconstruction script executes end-to-end without seam lines or memory overflow in $< 15$ minutes on $1\times$ A100.

#### Sprint 12: Deployment, QGIS Plugin, NER Pilot & Final Report
- **User Story:** As a Solutions Architect, I need the complete framework accessible via FastAPI and QGIS, validated on 10 unseen NER pilot scenes, with final documentation delivered.
- **Tasks:**
  1. Package the inference pipeline into a production Docker image with a FastAPI service (`POST /reconstruct`).
  2. Develop a native QGIS 3.x plugin GUI allowing users to select a cloudy LISS-IV layer and generate cloud-free outputs.
  3. Run an operational pilot on 10 challenging real-world scenes across Assam, Meghalaya, and Arunachal Pradesh.
  4. Compile final technical reports, user manuals, and research paper manuscript.
- **Definition of Done (DoD):** All 10 pilot scenes successfully processed; QGIS plugin functions seamlessly; documentation suite 100% complete.

---

## 4. Team Structure and Resource Allocation

### 4.1 Engineering Team Roles

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TEAM RESPONSIBILITY MATRIX                                │
├─────────────────────┬──────────────┬───────┬───────────────────────────────────────────┤
│ Role                │ Headcount    │ Alloc │ Key Responsibilities                      │
├─────────────────────┼──────────────┼───────┼───────────────────────────────────────────┤
│ **ML Lead**         │ 1 FTE        │ 100%  │ Diffusion architecture, loss formulation, │
│                     │              │       │ multi-modal cross-attention, training loop│
├─────────────────────┼──────────────┼───────┼───────────────────────────────────────────┤
│ **Geospatial Eng.** │ 1 FTE        │ 100%  │ Ingestion, calibration, co-registration,  │
│                     │              │       │ QGIS plugin development, projection QA    │
├─────────────────────┼──────────────┼───────┼───────────────────────────────────────────┤
│ **MLOps / Data Eng**│ 1 FTE        │ 75%   │ Prefect pipeline, S3/MinIO, Docker, API,  │
│                     │              │       │ CI/CD, distributed cluster management     │
├─────────────────────┼──────────────┼───────┼───────────────────────────────────────────┤
│ **Domain Specialist/│ 0.5 FTE      │ 50%   │ Agricultural/Forestry validation, LULC    │
│ QA Evaluator**      │              │       │ downstream accuracy testing, visual QA    │
└─────────────────────┴──────────────┴───────┴───────────────────────────────────────────┘
```

---

## 5. Hardware, Infrastructure, and Budget Estimate

### 5.1 Cloud vs On-Premises Cost Comparison (6-Month Project Scope)

| Infrastructure Item | AWS Cloud (US/India Region) | Lambda Labs GPU Cloud | On-Premises Cluster (ISRO/NRSC) |
|---------------------|-----------------------------|-----------------------|---------------------------------|
| **Model Training** | 2× A100 (80GB) Spot: ~300 hrs @ $3.50/hr = **$1,050** | 2× A100 (80GB) On-Demand: 300 hrs @ $2.20/hr = **$660** | Internal compute quota (Zero marginal cloud cost) |
| **Development & Ablation** | 1× A10G / RTX 4090: 500 hrs @ $1.00/hr = **$500** | 1× A10: 500 hrs @ $0.60/hr = **$300** | Workstation with 1× RTX 4090 |
| **Object Storage** | S3 Standard: 5 TB storage + egress = **$680** | S3 / Wasabi: 5 TB storage = **$180** | On-prem NAS / SAN array |
| **Inference Staging** | 1× T4 (g4dn.xlarge) spot: 200 hrs = **$120** | 1× A10 instance: 200 hrs = **$120** | Local GPU server |
| **Total Cloud Budget**| **$2,350 USD** | **$1,260 USD** | **Hardware Depreciation Only** |

---

## 6. Risk Management and Mitigation Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RISK ASSESSMENT & MITIGATION MATRIX                             │
├─────────────────────────┬────────┬────────┬────────────────────────────────────────────┤
│ Risk Description        │ Impact │ Likeli-│ Concrete Engineering Mitigation            │
│                         │        │ hood   │                                            │
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **Bhoonidhi Throttling**│ High   │ Medium │ Maintain local S3 cache of 50 base scenes; │
│ Manual order bottlenecks│        │        │ utilize synthetic cloud generator on clear │
│ stall training data     │        │        │ scenes to multiply dataset by 10×.         │
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **Terrain Shadows in NER**│ Medium│ High   │ Incorporate COP-30 DEM hillshade into loss │
│ Deep Himalayan gorges   │        │        │ weighting; differentiate real shadow from  │
│ misclassified as clouds │        │        │ cloud shadow using solar azimuth geometry. │
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **Generative Hallucina-**│ High  │ Medium │ Strictly enforce SAR backscatter guidance; │
│ **tion / Invented Towns**│       │        │ add spectral penalty loss and enforce      │
│ Diffusion hallucinates  │        │        │ zero-hallucination threshold via SAM test. │
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **Sub-Pixel Mismatch**  │ High   │ Medium │ Use robust multi-scale FFT matching        │
│ Mountainous terrain     │        │        │ (AROSICS) with high-coherence tie points;  │
│ distorts SAR-optical fit│        │        │ reject tiles with co-registration RMSE > 1px│
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **GPU VRAM Saturation** │ Medium │ High   │ Use Latent Diffusion (8× compression) with │
│ Large 512×512 batches   │        │        │ PyTorch 2.0 FlashAttention & FP16 AMP.     │
│ overflow 24GB VRAM      │        │        │ Batch size dynamic scaling.                │
├─────────────────────────┼────────┼────────┼────────────────────────────────────────────┤
│ **Seasonal Phenological**│ Medium│ High   │ Limit temporal reference window to ±30 days│
│ **Discontinuity**       │        │        │ of target season; apply histogram matching │
│ Winter leaf-off vs sum- │        │        │ across cloud boundaries.                   │
│ mer leaf-on mismatch    │        │        │                                            │
└─────────────────────────┴────────┴────────┴────────────────────────────────────────────┘
```

---

## 7. Quality Gates and Definition of Done (DoD)

To advance from one project phase to the next, the system must pass strict automated Quality Gates:

```
 [Phase 1: Ingest] ────► Gate 1: Checksums valid, CRS verified, Metadata parsed
        │
 [Phase 2: Masking] ───► Gate 2: Cloud detector IoU ≥ 0.92, AROSICS RMSE < 0.5px
        │
 [Phase 3: Base GAN] ──► Gate 3: Baseline benchmark established (PSNR ≥ 29 dB)
        │
 [Phase 4: Diffusion] ─► Gate 4: Production model PSNR ≥ 32.5 dB, SAM ≤ 2.8°
        │
 [Phase 5: Mosaic] ────► Gate 5: Zero boundary seam (CBDI < 0.05), Valid COG
        │
 [Phase 6: Pilot] ─────► Gate 6: 10/10 pilot scenes validated, QGIS Plugin verified
```

### Automated CI/CD Testing Pipeline

Every pull request triggers automated tests executed in GitHub Actions with GPU runners:
1. **Unit Tests (`pytest tests/unit`):** Tests radiometric conversions, tile window math, and tensor shapes.
2. **Geospatial Regression Tests (`pytest tests/geo`):** Verifies affine transform consistency, CRS preservation, and nodata integrity.
3. **Spectral Fidelity Smoke Tests (`pytest tests/spectral`):** Runs mini-inference on 5 synthetic cloudy tiles and asserts SAM $< 3.5^\circ$ and SSIM $> 0.90$.

---

## 8. Verification and Deliverables Checklist

| Category | Deliverable Item | Status | Verification Command / Artifact |
|----------|------------------|--------|---------------------------------|
| **Documentation** | `docs/01-project-overview.md` | Completed | Executive project summary |
| **Documentation** | `docs/02-high-level-design-HLD.md` | Completed | High-level system architecture |
| **Documentation** | `docs/03-low-level-design-LLD.md` | Completed | Class diagrams, interfaces & API schemas |
| **Documentation** | `docs/04-project-architecture.md` | Completed | Deep dive on component interactions |
| **Documentation** | `docs/05-data-sources-and-downloading.md`| Completed | Bhoonidhi & Copernicus harvest commands |
| **Documentation** | `docs/06-model-training-and-algorithms.md`| Completed | Diffusion, GAN, and transformer math |
| **Documentation** | `docs/07-complete-workflow-and-pipeline.md`| Completed | End-to-end operational pipeline code |
| **Documentation** | `docs/08-detailed-implementation-plan.md`| Completed | Master 24-week engineering roadmap |
| **Software** | Modular Ingestion & Preprocessing Suite | Ready for Phase 1 | `pipeline/stage1_ingest.py` |
| **Software** | DeepLabv3+ Cloud & Shadow Detector | Ready for Phase 2 | `pipeline/stage3_cloud_mask.py` |
| **Software** | Latent Diffusion Reconstruction Engine | Ready for Phase 4 | `pipeline/stage6_reconstruct.py` |
| **Software** | Seamless Mosaicing & ARD Packaging | Ready for Phase 5 | `pipeline/stage7_mosaic.py` |
| **Software** | FastAPI Microservice & QGIS Plugin | Ready for Phase 6 | `api/main.py`, `qgis_plugin/` |

This detailed implementation plan provides a comprehensive, production-grade roadmap to guide the development and deployment of the Generative AI Cloud Removal system for LISS-IV satellite imagery.
