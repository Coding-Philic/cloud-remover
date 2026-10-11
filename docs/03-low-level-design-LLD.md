# 03 — Low-Level Design (LLD)
## Generative AI-Based Cloud Removal for LISS-IV Imagery

---

## 1. Project Directory Structure

```
cloud-removal-liss4/
│
├── configs/                          # All configuration files
│   ├── config.yaml                   # Master config (paths, model selection)
│   ├── train_diffusion.yaml          # Diffusion model training config
│   ├── train_gan.yaml                # GAN model training config
│   ├── train_transformer.yaml        # Transformer model training config
│   └── inference.yaml                # Inference config
│
├── data/                             # Data directory (gitignored, DVC-tracked)
│   ├── raw/                          # Raw downloaded imagery
│   │   ├── liss4/                    # LISS-IV scenes from Bhoonidhi
│   │   ├── sentinel1/               # Sentinel-1 GRD products
│   │   ├── sentinel2/               # Sentinel-2 L2A products
│   │   ├── dem/                      # SRTM or Copernicus DEM tiles
│   │   └── temporal_refs/           # Cloud-free temporal reference images
│   │
│   ├── processed/                    # Preprocessed, co-registered data
│   │   ├── tiles/                    # 512x512 tiles (NPZ or TIFF)
│   │   ├── masks/                    # Cloud masks per tile
│   │   └── stacks/                   # Multi-modal stacked arrays
│   │
│   └── splits/                       # Train/val/test split definitions
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                         # Data loading and processing
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   ├── datamodule.py             # Lightning DataModule
│   │   ├── transforms.py             # Augmentations (Albumentations)
│   │   ├── cloud_simulator.py        # Synthetic cloud generation
│   │   └── tiler.py                  # Scene → tiles and tiles → scene
│   │
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   ├── preprocess_liss4.py       # LISS-IV calibration and normalization
│   │   ├── preprocess_sentinel1.py   # SAR preprocessing (speckle filter, dB convert)
│   │   ├── preprocess_sentinel2.py   # S2 band extraction and resampling
│   │   ├── coregister.py             # Multi-source co-registration
│   │   └── stack_builder.py          # Build multi-channel input stacks
│   │
│   ├── masking/                      # Cloud mask generation
│   │   ├── __init__.py
│   │   ├── cloud_detector.py         # DL-based cloud segmentation model
│   │   ├── rule_based.py             # Threshold-based cloud detection
│   │   └── shadow_estimator.py       # Cloud shadow detection
│   │
│   ├── models/                       # Generative model architectures
│   │   ├── __init__.py
│   │   ├── diffusion/                # Diffusion model family
│   │   │   ├── __init__.py
│   │   │   ├── unet.py               # Conditional U-Net denoiser
│   │   │   ├── scheduler.py          # Noise scheduler (DDPM/DDIM)
│   │   │   ├── vae.py                # Variational Autoencoder (latent space)
│   │   │   └── diffusion_model.py    # Full diffusion pipeline
│   │   │
│   │   ├── gan/                      # GAN model family
│   │   │   ├── __init__.py
│   │   │   ├── generator.py          # U-Net / ResNet generator
│   │   │   ├── discriminator.py      # PatchGAN discriminator
│   │   │   └── gan_model.py          # Full GAN training pipeline
│   │   │
│   │   ├── transformer/              # Transformer model family
│   │   │   ├── __init__.py
│   │   │   ├── swin_ir.py            # Swin Transformer backbone
│   │   │   └── transformer_model.py  # Full Transformer pipeline
│   │   │
│   │   ├── fusion/                   # Multi-modal fusion modules
│   │   │   ├── __init__.py
│   │   │   ├── fusion_encoder.py     # Encode SAR + DEM + temporal
│   │   │   └── attention_fusion.py   # Cross-attention fusion
│   │   │
│   │   └── losses.py                 # All loss functions
│   │
│   ├── inference/                    # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predictor.py              # Tiled inference engine
│   │   ├── blender.py                # Tile seam blending
│   │   └── exporter.py              # GeoTIFF export with metadata
│   │
│   ├── evaluation/                   # Metrics and evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                # PSNR, SSIM, SAM, LPIPS, FID, CC
│   │   ├── visual_compare.py         # Side-by-side comparison generator
│   │   └── benchmark.py              # Architecture comparison framework
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── geo_utils.py              # CRS, geotransform, reprojection helpers
│       ├── io_utils.py               # Read/write rasters (rasterio wrappers)
│       └── logger.py                 # Logging configuration
│
├── scripts/                          # Runnable scripts
│   ├── download_data.py              # Download all datasets
│   ├── preprocess_all.py             # Run full preprocessing pipeline
│   ├── train.py                      # Training entry point
│   ├── inference.py                  # Inference entry point
│   ├── evaluate.py                   # Evaluation entry point
│   └── export_onnx.py               # Export model to ONNX
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_cloud_mask_analysis.ipynb
│   ├── 03_training_monitoring.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/                            # Unit and integration tests
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_tiler.py
│   └── test_inference.py
│
├── docker/
│   ├── Dockerfile.train              # Training environment
│   └── Dockerfile.infer              # Inference environment
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── README.md                         # Project README
└── .gitignore
```

---

## 2. Core Class Designs

### 2.1 Data Classes

#### `Sample` — Single Training/Inference Sample

```python
from dataclasses import dataclass
from torch import Tensor
from typing import Optional

@dataclass
class Sample:
    """Represents a single tile for training or inference."""

    # Core inputs
    cloudy_image: Tensor          # (3, H, W) — LISS-IV with clouds, float32 [0, 1]
    cloud_mask: Tensor            # (1, H, W) — binary mask, 1 = cloud, 0 = clear
    ground_truth: Optional[Tensor]  # (3, H, W) — cloud-free GT (None during real inference)

    # Auxiliary inputs (optional — zero-filled if unavailable)
    sar: Optional[Tensor]         # (2, H, W) — Sentinel-1 VV/VH, normalized [-1, 1]
    sentinel2: Optional[Tensor]   # (4, H, W) — S2 B/G/R/NIR, float32 [0, 1]
    dem: Optional[Tensor]         # (1, H, W) — elevation, normalized [0, 1]
    temporal_ref: Optional[Tensor]  # (3, H, W) — previous cloud-free LISS-IV

    # Metadata
    tile_id: str                  # Unique identifier: "{scene_id}_{row}_{col}"
    scene_id: str                 # Source scene identifier
    crs: str                      # Coordinate reference system (e.g., "EPSG:32646")
    transform: tuple              # Affine geotransform (6 values)
```

#### `LISS4CloudDataset` — PyTorch Dataset

```python
class LISS4CloudDataset(torch.utils.data.Dataset):
    """
    Loads tiled patches for cloud removal training.

    For training: loads cloud-free GT, applies synthetic clouds.
    For inference: loads real cloudy imagery with detected masks.
    """

    def __init__(
        self,
        split_csv: str,           # Path to train.csv / val.csv / test.csv
        data_root: str,           # Path to processed data directory
        tile_size: int = 512,
        use_sar: bool = True,
        use_s2: bool = True,
        use_dem: bool = True,
        use_temporal: bool = True,
        augment: bool = True,     # Enable augmentations (training only)
        cloud_sim_mode: str = "perlin",  # "perlin" | "real_mask" | "mixed"
    ):
        ...

    def __len__(self) -> int:
        """Return number of tiles in this split."""
        ...

    def __getitem__(self, idx: int) -> Sample:
        """
        Load tile and create training sample.

        Training flow:
        1. Load cloud-free GT tile from disk
        2. Generate synthetic cloud mask (Perlin noise or real mask overlay)
        3. Create cloudy image: cloudy = GT * (1 - mask) + white * mask
        4. Load auxiliary data (SAR, S2, DEM, temporal) if available
        5. Apply augmentations
        6. Return Sample dataclass

        Inference flow:
        1. Load real cloudy tile
        2. Load detected cloud mask
        3. Load auxiliary data if available
        4. Return Sample (ground_truth = None)
        """
        ...
```

#### `LISS4DataModule` — Lightning DataModule

```python
class LISS4DataModule(pl.LightningDataModule):
    """Manages data loading for training, validation, and testing."""

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config["batch_size"]       # e.g., 8
        self.num_workers = config["num_workers"]     # e.g., 4
        self.tile_size = config["tile_size"]          # 512

    def setup(self, stage: str = None):
        """Create dataset instances for each split."""
        ...

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with augmentations and shuffling."""
        ...

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader without augmentations."""
        ...

    def test_dataloader(self) -> DataLoader:
        """Test dataloader for final evaluation."""
        ...
```

---

### 2.2 Model Classes

#### Diffusion Model (Primary)

```python
class ConditionalDiffusionModel(pl.LightningModule):
    """
    Conditional Latent Diffusion Model for cloud removal.

    Architecture:
        1. FusionEncoder: Encodes multi-modal auxiliary inputs (SAR, S2, DEM, temporal)
           into conditioning features.
        2. VAE: Encodes ground truth into latent space (frozen, pre-trained).
        3. U-Net: Conditional denoiser — takes noisy latent + timestep + condition features,
           predicts noise.
        4. VAE Decoder: Decodes denoised latent back to pixel space.

    Training:
        - Forward diffusion: add noise to GT latent at random timestep
        - Predict noise using conditioned U-Net
        - Loss = MSE(predicted_noise, actual_noise) + auxiliary losses

    Inference:
        - Start from random noise in latent space
        - Iteratively denoise using DDIM (10–50 steps)
        - Decode final latent to pixel space
    """

    def __init__(self, config: dict):
        super().__init__()
        self.fusion_encoder = FusionEncoder(in_channels=11, embed_dim=256)
        self.vae = load_pretrained_vae("stabilityai/sd-vae-ft-mse")
        self.vae.requires_grad_(False)  # Freeze VAE
        self.unet = ConditionalUNet(
            in_channels=4,             # Latent channels
            out_channels=4,
            cond_channels=256,         # From fusion encoder
            block_channels=(128, 256, 512, 512),
            attention_levels=(False, True, True, True),
            num_res_blocks=2,
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="cosine",
        )
        self.loss_fn = CombinedLoss(config["loss_weights"])

    def training_step(self, batch: Sample, batch_idx: int) -> Tensor:
        """Single training step."""
        # 1. Encode GT to latent space
        z0 = self.vae.encode(batch.ground_truth)

        # 2. Sample random timestep and add noise
        t = torch.randint(0, 1000, (z0.shape[0],), device=self.device)
        noise = torch.randn_like(z0)
        z_noisy = self.noise_scheduler.add_noise(z0, noise, t)

        # 3. Build conditioning input
        cond_input = self._build_condition(batch)
        cond_features = self.fusion_encoder(cond_input)

        # 4. Predict noise
        noise_pred = self.unet(z_noisy, t, cond_features)

        # 5. Compute loss
        loss = F.mse_loss(noise_pred, noise)

        # 6. Optional: decode and compute pixel-space auxiliary losses
        if batch_idx % 10 == 0:  # Every 10 steps for efficiency
            with torch.no_grad():
                z_pred = self.noise_scheduler.step(noise_pred, t[0], z_noisy).prev_sample
                pred_image = self.vae.decode(z_pred)
            aux_loss = self.loss_fn(pred_image, batch.ground_truth, batch.cloud_mask)
            loss = loss + 0.1 * aux_loss

        return loss

    @torch.no_grad()
    def generate(self, batch: Sample, num_steps: int = 20) -> Tensor:
        """Generate cloud-free image via iterative denoising."""
        cond_input = self._build_condition(batch)
        cond_features = self.fusion_encoder(cond_input)

        # Start from pure noise
        z = torch.randn(1, 4, 64, 64, device=self.device)

        # DDIM denoising loop
        self.noise_scheduler.set_timesteps(num_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(z, t, cond_features)
            z = self.noise_scheduler.step(noise_pred, t, z).prev_sample

        # Decode latent to pixel space
        reconstructed = self.vae.decode(z)
        return reconstructed
```

#### GAN Model (Baseline)

```python
class CloudRemovalGAN(pl.LightningModule):
    """
    Conditional GAN for cloud removal (Pix2Pix / SpaGAN style).

    Generator: U-Net with skip connections + SPADE normalization in decoder
    Discriminator: PatchGAN (70×70 receptive field)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.generator = UNetGenerator(
            in_channels=14,           # Full multi-modal input
            out_channels=3,           # Reconstructed G/R/NIR
            features=[64, 128, 256, 512, 512, 512],
            use_spade=True,
        )
        self.discriminator = PatchDiscriminator(
            in_channels=3 + 14,       # Real/fake + condition
            features=[64, 128, 256, 512],
        )
        self.automatic_optimization = False  # Manual G/D optimization

    def training_step(self, batch: Sample, batch_idx: int):
        g_opt, d_opt = self.optimizers()
        cond = self._build_condition(batch)
        fake = self.generator(cond)

        # --- Discriminator step ---
        d_opt.zero_grad()
        real_pair = torch.cat([batch.ground_truth, cond], dim=1)
        fake_pair = torch.cat([fake.detach(), cond], dim=1)
        d_loss = (
            F.mse_loss(self.discriminator(real_pair), torch.ones_like(...)) +
            F.mse_loss(self.discriminator(fake_pair), torch.zeros_like(...))
        ) * 0.5
        self.manual_backward(d_loss)
        d_opt.step()

        # --- Generator step ---
        g_opt.zero_grad()
        fake_pair = torch.cat([fake, cond], dim=1)
        g_adv = F.mse_loss(self.discriminator(fake_pair), torch.ones_like(...))
        g_rec = self.loss_fn(fake, batch.ground_truth, batch.cloud_mask)
        g_loss = g_rec + 0.1 * g_adv
        self.manual_backward(g_loss)
        g_opt.step()
```

#### Transformer Model (Baseline)

```python
class CloudRemovalTransformer(pl.LightningModule):
    """
    Swin Transformer-based cloud removal (SwinIR / Restormer style).

    Direct image-to-image translation without iterative sampling.
    Fastest inference among the three architectures.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = SwinTransformerEncoder(
            in_channels=14,
            embed_dim=96,
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            window_size=8,
        )
        self.decoder = PixelShuffleDecoder(
            in_channels=96,
            out_channels=3,
        )
        self.loss_fn = CombinedLoss(config["loss_weights"])

    def forward(self, batch: Sample) -> Tensor:
        cond = self._build_condition(batch)
        features = self.encoder(cond)
        reconstructed = self.decoder(features)
        return reconstructed
```

---

### 2.3 Loss Functions

```python
class CombinedLoss(nn.Module):
    """
    Multi-objective loss for cloud removal training.

    Components:
    - L1 (reconstruction): Pixel-level accuracy
    - MS-SSIM: Structural similarity preservation
    - LPIPS (perceptual): Deep feature-level similarity
    - SAM (spectral angle): Spectral consistency
    - Masked weighting: Higher loss weight on cloud regions
    """

    def __init__(self, weights: dict):
        super().__init__()
        self.w_l1 = weights.get("l1", 10.0)
        self.w_ssim = weights.get("ssim", 5.0)
        self.w_perceptual = weights.get("perceptual", 2.0)
        self.w_spectral = weights.get("spectral", 3.0)
        self.w_mask_focus = weights.get("mask_focus", 2.0)

        self.lpips = lpips.LPIPS(net="alex")
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=True)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # L1 reconstruction loss (full image)
        l1_loss = F.l1_loss(pred, target)

        # L1 on masked (cloud) region only — with higher weight
        l1_masked = F.l1_loss(pred * mask, target * mask)

        # MS-SSIM loss
        ssim_loss = 1.0 - self.ms_ssim(pred, target)

        # Perceptual loss (LPIPS)
        perceptual_loss = self.lpips(pred, target).mean()

        # Spectral Angle Mapper loss
        spectral_loss = self._sam_loss(pred, target)

        # Combined
        total = (
            self.w_l1 * l1_loss +
            self.w_mask_focus * l1_masked +
            self.w_ssim * ssim_loss +
            self.w_perceptual * perceptual_loss +
            self.w_spectral * spectral_loss
        )
        return total

    @staticmethod
    def _sam_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Spectral Angle Mapper — measures angle between spectral vectors.
        Lower angle = better spectral consistency.
        Returns mean angle in radians (for gradient flow).
        """
        dot_product = (pred * target).sum(dim=1, keepdim=True)
        norm_pred = pred.norm(dim=1, keepdim=True)
        norm_target = target.norm(dim=1, keepdim=True)
        cos_angle = dot_product / (norm_pred * norm_target + eps)
        angle = torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
        return angle.mean()
```

---

### 2.4 Fusion Encoder

```python
class FusionEncoder(nn.Module):
    """
    Encodes multi-modal auxiliary inputs into conditioning features.

    Input: 11-channel stack [cloudy_masked(3) + mask(1) + SAR(2) + DEM(1) + temporal(3) + S2(1 sentinel NIR)]
    Output: Multi-scale feature maps for conditioning the denoiser.

    Design choices:
    - Separate normalization branches for SAR (log-scale dB) and optical (reflectance)
    - Channel attention (SE blocks) to let the model learn which modalities are informative
    - Multi-scale output for U-Net cross-attention at different resolution levels
    """

    def __init__(self, in_channels: int = 14, embed_dim: int = 256):
        super().__init__()

        # Modality-specific stem layers
        self.optical_stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.sar_stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.InstanceNorm2d(32),     # InstanceNorm for SAR (different statistics)
            nn.GELU(),
        )
        self.aux_stem = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1),  # mask(1) + S2(4) + DEM(1) + temporal(3)
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Fusion body
        self.fusion_conv = nn.Conv2d(128, embed_dim, 1)  # 32+32+64 = 128
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock(embed_dim, embed_dim) for _ in range(4)
        ])

        # Channel attention
        self.se_block = SEBlock(embed_dim, reduction=16)

    def forward(self, x: Tensor) -> Tensor:
        # Split channels by modality
        optical = x[:, :3]        # Cloudy LISS-IV
        sar = x[:, 3:5]          # VV, VH
        aux = x[:, 5:]           # mask + S2 + DEM + temporal

        # Modality-specific encoding
        f_opt = self.optical_stem(optical)
        f_sar = self.sar_stem(sar)
        f_aux = self.aux_stem(aux)

        # Concatenate and fuse
        fused = torch.cat([f_opt, f_sar, f_aux], dim=1)
        fused = self.fusion_conv(fused)

        # Deep feature extraction
        for block in self.encoder_blocks:
            fused = block(fused)

        # Channel attention
        fused = self.se_block(fused)
        return fused
```

---

## 3. Inference Pipeline Classes

### 3.1 Tiler — Scene to Tiles and Back

```python
class SceneTiler:
    """
    Splits a full satellite scene into overlapping tiles for inference,
    and reassembles tiles back into a full scene with blending.
    """

    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def split(self, scene: np.ndarray) -> list[dict]:
        """
        Split scene into tiles.

        Args:
            scene: (C, H, W) full scene array

        Returns:
            List of dicts: {"array": (C, tile_size, tile_size),
                           "row": int, "col": int,
                           "pad_bottom": int, "pad_right": int}
        """
        ...

    def merge(self, tiles: list[dict], scene_shape: tuple) -> np.ndarray:
        """
        Merge tiles back into full scene with Gaussian-weighted blending.

        In overlap regions, pixel values are weighted by a 2D Gaussian
        centered on each tile to create smooth transitions.
        """
        ...
```

### 3.2 Predictor — Full Inference Engine

```python
class CloudRemovalPredictor:
    """
    End-to-end inference: takes a cloudy scene → produces cloud-free scene.

    Steps:
    1. Load and preprocess input scene
    2. Generate or load cloud mask
    3. Load and co-register auxiliary data (if available)
    4. Tile the scene
    5. Run model inference on each tile (batched)
    6. Merge tiles with blending
    7. Apply post-processing (histogram matching, QA)
    8. Export as GeoTIFF
    """

    def __init__(self, model_path: str, config: dict, device: str = "cuda"):
        self.model = self._load_model(model_path)
        self.tiler = SceneTiler(config["tile_size"], config["overlap"])
        self.config = config
        self.device = device

    def predict(
        self,
        liss4_path: str,
        output_path: str,
        mask_path: str = None,
        sar_path: str = None,
        temporal_path: str = None,
    ) -> dict:
        """
        Run cloud removal on a full LISS-IV scene.

        Returns:
            dict with keys: "output_path", "qa_mask_path", "metrics" (if GT available)
        """
        ...
```

### 3.3 Exporter — GeoTIFF Output

```python
class GeoTiffExporter:
    """
    Exports reconstructed imagery as georeferenced GeoTIFF.

    Preserves:
    - Coordinate Reference System (CRS) from input
    - Affine geotransform
    - Band descriptions
    - Nodata value

    Also generates:
    - QA mask (0 = original pixel, 1 = reconstructed pixel)
    - Metadata JSON with processing parameters
    """

    def export(
        self,
        array: np.ndarray,         # (3, H, W) reconstructed image
        profile: dict,             # Rasterio profile from input
        output_path: str,
        qa_mask: np.ndarray = None,
    ) -> str:
        """Write GeoTIFF and return output path."""
        ...
```

---

## 4. API Contract (FastAPI)

### 4.1 Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| `POST` | `/api/v1/reconstruct` | Reconstruct cloud-free imagery from uploaded files | Multipart form: `image` (required), `mask`, `sar`, `temporal` | JSON: `{"job_id": "...", "status": "processing"}` |
| `GET` | `/api/v1/jobs/{job_id}` | Check job status | Path param: `job_id` | JSON: `{"status": "done", "result_url": "...", "metrics": {...}}` |
| `POST` | `/api/v1/reconstruct/sync` | Synchronous single-tile reconstruction | Multipart form: `image`, `mask` | Binary GeoTIFF response |
| `POST` | `/api/v1/mask/detect` | Generate cloud mask for input image | Multipart form: `image` | Binary mask GeoTIFF |
| `GET` | `/api/v1/health` | Health check | None | JSON: `{"status": "healthy", "model_loaded": true, "gpu_available": true}` |

### 4.2 Request/Response Examples

**POST `/api/v1/reconstruct/sync`:**

```bash
# Request
curl -X POST http://localhost:8000/api/v1/reconstruct/sync \
  -F "image=@cloudy_tile.tif" \
  -F "mask=@cloud_mask.tif" \
  -F "sar=@sentinel1_vvvh.tif" \
  --output reconstructed.tif

# Response: Binary GeoTIFF file (application/geo+tiff)
```

**Error Response:**
```json
{
  "error": "INVALID_INPUT",
  "message": "Input image must be a 3-band GeoTIFF. Received 4 bands.",
  "status_code": 400
}
```

---

## 5. Database Schema (SQLite — Tile Tracking)

```sql
-- Tile metadata for training/inference tracking
CREATE TABLE tiles (
    tile_id         TEXT PRIMARY KEY,     -- e.g., "scene001_r03_c07"
    scene_id        TEXT NOT NULL,
    split           TEXT CHECK(split IN ('train', 'val', 'test')),
    row_idx         INTEGER,
    col_idx         INTEGER,
    cloud_percent   REAL,                 -- Cloud coverage percentage
    has_sar         BOOLEAN DEFAULT 0,
    has_temporal    BOOLEAN DEFAULT 0,
    has_s2          BOOLEAN DEFAULT 0,
    path_gt         TEXT,                 -- Path to ground truth tile
    path_cloudy     TEXT,                 -- Path to cloudy tile
    path_mask       TEXT,                 -- Path to cloud mask
    bbox_minx       REAL,
    bbox_miny       REAL,
    bbox_maxx       REAL,
    bbox_maxy       REAL,
    crs             TEXT DEFAULT 'EPSG:32646',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Experiment tracking
CREATE TABLE experiments (
    experiment_id   TEXT PRIMARY KEY,
    model_type      TEXT NOT NULL,        -- 'diffusion', 'gan', 'transformer'
    config_json     TEXT,                 -- Full training config as JSON
    best_psnr       REAL,
    best_ssim       REAL,
    best_sam        REAL,
    checkpoint_path TEXT,
    status          TEXT DEFAULT 'running',
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMP
);

-- Per-tile evaluation results
CREATE TABLE evaluations (
    eval_id         TEXT PRIMARY KEY,
    experiment_id   TEXT REFERENCES experiments(experiment_id),
    tile_id         TEXT REFERENCES tiles(tile_id),
    psnr            REAL,
    ssim            REAL,
    sam_degrees     REAL,
    lpips           REAL,
    cc_green        REAL,
    cc_red          REAL,
    cc_nir          REAL,
    evaluated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. Configuration Schema

```yaml
# configs/train_diffusion.yaml

# ---- Data ----
data:
  root: "./data/processed"
  splits_dir: "./data/splits"
  tile_size: 512
  overlap: 64
  batch_size: 8
  num_workers: 4
  use_sar: true
  use_s2: true
  use_dem: true
  use_temporal: true
  cloud_sim_mode: "mixed"      # "perlin", "real_mask", "mixed"
  cloud_coverage_range: [0.1, 0.8]  # Min/max synthetic cloud coverage

# ---- Model ----
model:
  type: "diffusion"            # "diffusion", "gan", "transformer"
  diffusion:
    num_train_timesteps: 1000
    beta_schedule: "cosine"    # "linear", "cosine", "scaled_linear"
    inference_steps: 20        # DDIM steps for inference
    vae_pretrained: "stabilityai/sd-vae-ft-mse"
    unet_channels: [128, 256, 512, 512]
    attention_resolutions: [32, 16, 8]
    num_res_blocks: 2
  fusion:
    embed_dim: 256
    num_blocks: 4
    use_channel_attention: true

# ---- Loss ----
loss:
  l1: 10.0
  ssim: 5.0
  perceptual: 2.0
  spectral: 3.0
  mask_focus: 2.0
  adversarial: 0.0             # Set > 0 to enable PatchGAN discriminator

# ---- Training ----
training:
  max_epochs: 200
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  optimizer: "adamw"           # "adam", "adamw"
  scheduler: "cosine"          # "cosine", "step", "plateau"
  warmup_epochs: 5
  gradient_clip_val: 1.0
  precision: "16-mixed"        # "32", "16-mixed", "bf16-mixed"
  accumulate_grad_batches: 2
  seed: 42

# ---- Hardware ----
hardware:
  accelerator: "gpu"
  devices: 1                   # Number of GPUs
  strategy: "auto"             # "auto", "ddp", "fsdp"

# ---- Logging ----
logging:
  project_name: "liss4-cloud-removal"
  log_every_n_steps: 50
  save_top_k: 3                # Keep top 3 checkpoints by val_psnr
  monitor_metric: "val/psnr"
  mode: "max"

# ---- Checkpointing ----
checkpoint:
  dirpath: "./checkpoints"
  filename: "{epoch}-{val_psnr:.2f}"
```

---

## 7. Testing Strategy

| Test Level | Scope | Tools | Examples |
|-----------|-------|-------|---------|
| **Unit** | Individual functions and classes | `pytest` | Test `SAM_loss` output range, test `SceneTiler.split()` → `merge()` roundtrip, test dataset `__len__` |
| **Integration** | Multi-component pipelines | `pytest` + fixtures | Test preprocess → tile → inference → mosaic end-to-end on a small synthetic scene |
| **Model** | Forward/backward pass, shapes, export | `pytest` + `torch.testing` | Test model forward produces correct output shape, test ONNX export succeeds |
| **Data Validation** | Input data quality | Custom validators | Check CRS consistency, nodata values, value ranges, band count |
| **Visual QA** | Qualitative output inspection | Jupyter notebooks | Side-by-side visual comparison of input/output/ground truth |

---

## 8. Error Handling Strategy

| Error Type | Handling |
|-----------|---------|
| Missing auxiliary data (SAR/S2/DEM/temporal) | Zero-fill missing channels, set availability flag, log warning |
| CRS mismatch between inputs | Auto-reproject to LISS-IV's CRS with warning |
| Tile with >95% cloud cover | Process but flag QA=2 ("low confidence"), log warning |
| GPU out of memory | Reduce batch size dynamically, enable gradient checkpointing |
| Invalid input file format | Return 400 error with descriptive message |
| Model checkpoint not found | Return 500 error, log critical |

---

> **Next →** Read [04-project-architecture.md](./04-project-architecture.md) for end-to-end architecture diagrams.
