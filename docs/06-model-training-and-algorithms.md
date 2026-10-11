# 06 — Model Training Guide and Algorithms
## Generative AI-Based Cloud Removal for LISS-IV Imagery

---

## 1. Algorithm Selection — Why These Approaches?

### 1.1 The Cloud Removal Problem as an Image Inpainting Task

Cloud removal is fundamentally an **image inpainting** problem — we have an image with missing/corrupted regions (clouds) and need to fill them with plausible content. However, satellite cloud removal differs from natural image inpainting:

| Aspect | Natural Image Inpainting | Satellite Cloud Removal |
|--------|-------------------------|------------------------|
| Missing region | Typically small, regular shapes | Large, irregular cloud shapes (10–80% of tile) |
| Spectral bands | RGB (3 channels) | Multispectral (3+ channels with physical meaning) |
| Texture | Complex but no physical constraints | Must preserve NDVI, spectral correlations |
| Multi-modal input | Usually single image | SAR, temporal, DEM available as auxiliary |
| Evaluation | Visual quality (FID, LPIPS) | Physical accuracy (SAM, band CC, NDVI error) |

### 1.2 Algorithm Comparison

We evaluate three families of generative models:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ALGORITHM COMPARISON MATRIX                        │
├───────────────────┬────────────┬──────────────┬──────────────────────┤
│ Criterion         │ GAN        │ Diffusion    │ Transformer          │
│                   │ (Pix2Pix/  │ (DDPM/DDIM)  │ (SwinIR/Restormer)   │
│                   │ SpaGAN)    │              │                      │
├───────────────────┼────────────┼──────────────┼──────────────────────┤
│ Image Quality     │ Good       │ Excellent    │ Very Good            │
│ Texture Detail    │ Good       │ Excellent    │ Very Good            │
│ Spectral Fidelity │ Good       │ Very Good    │ Very Good            │
│ Training Stability│ Poor       │ Excellent    │ Good                 │
│ Inference Speed   │ Very Fast  │ Slow         │ Fast                 │
│                   │ (~50ms)    │ (~2-5s)      │ (~200ms)             │
│ VRAM (Training)   │ 12-16 GB   │ 24-40 GB     │ 16-24 GB             │
│ Mode Collapse     │ Possible   │ No           │ No                   │
│ Large Mask Fill   │ Poor       │ Good         │ Good                 │
│ Multi-modal Input │ Easy       │ Easy         │ Easy                 │
│ Research Maturity │ High       │ High         │ Medium               │
├───────────────────┼────────────┼──────────────┼──────────────────────┤
│ Overall Rating    │ ★★★☆☆      │ ★★★★★        │ ★★★★☆                │
└───────────────────┴────────────┴──────────────┴──────────────────────┘
```

### 1.3 Why Conditional Latent Diffusion is the Primary Choice

1. **Best reconstruction quality:** Diffusion models generate the most realistic textures and details, critical for 5.8m satellite imagery where individual features (buildings, field boundaries) matter

2. **Stable training:** Unlike GANs, diffusion models don't suffer from mode collapse or training oscillation. The loss monotonically decreases — no adversarial min-max game

3. **Latent space efficiency:** By working in a compressed latent space (64×64 instead of 512×512), we reduce memory and compute by ~50× while maintaining visual quality

4. **Conditional generation is natural:** The denoiser can be conditioned on arbitrary auxiliary inputs (SAR, temporal, mask) through cross-attention — no architectural hacks needed

5. **Controllable quality-speed tradeoff:** DDIM sampling allows choosing between quality (50 steps) and speed (10 steps) at inference time without retraining

---

## 2. Algorithm Deep Dives

### 2.1 Conditional Latent Diffusion Model (Primary)

#### How Diffusion Models Work

Diffusion models learn to generate images by learning to **reverse a gradual noising process**:

```
FORWARD PROCESS (Training — adds noise):
──────────────────────────────────────
  Clean Image (z₀) → Add noise → z₁ → Add more noise → z₂ → ... → z_T (pure noise)

  Mathematically: z_t = √(ᾱ_t) · z₀ + √(1 - ᾱ_t) · ε,  where ε ~ N(0, I)

  ᾱ_t follows a "noise schedule" (cosine schedule preferred):
  - At t=0:   ᾱ₀ ≈ 1.0   → z₀ ≈ clean image
  - At t=500: ᾱ₅₀₀ ≈ 0.5 → z₅₀₀ = mix of image and noise
  - At t=1000: ᾱ₁₀₀₀ ≈ 0  → z₁₀₀₀ ≈ pure noise

REVERSE PROCESS (Inference — removes noise):
──────────────────────────────────────────
  Pure Noise (z_T) → Denoise → z_{T-1} → Denoise → ... → z₀ (clean image)

  The neural network (U-Net) learns: given z_t and t, predict the noise ε
  Then: z_{t-1} = f(z_t, ε_predicted, t)    (using DDIM or DDPM update rule)
```

#### Our Adaptation for Cloud Removal

Standard diffusion generates images from scratch. We modify it for **conditional inpainting**:

```
Instead of: z_T (random) → denoise → clean image
We do:      z_T (random) + CONDITION (cloudy image + mask + SAR + ...) → denoise → cloud-free image

The condition is injected via:
1. Cross-attention in the U-Net: condition features attend to noisy latent
2. Concatenation: mask and cloudy image appended to latent channels
```

#### Why "Latent" Diffusion?

```
Pixel-space diffusion:  Operate on 512×512×3 = 786,432 values
Latent-space diffusion: Operate on  64×64×4  =  16,384 values  (48× fewer!)

How: A pre-trained VAE compresses the image:
  512×512×3 → [VAE Encoder] → 64×64×4 (latent) → [VAE Decoder] → 512×512×3

The diffusion process happens entirely in this compressed latent space.
VAE is frozen (not trained) — we use Stability AI's pre-trained VAE.
```

#### Training Algorithm (Pseudocode)

```python
def train_step(model, batch):
    """One training step for conditional latent diffusion."""

    # 1. Get ground truth and encode to latent space
    gt_image = batch.ground_truth                    # (B, 3, 512, 512)
    z0 = vae.encode(gt_image).latent                 # (B, 4, 64, 64)

    # 2. Sample random timestep for each sample in batch
    t = torch.randint(0, 1000, (B,))                 # (B,)

    # 3. Sample random noise
    noise = torch.randn_like(z0)                     # (B, 4, 64, 64)

    # 4. Create noisy latent at timestep t
    z_t = sqrt(alpha_bar[t]) * z0 + sqrt(1 - alpha_bar[t]) * noise

    # 5. Build conditioning features from auxiliary data
    cond_stack = stack([                              # (B, 14, 512, 512)
        batch.cloudy_image,   # 3 channels
        batch.cloud_mask,     # 1 channel
        batch.sar,            # 2 channels (or zeros)
        batch.sentinel2,      # 4 channels (or zeros)
        batch.dem,            # 1 channel  (or zeros)
        batch.temporal_ref,   # 3 channels (or zeros)
    ], dim=1)
    cond_features = fusion_encoder(cond_stack)        # (B, 256, H', W')

    # 6. Predict noise using conditioned U-Net
    noise_pred = unet(z_t, t, cond_features)         # (B, 4, 64, 64)

    # 7. Simple noise prediction loss
    loss = MSE(noise_pred, noise)

    return loss
```

#### Inference Algorithm (DDIM Sampling)

```python
def generate(model, batch, num_steps=20):
    """Generate cloud-free image using DDIM sampling."""

    # 1. Build conditioning
    cond_stack = build_condition(batch)
    cond_features = fusion_encoder(cond_stack)

    # 2. Start from random noise in latent space
    z = torch.randn(1, 4, 64, 64)                    # Pure noise

    # 3. Set up DDIM timesteps (e.g., 20 evenly spaced from 1000 to 0)
    timesteps = linspace(999, 0, num_steps)           # [999, 947, 894, ..., 0]

    # 4. Iterative denoising
    for i, t in enumerate(timesteps):
        # Predict noise at current step
        noise_pred = unet(z, t, cond_features)

        # DDIM update: compute z at next (less noisy) timestep
        # z_{t-1} = sqrt(α_{t-1}) * predicted_z0 + sqrt(1-α_{t-1}) * direction
        pred_z0 = (z - sqrt(1-alpha_bar[t]) * noise_pred) / sqrt(alpha_bar[t])
        t_next = timesteps[i+1] if i < len(timesteps)-1 else 0
        direction = sqrt(1 - alpha_bar[t_next]) * noise_pred
        z = sqrt(alpha_bar[t_next]) * pred_z0 + direction

    # 5. Decode latent to pixel space
    reconstructed = vae.decode(z)                     # (1, 3, 512, 512)

    # 6. Composite: keep original clear pixels, use reconstruction only for clouds
    output = batch.cloudy_image * (1 - batch.cloud_mask) + reconstructed * batch.cloud_mask

    return output
```

---

### 2.2 GAN — Pix2Pix / SpaGAN (Baseline 1)

#### How GANs Work for Cloud Removal

```
GENERATOR (G):
  Input:  Cloudy image + mask + auxiliary data (14 channels)
  Output: Reconstructed cloud-free image (3 channels)
  Architecture: U-Net with skip connections
    - Encoder: progressively downsample and extract features
    - Decoder: progressively upsample with skip connections from encoder
    - SPADE normalization in decoder: uses mask to spatially modulate features

DISCRIMINATOR (D):
  Input:  Real or fake image (3ch) concatenated with condition (14ch)
  Output: Patch-level real/fake prediction (not a single yes/no)
  Architecture: PatchGAN (70×70 receptive field)
    - Each output pixel judges a 70×70 patch of the input

TRAINING:
  D tries to distinguish real cloud-free images from G's fakes
  G tries to fool D while also minimizing pixel-level and perceptual losses

  G_loss = L1(G(x), y) * 100 + VGG_perceptual(G(x), y) * 10 + Adversarial * 1
  D_loss = BCE(D(real), 1) + BCE(D(fake), 0)
```

#### Why GAN as Baseline?

- **Fastest inference** (~50ms single forward pass, no iterative sampling)
- **Well-studied** for satellite image translation (Pix2Pix, SpaGAN, DSen2 lineage)
- **Good enough** for many applications — acceptable quality with minimal compute
- **Weakness:** Training instability, possible mode collapse, less texture detail than diffusion

#### Training Configuration

```yaml
# configs/train_gan.yaml
model:
  type: "gan"
  generator:
    architecture: "unet"       # "unet" or "resnet9"
    features: [64, 128, 256, 512, 512, 512]
    use_spade: true
    dropout: 0.5               # In bottleneck layers
  discriminator:
    type: "patchgan"
    n_layers: 3
    features: [64, 128, 256, 512]

loss:
  l1: 100.0
  perceptual_vgg: 10.0
  adversarial: 1.0
  spectral_sam: 5.0
  ssim: 5.0

training:
  max_epochs: 300
  learning_rate_g: 2.0e-4
  learning_rate_d: 2.0e-4
  optimizer: "adam"
  beta1: 0.5
  beta2: 0.999
  lr_decay_start: 150          # Start linear decay at epoch 150
```

---

### 2.3 Transformer — SwinIR / Restormer (Baseline 2)

#### How Transformers Work for Cloud Removal

```
INPUT: 14-channel condition stack (14, 512, 512)
    │
    ▼
PATCH EMBEDDING: Split into non-overlapping patches
    Each patch: 4×4 pixels → 1 token (vector)
    Result: 128×128 = 16,384 tokens, each 96-dimensional
    │
    ▼
SWIN TRANSFORMER BLOCKS (×24):
    For each block:
    1. Window Multi-Head Self-Attention (W-MSA)
       - Divide tokens into 8×8 windows
       - Each token attends to all others in its window
       - Complexity: O(window_size² × num_windows) instead of O(N²)

    2. Shifted Window MSA (SW-MSA)
       - Shift windows by half their size
       - Enables cross-window communication
       - Alternates with W-MSA every other block

    3. Feed-Forward Network (MLP)
       - 2-layer MLP with GELU activation
       - Expand to 4× dim, then project back

    Every 6 blocks: Patch Merging
       - 2×2 neighboring tokens → 1 token (2× dim)
       - Reduces spatial resolution, increases channel dim
    │
    ▼
PIXEL SHUFFLE DECODER:
    Upsample back to original resolution
    3-channel output: reconstructed Green, Red, NIR
```

#### Why Transformer as Baseline?

- **Long-range context:** Self-attention can relate distant pixels — important for large cloud gaps where the model must look far for context
- **No iterative sampling:** Single forward pass like GAN, but without adversarial training instability
- **State-of-the-art** for image restoration tasks (super-resolution, denoising, dehazing)
- **Weakness:** Higher memory than GAN, may need more training data

#### Training Configuration

```yaml
# configs/train_transformer.yaml
model:
  type: "transformer"
  backbone: "swin_ir"
  embed_dim: 96
  depths: [6, 6, 6, 6]
  num_heads: [6, 6, 6, 6]
  window_size: 8
  mlp_ratio: 4.0
  patch_size: 4
  upsampler: "pixelshuffle"

loss:
  l1: 10.0
  ssim: 5.0
  perceptual: 2.0
  spectral_sam: 3.0
  mask_focus: 2.0

training:
  max_epochs: 250
  learning_rate: 2.0e-4
  optimizer: "adamw"
  weight_decay: 0.0
  scheduler: "cosine"
  warmup_epochs: 10
  batch_size: 8
```

---

## 3. Training Procedure — Step by Step

### 3.1 Data Preparation

```
Step 1: Organize raw data
─────────────────────────
  data/raw/liss4/         ← LISS-IV scenes from Bhoonidhi
  data/raw/sentinel1/     ← S1 GRD products
  data/raw/sentinel2/     ← S2 L2A products
  data/raw/dem/           ← SRTM/COP DEM

Step 2: Preprocess each source
──────────────────────────────
  python scripts/preprocess_all.py --config configs/config.yaml

  This runs:
  - LISS-IV: merge bands → calibrate → reproject to UTM → normalize [0,1]
  - Sentinel-1: orbit file → calibrate σ₀ → speckle filter → dB → normalize [-1,1]
  - Sentinel-2: extract B2/B3/B4/B8 → reproject → resample to 5.8m → normalize [0,1]
  - DEM: reproject → resample to 5.8m → normalize [0,1]

Step 3: Co-register all sources to LISS-IV grid
──────────────────────────────────────────────────
  - Target: LISS-IV CRS + transform + resolution
  - Use gdal.Warp with target_aligned_pixels for exact pixel alignment
  - Sub-pixel correction with phase correlation (if needed)

Step 4: Generate cloud masks for cloud-free scenes
───────────────────────────────────────────────────
  - For training: we START with cloud-free LISS-IV and ADD synthetic clouds
  - Synthetic cloud masks: Perlin noise thresholding + real cloud mask library
  - This gives us perfect (input, ground truth) pairs

Step 5: Tile into 512×512 patches
─────────────────────────────────
  python scripts/preprocess_all.py --mode tile
  
  - Stride: 448 (overlap: 64)
  - Save as .npz with keys: "stack" (14,512,512), "gt" (3,512,512), "mask" (1,512,512)

Step 6: Create train/val/test splits
─────────────────────────────────────
  - Split by SCENE (not tile) to prevent data leakage
  - 70% train / 15% val / 15% test (by scene count)
  - Stratify by cloud coverage percentage
  - Save as CSV: tile_id, scene_id, cloud_pct, has_sar, has_temporal
```

### 3.2 Synthetic Cloud Generation

A critical training design decision: since ground truth under real clouds doesn't exist, we train on **synthetic clouds applied to cloud-free imagery**.

```python
class CloudSimulator:
    """
    Generates realistic synthetic cloud masks for training.

    Methods:
    1. Perlin noise: smooth, organic cloud shapes
    2. Real mask transfer: copy real cloud masks from other scenes
    3. Mixed: randomly choose between methods

    Cloud properties simulated:
    - Opaque thick clouds (white, complete occlusion)
    - Semi-transparent thin clouds (partial occlusion, alpha blending)
    - Cloud shadows (dark, offset from cloud position)
    """

    def generate_mask(self, height, width, coverage_range=(0.1, 0.8)):
        """Generate a synthetic cloud mask."""
        coverage = random.uniform(*coverage_range)

        # Method 1: Perlin noise
        noise = generate_perlin_noise_2d((height, width), res=(4, 4))
        threshold = np.percentile(noise, (1 - coverage) * 100)
        mask = (noise > threshold).astype(np.float32)

        # Apply morphological operations for realistic shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (11, 11), 3)
        mask = (mask > 0.5).astype(np.float32)

        return mask

    def apply_cloud(self, image, mask, alpha=0.9):
        """
        Apply synthetic cloud to a clean image.

        image: (3, H, W) cloud-free
        mask:  (1, H, W) binary cloud mask
        alpha: opacity (1.0 = fully opaque white)
        """
        cloud_color = 0.95  # Near-white
        cloudy = image * (1 - mask * alpha) + cloud_color * mask * alpha
        return cloudy
```

### 3.3 Training Commands

```bash
# ============================================
# TRAINING COMMANDS
# ============================================

# --- Diffusion Model (Primary) ---
python scripts/train.py \
  --config configs/train_diffusion.yaml \
  --data.root ./data/processed \
  --data.batch_size 8 \
  --training.max_epochs 200 \
  --training.learning_rate 1e-4 \
  --training.precision 16-mixed \
  --hardware.devices 1 \
  --logging.project_name liss4-cloud-removal

# With multiple GPUs (DDP):
python scripts/train.py \
  --config configs/train_diffusion.yaml \
  --hardware.devices 4 \
  --hardware.strategy ddp \
  --data.batch_size 4  # Per-GPU batch size (effective batch = 4 × 4 = 16)

# Resume from checkpoint:
python scripts/train.py \
  --config configs/train_diffusion.yaml \
  --ckpt_path checkpoints/last.ckpt

# --- GAN Model (Baseline) ---
python scripts/train.py \
  --config configs/train_gan.yaml \
  --training.max_epochs 300

# --- Transformer Model (Baseline) ---
python scripts/train.py \
  --config configs/train_transformer.yaml \
  --training.max_epochs 250
```

### 3.4 Hyperparameter Guide

#### Diffusion Model Hyperparameters

| Parameter | Recommended | Range | Effect |
|-----------|-------------|-------|--------|
| `learning_rate` | 1e-4 | 5e-5 to 3e-4 | Higher → faster convergence but potential instability |
| `batch_size` | 8 | 2–16 | Higher → better gradient estimates, needs more VRAM |
| `num_train_timesteps` | 1000 | 500–2000 | More steps → finer noise schedule, slight quality gain |
| `beta_schedule` | cosine | linear, cosine | Cosine provides more uniform SNR across timesteps |
| `ddim_inference_steps` | 20 | 10–50 | More steps → better quality, slower inference |
| `loss_weight_l1` | 10.0 | 5–20 | Higher → more pixel-accurate, potentially less texture |
| `loss_weight_ssim` | 5.0 | 1–10 | Higher → better structural preservation |
| `loss_weight_spectral` | 3.0 | 1–5 | Higher → better spectral consistency (SAM) |
| `max_epochs` | 200 | 100–400 | Monitor val loss — stop when plateau |
| `gradient_clip_val` | 1.0 | 0.5–2.0 | Prevents exploding gradients |
| `precision` | 16-mixed | 32, 16-mixed, bf16 | Mixed precision saves ~40% VRAM with minimal quality loss |

#### GAN Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `lr_generator` | 2e-4 | Standard for Pix2Pix |
| `lr_discriminator` | 2e-4 | Keep same as generator |
| `beta1` | 0.5 | Lower than usual — stabilizes GAN training |
| `l1_weight` | 100 | High L1 weight prevents mode collapse |
| `adversarial_weight` | 1.0 | Low relative to L1 — reconstruction accuracy first |
| `lr_decay_epoch` | 150 | Start linear LR decay halfway through training |

#### Transformer Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `learning_rate` | 2e-4 | Transformers often benefit from slightly higher LR |
| `window_size` | 8 | Swin window — 8 is standard for 512 input |
| `embed_dim` | 96 | Swin-Tiny default — increase to 128/192 for Swin-Small/Base |
| `warmup_epochs` | 10 | Transformers benefit from LR warmup |
| `weight_decay` | 0 | SwinIR paper uses 0 weight decay |

### 3.5 Training Monitoring

```python
# Key metrics to monitor during training:

# 1. Training loss curve — should decrease monotonically (especially for diffusion)
# 2. Validation PSNR — should increase, plateau indicates convergence
# 3. Validation SSIM — should increase toward 0.90+
# 4. Validation SAM — should decrease toward < 3.0°
# 5. Learning rate — verify schedule is applied correctly
# 6. Gradient norm — should stay < gradient_clip_val

# TensorBoard:
tensorboard --logdir lightning_logs/ --port 6006

# W&B (if configured):
# Metrics automatically logged to wandb.ai dashboard
```

### 3.6 Common Training Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|---------|
| **OOM (Out of Memory)** | CUDA OOM error | Reduce batch_size, enable gradient checkpointing, use mixed precision |
| **Training loss doesn't decrease** | Loss plateau from epoch 1 | Check data loading (visualize samples), reduce learning rate, check normalization |
| **GAN mode collapse** | Generator produces identical outputs | Increase L1 weight, reduce adversarial weight, use spectral normalization |
| **Blurry outputs** | Low SSIM, visually blurry | Add perceptual loss (LPIPS/VGG), increase SSIM loss weight |
| **Spectral distortion** | High SAM, wrong colors | Increase spectral loss weight, verify band normalization |
| **Seam artifacts** | Visible tile boundaries in mosaic | Increase overlap (128), improve blending weights |
| **Slow convergence** | >200 epochs without improvement | Use pre-trained weights (transfer learning), increase batch size |
| **Val loss diverges from train** | Overfitting | Add augmentations, reduce model size, increase training data |

---

## 4. Transfer Learning Strategy

### 4.1 Pre-trained Components

| Component | Source | Freeze? | Purpose |
|-----------|--------|---------|---------|
| VAE (encoder + decoder) | `stabilityai/sd-vae-ft-mse` | Yes (frozen) | Compress/decompress images to/from latent space |
| U-Net denoiser | `runwayml/stable-diffusion-v1-5` (optional) | No (fine-tune) | Can initialize from SD weights, then fine-tune on satellite data |
| Fusion Encoder | Train from scratch | No | No pre-trained model exists for SAR+optical fusion |
| Cloud Detector (masking) | `segmentation-models-pytorch` + ImageNet backbone | Partially | ResNet/EfficientNet encoder pre-trained on ImageNet |

### 4.2 Fine-Tuning Protocol

```
Phase 1: Pre-train on Sentinel-2 (abundant data)
────────────────────────────────────────────────
  - Sentinel-2 has similar spectral bands but 10m resolution
  - Much more freely available data than LISS-IV
  - Train cloud removal on S2 first (50,000+ tiles easily available)
  - Use SEN2-CLOUD dataset or create synthetic clouds on S2

Phase 2: Fine-tune on LISS-IV (target data)
────────────────────────────────────────────
  - Load Phase 1 weights
  - Fine-tune on LISS-IV data (smaller dataset)
  - Lower learning rate (1e-5 to 5e-5) to preserve learned features
  - Train for fewer epochs (50-100)

Phase 3: Domain adaptation (optional)
──────────────────────────────────────
  - Fine-tune specifically on NER region tiles
  - Address domain shift: NER has unique terrain, vegetation, cloud patterns
```

---

## 5. Evaluation Methodology

### 5.1 Quantitative Metrics

| Metric | What It Measures | Formula / Method | Target |
|--------|-----------------|-----------------|--------|
| **PSNR** (dB) | Pixel-level accuracy | 10 · log₁₀(MAX² / MSE) | > 30 dB |
| **SSIM** | Structural similarity | Luminance × Contrast × Structure | > 0.90 |
| **SAM** (degrees) | Spectral angle between pixel vectors | arccos(dot(pred, gt) / (‖pred‖·‖gt‖)) | < 3.0° |
| **LPIPS** | Perceptual similarity (deep features) | VGG/AlexNet feature distance | < 0.15 |
| **CC** | Per-band correlation coefficient | Pearson correlation | > 0.93 |
| **FID** | Distribution quality | Fréchet distance in Inception feature space | < 50 |
| **NDVI MAE** | Vegetation index accuracy | ‖NDVI_pred - NDVI_gt‖ | < 0.03 |

### 5.2 Evaluation Script

```bash
# Run evaluation on test set
python scripts/evaluate.py \
  --model_path checkpoints/best_diffusion.ckpt \
  --test_data data/splits/test.csv \
  --output_dir results/diffusion/ \
  --metrics psnr,ssim,sam,lpips,cc,fid,ndvi_mae

# Compare architectures
python scripts/evaluate.py \
  --model_paths checkpoints/best_diffusion.ckpt,checkpoints/best_gan.ckpt,checkpoints/best_transformer.ckpt \
  --model_names "Diffusion,GAN,Transformer" \
  --output_dir results/comparison/ \
  --generate_report
```

### 5.3 Evaluation by Cloud Coverage

We stratify evaluation by cloud coverage to understand performance under different conditions:

| Cloud Coverage | Description | Expected Performance |
|---------------|-------------|---------------------|
| 0–20% (light) | Small scattered clouds | Excellent (plenty of context) |
| 20–50% (moderate) | Significant cloud patches | Good (sufficient context + SAR helps) |
| 50–80% (heavy) | Most of tile clouded | Acceptable (SAR + temporal critical) |
| 80–100% (extreme) | Nearly all pixels clouded | Challenging (heavy reliance on auxiliary data) |

---

## 6. Model Export and Optimization

### 6.1 ONNX Export

```python
# Export trained model to ONNX for optimized inference
python scripts/export_onnx.py \
  --checkpoint checkpoints/best_diffusion.ckpt \
  --output models/diffusion_v1.onnx \
  --input_shape 1,14,512,512 \
  --opset 17
```

### 6.2 Inference Optimization Checklist

```
□ Use DDIM with 10–20 steps (instead of 1000 DDPM steps)
□ Export to ONNX and use ONNX Runtime
□ Enable mixed precision (FP16) inference
□ Batch multiple tiles (batch_size=4–8)
□ Pre-compute and cache auxiliary data co-registration
□ Use torch.compile() for PyTorch 2.x (if not using ONNX)
□ Profile with torch.profiler to find bottlenecks
```

---

> **Next →** Read [07-complete-workflow-pipeline.md](./07-complete-workflow-pipeline.md) for the step-by-step operational pipeline.
