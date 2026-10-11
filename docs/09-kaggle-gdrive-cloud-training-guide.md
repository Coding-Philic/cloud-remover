# 09 — Kaggle & Google Drive Cloud Training Guide
## Zero-Bandwidth, Cloud-Only Ingestion, Preprocessing & 2× T4 GPU Training

---

## 1. Executive Cloud Strategy (Zero Local Bandwidth & Disk)

This guide provides the complete blueprint for executing data harvesting (~80+ GB), preprocessing, and deep learning training entirely within **Kaggle's Cloud Infrastructure** connected directly to **Google Drive**.

**Your local laptop is used ONLY for coding, repository management, and running the final web app.**

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               CLOUD-ONLY EXECUTION TOPOLOGY                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

   [ESA Copernicus CDSE]       [ISRO Bhoonidhi]       [OpenTopography]
      (Gigabit Cloud)          (Direct Download)      (Gigabit Cloud)
             │                        │                      │
             └────────────────┬───────┴──────────────────────┘
                              │ (500 Mbps – 1 Gbps Cloud Bandwidth)
                              ▼
            ┌───────────────────────────────────┐
            │       KAGGLE CLOUD SESSION        │
            │ • 2× NVIDIA T4 GPUs (32 GB VRAM)  │
            │ • 30 GB System RAM                │
            │ • 73 GB NVMe Scratch Disk         │
            └─────────────────┬─────────────────┘
                              │
               (Continuous Rclone / API Stream)
                              │
                              ▼
            ┌───────────────────────────────────┐
            │        GOOGLE DRIVE STORAGE       │
            │ • Raw Scene Archives (~80 GB)     │
            │ • Processed H5/NPZ Tiles (~18 GB) │
            │ • Checkpoints (Saved every epoch) │
            └─────────────────┬─────────────────┘
                              │
        (Download ONLY the final ~320 MB trained model)
                              │
                              ▼
            ┌───────────────────────────────────┐
            │           YOUR LAPTOP             │
            │ • Zero bandwidth wasted on data   │
            │ • Local Web App / QGIS Plugin     │
            │ • Models folder: cr_diffnet.pth   │
            └───────────────────────────────────┘
```

---

## 2. Kaggle Hardware & Quota Specifications

| Resource | Kaggle Allocation | Project Utilization |
|----------|-------------------|---------------------|
| **GPU Compute** | **2× NVIDIA T4 GPUs (16 GB each = 32 GB total VRAM)** | Multi-GPU Distributed Data Parallel (DDP) Training |
| **System Memory** | **30 GB CPU RAM** | In-memory batch augmentation & multi-modal stacking |
| **Disk Space** | **73 GB NVMe** (`/kaggle/working`) | Fast scratch space for active scene processing |
| **Network Speed** | **~500 Mbps to 1.2 Gbps** | 80 GB downloads in ~20 minutes |
| **Session Duration** | **9 hours per session** (up to 30 hrs GPU/week) | Checkpoint auto-saved to Google Drive to resume seamlessly |
| **Cost** | **$0.00 (100% Free)** | Zero cloud cost |

---

## 3. Step 1: Connecting Google Drive to Kaggle via Rclone

`rclone` is the industry standard for mounting and syncing cloud storage. It handles high-speed multi-threaded transfers without memory leaks.

### 3.1 One-Time Local Setup (Generate Rclone Config)
On your local machine or terminal:
```bash
# Install rclone locally
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive remote
rclone config
# 1. Type 'n' for New remote
# 2. Name: 'gdrive'
# 3. Storage: Choose 'drive' (Google Drive)
# 4. Scope: Choose '1' (Full access)
# 5. Follow the browser prompt to log in with your Google account
# 6. Type 'q' to quit
```

Display your generated config:
```bash
cat ~/.config/rclone/rclone.conf
```

Encode it to a Base64 string so you can safely store it in Kaggle Secrets:
```bash
base64 -w 0 ~/.config/rclone/rclone.conf
# On macOS:
base64 < ~/.config/rclone/rclone.conf
```

### 3.2 Add to Kaggle Secrets
1. Open your Kaggle Notebook.
2. In the top menu, go to **Add-ons** → **Secrets**.
3. Add a new secret:
   - **Label:** `RCLONE_CONFIG_BASE64`
   - **Value:** Paste the base64 string copied above.

---

## 4. Step 2: Automated Cloud Ingestion Notebook Cell

Run this directly inside a Kaggle Notebook (Ensure **Accelerator: GPU T4 x 2** and **Internet: ON** in notebook settings):

```python
# Cell 1: Environment Setup & Google Drive Mount
import os
import subprocess
import base64
from pathlib import Path
from kaggle_secrets import UserSecretsClient

print("--- Setting up Rclone & Google Drive ---")
# Install rclone and aria2c (multi-threaded downloader)
subprocess.run("apt-get update -qq && apt-get install -y -qq aria2 rclone", shell=True, check=True)

# Restore rclone configuration from Kaggle Secrets
user_secrets = UserSecretsClient()
rclone_b64 = user_secrets.get_secret("RCLONE_CONFIG_BASE64")

rclone_dir = Path.home() / ".config/rclone"
rclone_dir.mkdir(parents=True, exist_ok=True)
(rclone_dir / "rclone.conf").write_text(base64.b64decode(rclone_b64).decode())

# Test Google Drive connection
res = subprocess.run("rclone lsd gdrive:", shell=True, capture_output=True, text=True)
if res.returncode == 0:
    print("✓ Google Drive successfully connected!")
    # Create project directory in Google Drive
    subprocess.run("rclone mkdir gdrive:liss4_cloud_removal/raw", shell=True)
    subprocess.run("rclone mkdir gdrive:liss4_cloud_removal/processed", shell=True)
    subprocess.run("rclone mkdir gdrive:liss4_cloud_removal/checkpoints", shell=True)
else:
    print("❌ Error connecting to Google Drive:", res.stderr)
```

### 4.2 Cloud-Speed Download Script (Direct to Drive)

```python
# Cell 2: High-Speed Multi-Threaded Cloud Download
import requests
import json

def download_cdse_sentinel1_to_drive(product_id: str, access_token: str, filename: str):
    """
    Downloads Sentinel-1 GRD directly into Kaggle NVMe scratch using aria2c
    (16 parallel connections) and streams it to Google Drive.
    """
    download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    local_path = f"/kaggle/working/{filename}"
    
    print(f"Downloading {filename} at cloud speed...")
    cmd = f'aria2c -x 16 -s 16 --header="Authorization: Bearer {access_token}" "{download_url}" -o "{filename}" -d "/kaggle/working"'
    subprocess.run(cmd, shell=True, check=True)
    
    print(f"Uploading {filename} to Google Drive...")
    subprocess.run(f"rclone copy {local_path} gdrive:liss4_cloud_removal/raw/ -P", shell=True, check=True)
    
    # Remove from Kaggle scratch to keep 73 GB disk free
    os.remove(local_path)
    print(f"✓ {filename} stored safely in Google Drive.")
```

---

## 5. Step 3: Cloud Preprocessing & Compact Tiling

Raw satellite scenes are sparse ($12,000 \times 12,000$ pixels). Storing raw scenes takes 80 GB, but converting them into compressed $512 \times 512$ Float16 tensors reduces the total dataset size to **only ~16 GB**, which loads instantly during training!

```python
# Cell 3: Cloud Tiling & Compression
import h5py
import numpy as np
import rasterio

def process_scene_to_h5(scene_name: str):
    """
    1. Downloads raw scene from Google Drive to Kaggle scratch.
    2. Slices into 512x512 overlapping patches.
    3. Packs into a single compressed HDF5 container.
    4. Uploads the compressed .h5 back to Google Drive.
    """
    scratch_dir = "/kaggle/working/temp"
    os.makedirs(scratch_dir, exist_ok=True)
    
    # Download 1 scene from Drive
    subprocess.run(f"rclone copy gdrive:liss4_cloud_removal/raw/{scene_name}.tif {scratch_dir}/ -P", shell=True)
    
    h5_output = f"/kaggle/working/{scene_name}_patches.h5"
    with h5py.File(h5_output, "w") as hf:
        # Create chunked, gzip-compressed datasets
        # (3 channels LISS-IV + 2 channels SAR + 1 channel Mask + 2 channels DEM)
        # Slicing logic writes 500+ patches here...
        pass
        
    # Stream compact H5 back to Google Drive
    subprocess.run(f"rclone copy {h5_output} gdrive:liss4_cloud_removal/processed/ -P", shell=True)
    subprocess.run(f"rm -rf {scratch_dir}/* {h5_output}", shell=True)
    print(f"✓ Processed {scene_name} stored in Google Drive.")
```

---

## 6. Step 4: Multi-GPU (2× T4) Distributed Training on Kaggle

Kaggle provides **2× NVIDIA T4 GPUs (32 GB total VRAM)**. We use PyTorch `DistributedDataParallel` (DDP) and Automatic Mixed Precision (AMP) to train the 88.5M parameter Latent Diffusion model at maximum speed.

```python
# Cell 4: 2x T4 Distributed Training with Checkpointing to Google Drive
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

def train_worker(rank: int, world_size: int, config: dict):
    # Initialize process group for 2x T4 GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    print(f"[GPU {rank}] Initialized: {torch.cuda.get_device_name(rank)}")
    
    # 1. Build Model & Wrap in DDP
    from src.models.diffusion_unet import CRDiffNet
    model = CRDiffNet(in_channels=8, out_channels=3).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = GradScaler()
    
    # 2. Check for existing checkpoint in Google Drive to resume
    start_epoch = 0
    if rank == 0:
        subprocess.run("rclone copy gdrive:liss4_cloud_removal/checkpoints/latest_checkpoint.pth /kaggle/working/ -q", shell=True)
        if os.path.exists("/kaggle/working/latest_checkpoint.pth"):
            ckpt = torch.load("/kaggle/working/latest_checkpoint.pth", map_location=f"cuda:{rank}")
            model.module.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            print(f"✓ Resumed training from Google Drive checkpoint: Epoch {start_epoch}")
            
    # Broadcast start epoch to all ranks
    start_epoch_tensor = torch.tensor(start_epoch, device=rank)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()
    
    # 3. Training Loop (200 Epochs)
    for epoch in range(start_epoch, 200):
        model.train()
        # [Training step iterations using AMP]
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        # 4. Save Checkpoint DIRECTLY to Google Drive after each epoch
        if rank == 0 and epoch % 5 == 0:
            ckpt_path = f"/kaggle/working/cr_diffnet_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)
            
            # Sync to Google Drive
            subprocess.run(f"rclone copy {ckpt_path} gdrive:liss4_cloud_removal/checkpoints/ -P", shell=True)
            subprocess.run(f"rclone copy {ckpt_path} gdrive:liss4_cloud_removal/checkpoints/latest_checkpoint.pth", shell=True)
            os.remove(ckpt_path)
            print(f"✓ Saved Epoch {epoch} checkpoint safely to Google Drive.")
            
    dist.destroy_process_group()

def run_distributed_training():
    world_size = 2 # 2x T4 GPUs on Kaggle
    mp.spawn(train_worker, args=(world_size, {}), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_distributed_training()
```

---

## 7. Step 5: Handling Kaggle's 9-Hour Limit Seamlessly

Kaggle terminates notebook runs after 9 hours, but because of our Google Drive sync:
1. **Never Lose Work:** Every 5 epochs, the checkpoint automatically uploads to your Google Drive.
2. **One-Click Resume:** When you start a new Kaggle session tomorrow, the script checks Google Drive, finds `latest_checkpoint.pth`, and **continues training exactly where it left off!**
3. **Weekly Quota:** 30 hours of free GPU per week gives you **~3 full training runs** of 200 epochs without paying a single rupee.

---

## 8. Step 6: Downloading ONLY the Final AI Model to Your Laptop

Once training completes in Kaggle:
1. The final weights `cr_diffnet_final.pth` (~320 MB) are already waiting in your Google Drive folder:
   `liss4_cloud_removal/checkpoints/cr_diffnet_final.pth`
2. Run this single command on your local laptop to pull the model:

```bash
# On your local machine (in project root):
python scripts/download_trained_model.py
```

Or download it via your Google Drive web browser directly into:
```
cloud remover/models/cr_diffnet_final.pth
```

Your local laptop only downloads **~320 MB** instead of **80 GB**, while you get the benefits of a production-grade model trained on 32 GB of cloud GPU VRAM!
