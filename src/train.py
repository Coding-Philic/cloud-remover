"""
Distributed Multi-GPU (2x T4) Training Script for CR-DiffNet
Latent Diffusion with Multi-Modal Conditioning (SAR + DEM) on LISS-IV / Sentinel Imagery.
Auto-resumes from and syncs checkpoints to Google Drive via Rclone.
"""

import os
import sys
import argparse
import subprocess
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

from src.models.vae import SpatialCompressionVAE
from src.models.diffusion_unet import CRDiffNet
from src.models.losses import CompositeCloudRemovalLoss
from src.data.synthetic_clouds import SyntheticCloudInjector


class HDF5CloudRemovalDataset(Dataset):
    """
    Reads preprocessed Float16 tensor patches from HDF5 container.
    Injects synthetic cloud and shadow masks on-the-fly for self-supervised training.
    """
    def __init__(self, h5_path: str, is_train: bool = True):
        self.h5_path = h5_path
        self.is_train = is_train
        self.cloud_injector = SyntheticCloudInjector()

        # Open in read-only mode to get dataset length
        with h5py.File(h5_path, "r") as hf:
            self.length = hf["optical"].shape[0]

        self.hf = None  # Lazy handle per worker process

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

        opt_clean = self.hf["optical"][idx].astype(np.float32)  # (3, 512, 512)
        sar_maps = self.hf["sar"][idx].astype(np.float32)       # (2, 512, 512)
        dem_maps = self.hf["dem"][idx].astype(np.float32)       # (2, 512, 512)

        # 1. On-the-fly physics-based cloud & shadow injection
        coverage = np.random.uniform(0.15, 0.70) if self.is_train else 0.40
        cloudy_opt, mask = self.cloud_injector.apply_clouds(
            opt_clean,
            coverage=coverage,
            sun_azimuth_deg=np.random.uniform(115.0, 160.0),
            sun_elevation_deg=np.random.uniform(35.0, 65.0)
        )

        # 2. Random data augmentations (flips & 90-deg rotations)
        if self.is_train:
            if np.random.rand() > 0.5:
                opt_clean = np.flip(opt_clean, axis=2).copy()
                cloudy_opt = np.flip(cloudy_opt, axis=2).copy()
                sar_maps = np.flip(sar_maps, axis=2).copy()
                dem_maps = np.flip(dem_maps, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()

            if np.random.rand() > 0.5:
                opt_clean = np.flip(opt_clean, axis=1).copy()
                cloudy_opt = np.flip(cloudy_opt, axis=1).copy()
                sar_maps = np.flip(sar_maps, axis=1).copy()
                dem_maps = np.flip(dem_maps, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        # Auxiliary map stack: [SAR_VV, SAR_VH, DEM_Elev, DEM_Slope] -> (4, 512, 512)
        aux_stack = np.concatenate([sar_maps, dem_maps], axis=0)

        return {
            "clean_target": torch.from_numpy(opt_clean),
            "cloudy_optical": torch.from_numpy(cloudy_opt),
            "mask": torch.from_numpy(mask),
            "aux_maps": torch.from_numpy(aux_stack)
        }


def setup_ddp():
    """Initializes Distributed Data Parallel process group."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        # Single GPU mode
        local_rank = 0 if torch.cuda.is_available() else -1
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)
        return 0, 1, local_rank, False


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train():
    parser = argparse.ArgumentParser(description="Train CR-DiffNet on Kaggle / Multi-GPU")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to liss4_training_dataset.h5")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="/kaggle/working/checkpoints", help="Local checkpoint dir")
    parser.add_argument("--gdrive_sync", action="store_true", default=True, help="Sync checkpoints to Google Drive")
    args = parser.parse_args()

    rank, world_size, local_rank, is_distributed = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if local_rank >= 0 else "cpu")
    is_main = (rank == 0)

    if is_main:
        print("=" * 70)
        print("🚀 STARTING CR-DIFFNET MULTI-GPU DISTRIBUTED TRAINING")
        print(f"GPUs active: {world_size} | Local Rank: {local_rank} | Device: {torch.cuda.get_device_name(device)}")
        print(f"Dataset path: {args.h5_path}")
        print("=" * 70)
        os.makedirs(args.ckpt_dir, exist_ok=True)

    # 1. Dataset & Distributed DataLoader
    dataset = HDF5CloudRemovalDataset(args.h5_path, is_train=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    if is_main:
        print(f"✓ Total Training Patches: {len(dataset):,} | Batches per Epoch: {len(loader)}")

    # 2. Instantiate Models
    # Frozen Spatial VAE (Compresses 512x512 down to 64x64 latent space)
    vae = SpatialCompressionVAE(in_channels=3, latent_channels=4).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Denoising U-Net with Cross-Attention Multi-Modal Conditioning
    model = CRDiffNet(aux_in_ch=4, context_dim=512).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. Optimizer, Scheduler & Scaler
    raw_model = model.module if is_distributed else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    criterion = CompositeCloudRemovalLoss()

    # 4. Resume from Google Drive checkpoint if available
    start_epoch = 0
    latest_ckpt_path = os.path.join(args.ckpt_dir, "latest_checkpoint.pth")
    is_direct_drive = "drive/MyDrive" in str(args.ckpt_dir)
    import shutil
    has_rclone = shutil.which("rclone") is not None

    if is_main and args.gdrive_sync and (not is_direct_drive) and has_rclone:
        subprocess.run(f"rclone copy gdrive:liss4_cloud_removal/checkpoints/latest_checkpoint.pth {args.ckpt_dir}/ -q", shell=True)

    if is_distributed:
        dist.barrier()

    if os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"✓ Resumed seamlessly from Google Drive checkpoint: Epoch {start_epoch}")

    # Broadcast start epoch
    if is_distributed:
        epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(epoch_tensor, src=0)
        start_epoch = epoch_tensor.item()

    # Noise Scheduler parameters (Linear beta schedule 1000 steps)
    timesteps_total = 1000
    betas = torch.linspace(1e-4, 0.02, timesteps_total, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        raw_model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", disable=not is_main)

        for batch in pbar:
            clean_tgt = batch["clean_target"].to(device, non_blocking=True)       # (B, 3, 512, 512)
            cloudy_opt = batch["cloudy_optical"].to(device, non_blocking=True)     # (B, 3, 512, 512)
            aux_maps = batch["aux_maps"].to(device, non_blocking=True)             # (B, 4, 512, 512)
            mask = batch["mask"].to(device, non_blocking=True)                     # (B, 1, 512, 512)
            B = clean_tgt.shape[0]

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Encode target and condition optical into latent space (64x64)
                with torch.no_grad():
                    _, z_target = vae(clean_tgt)
                    _, z_cond = vae(cloudy_opt)

                # Sample random diffusion timesteps
                t = torch.randint(0, timesteps_total, (B,), device=device).long()
                noise = torch.randn_like(z_target)

                # Forward diffusion step (add noise to target latent)
                a_cp = alphas_cumprod[t].view(B, 1, 1, 1)
                noisy_latents = torch.sqrt(a_cp) * z_target + torch.sqrt(1.0 - a_cp) * noise

                # Predict noise using CRDiffNet
                noise_pred = model(noisy_latents, z_cond, t, aux_maps)

                # Composite loss
                loss, loss_dict = criterion(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if is_main:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "noise_mse": f"{loss_dict['loss_noise']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if is_main:
            print(f"📊 Epoch {epoch+1} Completed | Mean Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save and sync checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                ckpt_save_path = os.path.join(args.ckpt_dir, f"cr_diffnet_epoch_{epoch+1}.pth")
                state = {
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss
                }
                torch.save(state, ckpt_save_path)
                torch.save(state, latest_ckpt_path)
                print(f"💾 Checkpoint saved locally: {ckpt_save_path}")

                if is_direct_drive:
                    print(f"☁️ Checkpoint saved directly to Google Drive: {ckpt_save_path}")
                elif args.gdrive_sync and has_rclone:
                    subprocess.run(f"rclone copy {ckpt_save_path} gdrive:liss4_cloud_removal/checkpoints/ -q", shell=True)
                    subprocess.run(f"rclone copy {latest_ckpt_path} gdrive:liss4_cloud_removal/checkpoints/ -q", shell=True)
                    print(f"☁️ Checkpoint synced safely to Google Drive: cr_diffnet_epoch_{epoch+1}.pth")

    # Final Export
    if is_main:
        final_model_path = os.path.join(args.ckpt_dir, "cr_diffnet_final.pth")
        torch.save(raw_model.state_dict(), final_model_path)
        if is_direct_drive:
            print(f"☁️ Final model saved directly to Google Drive: {final_model_path}")
        elif args.gdrive_sync and has_rclone:
            subprocess.run(f"rclone copy {final_model_path} gdrive:liss4_cloud_removal/checkpoints/ -q", shell=True)
        print("\n🎉 TRAINING FULLY CONVERGED! Final model exported to Google Drive: cr_diffnet_final.pth (~320 MB)")

    cleanup_ddp()


if __name__ == "__main__":
    train()
