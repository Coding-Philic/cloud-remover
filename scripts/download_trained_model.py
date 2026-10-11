"""
Download Trained Model Weights from Google Drive to Local Project
Usage:
    python scripts/download_trained_model.py --file-id <GOOGLE_DRIVE_FILE_ID>
    or
    python scripts/download_trained_model.py --rclone-path gdrive:liss4_cloud_removal/checkpoints/cr_diffnet_final.pth
"""

import os
import sys
import argparse
from pathlib import Path

def download_via_gdown(file_id: str, output_path: Path):
    try:
        import gdown
    except ImportError:
        print("Installing gdown for direct Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading model checkpoint from Google Drive ID: {file_id}")
    gdown.download(url, str(output_path), quiet=False)

def download_via_rclone(rclone_path: str, output_path: Path):
    import subprocess
    print(f"Syncing model checkpoint via rclone: {rclone_path} -> {output_path}")
    cmd = f"rclone copy {rclone_path} {output_path.parent} -P"
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError("Rclone download failed.")

def main():
    parser = argparse.ArgumentParser(description="Download trained model checkpoint from Google Drive to local project")
    parser.add_argument("--file-id", type=str, default=None, help="Google Drive shared file ID")
    parser.add_argument("--rclone-path", type=str, default="gdrive:liss4_cloud_removal/checkpoints/cr_diffnet_final.pth", help="Rclone remote path")
    parser.add_argument("--output", type=str, default="models/cr_diffnet_final.pth", help="Local destination path")
    args = parser.parse_args()

    out_file = Path(args.output).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Target local model destination: {out_file}")

    if args.file_id:
        download_via_gdown(args.file_id, out_file)
    else:
        # Check if rclone is available
        import shutil
        if shutil.which("rclone"):
            download_via_rclone(args.rclone_path, out_file)
        else:
            print("Note: rclone not detected locally. You can either:")
            print("  1. Provide --file-id <YOUR_GOOGLE_DRIVE_FILE_ID>")
            print("  2. Download cr_diffnet_final.pth manually from Google Drive into:")
            print(f"     {out_file}")
            return

    if out_file.exists() and out_file.stat().st_size > 1000:
        print(f"✓ Success! Trained model is ready at: {out_file} ({out_file.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("⚠️ File downloaded but seems empty. Please check your credentials.")

if __name__ == "__main__":
    main()
