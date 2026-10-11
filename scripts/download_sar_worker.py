import os
import sys
import json
import urllib.request
import subprocess
import requests
from tqdm import tqdm

BATCH_LIMIT = 2

# 1. Fetch files ALREADY in Google Drive
try:
    existing_s1 = set(subprocess.check_output("rclone lsf gdrive:liss4_cloud_removal/raw/s1/", shell=True).decode().splitlines())
except Exception:
    existing_s1 = set()

# 2. Query Sentinel-1 SAR scenes over North East India
stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
s1_payload = {
    "collections": ["sentinel-1-grd"],
    "bbox": [89.0, 24.5, 94.5, 28.5],
    "datetime": "2024-01-01T00:00:00Z/2024-08-31T23:59:59Z",
    "limit": 40
}
req = urllib.request.Request(stac_url, data=json.dumps(s1_payload).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=30) as resp:
    s1_features = json.loads(resp.read().decode()).get("features", [])

all_bands = []
for feat in s1_features:
    for pol in ["vv", "vh"]:
        if pol in feat["assets"]:
            all_bands.append((feat["id"], pol, feat["assets"][pol]["href"]))

total_bands = len(all_bands)
already_in_drive = len([b for b in all_bands if f"{b[0]}_{b[1].upper()}.tif" in existing_s1])

print(f"📊 Overall Progress: {already_in_drive}/{total_bands} bands in Drive ({(already_in_drive/total_bands)*100:.1f}%)")

if already_in_drive >= total_bands:
    sys.exit(99)

# 3. Fetch container SAS token (0 rate limit)
token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"
with urllib.request.urlopen(token_url, timeout=30) as r:
    sas_token = json.loads(r.read().decode())["token"]

session = requests.Session()
downloaded_in_batch = 0

for scene_id, pol, raw_url in all_bands:
    filename = f"{scene_id}_{pol.upper()}.tif"
    if filename in existing_s1:
        continue

    signed_url = f"{raw_url}?{sas_token}"
    drive_dest = f"gdrive:liss4_cloud_removal/raw/s1/{filename}"
    display_label = f"SAR {pol.upper()} {scene_id[:16]}"

    try:
        resp = session.get(signed_url, stream=True, timeout=(10, 60))
        total_size = int(resp.headers.get("content-length", 0))
        chunk_size = 4 * 1024 * 1024

        rcat_cmd = ["rclone", "rcat", drive_dest, "--drive-chunk-size", "128M", "--size", str(total_size)]
        proc = subprocess.Popen(rcat_cmd, stdin=subprocess.PIPE)

        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=f"📥 {display_label}", leave=True) as file_bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    proc.stdin.write(chunk)
                    file_bar.update(len(chunk))

        proc.stdin.close()
        proc.wait()
        resp.close()

        if proc.returncode == 0:
            existing_s1.add(filename)
            downloaded_in_batch += 1
            print(f"  ✓ [{already_in_drive + downloaded_in_batch}/{total_bands}] Saved in Google Drive: {filename}")

        if downloaded_in_batch >= BATCH_LIMIT:
            print(f"⚡ Batch of {BATCH_LIMIT} completed cleanly! Auto-restarting fresh process...")
            sys.exit(0)

    except Exception as e:
        print(f"⚠️ Error on {filename}: {e}. Retrying next...")
        continue

sys.exit(99)
