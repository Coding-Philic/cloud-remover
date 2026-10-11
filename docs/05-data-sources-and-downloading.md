# 05 — Data Sources and Downloading Guide
## Generative AI-Based Cloud Removal for LISS-IV Imagery

---

## 1. Data Sources Overview

| # | Dataset | Provider | Resolution | Bands/Channels | Coverage | Cost | Role in Project |
|---|---------|----------|-----------|----------------|----------|------|-----------------|
| 1 | **LISS-IV** | ISRO (Bhoonidhi) | 5.8 m | 3 (G, R, NIR) | India | Free for Indian institutions | **Primary** — target imagery for cloud removal |
| 2 | **Sentinel-1 SAR** | ESA (Copernicus) | 10 m | 2 (VV, VH) | Global | Free | **Auxiliary** — cloud-penetrating radar structure |
| 3 | **Sentinel-2 L2A** | ESA (Copernicus) | 10 m | 13 (use 4: B/G/R/NIR) | Global | Free | **Auxiliary** — spectral reference |
| 4 | **SRTM DEM** | NASA / USGS | 30 m | 1 (elevation) | 60°N–56°S | Free | **Auxiliary** — terrain information |
| 5 | **Copernicus DEM (COP-30)** | ESA / Airbus | 30 m | 1 (elevation) | Global | Free | **Auxiliary** — alternative DEM |
| 6 | **Temporal Reference** | Same as Primary | Same | Same | Same area, different date | Same | **Auxiliary** — past cloud-free observation |

---

## 2. LISS-IV Data (Primary Dataset)

### 2.1 About LISS-IV

- **Satellite:** Resourcesat-2 (launched 2011) and Resourcesat-2A (launched 2016)
- **Sensor:** Linear Imaging Self-Scanning Sensor – IV
- **Spectral Bands:**
  - Band 2 (Green): 0.52 – 0.59 µm
  - Band 3 (Red): 0.62 – 0.68 µm
  - Band 4 (NIR): 0.77 – 0.86 µm
- **Spatial Resolution:** 5.8 m
- **Swath Width:** 70 km (mono mode), 23.9 km (stereo mode)
- **Radiometric Resolution:** 10 bits
- **Revisit:** 5 days (with Resourcesat-2 + 2A combined)
- **Data Level:** Level-2 (radiometrically and geometrically corrected)

### 2.2 How to Access — Bhoonidhi Portal

Bhoonidhi (https://bhoonidhi.nrsc.gov.in) is ISRO's unified geospatial data portal.

#### Step 1: Registration
```
1. Go to https://bhoonidhi.nrsc.gov.in
2. Click "Register" / "Sign Up"
3. Fill in details:
   - Name, Email, Institution
   - Indian institutional affiliation required
   - Purpose: "Research / Academic"
4. Wait for approval (usually 1–3 business days)
5. Receive login credentials via email
```

#### Step 2: Search for LISS-IV Data
```
1. Login to Bhoonidhi
2. Navigate to "Data Search" / "Browse Catalog"
3. Set search parameters:
   - Satellite: Resourcesat-2 / Resourcesat-2A
   - Sensor: LISS-IV
   - Date Range: e.g., 2022-01-01 to 2024-12-31
   - Area of Interest: Draw polygon over NER
     (approx bounds: 88°E–97°E, 21°N–29°N)
   - Cloud Cover: 0–100% (we need both cloudy AND clear scenes)
   - Product Level: Level-2 (Ortho)
4. Review search results
5. Add desired scenes to cart
```

#### Step 3: Order and Download
```
1. Place order for selected scenes
2. Order is processed by NRSC (may take hours to days)
3. Receive download notification email
4. Download ZIP files from the provided links
5. Each ZIP contains:
   - BAND2.tif (Green)
   - BAND3.tif (Red)
   - BAND4.tif (NIR)
   - metadata.xml
   - browse image (JPEG)
```

#### Step 4: Organize Downloaded Data
```bash
# Create directory structure
mkdir -p data/raw/liss4

# Unzip downloaded scenes
cd data/raw/liss4
for zip_file in ~/Downloads/LISS4_*.zip; do
    scene_name=$(basename "$zip_file" .zip)
    mkdir -p "$scene_name"
    unzip "$zip_file" -d "$scene_name"
done

# Verify file structure
find data/raw/liss4 -name "*.tif" | head -20
```

#### Alternative: Using Bhuvan API (Limited)
```python
"""
Note: Bhuvan/Bhoonidhi has limited API access for LISS-IV.
This script demonstrates what's possible with the web services.
For bulk LISS-IV, manual ordering via the portal is typically required.
"""

import requests

# Bhuvan WMS endpoint (for preview/visualization only, not full-res download)
bhuvan_wms = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"

params = {
    "service": "WMS",
    "request": "GetMap",
    "layers": "resourcesat2:liss4",
    "bbox": "92.0,26.0,93.0,27.0",  # NER subset
    "width": 1024,
    "height": 1024,
    "srs": "EPSG:4326",
    "format": "image/geotiff",
}

response = requests.get(bhuvan_wms, params=params)
with open("preview.tif", "wb") as f:
    f.write(response.content)
```

### 2.3 Recommended Collection Strategy

For this project, collect:

| Category | Min Scenes | Cloud Cover | Purpose |
|----------|-----------|-------------|---------|
| Cloud-free | 50–100 | < 10% | Ground truth for training |
| Partially cloudy | 50–100 | 20–60% | Real cloudy samples for testing |
| Heavily cloudy | 20–30 | 60–90% | Stress-test evaluation |
| Temporal pairs | 30–50 pairs | Mixed | Same area, one clear + one cloudy |

**Focus Area:** NER states — Assam, Meghalaya, Nagaland, Manipur, Mizoram, Tripura, Arunachal Pradesh, Sikkim.

---

## 3. Sentinel-1 SAR Data (Auxiliary)

### 3.1 About Sentinel-1

- **Satellite:** Sentinel-1A (2014), Sentinel-1C (2024, replacement for 1B)
- **Sensor:** C-band SAR (5.405 GHz)
- **Mode:** Interferometric Wide Swath (IW)
- **Product:** Ground Range Detected (GRD)
- **Polarization:** Dual (VV + VH)
- **Resolution:** 10 m × 10 m (IW GRD)
- **Swath:** 250 km
- **Revisit:** 6 days (per satellite), 3 days (constellation)
- **Key Advantage:** Sees through clouds — provides structural/textural information regardless of weather

### 3.2 Download via Copernicus Data Space Ecosystem (CDSE)

#### Method 1: `cdsetool` (Recommended — Command Line)

```bash
# Install cdsetool
pip install cdsetool

# Register at https://dataspace.copernicus.eu (free)

# Search for Sentinel-1 GRD data over NER
cdsetool query \
  --collection SENTINEL-1 \
  --start-date 2024-06-01 \
  --end-date 2024-06-30 \
  --bbox 92.0,26.0,93.0,27.0 \
  --product-type GRD \
  --sensor-mode IW \
  --output results_s1.json

# Download matching products
cdsetool download \
  --input results_s1.json \
  --output data/raw/sentinel1/ \
  --username YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --concurrency 4
```

#### Method 2: `sentinelsat` (Python API)

```python
from sentinelsat import SentinelAPI

# Connect to Copernicus Data Space
api = SentinelAPI(
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    api_url="https://catalogue.dataspace.copernicus.eu/odata/v1",
)

# Search for Sentinel-1 GRD over NER
from shapely.geometry import box
ner_aoi = box(92.0, 26.0, 93.0, 27.0)  # minx, miny, maxx, maxy

products = api.query(
    area=ner_aoi.wkt,
    date=("20240601", "20240630"),
    platformname="Sentinel-1",
    producttype="GRD",
    sensoroperationalmode="IW",
)

print(f"Found {len(products)} Sentinel-1 products")

# Download all results
api.download_all(products, directory_path="data/raw/sentinel1/")
```

#### Method 3: Google Earth Engine (Python API)

```python
import ee

ee.Initialize(project="your-gee-project")

# Define NER area of interest
ner_aoi = ee.Geometry.Rectangle([92.0, 26.0, 93.0, 27.0])

# Get Sentinel-1 GRD collection
s1_collection = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(ner_aoi)
    .filterDate("2024-06-01", "2024-06-30")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .select(["VV", "VH"])
)

print(f"Found {s1_collection.size().getInfo()} images")

# Export to Google Drive
for i, image_id in enumerate(s1_collection.aggregate_array("system:id").getInfo()):
    image = ee.Image(image_id).clip(ner_aoi)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"S1_GRD_{i:03d}",
        folder="sentinel1_data",
        region=ner_aoi,
        scale=10,
        crs="EPSG:32646",
        maxPixels=1e10,
    )
    task.start()
    print(f"Started export task {i}")
```

### 3.3 SAR Preprocessing Commands

```bash
# Using ESA SNAP GPT (Graph Processing Tool)
# Install SNAP: https://step.esa.int/main/download/snap-download/

# Full preprocessing pipeline for Sentinel-1 GRD:
# 1. Apply orbit file
# 2. Remove thermal noise
# 3. Calibrate to sigma0
# 4. Terrain correction (Range-Doppler)
# 5. Convert to dB

# Create SNAP processing graph (save as s1_preprocess.xml)
cat > s1_preprocess.xml << 'EOF'
<graph id="S1_Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>${input}</file></parameters>
  </node>
  <node id="ApplyOrbitFile">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="ApplyOrbitFile"/></sources>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters><outputSigmaBand>true</outputSigmaBand></parameters>
  </node>
  <node id="SpeckleFilter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <filter>Lee Sigma</filter>
      <filterSizeX>5</filterSizeX>
      <filterSizeY>5</filterSizeY>
    </parameters>
  </node>
  <node id="TerrainCorrection">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="SpeckleFilter"/></sources>
    <parameters>
      <demName>SRTM 3Sec</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>EPSG:32646</mapProjection>
    </parameters>
  </node>
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources><sourceProduct refid="TerrainCorrection"/></sources>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="LinearToFromdB"/></sources>
    <parameters>
      <file>${output}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>
EOF

# Run SNAP GPT
gpt s1_preprocess.xml \
  -Pinput=data/raw/sentinel1/S1A_IW_GRDH_*.zip \
  -Poutput=data/processed/sentinel1/s1_preprocessed.tif
```

```python
# Alternative: Python-based SAR preprocessing (using rasterio + scipy)
import rasterio
import numpy as np
from scipy.ndimage import uniform_filter

def preprocess_sar(input_path, output_path):
    """Basic SAR preprocessing in Python."""
    with rasterio.open(input_path) as src:
        vv = src.read(1).astype(np.float32)  # VV band
        vh = src.read(2).astype(np.float32)  # VH band
        profile = src.profile.copy()

    # Convert linear to dB (if not already)
    vv_db = 10 * np.log10(np.clip(vv, 1e-10, None))
    vh_db = 10 * np.log10(np.clip(vh, 1e-10, None))

    # Lee speckle filter (simplified)
    vv_filtered = lee_filter(vv_db, size=5)
    vh_filtered = lee_filter(vh_db, size=5)

    # Normalize to [-1, 1] range
    # Typical SAR dB range: -30 to +5
    vv_norm = np.clip((vv_filtered + 30) / 35, 0, 1) * 2 - 1
    vh_norm = np.clip((vh_filtered + 30) / 35, 0, 1) * 2 - 1

    # Write output
    profile.update(count=2, dtype='float32')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(vv_norm, 1)
        dst.write(vh_norm, 2)

def lee_filter(image, size=5):
    """Lee speckle filter."""
    mean = uniform_filter(image, size)
    sqr_mean = uniform_filter(image**2, size)
    variance = sqr_mean - mean**2
    overall_var = np.var(image)
    weight = np.clip(variance / (variance + overall_var + 1e-10), 0, 1)
    return mean + weight * (image - mean)
```

---

## 4. Sentinel-2 L2A Data (Auxiliary)

### 4.1 About Sentinel-2

- **Satellites:** Sentinel-2A (2015), Sentinel-2B (2017)
- **Bands of interest:** B2 (Blue, 10m), B3 (Green, 10m), B4 (Red, 10m), B8 (NIR, 10m)
- **Product Level:** L2A (atmospherically corrected, Surface Reflectance)
- **Revisit:** 5 days (2 satellites)

### 4.2 Download Commands

#### Using `cdsetool`

```bash
# Search for Sentinel-2 L2A data
cdsetool query \
  --collection SENTINEL-2 \
  --start-date 2024-06-01 \
  --end-date 2024-06-30 \
  --bbox 92.0,26.0,93.0,27.0 \
  --product-type S2MSI2A \
  --cloud-cover 0,50 \
  --output results_s2.json

# Download
cdsetool download \
  --input results_s2.json \
  --output data/raw/sentinel2/ \
  --username YOUR_USERNAME \
  --password YOUR_PASSWORD \
  --concurrency 4
```

#### Using Google Earth Engine

```python
import ee

ee.Initialize(project="your-gee-project")

ner_aoi = ee.Geometry.Rectangle([92.0, 26.0, 93.0, 27.0])

# Get Sentinel-2 L2A Surface Reflectance
s2_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(ner_aoi)
    .filterDate("2024-06-01", "2024-06-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
    .select(["B2", "B3", "B4", "B8"])  # Blue, Green, Red, NIR
)

print(f"Found {s2_collection.size().getInfo()} images")

# Export each image
for i, img_info in enumerate(s2_collection.getInfo()["features"]):
    image = ee.Image(img_info["id"]).clip(ner_aoi)
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"S2_L2A_{i:03d}",
        folder="sentinel2_data",
        region=ner_aoi,
        scale=10,
        crs="EPSG:32646",
        maxPixels=1e10,
    )
    task.start()
```

#### Using `sentinelsat`

```python
from sentinelsat import SentinelAPI
from shapely.geometry import box

api = SentinelAPI("YOUR_USERNAME", "YOUR_PASSWORD",
                  api_url="https://catalogue.dataspace.copernicus.eu/odata/v1")

ner_aoi = box(92.0, 26.0, 93.0, 27.0)

products = api.query(
    area=ner_aoi.wkt,
    date=("20240601", "20240630"),
    platformname="Sentinel-2",
    producttype="S2MSI2A",
    cloudcoverpercentage=(0, 50),
)

api.download_all(products, directory_path="data/raw/sentinel2/")
```

### 4.3 Sentinel-2 Preprocessing

```python
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def preprocess_sentinel2(s2_dir, output_path, target_crs, target_transform, target_shape):
    """
    Extract 10m bands from Sentinel-2 L2A and resample to LISS-IV grid.

    Args:
        s2_dir: Path to Sentinel-2 SAFE directory
        output_path: Output GeoTIFF path
        target_crs: LISS-IV CRS (e.g., EPSG:32646)
        target_transform: LISS-IV affine transform
        target_shape: (height, width) of LISS-IV scene
    """
    # Find 10m band files
    band_files = {
        "B02": find_band_file(s2_dir, "B02_10m"),  # Blue
        "B03": find_band_file(s2_dir, "B03_10m"),  # Green
        "B04": find_band_file(s2_dir, "B04_10m"),  # Red
        "B08": find_band_file(s2_dir, "B08_10m"),  # NIR
    }

    bands = []
    for band_name, band_path in band_files.items():
        with rasterio.open(band_path) as src:
            # Read and reproject to LISS-IV grid
            dest = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.cubic,
            )
            # Normalize: S2 L2A values are in [0, 10000] → [0, 1]
            dest = dest / 10000.0
            bands.append(dest)

    # Stack and write
    stack = np.stack(bands, axis=0)  # (4, H, W)
    profile = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "float32",
        "crs": target_crs,
        "transform": target_transform,
        "height": target_shape[0],
        "width": target_shape[1],
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)
```

---

## 5. DEM Data (Auxiliary)

### 5.1 Option A: SRTM 30m (NASA/USGS)

```bash
# Method 1: Download from USGS EarthExplorer
# 1. Register at https://earthexplorer.usgs.gov
# 2. Search for "SRTM 1 Arc-Second Global" (30m)
# 3. Select tiles covering NER: N26E092, N26E093, N27E092, etc.
# 4. Download GeoTIFF files

# Method 2: Using elevation package (Python)
pip install elevation

# Download SRTM 30m DEM for NER
eio clip -o data/raw/dem/srtm_ner.tif \
  --bounds 92.0 26.0 93.0 27.0
```

```python
# Method 3: Using Google Earth Engine
import ee

ee.Initialize(project="your-gee-project")

ner_aoi = ee.Geometry.Rectangle([92.0, 26.0, 93.0, 27.0])

# SRTM 30m DEM
srtm = ee.Image("USGS/SRTMGL1_003").clip(ner_aoi)

task = ee.batch.Export.image.toDrive(
    image=srtm.select("elevation"),
    description="SRTM_DEM_NER",
    folder="dem_data",
    region=ner_aoi,
    scale=30,
    crs="EPSG:32646",
    maxPixels=1e10,
)
task.start()
```

### 5.2 Option B: Copernicus DEM (COP-30)

```python
# Using OpenTopography API
import requests

url = "https://portal.opentopography.org/API/globaldem"
params = {
    "demtype": "COP30",
    "south": 26.0,
    "north": 27.0,
    "west": 92.0,
    "east": 93.0,
    "outputFormat": "GTiff",
    "API_Key": "YOUR_API_KEY",  # Register at opentopography.org
}

response = requests.get(url, params=params)
with open("data/raw/dem/cop30_ner.tif", "wb") as f:
    f.write(response.content)
```

### 5.3 DEM Preprocessing

```python
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

def preprocess_dem(dem_path, output_path, target_crs, target_transform, target_shape):
    """
    Reproject DEM to LISS-IV grid and normalize.
    """
    with rasterio.open(dem_path) as src:
        dest = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.cubic,
        )

    # Normalize elevation to [0, 1]
    # NER elevation range: ~0 m (plains) to ~3,500 m (mountains)
    dem_normalized = np.clip(dest / 3500.0, 0.0, 1.0)

    # Replace nodata (-32768 typically) with 0
    dem_normalized[dest < -100] = 0.0

    profile = {
        "driver": "GTiff", "count": 1, "dtype": "float32",
        "crs": target_crs, "transform": target_transform,
        "height": target_shape[0], "width": target_shape[1],
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dem_normalized, 1)
```

---

## 6. Complete Download Script

```python
#!/usr/bin/env python3
"""
download_all_data.py — Master download script for cloud removal project.

Usage:
    python scripts/download_data.py --aoi 92.0,26.0,93.0,27.0 \
        --start 2024-06-01 --end 2024-06-30 \
        --output data/raw/
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

def download_sentinel1(aoi_bbox, start_date, end_date, output_dir, username, password):
    """Download Sentinel-1 GRD IW data."""
    from sentinelsat import SentinelAPI
    from shapely.geometry import box

    api = SentinelAPI(username, password,
                      api_url="https://catalogue.dataspace.copernicus.eu/odata/v1")

    aoi = box(*aoi_bbox)
    products = api.query(
        area=aoi.wkt,
        date=(start_date, end_date),
        platformname="Sentinel-1",
        producttype="GRD",
        sensoroperationalmode="IW",
    )

    print(f"[S1] Found {len(products)} products")
    os.makedirs(output_dir, exist_ok=True)
    api.download_all(products, directory_path=output_dir)
    return list(products.keys())


def download_sentinel2(aoi_bbox, start_date, end_date, output_dir, username, password,
                        max_cloud=50):
    """Download Sentinel-2 L2A data."""
    from sentinelsat import SentinelAPI
    from shapely.geometry import box

    api = SentinelAPI(username, password,
                      api_url="https://catalogue.dataspace.copernicus.eu/odata/v1")

    aoi = box(*aoi_bbox)
    products = api.query(
        area=aoi.wkt,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        producttype="S2MSI2A",
        cloudcoverpercentage=(0, max_cloud),
    )

    print(f"[S2] Found {len(products)} products (cloud < {max_cloud}%)")
    os.makedirs(output_dir, exist_ok=True)
    api.download_all(products, directory_path=output_dir)
    return list(products.keys())


def download_srtm_dem(aoi_bbox, output_path):
    """Download SRTM DEM via the elevation package."""
    import subprocess
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bounds = f"{aoi_bbox[0]} {aoi_bbox[1]} {aoi_bbox[2]} {aoi_bbox[3]}"
    subprocess.run(
        f"eio clip -o {output_path} --bounds {bounds}",
        shell=True, check=True
    )
    print(f"[DEM] Downloaded to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download all data for cloud removal project")
    parser.add_argument("--aoi", type=str, required=True,
                        help="AOI bounding box: minx,miny,maxx,maxy")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/raw/", help="Output root directory")
    parser.add_argument("--username", type=str, help="Copernicus username")
    parser.add_argument("--password", type=str, help="Copernicus password")
    parser.add_argument("--skip-s1", action="store_true", help="Skip Sentinel-1 download")
    parser.add_argument("--skip-s2", action="store_true", help="Skip Sentinel-2 download")
    parser.add_argument("--skip-dem", action="store_true", help="Skip DEM download")
    args = parser.parse_args()

    bbox = [float(x) for x in args.aoi.split(",")]
    output = Path(args.output)

    print(f"=== Cloud Removal Data Downloader ===")
    print(f"AOI: {bbox}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Output: {output}")
    print()

    # Sentinel-1
    if not args.skip_s1 and args.username:
        print("[1/3] Downloading Sentinel-1 SAR...")
        download_sentinel1(bbox, args.start, args.end,
                          str(output / "sentinel1"), args.username, args.password)

    # Sentinel-2
    if not args.skip_s2 and args.username:
        print("[2/3] Downloading Sentinel-2 L2A...")
        download_sentinel2(bbox, args.start, args.end,
                          str(output / "sentinel2"), args.username, args.password)

    # DEM
    if not args.skip_dem:
        print("[3/3] Downloading SRTM DEM...")
        download_srtm_dem(bbox, str(output / "dem" / "srtm_ner.tif"))

    print("\n=== Download Complete ===")
    print("Note: LISS-IV data must be downloaded manually from Bhoonidhi portal.")
    print("  URL: https://bhoonidhi.nrsc.gov.in")


if __name__ == "__main__":
    main()
```

---

## 7. Data Requirements Summary

### For Training

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Cloud-free LISS-IV tiles (512×512) | 5,000 | 20,000+ |
| Cloudy LISS-IV tiles (for testing) | 500 | 2,000+ |
| Sentinel-1 SAR matched tiles | 3,000 | 15,000+ |
| Sentinel-2 matched tiles | 2,000 | 10,000+ |
| DEM tiles | Covers study area | Same |
| Temporal reference tiles | 2,000 | 10,000+ |

### Storage Estimates

| Data | Per Scene (70km × 70km) | 100 Scenes | Format |
|------|------------------------|------------|--------|
| LISS-IV raw | ~2 GB | ~200 GB | GeoTIFF UInt16 |
| Sentinel-1 raw | ~1 GB | ~100 GB | GeoTIFF |
| Sentinel-2 raw | ~800 MB | ~80 GB | JP2 |
| DEM | ~50 MB/tile | ~500 MB | GeoTIFF |
| Processed tiles (all) | — | ~150 GB | NPZ Float16 |
| **Total** | — | **~530 GB** | — |

---

## 8. Environment Setup

```bash
# Create conda environment
conda create -n cloud-removal python=3.10 -y
conda activate cloud-removal

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.1.0

# Install geospatial tools
pip install rasterio==1.3.9 gdal==3.8.0 pyproj geopandas shapely
pip install sentinelsat cdsetool elevation

# Install CV and augmentation
pip install opencv-python scikit-image albumentations

# Install metrics and losses
pip install torchmetrics lpips pytorch-msssim

# Install ML tools
pip install wandb tensorboard

# Install diffusion model utilities
pip install diffusers transformers accelerate xformers

# Install API framework
pip install fastapi uvicorn python-multipart

# Install utility
pip install numpy pandas matplotlib tqdm pyyaml

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 9. Zero-Bandwidth Cloud Ingestion: Direct Kaggle to Google Drive

To download ~80+ GB of raw satellite scenes without consuming local laptop storage or home internet bandwidth, we execute all harvesting inside **Kaggle's Cloud Infrastructure** connected directly to **Google Drive**.

### 9.1 Why Kaggle + Google Drive?
1. **Cloud Bandwidth:** Kaggle download speeds reach **500 Mbps – 1 Gbps** directly from ESA Copernicus CDSE and AWS servers (downloading 80 GB takes under 25 minutes).
2. **Zero Laptop Storage:** Raw data streams directly to Google Drive via `rclone` or Google Drive API.
3. **Free Cloud Compute:** Kaggle provides **2× NVIDIA T4 GPUs (32 GB total VRAM)** and 30 GB CPU RAM for 30 hours per week for free.
4. **Final Local Footprint:** Only the final trained model weights (`cr_diffnet_final.pth`, ~300 MB) are downloaded to your laptop.

### 9.2 Kaggle Cloud Download Script (`kaggle_download_to_gdrive.py`)

```python
# Run this inside a Kaggle Notebook (with Internet ON)
import os
import subprocess
from pathlib import Path

# 1. Install rclone in Kaggle environment
subprocess.run("curl https://rclone.org/install.sh | sudo bash", shell=True, check=True)

# 2. Configure Rclone to Google Drive (using your rclone.conf stored in Kaggle Secrets)
# Secret name: RCLONE_CONFIG_BASE64
rclone_conf = os.environ.get("RCLONE_CONFIG_BASE64")
if rclone_conf:
    import base64
    config_path = Path.home() / ".config/rclone/rclone.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(base64.b64decode(rclone_conf).decode())
    print("✓ Google Drive mounted via Rclone")

# 3. Download Sentinel-1 SAR & DEM directly to Kaggle NVMe scratch
print("Downloading auxiliary datasets at cloud speed...")
# Example: CDSE curl / aria2c multi-connection download (16 parallel threads)
# aria2c -x 16 -s 16 https://zipper.dataspace.copernicus.eu/odata/v1/Products(...) -o scene.zip

# 4. Stream processed files directly to Google Drive
# rclone copy /kaggle/working/data gdrive:liss4_project/data --transfers 8 -P
```

> **Complete Step-by-Step Guide:** For the full setup, rclone configuration, and automated scripts, see [09-kaggle-gdrive-cloud-training-guide.md](./09-kaggle-gdrive-cloud-training-guide.md).

---

> **Next →** Read [06-model-training-and-algorithms.md](./06-model-training-and-algorithms.md) for algorithm details, or [09-kaggle-gdrive-cloud-training-guide.md](./09-kaggle-gdrive-cloud-training-guide.md) for the complete cloud pipeline.
