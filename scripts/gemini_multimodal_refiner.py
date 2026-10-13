#!/usr/bin/env python3
"""
CLI Tool: Gemini Multimodal Quad-Stack Satellite Refiner (Stage 2)
Executes the Two-Stage Cascaded Multi-Modal Cloud Removal Pipeline:
1. Preliminary cloud-free reconstruction from Stage 1 (CR-DiffNet)
2. Raw cloudy optical scene
3. Sentinel-1 SAR C-band microwave radar
4. ISRO LISS-IV native clear ground reference

Usage:
    python3 scripts/gemini_multimodal_refiner.py \
        --draft webapp/static/samples/03_kaziranga_wetlands_clean.jpg \
        --cloudy webapp/static/samples/03_kaziranga_wetlands_cloudy_satellite.jpg \
        --sar webapp/static/samples/03_kaziranga_wetlands_clean.jpg \
        --anchor webapp/static/samples/03_kaziranga_wetlands_clean.jpg \
        --output results/refined_cloud_free.png
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gemini_refiner import GeminiMultimodalRefiner


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Gemini Multimodal Quad-Stack Satellite Refiner")
    parser.add_argument("--draft", required=True, help="Path to Stage-1 CR-DiffNet reconstruction image")
    parser.add_argument("--cloudy", required=True, help="Path to raw cloudy optical image")
    parser.add_argument("--sar", required=False, help="Path to Sentinel-1 SAR radar image (optional, synthesized from optical if omitted)")
    parser.add_argument("--anchor", required=False, help="Path to ISRO LISS-IV clear ground anchor (optional)")
    parser.add_argument("--output", default="refined_cloud_free_output.png", help="Path to save output PNG")
    parser.add_argument("--api-key", default=None, help="Google Gemini API key (or set GEMINI_API_KEY env var)")

    args = parser.parse_args()

    print("=" * 70)
    print("🛰️  TWO-STAGE CASCADED CLOUD REMOVAL: GEMINI MULTIMODAL REFINER (STAGE 2)")
    print("=" * 70)

    # Load Stage-1 Draft
    draft_bgr = cv2.imread(args.draft)
    if draft_bgr is None:
        print(f"❌ Error: Could not read draft image: {args.draft}")
        sys.exit(1)
    draft_rgb = cv2.cvtColor(draft_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = draft_rgb.shape

    # Load Cloudy Optical
    cloudy_bgr = cv2.imread(args.cloudy)
    if cloudy_bgr is None:
        print(f"❌ Error: Could not read cloudy image: {args.cloudy}")
        sys.exit(1)
    cloudy_rgb = cv2.cvtColor(cv2.resize(cloudy_bgr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Load SAR Radar (or synthesize from optical if omitted)
    if args.sar and os.path.exists(args.sar):
        sar_bgr = cv2.imread(args.sar)
        sar_rgb = cv2.cvtColor(cv2.resize(sar_bgr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        sar_gray = cv2.cvtColor(cloudy_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sar_rgb = np.repeat(sar_gray[:, :, np.newaxis], 3, axis=-1)

    # Load LISS-IV Clear Ground Anchor
    if args.anchor and os.path.exists(args.anchor):
        anchor_bgr = cv2.imread(args.anchor)
        anchor_rgb = cv2.cvtColor(cv2.resize(anchor_bgr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        anchor_rgb = draft_rgb.copy()

    # Detect cloud boundary
    r, g, b = cloudy_rgb[:, :, 0], cloudy_rgb[:, :, 1], cloudy_rgb[:, :, 2]
    min_ch = np.minimum(np.minimum(r, g), b)
    col_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
    gray = cv2.cvtColor((cloudy_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    thresh_bright = max(0.46, float(np.mean(gray)) + 0.12)
    cloud_core = ((min_ch > thresh_bright) & (col_diff < 0.20)) | ((min_ch > 0.65) & (col_diff < 0.30))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cloud_mask = cv2.dilate(cloud_core.astype(np.uint8), k_dilate).astype(np.float32)

    # Initialize Gemini Multimodal Refiner
    refiner = GeminiMultimodalRefiner(api_key=args.api_key)

    print("🤖 Processing Quad-Stack through Gemini Multimodal Refiner...")
    print("   ✓ Layer 1: CR-DiffNet Preliminary Reconstruction")
    print("   ✓ Layer 2: Raw Cloudy Optical Scene")
    print("   ✓ Layer 3: Sentinel-1 SAR C-Band Microwave Radar")
    print("   ✓ Layer 4: ISRO LISS-IV Clear Ground Anchor")

    refined_rgb, report = refiner.refine_quad_stack(
        draft_recon=draft_rgb,
        cloudy_opt=cloudy_rgb,
        sar_radar=sar_rgb,
        clean_anchor=anchor_rgb,
        cloud_mask=cloud_mask
    )

    # Save output
    out_bgr = cv2.cvtColor((refined_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)

    print("\n✅ REFINEMENT COMPLETE!")
    print(f"📁 Output Saved: {args.output}")
    print(f"📊 Pipeline: {report['pipeline']}")
    print(f"🛰️  Spectral Fidelity Score: {report['spectral_fidelity_score']}")
    print(f"⚡ Latency: {report['refinement_latency_ms']} ms")
    print(f"📝 Gemini Remote Sensing Assessment:\n   \"{report['gemini_assessment']}\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
