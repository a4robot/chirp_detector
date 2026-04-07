"""
auto_analyze.py — Robust batch analyzer for real scans
======================================================
1.  Automatically crops scanner white margin.
2.  Resizes horizontally to 14975 (preserving row count).
3.  Runs the rotation-aware DSP pipeline.
"""

import os
import sys
import numpy as np
from PIL import Image
import json

from src.config import ScanConfig, REST_PNG, REPORT
from src.analyzer import DSPReconstructor

# PIL safety
Image.MAX_IMAGE_PIXELS = None

def get_chart_mask(img_arr):
    # Chart background is dark (<100 approx)
    # Background scanner margin is white (>230 approx)
    # Check middle columns to handle left/right slant
    h, w = img_arr.shape
    mid = img_arr[:, w//4 : 3*w//4]
    row_means = np.mean(mid, axis=1)
    
    # Simple thresholding
    is_chart = (row_means < 200)
    indices = np.where(is_chart)[0]
    if len(indices) == 0:
        return 0, h-1
    return indices[0], indices[-1]

def process_scan(scan_path, ref_path=None):
    print(f"\nAnalyzing scan: {scan_path}")
    base = os.path.basename(scan_path).split('.')[0]
    out_png = f"data/restored_{base}.png"
    
    # 1. Load
    img = Image.open(scan_path).convert('L')
    arr = np.array(img)
    H, W = arr.shape
    print(f"  Raw dimensions: {W} x {H}")
    
    # 2. Detect vertical bounds (remove scanner margin)
    r_start, r_end = get_chart_mask(arr)
    print(f"  Chart vertical ROI: {r_start} to {r_end} ({r_end - r_start + 1} rows)")
    chart_crop = arr[r_start:r_end+1, :]
    
    # 3. Handle X-scale (16384 -> 14975)
    # The chart starts at col 0 based on X-tracker center calculations
    # but we should check if there's a left/right white margin too.
    prof_h = np.mean(chart_crop, axis=0)
    c_mask = (prof_h < 230)
    c_indices = np.where(c_mask)[0]
    if len(c_indices) > 0:
        c_start, c_end = c_indices[0], c_indices[-1]
        print(f"  Chart horizontal ROI: {c_start} to {c_end} ({c_end-c_start+1} cols)")
        chart_crop = chart_crop[:, c_start:c_end+1]
    
    # Resize horizontally ONLY to 14975 (canonical width)
    # This ensures tags/x-tracker are at the expected column indices.
    h_curr, w_curr = chart_crop.shape
    target_w = 14975
    print(f"  Resizing width {w_curr} -> {target_w} (preserving rows {h_curr})...")
    img_canonical = Image.fromarray(chart_crop).resize((target_w, h_curr), Image.Resampling.LANCZOS)
    scanned_canonical = np.array(img_canonical)
    
    # 4. Normalize contrast (the decoder is sensitive to intensity)
    # Map 10th percentile to 0, 99th to 255.
    p5, p95 = np.percentile(scanned_canonical, [5, 95])
    print(f"  Normalization [5%, 95%]: ({p5:.1f}, {p95:.1f})")
    scanned_canonical = np.clip((scanned_canonical.astype(float) - p5) * (255.0 / (p95 - p5 + 1e-6)), 0, 255).astype(np.uint8)
    
    # 5. Restore
    cfg = ScanConfig()
    recon = DSPReconstructor(cfg)
    
    # Pipeline
    print("  [1/2] Vertical phase mapping...")
    y_phase = recon.restore_vertical_phase_mapping(scanned_canonical)
    
    print("  [2/2] Horizontal vibration tracking...")
    x_shifts = recon.detect_horizontal_vibration(scanned_canonical, y_phase)
    
    print("  Generating reconstruction...")
    recipe = recon.generate_reconstruction_recipe(y_phase, x_shifts)
    restored = recon.reconstruct_perfect_image(scanned_canonical, recipe)
    
    Image.fromarray(restored).save(out_png)
    print(f"  Saved restored to → {out_png}")
    
    # Comparison
    if ref_path:
        ref_img = Image.open(ref_path).convert('L')
        ref_arr = np.array(ref_img)
        h = min(ref_arr.shape[0], restored.shape[0])
        w = min(ref_arr.shape[1], restored.shape[1])
        metrics = recon.evaluate_restoration_quality(ref_arr[:h, :w], restored[:h, :w])
        print(f"  Quality: PSNR={metrics['psnr']} dB, MAE={metrics['mae']}")

if __name__ == "__main__":
    scans = [
        "data/image_1775189755.png",
        "data/image_1775191033.png",
        "data/image_1775191067.png"
    ]
    ref = "data/reference_image.png"
    for s in scans:
        if os.path.exists(s):
            try:
                process_scan(s, ref if os.path.exists(ref) else None)
            except Exception as e:
                print(f"  CRASH: {e}")
                import traceback
                traceback.print_exc()
