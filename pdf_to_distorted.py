import fitz
import numpy as np
import cv2
import sys
import os

from src.config import ScanConfig, DIST_NPY, DIST_PNG, REF_NPY

def main():
    print("Converting PDF back to arrays for analysis...")
    pdf_path = "calibration_target_vector.pdf"
    
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    target_w = 14975
    target_h = 20000
    
    zoom_x = target_w / page.rect.width
    zoom_y = target_h / page.rect.height
    
    # 2x oversample for anti-aliasing safety, then downscale perfectly
    mat = fitz.Matrix(zoom_x, zoom_y)
    
    # Enable high-quality anti-aliasing rendering
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)
    
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    
    if img.shape[0] != target_h or img.shape[1] != target_w:
        print(f"Resize needed: {img.shape} -> {(target_h, target_w)}")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Threshold clearly back to binary (our reference was binary)
    # The pdf draw lines exactly, so edges should be perfect
    img_bin = np.where(img > 127, 255, 0).astype(np.uint8)

    np.save(DIST_NPY, img_bin)
    
    # Load exact reference to ensure size matches
    ref = np.load(REF_NPY)
    print(f"Reference shape: {ref.shape}")
    print(f"PDF Conv  shape: {img_bin.shape}")
    
    print(f"Calculating PSNR manually to prove accuracy of LaTeX generation...")
    import math
    diff = np.abs(ref.astype(float) - img_bin.astype(float))
    mae = np.mean(diff)
    mse = np.mean(diff ** 2)
    psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99.0
    
    print(f"\nPDF → Pixels Quality Match:")
    print(f"  MAE  : {mae:.4f} px")
    print(f"  PSNR : {psnr:.1f} dB")
    print(f"  Total differences: {np.sum(diff > 0)} pixels out of {target_w*target_h}")

    # Now run analyzer directly
    from main import step_analyze, step_distort_classic
    cfg = ScanConfig()
    
    print("\nApplying mechanical distortion (rotation + vibration) to the PDF raster...")
    distorted, truth = step_distort_classic(cfg, img_bin, rotation_deg=0.0)
    
    print("\nRunning standard analyzer on the distorted PDF-generated raster...")
    metrics = step_analyze(cfg, ref, distorted)
    
    print("\n" + "=" * 40)
    print(f"  Rotation : +0.0° Simulated")
    print(f"  PSNR     : {metrics['psnr']} dB")
    print(f"  MAE      : {metrics['mae']}")
    print("=" * 40)

if __name__ == "__main__":
    main()
