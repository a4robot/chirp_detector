"""
analyze_scan_golden.py — Analyze a golden scanned calibration chart image with vertical flip
==========================================================================================

Usage:
  # Basic: just restore the scanned image (auto-flips vertically)
  PYTHONPATH=. python analyze_scan_golden.py data/scanned_image.png

  # With reference for PSNR/MAE quality comparison
  PYTHONPATH=. python analyze_scan_golden.py data/scanned_image.png --reference data/reference_image.png
"""

import argparse
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from src.config import ScanConfig, REST_NPY, REST_PNG, REPORT
from src.analyzer import DSPReconstructor


def load_grayscale(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img)


def crop_to_common(a: np.ndarray, b: np.ndarray):
    """Crop both arrays to their overlapping region (min height × min width)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a scanned calibration chart (Golden Version with Vertical Flip)."
    )
    parser.add_argument("scan", help="Path to scanned image (PNG/BMP/JPG/TIFF)")
    parser.add_argument("--reference", "-r", default=None,
                        help="Optional reference image for PSNR/MAE evaluation")
    parser.add_argument("--x", type=int, default=None,
                        help="Override X-coordinate for chirp analysis (like plot_chirp.py)")
    parser.add_argument("--width", type=int, default=20,
                        help="Width of strip for chirp analysis (default: 20)")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable vertical flipping (flips by default in this version)")
    args = parser.parse_args()

    cfg = ScanConfig()
    recon = DSPReconstructor(cfg)

    # ── Load scanned image ────────────────────────────────────────────────────
    print(f"Loading scanned image: {args.scan}")
    scanned = load_grayscale(args.scan)
    
    if not args.no_flip:
        print("  Flipping image vertically...")
        scanned = np.flipud(scanned)
        
    print(f"  Shape: {scanned.shape}")

    # ── Step 1: Decode rotation + vertical phase mapping ─────────────────────
    print("\n[1/2] Restoring vertical phase mapping (rotation tags)...")
    y_phase = recon.restore_vertical_phase_mapping(scanned, override_x=args.x, override_width=args.width)

    # ── Step 2: Detect horizontal vibration (x-tracker) ──────────────────────
    print("\n[2/2] Detecting horizontal vibration (x-tracker)...")
    x_shifts = recon.detect_horizontal_vibration(scanned, y_phase)

    # Save intermediate X-restored image for verification
    print("  Generating intermediate [X-restored] image...")
    x_only = recon.apply_x_shifts_only(scanned, x_shifts)
    Image.fromarray(x_only).save("data/golden_x_restored_only.png")
    print(f"    Saved purely horizontal restoration to: data/golden_x_restored_only.png")

    # ── Reconstruct ──────────────────────────────────────────────────────────
    recipe = recon.generate_reconstruction_recipe(y_phase, x_shifts)
    restored = recon.reconstruct_perfect_image(scanned, recipe)

    np.save("data/golden_" + REST_NPY.split('/')[-1], restored)
    Image.fromarray(restored).save("data/golden_" + REST_PNG.split('/')[-1])
    print(f"\nRestored image saved → data/golden_{REST_PNG.split('/')[-1]}  shape={restored.shape}")

    # ── Optional: quality comparison against reference ────────────────────────
    if args.reference:
        print(f"\nLoading reference for quality comparison: {args.reference}")
        if args.reference.endswith('.npy'):
            ref = np.load(args.reference)
        else:
            ref = load_grayscale(args.reference)
        print(f"  Reference shape: {ref.shape}")
        print(f"  Restored  shape: {restored.shape}")

        ref_crop, res_crop = crop_to_common(ref, restored)
        print(f"  Cropped to common region: {ref_crop.shape}")

        metrics = recon.evaluate_restoration_quality(ref_crop, res_crop)
        print(f"\n{'=' * 40}")
        print(f"  PSNR : {metrics['psnr']} dB")
        print(f"  MAE  : {metrics['mae']}")
        print(f"{'=' * 40}")
    else:
        print("\nTip: Add --reference <file> to compute PSNR/MAE quality metrics.")


if __name__ == "__main__":
    main()
