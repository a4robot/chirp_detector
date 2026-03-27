"""
creator.py — Programmer 1: The Creator
=======================================
Generates the mathematically perfect 3-column Ground Truth reference image.

Layout (1000 rows × 300 columns)
─────────────────────────────────
Col  Pixels    Content
 1    0– 99    X-Axis Reference  : white bg + 10-px-wide black centre line
 2  100–199    Low Chirp Fwd     :  5 → 25 cycles/image  (top → bottom)
 3  200–299    Low Chirp Rev     : 25 →  5 cycles/image  (bottom → top mirror)

Chirp formula (constant-amplitude binary spatial chirp)
────────────────────────────────────────────────────────
    t ∈ [0, 1)  (fractional row position)
    phase(t) = 2π · (f0·t  +  ½·(f1−f0)·t²)
    pixel(t) = 255 if sin(phase(t)) ≥ 0 else 0

For the reversed strip  f0=25, f1=5  →  k = −20  (negative chirp rate).
The formula is identical; only t ∈ [0,1) and the sign of k differ.
"""

import numpy as np
from PIL import Image
from src.config import ScanConfig, REF_NPY, REF_PNG


class ImageSynthesizer:
    """Craft the zero-distortion, zero-noise 'Ground Truth' reference image."""

    def __init__(self, config: ScanConfig):
        self.cfg = config

    # ── Public API ─────────────────────────────────────────────────────────

    def synthesize_reference_image(self) -> np.ndarray:
        """
        Build and return the full 1000 × 300 reference image as a uint8 array.

        Returns
        -------
        np.ndarray  shape (height, width), dtype uint8, values 0 or 255
        """
        h, w = self.cfg.height, self.cfg.width
        image = np.zeros((h, w), dtype=np.uint8)

        self._draw_col1_x_reference(image)

        # Col 2: forward low chirp  (5 → 25, frequency increases downward)
        self._draw_chirp_strip(image,
                               self.cfg.col2_start, self.cfg.col2_end,
                               self.cfg.col2_f0,    self.cfg.col2_f1)

        # Col 3: reversed low chirp (25 → 5, frequency decreases downward)
        self._draw_chirp_strip(image,
                               self.cfg.col3_start, self.cfg.col3_end,
                               self.cfg.col3_f0,    self.cfg.col3_f1)

        return image

    # ── Private helpers ────────────────────────────────────────────────────

    def _draw_col1_x_reference(self, image: np.ndarray) -> None:
        """
        Column 1: solid white background with a 10-px-wide black centre line.
        Used by the DSP analyser to track sub-pixel horizontal vibration.
        """
        image[:, self.cfg.col1_start:self.cfg.col1_end] = 255
        image[:, self.cfg.x_line_start:self.cfg.x_line_end] = 0

    def _draw_chirp_strip(self, image: np.ndarray,
                          x_start: int, x_end: int,
                          f0: float, f1: float) -> None:
        """
        Paint a binary spatial chirp into one vertical strip.

        Pixel is fully ON (255) or fully OFF (0); only fringe spacing changes.
        Supports both positive (f0<f1) and negative (f0>f1) chirp rates.

        Parameters
        ----------
        x_start, x_end : column slice for this strip
        f0             : instantaneous frequency at row   0  [cycles/image-height]
        f1             : instantaneous frequency at row H-1  [cycles/image-height]
        """
        H = self.cfg.height
        t = np.linspace(0.0, 1.0, H, endpoint=False)
        k = f1 - f0                                   # negative for reversed strip

        phase  = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2)
        column = np.where(np.sin(phase) >= 0, np.uint8(255), np.uint8(0))

        image[:, x_start:x_end] = column[:, np.newaxis]


def main():
    """CLI entry-point: generate, save, and report the Ground Truth image."""
    cfg   = ScanConfig()
    synth = ImageSynthesizer(cfg)

    ref_image = synth.synthesize_reference_image()

    np.save(REF_NPY, ref_image)
    Image.fromarray(ref_image).save(REF_PNG)

    print("=" * 58)
    print("  Ground Truth Reference — Generation Complete")
    print("=" * 58)
    print(f"  Shape  : {ref_image.shape}  dtype={ref_image.dtype}")
    print(f"  Layout :")
    print(f"    Col 1 (X-Ref) cols   0–99 : white bg + 10-px black line")
    print(f"    Col 2 (Fwd)   cols 100–199: {cfg.col2_f0:.0f} → {cfg.col2_f1:.0f} cy  (freq ↑ top→bot)")
    print(f"    Col 3 (Rev)   cols 200–299: {cfg.col3_f0:.0f} → {cfg.col3_f1:.0f} cy  (freq ↓ top→bot)")
    print(f"  Saved  : {REF_NPY}")
    print(f"           {REF_PNG}")
    print("=" * 58)


if __name__ == "__main__":
    main()
