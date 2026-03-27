"""
creator.py — Programmer 1: The Creator
=======================================
Generates the mathematically perfect 4-column Ground Truth reference image.

Layout (1000 rows × 400 columns)
─────────────────────────────────
Col  Pixels    Content
 1    0–99     X-Axis Reference  : white bg + 10-px-wide black centre line
 2  100–199    Low  Chirp        :  5 → 25 cycles/image  (grayscale cosine)
 3  200–299    Mid  Chirp        : 15 → 35 cycles/image
 4  300–399    High Chirp        : 25 → 45 cycles/image

Chirp formula (constant-amplitude, linear-frequency sweep)
───────────────────────────────────────────────────────────
    t ∈ [0, 1)  (fractional row position)
    phase(t) = 2π · (f0·t  +  ½·(f1-f0)·t²)
    pixel(t) = 127.5 · (1 + cos(phase(t)))        ←  0..255, no clipping
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
        Build and return the full 1000 × 400 reference image as a uint8 array.

        Returns
        -------
        np.ndarray  shape (height, width), dtype uint8, values 0-255
        """
        h, w = self.cfg.height, self.cfg.width
        image = np.zeros((h, w), dtype=np.uint8)

        self._draw_col1_x_reference(image)
        self._draw_chirp_strip(image, self.cfg.col2_start, self.cfg.col2_end,
                               self.cfg.col2_f0, self.cfg.col2_f1)
        self._draw_chirp_strip(image, self.cfg.col3_start, self.cfg.col3_end,
                               self.cfg.col3_f0, self.cfg.col3_f1)
        self._draw_chirp_strip(image, self.cfg.col4_start, self.cfg.col4_end,
                               self.cfg.col4_f0, self.cfg.col4_f1)

        return image

    # ── Private helpers ────────────────────────────────────────────────────

    def _draw_col1_x_reference(self, image: np.ndarray) -> None:
        """
        Column 1: solid white background with a 10-px-wide black centre line.

        The crisp black line is used by the DSP analyser to track sub-pixel
        horizontal vibration shifts on every scan line.
        """
        # Flood the entire column band with white
        image[:, self.cfg.col1_start:self.cfg.col1_end] = 255
        # Paint the black reference line (10 px, centred at pixel 50)
        image[:, self.cfg.x_line_start:self.cfg.x_line_end] = 0

    def _draw_chirp_strip(self, image: np.ndarray,
                          x_start: int, x_end: int,
                          f0: float, f1: float) -> None:
        """
        Paint a spatial chirp (swept-frequency cosine) into one vertical strip.

        Intensity is kept at FULL AMPLITUDE throughout (0 → 255); only the
        spatial frequency of the fringes changes from f0 (top) to f1 (bottom).

        Parameters
        ----------
        x_start, x_end : column range of the strip
        f0             : instantaneous frequency at row 0  [cycles / image-height]
        f1             : instantaneous frequency at row H  [cycles / image-height]
        """
        H  = self.cfg.height
        t  = np.linspace(0.0, 1.0, H, endpoint=False)   # shape (H,)
        k  = f1 - f0                                      # chirp rate

        # Linear-phase (quadratic-phase) chirp — exact formula
        phase  = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2)

        # Binary spatial chirp: 255 if sin(phase) >= 0 else 0
        column = np.where(np.sin(phase) >= 0, 255, 0).astype(np.uint8)

        # Broadcast the 1-D column vector across every pixel in the strip
        image[:, x_start:x_end] = column[:, np.newaxis]


def main():
    """CLI entry-point: generate, save, and report the Ground Truth image."""
    cfg   = ScanConfig()
    synth = ImageSynthesizer(cfg)

    ref_image = synth.synthesize_reference_image()

    # Save lossless artefacts
    np.save(REF_NPY, ref_image)
    Image.fromarray(ref_image).save(REF_PNG)          # PNG = lossless

    print("=" * 52)
    print("  Ground Truth Reference — Generation Complete")
    print("=" * 52)
    print(f"  Shape  : {ref_image.shape} (H × W), dtype={ref_image.dtype}")
    print(f"  Layout :")
    print(f"    Col 1 (X-Ref)  cols   0–99  : white bg + 10-px black line")
    print(f"    Col 2 (Low)   cols 100–199  :  {cfg.col2_f0:.0f} → {cfg.col2_f1:.0f} cycles")
    print(f"    Col 3 (Mid)   cols 200–299  : {cfg.col3_f0:.0f} → {cfg.col3_f1:.0f} cycles")
    print(f"    Col 4 (High)  cols 300–399  : {cfg.col4_f0:.0f} → {cfg.col4_f1:.0f} cycles")
    print(f"  Saved  : {REF_NPY}")
    print(f"           {REF_PNG}")
    print("=" * 52)


if __name__ == "__main__":
    main()
