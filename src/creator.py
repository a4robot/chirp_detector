"""
creator.py — Programmer 1: The Creator
=======================================
Generates the 4-strip staggered binary reference image.

Layout (1000 rows × 1000 columns)
─────────────────────────────────
Col  Pixels    Content
 X     0–  99  X-Axis Reference  : white bg + 10-px black centre line at 45-55
gap  100– 149  Black
 1   150– 249  Fwd  chirp (phase 0)
gap  250– 299  Black
 2   300– 399  Rev  chirp (phase π/4)
gap  400– 449  Black
 3   450– 549  Fwd  chirp (phase π/2)
gap  550– 599  Black
 4   600– 699  Rev  chirp (phase 3π/4)
gap  700– 999  Black
"""

import numpy as np
from PIL import Image
from src.config import ScanConfig, REF_NPY, REF_PNG


class ImageSynthesizer:
    """Craft the zero-distortion, zero-noise 'Ground Truth' reference image."""

    def __init__(self, config: ScanConfig):
        self.cfg = config

    def synthesize_reference_image(self) -> np.ndarray:
        """Build and return the full reference image (uint8, grayscale)."""
        cfg = self.cfg
        image = np.zeros((cfg.height, cfg.width), dtype=np.uint8)

        # ── X-Axis Reference (cols 0–99): white bg + black centre line ─────────
        image[:, cfg.col1_start:cfg.col1_end] = 255
        image[:, cfg.x_line_start:cfg.x_line_end] = 0

        # ── 4 Staggered Binary Chirp Strips ────────────────────────────────────
        t = np.linspace(0.0, 1.0, cfg.height, endpoint=False)
        for col_start, col_end, mode, phase in cfg.strips:
            f0, f1 = cfg.f0, cfg.f1
            if mode == "rev":
                f0, f1 = f1, f0
            k = f1 - f0
            p = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phase
            column = np.where(np.sin(p) >= 0, np.uint8(255), np.uint8(0))
            image[:, col_start:col_end] = column[:, np.newaxis]

        return image


def main():
    cfg   = ScanConfig()
    synth = ImageSynthesizer(cfg)
    ref   = synth.synthesize_reference_image()

    np.save(REF_NPY, ref)
    Image.fromarray(ref).save(REF_PNG)
    print(f"Reference image saved: {ref.shape}  →  {REF_PNG}")


if __name__ == "__main__":
    main()
