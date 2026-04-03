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


from src.rotation_tag import build_layout, synthesize_rotation_tag, TOTAL_RIGHT_WIDTH

class ImageSynthesizer:
    """Craft the zero-distortion, zero-noise 'Ground Truth' reference image."""

    def __init__(self, config: ScanConfig):
        self.cfg = config

    def synthesize_reference_image(self) -> np.ndarray:
        """Build and return the full reference image (uint8, grayscale)."""
        cfg = self.cfg
        image = np.zeros((cfg.height, cfg.width), dtype=np.uint8)

        # ── Right X-Tracker ──
        # Placed in the safe zone after the right rotation tag's max drift.
        # Track line is drawn white on a black background
        image[:, cfg.x_line_start:cfg.x_line_end] = 255


        # ── 4 Staggered Binary Chirp Strips ────────────────────────────────────
        # t = np.linspace(0.0, 1.0, cfg.height, endpoint=False)
        # for col_start, col_end, mode, phase, f1_strip in cfg.strips:
        #     f0, f1 = cfg.f0, f1_strip
        #     if mode == "rev":
        #         f0, f1 = f1, f0
        #     k = f1 - f0
        #     p = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phase
        #     column = np.where(np.sin(p) >= 0, np.uint8(255), np.uint8(0))
        #     image[:, col_start:col_end] = column[:, np.newaxis]

        # ── Centered 16×6 Checkerboard (1000 px/square) ────────────────────────
        # Grid: 6 cols × 16 rows = 6 000 × 16 000 px, perfectly centered.
        checker_cols  = 11
        checker_rows  = 20
        square_px     = 1000
        grid_w = checker_cols * square_px   #  6 000
        grid_h = checker_rows * square_px   # 16 000
        cx0 = (cfg.width  - grid_w) // 2   # left edge of grid
        cy0 = (cfg.height - grid_h) // 2   # top  edge of grid
        # Fill area is naturally black, so we don't draw a white background box
        # We just draw white squares
        for gr in range(checker_rows):
            for gc in range(checker_cols):
                if (gr + gc) % 2 == 0:
                    x1 = cx0 + gc * square_px
                    y1 = cy0 + gr * square_px
                    x2 = min(x1 + square_px, cfg.width)
                    y2 = min(y1 + square_px, cfg.height)
                    image[y1:y2, x1:x2] = 255   # white square on black background

        # ── Rotation Tags ──────────────────────────────────────────────────────
        # Left tag starts at col 200 (padding) — positive-angle strips fan RIGHT
        # at the top, so max right extent stays within the margin.
        # Right tag starts offset by ~550px (350 drift + 200 padding) from the right edge.
        layout = build_layout(left_x0=200, right_x0=cfg.width - TOTAL_RIGHT_WIDTH - 350 - 200)
        image = synthesize_rotation_tag(image, layout)

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
