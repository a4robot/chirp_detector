"""
creator.py — Programmer 1: The Creator
=======================================
Generates the mathematically perfect 4-column Ground Truth reference image.

Layout (1000 rows × 400 columns)
─────────────────────────────────
Col  Pixels    Content
 1    0– 99    X-Axis Reference  : white bg + 10-px-wide black centre line
 2  100–199    Low Chirp Fwd     :  5 → 25 cycles  (freq ↑ top→bottom)
 3  200–299    Low Chirp Rev     : 25 →  5 cycles  (freq ↓ top→bottom)
 4  300–399    V-Chirp           :  5 → 25 → 5     (freq ↑ top→mid, ↓ mid→bot)

Binary spatial chirp formula
─────────────────────────────
    t ∈ [0, 1)  (fractional row within the half being generated)
    phase(t) = 2π · (f0·t  +  ½·k·t²),   k = f1 − f0
    pixel(t) = 255  if sin(phase) ≥ 0  else  0
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
        """Build and return the full 1000 × 400 reference image (uint8)."""
        image = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)

        self._draw_col1_x_reference(image)

        # Col 2: forward low chirp (5 → 25, freq ↑)
        self._draw_chirp_strip(image,
                               self.cfg.col2_start, self.cfg.col2_end,
                               self.cfg.col2_f0,    self.cfg.col2_f1)

        # Col 3: reversed low chirp (25 → 5, freq ↓)
        self._draw_chirp_strip(image,
                               self.cfg.col3_start, self.cfg.col3_end,
                               self.cfg.col3_f0,    self.cfg.col3_f1)

        # Col 4: V-chirp (5→25 top-half, 25→5 bottom-half)
        self._draw_v_chirp_strip(image,
                                 self.cfg.col4_start, self.cfg.col4_end,
                                 self.cfg.col4_f_edge, self.cfg.col4_f_peak)

        return image

    # ── Private helpers ────────────────────────────────────────────────────

    def _draw_col1_x_reference(self, image: np.ndarray) -> None:
        """Column 1: white bg + 10-px black centre line for vibration tracking."""
        image[:, self.cfg.col1_start:self.cfg.col1_end] = 255
        image[:, self.cfg.x_line_start:self.cfg.x_line_end] = 0

    def _draw_chirp_strip(self, image: np.ndarray,
                          x_start: int, x_end: int,
                          f0: float, f1: float) -> None:
        """
        Binary spatial chirp over the full image height.

        Supports both positive (ascending) and negative (descending) chirp rates.
        """
        H = self.cfg.height
        t = np.linspace(0.0, 1.0, H, endpoint=False)
        k = f1 - f0
        phase  = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2)
        column = np.where(np.sin(phase) >= 0, np.uint8(255), np.uint8(0))
        image[:, x_start:x_end] = column[:, np.newaxis]

    def _draw_v_chirp_strip(self, image: np.ndarray,
                            x_start: int, x_end: int,
                            f_edge: float, f_peak: float) -> None:
        """
        V-Chirp: binary spatial chirp that peaks in frequency at the midpoint row.

        Top half (rows 0 → H//2 - 1): frequency rises from f_edge → f_peak.
        Bottom half (rows H//2 → H-1): frequency falls from f_peak → f_edge.

        The two halves are generated as independent chirps each compressed into
        H//2 rows, then concatenated.  The phase is continuous at the join
        because both halves start their own t ∈ [0, 1) sweep — the visual
        continuity is that both halves share the same phase at t=0 (sin ≥ 0).

        Parameters
        ----------
        f_edge : frequency [cy/img-height] at row 0 and row H-1
        f_peak : frequency [cy/img-height] at row H//2 (midpoint)
        """
        H    = self.cfg.height
        half = H // 2
        rest = H - half   # handles odd H

        column = np.concatenate([
            self._binary_chirp_segment(half, f_edge, f_peak),  # top:  ↑
            self._binary_chirp_segment(rest, f_peak, f_edge),  # bot:  ↓
        ]).astype(np.uint8)

        image[:, x_start:x_end] = column[:, np.newaxis]

    @staticmethod
    def _binary_chirp_segment(n_rows: int, f0: float, f1: float) -> np.ndarray:
        """Return a 1-D binary chirp of length n_rows, f0 at start, f1 at end."""
        t     = np.linspace(0.0, 1.0, n_rows, endpoint=False)
        k     = f1 - f0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2)
        return np.where(np.sin(phase) >= 0, np.uint8(255), np.uint8(0))


def main():
    cfg   = ScanConfig()
    synth = ImageSynthesizer(cfg)
    ref   = synth.synthesize_reference_image()

    np.save(REF_NPY, ref)
    Image.fromarray(ref).save(REF_PNG)

    print("=" * 60)
    print("  Ground Truth Reference — Generation Complete")
    print("=" * 60)
    print(f"  Shape  : {ref.shape}  dtype={ref.dtype}")
    print(f"  Layout :")
    print(f"    Col 1 (X-Ref)  cols   0–99  : white bg + 10-px black line")
    print(f"    Col 2 (Fwd)    cols 100–199  : {cfg.col2_f0:.0f}→{cfg.col2_f1:.0f} cy  (freq ↑)")
    print(f"    Col 3 (Rev)    cols 200–299  : {cfg.col3_f0:.0f}→{cfg.col3_f1:.0f} cy  (freq ↓)")
    print(f"    Col 4 (V)      cols 300–399  : {cfg.col4_f_edge:.0f}→{cfg.col4_f_peak:.0f}→"
          f"{cfg.col4_f_edge:.0f} cy  (V-chirp, peak at row {cfg.height//2})")
    print(f"  Saved  : {REF_NPY}")
    print(f"           {REF_PNG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
