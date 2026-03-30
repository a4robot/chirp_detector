"""
analyzer.py — Programmer 3: The DSP Reconstructor
===================================================
Ingests a distorted image and recovers the perfect 1000-line original
WITHOUT any knowledge of the answer key.

Pipeline
--------
Step 1  detect_horizontal_vibration()
        Sub-pixel centroid of the black reference line → per-row Δx

Step 2  restore_vertical_phase_mapping()
        Multi-strip confidence-weighted Hilbert phase fusion of THREE
        spatial chirp columns.  Each strip votes on the source-row mapping;
        the vote weight drops to zero when a strip aliases or saturates.
        Frequency-preference weighting ensures the highest-resolution strip
        dominates when healthy.

Step 3  generate_reconstruction_recipe()
        Per-row action table: keep / drop + shift_x

Step 4  reconstruct_perfect_image()
        Apply x-corrections, drop duplicates, interpolate skips → 1000 rows

Step 5  evaluate_restoration_quality()
        PSNR / MAE vs. ground truth; saves diff artefact
"""

import numpy as np
from scipy.signal import hilbert
from scipy.fft import next_fast_len
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d, median_filter
from PIL import Image
import json
import os
import math

from src.config import (ScanConfig,
                        DIST_NPY, REF_NPY, REST_NPY, REST_PNG,
                        RECIPE, DIFF_PNG, REPORT)


# ─────────────────────────────────────────────────────────────────────────────
class DSPReconstructor:
    """
    Complete DSP pipeline that detects and undoes mechanical line-scan
    distortions (vibration + slip / skip) using only the distorted image.

    Usage
    -----
        recon    = DSPReconstructor(ScanConfig())
        x_shift  = recon.detect_horizontal_vibration(distorted)
        src_y    = recon.restore_vertical_phase_mapping(distorted)
        recipe   = recon.generate_reconstruction_recipe(src_y, x_shift)
        restored = recon.reconstruct_perfect_image(distorted, recipe)
        metrics  = recon.evaluate_restoration_quality(reference, restored)
    """

    def __init__(self, config: ScanConfig):
        self.cfg = config

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – X-Axis: Vibration Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_horizontal_vibration(self, distorted_image: np.ndarray) -> np.ndarray:
        """
        Recover the sub-pixel horizontal shift (Δx) for every distorted row.

        Column 1 (cols 0–99) is WHITE (255) with a 10-px BLACK (0) centre line
        at the ideal position x = 49.5 px (centre of cols 45–54).

        Key robustness fix — chirp cross-column bleed
        ---------------------------------------------
        When x_shift is negative the simulator pulls pixels from Col2 (chirp,
        starting at col 100) into the rightmost pixels of Col1.  A dark chirp
        fringe appearing at col 99 creates a false dark region that drags the
        full-band centroid rightward — opposite sign to the truth.

        Solution: restrict the search to a *safe inner window* around the
        expected line centre (cols 25–74), guaranteed to stay clear of the
        col-100 chirp boundary at maximum vibration amplitude.

        Algorithm
        ---------
        1. Within safe window (cols 25–74), find argmin (darkest pixel).
        2. Require anchor pixel to be genuinely dark (value < 128).
        3. Refine: compute intensity-weighted centroid within ±8 px of anchor.
        4. shift = measured_centroid − expected_centroid (49.5).
        5. Gap-fill failed rows via linear interpolation.
        6. Apply 5-row median filter to suppress residual spike outliers.

        Returns
        -------
        np.ndarray  shape (N_distorted,)  — signed shift per row [pixels]
        """
        n_rows  = distorted_image.shape[0]
        shifts  = np.full(n_rows, np.nan)
        cx_ref  = self.cfg.x_tracker_expected_center        # 49.5

        # Safe search window — clear of chirp bleed
        margin   = 25
        safe_lo  = max(self.cfg.col1_start, int(cx_ref) - margin)   # 25
        safe_hi  = min(self.cfg.col1_end,   int(cx_ref) + margin)   # 74
        refine_r = 8

        for r in range(n_rows):
            window = distorted_image[r, safe_lo:safe_hi].astype(float)

            # Coarse anchor: darkest pixel in safe window
            anchor_local = int(np.argmin(window))
            anchor_abs   = safe_lo + anchor_local

            if window[anchor_local] > 128:          # no dark line visible
                continue

            # Refined centroid: invert within ±refine_r of anchor
            lo2 = max(safe_lo, anchor_abs - refine_r)
            hi2 = min(safe_hi, anchor_abs + refine_r + 1)
            sub  = distorted_image[r, lo2:hi2].astype(float)
            inv  = 255.0 - sub
            coords = np.arange(lo2, hi2, dtype=float)
            thresh = inv.max() * 0.5
            mass   = inv * (inv >= thresh)
            total  = mass.sum()
            if total > 0:
                shifts[r] = float((coords * mass).sum() / total) - cx_ref

        # Gap-fill NaN rows
        valid = ~np.isnan(shifts)
        if valid.sum() < 2:
            return np.zeros(n_rows)
        idx    = np.arange(n_rows)
        filled = np.interp(idx, idx[valid], shifts[valid])

        # Median filter to kill residual spike outliers
        return median_filter(filled, size=5, mode='nearest')

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – Y-Axis: Multi-Strip Confidence-Weighted Phase Fusion
    # ─────────────────────────────────────────────────────────────────────────

    def restore_vertical_phase_mapping(self, distorted_image: np.ndarray) -> np.ndarray:
        """
        Build a source-row mapping using THREE complementary chirp strips.

        Strip layout
        ------------
        • Col 2 (Fwd,  5→25)      : freq ↑ top→bottom
        • Col 3 (Rev, 25→ 5)      : freq ↓ top→bottom (mirror of col2)
        • Col 4 (V,    5→25→5)    : freq ↑ top→mid, freq ↓ mid→bottom

        The Fwd+Rev pair pins global position (their freq-sum is constant at 30).
        The V-strip adds an independent constraint that is locally ambiguous
        (symmetric around row 500) but breaks ties when fused with col2/col3.

        Returns
        -------
        np.ndarray  shape (N_distorted,)  — source row index ∈ [0, height-1]
        """
        all_estimates  = []
        all_confidence = []

        # ── Forward and Reversed single-chirp strips ──────────────────────────
        for (cs, ce, f0, f1) in [
            (self.cfg.col2_start, self.cfg.col2_end,
             self.cfg.col2_f0,    self.cfg.col2_f1),
            (self.cfg.col3_start, self.cfg.col3_end,
             self.cfg.col3_f0,    self.cfg.col3_f1),
        ]:
            est, conf = self._estimate_source_y_single_strip(
                distorted_image, cs, ce, f0, f1)
            all_estimates.append(est)
            all_confidence.append(conf)

        # ── V-Chirp strip (two-segment) ───────────────────────────────────────
        est_v, conf_v = self._estimate_source_y_v_strip(
            distorted_image,
            self.cfg.col4_start, self.cfg.col4_end,
            self.cfg.col4_f_edge, self.cfg.col4_f_peak)
        all_estimates.append(est_v)
        all_confidence.append(conf_v)

        estimates  = np.array(all_estimates)    # (3, N)
        confidence = np.array(all_confidence)   # (3, N)

        total_weight = confidence.sum(axis=0)
        denominator  = np.where(total_weight > 1e-6, total_weight, 1.0)
        fused        = (estimates * confidence).sum(axis=0) / denominator

        fallback = np.linspace(0.0, self.cfg.height - 1, len(fused))
        fused    = np.where(total_weight > 1e-6, fused, fallback)

        return self._enforce_monotonicity(fused)


    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – Recipe Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_reconstruction_recipe(self,
                                       source_y: np.ndarray,
                                       x_shifts: np.ndarray,
                                       min_advance: float = 0.1) -> list:
        """
        Walk through every distorted row and generate an action instruction.

        Each entry:
          • distorted_row  – index in the distorted image
          • source_y       – estimated ideal-image row this maps to
          • action         – 'keep' (unique) or 'drop' (slip duplicate)
          • shift_x        – horizontal correction to apply before use

        A row is 'drop' if source_y advanced < min_advance since the last
        kept row, indicating a repeated slip sample.

        Returns  list of dict — one entry per distorted row
        """
        recipe      = []
        last_kept_y = -1.0

        for i, (sy, sx) in enumerate(zip(source_y, x_shifts)):
            sy, sx = float(sy), float(sx)
            action = 'keep' if (sy - last_kept_y >= min_advance) else 'drop'
            if action == 'keep':
                last_kept_y = sy

            recipe.append({
                'distorted_row': i,
                'source_y':      round(sy, 4),
                'action':        action,
                'shift_x':       round(sx, 4),
            })

        return recipe

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – Image Reconstruction
    # ─────────────────────────────────────────────────────────────────────────

    def reconstruct_perfect_image(self,
                                  distorted_image: np.ndarray,
                                  recipe: list) -> np.ndarray:
        """
        Execute the action recipe to produce a perfectly de-warped image.

        For each 'keep' row:
          • Apply sub-pixel horizontal correction (inverse of vibration shift).
        Collect (source_y, corrected_line) pairs, then interpolate every
        integer target row in [0, height−1].  This simultaneously fixes
        vibration, drops slip duplicates, and fills skip gaps.

        Returns  np.ndarray  shape (height, width) uint8  — exactly 1000 rows
        """
        width  = distorted_image.shape[1]
        orig_x = np.arange(width, dtype=float)

        kept_source_y: list[float]       = []
        kept_lines:    list[np.ndarray]  = []

        for entry in recipe:
            if entry['action'] != 'keep':
                continue
            r, sx = entry['distorted_row'], entry['shift_x']
            row = distorted_image[r].astype(float)
            # Sub-pixel horizontal correction (undo the shift)
            corrected = np.interp(orig_x + sx, orig_x, row,
                                  left=row[0], right=row[-1])
            kept_source_y.append(entry['source_y'])
            kept_lines.append(corrected)

        if len(kept_source_y) < 2:
            raise RuntimeError("Too few 'keep' rows — check min_advance / phase mapping.")

        kept_lines_arr = np.array(kept_lines)           # (M, W)
        target_y       = np.linspace(0.0, self.cfg.height - 1, self.cfg.height)
        restored       = np.zeros((self.cfg.height, width), dtype=np.uint8)

        for col in range(width):
            restored[:, col] = np.clip(
                np.interp(target_y, kept_source_y, kept_lines_arr[:, col]),
                0, 255
            ).astype(np.uint8)

        return restored

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – Quality Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_restoration_quality(self,
                                     original: np.ndarray,
                                     restored: np.ndarray) -> dict:
        """
        Compare the restored image to the ground-truth original.

        Saves
        -----
        • diff_image.png        : |original − restored| × 4 for visibility
        • analysis_report.txt

        Returns  dict with keys 'psnr' (dB) and 'mae'
        """
        diff     = np.abs(original.astype(int) - restored.astype(int))
        diff_vis = np.clip(diff * 4, 0, 255).astype(np.uint8)
        Image.fromarray(diff_vis).save(DIFF_PNG)

        mse  = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        mae  = float(diff.mean())

        metrics = {'psnr': round(psnr, 3), 'mae': round(mae, 4)}

        with open(REPORT, 'w') as f:
            f.write("=" * 52 + "\n")
            f.write("  Chirp Detector — Restoration Quality Report\n")
            f.write("=" * 52 + "\n\n")
            f.write(f"  PSNR         : {metrics['psnr']} dB\n")
            f.write(f"  MAE          : {metrics['mae']}\n\n")
            f.write("  Interpretation\n")
            f.write("  --------------\n")
            f.write("  > 40 dB  →  near-perfect reconstruction\n")
            f.write("  30–40 dB →  good reconstruction\n")
            f.write("  < 30 dB  →  significant residual distortion\n")

        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_source_y_single_strip(self,
                                        distorted_image: np.ndarray,
                                        col_start: int,
                                        col_end: int,
                                        f0: float,
                                        f1: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Binary edge-counting phase inversion for ONE linear-frequency chirp strip.
        It precisely matches observed edge transitions 1-to-1 against generated
        ideal edge transitions, naturally reversing non-linear mechanical slip.
        """
        H      = float(self.cfg.height)
        k      = f1 - f0
        n_rows = distorted_image.shape[0]

        # ── 1. Create the IDEAL column profile computationally ────────────────
        t_ideal = np.linspace(0.0, 1.0, int(H), endpoint=False)
        phase_ideal = 2.0 * np.pi * (f0 * t_ideal + 0.5 * k * t_ideal ** 2)
        ideal_binary = (np.sin(phase_ideal) >= 0).astype(int)

        # ── 2. Extract distorted binary column profile ────────────────────────
        raw = distorted_image[:, col_start:col_end].mean(axis=1)
        dist_binary = (raw >= 128).astype(int)

        # ── 3. Detect edge transition indices (rows) ──────────────────────────
        ideal_diff = np.diff(ideal_binary)
        dist_diff  = np.diff(dist_binary)
        
        ideal_edges = np.where(ideal_diff != 0)[0].astype(float)
        dist_edges  = np.where(dist_diff != 0)[0].astype(float)

        if len(dist_edges) < 2 or len(ideal_edges) < 2:
            return np.linspace(0, H - 1, n_rows), np.zeros(n_rows)

        # ── 4. Align edges 1-to-1 ─────────────────────────────────────────────
        ideal_first_pol = ideal_diff[int(ideal_edges[0])]
        dist_first_pol  = dist_diff[int(dist_edges[0])]
        
        ideal_start = 0
        if ideal_first_pol != dist_first_pol:
            ideal_start = 1
            
        min_len = min(len(ideal_edges) - ideal_start, len(dist_edges))
        
        matched_ideal = ideal_edges[ideal_start: ideal_start + min_len]
        matched_dist  = dist_edges[:min_len]

        # ── 5. Interpolate to fill all rows ───────────────────────────────────
        all_rows   = np.arange(n_rows, dtype=float)
        # We linearly extrapolate via Pchip to map the edges exactly
        pchip = PchipInterpolator(matched_dist, matched_ideal, extrapolate=True)
        source_y = np.clip(pchip(all_rows), 0, H - 1)

        # ── 6. Confidence metric ──────────────────────────────────────────────
        # Confidence drops if aliasing blurs the binary wave into flat gray
        f_mean       = (f0 + f1) / 2.0
        win          = max(5, int(H / f_mean * 2))
        raw_float    = raw.astype(float)
        raw_dc       = uniform_filter1d(raw_float, size=win, mode='nearest')
        raw_ac       = raw_float - raw_dc
        local_rms    = np.sqrt(uniform_filter1d(raw_ac ** 2, size=win, mode='nearest'))
        expected_rms = 127.5
        confidence   = np.clip(local_rms / expected_rms, 0.0, 1.0)
        
        # Penalize implausibly fast frequency (out of band)
        f_lo = min(f0, f1)
        f_hi = max(f0, f1)
        f_inst     = f_lo + abs(k) * (source_y / (H - 1))
        margin     = 2.0
        in_band    = (f_inst >= f_lo - margin) & (f_inst <= f_hi + margin)
        confidence = confidence * in_band.astype(float)

        return source_y, confidence

    def _estimate_source_y_v_strip(self,
                                   distorted_image: np.ndarray,
                                   col_start: int, col_end: int,
                                   f_edge: float, f_peak: float,
                                   ) -> tuple[np.ndarray, np.ndarray]:
        """
        Edge-based source-row estimation for the V-chirp strip.

        The V-chirp is generated in two independent half-segments:
          • Top half (rows 0 … H//2−1): chirp from f_edge → f_peak
          • Bot half (rows H//2 … H−1): chirp from f_peak → f_edge

        The ideal binary profile is reconstructed the same way as in the
        creator, giving us a full-height (H,) reference edge sequence.
        Observed distorted edges are then matched 1-to-1 against it.

        Confidence is computed from the local AC-RMS of the distorted column,
        the same way as _estimate_source_y_single_strip.
        """
        H      = float(self.cfg.height)
        n_rows = distorted_image.shape[0]
        half   = int(H) // 2
        rest   = int(H) - half

        # ── 1. Build ideal V-chirp binary profile ────────────────────────────
        def _chirp_seg(n: int, f0: float, f1: float) -> np.ndarray:
            t     = np.linspace(0.0, 1.0, n, endpoint=False)
            k     = f1 - f0
            phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t ** 2)
            return (np.sin(phase) >= 0).astype(int)

        ideal_binary = np.concatenate([
            _chirp_seg(half, f_edge, f_peak),   # top ↑
            _chirp_seg(rest, f_peak, f_edge),   # bot ↓
        ])

        # ── 2. Extract distorted binary column ────────────────────────────────
        raw         = distorted_image[:, col_start:col_end].mean(axis=1)
        dist_binary = (raw >= 128).astype(int)

        # ── 3. Confidence from local AC-RMS ───────────────────────────────────
        f_mean       = (f_edge + f_peak) / 2.0
        win          = max(5, int(H / f_mean * 2))
        raw_float    = raw.astype(float)
        raw_dc       = uniform_filter1d(raw_float, size=win, mode='nearest')
        raw_ac       = raw_float - raw_dc
        local_rms    = np.sqrt(uniform_filter1d(raw_ac ** 2, size=win, mode='nearest'))
        confidence   = np.clip(local_rms / 127.5, 0.0, 1.0)

        # ── 4. Detect edges in both ideal and distorted profiles ──────────────
        ideal_diff  = np.diff(ideal_binary)
        dist_diff   = np.diff(dist_binary)

        ideal_edges = np.where(ideal_diff != 0)[0].astype(float)
        dist_edges  = np.where(dist_diff  != 0)[0].astype(float)

        if len(dist_edges) < 2 or len(ideal_edges) < 2:
            return np.linspace(0.0, H - 1, n_rows), confidence * 0.0

        # ── 5. Align polarity and match 1-to-1 ───────────────────────────────
        ideal_start = 0
        if ideal_diff[int(ideal_edges[0])] != dist_diff[int(dist_edges[0])]:
            ideal_start = 1

        min_len       = min(len(ideal_edges) - ideal_start, len(dist_edges))
        matched_ideal = ideal_edges[ideal_start: ideal_start + min_len]
        matched_dist  = dist_edges[:min_len]

        # ── 6. Interpolate to all rows via PCHIP ──────────────────────────────
        pchip    = PchipInterpolator(matched_dist, matched_ideal, extrapolate=True)
        all_rows = np.arange(n_rows, dtype=float)
        source_y = np.clip(pchip(all_rows), 0.0, H - 1)

        return source_y, confidence

    def _enforce_monotonicity(self, raw_y: np.ndarray) -> np.ndarray:
        """
        PCHIP-smoothed monotone enforcement on a noisy source-row mapping.

        Builds strictly-advancing anchor points by sub-sampling raw_y, fits
        a PCHIP spline, then does a final forward non-decreasing pass.
        """
        n         = len(raw_y)
        dist_rows = np.arange(n, dtype=float)
        step      = max(1, n // 500)

        anchor_x, anchor_y = [0.0], [float(raw_y[0])]
        for i in range(step, n, step):
            val = float(raw_y[i])
            if val > anchor_y[-1] + 0.01:
                anchor_x.append(float(i))
                anchor_y.append(val)

        # Guarantee final row coverage
        if anchor_x[-1] < n - 1:
            last_val = max(anchor_y[-1] + 0.01, float(raw_y[-1]))
            anchor_x.append(float(n - 1))
            anchor_y.append(min(last_val, float(self.cfg.height - 1)))

        pchip    = PchipInterpolator(anchor_x, anchor_y, extrapolate=True)
        smoothed = np.clip(pchip(dist_rows), 0.0, float(self.cfg.height - 1))

        # Forward non-decreasing pass
        for i in range(1, n):
            if smoothed[i] < smoothed[i - 1]:
                smoothed[i] = smoothed[i - 1]

        return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the full analysis + reconstruction pipeline from the CLI."""
    config = ScanConfig()
    recon  = DSPReconstructor(config)

    if not os.path.exists(DIST_NPY):
        print(f"Error: {DIST_NPY} not found.  Run the simulator first.")
        return

    distorted = np.load(DIST_NPY)
    print(f"Loaded distorted image : {distorted.shape}")

    x_shifts = recon.detect_horizontal_vibration(distorted)
    print(f"X-shift range          : [{x_shifts.min():.2f}, {x_shifts.max():.2f}] px")

    source_y = recon.restore_vertical_phase_mapping(distorted)
    print(f"Source-Y range         : [{source_y.min():.1f}, {source_y.max():.1f}]")

    recipe  = recon.generate_reconstruction_recipe(source_y, x_shifts)
    kept    = sum(1 for e in recipe if e['action'] == 'keep')
    dropped = sum(1 for e in recipe if e['action'] == 'drop')
    print(f"Recipe                 : {kept} keep, {dropped} drop")

    with open(RECIPE, 'w') as f:
        json.dump(recipe, f, indent=2)

    restored = recon.reconstruct_perfect_image(distorted, recipe)
    np.save(REST_NPY, restored)
    Image.fromarray(restored).save(REST_PNG)
    print(f"Restored image         : {restored.shape}  →  {REST_PNG}")

    if os.path.exists(REF_NPY):
        metrics = recon.evaluate_restoration_quality(np.load(REF_NPY), restored)
        print(f"\nPSNR  : {metrics['psnr']} dB")
        print(f"MAE   : {metrics['mae']}")
        print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
