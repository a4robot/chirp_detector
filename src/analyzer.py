import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import PchipInterpolator
from PIL import Image
import json
import os
import math
from src.config import ScanConfig, DIST_NPY, REF_NPY, REST_NPY, REST_PNG, RECIPE, DIFF_PNG, REPORT

class DSPReconstructor:
    """The Analyzer: professional DSP pipeline to detect and undo mechanical distortions."""

    def __init__(self, config: ScanConfig):
        self.config = config

    def detect_horizontal_vibration(self, distorted_image: np.ndarray) -> np.ndarray:
        """Step 1: Recover sub-pixel horizontal shifts for every row."""
        n_rows = distorted_image.shape[0]
        shifts = np.full(n_rows, np.nan)

        for r in range(n_rows):
            # Centroid detection logic
            cx = int(round(self.config.x_tracker_expected_center))
            lo = max(0, cx - 20)
            hi = min(len(distorted_image[r]), cx + 21)
            window = distorted_image[r, lo:hi].astype(float)

            if window.max() < 30: continue
            
            thresh = window.max() * 0.5
            mask = window >= thresh
            coords = np.arange(lo, hi)
            mass = window * mask
            total = mass.sum()
            if total > 0:
                shifts[r] = float((coords * mass).sum() / total) - self.config.x_tracker_expected_center

        # Gap-fill and interpolate
        valid = ~np.isnan(shifts)
        if valid.sum() < 2:
            return np.zeros(n_rows)
        
        idx = np.arange(n_rows)
        return np.interp(idx, idx[valid], shifts[valid])

    def restore_vertical_phase_mapping(self, distorted_image: np.ndarray) -> np.ndarray:
        """Step 2: Use phase tracking to map distorted rows into ideal 1000-line signal."""
        chirp = distorted_image[:, self.config.y_chirp_start:self.config.y_chirp_end].mean(axis=1).astype(float)
        
        # Build ideal phase
        t = np.linspace(0, 1, self.config.height, endpoint=False)
        ideal = (np.sin(2 * np.pi * self.config.y_chirp_cycles * t) + 1) / 2 * 255
        
        # Phase-inversion via Hilbert
        phase_dist  = np.unwrap(np.angle(hilbert(chirp - chirp.mean())))
        phase_ideal = np.unwrap(np.angle(hilbert(ideal - ideal.mean())))
        
        # Raw inverse interpolation
        ideal_rows = np.arange(self.config.height, dtype=float)
        source_y_raw = np.interp(phase_dist, phase_ideal, ideal_rows, left=0.0, right=float(self.config.height - 1))
        
        # Monotone smoothing via PCHIP
        return self._enforce_monotonicity(source_y_raw)

    def generate_reconstruction_recipe(self, source_y: np.ndarray, 
                                     x_shifts: np.ndarray, 
                                     min_advance: float = 0.1) -> list:
        """Step 3: Classify each row (keep/drop) and package with reconstruction info."""
        recipe = []
        last_kept_y = -1.0
        for i, (sy, sx) in enumerate(zip(source_y, x_shifts)):
            action = 'keep' if (sy - last_kept_y >= min_advance) else 'drop'
            if action == 'keep': last_kept_y = sy
            
            recipe.append({
                'distorted_row': i,
                'source_y': round(float(sy), 4),
                'action': action,
                'shift_x': round(float(sx), 4)
            })
        return recipe

    def reconstruct_perfect_image(self, distorted_image: np.ndarray, 
                                 recipe: list) -> np.ndarray:
        """Step 4: Execute the 'Restoration Recipe' to fix vibration and interpolate gaps."""
        width = distorted_image.shape[1]
        kept_source_y, kept_lines = [], []

        for entry in recipe:
            if entry['action'] != 'keep': continue
            r, sx = entry['distorted_row'], entry['shift_x']
            # Sub-pixel horizontal correction
            orig_x = np.arange(width, dtype=float)
            corrected = np.interp(orig_x + sx, orig_x, distorted_image[r].astype(float), left=0, right=0)
            kept_source_y.append(entry['source_y'])
            kept_lines.append(corrected)

        # Full Vertical Resampling
        target_y = np.linspace(0, self.config.height - 1, self.config.height)
        kept_lines = np.array(kept_lines)
        restored = np.zeros((self.config.height, width), dtype=np.uint8)

        for col in range(width):
            restored[:, col] = np.clip(np.interp(target_y, kept_source_y, kept_lines[:, col]), 0, 255).astype(np.uint8)

        return restored

    def evaluate_restoration_quality(self, original: np.ndarray, 
                                    restored: np.ndarray) -> dict:
        """Step 5: Validation: Report PSNR and save diff artifacts."""
        diff = np.abs(original.astype(int) - restored.astype(int)).astype(np.uint8)
        Image.fromarray(np.clip(diff * 4, 0, 255).astype(np.uint8)).save(DIFF_PNG)

        mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
        psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        
        metrics = {'psnr': round(psnr, 3), 'mae': round(float(diff.mean()), 4)}
        with open(REPORT, 'w') as f:
            f.write(f"PSNR: {metrics['psnr']} dB\nMAE:  {metrics['mae']}\n")
        return metrics

    def _enforce_monotonicity(self, raw_y: np.ndarray) -> np.ndarray:
        """Internal helper for robust PCHIP-based monotone smoothing."""
        dist_rows = np.arange(len(raw_y), dtype=float)
        step = max(1, len(raw_y) // 500)
        anchor_x, anchor_y = [0.0], [raw_y[0]]
        for i in range(step, len(raw_y), step):
            if raw_y[i] > anchor_y[-1] + 0.01:
                anchor_x.append(float(i)); anchor_y.append(raw_y[i])
        
        pchip = PchipInterpolator(anchor_x, anchor_y, extrapolate=True)
        smoothed = np.clip(pchip(dist_rows), 0, float(self.config.height - 1))
        # Final non-decreasing pass
        for i in range(1, len(smoothed)):
            if smoothed[i] < smoothed[i-1]: smoothed[i] = smoothed[i-1]
        return smoothed

def main():
    """CLI to analyze and reconstruct image."""
    config = ScanConfig()
    recon = DSPReconstructor(config)
    
    distorted = np.load(DIST_NPY)
    # Pipeline
    x_shifts = recon.detect_horizontal_vibration(distorted)
    source_y = recon.restore_vertical_phase_mapping(distorted)
    recipe   = recon.generate_reconstruction_recipe(source_y, x_shifts)
    restored = recon.reconstruct_perfect_image(distorted, recipe)
    
    np.save(REST_NPY, restored)
    Image.fromarray(restored).save(REST_PNG)
    with open(RECIPE, 'w') as f:
        json.dump(recipe, f, indent=2)

    if os.path.exists(REF_NPY):
        metrics = recon.evaluate_restoration_quality(np.load(REF_NPY), restored)
        print(f"Restoration Complete: PSNR={metrics['psnr']} dB")

if __name__ == "__main__":
    main()
