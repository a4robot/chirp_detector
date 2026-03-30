"""
analyzer.py — Programmer 3: The DSP Reconstructor (Point-Pool Binary Fusion)
=============================================================================
Implements the "Point-Pool" decoder:
  1. Pools all binary edges from 4 staggered strips into one coordinate list.
  2. Fits a single master PCHIP for the vertical motion map.
  3. Robust sliding-window edge alignment handles initial slip/skip.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from PIL import Image
import json, os, math

from src.config import (ScanConfig,
                        DIST_NPY, REF_NPY, REST_NPY, REST_PNG,
                        RECIPE, DIFF_PNG, REPORT)


class DSPReconstructor:

    def __init__(self, config: ScanConfig):
        self.cfg = config

    def _pava(self, x):
        """
        Pool Adjacent Violators Algorithm (PAVA)
        Strictly monotonic non-decreasing isotonic regression.
        """
        n = x.shape[0]
        v = np.copy(x)
        lvl = np.arange(n)
        w = np.ones(n)
        while True:
            diff = np.diff(v)
            if np.all(diff >= 0):
                break
            i = np.where(diff < 0)[0][0]
            # Pool i and i+1
            v_new = (w[i] * v[i] + w[i+1] * v[i+1]) / (w[i] + w[i+1])
            v[i] = v_new
            w[i] += w[i+1]
            v = np.delete(v, i+1)
            w = np.delete(w, i+1)
            
            # Expand the pooled value back to the representative indices if needed,
            # but simpler is to reconstruct the full array at the end from weights.
            # Let's use a more standard iterative approach for robustness.
            
        # Re-expand pooled values
        # (Actually, a simpler O(n) stack-based PAVA is better)
        return self._pava_stack(x)

    def _pava_stack(self, y):
        n = len(y)
        if n == 0: return y
        # Stack stores (value, weight, count)
        # We use count to know how many elements this pool represents
        stack = [] # list of [sum_y, weight]
        for val in y:
            cur_sum = float(val)
            cur_weight = 1.0
            while stack and (stack[-1][0]/stack[-1][1] > cur_sum/cur_weight):
                prev_sum, prev_weight = stack.pop()
                cur_sum += prev_sum
                cur_weight += prev_weight
            stack.append((cur_sum, cur_weight))
        
        # Expand
        res = np.zeros(n)
        idx = 0
        for s, w in stack:
            avg = s / w
            count = int(round(w))
            res[idx:idx+count] = avg
            idx += count
        return res

    # ── Step 1: Horizontal vibration from X-ref line ─────────────────────────

    def detect_horizontal_vibration(self, dist_img: np.ndarray) -> np.ndarray:
        cfg    = self.cfg
        n      = dist_img.shape[0]
        shifts = np.full(n, np.nan)
        cx     = cfg.x_tracker_expected_center
        s, e   = cfg.col1_start, cfg.col1_end   # 0, 100
        refine = 12

        for r in range(n):
            win = dist_img[r, s:e].astype(float)
            # anchor: darkest pixel in inner safe window (avoid bleed at edges)
            safe_margin = max(2, min(15, (e - s - 10) // 2))
            safe_s = s + safe_margin
            safe_e = e - safe_margin
            anchor = int(np.argmin(win[safe_margin:-safe_margin])) + safe_s
            if win[anchor - s] > 128:
                continue
            lo = max(s, anchor - refine)
            hi = min(e, anchor + refine + 1)
            sub    = dist_img[r, lo:hi].astype(float)
            inv    = 255.0 - sub
            coords = np.arange(lo, hi, dtype=float)
            thresh = inv.max() * 0.5
            mass   = inv * (inv >= thresh)
            tot    = mass.sum()
            if tot > 0:
                shifts[r] = (coords * mass).sum() / tot - cx

        valid = ~np.isnan(shifts)
        if valid.sum() < 2:
            return np.zeros(n)
        idx    = np.arange(n)
        filled = np.interp(idx, idx[valid], shifts[valid])
        return median_filter(filled, size=5, mode='nearest')

    # ── Step 2: Vertical phase mapping via unified Point-Pool ─────────────────

    def restore_vertical_phase_mapping(self, dist_img: np.ndarray) -> np.ndarray:
        cfg    = self.cfg
        H      = cfg.height
        t_ref  = np.linspace(0.0, 1.0, H, endpoint=False)
        all_pts = []

        for cs, ce, mode, phase, f1_strip in cfg.strips:
            f0, f1 = cfg.f0, f1_strip
            if mode == "rev":
                f0, f1 = f1, f0
            k       = f1 - f0
            # Build the ideal model using THIS strip's own f1_strip chirp rate.
            # This is the key fix: incommensurable Vernier strips each get their
            # own ideal interval sequence, preventing cross-contamination in the
            # sliding-window alignment step.
            p_ideal = 2.0 * np.pi * (f0 * t_ref + 0.5 * k * t_ref ** 2) + phase
            ideal   = (np.sin(p_ideal) >= 0).astype(int)

            raw  = dist_img[:, cs:ce].mean(axis=1)
            dist_col = (raw >= 128).astype(int)

            # Pass the expected mean interval for this strip so _match_edges
            # can use a frequency-appropriate search window.
            expected_interval = H / (f1_strip + f0)  # approx mean half-period
            de, ie = self._match_edges(dist_col, ideal, expected_interval)
            for d, i in zip(de, ie):
                all_pts.append((float(d), float(i)))

        if len(all_pts) < 10:
            return np.linspace(0, H - 1, dist_img.shape[0])

        # Pool → average by distorted-row (D) → Isotonic (PAVA) → PCHIP
        bucket: dict = {}
        for d, i in all_pts:
            bucket.setdefault(d, []).append(i)
        
        s_d = np.array(sorted(bucket.keys()))
        s_i_raw = np.array([float(np.mean(bucket[k])) for k in s_d])

        # Step 4: Isotonic Regression (Mathematical Upgrade)
        # Instead of RANSAC windows, we enforce the physical law of monotonicity globally.
        s_i_iso = self._pava_stack(s_i_raw)

        # To keep PCHIP happy, we need STRICT monotonicity (non-zero derivatives)
        # and we filter out points that were heavily "pooled" (outliers).
        clean_d, clean_i = [], []
        
        if len(s_d) > 0:
            clean_d.append(s_d[0])
            clean_i.append(s_i_iso[0])
            for j in range(1, len(s_d)):
                # If isotonic regression flattened this point into its neighbor,
                # it means it was a violator/noise. We only keep points that
                # maintain a minimum positive delta to ensure sub-pixel curve fit.
                if s_i_iso[j] > clean_i[-1] + 1e-5:
                    clean_d.append(s_d[j])
                    clean_i.append(s_i_iso[j])

        if len(clean_d) < 5:
            return np.linspace(0, H - 1, dist_img.shape[0])

        pchip  = PchipInterpolator(clean_d, clean_i, extrapolate=True)
        rows   = np.arange(dist_img.shape[0], dtype=float)
        sy     = np.clip(pchip(rows), 0, H - 1)
        return self._mono(sy)

    def _match_edges(self, db, ib, expected_interval: float = 20.0):
        """Slide the ideal edge sequence to find the best polarity-aligned start.

        The search window is now calibrated to the per-strip chirp rate:
        - For a low-frequency strip (large expected_interval), we look further.
        - For a high-frequency strip (small expected_interval), we look tighter.
        This prevents Vernier strips with very different f1_strip from being
        mis-aligned against a shared global ideal sequence.
        """
        de = np.where(np.diff(db) != 0)[0].astype(float)
        ie = np.where(np.diff(ib) != 0)[0].astype(float)
        dd = np.diff(db)[de.astype(int)]
        id_ = np.diff(ib)[ie.astype(int)]
        if len(de) < 5 or len(ie) < 5:
            return [], []

        # Scale search window to at least 3 full expected half-periods so we
        # can find the correct polarity start even with a large initial slip.
        max_search = max(40, int(3 * expected_interval))
        max_search = min(max_search, len(ie) - 5)

        # Scoring uses a small local window (n=5) for robustness.
        # We normalize interval errors by the expected interval to make the
        # scoring frequency-invariant (otherwise high-freq strips always win).
        n_score_pairs = 5

        best_score, best_off = -1.0, 0
        for off in range(max_search):
            if id_[off] != dd[0]:          # polarity gate: must start same direction
                continue
            m = min(n_score_pairs, len(de), len(ie) - off)
            if m < 3:
                continue
            
            # Core Score: Inverse of Normalized Interval Error.
            # We add a tiny 'minimal-drift' bias (dist/5000) to break ties 
            # in periodic low-frequency signals (e.g. Strip 0 in Vernier).
            # This prevents jumping to a different chirp period if intervals are similar.
            interval_err = np.sum(np.abs(np.diff(de[:m]) - np.diff(ie[off:off+m])))
            dist_bias = np.abs(ie[off] - de[0]) / 5000.0
            score = 1.0 / ( (interval_err / expected_interval) + dist_bias + 1e-6 )
            
            if score > best_score:
                best_score, best_off = score, off

        m_len = min(len(de), len(ie) - best_off)
        return de[:m_len], ie[best_off: best_off + m_len]

    def _mono(self, sy):
        for i in range(1, len(sy)):
            if sy[i] < sy[i - 1]:
                sy[i] = sy[i - 1]
        return sy

    # ── Step 3 & 4: Recipe + Reconstruction ──────────────────────────────────

    def generate_reconstruction_recipe(self, sy, sx):
        recipe, last = [], -1.0
        for i, (y, x) in enumerate(zip(sy, sx)):
            ok = y - last >= 0.1
            if ok:
                last = y
            recipe.append({'distorted_row': i, 'source_y': float(y),
                            'action': 'keep' if ok else 'drop',
                            'shift_x': float(x)})
        return recipe

    def reconstruct_perfect_image(self, dist_img, recipe):
        W  = dist_img.shape[1]
        ox = np.arange(W, dtype=float)
        ks, kl = [], []
        for e in recipe:
            if e['action'] != 'keep':
                continue
            row = dist_img[e['distorted_row']].astype(float)
            kl.append(np.interp(ox + e['shift_x'], ox, row, left=row[0], right=row[-1]))
            ks.append(e['source_y'])
        ka = np.array(kl)
        ty = np.linspace(0, self.cfg.height - 1, self.cfg.height)
        out = np.zeros((self.cfg.height, W), dtype=np.uint8)
        for c in range(W):
            out[:, c] = np.clip(np.interp(ty, ks, ka[:, c]), 0, 255).astype(np.uint8)
        return out

    # ── Step 5: Quality ───────────────────────────────────────────────────────

    def evaluate_restoration_quality(self, original, restored):
        diff = np.abs(original.astype(float) - restored.astype(float))
        mse  = np.mean(diff ** 2)
        psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99.0
        mae  = float(diff.mean())
        m    = {'psnr': round(psnr, 2), 'mae': round(mae, 3)}
        with open(REPORT, 'w') as f:
            f.write(f"PSNR: {m['psnr']} dB\nMAE: {m['mae']}\n")
        return m
