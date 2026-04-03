"""
analyzer.py — Programmer 3: The DSP Reconstructor (Rotation-Tag Edition)
=========================================================================
Uses the rotation-tag ladder embedded in the reference image to recover
the Y-axis distortion map:
  1. decode_rotation_angle  → best-matching pre-slanted strip (lowest CV)
  2. build_y_map_from_signal → PCHIP Y mapping from that strip's edge sequence
  3. reconstruct_perfect_image → remap distorted rows back to ideal grid

No X-tracker or chirp strips required.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from PIL import Image
import json, os, math

from src.config import (ScanConfig,
                        DIST_NPY, REF_NPY, REST_NPY, REST_PNG,
                        RECIPE, DIFF_PNG, REPORT)
from src.rotation_tag import (build_layout, decode_rotation_angle,
                               build_y_map_from_signal,
                               TOTAL_RIGHT_WIDTH, CHIRP_F0, CHIRP_FREQ)


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

    # ── Step 1: Horizontal vibration from right-side X-tracker ───────────────

    def detect_horizontal_vibration(self, dist_img: np.ndarray, y_map: np.ndarray) -> np.ndarray:
        """Adaptive centroid extractor reading the right-side X-tracker strip.

        Scales margins and thresholds with band width so it stays robust.
        Dynamically follows the tilted tracking strip if a physical rotation
        was decoded in the Y-phase step. Uses the Y-map to precisely predict
        the slant displacement against non-linear slip/skip.
        """
        import math
        cfg    = self.cfg
        n, W_img = dist_img.shape
        shifts = np.full(n, np.nan)
        
        orig_cx = cfg.x_tracker_expected_center
        bw      = cfg.col1_end - cfg.col1_start

        safe_margin  = max(1, bw // 10)
        refine       = max(3, bw // 2)
        thresh_ratio = min(0.5, max(0.3, 0.3 + 0.2 * (bw - 8) / 92.0))

        physical_cw_deg = getattr(self, 'physical_cw_deg', 0.0)
        theta_rad = math.radians(physical_cw_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        img_cx, img_cy = W_img / 2.0, n / 2.0

        for r in range(n):
            # To predict the tracking line's X displacement, we MUST use the physical 
            # Y coordinate of the scan, not the raw row index `r`.
            # y_map[r] gives the physical unrotated Y coordinate.
            phys_y = y_map[r] if y_map is not None else float(r)
            
            # Since y_map[r] was measured at the rotation tag, we adjust it perfectly
            # for the X-tracker's horizontal position. (optional but highly rigorous)
            # The right rotation layout tag was approximately at tag_x. But practically, 
            # the offset difference is tiny. We use phys_y directly as it represents 
            # the center-axis or tag-axis Y-time.
            # Actually, because the Y-map returned from analyzer includes the rotational 
            # y_shift of the tag, y_map[r] is exactly the physical Y at the tag.
            # For the X tracker, the slant is from img_cy. Let's just use phys_y:
            
            dyn_cx = img_cx + (orig_cx - img_cx) * cos_t - (phys_y - img_cy) * sin_t
            
            s = int(math.floor(dyn_cx - bw / 2.0))
            e = int(math.ceil(dyn_cx + bw / 2.0))
            
            # Clip bounds
            s = max(0, min(W_img - 1, s))
            e = max(0, min(W_img - 1, e))

            if e - s < 5:
                continue

            win = dist_img[r, s:e].astype(float)

            if bw <= 8:
                signal = win
                coords = np.arange(s, e, dtype=float)
                thresh = signal.max() * 0.3
                mass   = signal * (signal >= thresh)
                tot    = mass.sum()
                if tot > 0 and signal.max() > 80:
                    shifts[r] = (coords * mass).sum() / tot - dyn_cx
                continue

            inner = win[safe_margin:-safe_margin] if safe_margin < bw // 2 else win
            anchor_local = int(np.argmax(inner))
            anchor = anchor_local + s + (safe_margin if safe_margin < bw // 2 else 0)

            if win[anchor - s] < 128:
                continue

            lo  = max(s, anchor - refine)
            hi  = min(e, anchor + refine + 1)
            sub = dist_img[r, lo:hi].astype(float)
            signal = sub
            coords = np.arange(lo, hi, dtype=float)
            thresh = signal.max() * thresh_ratio
            mass   = signal * (signal >= thresh)
            tot    = mass.sum()
            if tot > 0:
                shifts[r] = (coords * mass).sum() / tot - dyn_cx

        valid = ~np.isnan(shifts)
        if valid.sum() < 2:
            return np.zeros(n)
        idx    = np.arange(n)
        filled = np.interp(idx, idx[valid], shifts[valid])
        return median_filter(filled, size=5, mode='nearest')

    # ── Step 2: Vertical phase mapping via Rotation-Tag ladder ────────────────

    def restore_vertical_phase_mapping(self, dist_img: np.ndarray) -> np.ndarray:
        import cv2
        cfg   = self.cfg
        H     = dist_img.shape[0]
        REF_H = cfg.height

        # ── 2a: Decode physical rotation from the tag ladder ──────────────────
        layout = build_layout(
            left_x0=200,
            right_x0=cfg.width - TOTAL_RIGHT_WIDTH - 350 - 200
        )
        angle, best_signal, score = decode_rotation_angle(
            dist_img, layout, verbose=False
        )
        print(f"      Rotation tag decode: angle={angle:+.2f}°  var={score:.0f}")

        # ── 2b: Remove physical tilt from Y-map calculation ───────────────────
        # Identify the physical rotation (angle_tag is negative of physical CW tilt)
        self.physical_cw_deg = -angle
        print(f"      Mapped physical tilt geometry to: {self.physical_cw_deg:+.2f}° CW.")
        
        H, W = dist_img.shape
        cx = W / 2.0
        
        tag_x = cx  # default
        for col_start, col_end, strip_angle in layout.strips:
            if abs(strip_angle - angle) < 1e-4:
                tag_x = (col_start + col_end) / 2.0
                break
        
        import math
        # OpenCV CW rotation: Y' = Y + (X - cx) * sin(theta_cw)
        theta_cw_rad = math.radians(self.physical_cw_deg)
        y_shift = (tag_x - cx) * math.sin(theta_cw_rad)
        print(f"      Calculated Y-shift for {angle:+.2f}° tag: {y_shift:+.1f} px")

        # We use the best_signal from dist_img because it was perfectly vertical 
        # there (blurred only by true X-vibration).
        raw_map = build_y_map_from_signal(best_signal, f0=CHIRP_F0, f1=CHIRP_FREQ, y_shift=y_shift)
        scale   = (REF_H - 1) / max(1, len(best_signal) - 1)
        y_map   = np.clip(raw_map * scale, 0, REF_H - 1)

        return self._mono(y_map)


    # _match_edges kept for potential future use but no longer called in main path
    def _match_edges(self, db, ib, expected_interval: float = 20.0):
        """(Legacy) Slide the ideal edge sequence to find the best polarity-aligned start."""
        de = np.where(np.diff(db) != 0)[0].astype(float)
        ie = np.where(np.diff(ib) != 0)[0].astype(float)
        dd = np.diff(db)[de.astype(int)]
        id_ = np.diff(ib)[ie.astype(int)]
        if len(de) < 5 or len(ie) < 5:
            return [], []
        max_search = max(40, int(3 * expected_interval))
        max_search = min(max_search, len(ie) - 5)
        n_score_pairs = 5
        best_score, best_off = -1.0, 0
        for off in range(max_search):
            if id_[off] != dd[0]:
                continue
            m = min(n_score_pairs, len(de), len(ie) - off)
            if m < 3:
                continue
            interval_err = np.sum(np.abs(np.diff(de[:m]) - np.diff(ie[off:off+m])))
            dist_bias = np.abs(ie[off] - de[0]) / 5000.0
            score = 1.0 / ((interval_err / expected_interval) + dist_bias + 1e-6)
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

        from PIL import Image
        Image.fromarray(out).save('data/straightened_rotated.png')
        physical_cw_deg = getattr(self, 'physical_cw_deg', 0.0)
        if abs(physical_cw_deg) > 0.05:
            import cv2
            H_out, W_out = out.shape
            cx, cy = W_out / 2.0, H_out / 2.0
            # To cancel a physical CW tilt, rotate by CCW length.
            # cv2 positive angle rotates CCW.
            M = cv2.getRotationMatrix2D((cx, cy), physical_cw_deg, 1.0)
            out = cv2.warpAffine(out, M, (W_out, H_out), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            print(f"      Reconstruction geometrically derotated by {physical_cw_deg:+.2f}° CCW.")

        return out

    # ── Step 5: Quality ───────────────────────────────────────────────────────

    def evaluate_restoration_quality(self, original, restored):
        # Evaluate over the full image (rotation tags only, no chirp strips).
        # Crop to the smaller of the two heights if they differ.
        min_h = min(original.shape[0], restored.shape[0])
        o_crop = original[:min_h, :]
        r_crop = restored[:min_h, :]

        diff = np.abs(o_crop.astype(float) - r_crop.astype(float))
        mse  = np.mean(diff ** 2)
        psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99.0
        mae  = float(diff.mean())
        m    = {'psnr': round(psnr, 2), 'mae': round(mae, 3)}
        with open(REPORT, 'w') as f:
            f.write(f"PSNR: {m['psnr']} dB\nMAE: {m['mae']}\n")
        return m
