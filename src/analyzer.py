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
import cv2
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
        """Flood-fill X-tracker.

        1. Threshold a generous strip around the predicted center.
        2. Flood-fill from the center seed to extract the connected white blob.
        3. Per-row centroid of that blob → per-row shift.
        4. Interpolate + median-smooth, save a full debug overlay.
        """
        import math
        import cv2
        cfg      = self.cfg
        n, W_img = dist_img.shape

        global_x_shift  = getattr(self, 'global_x_shift', 0)
        physical_cw_deg = getattr(self, 'physical_cw_deg', 0.0)
        theta_rad       = math.radians(physical_cw_deg)
        bw              = cfg.col1_end - cfg.col1_start

        # ── Anchor: Dynamically locate x-tracker starting column ─────────────
        # Instead of relying on global_x_shift (which fails if left tags are clipped),
        # we cross-correlate a 1D template of the expected x-tracker [black, white, black]
        # signature against the row-averaged top region of the image.
        
        _H = min(2000, n)
        profile_2d = np.mean(dist_img[:_H, :], axis=0).astype(np.float32).reshape(1, -1)
        
        bw_full = cfg.col1_end - cfg.col1_start
        template = np.zeros(bw_full, dtype=np.float32)
        start_w = cfg.x_line_start - cfg.col1_start
        end_w   = cfg.x_line_end   - cfg.col1_start
        template[start_w:end_w] = 255.0
        
        res = cv2.matchTemplate(profile_2d, template.reshape(1, -1), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.5:
            center_offset = cfg.x_tracker_expected_center - cfg.col1_start
            x_center0 = max_loc[0] + center_offset
            print(f"      X-Tracker anchor dynamically found at x={x_center0:.1f} (match={max_val:.2f})")
        else:
            x_center0 = cfg.x_tracker_expected_center + global_x_shift
            print(f"      X-Tracker search failed (match={max_val:.2f}), using geometry anchor x={x_center0:.1f}")

        tan_theta  = math.tan(theta_rad)

        def expected_cx(row: int) -> float:
            # Positive CW tilt means top-right → bottom-left slant.
            # dx/dr = -tan(theta) for a CW angle.
            return x_center0 - row * tan_theta

        # ── Early exit: tracker strip center is outside this (cropped) image ───
        mid_cx = expected_cx(n // 2)
        strip_hw = max(bw // 2, 10)
        if mid_cx - strip_hw >= W_img or mid_cx + strip_hw <= 0:
            print(f"      X-Tracker: strip off-screen (x={mid_cx:.0f}, W={W_img}) — zero shifts")
            return np.zeros(n)
        # Use a generous half-width so the blob can wobble freely
        half_strip = max(bw * 3, 60)

        # Compute column bounds for each row
        col_lo = np.zeros(n, dtype=int)
        col_hi = np.zeros(n, dtype=int)
        for r in range(n):
            cx = expected_cx(r)
            col_lo[r] = max(0, int(cx - half_strip))
            col_hi[r] = min(W_img, int(cx + half_strip))

        # ── Threshold the strip region ────────────────────────────────────────
        # Work on the full-size mask so flood-fill connectivity is reliable
        thresh_val = 160          # white strip should be well above this
        bin_mask = np.zeros((n, W_img), dtype=np.uint8)
        for r in range(n):
            row_slice = dist_img[r, col_lo[r]:col_hi[r]]
            bright = (row_slice >= thresh_val).astype(np.uint8) * 255
            bin_mask[r, col_lo[r]:col_hi[r]] = bright

        # Bridge small horizontal gaps (dust, noise, signal dropout)
        # using a vertical closing morphological operation.
        kernel = np.ones((25, 1), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        # ── Flood-fill from seed ──────────────────────────────────────────────
        # Find a seed point near the vertical mid-point of the image.
        # Since physical scans can bend, we use the same robust 1D template match
        # to find the tracker exactly at the middle row!
        seed_row = n // 2
        found_seed = False
        
        # Take a 100-row slice around the middle to average out noise
        r_start = max(0, seed_row - 50)
        r_end   = min(n, seed_row + 50)
        mid_profile_2d = np.mean(dist_img[r_start:r_end, :], axis=0).astype(np.float32).reshape(1, -1)
        
        res_mid = cv2.matchTemplate(mid_profile_2d, template.reshape(1, -1), cv2.TM_CCOEFF_NORMED)
        _, max_val_mid, _, max_loc_mid = cv2.minMaxLoc(res_mid)
        
        if max_val_mid > 0.5:
            seed_col = int(max_loc_mid[0] + center_offset)
            found_seed = True
            print(f"      X-Tracker seed found dynamically at row {seed_row}, col {seed_col} (match={max_val_mid:.2f})")
        else:
            print(f"      X-Tracker seed dynamic search failed (match={max_val_mid:.2f})")
            
        flood_flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        ff_mask = np.zeros((n + 2, W_img + 2), dtype=np.uint8)
        if found_seed:
            cv2.floodFill(bin_mask, ff_mask,
                          seedPoint=(seed_col, seed_row),
                          newVal=128,
                          loDiff=0, upDiff=0,
                          flags=flood_flags)
        # ff_mask interior (1:-1, 1:-1) is 255 where filled
        filled_blob = ff_mask[1:-1, 1:-1]

        # ── Per-row centroid of the blob ──────────────────────────────────────
        shifts    = np.full(n, np.nan)
        blob_cols = np.full(n, np.nan)  # absolute detected center

        for r in range(n):
            row = filled_blob[r, col_lo[r]:col_hi[r]]
            if row.any():
                cols = np.where(row)[0] + col_lo[r]
                # Sanity: blob row must be narrow (≤ 3×bw) to be the tracking strip
                blob_width = int(cols.max() - cols.min() + 1)
                if blob_width > bw * 3:
                    continue  # wrong blob (noise region spilling in)
                cx_detected = float(cols.mean())
                blob_cols[r] = cx_detected
                shifts[r]    = cx_detected - expected_cx(r)

        valid = ~np.isnan(shifts)
        if valid.sum() < 2:
            res_for_dbg = np.zeros(n)
            print("      X-Tracker: flood-fill found no connected blob — returning zero shifts")
        else:
            # Re-fit the global physical tilt from the continuous X-Tracker!
            # The X-Tracker slope dictates the absolute physical geometry perfectly.
            valid_idx = np.where(valid)[0]
            m, c = np.polyfit(valid_idx, blob_cols[valid_idx], 1)
            
            # The red expected line now perfectly tracks the physical geometry
            def expected_cx(row: int) -> float:
                return float(m * row + c)
            
            # Re-compute shifts as purely the high-frequency vibration deviation
            for r in valid_idx:
                shifts[r] = blob_cols[r] - expected_cx(r)
                
            # physical_cw_deg: dx/dr = -tan(theta) for a CW tilt
            new_tilt_deg = math.degrees(math.atan(-m))
            print(f"      X-Tracker geometric linear fit: updated global tilt to {new_tilt_deg:+.3f}° CW")
            self.physical_cw_deg = new_tilt_deg
            
            idx         = np.arange(n)
            filled_vals = np.interp(idx, valid_idx, shifts[valid_idx])
            res_for_dbg = median_filter(filled_vals, size=21, mode='nearest')
            n_rows      = int(valid.sum())
            print(f"      X-Tracker flood-fill: tracked {n_rows}/{n} rows  "
                  f"vibration range [{res_for_dbg.min():.1f}, {res_for_dbg.max():.1f}] px")

        # ── Debug overlay ─────────────────────────────────────────────────────
        try:
            debug_w = 400
            view_cx = x_center0
            s_dbg   = max(0, int(view_cx - debug_w // 2))
            e_dbg   = min(W_img, s_dbg + debug_w)

            h_step   = max(1, n // 4000)
            dbg_rows = np.arange(0, n, h_step)

            # Background: grayscale scan crop
            dbg_gray = dist_img[::h_step, s_dbg:e_dbg].copy()
            dbg_bgr  = cv2.cvtColor(dbg_gray, cv2.COLOR_GRAY2BGR)

            # Tint flood-filled pixels cyan (B=255, G=200, R=0)
            for i, r in enumerate(dbg_rows):
                if i >= dbg_bgr.shape[0]:
                    break
                blob_row = filled_blob[r, s_dbg:e_dbg]
                dbg_bgr[i][blob_row > 0] = [200, 200, 0]   # cyan-ish tint

            # Expected center line — RED
            for i, r in enumerate(dbg_rows):
                if i >= dbg_bgr.shape[0]:
                    break
                ex_x = int(round(expected_cx(r) - s_dbg))
                if 0 <= ex_x < dbg_bgr.shape[1]:
                    dbg_bgr[i, ex_x] = [0, 0, 255]

            # Detected center (flood-fill centroid) — GREEN (3 px wide)
            for i, r in enumerate(dbg_rows):
                if i >= dbg_bgr.shape[0]:
                    break
                ac_x = int(round(expected_cx(r) + res_for_dbg[r] - s_dbg))
                if 0 <= ac_x < dbg_bgr.shape[1]:
                    x_lo = max(0, ac_x - 1)
                    x_hi = min(dbg_bgr.shape[1], ac_x + 2)
                    dbg_bgr[i, x_lo:x_hi] = [0, 255, 0]

            os.makedirs("data/debug", exist_ok=True)
            cv2.imwrite("data/debug/debug_x_tracker.png", dbg_bgr)
            print("      X-Tracker flood-fill debug saved: data/debug/debug_x_tracker.png")
        except Exception as exc:
            print(f"      (X-Tracker debug suppressed: {exc})")

        if valid.sum() < 2:
            return np.zeros(n)
        return res_for_dbg

    def apply_x_shifts_only(self, dist_img: np.ndarray, x_shifts: np.ndarray) -> np.ndarray:
        """Apply horizontal shift correction ONLY, keeping vertical distortion intact."""
        n, w = dist_img.shape
        out = np.zeros_like(dist_img)
        ox = np.arange(w, dtype=float)
        for r in range(n):
            row = dist_img[r].astype(float)
            shift = x_shifts[r]
            out[r] = np.interp(ox + shift, ox, row, left=row[0], right=row[-1]).astype(np.uint8)
        return out

    # ── Step 2: Vertical phase mapping via Rotation-Tag ladder ────────────────

    def restore_vertical_phase_mapping(self, dist_img: np.ndarray, override_x: int = None, override_width: int = None, flip_vertical: bool = False, known_angle_deg: float = None) -> np.ndarray:
        
        # ── Dynamic Scale Factor ──
        # Assume original design was 14975 px wide. Detect current scale.
        scale_factor = dist_img.shape[1] / 14975.0
        if abs(scale_factor - 1.0) > 1e-3:
            print(f"  Detected image width {dist_img.shape[1]}px. Scaling geometry by factor: {scale_factor:.4f}")
            self.cfg = self.cfg.scale(scale_factor)
            
        cfg   = self.cfg
        REF_H = cfg.height
        print("  Decoding physical rotation angle...")
        from src.rotation_tag import build_layout, get_total_right_width, decode_rotation_angle, build_y_map_from_signal, CHIRP_F0, CHIRP_FREQ
        
        # Build layout with scaled factor
        layout = build_layout(left_x0=int(round(200 * scale_factor)), 
                              right_x0=cfg.width - get_total_right_width(scale_factor) - int(round(350 * scale_factor)) - int(round(200 * scale_factor)),
                              scale_factor=scale_factor)

        # Dynamically align layout to physical scan using cv2 matchTemplate
        import cv2
        from src.config import REF_PNG
        
        # Template is taken from original scale reference, so we must scale it
        ref_img = cv2.imread(REF_PNG, cv2.IMREAD_GRAYSCALE)
        _H_ref = min(2000, ref_img.shape[0])
        template_left = ref_img[:_H_ref, 200 : 200 + 20]
        
        if abs(scale_factor - 1.0) > 1e-3:
            new_w = int(round(template_left.shape[1] * scale_factor))
            new_h = int(round(template_left.shape[0] * scale_factor))
            template_left = cv2.resize(template_left, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        _H = min(template_left.shape[0], dist_img.shape[0])
        template_left = template_left[:_H, :]
        
        # Reduce search range to prevent aliasing on the periodic tags
        c_start = max(0, layout.left_x0 - 15)
        c_end   = min(dist_img.shape[1], layout.left_x0 + 35)
        
        # Try both direct and inverted to handle both simulation and real scans
        crop_direct = dist_img[:_H, c_start:c_end].astype(np.uint8)
        crop_inverted = 255 - crop_direct
        
        res_direct = cv2.matchTemplate(crop_direct, template_left, cv2.TM_CCOEFF_NORMED)
        _, max_val_dir, _, max_loc_dir = cv2.minMaxLoc(res_direct)
        
        res_inv = cv2.matchTemplate(crop_inverted, template_left, cv2.TM_CCOEFF_NORMED)
        _, max_val_inv, _, max_loc_inv = cv2.minMaxLoc(res_inv)
        
        if max_val_dir > max_val_inv:
            best_match_val = float(max_val_dir)
            best_x = c_start + max_loc_dir[0]
        else:
            best_match_val = float(max_val_inv)
            best_x = c_start + max_loc_inv[0]
            
        if best_match_val > 0.1:
            shift = best_x - layout.left_x0
            self.global_x_shift = shift
            print(f"      Global Tag Alignment: {shift:+} px (match={best_match_val:.2f})")
            for idx in range(len(layout.strips)):
                cs, ce, ad = layout.strips[idx]
                layout.strips[idx] = (cs + shift, ce + shift, ad)

        # Save original (unshifted) layout for later reference
        orig_layout = build_layout(left_x0=int(round(200 * scale_factor)), 
                                   right_x0=cfg.width - get_total_right_width(scale_factor) - int(round(350 * scale_factor)) - int(round(200 * scale_factor)),
                                   scale_factor=scale_factor)

        if known_angle_deg is not None:
            # Bypass variance-based decode — use caller-supplied angle directly.
            angle = known_angle_deg
            score = 0.0
            print(f"      Rotation tag decode: angle={angle:+.2f}° (caller-supplied, skipping variance decode)")
        else:
            angle, best_signal, score = decode_rotation_angle(
                dist_img, layout, verbose=False
            )
            print(f"      Rotation tag decode: angle={angle:+.2f}°  var={score:.0f}")

        # ── 2b: Remove physical tilt from Y-map calculation ───────────────────
        self.physical_cw_deg = -angle
        print(f"      Mapped physical tilt geometry to: {self.physical_cw_deg:+.2f}° CW.")
        
        H, W = dist_img.shape
        cx = W / 2.0
        
        tag_x = cx
        tag_strip_source = orig_layout.strips if known_angle_deg is not None else layout.strips
        for col_start, col_end, strip_angle in tag_strip_source:
            if abs(strip_angle - angle) < 1e-4:
                tag_x = (col_start + col_end) / 2.0
                break
        
        import math
        theta_cw_rad = math.radians(self.physical_cw_deg)
        y_shift = (tag_x - cx) * math.sin(theta_cw_rad)
        print(f"      Calculated Y-shift for {angle:+.2f}° tag: {y_shift:+.1f} px")

        # ── 2c: Build Y-map from the median-filtered best strip ───────────────
        # Rather than a single pixel column (noisy), take the median across the
        # full width of the best strip so the signal is robust to scan noise.
        if override_x is not None:
            cs_i = max(0, override_x)
            ce_i = min(dist_img.shape[1], override_x + override_width)
            print(f"      Using OVERRIDE chirp strip at x={cs_i}:{ce_i}")
            best_signal = np.median(dist_img[:, cs_i:ce_i], axis=1).astype(np.uint8)
            
            # Recalculate y_shift for the new tag_x position
            tag_x = (cs_i + ce_i) / 2.0
            y_shift = (tag_x - cx) * math.sin(theta_cw_rad)
            print(f"      Recalculated Y-shift for overridden x={tag_x:.1f}: {y_shift:+.1f} px")
        else:
            # Extract the chirp signal from the strip at its actual diagonal position in the
            # distorted image. After physical rotation by angle_deg CW, the strip that was
            # painted diagonally at -angle_deg in the reference becomes vertical at col_base.
            # But due to the rotation warp, the strip in the distorted image is not exactly
            # vertical — its column drifts by approximately -sin(angle_rad) * (row - cy) per row.
            # At the pivot row (cy), the strip is at col_base; elsewhere it shifts.
            strip_source = orig_layout.strips if known_angle_deg is not None else layout.strips
            for col_start, col_end, strip_angle in strip_source:
                if abs(strip_angle - angle) < 1e-4:
                    cs_base = int(col_start)
                    ce_base = int(col_end)
                    H_img = dist_img.shape[0]
                    W_img = dist_img.shape[1]
                    cy_img = H_img / 2.0
                    # Drift rate: -2*sin(angle) per row
                    # Factor of 2: rotation moves the painted strip position by -sin(angle)/row,
                    # and the strip's own diagonal adds another -sin(angle)/row → -2*sin(angle) total.
                    sin_a = math.sin(math.radians(angle))
                    rows_idx = np.arange(H_img)
                    col_offsets = np.round(-2.0 * sin_a * (rows_idx - cy_img)).astype(int)
                    c0s = np.clip(cs_base + col_offsets, 0, W_img - 1)
                    c1s = np.clip(ce_base + col_offsets, 0, W_img)
                    diag_signal = np.array([
                        float(np.median(dist_img[r, c0s[r]:max(c0s[r]+1, c1s[r])]))
                        for r in range(H_img)
                    ], dtype=np.float32)
                    best_signal = np.clip(diag_signal, 0, 255).astype(np.uint8)
                    print(f"      Extracting diagonal strip signal (drift={-sin_a:.5f}/row)")
                    break


        raw_map = build_y_map_from_signal(best_signal, f0=CHIRP_F0, f1=CHIRP_FREQ,
                                           y_shift=y_shift, ideal_H=REF_H,
                                           pad_px=int(round(200 * getattr(self, '_last_scale_factor', scale_factor))),
                                           reverse_chirp=flip_vertical)
        return self._mono(raw_map)


    def _match_edges(self, db, ib):
        """Slide the ideal edge sequence to find the best interval-matching start."""
        de = np.where(np.diff(db) != 0)[0].astype(float)
        ie = np.where(np.diff(ib) != 0)[0].astype(float)
        if len(de) < 5 or len(ie) < 5:
            return [], []
        dd = np.diff(db)[de.astype(int)]
        id_ = np.diff(ib)[ie.astype(int)]
        best_score, best_off = -1.0, 0
        for off in range(min(40, len(ie) - 5)):
            m = min(10, len(de), len(ie) - off)
            interval_err = np.sum(np.abs(np.diff(de[:m]) - np.diff(ie[off:off+m])))
            # Small bonus for polarity match, but interval error dominates
            polarity_bonus = 1.0 if id_[off] == dd[0] else 0.0
            score = polarity_bonus / (interval_err + 1e-6) + 1.0 / (interval_err + 1e-6)
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

        import cv2
        cv2.imwrite('data/straightened_rotated.png', out)
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
