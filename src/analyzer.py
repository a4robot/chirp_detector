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

        for cs, ce, mode, phase in cfg.strips:
            f0, f1 = cfg.f0, cfg.f1
            if mode == "rev":
                f0, f1 = f1, f0
            k       = f1 - f0
            p_ideal = 2.0 * np.pi * (f0 * t_ref + 0.5 * k * t_ref ** 2) + phase
            ideal   = (np.sin(p_ideal) >= 0).astype(int)

            raw  = dist_img[:, cs:ce].mean(axis=1)
            dist = (raw >= 128).astype(int)

            de, ie = self._match_edges(dist, ideal)
            for d, i in zip(de, ie):
                all_pts.append((float(d), float(i)))

        if len(all_pts) < 10:
            return np.linspace(0, H - 1, dist_img.shape[0])

        # Pool → deduplicate distorted-row duplicates → PCHIP
        bucket: dict = {}
        for d, i in all_pts:
            bucket.setdefault(d, []).append(i)
        s_d = sorted(bucket)
        s_i = [float(np.mean(bucket[k])) for k in s_d]

        # Median outlier removal
        clean_d, clean_i = [], []
        for j in range(len(s_i)):
            lo, hi = max(0, j - 5), min(len(s_i), j + 6)
            if abs(s_i[j] - float(np.median(s_i[lo:hi]))) < 20.0:
                clean_d.append(s_d[j])
                clean_i.append(s_i[j])

        if len(clean_d) < 5:
            return np.linspace(0, H - 1, dist_img.shape[0])

        pchip  = PchipInterpolator(clean_d, clean_i, extrapolate=True)
        rows   = np.arange(dist_img.shape[0], dtype=float)
        sy     = np.clip(pchip(rows), 0, H - 1)
        return self._mono(sy)

    def _match_edges(self, db, ib):
        """Slide the ideal edge sequence to find the best polarity-aligned start."""
        de = np.where(np.diff(db) != 0)[0].astype(float)
        ie = np.where(np.diff(ib) != 0)[0].astype(float)
        dd = np.diff(db)[de.astype(int)]
        id = np.diff(ib)[ie.astype(int)]
        if len(de) < 5 or len(ie) < 5:
            return [], []

        best_score, best_off = -1.0, 0
        for off in range(min(40, len(ie) - 5)):
            if id[off] != dd[0]:
                continue
            m = min(10, len(de), len(ie) - off)
            score = 1.0 / (np.sum(np.abs(np.diff(de[:m]) - np.diff(ie[off:off+m]))) + 1e-6)
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
