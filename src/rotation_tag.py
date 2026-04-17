"""
rotation_tag.py — Rotation-Robust Tag System
=============================================
Implements the ±2° rotation-tolerant tracking system using a pre-rotated
angle ladder of single binary chirp strips.

CONCEPT
-------
A "rotation ladder" is embedded on each side of the image:
  • LEFT  side: N_ANGLES strips at θ = 0.0 → +MAX_DEG  (step 0.1°)
  • RIGHT side: N_ANGLES strips at θ = -MAX_DEG → -0.1°

Each strip is a 1D binary chirp at f=50 (best single-strip frequency),
pre-slanted so that a line-scan camera at that exact rotation angle would
see perfectly regular, undistorted edge intervals.

DECODE
------
When the distorted image arrives at the analyser, each strip is read along
its own diagonal trajectory.  The strip whose edge intervals are the MOST
REGULAR (lowest coefficient of variation) is the winner — that strip's
angle is the true physical rotation.  The Y-axis distortion map is then
built from that strip's edge sequence.

GEOMETRY
--------
A strip at angle θ (degrees, CCW positive) means the physical feature line
is tilted.  When we sample column col_c of the image at image row r, the
scan physically corresponds to a point that is laterally offset by:
  x_offset = r * tan(θ)
So to "look through" the strip at angle θ, we map each image row r to:
  sample_col = col_centre + r * tan(θ_rad)
and bilinearly interpolate the pixel value.  When θ matches the true tilt,
the resulting 1D signal is distortion-free, producing clean, periodic edges
that score well.

STRIP WIDTH
-----------
8 px per strip.  The chirp at f=50 has a half-period of ≈200 px at the
fast end, so 8 px is sufficient to read the binary edge crossing reliably
via a 1D mean along the strip width.

TOTAL FOOTPRINT
  LEFT : 21 strips × 8 px = 168 px
  RIGHT: 20 strips × 8 px = 160 px (negative angles, exclude 0° duplicate)
  TOTAL: 328 px out of 16 384 px = 2.0% overhead
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ANGLE_DEG: float = 2.0          # ±2° coverage
ANGLE_STEP_DEG: float = 0.1         # 0.1° resolution
STRIP_WIDTH_PX: int  = 20           # widened to tolerate lateral vibration + geometric cross-talk
CHIRP_FREQ: float    = 50.0         # best single-strip frequency (prior benchmarks)
CHIRP_F0: float      = 5.0          # start frequency

# Build angle lists
_LEFT_ANGLES: List[float]  = [round(i * ANGLE_STEP_DEG, 2)
                               for i in range(0, int(MAX_ANGLE_DEG / ANGLE_STEP_DEG) + 1)]
_RIGHT_ANGLES: List[float] = [round(-i * ANGLE_STEP_DEG, 2)
                               for i in range(1, int(MAX_ANGLE_DEG / ANGLE_STEP_DEG) + 1)]

ALL_ANGLES: List[float] = _LEFT_ANGLES + _RIGHT_ANGLES   # 41 strips total

STRIP_SPACING_PX: int = 20
TOTAL_LEFT_WIDTH:  int = (len(_LEFT_ANGLES)  * STRIP_WIDTH_PX) + ((len(_LEFT_ANGLES) - 1)  * STRIP_SPACING_PX)
TOTAL_RIGHT_WIDTH: int = (len(_RIGHT_ANGLES) * STRIP_WIDTH_PX) + ((len(_RIGHT_ANGLES) - 1) * STRIP_SPACING_PX)
TOTAL_TAG_WIDTH:   int = TOTAL_LEFT_WIDTH + TOTAL_RIGHT_WIDTH
        


# ── Layout descriptor ─────────────────────────────────────────────────────────

@dataclass
class RotationTagLayout:
    """Describes the pixel layout of the rotation tag on the image canvas."""
    left_x0:  int             # first column of the whole tag
    strips:   List[Tuple[int, int, float]] = field(default_factory=list)
    # Each entry: (col_start, col_end, angle_deg)

    @property
    def right_x0(self) -> int:
        if not self.strips:
            return self.left_x0
        return self.strips[len(_LEFT_ANGLES)][0]

    @property
    def total_width(self) -> int:
        return TOTAL_TAG_WIDTH


def build_layout(left_x0: int = 0, right_x0: int = 16000) -> RotationTagLayout:
    """
    Build the strip layout:
      - 0° to +2.0° on the left starting at *left_x0*.
      - -0.1° to -2.0° on the right starting at *right_x0*.
    """
    layout = RotationTagLayout(left_x0=left_x0)
    
    # Left tag: 0 to +2.0
    x = left_x0
    for angle in _LEFT_ANGLES:
        layout.strips.append((x, x + STRIP_WIDTH_PX, angle))
        x += STRIP_WIDTH_PX + STRIP_SPACING_PX
        
    # Right tag: -0.1 to -2.0
    x = right_x0
    for angle in _RIGHT_ANGLES:
        layout.strips.append((x, x + STRIP_WIDTH_PX, angle))
        x += STRIP_WIDTH_PX + STRIP_SPACING_PX
        
    return layout


# ── Image synthesis ───────────────────────────────────────────────────────────

def synthesize_rotation_tag(image: np.ndarray,
                             layout: RotationTagLayout) -> np.ndarray:
    """
    Paint the rotation tag directly onto *image* (uint8 grayscale, H×W).
    Each strip i is drawn as a slanted ribbon corresponding to angle_i.
    When the image is rotated by angle_i, this ribbon becomes exactly vertical.
    """
    H = image.shape[0]
    W = image.shape[1]
    
    f0, f1 = CHIRP_F0, CHIRP_FREQ
    k = f1 - f0

    # Base chirp signal evaluated over rows (1D spatial chirp)
    # We want the phase to depend purely on the original row (before any rotation).
    # Since the ribbon is drawn slanted, the vertical coordinate isn't just `r`.
    # Wait, the simplest way to make it perfectly invariant to X shift is
    # evaluating phase = Chirp(r) for all X inside the ribbon.
    t_arr = np.linspace(0.0, 1.0, H, endpoint=False)
    phi_arr = 2.0 * np.pi * (f0 * t_arr + 0.5 * k * t_arr ** 2)
    signal_arr = np.where(np.sin(phi_arr) >= 0, 255, 0).astype(np.uint8)

    # To avoid the ribbons overwriting each other where they cross, we draw them
    # in order. Better yet, we should base the separation on the pivot point.
    # We pivot around cy.
    cy = H / 2.0

    for col_start, col_end, angle_deg in layout.strips:
        # We pre-rotate by -angle_deg so that when the image is mathematically 
        # rotated by +angle_deg, the ribbon becomes perfectly upright.
        theta = np.deg2rad(-angle_deg)
        tan_t = np.tan(theta)
        
        # Center of the strip at the pivot row cy
        x_base = col_start
        width = col_end - col_start

        # Draw with 200px top/bottom padding as requested
        for r in range(200, H - 200):
            # To make this ribbon perfectly vertical at col_start when the image is
            # rotated by +angle_deg, it must be drawn at:
            # X_draw = X_upright + (r - cy) * tan(theta)
            shift_x = (r - cy) * tan_t
            
            x_lo = int(round(col_start + shift_x))
            x_hi = int(round(col_end + shift_x))
            
            # Clip bounds
            x_lo = max(0, min(W-1, x_lo))
            x_hi = max(0, min(W-1, x_hi))
            
            if x_lo < x_hi:
                image[r, x_lo:x_hi] = signal_arr[r]

    return image


# ── Rotation decoder ──────────────────────────────────────────────────────────

def decode_rotation_angle(image: np.ndarray,
                           layout: RotationTagLayout,
                           verbose: bool = False) -> Tuple[float, np.ndarray, float]:
    """
    Find the best-matching pre-rotated strip in *layout*.
    Since the physical image was rotated, the ribbon that matches the physical
    rotation will now be perfectly vertical at its designated bounds [col_start : col_end].
    Therefore, a strict straight vertical column read will perfectly sample it!
    """
    H = image.shape[0]
    best_angle = 0.0
    best_score = -1.0 # We want MAX variance now
    best_signal = None

    for col_start, col_end, angle_deg in layout.strips:
        H_img = image.shape[0]
        y_center = H_img // 2
        safe_y_start = max(0, y_center - 2000)
        safe_y_end   = min(H_img, y_center + 2000)
        c_start = max(0, col_start - 10)
        c_end   = min(image.shape[1], col_end + 10)
        
        strip_crop_mid = image[safe_y_start:safe_y_end, c_start:c_end]
        col_vars = np.var(strip_crop_mid, axis=0)
        score = float(np.max(col_vars))

        if verbose:
            print(f"  angle={angle_deg:+6.2f}°  var={score:.3f}")
        
        if score > best_score:
            best_score  = score
            best_angle  = angle_deg
            
            best_pixel_idx = int(np.argmax(col_vars))
            best_col = c_start + best_pixel_idx
            best_signal = image[:, best_col]

    return best_angle, best_signal, best_score



TARGET_MATCHED_EDGES: int = 52
"""Legacy constant kept for backward-compatibility.
The new typed-edge pipeline (build_y_map_from_signal) derives the target
dynamically from compute_ideal_edges_typed() — do not rely on this value."""


def compute_ideal_edges(f0:      float = CHIRP_F0,
                        f1:      float = CHIRP_FREQ,
                        ideal_H: int   = 20000) -> np.ndarray:
    """Ideal zero-crossing edge positions in [0, ideal_H] coordinate space."""
    t_fine = np.linspace(0, 1, ideal_H * 4)
    k      = f1 - f0
    phi    = 2.0 * np.pi * (f0 * t_fine + 0.5 * k * t_fine ** 2)
    s      = np.sign(np.sin(phi))
    s[s == 0] = 1
    return np.where(np.diff(s) != 0)[0].astype(float) / 4.0


def detect_chirp_edges(signal:  np.ndarray,
                       f0:      float = CHIRP_F0,
                       f1:      float = CHIRP_FREQ,
                       ideal_H: int   = 20000,
                       pad_px:  int   = 200) -> np.ndarray:
    """
    Detect sub-pixel zero-crossing edges from a 1-D binary chirp signal.

    Edges are filtered to the valid strip band (after top/bottom padding)
    and noise-rejected via a minimum-interval gate.
    """
    H       = len(signal)
    scale_y = H / float(ideal_H)

    centered  = signal.astype(float) - 127.5
    signs     = np.sign(centered)
    signs[signs == 0] = 1
    cross_idx = np.where(np.diff(signs) != 0)[0]
    if len(cross_idx) == 0:
        return np.array([])

    obs = []
    for i in cross_idx:
        a, b = centered[i], centered[i + 1]
        frac = -a / (b - a) if (b - a) != 0 else 0.5
        obs.append(i + frac)
    obs = np.array(obs)

    # Valid row window from the ideal first edge
    ideal_all       = compute_ideal_edges(f0, f1, ideal_H)
    first_chirp_ref = ideal_all[0] if len(ideal_all) > 0 else 0.0
    floor_px        = first_chirp_ref * scale_y * 0.90
    ceil_px         = (ideal_H - pad_px) * scale_y
    obs = obs[(obs >= floor_px) & (obs <= ceil_px)]

    # Minimum-interval filter (reject sub-half-period glitches)
    min_half_ref = ideal_H / (2.0 * f1)
    min_interval = max(30.0, min_half_ref * scale_y * 0.80)
    if len(obs) > 1:
        kept = [obs[0]]
        for e in obs[1:]:
            if e - kept[-1] >= min_interval:
                kept.append(e)
        obs = np.array(kept)

    return obs


def _try_match(obs_edges:   np.ndarray,
               ideal_edges: np.ndarray,
               scale_y:     float,
               j0:          int,
               tolerance:   float) -> Tuple[int, list, list]:
    """Single greedy one-to-one matching pass with given (j0, tolerance)."""
    dist_list  = []
    ideal_list = []
    obs_ptr    = 0

    for ji in range(j0, min(j0 + len(obs_edges) + 10, len(ideal_edges))):
        if obs_ptr >= len(obs_edges):
            break
        ideal_pos    = ideal_edges[ji] * scale_y
        obs_pos      = obs_edges[obs_ptr]
        exp_interval = (ideal_edges[ji] - ideal_edges[ji - 1]) * scale_y if ji > 0 else 500.0 * scale_y
        tol_px       = max(exp_interval * tolerance, 50.0)

        if abs(obs_pos - ideal_pos) <= tol_px:
            dist_list.append(obs_pos)
            ideal_list.append(ideal_edges[ji])
            obs_ptr += 1
        else:
            # Try skipping one obs edge (handles a single noise spike)
            if obs_ptr + 1 < len(obs_edges):
                next_obs = obs_edges[obs_ptr + 1]
                if abs(next_obs - ideal_pos) < abs(obs_pos - ideal_pos):
                    obs_ptr += 1
                    if abs(obs_edges[obs_ptr] - ideal_pos) <= tol_px:
                        dist_list.append(obs_edges[obs_ptr])
                        ideal_list.append(ideal_edges[ji])
                        obs_ptr += 1

    return len(dist_list), dist_list, ideal_list


def iterative_edge_match(obs_edges:   np.ndarray,
                          ideal_edges: np.ndarray,
                          scale_y:     float,
                          target_n:    int   = TARGET_MATCHED_EDGES,
                          y_shift:     float = 0.0) -> Tuple[np.ndarray, np.ndarray, int, float, bool]:
    """
    Search (j0 × tolerance) until exactly *target_n* edges are matched.

    Sweeps j0 = j_est ± 30, tolerance ∈ [0.20, 0.60] step 0.05.
    Returns the first combination that hits exactly target_n.
    Returns success=False (no exception) if target cannot be met.

    Returns
    -------
    (dist_s, ideal_s, best_j0, best_tol, success)
    """
    if len(obs_edges) == 0:
        return np.array([]), np.array([]), 0, 0.0, False

    first_obs_eff = obs_edges[0] / scale_y - y_shift / max(scale_y, 1e-9)
    j_est         = max(0, int(np.argmin(np.abs(ideal_edges - first_obs_eff))))

    tolerances  = np.round(np.arange(0.20, 0.61, 0.05), 2).tolist()
    best_cnt    = 0

    for dj in range(-30, 31):
        j0 = j_est + dj
        if j0 < 0 or j0 + target_n > len(ideal_edges):
            continue
        for tol in tolerances:
            count, dist_list, ideal_list = _try_match(obs_edges, ideal_edges, scale_y, j0, tol)
            if count == target_n:
                dist_s  = np.array(dist_list)
                ideal_s = np.array(ideal_list)
                _, uniq = np.unique(dist_s, return_index=True)
                dist_s, ideal_s = dist_s[uniq], ideal_s[uniq]
                print(f"      [Y-map] Iterative match: j0={j0}, tol={tol:.2f} → {count}/{target_n} ✓")
                return dist_s, ideal_s, j0, tol, True
            if count > best_cnt:
                best_cnt = count

    print(f"      [Y-map] ABORT — target={target_n} unreachable. best_so_far={best_cnt}, obs={len(obs_edges)}")
    return np.array([]), np.array([]), j_est, 0.0, False


def build_spline_map(dist_s:  np.ndarray,
                     ideal_s: np.ndarray,
                     H:       int,
                     ideal_H: int) -> np.ndarray:
    """
    Build a PCHIP spline Y-map from matched (distorted-row, ideal-row) pairs.
    Boundary anchors are extrapolated from the slope at each end.
    """
    if len(dist_s) < 2:
        return np.arange(H, dtype=float)

    slope_s = (ideal_s[1]  - ideal_s[0])  / max(dist_s[1]  - dist_s[0],  1.0)
    slope_e = (ideal_s[-1] - ideal_s[-2]) / max(dist_s[-1] - dist_s[-2], 1.0)
    ev_s    = max(0.0,             float(ideal_s[0]  - slope_s * dist_s[0]))
    ev_e    = min(float(ideal_H - 1), float(ideal_s[-1] + slope_e * (H - 1 - dist_s[-1])))

    dist_ext  = np.concatenate([[0.0],   dist_s,  [float(H - 1)]])
    ideal_ext = np.concatenate([[ev_s],  ideal_s, [ev_e]])

    spline  = PchipInterpolator(dist_ext, ideal_ext, extrapolate=False)
    raw_map = spline(np.arange(H, dtype=float))
    return np.clip(raw_map, 0.0, float(ideal_H - 1))


def compute_ideal_edges_typed(f0:      float = CHIRP_F0,
                               f1:      float = CHIRP_FREQ,
                               ideal_H: int   = 20000,
                               pad_px:  int   = 200
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Ideal chirp edges filtered to the drawn strip [pad_px, ideal_H-pad_px-1].

    Includes the virtual strip-start rising edge when the chirp opens HIGH at
    pad_px (the blank-to-chirp boundary is a real, observable rising edge).

    Returns
    -------
    positions : np.ndarray  — row positions in reference space
    types     : np.ndarray  — +1 rising (LOW→HIGH),  -1 falling (HIGH→LOW)
    """
    t_fine = np.linspace(0, 1, ideal_H * 4)
    k      = f1 - f0
    phi    = 2.0 * np.pi * (f0 * t_fine + 0.5 * k * t_fine ** 2)
    s      = np.sign(np.sin(phi))
    s[s == 0] = 1
    diff_s     = np.diff(s)
    cross_idx  = np.where(diff_s != 0)[0]

    positions  = cross_idx.astype(float) / 4.0
    types      = np.sign(diff_s[cross_idx]).astype(int)   # +1 rising, -1 falling

    # Filter to physically drawn strip rows
    strip_start = float(pad_px)
    strip_end   = float(ideal_H - pad_px - 1)
    mask        = (positions >= strip_start) & (positions <= strip_end)
    positions   = positions[mask]
    types       = types[mask]

    # Virtual start: chirp opens HIGH at pad_px → blank→chirp is a rising edge
    t_s   = pad_px / float(ideal_H)
    phi_s = 2.0 * np.pi * (f0 * t_s + 0.5 * k * t_s ** 2)
    if np.sin(phi_s) >= 0:
        positions = np.concatenate([[strip_start], positions])
        types     = np.concatenate([[+1],          types])

    return positions, types


def detect_chirp_edges_typed(signal:  np.ndarray,
                              f0:      float = CHIRP_F0,
                              f1:      float = CHIRP_FREQ,
                              ideal_H: int   = 20000,
                              pad_px:  int   = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Detect typed sub-pixel zero-crossing edges from a 1-D binary chirp signal.

    Sets floor_px one row before the strip start so the blank→chirp rising edge
    is naturally captured rather than filtered away (the old floor_px=1346 cut
    it out, leaving the count odd and causing cascade matching errors).

    Returns
    -------
    positions : np.ndarray  — edge positions in scan space
    types     : np.ndarray  — +1 rising, -1 falling
    """
    H       = len(signal)
    scale_y = H / float(ideal_H)

    centered   = signal.astype(float) - 127.5
    signs      = np.sign(centered)
    signs[signs == 0] = 1
    diff_signs = np.diff(signs)
    cross_idx  = np.where(diff_signs != 0)[0]

    if len(cross_idx) == 0:
        return np.array([]), np.array([], dtype=int)

    obs, obs_t = [], []
    for i in cross_idx:
        a, b = centered[i], centered[i + 1]
        frac = -a / (b - a) if (b - a) != 0 else 0.5
        obs.append(i + frac)
        obs_t.append(int(np.sign(diff_signs[i])))
    obs   = np.array(obs)
    obs_t = np.array(obs_t, dtype=int)

    # Expand floor/ceil by 500px to account for distortion drift near boundaries
    floor_px = max(0.0, (pad_px - 500) * scale_y)
    ceil_px  = (ideal_H - pad_px + 500) * scale_y
    mask     = (obs >= floor_px) & (obs <= ceil_px)
    obs, obs_t = obs[mask], obs_t[mask]

    # Minimum-interval noise filter
    min_half_ref = ideal_H / (2.0 * f1)
    min_interval = max(30.0, min_half_ref * scale_y * 0.80)
    if len(obs) > 1:
        kept, kept_t = [obs[0]], [obs_t[0]]
        for e, et in zip(obs[1:], obs_t[1:]):
            if e - kept[-1] >= min_interval:
                kept.append(e)
                kept_t.append(et)
        obs, obs_t = np.array(kept), np.array(kept_t, dtype=int)

    return obs, obs_t


def _match_typed_subseq(obs:       np.ndarray,
                         ideal:     np.ndarray,
                         scale_y:   float,
                         tolerance: float = 0.40) -> Tuple[np.ndarray, np.ndarray]:
    """Index-based match for a single edge type (all-rising or all-falling).

    The distortion is a *nonlinear* row mapping, so matching by position error
    is fundamentally wrong — an obs edge at row 2400 may legitimately correspond
    to ideal row 2540 because the scan stretched or skipped rows in between.

    Correct approach: the i-th observed edge of a given type must correspond to
    the (j0 + i)-th ideal edge of that type, where j0 is the starting offset.
    We estimate j0 from the first obs edge (linear approximation is good enough
    for a small offset search) then match the rest purely by index.

    Returns (matched_obs_scan, matched_ideal_ref).
    """
    if len(obs) == 0 or len(ideal) == 0:
        return np.array([]), np.array([])

    ideal_scan = ideal * scale_y
    j_est      = max(0, int(np.argmin(np.abs(ideal_scan - obs[0]))))

    best_n, best_dist, best_ideal_m = 0, np.array([]), np.array([])

    for dj in range(-3, 4):
        j0 = j_est + dj
        if j0 < 0:
            continue
        n = min(len(obs), len(ideal) - j0)
        if n <= 0:
            continue
        if n > best_n:
            best_n       = n
            best_dist    = obs[:n].copy()
            best_ideal_m = ideal[j0: j0 + n].copy()

    return best_dist, best_ideal_m



def typed_edge_match(obs_pos:     np.ndarray,
                     obs_types:   np.ndarray,
                     ideal_pos:   np.ndarray,
                     ideal_types: np.ndarray,
                     scale_y:     float,
                     tolerance:   float = 0.40) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Match chirp edges with strict type separation: RISING→RISING, FALLING→FALLING.

    Returns
    -------
    dist_s  : matched observed positions (scan space), sorted
    ideal_s : matched ideal positions (reference space), sorted
    success : True when both rising and falling have ≥2 matched pairs
    """
    obs_r, obs_f     = obs_pos[obs_types == +1],   obs_pos[obs_types == -1]
    ideal_r, ideal_f = ideal_pos[ideal_types == +1], ideal_pos[ideal_types == -1]

    dist_r,  ideal_r_m = _match_typed_subseq(obs_r, ideal_r, scale_y, tolerance)
    dist_f,  ideal_f_m = _match_typed_subseq(obs_f, ideal_f, scale_y, tolerance)

    print(f"      [Y-map typed] rising {len(dist_r)}/{len(ideal_r)}  "
          f"falling {len(dist_f)}/{len(ideal_f)}")

    if len(dist_r) < 2 and len(dist_f) < 2:
        return np.array([]), np.array([]), False

    dist_s  = np.concatenate([dist_r,  dist_f])
    ideal_s = np.concatenate([ideal_r_m, ideal_f_m])
    order   = np.argsort(dist_s)
    dist_s, ideal_s = dist_s[order], ideal_s[order]

    _, uniq = np.unique(dist_s, return_index=True)
    dist_s, ideal_s = dist_s[uniq], ideal_s[uniq]

    return dist_s, ideal_s, (len(dist_r) >= 2 and len(dist_f) >= 2)


def build_y_map_from_signal(signal:        np.ndarray,
                             f0:           float = CHIRP_F0,
                             f1:           float = CHIRP_FREQ,
                             y_shift:      float = 0.0,
                             ideal_H:      int   = 20000,
                             reverse_chirp: bool = False,
                             target_n:     int   = TARGET_MATCHED_EDGES) -> np.ndarray:
    """Build a PCHIP Y-axis mapping using type-separated rising/falling edge matching.

    Parameters
    ----------
    signal        : 1-D uint8 chirp strip signal from the distorted scan.
    f0 / f1       : Chirp start / end frequency (must match synthesis).
    y_shift       : Unused in typed pipeline (kept for API compatibility).
    ideal_H       : Canonical reference height.
    reverse_chirp : True when the target was scanned with a vertical flip.
    target_n      : Ignored — typed pipeline derives count from design parameters.

    Raises
    ------
    ValueError
        If typed matching cannot produce ≥2 pairs per edge type.
        Callers must catch and abort reconstruction — do not proceed.
    """
    H           = len(signal)
    work_signal = signal[::-1].copy() if reverse_chirp else signal
    scale_y     = H / float(ideal_H)

    obs_pos,   obs_types   = detect_chirp_edges_typed(work_signal, f0=f0, f1=f1, ideal_H=ideal_H)
    ideal_pos, ideal_types = compute_ideal_edges_typed(f0=f0, f1=f1, ideal_H=ideal_H)

    n_r_ideal = int(np.sum(ideal_types == +1))
    n_f_ideal = int(np.sum(ideal_types == -1))
    n_r_obs   = int(np.sum(obs_types   == +1))
    n_f_obs   = int(np.sum(obs_types   == -1))
    print(f"      [Y-map typed] obs R={n_r_obs} F={n_f_obs}  "
          f"ideal R={n_r_ideal} F={n_f_ideal}  total_ideal={len(ideal_pos)}")

    dist_s, ideal_s, success = typed_edge_match(
        obs_pos, obs_types, ideal_pos, ideal_types, scale_y
    )

    if not success:
        raise ValueError(
            f"[Y-map] Typed match failed — insufficient pairs. "
            f"obs R={n_r_obs} F={n_f_obs}. Reconstruction aborted."
        )

    raw_map = build_spline_map(dist_s, ideal_s, H, ideal_H)

    if reverse_chirp:
        raw_map_flip = np.empty_like(raw_map)
        for i in range(H):
            raw_map_flip[i] = (ideal_H - 1) - raw_map[H - 1 - i]
        return raw_map_flip

    return raw_map


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    print("=== Rotation Tag Self-Test ===")
    print(f"  Total strips : {len(ALL_ANGLES)}")

    H, W = 1000, TOTAL_TAG_WIDTH + 100
    layout = build_layout(left_x0=0, right_x0=TOTAL_LEFT_WIDTH + 50) 
    
    img = synthesize_rotation_tag(img, layout)
    
    from PIL import Image
    # (Optional) Ensure the directory exists or change to a local save path
    Image.fromarray(img).save("rotation_tag_test.png") 
    
    best_angle, _, score = decode_rotation_angle(img, layout, verbose=True)
    print(f"  Decoded angle on clean image: {best_angle:+.2f}°  (var={score:.3f})")
