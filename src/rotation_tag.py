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


def build_y_map_from_signal(signal: np.ndarray,
                             f0: float = CHIRP_F0,
                             f1: float = CHIRP_FREQ,
                             y_shift: float = 0.0) -> np.ndarray:
    """Build a PCHIP Y-axis mapping from the winning 1D signal.

    y_shift: Physical rotation offsets the tag vertically (e.g., tags on the right
             shift down when the object is rotated CW). Subtract this shift when
             identifying which ideal edge corresponds to the first observed edge.
    """
    H = len(signal)

    # -- Detect observed edges ------------------------------------------------
    centered = signal.astype(float) - 127.5
    signs = np.sign(centered)
    signs[signs == 0] = 1
    cross_idx = np.where(np.diff(signs) != 0)[0]

    if len(cross_idx) < 4:
        return np.arange(H, dtype=float)

    obs_edges = []
    for i in cross_idx:
        a, b = centered[i], centered[i + 1]
        frac = -a / (b - a) if (b - a) != 0 else 0.5
        obs_edges.append(i + frac)
    obs_edges = np.array(obs_edges)

    # -- Filter out spurious boundary edges in the padding zone ----------------
    PAD_PX = 200
    obs_edges = obs_edges[(obs_edges >= PAD_PX) & (obs_edges <= H - PAD_PX)]

    if len(obs_edges) < 4:
        return np.arange(H, dtype=float)

    # -- Ideal edge positions (full span 0..H-1) --------------------------------
    t_fine = np.linspace(0, 1, H * 4)
    k = f1 - f0
    phi_fine = 2.0 * np.pi * (f0 * t_fine + 0.5 * k * t_fine ** 2)
    s_fine = np.sign(np.sin(phi_fine))
    s_fine[s_fine == 0] = 1
    ideal_edges = np.where(np.diff(s_fine) != 0)[0].astype(float) / 4.0

    # -- Align observed edges to ideal by position, correcting for y_shift -----
    # Filter false cut-off edges at the top boundary
    while len(obs_edges) > 0:
        eff_obs = obs_edges[0] - y_shift
        j0 = int(np.argmin(np.abs(ideal_edges - eff_obs)))
        if np.abs(ideal_edges[j0] - eff_obs) > 400:
            obs_edges = obs_edges[1:]
        else:
            break

    # Filter false cut-off edges at the bottom boundary
    while len(obs_edges) > 0:
        eff_obs = obs_edges[-1] - y_shift
        j1 = int(np.argmin(np.abs(ideal_edges - eff_obs)))
        if np.abs(ideal_edges[j1] - eff_obs) > 400:
            obs_edges = obs_edges[:-1]
        else:
            break

    if len(obs_edges) < 4:
        return np.arange(H, dtype=float)

    first_obs = obs_edges[0]
    effective_obs = first_obs - y_shift
    j0 = int(np.argmin(np.abs(ideal_edges - effective_obs)))

    n = min(len(obs_edges), len(ideal_edges) - j0)
    dist_s  = obs_edges[:n]
    ideal_s = ideal_edges[j0: j0 + n]

    # Clean duplicates
    _, uniq = np.unique(dist_s, return_index=True)
    dist_s  = dist_s[uniq]
    ideal_s = ideal_s[uniq]

    if len(dist_s) < 2:
        return np.arange(H, dtype=float)

    # -- Anchor spline at image boundaries ------------------------------------
    # To mathematically preserve the geometric rotation before cv2.warpAffine applies,
    # the mapped ideal_s MUST include the physical rotation offset (y_shift).
    ideal_s_shifted = ideal_s + y_shift
    
    # Rows in the padding zone (no signal) should map to identity.
    # The physical line scanner starts at row 0 and ends at row H-1, so 
    # Y_cam(0) = 0 and Y_cam(H-1) = H-1.
    anchor_0 = 0.0
    anchor_H = float(H - 1)

    dist_s  = np.concatenate([[0.0],  dist_s,  [float(H - 1)]])
    ideal_s = np.concatenate([[anchor_0],  ideal_s_shifted, [anchor_H]])
    
    _, uniq = np.unique(dist_s, return_index=True)
    dist_s  = dist_s[uniq]
    ideal_s = ideal_s[uniq]

    spline = PchipInterpolator(dist_s, ideal_s, extrapolate=True)
    # Don't strictly clip to H-1 if the rotation geometrically extends beyond it, 
    # but we can clip loosely or leave it. Actually, the reference image operates 
    # in [0, H-1]. The geometric derotation will shift it back.
    # We will let the spline extrapolate naturally, then clip.
    return spline(np.arange(H, dtype=float))


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Rotation Tag Self-Test ===")
    print(f"  Total strips : {len(ALL_ANGLES)}")
    
    H, W = 1000, TOTAL_TAG_WIDTH + 100
    img = np.zeros((H, W), dtype=np.uint8)
    
    # FIX: Explicitly set right_x0 so it fits within W
    # We'll place it right after the left tag + some padding
    layout = build_layout(left_x0=0, right_x0=TOTAL_LEFT_WIDTH + 50) 
    
    img = synthesize_rotation_tag(img, layout)
    
    from PIL import Image
    # (Optional) Ensure the directory exists or change to a local save path
    Image.fromarray(img).save("rotation_tag_test.png") 
    
    best_angle, _, score = decode_rotation_angle(img, layout, verbose=True)
    print(f"  Decoded angle on clean image: {best_angle:+.2f}°  (var={score:.3f})")
