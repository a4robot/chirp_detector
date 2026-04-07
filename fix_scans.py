import numpy as np
import os
from PIL import Image
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import PchipInterpolator

import sys
sys.path.insert(0, '.')
from src.config import ScanConfig

Image.MAX_IMAGE_PIXELS = None


def extract_checker_y_edges(scan, ref, scale_x, shift_x):
    """Extract Y-axis control points from vertical checkerboard cell boundaries.
    
    The checkerboard cells produce sharp vertical transitions every ~1000 ref rows.
    We detect these in both the scan and reference, match by polarity, and return
    paired (scan_row, ref_row) control points.
    """
    H_scan = scan.shape[0]
    
    # Average across the first checker column corridor in scan coords
    # Ref checker left edge is at x≈2000; in scan coords that's (2000 - shift_x) / scale_x ≈ 2189
    ref_col_start = 2000
    ref_col_end = 3000
    scan_col_start = int((ref_col_start - shift_x) / scale_x)
    scan_col_end = int((ref_col_end - shift_x) / scale_x)
    
    # Scan vertical profile
    scan_v = np.mean(scan[:, scan_col_start:scan_col_end].astype(float), axis=1)
    scan_v_smooth = uniform_filter1d(scan_v, size=20)
    scan_v_bin = (scan_v_smooth > 127).astype(float)
    transitions = np.where(np.abs(np.diff(scan_v_bin)) > 0.5)[0]
    transitions = transitions[(transitions > 1000) & (transitions < H_scan - 500)]
    
    # Filter: require minimum interval > 500px
    scan_edges = [transitions[0]]
    for e in transitions[1:]:
        if e - scan_edges[-1] > 500:
            scan_edges.append(e)
    scan_edges = np.array(scan_edges, dtype=float)
    
    # Reference vertical profile
    ref_v = np.mean(ref[:, ref_col_start:ref_col_end].astype(float), axis=1)
    ref_v_bin = (ref_v > 127).astype(float)
    ref_edges = np.where(np.abs(np.diff(ref_v_bin)) > 0.5)[0].astype(float)
    
    # Determine polarity offset: does the first scan edge match the first ref edge?
    # Scan first edge: check if it's rising (B→W) or falling (W→B)
    scan_rising = scan_v_smooth[int(scan_edges[0]) + 20] > scan_v_smooth[int(scan_edges[0]) - 20]
    ref_rising = ref_v[int(ref_edges[0]) + 10] > ref_v[int(ref_edges[0]) - 10]
    
    if scan_rising == ref_rising:
        # Same polarity: direct 1-to-1 match
        n = min(len(scan_edges), len(ref_edges))
        return scan_edges[:n], ref_edges[:n]
    else:
        # Opposite polarity: scan edge i → ref edge i+1
        n = min(len(scan_edges), len(ref_edges) - 1)
        return scan_edges[:n], ref_edges[1:1 + n]


def measure_x_calibration_per_row(scan, content_row_range=None, n_samples=20, scale_x=1.011073):
    """Measure the X position of the checker col1→col2 boundary at multiple Y positions.
    
    The column 1→2 boundary (ref x≈2987) always has a transition regardless of
    cell polarity at that row. We search in a window around the expected position.
    Returns (scan_rows, scan_x_positions) of this boundary.
    """
    H_scan = scan.shape[0]
    if content_row_range is None:
        r_lo, r_hi = 2000, H_scan - 500
    else:
        r_lo, r_hi = content_row_range
    
    # Expected column 1→2 boundary in scan coords: (2987 + 213) / 1.011 ≈ 3164
    # Search in a ±200px window
    expected_x = 3164
    search_lo = expected_x - 200
    search_hi = expected_x + 200
    
    step = (r_hi - r_lo) // (n_samples + 1)
    
    scan_rows = []
    scan_x_positions = []
    
    for i in range(1, n_samples + 1):
        row = r_lo + i * step
        if row >= H_scan:
            continue
        
        bits = (scan[row, :] > 127).astype(int)
        
        # Find the transition closest to expected_x in the search window
        best_edge = None
        best_dist = 1e6
        for x in range(search_lo, min(search_hi, scan.shape[1])):
            if bits[x] != bits[x - 1]:
                dist = abs(x - expected_x)
                if dist < best_dist:
                    best_dist = dist
                    best_edge = x
        
        if best_edge is not None:
            scan_rows.append(float(row))
            scan_x_positions.append(float(best_edge))
    
    return np.array(scan_rows), np.array(scan_x_positions)


def fix_scans():
    cfg = ScanConfig()
    ideal_H = cfg.height  # 20000
    ideal_W = cfg.width   # 14975
    
    # Initial X calibration (may be refined per-scan)
    scale_x = 1.011073
    shift_x = -212.76
    
    ref = np.array(Image.open('data/reference_image.png').convert('L'))
    print(f"Reference: {ideal_H}x{ideal_W}")
    
    scan_files = sorted([
        f for f in os.listdir('data')
        if f.startswith('image_') and f.endswith('.png')
    ])
    
    for fname in scan_files:
        file_path = os.path.join('data', fname)
        print(f"\n{'='*60}")
        print(f"--- Aligning {file_path} ---")
        scan = np.array(Image.open(file_path).convert('L'))
        
        # Pad 1000px black border on top and bottom so edge extraction and
        # spline extrapolation don't clip content at the boundaries.
        PAD = 1000
        scan = np.pad(scan, ((PAD, PAD), (0, 0)), mode='constant', constant_values=0)
        H_scan, W_scan = scan.shape
        print(f"  Scan size: {H_scan}x{W_scan} (after ±{PAD}px Y padding)")
        
        # ── Step 1: Y-axis mapping from checkerboard vertical edges ────────
        scan_y_edges, ref_y_edges = extract_checker_y_edges(scan, ref, scale_x, shift_x)
        print(f"  Y control points: {len(scan_y_edges)} matched checker edges")
        print(f"    scan [{scan_y_edges[0]:.0f}, {scan_y_edges[-1]:.0f}] → ref [{ref_y_edges[0]:.0f}, {ref_y_edges[-1]:.0f}]")
        
        y_spline = PchipInterpolator(scan_y_edges, ref_y_edges, extrapolate=True)
        y_map = y_spline(np.arange(H_scan, dtype=float))
        
        # Invert: for each target ref row, find the source scan row
        source_scan_rows = np.interp(
            np.arange(ideal_H, dtype=float),
            y_map,
            np.arange(H_scan, dtype=float)
        )
        
        # ── Step 2: X-axis mapping (with rotation correction) ──────────────
        # Measure checker left edge position at multiple Y bands
        # Use the checker Y range to avoid measuring in the black padding
        content_range = (int(scan_y_edges[0]), int(scan_y_edges[-1]))
        band_rows, band_x = measure_x_calibration_per_row(scan, content_row_range=content_range)
        
        # The ref checker col1→col2 boundary is at x=2987. For each band:
        # ref_x = scale_x * scan_x + shift_x  →  scale_x stays ~1.011
        # shift_x may vary with Y due to rotation
        ref_checker_x = 2987.0
        band_shift = ref_checker_x - scale_x * band_x
        
        # Compute per-target-row shift via interpolation
        # Map band_rows (in scan coords) to ref coords
        band_ref_rows = y_spline(band_rows)
        shift_per_ref_row = np.interp(
            np.arange(ideal_H, dtype=float),
            band_ref_rows,
            band_shift
        )
        
        print(f"  X shift range: [{shift_per_ref_row.min():.1f}, {shift_per_ref_row.max():.1f}] "
              f"(Δ={shift_per_ref_row.max()-shift_per_ref_row.min():.1f}px rotation)")
        
        # ── Step 3: Reconstruction ────────────────────────────────────────
        out = np.zeros((ideal_H, ideal_W), dtype=np.uint8)
        ox = np.arange(ideal_W, dtype=float)
        
        for t in range(ideal_H):
            src_y = source_scan_rows[t]
            if src_y < 0 or src_y >= H_scan - 1:
                continue
            
            # Per-row X mapping: scan_x = (ref_x - shift) / scale_x
            shift_t = shift_per_ref_row[t]
            source_x = (ox - shift_t) / scale_x
            
            # Bilinear interpolation in Y
            y0 = int(src_y)
            y1 = min(y0 + 1, H_scan - 1)
            fy = src_y - y0
            
            row0 = scan[y0, :].astype(float)
            row1 = scan[y1, :].astype(float)
            blended_row = row0 * (1 - fy) + row1 * fy
            
            # Interpolate in X
            out[t, :] = np.clip(
                np.interp(source_x, np.arange(W_scan, dtype=float), blended_row),
                0, 255
            ).astype(np.uint8)
        
        # ── Step 4: Save and evaluate ─────────────────────────────────────
        out_name = fname.replace('image_', 'restored_')
        out_path = os.path.join('data', out_name)
        Image.fromarray(out).save(out_path)
        print(f"  Saved → {out_path}")
        
        # Quality metrics on checkerboard core
        r1, r2 = 3000, 18000
        c1, c2 = 3000, 12000
        ref_crop = ref[r1:r2, c1:c2].astype(float)
        out_crop = out[r1:r2, c1:c2].astype(float)
        mae = np.mean(np.abs(ref_crop - out_crop))
        bin_mae = np.mean(np.abs(
            (ref_crop > 127).astype(float) - (out_crop > 127).astype(float)
        ))
        print(f"  Checkerboard MAE: {mae:.2f}  |  Binary MAE: {bin_mae:.4f}")


if __name__ == '__main__':
    fix_scans()
