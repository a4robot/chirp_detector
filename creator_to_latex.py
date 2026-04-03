import numpy as np
import os
import math
from src.config import ScanConfig
from src.rotation_tag import build_layout, TOTAL_RIGHT_WIDTH, CHIRP_F0, CHIRP_FREQ

def generate_latex():
    cfg = ScanConfig()
    H = cfg.height
    W = cfg.width
    SCALE = 500.0 / 16384.0 # mm per px

    W_mm = W * SCALE
    H_mm = H * SCALE

    out = []
    out.append(r"\documentclass{standalone}")
    out.append(r"\usepackage{tikz}")
    out.append(r"\begin{document}")
    out.append(rf"\begin{{tikzpicture}}[x=1mm, y=-1mm]")

    # 1. Right X-Tracker
    c1s_mm = cfg.col1_start * SCALE
    c1e_mm = cfg.col1_end * SCALE
    xls_mm = cfg.x_line_start * SCALE
    xle_mm = cfg.x_line_end * SCALE

    # Background of canvas to black
    out.append(rf"\fill[black] (0, 0) rectangle ({W_mm:.4f}, {H_mm:.4f});")

    # Fill X-Tracker line with white (canvas is black)
    out.append(rf"\fill[white] ({xls_mm:.4f}, 0) rectangle ({xle_mm:.4f}, {H_mm:.4f});")

    # 2. Checkerboard
    checker_cols  = 11
    checker_rows  = 20
    square_px     = 1000
    grid_w_px = checker_cols * square_px
    grid_h_px = checker_rows * square_px
    cx0 = (W  - grid_w_px) / 2.0
    cy0 = (H - grid_h_px) / 2.0

    grid_x0 = cx0 * SCALE
    grid_y0 = cy0 * SCALE
    sq_mm = square_px * SCALE

    for gr in range(checker_rows):
        for gc in range(checker_cols):
            if (gr + gc) % 2 == 0:
                x_mm = grid_x0 + gc * sq_mm
                y_mm = grid_y0 + gr * sq_mm
                
                # Clip width/height to edge
                x2 = x_mm + sq_mm if (x_mm + sq_mm) <= W_mm else W_mm
                y2 = y_mm + sq_mm if (y_mm + sq_mm) <= H_mm else H_mm
                
                if x2 > x_mm and y2 > y_mm:
                    out.append(rf"\fill[white] ({x_mm:.4f}, {y_mm:.4f}) rectangle ({x2:.4f}, {y2:.4f});")

    # 3. Rotation Tags
    layout = build_layout(left_x0=200, right_x0=W - TOTAL_RIGHT_WIDTH - 350 - 200)
    f0, f1 = CHIRP_F0, CHIRP_FREQ
    k = f1 - f0
    
    t_arr = np.linspace(0.0, 1.0, H, endpoint=False)
    phi_arr = 2.0 * np.pi * (f0 * t_arr + 0.5 * k * t_arr ** 2)
    
    # Clip to drawn area
    r_min = 200
    r_max = H - 200

    # find white segments
    is_white = (np.sin(phi_arr) >= 0)[r_min:r_max]
    diffs = np.diff(np.concatenate([[0], is_white.astype(int), [0]]))
    starts = np.where(diffs == 1)[0] + r_min
    ends = np.where(diffs == -1)[0] + r_min - 1

    cy = H / 2.0

    for col_start, col_end, angle_deg in layout.strips:
        theta = np.deg2rad(-angle_deg)
        tan_t = math.tan(theta)
        
        for y_start, y_end in zip(starts, ends):
            y1 = y_start
            y2 = y_end + 1
            
            sx1 = (y1 - cy) * tan_t
            sx2 = (y2 - cy) * tan_t
            
            x1_lo = (col_start + sx1) * SCALE
            x1_hi = (col_end + sx1) * SCALE
            x2_lo = (col_start + sx2) * SCALE
            x2_hi = (col_end + sx2) * SCALE
            y1_mm = y1 * SCALE
            y2_mm = y2 * SCALE
            
            # constrain within boundaries just in case
            if x1_hi < 0 or x2_lo > W_mm: continue
            
            out.append(rf"\fill[white] ({x1_lo:.4f},{y1_mm:.4f}) -- ({x1_hi:.4f},{y1_mm:.4f}) -- ({x2_hi:.4f},{y2_mm:.4f}) -- ({x2_lo:.4f},{y2_mm:.4f}) -- cycle;")

    out.append(r"\end{tikzpicture}")
    out.append(r"\end{document}")

    with open("calibration_target_vector.tex", "w") as f:
        f.write("\n".join(out))

    print("LaTeX successfully written to calibration_target_vector.tex")

if __name__ == '__main__':
    generate_latex()
