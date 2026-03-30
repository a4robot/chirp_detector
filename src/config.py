import os
from dataclasses import dataclass

@dataclass
class ScanConfig:
    """
    Central configuration for the 4-column, 1000×400 ground-truth image.

    Layout
    ──────
    Col 1 (  0– 99): X-Axis Reference  — white bg + 10-px black centre line
    Col 2 (100–199): Low Chirp Fwd     —  5 → 25 cycles  (freq ↑, top→bottom)
    Col 3 (200–299): Low Chirp Rev     — 25 →  5 cycles  (freq ↓, top→bottom)
    Col 4 (300–399): V-Chirp           —  5 → 25 → 5     (freq ↑ top→mid, ↓ mid→bot)

    The three chirp strips form a complementary set:
      • Fwd  :  f_inst(t) = 5  + 20·t
      • Rev  :  f_inst(t) = 25 − 20·t
      • V    :  f_inst(t) = 5  + 20·(2t)  for t ∈ [0, 0.5)
                           = 25 − 20·(2t−1) for t ∈ [0.5, 1)
    Sum of Fwd+Rev is constant (30 cy), uniquely pinning t globally.
    The V-strip adds a second constraint that resolves the top/bottom half
    ambiguity that would otherwise arise if Fwd or Rev alone were used.
    """

    # ── Image Dimensions ──────────────────────────────────────────────────────
    height: int = 1000
    width:  int = 400      # 4 × 100-px columns

    # ── Column 1: X-Axis Reference (Vibration Tracker) ───────────────────────
    col1_start: int = 0
    col1_end:   int = 100
    x_line_start: int = 45
    x_line_end:   int = 55
    x_tracker_expected_center: float = 49.5

    # ── Column 2: Low Chirp Forward  (5 → 25 cycles, top-to-bottom) ──────────
    col2_start: int   = 100
    col2_end:   int   = 200
    col2_f0:    float = 5.0
    col2_f1:    float = 25.0

    # ── Column 3: Low Chirp Reversed (25 → 5 cycles, bottom-to-top) ──────────
    col3_start: int   = 200
    col3_end:   int   = 300
    col3_f0:    float = 25.0
    col3_f1:    float = 5.0

    # ── Column 4: V-Chirp (5→25 top-to-mid, then 25→5 mid-to-bottom) ─────────
    # The column is split at the midpoint row.  Each half is an independent
    # linear-frequency chirp compressed into 500 rows.
    # f_edge = frequency at both ends (top row and bottom row)
    # f_peak = frequency at the midpoint row (row 500)
    col4_start:  int   = 300
    col4_end:    int   = 400
    col4_f_edge: float = 5.0    # frequency at row 0 and row 999
    col4_f_peak: float = 25.0   # frequency at row 500 (midpoint)

# ─────────────────────────────────────────────────────────────────────────────
# Path Management
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_data_path(filename: str) -> str:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return os.path.join(DATA_DIR, filename)

REF_NPY   = get_data_path("reference_image.npy")
REF_PNG   = get_data_path("reference_image.png")
DIST_NPY  = get_data_path("distorted_image.npy")
DIST_PNG  = get_data_path("distorted_image.png")
REST_NPY  = get_data_path("restored_image.npy")
REST_PNG  = get_data_path("restored_image.png")
ANS_KEY   = get_data_path("answer_key.json")
RECIPE    = get_data_path("recipe.json")
DIFF_PNG  = get_data_path("diff_image.png")
REPORT    = get_data_path("analysis_report.txt")
