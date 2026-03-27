import os
from dataclasses import dataclass

@dataclass
class ScanConfig:
    """
    Central configuration for the 3-column, 1000×300 ground-truth image.

    Layout
    ──────
    Col 1 (  0– 99): X-Axis Reference  — white bg + 10-px black centre line
    Col 2 (100–199): Low Chirp Fwd     —  5 → 25 cycles  (top-to-bottom)
    Col 3 (200–299): Low Chirp Rev     — 25 →  5 cycles  (bottom-to-top mirror)
    """

    # ── Image Dimensions ──────────────────────────────────────────────────────
    height: int = 1000
    width:  int = 300      # 3 × 100-px columns

    # ── Column 1: X-Axis Reference (Vibration Tracker) ───────────────────────
    col1_start: int = 0
    col1_end:   int = 100
    # 10-px-wide black line centred at pixel 50 (cols 45–54)
    x_line_start: int = 45
    x_line_end:   int = 55
    x_tracker_expected_center: float = 49.5   # (45 + 54) / 2

    # ── Column 2: Low Chirp Forward  (5 → 25 cycles, top-to-bottom) ──────────
    col2_start: int   = 100
    col2_end:   int   = 200
    col2_f0:    float = 5.0
    col2_f1:    float = 25.0

    # ── Column 3: Low Chirp Reversed (25 → 5 cycles, bottom-to-top) ──────────
    # Same frequency range as col2 but mirrored: f0=25 at top, f1=5 at bottom.
    # The negative chirp rate (k = f1 − f0 = −20) is handled naturally by the
    # phase formula.  Together with col2 they form a complementary pair whose
    # instantaneous-frequency sum is constant (30 cy/img everywhere) and whose
    # difference linearly encodes row position.
    col3_start: int   = 200
    col3_end:   int   = 300
    col3_f0:    float = 25.0   # ← high frequency at top
    col3_f1:    float = 5.0    # ← low  frequency at bottom

# ─────────────────────────────────────────────────────────────────────────────
# Path Management
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def get_data_path(filename: str) -> str:
    """Ensures data directory exists and returns absolute path for a file."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    return os.path.join(DATA_DIR, filename)

# Standard File Paths
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
