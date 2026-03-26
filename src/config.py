import os
from dataclasses import dataclass

@dataclass
class ScanConfig:
    """Central configuration for the 4-column, 1000x400 ground-truth image."""

    # ── Image Dimensions ──────────────────────────────────────────────────────
    height: int = 1000
    width:  int = 400      # 4 × 100-px columns

    # ── Column 1: X-Axis Reference (Vibration Tracker) ───────────────────────
    # Solid white (255) background with a 10-px-wide black (0) line at centre.
    col1_start: int = 0
    col1_end:   int = 100
    # Black line: columns 45-54 (centre of a 100-px band → pixel 50 is centre)
    x_line_start: int = 45
    x_line_end:   int = 55   # centre = 50.0
    x_tracker_expected_center: float = 49.5   # centre of cols 45-54: (45+54)/2

    # ── Columns 2-4: Spatial Chirp Strips ────────────────────────────────────
    # All three strips are grayscale (full-amplitude cosine, NOT binarised)
    # so that the intensity alone encodes position — perfect for a reference.
    #
    # Column 2 (Low):  5  → 25 cycles/image
    col2_start: int   = 100
    col2_end:   int   = 200
    col2_f0:    float = 5.0
    col2_f1:    float = 25.0

    # Column 3 (Mid): 15 → 35 cycles/image
    col3_start: int   = 200
    col3_end:   int   = 300
    col3_f0:    float = 15.0
    col3_f1:    float = 35.0

    # Column 4 (High): 25 → 45 cycles/image
    col4_start: int   = 300
    col4_end:   int   = 400
    col4_f0:    float = 25.0
    col4_f1:    float = 45.0

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
