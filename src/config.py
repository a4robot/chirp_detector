import os
from dataclasses import dataclass

@dataclass
class ScanConfig:
    """Central configuration for image dimensions and tracker locations."""
    # Dimensions
    height: int = 1000
    width: int  = 500

    # Y-Axis Sine Tracker (Chirp)
    y_chirp_start: int  = 50
    y_chirp_end: int    = 150
    y_chirp_cycles: int = 50

    # X-Axis Vibration Tracker (Vertical Line)
    x_tracker_start: int = 300
    x_tracker_width: int = 5
    x_tracker_end: int   = 305  # x_tracker_start + x_tracker_width
    x_tracker_expected_center: float = 302.0

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
