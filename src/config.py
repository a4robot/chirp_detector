import os
from dataclasses import dataclass
import numpy as np

@dataclass
class ScanConfig:
    """
    Central configuration for the 4-strip staggered binary line-scan system.

    Layout (1000 rows × 1000 columns)
    ──────────────────────────────────
    Col  Pixels      Content
    X    0–  99     X-Axis Reference  : white bg + 10-px black centre line
    gap  100– 149   Black gap
     1   150– 249   Fwd  chirp  (phase 0)
    gap  250– 299   Black gap
     2   300– 399   Rev  chirp  (phase π/4)
    gap  400– 449   Black gap
     3   450– 549   Fwd  chirp  (phase π/2)
    gap  550– 599   Black gap
     4   600– 699   Rev  chirp  (phase 3π/4)
    gap  700– 999   Black

    The X-reference is on the LEFT, isolated from the chirp strips by a 50-px
    black gap. This is the same design as the 34.8 dB gold standard that was
    validated in the binary-vs-continuous benchmark.
    """

    height: int = 1000
    width:  int = 1000

    # ── X-Axis Reference (Vibration Tracker) ─────────────────────────────────
    col1_start:  int   = 0
    col1_end:    int   = 100
    x_line_start: int  = 45
    x_line_end:   int  = 55
    x_tracker_expected_center: float = 49.5

    # ── 4 Staggered Binary Chirp Strips ──────────────────────────────────────
    # phase-shifted to interleave edges for the Point-Pool decoder
    f0: float = 5.0
    f1: float = 25.0

    # computed in __post_init__
    strips: object = None   # list of (col_start, col_end, mode, phase_offset)

    def __post_init__(self):
        self.strips = [
            (150, 250, "fwd", 0.0),
            (300, 400, "rev", np.pi / 4),
            (450, 550, "fwd", np.pi / 2),
            (600, 700, "rev", 3 * np.pi / 4),
        ]


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
