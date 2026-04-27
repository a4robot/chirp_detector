import os
from dataclasses import dataclass
import numpy as np

# Canonical design width in pixels (457/500 * 16384 = 14974.976).
# All horizontal layout fields below are calibrated against this width;
# scaling against this constant adapts the geometry to arbitrary scan widths.
DESIGN_WIDTH_PX: int = 14975

@dataclass
class ScanConfig:
    """
    Central configuration for the N-strip staggered binary line-scan system.
    Dynamically generates the layout based on strip quantity and dimensions.
    """
    height: int = 20000
    width:  int = DESIGN_WIDTH_PX

    # ── Right X-Tracker (Horizontal Vibration Tracker) ───────────────────────
    # Shifted left by 1409 px from 16384 baseline
    col1_start:  int   = 13427
    col1_end:    int   = 13577
    x_line_start: int  = 13492
    x_line_end:   int  = 13512
    x_tracker_expected_center: float = 13502.0

    # ── N Staggered Binary Chirp Strips ──────────────────────────────────────
    f0: float = 5.0
    f1: float = 25.0
    
    # ── Space Optimization Settings ──────────────────────────────────────────
    n_strips: int = 6
    strip_width: int = 50
    gap_width: int = 50
    first_gap: int = 0        # Gap between x-tracker and first strip (0 = adjacent)

    # ── Vernier Multi-Frequency Effect ───────────────────────────────────────
    vernier_frequencies: list = None

    strips: list = None

    def __post_init__(self):
        self.strips = []
        current_x = self.col1_end + self.first_gap
        
        for i in range(self.n_strips):
            # Alternating direction makes adjacent frequencies somewhat distinct.
            mode = "fwd" if i % 2 == 0 else "rev"
            
            # Phase staggering
            phase = i * np.pi / self.n_strips
            
            # Vernier frequency mapping
            f1_strip = self.f1
            if self.vernier_frequencies and i < len(self.vernier_frequencies):
                f1_strip = self.vernier_frequencies[i]
                
            self.strips.append((current_x, current_x + self.strip_width, mode, phase, f1_strip))
            current_x += self.strip_width + self.gap_width

    def scale(self, factor: float) -> "ScanConfig":
        """
        Returns a new ScanConfig instance with all horizontal/vertical pixel parameters
        scaled by the given factor.
        """
        from copy import deepcopy
        scaled = deepcopy(self)
        scaled.width = int(round(self.width * factor))
        scaled.height = int(round(self.height * factor))

        scaled.col1_start = int(round(self.col1_start * factor))
        scaled.col1_end = int(round(self.col1_end * factor))
        scaled.x_line_start = int(round(self.x_line_start * factor))
        scaled.x_line_end = int(round(self.x_line_end * factor))
        scaled.x_tracker_expected_center = self.x_tracker_expected_center * factor

        scaled.strip_width = int(round(self.strip_width * factor))
        scaled.gap_width = int(round(self.gap_width * factor))
        scaled.first_gap = int(round(self.first_gap * factor))

        # Re-run post_init to regenerate the staggered strips with scaled widths/gaps
        scaled.__post_init__()

        return scaled

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
