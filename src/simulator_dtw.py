"""
simulator_dtw.py — DTW-powered Distortion Engine
=================================================
Bridges the chirp_detector pipeline with the rich distortion library
from the ``dtw`` workspace (``/Users/bpasu/Documents/dtw``).

The ``DTWDistortionEngine`` drives the same interface as the original
``MechanicalChaosEngine``:

    distorted_image, answer_key = engine.apply_mechanical_distortions(ref)

…but delegates the actual pixel-level warping to
``dtw.utils.image_utils.apply_distortion``, which supports:

  * Rotation
  * Sinusoidal Y-stretch   (slow swell)
  * Random Y-stretch       (slip / skip)
  * Sinusoidal X-oscillation (smooth vibration)
  * Random X-oscillation   (jitter events)

Usage (from chirp_detector root):
    python main.py --mode dtw
"""

import sys
import os
import numpy as np
from PIL import Image

# ── Path bootstrap: make the dtw workspace importable ─────────────────────────
_DTW_WORKSPACE = "/Users/bpasu/Documents/dtw"
_DTW_SRC       = os.path.join(_DTW_WORKSPACE, "src")

if _DTW_SRC not in sys.path:
    sys.path.insert(0, _DTW_SRC)

try:
    from dtw.utils.image_utils import apply_distortion   # noqa: E402
    _DTW_AVAILABLE = True
except ImportError as _e:
    _DTW_AVAILABLE = False
    _DTW_IMPORT_ERROR = str(_e)

from src.config import ScanConfig, DIST_NPY, DIST_PNG, ANS_KEY


# ── DTW Distortion Presets ────────────────────────────────────────────────────

PRESETS = {
    # Name           : kwargs forwarded to apply_distortion
    "mild": dict(
        angle_deg   = 0.5,
        use_sin_y   = True,  sin_y_params  = {"max_stretch": 3.0,  "freq": 0.015},
        use_sin_x   = True,  sin_x_params  = {"max_oscillation": 2.0, "freq": 0.02},
    ),
    "moderate": dict(
        angle_deg   = 1.0,
        use_sin_y   = True,  sin_y_params  = {"max_stretch": 8.0,  "freq": 0.012},
        use_rand_y  = True,  rand_y_params = {"max_stretch": 4.0,  "slip_probability": 60},
        use_sin_x   = True,  sin_x_params  = {"max_oscillation": 3.0, "freq": 0.018},
        use_rand_x  = True,  rand_x_params = {"max_oscillation": 1.5, "slip_probability": 80},
    ),
    "chaos": dict(
        angle_deg   = 2.0,
        use_sin_y   = True,  sin_y_params  = {"max_stretch": 15.0, "freq": 0.010},
        use_rand_y  = True,  rand_y_params = {"max_stretch": 10.0, "slip_probability": 30},
        use_sin_x   = True,  sin_x_params  = {"max_oscillation": 5.0, "freq": 0.025},
        use_rand_x  = True,  rand_x_params = {"max_oscillation": 3.0, "slip_probability": 40},
    ),
}


class DTWDistortionEngine:
    """
    Distortion engine powered by the DTW workspace's ``apply_distortion``.

    Parameters
    ----------
    config : ScanConfig
        The shared pipeline config (carries image dimensions).
    preset : str
        One of ``"mild"``, ``"moderate"``, ``"chaos"``.  Controls the
        blend of sinusoidal and random distortions applied.
    seed : int or None
        Random seed for reproducible runs.
    """

    def __init__(self, config: ScanConfig, preset: str = "moderate", seed: int = None):
        if not _DTW_AVAILABLE:
            raise ImportError(
                f"DTW workspace not importable from '{_DTW_SRC}'.\n"
                f"Original error: {_DTW_IMPORT_ERROR}\n\n"
                "Make sure the dtw workspace venv dependencies are installed:\n"
                "  cd /Users/bpasu/Documents/dtw && pip install -r requirements.txt"
            )
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")

        self.config = config
        self.preset = preset
        self.seed   = seed

    # ── Public API (same signature as MechanicalChaosEngine) ─────────────────

    def apply_mechanical_distortions(self, input_image: np.ndarray):
        """
        Apply DTW-based distortions and return (distorted_image, answer_key).

        The answer_key is a list of per-row dicts compatible with the
        existing analyzer and benchmark scripts:

            {
              "distorted_line": int,
              "source_y_coord": float,   # row offset in original image
              "x_shift":        float,   # horizontal shift applied (px)
            }
        """
        kwargs = dict(PRESETS[self.preset])   # copy so we don't mutate

        if hasattr(self, 'rotation_angle'):
            kwargs['angle_deg'] = self.rotation_angle

        if self.seed is not None:
            kwargs["seed"] = self.seed

        kwargs["return_flow"] = False         # we use jitter arrays directly

        # apply_distortion returns (stretched, jitter_y, jitter_x)
        distorted, jitter_y, jitter_x = apply_distortion(input_image, **kwargs)

        # Ensure correct dtype
        distorted = np.clip(distorted, 0, 255).astype(np.uint8)

        # Build answer_key
        n = distorted.shape[0]
        answer_key = []
        for i in range(n):
            answer_key.append({
                "distorted_line": i,
                "source_y_coord": float(i + jitter_y[i]),
                "x_shift":        float(jitter_x[i]),
            })

        return distorted, answer_key

    # ── Convenience: run & save (mirrors simulator.py's main()) ──────────────

    def run_and_save(self, ref_npy_path: str):
        """Load reference .npy, distort with DTW, save outputs."""
        import json
        ref_image = np.load(ref_npy_path)
        distorted, answer_key = self.apply_mechanical_distortions(ref_image)

        np.save(DIST_NPY, distorted)
        Image.fromarray(distorted).save(DIST_PNG)
        with open(ANS_KEY, "w") as f:
            json.dump(answer_key, f, indent=2)

        print(f"[DTW] Distortion preset   : {self.preset}")
        print(f"[DTW] Distorted shape     : {distorted.shape}")
        print(f"[DTW] Y-jitter range      : "
              f"{np.array([r['source_y_coord'] - r['distorted_line'] for r in answer_key]).min():.2f}"
              f" … "
              f"{np.array([r['source_y_coord'] - r['distorted_line'] for r in answer_key]).max():.2f} px")
        print(f"[DTW] X-shift range       : "
              f"{min(r['x_shift'] for r in answer_key):.2f}"
              f" … "
              f"{max(r['x_shift'] for r in answer_key):.2f} px")
        print(f"[DTW] Saved → {DIST_PNG}")
        return distorted, answer_key


# ── CLI entry-point ────────────────────────────────────────────────────────────

def main():
    import argparse, json
    from src.config import REF_NPY

    parser = argparse.ArgumentParser(description="DTW-powered chirp distortion")
    parser.add_argument("--preset", choices=list(PRESETS), default="moderate",
                        help="Distortion severity preset (default: moderate)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if not os.path.exists(REF_NPY):
        print(f"Error: {REF_NPY} not found. Run the creator first.")
        return

    cfg    = ScanConfig()
    engine = DTWDistortionEngine(cfg, preset=args.preset, seed=args.seed)
    engine.run_and_save(REF_NPY)


if __name__ == "__main__":
    main()
