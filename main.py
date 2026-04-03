"""
main.py — Chirp Detector Pipeline (Unified Entry-Point)
========================================================

Modes
-----
  classic   (default)
      Creator → MechanicalChaosEngine → DSPReconstructor
      Uses the built-in slip/skip/vibration simulator.

  dtw
      Creator → DTWDistortionEngine → DSPReconstructor
      Delegates distortion to the DTW workspace's apply_distortion
      (sinusoidal + random Y-stretch / X-oscillation + optional rotation).

Usage
-----
  python main.py                          # classic mode
  python main.py --mode dtw               # DTW distortion, moderate preset
  python main.py --mode dtw --preset chaos --seed 7
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
import json

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (ScanConfig, REF_NPY, REF_PNG,
                        DIST_NPY, DIST_PNG, ANS_KEY,
                        REST_NPY, REST_PNG, RECIPE)
from src.creator   import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer  import DSPReconstructor


# ── Step helpers ──────────────────────────────────────────────────────────────

def step_create(config: ScanConfig) -> np.ndarray:
    print("[1/3] Synthesizing Golden Reference (N-Strip Staggered Binary)...")
    synth     = ImageSynthesizer(config)
    reference = synth.synthesize_reference_image()
    np.save(REF_NPY, reference)
    Image.fromarray(reference).save(REF_PNG)
    print(f"      Saved → {REF_PNG}  shape={reference.shape}")
    return reference


def step_distort_classic(config: ScanConfig, reference: np.ndarray,
                          rotation_deg: float = 0.0):
    print(f"[2/3] Simulating Mechanical Chaos (Classic: Slip, Skip, Vibration, Rotation={rotation_deg:+.1f}°)...")
    engine            = MechanicalChaosEngine(config)
    distorted, truth  = engine.apply_mechanical_distortions(reference, rotation_deg=rotation_deg)
    np.save(DIST_NPY, distorted)
    Image.fromarray(distorted).save(DIST_PNG)
    with open(ANS_KEY, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"      Saved → {DIST_PNG}  shape={distorted.shape}")
    return distorted, truth


def step_distort_dtw(config: ScanConfig, reference: np.ndarray,
                     preset: str = "moderate", seed: int = 42):
    """Use DTW workspace distortion pipeline."""
    print(f"[2/3] Applying DTW Distortion  (preset={preset}, seed={seed})...")
    from src.simulator_dtw import DTWDistortionEngine
    engine            = DTWDistortionEngine(config, preset=preset, seed=seed)
    distorted, truth  = engine.apply_mechanical_distortions(reference)
    np.save(DIST_NPY, distorted)
    Image.fromarray(distorted).save(DIST_PNG)
    with open(ANS_KEY, "w") as f:
        json.dump(truth, f, indent=2)
    y_shifts = np.array([r["source_y_coord"] - r["distorted_line"] for r in truth])
    x_shifts = np.array([r["x_shift"] for r in truth])
    print(f"      Y-jitter  {y_shifts.min():.2f}…{y_shifts.max():.2f} px  |  "
          f"X-shift  {x_shifts.min():.2f}…{x_shifts.max():.2f} px")
    print(f"      Saved → {DIST_PNG}  shape={distorted.shape}")
    return distorted, truth


def step_analyze(config: ScanConfig, reference: np.ndarray, distorted: np.ndarray):
    print("[3/3] Executing Rotation-Tag DSP Reconstruction...")
    recon   = DSPReconstructor(config)

    # Step 2a/2b: decode rotation + extract y-phase
    y_phase = recon.restore_vertical_phase_mapping(distorted)

    # Step 1: x-tracker directly on the distorted image (dynamically follows slant)
    x_shifts = recon.detect_horizontal_vibration(distorted, y_phase)

    print("\n      [DEBUG] Pipeline internal vectors:")
    for i in range(10):
        print(f"      Row {i:4d}: y_map={y_phase[i]:.2f}, x_shift={x_shifts[i]:.2f}")

    # Reconstruct mathematically, then geometrically deform to orthogonal baseline
    recipe   = recon.generate_reconstruction_recipe(y_phase, x_shifts)
    restored = recon.reconstruct_perfect_image(distorted, recipe)

    np.save(REST_NPY, restored)
    Image.fromarray(restored).save(REST_PNG)

    metrics = recon.evaluate_restoration_quality(reference, restored)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(mode: str = "classic", preset: str = "moderate",
                 seed: int = 42, rotation_deg: float = 0.0):
    """Unified runner for the full Chirp Detector pipeline."""
    np.random.seed(seed)
    config = ScanConfig()

    # Step 1 – always the same
    reference = step_create(config)

    # Step 2 – distortion (classic, DTW, or passthrough for sanity-check)
    if mode == "classic":
        distorted, _ = step_distort_classic(config, reference, rotation_deg=rotation_deg)
    elif mode == "dtw":
        distorted, _ = step_distort_dtw(config, reference, preset=preset, seed=seed)
    elif mode == "passthrough":
        print("[2/3] Passthrough mode — using reference image as-is (no distortion).")
        distorted = reference.copy()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'classic', 'dtw', or 'passthrough'.")

    # Step 3 – always the same analyzer
    metrics = step_analyze(config, reference, distorted)

    print("\n" + "=" * 40)
    print(f"  Mode     : {mode}" + (f"  [{preset}]" if mode == "dtw" else ""))
    print(f"  Rotation : {rotation_deg:+.1f}°")
    print(f"  PSNR     : {metrics['psnr']} dB")
    print(f"  MAE      : {metrics['mae']}")
    print("=" * 40)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chirp Detector Pipeline")
    parser.add_argument("--mode",     choices=["classic", "dtw", "passthrough"], default="classic")
    parser.add_argument("--preset",   choices=["mild", "moderate", "chaos"],     default="moderate")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="Physical rotation to simulate in degrees (classic mode only)")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, preset=args.preset, seed=args.seed,
                 rotation_deg=args.rotation)
