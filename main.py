import sys
import os
import argparse
import numpy as np
from PIL import Image
import json

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ScanConfig, REF_NPY, REF_PNG, DIST_NPY, DIST_PNG, ANS_KEY, REST_NPY, REST_PNG, RECIPE
from src.creator import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer import DSPReconstructor

def run_pipeline():
    """Unified runner for the full high-precision Chirp Detector pipeline."""
    # LOCK SEED FOR DETERMINISTIC COMPARISON
    np.random.seed(42)
    
    config = ScanConfig()
    
    # ── Step 1: Synthesize Reference ──
    print("[1/3] Synthesizing Golden Reference (4-Strip Staggered Binary)...")
    synth = ImageSynthesizer(config)
    reference = synth.synthesize_reference_image()
    np.save(REF_NPY, reference)
    Image.fromarray(reference).save(REF_PNG)

    # ── Step 2: Simulate Mechanical Chaos ──
    print("[2/3] Simulating Mechanical Chaos (Slip, Skip, Vibration)...")
    engine = MechanicalChaosEngine(config)
    distorted, truth = engine.apply_mechanical_distortions(reference)
    np.save(DIST_NPY, distorted)
    Image.fromarray(distorted).save(DIST_PNG)
    with open(ANS_KEY, 'w') as f:
        json.dump(truth, f, indent=2)

    # ── Step 3: DSP Point-Pool Reconstruction ──
    print("[3/3] Executing DSP Point-Pool Reconstruction...")
    recon = DSPReconstructor(config)
    
    x_shifts = recon.detect_horizontal_vibration(distorted)
    y_phase  = recon.restore_vertical_phase_mapping(distorted)
    
    recipe   = recon.generate_reconstruction_recipe(y_phase, x_shifts)
    restored = recon.reconstruct_perfect_image(distorted, recipe)
    
    np.save(REST_NPY, restored)
    Image.fromarray(restored).save(REST_PNG)
    
    metrics = recon.evaluate_restoration_quality(reference, restored)
    
    print("\n" + "="*40)
    print(f" PSNR: {metrics['psnr']} dB")
    print(f" MAE:  {metrics['mae']}")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
