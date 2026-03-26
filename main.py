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
    """Unified runner for the full Chirp Detector pipeline."""
    config = ScanConfig()
    
    # ── Step 1: Synthesize Reference ──
    print("[1/3] Synthesizing Golden Reference Image...")
    synth = ImageSynthesizer(config)
    reference = synth.synthesize_reference_image()
    np.save(REF_NPY, reference)
    Image.fromarray(reference).save(REF_PNG)

    # ── Step 2: Simulate Mechanical Chaos ──
    print("[2/3] Simulating Mechanical Chaos (Distortions)...")
    engine = MechanicalChaosEngine(config)
    distorted, truth = engine.apply_mechanical_distortions(reference)
    np.save(DIST_NPY, distorted)
    Image.fromarray(distorted).save(DIST_PNG)
    with open(ANS_KEY, 'w') as f:
        json.dump(truth, f, indent=2)

    # ── Step 3: DSP Reconstruction ──
    print("[3/3] Executing DSP Reconstruction Pipeline...")
    recon = DSPReconstructor(config)
    
    # 3a. Recover shift and phase mapping
    x_shifts = recon.detect_horizontal_vibration(distorted)
    y_phase  = recon.restore_vertical_phase_mapping(distorted)
    
    # 3b. Generate recipe and reconstruct
    recipe   = recon.generate_reconstruction_recipe(y_phase, x_shifts)
    restored = recon.reconstruct_perfect_image(distorted, recipe)
    
    # 3c. Save and Validate
    np.save(REST_NPY, restored)
    Image.fromarray(restored).save(REST_PNG)
    with open(RECIPE, 'w') as f:
        json.dump(recipe, f, indent=2)

    metrics = recon.evaluate_restoration_quality(reference, restored)
    
    print("\n" + "="*40)
    print(" Pipeline Execution Complete ")
    print("="*40)
    print(f" PSNR: {metrics['psnr']} dB")
    print(f" MAE:  {metrics['mae']}")
    print(f" Output: {REST_PNG}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Chirp Detector Pipeline")
    args = parser.parse_args()
    
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[FATAL] Pipeline Failed: {str(e)}")
        sys.exit(1)
