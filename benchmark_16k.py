import numpy as np
import time
from src.config import ScanConfig
from src.creator import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer import DSPReconstructor

n_options = [1, 2, 3, 4, 5, 6] 
w_options = [10, 25, 50]  
gaps = [10, 25, 50]

print(f"Running 16k Benchmark - 16384px height (tight X-tracker)")
print(f"{'N':<3} | {'StrpW':<5} | {'GapW':<4} | {'Tot W':<5} | {'PSNR (dB)':<9} | {'MAE':<6}")
print("-" * 55)

best_psnr = -1
best_cfg = None

for n in n_options:
    for w in w_options:
        for gap in gaps:
            np.random.seed(42) 
            
            # Using 16384px height, and tight X-tracker (50px)
            cfg = ScanConfig(n_strips=n, strip_width=w, gap_width=gap)
            cfg.height = 16384
            # Keep X-tracker tight:
            cfg.col1_start = 0
            cfg.col1_end = 50
            cfg.x_line_start = 20
            cfg.x_line_end = 30
            cfg.x_tracker_expected_center = 24.5
            # Force config's post_init logic for strips/width update:
            cfg.__post_init__()
            
            try:
                synth = ImageSynthesizer(cfg)
                ref = synth.synthesize_reference_image()
                
                sim = MechanicalChaosEngine(cfg)
                dist, _ = sim.apply_mechanical_distortions(ref)
                
                recon = DSPReconstructor(cfg)
                sx = recon.detect_horizontal_vibration(dist)
                sy = recon.restore_vertical_phase_mapping(dist)
                rec = recon.generate_reconstruction_recipe(sy, sx)
                rest = recon.reconstruct_perfect_image(dist, rec)
                
                m = recon.evaluate_restoration_quality(ref, rest)
                
                print(f"{n:<3} | {w:<5} | {gap:<4} | {cfg.width:<5} | {m['psnr']:<9} | {m['mae']:<6}")
                
                if m['psnr'] > best_psnr:
                    best_psnr = m['psnr']
                    best_cfg = (n, w, gap, cfg.width)
            except Exception as e:
                print(f"{n:<3} | {w:<5} | {gap:<4} | ERROR: {str(e)}")

print("-" * 55)
print(f"Optimal 16k Result: N={best_cfg[0]}, StripW={best_cfg[1]}, Gap={best_cfg[2]}, Tot Width={best_cfg[3]} -> {best_psnr} dB")
