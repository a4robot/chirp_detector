import numpy as np
import time
from src.config import ScanConfig
from src.creator import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer import DSPReconstructor

# "Cleverly" search N using geometric progression / powers of 2 
n_options = [2, 4, 6, 8, 12, 16] 
w_options = [10, 25, 50, 100]  
gaps = [10, 25, 50]

print(f"{'N':<3} | {'StrpW':<5} | {'GapW':<4} | {'Tot W':<5} | {'PSNR (dB)':<9} | {'MAE':<6}")
print("-" * 55)

best_psnr = 0
best_cfg = None

for n in n_options:
    for w in w_options:
        for gap in gaps:
            # Lock seed for apples-to-apples comparison on exactly the same distortion!
            np.random.seed(42) 
            
            try:
                cfg = ScanConfig(n_strips=n, strip_width=w, gap_width=gap)
                
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
print(f"Optimal Result: N={best_cfg[0]}, StripW={best_cfg[1]}, Gap={best_cfg[2]}, Tot Width={best_cfg[3]} -> {best_psnr} dB")
