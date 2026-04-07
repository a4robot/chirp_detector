import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ScanConfig
from src.creator import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer import DSPReconstructor

def main():
    os.makedirs('report_assets', exist_ok=True)
    
    # 1. Setup 165px Config
    cfg = ScanConfig()
    cfg.height = 1000
    cfg.col1_end = 15      # Tracker band
    cfg.x_line_start = 6
    cfg.x_line_end = 9
    cfg.first_gap = 0
    cfg.strip_width = 50
    cfg.gap_width = 50
    cfg.n_strips = 2
    cfg.vernier_frequencies = [7.0, 11.3]
    cfg.__post_init__()
    
    # We want exactly 165px width for the report slice
    actual_w = 165
    cfg.width = actual_w 
    
    # 2. Synthesize Reference
    synth = ImageSynthesizer(cfg)
    ref = synth.synthesize_reference_image()
    # Ensure it's exactly 165px (ImageSynthesizer might add bit of trailing gap)
    ref = ref[:, :actual_w]
    
    # 3. Apply Distortions
    engine = MechanicalChaosEngine(cfg)
    # Simulator internally re-samples based on h,w. 
    # Let's ensure it uses our exact ref dimensions.
    dist, ans = engine.apply_mechanical_distortions(ref)
    
    # 4. Reconstruct
    recon = DSPReconstructor(cfg)
    sx = recon.detect_horizontal_vibration(dist)
    sy = recon.restore_vertical_phase_mapping(dist)
    recipe = recon.generate_reconstruction_recipe(sy, sx)
    restored = recon.reconstruct_perfect_image(dist, recipe)
    restored = restored[:, :actual_w] # clip to original width
    
    # 5. Save Snippets
    # Let's pick a region with visible distortion (around row 300)
    y_start, y_end = 250, 450
    
    def save_box(img_data, name, label):
        # Crop
        crop = img_data[y_start:y_end, :]
        # Scale up for visibility in the report
        scale = 3
        h, w = crop.shape
        pil_img = Image.fromarray(crop).resize((w*scale, h*scale), Image.NEAREST)
        
        # Add Label
        draw = ImageDraw.Draw(pil_img)
        # Using default font (since we can't rely on specific ttf paths)
        draw.text((5, 5), label, fill=128)
        
        path = os.path.join('report_assets', f"{name}.png")
        pil_img.save(path)
        return path

    ref_path = save_box(ref, "reference", "Reference (165px GT)")
    dist_path = save_box(dist, "distorted", "Distorted (Mechanical Chaos)")
    rest_path = save_box(restored, "restored", "Restored (V3 DSP)")
    
    # 6. Create Combined Comparison
    # Create a side-by-side image
    W_sub = actual_w * 3
    H_sub = (y_end - y_start) * 3
    combined = Image.new('L', (W_sub * 3 + 20, H_sub), 255)
    
    img_ref = Image.open(ref_path)
    img_dist = Image.open(dist_path)
    img_rest = Image.open(rest_path)
    
    combined.paste(img_ref, (0, 0))
    combined.paste(img_dist, (W_sub + 10, 0))
    combined.paste(img_rest, (2 * W_sub + 20, 0))
    
    combined_path = os.path.join('report_assets', "comparison_165px.png")
    combined.save(combined_path)
    
    print(f"Report assets generated in report_assets/")
    print(f"Master comparison: {os.path.abspath(combined_path)}")

if __name__ == "__main__":
    main()
