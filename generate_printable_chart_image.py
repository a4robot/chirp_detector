import numpy as np
from PIL import Image
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ScanConfig
from src.creator import ImageSynthesizer

# Remove PIL image size limit for the giant 327MP calibration image
Image.MAX_IMAGE_PIXELS = None

def main():
    W = 16384
    H = 20000
    
    out_img = np.full((H, W), 255, dtype=np.uint8)
    section_h = H // 4
    
    configs_list = [
        # name, tracker_band, first_gap, strip_width, gap_width, n_strips
        ("1200px", 100, 100, 100, 80, 6),
        ("600px",  100,  50,  75,  50, 4),
        ("300px",   50,  50,  75,  50, 2),
        ("165px",   15,   0,  50,  50, 2)
    ]
    
    for i, (name, t_band, f_gap, s_w, g_w, n_str) in enumerate(configs_list):
        print(f"Synthesizing section {i+1}/4: {name} footprint...")
        
        cfg = ScanConfig()
        cfg.height = section_h
        cfg.col1_start = 0
        cfg.col1_end = t_band
        
        # Center the tracking line within the tracker_band
        cfg.x_line_start = t_band // 2 - 2
        if cfg.x_line_start < 0: cfg.x_line_start = 0
        cfg.x_line_end = cfg.x_line_start + min(4, t_band)
        
        cfg.first_gap = f_gap
        cfg.strip_width = s_w
        cfg.gap_width = g_w
        cfg.n_strips = n_str
        
        # Distribute Vernier frequencies nicely depending on N
        cfg.vernier_frequencies = [7.0, 11.3, 14.5, 17.8, 21.0, 25.0][:n_str]
        
        cfg.__post_init__()
        
        synth = ImageSynthesizer(cfg)
        ref_block = synth.synthesize_reference_image()
        
        actual_w = ref_block.shape[1]
        
        row_start = i * section_h
        row_end = (i + 1) * section_h
        
        # Paste on left side
        out_img[row_start:row_end, 0:actual_w] = ref_block
        
        # Paste on right side (mirrored horizontally)
        mirrored_block = np.fliplr(ref_block)
        out_img[row_start:row_end, W - actual_w : W] = mirrored_block
        
        # Add a black solid line divider between sections
        if i < 3:
            out_img[row_end-10:row_end, :] = 0
            
    os.makedirs('figures', exist_ok=True)
    save_path = "figures/printable_calibration_chart_16k.png"
    
    print(f"Assembling target image ({W}x{H} pixels)...")
    # Save as PNG. This will take ~5-15 seconds and result in a 5-20MB compressed file.
    img = Image.fromarray(out_img)
    img.save(save_path, optimize=True)
    print(f"Successfully saved printable target to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()
