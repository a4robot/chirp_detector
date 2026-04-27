import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from src.rotation_tag import detect_chirp_edges_typed, CHIRP_F0, CHIRP_FREQ

img = np.array(Image.open('data/vector_target-1.png').convert('L'))
H, W = img.shape

# just take a simple column where the left tag is
# left tag is at x=200 in the PDF, which is 200 * 16384/14975 = 218.8
col = int(round(200 * 16384/14975.0)) + 5

signal = img[:, col]
obs_pos, obs_types = detect_chirp_edges_typed(signal, pad_px=int(round(200 * 16384/14975.0)))

print(f"Signal len: {len(signal)}")
print(f"obs_pos length: {len(obs_pos)}")
print(f"First 5 types: {obs_types[:5]}")
print(f"First 5 pos: {obs_pos[:5]}")
print(f"Signal values around 21661: {signal[21650:21670]}")
