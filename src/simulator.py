import numpy as np
import json
import os
from PIL import Image
from src.config import ScanConfig, DIST_NPY, DIST_PNG, ANS_KEY

class MechanicalChaosEngine:
    """Simulator to apply line-scan camera mechanical errors."""

    def __init__(self, config: ScanConfig):
        self.config = config

    def apply_mechanical_distortions(self, input_image: np.ndarray) -> (np.ndarray, list):
        """Simulate realistic scanning chaos on a perfect reference image."""
        h, w = input_image.shape
        target_length = h + np.random.randint(-50, 50)

        # 1. Compute Error Profiles (Y and X axis)
        motion_profile  = self._compute_motion_error_profile(h, target_length)
        vibration_shift = self._compute_vibration_profile(target_length)

        # 2. Resample Image (Y-axis)
        sampled_image = np.zeros((target_length, w))
        orig_y = np.arange(h)
        for col in range(w):
            sampled_image[:, col] = np.interp(motion_profile, orig_y, input_image[:, col])

        # 3. Resample Image (X-axis vibration)
        distorted_image = np.zeros((target_length, w), dtype=np.uint8)
        answer_key = []
        orig_x = np.arange(w)

        for i in range(target_length):
            shift = float(vibration_shift[i])
            # Correct the row with sub-pixel shift
            new_x = orig_x - shift
            row = sampled_image[i]
            distorted_image[i] = np.interp(
                new_x, orig_x, row,
                left=float(row[0]),    # replicate left edge pixel
                right=float(row[-1])   # replicate right edge pixel
            ).astype(np.uint8)

            # Record standard truth for validation
            answer_key.append({
                "distorted_line": i,
                "source_y_coord": float(motion_profile[i]),
                "x_shift": shift
            })

        return distorted_image, answer_key

    def _compute_motion_error_profile(self, input_height: int, 
                                     target_length: int) -> np.ndarray:
        """Internal helper for calculating chaotic Y-axis motion (slip/skip)."""
        steps = np.ones(target_length)
        
        # Add slip (0.2 - 0.4 region)
        steps[int(0.2*target_length):int(0.4*target_length)] = 0.5
        # Add skip (0.6 - 0.7 region)
        steps[int(0.6*target_length):int(0.7*target_length)] = 1.5
        
        # Add noise
        steps += np.random.normal(0, 0.05, target_length)
        steps = np.clip(steps, 0.1, 5.0)

        profile = np.cumsum(steps)
        profile = (profile - profile[0]) / (profile[-1] - profile[0]) * (input_height - 1)
        return profile

    def _compute_vibration_profile(self, length: int) -> np.ndarray:
        """Internal helper for calculating oscillatory X-axis mechanical vibration."""
        indices = np.arange(length)
        oscillation = 3 * np.sin(2 * np.pi * indices / 100)
        noise = np.random.normal(0, 0.5, length)
        # noise = np.zeros(length)
        return (oscillation + noise).astype(float)

def main():
    """CLI to apply distortions."""
    from src.config import REF_NPY
    if not os.path.exists(REF_NPY):
        print(f"Error: {REF_NPY} not found. Synthesize first.")
        return

    config = ScanConfig()
    engine = MechanicalChaosEngine(config)
    
    ref_image = np.load(REF_NPY)
    distorted_image, answer_key = engine.apply_mechanical_distortions(ref_image)
    
    np.save(DIST_NPY, distorted_image)
    Image.fromarray(distorted_image).save(DIST_PNG)
    with open(ANS_KEY, 'w') as f:
        json.dump(answer_key, f, indent=2)

    print(f"Simulation Complete: {distorted_image.shape}")
    print(f"Saved: {DIST_PNG}")

if __name__ == "__main__":
    main()
