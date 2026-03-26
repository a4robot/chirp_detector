import numpy as np
from PIL import Image
from src.config import ScanConfig, REF_NPY, REF_PNG

class ImageSynthesizer:
    """Class to craft the 'Gold Standard' reference image."""

    def __init__(self, config: ScanConfig):
        self.config = config

    def synthesize_reference_image(self) -> np.ndarray:
        """Create a perfect image with embedded Y and X tracking signals."""
        image = np.zeros((self.config.height, self.config.width), dtype=np.uint8)

        # 1. Overlay Y-Axis Tracker (Speed/Phase)
        y_wave = self._generate_y_phase_signal()
        image[:, self.config.y_chirp_start:self.config.y_chirp_end] = y_wave[:, np.newaxis]

        # 2. Overlay X-Axis Tracker (Vibration)
        image[:, self.config.x_tracker_start:self.config.x_tracker_end] = 255

        return image

    def _generate_y_phase_signal(self) -> np.ndarray:
        """Internal helper for exact phase-shifted sine wave generation."""
        t = np.linspace(0, 1, self.config.height, endpoint=False)
        wave = (np.sin(2 * np.pi * self.config.y_chirp_cycles * t) + 1) / 2 * 255
        return wave.astype(np.uint8)

def main():
    """CLI to generate/save reference image."""
    synth = ImageSynthesizer(ScanConfig())
    ref_image = synth.synthesize_reference_image()

    np.save(REF_NPY, ref_image)
    Image.fromarray(ref_image).save(REF_PNG)

    print(f"Synthesized Reference Image: {ref_image.shape}")
    print(f"Saved: {REF_NPY}")

if __name__ == "__main__":
    main()
