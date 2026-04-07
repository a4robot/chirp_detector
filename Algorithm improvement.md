## Algorithm improvement.
### 1. The Anatomy of a Slanted Chirp

When the workpiece is rotated by angle $\theta$, the camera is essentially slicing diagonally through the printed pattern. This causes two primary extraction failures:

* **The "Walk-Off" (X-Boundary Breach):** Your Vernier strips are ultra-lean (50px wide). If the pattern is rotated by just 2 degrees, the strip will laterally shift by ~35 pixels for every 1000 rows scanned. Very quickly, a static vertical X-Tracker will stop reading the black-and-white chirp and start reading the empty gap—or worse, bleed into the adjacent strip.
* **The Diagonal Phase Slicing:** The magic of the $\pi/N$ phase stagger assumes you are sampling the strips horizontally across the same physical $Y$-coordinate. If you extract data vertically from a slanted pattern, you are sampling Strip 1 at a different relative physical depth than Strip 2.

### 2. Extending NSSB: Dynamic Slant Extraction

To detect the chirp accurately despite the rotation, you don't need to change the core Point-Pool fusion. You just need to change **how you extract the 1D signal from the 2D image buffer**. You must abandon static vertical columns and implement **Slanted Vector Extraction**.

#### Step A: Dynamic Raycasting
Instead of pulling a straight vertical array `image[:, x_start:x_end]`, you need your X-Tracker to dictate a slanted extraction path. If you know the rotation angle $\theta$ (e.g., from the linear detrending slope $m$ discussed previously), the center coordinate of your strip at any given row $y$ becomes a dynamic function:

$$x(y) = x_0 + y \cdot \tan(\theta)$$

You then extract your 50px strip dynamically around $x(y)$ for each row.

#### Step B: Sub-Pixel Interpolation
Because a slanted line will not fall perfectly on integer pixel coordinates, you cannot simply use array slicing. You must use **Bilinear Interpolation** to sample the pixel intensities along the slanted extraction path. This ensures the edges of your binary chirp remain sharp and aren't heavily aliased into gray mush by the diagonal cut.

#### Step C: The Cosine Stretch
Once you have cleanly extracted the 1D signals along the slanted path, your strips are perfectly isolated again. The relative $\pi/N$ phase stagger is mathematically preserved along this new axis. 

The only remaining artifact is that the physical distance between the edges has been stretched. You simply update your ideal Point-Pool reference frequencies before running the sliding-window offset match:

$$f_{new} = f_{original} \cdot \cos(\theta)$$

---

By extracting along the slant, the Point-Pool algorithm gets fed exactly what it expects: clean, phase-locked, 1D binary square waves. 

To implement this dynamic extraction, **does your system process the line-scan buffer in real-time (row-by-row chunks), or do you capture the entire continuous image into memory before running the NSSB extraction?**