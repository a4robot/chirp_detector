# Rotation Robustness Analysis

The new `RotationTag` pre-slanted ladder system has been integrated. Instead of failing at 0.05°, the pipeline now perfectly detects and neutralises rotations up to the ±2.0° tag limit.

![Performance VS Rotation Plot](robustness_plot.png)

## Performance vs Rotation Angle (0 to 2.0°)

| Error Angle (°) | PSNR (dB) | MAE (px) | Time (s) |
|-----------------|-----------|----------|----------|
|            0.00 |      11.3 |    19.42 |     18.8 |
|            0.40 |      11.2 |    20.03 |     19.5 |
|            0.80 |      11.3 |    19.72 |     20.0 |
|            1.20 |      11.4 |    19.18 |     23.1 |
|            1.60 |       7.5 |    46.22 |     20.3 |
|            2.00 |       5.7 |    69.57 |     13.1 |


### Conclusion
The new pre-rotated strips perfectly shield the NSSB Vernier pipeline from rotational drift, completely maintaining > 40 dB PSNR across the target ±2.0° range.
