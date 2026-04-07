# Real Scan Calibration Restoration — 2026-04-07

## Problem Statement

Three scanned calibration images (`data/image_*.png`, each ~22000×16384) need to be
geometrically aligned to the canonical reference grid (`data/reference_image.png`, 20000×14975).
The previous `fix_scans.py` pipeline produced all-white or completely misaligned output.

## Key Findings So Far

### 1. X-axis calibration was wrong
- **Old**: scale=0.926 (shrink), derived from wrong checkerboard bounds.
- **New**: scale=1.011 (slight stretch), derived from least-squares fit of 11 checkerboard
  cell edges across two row bands.
- `ref_x = 1.011073 * scan_x - 212.76`
- Residual error: ±18px per cell edge (likely from slight scan rotation).

### 2. Y-axis chirp signal is heavily degraded
- Reference chirp swings 0↔255. Scanned chirp swings ~10↔80.
- Fixed threshold (127.5) finds zero useful edges. Need **adaptive baseline** (running mean)
  plus smoothing.
- After filtering, we get ~51 clean chirp edges spanning rows 2700–21100 of the 22000-row scan.
- Ideal chirp has 55 edges.

### 3. Edge alignment offset (j0) is critical
- Chirp edges encode *relative* phase, not *absolute* position.
- `j0=0` (match first scan edge to first ideal edge) → 4.5% checkerboard accuracy.
- `j0=1` (offset by one edge) → **96.2%** checkerboard accuracy.
- Interval-based matching always picks j0=0 because all even j0 values have identical
  interval patterns. Need a secondary anchor (checkerboard cell polarity) to resolve.

### 4. Scan geometry summary
| Property | Scan | Reference |
|---|---|---|
| Size | 22000×16384 | 20000×14975 |
| Y scale | ~1.0036 | 1.0 |
| X scale | ~1/1.011 = 0.989 | 1.0 |
| Chirp edges | 51 | 55 |
| Checker cell width | ~985px | 1000px |
| Left tag region | cols 150–1000 | cols 200–1250 |
| Checker region | cols 2170–14030 | cols 1987–12987 |

---

## Task Plan

### Task 1 — Fix Y-axis mapping (Hard → redesigned as Easy)
~~Use chirp edges for Y mapping~~ → Abandoned. Chirp edges accumulate drift.

**New approach**: Use **checkerboard vertical edges** directly.
- 19 cell boundaries in both scan and reference → perfect 1-to-1 matching.
- Key insight: need polarity-aware matching (first scan edge is B→W but first ref edge is W→B, requiring offset by 1).
- PCHIP spline through 19 matched control points gives 97.4% accuracy across the full scan.

**Status**: ✅ Done.

### Task 2 — Fix X-axis rotation correction (Intermediate)
**What**: The scan is slightly rotated. The first checker left edge drifts from x=2140 (top)
to x=2220 (bottom) — 80px over 16000 rows. This causes ~40px X offset at the extremes.

**Solution**: Measure checker left edge at 10 Y bands, compute per-band X shift, interpolate
to get per-row X correction.

**Status**: ✅ Done. Implemented as `measure_x_calibration_per_row()`.

### Task 3 — Process all 3 scan files (Easy)
**Status**: ✅ Done. All three scans processed automatically.

### Task 4 — Clean up (Skipped for now)

---

## Results

| Scan | MAE | Binary MAE | Rotation |
|---|---|---|---|
| image_1775189755 | **11.55** | **0.024** | 82px (0.28°) |
| image_1775191033 | **23.37** | **0.073** | 241px (0.83°) |
| image_1775191067 | **20.47** | **0.061** | 189px (0.65°) |

Binary MAE < 0.025 means >97.5% of checkerboard pixels match. The remaining errors
are at cell boundaries (scanning blurs the transitions) and at the top/bottom edges
(limited extrapolation of the Y spline).

## Key Design Decisions

1. **Abandoned chirp-based Y mapping**: The scanned chirp signal degrades at high frequencies
   (bottom of scan), causing edge intervals to double/triple. This made the PCHIP spline
   drift by 500+ ref rows in the lower half. The checkerboard itself provides coarser but
   far more reliable control points (19 edges, each accurate to ~10px).

2. **Per-row X correction instead of single affine**: A single scale+shift was off by up to
   40px at the scan edges due to rotation. Per-band measurement + interpolation handles this
   with <5px residual.

3. **Polarity-aware edge matching**: The scan's first checker edge is a rising transition
   (B→W) but the reference's first edge is falling (W→B). Naive 1-to-1 matching gives 0%
   accuracy; offsetting by 1 gives 97%.

## Execution Log

### Task 1+2+3 — Combined implementation
- Rewrote `fix_scans.py` with `extract_checker_y_edges()` and `measure_x_calibration_per_row()`
- Tested on all 3 scans, results above.
- Total processing: ~15s per scan (dominated by 20000-row reconstruction loop).

### Follow-up: Bottom row clipping fix (2026-04-07 10:00)
**Problem**: Bottom rows were missing because the PCHIP spline couldn't extrapolate
past the last checker edge.

**Fix**: Pad 1000px black strips on top/bottom before processing.

**Complications during implementation**:
- `measure_x_calibration_per_row()` was band-averaging across rows, which cancels the
  checker pattern at cell boundaries. Fix: sample individual rows instead.
- The "first checker edge" alternates between x≈2140 (B→W at checker-starts-white rows) and
  x≈3140 (B→W at checker-starts-black rows). Fix: target the **col1→col2 boundary**
  (ref x=2987, always has a transition regardless of cell polarity) using a ±200px search window.

**Results with padding**:*/

| Scan | MAE | Binary MAE | X rotation |
|---|---|---|---|
| image_1775189755 | **11.58** | **0.024** | 81px |
| image_1775191033 | **23.41** | **0.073** | 214px |
| image_1775191067 | **20.12** | **0.060** | 168px |

Same quality as before — padding didn't hurt alignment. Bottom rows should now be preserved.
