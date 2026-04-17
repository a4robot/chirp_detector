"""
evaluate_classic.py
====================
Runs the full classic-mode pipeline and plots 2D evaluation graphs:
  - Y-Map tracking   : Ground truth vs Detected, and per-row Error
  - X-Tracker tracking: Ground truth vs Detected, and per-row Error
  - Chirp signal      : Raw signal, detected edges, and ideal edges
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config    import ScanConfig, REF_NPY, DIST_NPY, ANS_KEY, get_data_path
from src.creator   import ImageSynthesizer
from src.simulator import MechanicalChaosEngine
from src.analyzer  import DSPReconstructor
from src.rotation_tag import (
    build_layout, decode_rotation_angle,
    compute_ideal_edges, detect_chirp_edges,
    compute_ideal_edges_typed, detect_chirp_edges_typed,
    TARGET_MATCHED_EDGES,
    TOTAL_RIGHT_WIDTH, CHIRP_F0, CHIRP_FREQ,
)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_and_evaluate(seed: int = 42, rotation_deg: float = 0.0):
    np.random.seed(seed)
    config = ScanConfig()

    # ── Step 1: Reference ────────────────────────────────────────────────────
    print("[1/3] Synthesizing reference...")
    synth     = ImageSynthesizer(config)
    reference = synth.synthesize_reference_image()
    np.save(REF_NPY, reference)

    # ── Step 2: Classic Distortion ───────────────────────────────────────────
    print("[2/3] Applying classic distortion...")
    engine           = MechanicalChaosEngine(config)
    distorted, truth = engine.apply_mechanical_distortions(reference, rotation_deg=rotation_deg)
    np.save(DIST_NPY, distorted)
    with open(ANS_KEY, 'w') as f:
        json.dump(truth, f)

    n_dist = distorted.shape[0]
    gt_y   = np.array([r['source_y_coord'] for r in truth])
    gt_x   = np.array([r['x_shift']        for r in truth])
    rows   = np.arange(n_dist)

    # ── Extract chirp signal (typed) ─────────────────────────────────────────
    print("[*] Extracting chirp signal for analysis...")
    layout = build_layout(left_x0=200,
                          right_x0=config.width - TOTAL_RIGHT_WIDTH - 350 - 200)
    _, best_signal, _ = decode_rotation_angle(distorted, layout)

    obs_pos,   obs_types   = detect_chirp_edges_typed(best_signal, f0=CHIRP_F0, f1=CHIRP_FREQ,
                                                       ideal_H=config.height)
    ideal_pos, ideal_types = compute_ideal_edges_typed(f0=CHIRP_F0, f1=CHIRP_FREQ,
                                                        ideal_H=config.height)
    scale_y    = len(best_signal) / float(config.height)
    ideal_scan = ideal_pos * scale_y    # ideal edges in scan-row space

    n_r_obs = int(np.sum(obs_types == +1))
    n_f_obs = int(np.sum(obs_types == -1))
    print(f"    obs: R={n_r_obs} F={n_f_obs} total={len(obs_pos)}  "
          f"ideal R={int(np.sum(ideal_types==+1))} F={int(np.sum(ideal_types==-1))} "
          f"total={len(ideal_pos)}  (target even)")


    # ── Step 3: Reconstruction (may abort if edges ≠ target) ─────────────────
    print("[3/3] Running analyzer...")
    recon       = DSPReconstructor(config)
    recon_ok    = False
    y_det       = None
    x_det       = None
    y_rmse_val  = None
    x_rmse_val  = None

    try:
        y_phase  = recon.restore_vertical_phase_mapping(distorted)
        x_shifts = recon.detect_horizontal_vibration(distorted, y_phase)

        n = min(len(gt_y), len(y_phase), len(x_shifts))
        y_det      = y_phase[:n]
        x_det      = x_shifts[:n]
        gt_y_c     = gt_y[:n]
        gt_x_c     = gt_x[:n]
        rows_c     = rows[:n]
        y_rmse_val = rmse(y_det, gt_y_c)
        x_rmse_val = rmse(x_det, gt_x_c)
        recon_ok   = True
        print(f"  Y-map   RMSE : {y_rmse_val:.3f} px")
        print(f"  X-track RMSE : {x_rmse_val:.3f} px")

    except ValueError as e:
        print(f"\n  ⚠  RECONSTRUCTION ABORTED: {e}\n")

    # ── Plotting ─────────────────────────────────────────────────────────────
    n_rows_plot = 3
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle('Classic Mode — Algorithm Evaluation', fontsize=15,
                 fontweight='bold', y=1.01)

    # Subsample factor for large arrays
    k = max(1, n_dist // 4000)

    # ── Row 1 & 2: Tracking plots (only if reconstruction succeeded) ─────────
    ax_y    = fig.add_subplot(3, 2, 1)
    ax_yerr = fig.add_subplot(3, 2, 2)
    ax_x    = fig.add_subplot(3, 2, 3)
    ax_xerr = fig.add_subplot(3, 2, 4)

    if recon_ok:
        r = rows_c[::k]

        # Y-map tracking
        ax_y.plot(r, gt_y_c[::k],  label='Ground Truth', color='#2ecc71', lw=1.2)
        ax_y.plot(r, y_det[::k],   label='Detected',     color='#e74c3c', lw=1.0,
                  linestyle='--', alpha=0.8)
        ax_y.set_title('Y-Map Tracking (source row per distorted row)')
        ax_y.set_xlabel('Distorted Row')
        ax_y.set_ylabel('Source Row (px)')
        ax_y.legend(); ax_y.grid(True, alpha=0.25)

        y_err = y_det - gt_y_c
        ax_yerr.plot(r, y_err[::k], color='#e67e22', lw=0.8)
        ax_yerr.axhline(0, color='black', lw=0.6)
        ax_yerr.fill_between(r, y_err[::k], 0, alpha=0.2, color='#e67e22')
        ax_yerr.set_title(f'Y-Map Error   RMSE = {y_rmse_val:.2f} px')
        ax_yerr.set_xlabel('Distorted Row'); ax_yerr.set_ylabel('Error (px)')
        ax_yerr.grid(True, alpha=0.25)

        # X-tracker tracking
        ax_x.plot(r, gt_x_c[::k],  label='Ground Truth', color='#3498db', lw=1.2)
        ax_x.plot(r, x_det[::k],   label='Detected',     color='#9b59b6', lw=1.0,
                  linestyle='--', alpha=0.8)
        ax_x.set_title('X-Tracker Tracking (horizontal shift per row)')
        ax_x.set_xlabel('Distorted Row'); ax_x.set_ylabel('X Shift (px)')
        ax_x.legend(); ax_x.grid(True, alpha=0.25)

        x_err = x_det - gt_x_c
        ax_xerr.plot(r, x_err[::k], color='#8e44ad', lw=0.8)
        ax_xerr.axhline(0, color='black', lw=0.6)
        ax_xerr.fill_between(r, x_err[::k], 0, alpha=0.2, color='#8e44ad')
        ax_xerr.set_title(f'X-Tracker Error   RMSE = {x_rmse_val:.2f} px')
        ax_xerr.set_xlabel('Distorted Row'); ax_xerr.set_ylabel('Error (px)')
        ax_xerr.grid(True, alpha=0.25)
    else:
        for ax, title in [(ax_y, 'Y-Map Tracking'), (ax_yerr, 'Y-Map Error'),
                          (ax_x,  'X-Tracker Tracking'), (ax_xerr, 'X-Tracker Error')]:
            ax.text(0.5, 0.5, 'RECONSTRUCTION\nABORTED\n(edge match failed)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=13, color='red')
            ax.set_title(title)

    # ── Row 3: Chirp signal (full width) ─────────────────────────────────────
    ax_chirp = fig.add_subplot(3, 1, 3)

    H_sig   = len(best_signal)
    sig_rows = np.arange(H_sig)
    ks = max(1, H_sig // 8000)   # downsample for render speed

    # ── Build ideal binary chirp signal in scan-row space ───────────────────
    t_ideal   = np.linspace(0.0, 1.0, config.height)
    k_chirp   = CHIRP_FREQ - CHIRP_F0
    phi_ideal = 2.0 * np.pi * (CHIRP_F0 * t_ideal + 0.5 * k_chirp * t_ideal ** 2)
    ideal_sig = np.where(np.sin(phi_ideal) >= 0, 1.0, 0.0)   # binary 0/1

    # Stretch ideal to scan-row length (scale_y ≈ 1 for classic mode)
    ideal_sig_scan = np.interp(
        np.linspace(0, config.height - 1, H_sig),
        np.arange(config.height),
        ideal_sig
    )

    # ── Plot ideal fill FIRST (behind distorted) ─────────────────────────────
    r_ds       = sig_rows[::ks]
    ideal_ds   = ideal_sig_scan[::ks]
    distort_ds = best_signal[::ks].astype(float) / 255.0

    ax_chirp.fill_between(r_ds, ideal_ds, alpha=0.35,
                          color='#27ae60', label='Ideal chirp')
    ax_chirp.plot(r_ds, ideal_ds, color='#1a7a42', lw=0.6, alpha=0.5)

    # ── Distorted signal on top (gray) ────────────────────────────────────────
    ax_chirp.fill_between(r_ds, distort_ds, alpha=0.30,
                          color='#7f8c8d', label='Distorted chirp')
    ax_chirp.plot(r_ds, distort_ds, color='#555', lw=0.5, alpha=0.5)

    # ── Detected edges (typed) ────────────────────────────────────────────────
    obs_r_pos = obs_pos[obs_types == +1]
    obs_f_pos = obs_pos[obs_types == -1]
    first = True
    for oe in obs_r_pos * scale_y / scale_y:   # already in scan space
        lbl = f'Rising obs ({len(obs_r_pos)})' if first else None
        ax_chirp.axvline(oe, color='#2980b9', alpha=0.7, lw=1.0,
                         linestyle='-', label=lbl)
        first = False
    first = True
    for oe in obs_f_pos:
        lbl = f'Falling obs ({len(obs_f_pos)})' if first else None
        ax_chirp.axvline(oe, color='#e74c3c', alpha=0.7, lw=1.0,
                         linestyle='--', label=lbl)
        first = False

    # ── Ideal edges (typed) ───────────────────────────────────────────────────
    ideal_r_scan = ideal_pos[ideal_types == +1] * scale_y
    ideal_f_scan = ideal_pos[ideal_types == -1] * scale_y
    first = True
    for ie in ideal_r_scan:
        lbl = f'Rising ideal ({len(ideal_r_scan)})' if first else None
        ax_chirp.axvline(ie, color='#1abc9c', alpha=0.35, lw=0.7, label=lbl)
        first = False
    first = True
    for ie in ideal_f_scan:
        lbl = f'Falling ideal ({len(ideal_f_scan)})' if first else None
        ax_chirp.axvline(ie, color='#e67e22', alpha=0.35, lw=0.7,
                         linestyle=':', label=lbl)
        first = False

    parity_ok = (len(obs_pos) % 2 == 0)
    match_sym = '✓' if parity_ok else '✗'
    ax_chirp.set_title(
        f'Chirp Signal — Ideal (green) vs Distorted (gray)   '
        f'obs R={len(obs_r_pos)} F={len(obs_f_pos)} total={len(obs_pos)} {match_sym}(even)  '
        f'ideal R={int(np.sum(ideal_types==+1))} F={int(np.sum(ideal_types==-1))}',
        fontsize=10
    )

    ax_chirp.set_xlabel('Scan Row')
    ax_chirp.set_ylabel('Amplitude (normalised)')
    ax_chirp.set_xlim(0, H_sig)
    ax_chirp.set_ylim(-0.05, 1.25)
    ax_chirp.legend(loc='upper right', fontsize=8, ncol=2)
    ax_chirp.grid(True, alpha=0.2)


    plt.tight_layout()
    out_path = get_data_path('evaluation_classic.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\n  Saved → {out_path}")


if __name__ == '__main__':
    run_and_evaluate()
