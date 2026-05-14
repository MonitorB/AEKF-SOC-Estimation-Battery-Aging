"""
parameter_identification_r1c1r2c2.py
=====================================
Extract R1, C1, R2, C2 from HPPC rest segments (post-1C discharge pulse)
for each SOH × SOC point.

Works with both HPPC structures:
  - 7 segments/SOC  (discharge-only, 300s or 600s rests)
  - 11 segments/SOC (discharge + charge pulses, 600s rests)

Fitting model (amplitude-based, original approach):
    V(t) = V_final - A1*exp(-t/tau1) - A2*exp(-t/tau2)

    then:  R1 = A1 / I_pulse,  C1 = tau1 / R1
           R2 = A2 / I_pulse,  C2 = tau2 / R2

Outputs:
    2rc_parameters/ecm_r1c1r2c2_2d.csv
    2rc_parameters/fit_plots/fit_quality_SOHxx.png
    Pop-up figures: R1, R2, C1, C2 vs SOC (3+2 layout) + averaged vs SOH (2×2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec

# ──────────────────────────── Configuration ────────────────────────────
HPPC_DIR = "hppc_data"
OUT_DIR  = "2rc_parameters"
PLOT_DIR = os.path.join(OUT_DIR, "fit_plots")

SOH_LIST = [100, 95, 90, 85, 80]
SOC_NOMINAL = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# Bounds for amplitude-based fitting: A1, tau1, A2, tau2, V_final
# tau1 (fast) ∈ [1, 50]s,  tau2 (slow) ∈ [50, 600]s
BOUNDS_LOWER = [0.001, 1.0,  0.001, 50.0,  3.0]
BOUNDS_UPPER = [0.5,   50.0, 0.5,   600.0, 4.5]

colors  = {100: '#1f77b4', 95: '#ff7f0e', 90: '#2ca02c', 85: '#d62728', 80: '#9467bd'}
markers = {100: 'o',       95: 's',       90: '^',       85: 'D',       80: 'v'}

# ──────────────────────────── Helpers ──────────────────────────────────

def two_rc_relaxation(t, A1, tau1, A2, tau2, V_final):
    """Voltage relaxation model (amplitude-based)."""
    return V_final - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)


def segment_data(time_s, current, threshold=0.05):
    """Split HPPC time series into constant-current segments."""
    segments = []
    seg_start = 0
    for i in range(1, len(current)):
        if abs(current[i] - current[i - 1]) > threshold:
            segments.append({
                'start_idx': seg_start,
                'end_idx': i - 1,
                'start_t': time_s[seg_start],
                'end_t': time_s[i - 1],
                'duration': time_s[i - 1] - time_s[seg_start],
                'mean_current': np.mean(current[seg_start:i]),
            })
            seg_start = i
    segments.append({
        'start_idx': seg_start,
        'end_idx': len(current) - 1,
        'start_t': time_s[seg_start],
        'end_t': time_s[-1],
        'duration': time_s[-1] - time_s[seg_start],
        'mean_current': np.mean(current[seg_start:]),
    })
    return segments


def detect_structure(segments):
    """
    Auto-detect HPPC structure from segment count.
    Returns segments-per-SOC (7 or 11).
    """
    # Total segments = 1 (initial charge) + N_soc * segs_per_soc
    n = len(segments) - 1  # exclude initial segment
    if n % 11 == 0 and n // 11 == 9:
        return 11
    elif n % 7 == 0 and n // 7 == 9:
        return 7
    else:
        # Fallback: try both
        print(f"  [WARN] Unexpected segment count: {len(segments)}, trying to auto-detect...")
        if n >= 99:  # 9 * 11
            return 11
        else:
            return 7


def get_post1C_rest_indices(segments, segs_per_soc):
    """
    Return (rest_segment_index, pulse_segment_index) for the rest after
    the 1C discharge pulse at each SOC level.

    7-seg structure:  0.5C, rest, 1C, REST, 2C, rest, stepdown
                      offsets: 0,  1,   2,   3,   4,   5,   6
                      → rest after 1C = offset 3, pulse = offset 2

    11-seg structure: 0.5C, rest, 1C, REST, 2C, rest, 0.5Cchg, rest, 1Cchg, rest, stepdown
                      offsets: 0,  1,   2,   3,   4,   5,   6,      7,    8,     9,    10
                      → rest after 1C dis = offset 3, pulse = offset 2
    """
    indices = []
    for group in range(9):
        base = 1 + group * segs_per_soc
        pulse_idx = base + 2   # 1C discharge pulse
        rest_idx  = base + 3   # rest after 1C discharge
        if rest_idx < len(segments):
            indices.append((rest_idx, pulse_idx))
    return indices


def fit_rest_segment(time_s, voltage, seg, I_pulse_mag):
    """
    Fit amplitude-based 2RC model to a single rest segment.
    Returns (R1, C1, R2, C2, t, fit_v, actual_v, success).
    """
    si, ei = seg['start_idx'], seg['end_idx'] + 1
    t_raw = time_s[si:ei]
    v_raw = voltage[si:ei]

    t = t_raw - t_raw[0]
    v = v_raw.copy()

    V_end = v[-1]
    V_start = v[0]
    delta_V = V_end - V_start
    A1_0 = delta_V * 0.4
    A2_0 = delta_V * 0.6
    p0 = [A1_0, 10.0, A2_0, 150.0, V_end]

    try:
        popt, _ = curve_fit(
            two_rc_relaxation, t, v,
            p0=p0,
            bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
            maxfev=20000,
        )
        A1, tau1, A2, tau2, V_final = popt

        # Ensure tau1 < tau2 (fast pair first)
        if tau1 > tau2:
            A1, tau1, A2, tau2 = A2, tau2, A1, tau1

        R1 = A1 / I_pulse_mag
        C1 = tau1 / R1 if R1 > 1e-9 else 0.0
        R2 = A2 / I_pulse_mag
        C2 = tau2 / R2 if R2 > 1e-9 else 0.0

        fit_v = two_rc_relaxation(t, *popt)
        return R1, C1, R2, C2, t, fit_v, v, True

    except Exception as e:
        print(f"    Fit failed: {e}")
        return 0, 0, 0, 0, t, np.full_like(t, np.nan), v, False


def plot_parameter_grid(result_df, col, label, unit, scale):
    """3-top 2-bottom subplot layout for parameter vs SOC."""
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f"{label} vs SOC — Individual SOH States",
                 fontsize=14, fontweight='bold')

    ax_positions = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
                    gs[1, 1:3], gs[1, 3:5]]

    for i, soh in enumerate(SOH_LIST):
        ax = fig.add_subplot(ax_positions[i])
        sub = result_df[result_df['SOH'] == soh].sort_values('SOC', ascending=False)
        ax.plot(sub['SOC'], sub[col] * scale,
                marker=markers[soh], color=colors[soh],
                linewidth=1.8, markersize=7)
        ax.set_title(f"SOH {soh}%", fontsize=12, fontweight='bold')
        ax.set_xlabel('SOC', fontsize=10)
        ax.set_ylabel(f"{label} [{unit}]", fontsize=10)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(SOC_NOMINAL)


def plot_avg_vs_soh(result_df):
    """2×2 grid: averaged R1, C1, R2, C2 vs SOH."""
    avg_df = result_df.groupby('SOH').mean(numeric_only=True).reset_index()
    avg_df = avg_df.sort_values('SOH', ascending=False)

    avg_params = [
        ('R1_ohm', 'R1 (Ohm) vs SOH',  'R1 (Ohm)', '#1f77b4'),
        ('C1_F',   'C1 (F) vs SOH',     'C1 (F)',   '#ff7f0e'),
        ('R2_ohm', 'R2 (Ohm) vs SOH',   'R2 (Ohm)', '#2ca02c'),
        ('C2_F',   'C2 (F) vs SOH',     'C2 (F)',   '#d62728'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("ECM Parameters vs SOH (Averaged Across SOC)",
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for ax, (col, title, ylabel, color) in zip(axes, avg_params):
        ax.plot(avg_df['SOH'], avg_df[col],
                marker='o', color=color, linewidth=1.8, markersize=8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('SOH (%)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(SOH_LIST)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ──────────────────────────── Main ─────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_rows = []

    for soh in SOH_LIST:
        fname = os.path.join(HPPC_DIR, f"HPPC_SOH{soh}.csv")
        if not os.path.isfile(fname):
            print(f"[WARN] File not found: {fname}")
            continue

        print(f"\n{'='*60}")
        print(f"  Processing SOH = {soh}%")
        print(f"{'='*60}")

        df = pd.read_csv(fname)
        time_s  = df['Time [h]'].values * 3600
        current = df['Current [A]'].values
        voltage = df['Voltage [V]'].values

        segments = segment_data(time_s, current)
        segs_per_soc = detect_structure(segments)
        print(f"  Detected {segs_per_soc} segments per SOC level")

        rest_pulse_indices = get_post1C_rest_indices(segments, segs_per_soc)

        # ── Fit-quality figure (saved to folder) ──
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        fig.suptitle(f"2RC Fit Quality — SOH {soh}%", fontsize=14, fontweight='bold')
        axes = axes.flatten()

        for j, ((rest_idx, pulse_idx), soc) in enumerate(
                zip(rest_pulse_indices, SOC_NOMINAL)):

            rest_seg  = segments[rest_idx]
            pulse_seg = segments[pulse_idx]
            I_pulse   = abs(pulse_seg['mean_current'])

            R1, C1, R2, C2, t, fit_v, actual_v, ok = fit_rest_segment(
                time_s, voltage, rest_seg, I_pulse
            )

            status = "OK" if ok else "FAIL"
            print(f"  SOC={soc:.1f}: R1={R1*1e3:.3f}mΩ  C1={C1:.1f}F  "
                  f"R2={R2*1e3:.3f}mΩ  C2={C2:.1f}F  [{status}]")

            all_rows.append({
                'SOH': soh, 'SOC': soc,
                'R1_ohm': R1, 'C1_F': C1,
                'R2_ohm': R2, 'C2_F': C2,
            })

            ax = axes[j]
            ax.plot(t, actual_v * 1e3, 'b.', markersize=2, label='Data')
            if ok:
                ax.plot(t, fit_v * 1e3, 'r-', linewidth=1.5, label='2RC Fit')
            ax.set_title(f"SOC = {soc:.0%}", fontsize=10)
            ax.set_xlabel("Time [s]", fontsize=8)
            ax.set_ylabel("Voltage [mV]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc='lower right')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(PLOT_DIR, f"fit_quality_SOH{soh}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved: {plot_path}")

    # ── Save 2D CSV ──
    result_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "ecm_r1c1r2c2_2d.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved 2D parameter table: {csv_path}")
    print(result_df.to_string(index=False))

    # ================================================================
    # Parameter variation plots (POP UP)
    # ================================================================
    plot_parameter_grid(result_df, 'R1_ohm', 'R1', 'mΩ', 1e3)
    plot_parameter_grid(result_df, 'R2_ohm', 'R2', 'mΩ', 1e3)
    plot_parameter_grid(result_df, 'C1_F',   'C1', 'F',  1)
    plot_parameter_grid(result_df, 'C2_F',   'C2', 'F',  1)
    plot_avg_vs_soh(result_df)

    plt.show()


if __name__ == "__main__":
    main()