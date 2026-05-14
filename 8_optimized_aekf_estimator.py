"""
aekf_estimator.py
==================
Task 5 - Stage 3: Adaptive Extended Kalman Filter (AEKF) for SOC estimation.

State vector:    x = [SOC, V1, V2]^T
Measurement:     y = V_terminal
Plant model:     2RC ECM (same equations as ecm_plant_model.py)

Adaptation:      Sage-Husa innovation-based tuning of R and Q.

Assumes SOH is KNOWN (selects the right parameter table from Task 4).
SOH switching / detection is left for Stage 4.

Demonstration:   Initial SOC guess deliberately set to 0.85 (true is 1.0)
                 to show convergence of the filter from a wrong initial
                 condition.

For each drive cycle (UDDS, HWFET) and each SOH state (100/95/90/85/80%),
this script:
  1) Loads voltage + current + true SOC from drive_cycle_data/
  2) Runs the AEKF on the current/voltage trace
  3) Reports convergence time and steady-state SOC RMSE
  4) Plots estimated SOC vs true SOC and the SOC error
  5) Writes CSV results to aekf_excel_output/ (one file per SOH and drive cycle)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib
ecm_module = importlib.import_module("6_ecm_plant_model")
ECMParameters = ecm_module.ECMParameters

# ============================================================
# Configuration
# ============================================================
R0_FILE       = "2rc_parameters/ecm_r0_2d.csv"
RC_FILE       = "2rc_parameters/ecm_r1c1r2c2_2d.csv"
OCV_FOLDER    = "ocv_data"
DRIVE_FOLDER  = "drive_cycle_data"
CSV_OUT_DIR   = "aekf_excel_output"

NOM_CAP   = 0.6806
SOH_LIST  = [100, 95, 90, 85, 80]

SOC_INIT_GUESS = 0.85   # deliberately wrong (true = 1.0) to test convergence
P_INIT = np.diag([0.15**2, 0.01**2, 0.01**2])   # 3×3 — augmented to 4×4 in __init__

# Noise parameters tuned by PSO (only tuned terms shown below)
# OLD (baseline values for PSO-tuned parameters):
# R_INIT = (10e-3) ** 2
# R_MIN = (2e-3) ** 2
# Q_SOC_INIT = (5e-6) ** 2
# Q_SOC_MAX = (1e-3) ** 2
#
# NEW (active) PSO-tuned values from aekf_pso_best.json:
R_INIT = 3.04895025775358e-05
R_MIN = 6.276052863910388e-07
Q_SOC_INIT = 8.667023582093448e-15
Q_SOC_MAX = 2.7671671065134328e-08



Q_INIT = np.diag([
    Q_SOC_INIT,
    (1e-3) ** 2,
    (1e-3) ** 2,
])                           # 3×3 — augmented to 4×4 in __init__
ADAPT_WINDOW = 25
ADAPT_ENABLED = True
R_MAX = (100e-3) ** 2
Q_SOC_MIN = (1e-8) ** 2
CONVERGE_TOL = 0.02



# Voltage-bias state (4th state) — absorbs static ECM/OCV model offset
# P_BIAS_INIT: ±50 mV initial uncertainty on the bias
# Q_VBIAS:     bias random-walk per step — very small so bias changes slowly
P_BIAS_INIT = (50e-3) ** 2
Q_VBIAS     = (0.1e-3) ** 2


class AEKF:
    """
    Adaptive Extended Kalman Filter for SOC estimation on a 2RC ECM.

    State vector: x = [SOC, V1, V2, Vbias]
      Vbias is an augmented state that absorbs the static ECM/OCV model offset
      (systematic voltage prediction error).  Modelling it as a Kalman state lets
      the filter separate "innovation from wrong SOC" from "innovation from model
      error", which a fixed EMA cannot do.
    """

    def __init__(self, params, soh, capacity,
                 soc_init=SOC_INIT_GUESS,
                 P_init=P_INIT, R_init=R_INIT, Q_init=Q_INIT,
                 q_vbias=Q_VBIAS,
                 adapt_window=ADAPT_WINDOW, adapt=ADAPT_ENABLED):
        self.params = params
        self.soh = soh
        self.capacity = capacity

        # 4-state: [SOC, V1, V2, Vbias]; Vbias starts at 0 (unknown)
        self.x = np.array([soc_init, 0.0, 0.0, 0.0])

        # Augment 3×3 P_init to 4×4 with large initial bias uncertainty
        P4 = np.zeros((4, 4))
        P4[:3, :3] = P_init.copy()
        P4[3, 3] = P_BIAS_INIT
        self.P = P4

        self.R = R_init

        # Augment 3×3 Q_init to 4×4 with small bias random-walk noise
        Q4 = np.zeros((4, 4))
        Q4[:3, :3] = Q_init.copy()
        Q4[3, 3] = q_vbias
        self.Q = Q4

        self.adapt = adapt
        self.adapt_window = adapt_window
        self.innovation_buffer = []

        self.history = {
            "soc": [], "v1": [], "v2": [], "vbias": [],
            "v_pred": [], "innovation": [],
            "R": [], "Q_soc": [], "K_soc": [],
        }

    def _ocv_slope(self, soc, eps=1e-3):
        soc_p = min(1.0, soc + eps)
        soc_m = max(0.0, soc - eps)
        ocv_p = self.params.get_ocv(soc_p, self.soh)
        ocv_m = self.params.get_ocv(soc_m, self.soh)
        return (ocv_p - ocv_m) / (soc_p - soc_m)

    def step(self, current_prev, current_now, voltage_meas, dt):
        """
        Run one Predict + Update cycle.
          current_prev : current at t_{k-1} [A], positive = discharge.
                         Used for RC dynamics and coulomb counting (ZOH over [t_{k-1}, t_k]).
          current_now  : current at t_k [A].
                         Used for the instantaneous R0 drop in the measurement equation.
          voltage_meas : measured cell voltage at t_k [V]
          dt           : time step [s]
        """
        soc, v1, v2, vbias = self.x

        # RC time-constant params at previous SOC (govern dynamics over the interval)
        _, R1, C1, R2, C2 = self.params.get_params(soc, self.soh)
        tau1 = R1 * C1
        tau2 = R2 * C2
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)

        # ---------- PREDICT (use current_prev: input held during [t_{k-1}, t_k]) ----------
        soc_p   = soc   - current_prev * dt / (self.capacity * 3600.0)
        v1_p    = a1 * v1   + R1 * (1 - a1) * current_prev
        v2_p    = a2 * v2   + R2 * (1 - a2) * current_prev
        vbias_p = vbias                    # bias is constant between steps (+ process noise)
        x_pred  = np.array([soc_p, v1_p, v2_p, vbias_p])

        F = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, a1,  0.0, 0.0],
            [0.0, 0.0, a2,  0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        P_pred = F @ self.P @ F.T + self.Q   # 4×4

        # ---------- UPDATE (use current_now for R0 — instantaneous at t_k) ----------
        R0, _, _, _, _ = self.params.get_params(soc_p, self.soh)
        ocv = self.params.get_ocv(soc_p, self.soh)
        v_pred_raw  = ocv - R0 * current_now - v1_p - v2_p
        v_pred_full = v_pred_raw + vbias_p

        dOCV_dSOC = self._ocv_slope(soc_p)
        # H = [dOCV/dSOC, -1, -1, +1] — Vbias enters the measurement additively
        H = np.array([[dOCV_dSOC, -1.0, -1.0, 1.0]])

        innovation = voltage_meas - v_pred_full

        S = float((H @ P_pred @ H.T)[0, 0]) + self.R

        K = (P_pred @ H.T / S).flatten()   # shape (4,)

        x_new = x_pred + K * innovation

        I4 = np.eye(4)
        P_new = (I4 - np.outer(K, H.flatten())) @ P_pred

        x_new[0] = np.clip(x_new[0], 0.0, 1.0)

        # ---------- ADAPTIVE TUNING (Sage-Husa) ----------
        if self.adapt:
            self.innovation_buffer.append(innovation)
            if len(self.innovation_buffer) > self.adapt_window:
                self.innovation_buffer.pop(0)

            if len(self.innovation_buffer) == self.adapt_window:
                # Mean square (not variance) — correct when innovations may be biased
                C_k = float(np.mean(np.square(self.innovation_buffer)))
                R_new = C_k - float((H @ P_pred @ H.T)[0, 0])
                self.R = float(np.clip(R_new, R_MIN, R_MAX))
                q_soc = float(K[0] ** 2 * C_k)
                q_soc = float(np.clip(q_soc, Q_SOC_MIN, Q_SOC_MAX))
                self.Q[0, 0] = q_soc

        self.x = x_new
        self.P = P_new

        self.history["soc"].append(self.x[0])
        self.history["v1"].append(self.x[1])
        self.history["v2"].append(self.x[2])
        self.history["vbias"].append(self.x[3])
        # v_pred for reporting: use predicted raw + updated bias (best current estimate)
        self.history["v_pred"].append(v_pred_raw + self.x[3])
        self.history["innovation"].append(innovation)
        self.history["R"].append(self.R)
        self.history["Q_soc"].append(self.Q[0, 0])
        self.history["K_soc"].append(K[0])


def run_aekf_on_csv(cycle_name, soh, params):
    fname = os.path.join(DRIVE_FOLDER, f"{cycle_name}_SOH{soh}.csv")
    df = pd.read_csv(fname)

    t          = df["Time [s]"].values
    current    = df["Current [A]"].values
    voltage    = df["Voltage [V]"].values
    soc_true   = df["True SOC"].values

    capacity = NOM_CAP * soh / 100.0

    aekf = AEKF(params, soh=soh, capacity=capacity,
                soc_init=SOC_INIT_GUESS)

    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        aekf.step(current[k - 1], current[k], voltage[k], dt)

    # Posterior SOC (corrected by voltage): the main AEKF output
    soc_est = np.array([SOC_INIT_GUESS] + aekf.history["soc"])

    # A priori SOC (estimated, predict-only, before voltage correction):
    # soc_prior[k] = soc_posterior[k-1] - I[k]*dt/(Q*3600)
    # Reconstructed from posterior without modifying the AEKF class.
    soc_prior = np.zeros(len(t))
    soc_prior[0] = SOC_INIT_GUESS
    for k in range(1, len(t)):
        dt_k = t[k] - t[k - 1]
        soc_prior[k] = soc_est[k - 1] - current[k - 1] * dt_k / (capacity * 3600.0)

    # Open-loop "measurement" SOC: pure Coulomb counting from same initial guess,
    # no voltage correction ever applied (baseline for comparison).
    soc_meas = np.zeros(len(t))
    soc_meas[0] = SOC_INIT_GUESS
    for k in range(1, len(t)):
        dt_k = t[k] - t[k - 1]
        soc_meas[k] = soc_meas[k - 1] - current[k - 1] * dt_k / (capacity * 3600.0)

    v_pred  = np.array([voltage[0]] + aekf.history["v_pred"])
    innov   = np.array([0.0] + aekf.history["innovation"])

    soc_err = soc_est - soc_true

    converge_idx = None
    streak = 0
    for k in range(len(soc_err)):
        if abs(soc_err[k]) < CONVERGE_TOL:
            streak += 1
            if streak >= 10:
                converge_idx = k - 9
                break
        else:
            streak = 0
    converge_time = t[converge_idx] if converge_idx is not None else None

    if converge_idx is not None:
        ss_slice = slice(converge_idx, None)
    else:
        ss_slice = slice(int(0.3 * len(t)), None)

    ss_rmse = float(np.sqrt(np.mean(soc_err[ss_slice] ** 2)) * 100)
    v_rmse  = float(np.sqrt(np.mean((v_pred - voltage) ** 2)) * 1000)

    return {
        "t": t, "current": current, "voltage": voltage,
        "soc_true": soc_true,
        "soc_prior": soc_prior,
        "soc_meas": soc_meas,
        "soc_est": soc_est,
        "soc_err": soc_err,
        "v_pred": v_pred, "innovation": innov,
        "converge_time": converge_time,
        "ss_rmse_pct": ss_rmse,
        "v_rmse_mV": v_rmse,
        "aekf": aekf,
    }


def export_csv_results(results, cycles, soh_list, out_dir=CSV_OUT_DIR):
    """Write AEKF_SOH{soh}_{cycle}.csv with the 4 SOC traces needed for analysis."""
    os.makedirs(out_dir, exist_ok=True)
    for soh in soh_list:
        for cycle_name in cycles:
            r = results[(cycle_name, soh)]
            df = pd.DataFrame({
                "Time [s]": r["t"],
                "Current [A]": r["current"],
                "Voltage [V]": r["voltage"],
                "True SOC": r["soc_true"],
                "Estimated SOC (AEKF prior)": r["soc_prior"],
                "Measurement SOC (open-loop)": r["soc_meas"],
                "Corrected SOC (AEKF posterior)": r["soc_est"],
            })
            path = os.path.join(out_dir, f"AEKF_SOH{soh}_{cycle_name}.csv")
            df.to_csv(path, index=False)
            print(f"Saved {path}")


def main():
    print("Loading ECM parameter tables...")
    params = ECMParameters(R0_FILE, RC_FILE, OCV_FOLDER, SOH_LIST)

    cycles = ["UDDS", "HWFET"]
    results = {}

    print("\nRunning AEKF (initial SOC guess = "
          f"{SOC_INIT_GUESS:.2f}, true = 1.00)")
    print("-" * 70)
    print(f"{'Cycle':<8}{'SOH':<8}{'Converge [s]':<15}"
          f"{'SOC RMSE [%]':<15}{'V RMSE [mV]':<15}")
    print("-" * 70)

    for cycle_name in cycles:
        for soh in SOH_LIST:
            r = run_aekf_on_csv(cycle_name, soh, params)
            results[(cycle_name, soh)] = r

            ct_str = f"{r['converge_time']:.1f}" if r['converge_time'] is not None else "  --  "
            print(f"{cycle_name:<8}{soh:<8}{ct_str:<15}"
                  f"{r['ss_rmse_pct']:<15.3f}{r['v_rmse_mV']:<15.1f}")

    export_csv_results(results, cycles, SOH_LIST)

    soh_colors = {100: '#1f77b4', 95: '#ff7f0e', 90: '#2ca02c',
                  85: '#d62728', 80: '#9467bd'}

    for cycle_name in cycles:
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        fig.suptitle(f"{cycle_name}: AEKF SOC Estimate vs True SOC",
                     fontsize=13, fontweight='bold')

        for ax, soh in zip(axes, SOH_LIST):
            r = results[(cycle_name, soh)]
            ax.plot(r["t"], r["soc_true"], 'k-', linewidth=1.4,
                    label='True SOC (PyBaMM)')
            ax.plot(r["t"], r["soc_est"], color=soh_colors[soh],
                    linewidth=1.2, linestyle='--',
                    label='AEKF estimate')
            ax.axhline(SOC_INIT_GUESS, color='grey', linewidth=0.5,
                       linestyle=':', label='Initial guess')
            ct = r['converge_time']
            ct_lbl = f"converged @ {ct:.0f}s" if ct is not None else "no converge"
            ax.set_ylabel(f"SOH {soh}%\nSOC")
            ax.set_title(f"SOC RMSE (steady) = {r['ss_rmse_pct']:.2f}%, {ct_lbl}",
                         fontsize=9)
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plt.figure(figsize=(11, 5))
        for soh in SOH_LIST:
            r = results[(cycle_name, soh)]
            plt.plot(r["t"], r["soc_err"] * 100, color=soh_colors[soh],
                     linewidth=1.2, label=f'SOH {soh}%')
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axhline(CONVERGE_TOL * 100, color='grey', linestyle=':', linewidth=0.7)
        plt.axhline(-CONVERGE_TOL * 100, color='grey', linestyle=':', linewidth=0.7)
        plt.title(f"{cycle_name}: SOC Estimation Error (Estimate - Truth)")
        plt.xlabel("Time [s]"); plt.ylabel("SOC error [%]")
        plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.figure(figsize=(11, 5))
        for soh in SOH_LIST:
            r = results[(cycle_name, soh)]
            plt.plot(r["t"], r["innovation"] * 1000, color=soh_colors[soh],
                     linewidth=0.8, label=f'SOH {soh}%')
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title(f"{cycle_name}: AEKF Innovation (Voltage Residual)")
        plt.xlabel("Time [s]"); plt.ylabel("Innovation [mV]")
        plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.show()
    print("\nAEKF validation complete!")


if __name__ == "__main__":
    main()
