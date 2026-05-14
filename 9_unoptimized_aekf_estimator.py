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
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib
# Reuse the parameter loader from Stage 2
ecm_module = importlib.import_module("6_ecm_plant_model")
ECMParameters = ecm_module.ECMParameters

# ============================================================
# Configuration
# ============================================================
R0_FILE       = "2rc_parameters/ecm_r0_2d.csv"
RC_FILE       = "2rc_parameters/ecm_r1c1r2c2_2d.csv"
OCV_FOLDER    = "ocv_data"
DRIVE_FOLDER  = "drive_cycle_data"

NOM_CAP   = 0.6806
SOH_LIST  = [100, 95, 90, 85, 80]

# Initial filter conditions (deliberately wrong to demo convergence)
SOC_INIT_GUESS = 0.85   # true is 1.0
P_INIT = np.diag([0.15**2, 0.01**2, 0.01**2])  # high SOC uncertainty, low RC uncertainty

# Initial noise covariances (will adapt)
# R is tuned to reflect the ECM bias floor (~80 mV stdev)
# Q on SOC is tiny because Coulomb counting is essentially exact
R_INIT = (0.030) ** 2     # measurement noise: ~30 mV stdev
Q_INIT = np.diag([
    (5e-6) ** 2,          # SOC process noise — small but allows correction
    (1e-3) ** 2,          # V1 process noise
    (1e-3) ** 2,          # V2 process noise
])
# Sage-Husa adaptation parameters
ADAPT_WINDOW = 50          # samples for innovation window
ADAPT_ENABLED = True       # toggle adaptive tuning
# R floor at 50 mV stdev so adaptation cannot make filter over-trust voltage
R_MIN, R_MAX = (0.020) ** 2, (0.300) ** 2
Q_SOC_MIN, Q_SOC_MAX = (1e-7) ** 2, (1e-5) ** 2
# Convergence criterion
CONVERGE_TOL = 0.02        # |error| < 2 %


# ============================================================
# AEKF class
# ============================================================
class AEKF:
    """
    Adaptive Extended Kalman Filter for SOC estimation on a 2RC ECM.
    State: x = [SOC, V1, V2]^T
    """

    def __init__(self, params, soh, capacity,
                 soc_init=SOC_INIT_GUESS,
                 P_init=P_INIT, R_init=R_INIT, Q_init=Q_INIT,
                 adapt_window=ADAPT_WINDOW, adapt=ADAPT_ENABLED):
        self.params = params
        self.soh = soh
        self.capacity = capacity   # Ah (already adjusted for SOH)

        # State and covariance
        self.x = np.array([soc_init, 0.0, 0.0])
        self.P = P_init.copy()

        # Noise covariances
        self.R = R_init       # scalar
        self.Q = Q_init.copy()

        # Adaptive tuning
        self.adapt = adapt
        self.adapt_window = adapt_window
        self.innovation_buffer = []

        # Voltage bias compensation (learned from early samples)
        self.v_bias = 0.0
        self.bias_samples = []
        self.bias_locked = False
        self.bias_n_samples = 30  # learn from first 30 samples (~30s)

        # Logging
        self.history = {
            "soc": [], "v1": [], "v2": [],
            "v_pred": [], "innovation": [],
            "R": [], "Q_soc": [],
            "K_soc": [],
        }

    # ---------- helper: numerical OCV slope at given SOC ----------
    def _ocv_slope(self, soc, eps=1e-3):
        soc_p = min(1.0, soc + eps)
        soc_m = max(0.0, soc - eps)
        ocv_p = self.params.get_ocv(soc_p, self.soh)
        ocv_m = self.params.get_ocv(soc_m, self.soh)
        return (ocv_p - ocv_m) / (soc_p - soc_m)

    # ---------- one filter step ----------
    def step(self, current, voltage_meas, dt):
        """
        Run one Predict + Update cycle.
          current      : measured cell current [A] (positive = discharge)
          voltage_meas : measured cell voltage [V]
          dt           : time step [s]
        """
        soc, v1, v2 = self.x

        # Lookup ECM parameters at current SOC and SOH
        R0, R1, C1, R2, C2 = self.params.get_params(soc, self.soh)
        tau1 = R1 * C1
        tau2 = R2 * C2
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)

        # ---------- PREDICT ----------
        # State propagation (zero-order hold on current)
        soc_p = soc - current * dt / (self.capacity * 3600.0)
        v1_p  = a1 * v1 + R1 * (1 - a1) * current
        v2_p  = a2 * v2 + R2 * (1 - a2) * current
        x_pred = np.array([soc_p, v1_p, v2_p])

        # State Jacobian F = d f / d x
        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, a1,  0.0],
            [0.0, 0.0, a2 ],
        ])

        # Covariance propagation
        P_pred = F @ self.P @ F.T + self.Q

        # ---------- UPDATE ----------
        # Predicted measurement: V_term = OCV(SOC) - R0*I - V1 - V2 + bias
        ocv = self.params.get_ocv(soc_p, self.soh)
        v_pred = ocv - R0 * current - v1_p - v2_p + self.v_bias

        # Measurement Jacobian H = d h / d x
        dOCV_dSOC = self._ocv_slope(soc_p)
        H = np.array([[dOCV_dSOC, -1.0, -1.0]])  # shape (1,3)

        # Innovation
        innovation = voltage_meas - v_pred         # scalar

        # Innovation covariance
        S = float((H @ P_pred @ H.T)[0, 0]) + self.R       # scalar

        # Kalman gain (3x1)
        K = (P_pred @ H.T / S).flatten()

        # State update
        x_new = x_pred + K * innovation

        # Covariance update (Joseph form would be more numerically stable
        # but the standard form is fine here)
        I3 = np.eye(3)
        P_new = (I3 - np.outer(K, H.flatten())) @ P_pred

        # Clamp SOC to [0, 1]
        x_new[0] = np.clip(x_new[0], 0.0, 1.0)

        # ---------- ADAPTIVE TUNING (Sage-Husa) ----------
        if self.adapt:
            self.innovation_buffer.append(innovation)
            if len(self.innovation_buffer) > self.adapt_window:
                self.innovation_buffer.pop(0)

            if len(self.innovation_buffer) == self.adapt_window:
                # Sample variance of recent innovations
                C_k = float(np.var(self.innovation_buffer))

                # Update R: C_k = H P_pred H^T + R  -> R = C_k - H P_pred H^T
                R_new = C_k - float((H @ P_pred @ H.T)[0, 0])
                self.R = float(np.clip(R_new, R_MIN, R_MAX))

                # Update Q on SOC channel only:
                # Q_new = K * C_k * K^T  (rank-1 update on SOC)
                q_soc = float(K[0] ** 2 * C_k)
                q_soc = float(np.clip(q_soc, Q_SOC_MIN, Q_SOC_MAX))
                self.Q[0, 0] = q_soc

        # Store state
        self.x = x_new
        # Learn voltage bias from first N samples then lock it
        if not self.bias_locked:
            # Use raw model prediction without current bias
            raw_pred = ocv - R0 * current - v1_p - v2_p
            self.bias_samples.append(voltage_meas - raw_pred)
            if len(self.bias_samples) >= self.bias_n_samples:
                self.v_bias = float(np.mean(self.bias_samples))
                self.bias_locked = True
        self.P = P_new

        # Log
        self.history["soc"].append(self.x[0])
        self.history["v1"].append(self.x[1])
        self.history["v2"].append(self.x[2])
        self.history["v_pred"].append(v_pred)
        self.history["innovation"].append(innovation)
        self.history["R"].append(self.R)
        self.history["Q_soc"].append(self.Q[0, 0])
        self.history["K_soc"].append(K[0])


# ============================================================
# Validation routine
# ============================================================
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
        aekf.step(current[k], voltage[k], dt)

    # Pad with the initial value to match length
    soc_est = np.array([SOC_INIT_GUESS] + aekf.history["soc"])
    v_pred  = np.array([voltage[0]] + aekf.history["v_pred"])
    innov   = np.array([0.0] + aekf.history["innovation"])

    soc_err = soc_est - soc_true

    # Convergence time: first time |error| stays below tol for 10 consecutive samples
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

    # Steady-state RMSE: after convergence (or last 70% if no convergence)
    if converge_idx is not None:
        ss_slice = slice(converge_idx, None)
    else:
        ss_slice = slice(int(0.3 * len(t)), None)

    ss_rmse = float(np.sqrt(np.mean(soc_err[ss_slice] ** 2)) * 100)  # percent
    v_rmse  = float(np.sqrt(np.mean((v_pred - voltage) ** 2)) * 1000)  # mV

    return {
        "t": t, "current": current, "voltage": voltage,
        "soc_true": soc_true, "soc_est": soc_est, "soc_err": soc_err,
        "v_pred": v_pred, "innovation": innov,
        "converge_time": converge_time,
        "ss_rmse_pct": ss_rmse,
        "v_rmse_mV": v_rmse,
        "aekf": aekf,
    }


# ============================================================
# Main
# ============================================================
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

    # ============================================================
    # Plots
    # ============================================================
    soh_colors = {100: '#1f77b4', 95: '#ff7f0e', 90: '#2ca02c',
                  85: '#d62728', 80: '#9467bd'}

    for cycle_name in cycles:
        # ---- SOC tracking ----
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        fig.suptitle(f"{cycle_name}: AEKF SOC Estimate vs True SOC",
                     fontsize=13, fontweight='bold')

        for ax, soh in zip(axes, SOH_LIST):
            r = results[(cycle_name, soh)]
            ax.plot(r["t"], r["soc_true"], 'k-', linewidth=1.4,
                    label='True SOC (PyBaMM)')
            ax.plot(r["t"], r["soc_est"], color=soh_colors[soh],
                    linewidth=1.2, linestyle='--',
                    label=f'AEKF estimate')
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

        # ---- SOC error ----
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

        # ---- Innovation (voltage residual) ----
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