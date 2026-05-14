"""
ecm_plant_model.py
==================
Task 5 - Stage 2: 2RC Equivalent Circuit Model (ECM) plant.

Loads the parameter tables from Task 4:
  - R0(SOC, SOH)             from 2rc_parameters/ecm_r0_2d.csv
  - R1, C1, R2, C2(SOC, SOH) from 2rc_parameters/ecm_r1c1r2c2_2d.csv
  - OCV(SOC) per SOH         from ocv_data/OCV_SOH*.csv

Implements the discrete 2RC state equations:
  V1[k+1] = V1[k] * exp(-dt/tau1) + R1 * I[k] * (1 - exp(-dt/tau1))
  V2[k+1] = V2[k] * exp(-dt/tau2) + R2 * I[k] * (1 - exp(-dt/tau2))
  SOC[k+1] = SOC[k] - I[k] * dt / (Q * 3600)
  V_term[k] = OCV(SOC[k]) + R0 * I[k] + V1[k] + V2[k]

Validates ECM voltage against PyBaMM drive-cycle voltage from Stage 1.

Sign convention: positive current = discharge (consistent with Stage 1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
import os

# ============================================================
# Configuration
# ============================================================
R0_FILE       = "2rc_parameters/ecm_r0_2d.csv"
RC_FILE       = "2rc_parameters/ecm_r1c1r2c2_2d.csv"
OCV_FOLDER    = "ocv_data"
DRIVE_FOLDER  = "drive_cycle_data"

NOM_CAP = 0.6806   # Ah, nominal capacity
SOH_LIST = [100, 95, 90, 85, 80]


# ============================================================
# Parameter table loader
# ============================================================
class ECMParameters:
    """
    Loads and interpolates the 2RC ECM parameters from CSV files.
    Provides callable lookups for R0, R1, C1, R2, C2 as f(SOC, SOH)
    and OCV as f(SOC, SOH).
    """

    def __init__(self, r0_file, rc_file, ocv_folder, soh_list):
        # ---- Load R0 table ----
        df_r0 = pd.read_csv(r0_file)
        # Handle either "SOH (%)" or "SOH" column name
        soh_col = "SOH (%)" if "SOH (%)" in df_r0.columns else "SOH"
        self.r0_table = df_r0.rename(columns={soh_col: "SOH",
                                               "R0 (Ohm)": "R0"})

        # ---- Load R1/C1/R2/C2 table ----
        df_rc = pd.read_csv(rc_file)
        soh_col = "SOH (%)" if "SOH (%)" in df_rc.columns else "SOH"
        self.rc_table = df_rc.rename(columns={soh_col: "SOH"})

        # ---- Build 2D interpolators on a regular grid ----
        # We'll grid each parameter on (SOH, SOC) and use RegularGridInterpolator
        self.soh_grid = np.array(sorted(soh_list, reverse=True), dtype=float)

        # Get a common SOC grid (use the RC table's SOC values from SOH 100)
        soc_vals_rc = sorted(self.rc_table[
            self.rc_table["SOH"] == 100
        ]["SOC"].unique())
        self.soc_grid = np.array(soc_vals_rc, dtype=float)

        # Build grids for each parameter: shape (n_soh, n_soc)
        self.R0_grid = self._build_grid(self.r0_table, "R0")
        self.R1_grid = self._build_grid(self.rc_table, "R1_ohm")
        self.C1_grid = self._build_grid(self.rc_table, "C1_F")
        self.R2_grid = self._build_grid(self.rc_table, "R2_ohm")
        self.C2_grid = self._build_grid(self.rc_table, "C2_F")

        # Create interpolators
        self.R0_interp = RegularGridInterpolator(
            (self.soh_grid[::-1], self.soc_grid),
            self.R0_grid[::-1, :],
            bounds_error=False, fill_value=None
        )
        self.R1_interp = RegularGridInterpolator(
            (self.soh_grid[::-1], self.soc_grid),
            self.R1_grid[::-1, :],
            bounds_error=False, fill_value=None
        )
        self.C1_interp = RegularGridInterpolator(
            (self.soh_grid[::-1], self.soc_grid),
            self.C1_grid[::-1, :],
            bounds_error=False, fill_value=None
        )
        self.R2_interp = RegularGridInterpolator(
            (self.soh_grid[::-1], self.soc_grid),
            self.R2_grid[::-1, :],
            bounds_error=False, fill_value=None
        )
        self.C2_interp = RegularGridInterpolator(
            (self.soh_grid[::-1], self.soc_grid),
            self.C2_grid[::-1, :],
            bounds_error=False, fill_value=None
        )

        # ---- Load OCV curves per SOH ----
        self.ocv_interps = {}
        for soh in soh_list:
            fname = os.path.join(ocv_folder, f"OCV_SOH{soh}.csv")
            df = pd.read_csv(fname)
            soc = df["SOC"].values
            ocv = df["OCV [V]"].values
            # Sort by SOC ascending
            order = np.argsort(soc)
            self.ocv_interps[soh] = interp1d(
                soc[order], ocv[order], kind='linear',
                bounds_error=False, fill_value=(ocv[order][0], ocv[order][-1])
            )

        print(f"Loaded ECM parameters:")
        print(f"  SOH grid: {self.soh_grid}")
        print(f"  SOC grid: {self.soc_grid}")
        print(f"  R0 range: {self.R0_grid.min():.4f} - {self.R0_grid.max():.4f} Ohm")
        print(f"  R1 range: {self.R1_grid.min():.4f} - {self.R1_grid.max():.4f} Ohm")
        print(f"  R2 range: {self.R2_grid.min():.4f} - {self.R2_grid.max():.4f} Ohm")

    def _build_grid(self, df, col):
        """Build (n_soh, n_soc) grid for a parameter."""
        grid = np.zeros((len(self.soh_grid), len(self.soc_grid)))
        for i, soh in enumerate(self.soh_grid):
            sub = df[df["SOH"] == soh].sort_values("SOC")
            soc_vals = sub["SOC"].values
            param_vals = sub[col].values
            # Interpolate onto common SOC grid
            grid[i, :] = np.interp(self.soc_grid, soc_vals, param_vals)
        return grid

    def get_params(self, soc, soh):
        """Return (R0, R1, C1, R2, C2) at given (SOC, SOH)."""
        # Clip SOC to grid range
        soc_c = np.clip(soc, self.soc_grid.min(), self.soc_grid.max())
        soh_c = np.clip(soh, self.soh_grid.min(), self.soh_grid.max())
        pt = np.array([[soh_c, soc_c]])
        R0 = float(self.R0_interp(pt)[0])
        R1 = float(self.R1_interp(pt)[0])
        C1 = float(self.C1_interp(pt)[0])
        R2 = float(self.R2_interp(pt)[0])
        C2 = float(self.C2_interp(pt)[0])
        return R0, R1, C1, R2, C2

    def get_ocv(self, soc, soh):
        """Return OCV at given SOC for the closest SOH in the table."""
        # Find nearest SOH from the loaded curves
        soh_keys = list(self.ocv_interps.keys())
        nearest = min(soh_keys, key=lambda x: abs(x - soh))
        return float(self.ocv_interps[nearest](soc))


# ============================================================
# ECM simulation
# ============================================================
def simulate_ecm(time, current, soc_init, soh, params, capacity):
    """
    Simulate the 2RC ECM voltage response to a current profile.

    Inputs:
      time     : (N,) time array [s]
      current  : (N,) current array [A], positive = discharge
      soc_init : initial SOC [-]
      soh      : SOH value [%] (assumed constant during the simulation)
      params   : ECMParameters object
      capacity : actual cell capacity [Ah] at this SOH

    Returns:
      v_term : (N,) ECM terminal voltage [V]
      soc    : (N,) SOC trace [-]
      v1     : (N,) RC1 voltage [V]
      v2     : (N,) RC2 voltage [V]
    """
    N = len(time)
    v_term = np.zeros(N)
    soc    = np.zeros(N)
    v1     = np.zeros(N)
    v2     = np.zeros(N)

    soc[0] = soc_init
    # Initial RC voltages: assume cell starts at rest (V1 = V2 = 0)
    v1[0] = 0.0
    v2[0] = 0.0

    # Output voltage at k=0
    R0, R1, C1, R2, C2 = params.get_params(soc[0], soh)
    ocv0 = params.get_ocv(soc[0], soh)
    v_term[0] = ocv0 - R0 * current[0] - v1[0] - v2[0]

    for k in range(N - 1):
        dt = time[k + 1] - time[k]

        # Lookup parameters at current SOC
        R0, R1, C1, R2, C2 = params.get_params(soc[k], soh)
        tau1 = R1 * C1
        tau2 = R2 * C2

        # Discrete RC voltage updates (zero-order hold on current)
        a1 = np.exp(-dt / tau1)
        a2 = np.exp(-dt / tau2)
        v1[k + 1] = a1 * v1[k] + R1 * (1 - a1) * current[k]
        v2[k + 1] = a2 * v2[k] + R2 * (1 - a2) * current[k]

        # Coulomb counting (positive I = discharge -> SOC drops)
        soc[k + 1] = soc[k] - current[k] * dt / (capacity * 3600.0)

        # Terminal voltage (sign: discharge current drops voltage)
        R0_next, _, _, _, _ = params.get_params(soc[k + 1], soh)
        ocv_next = params.get_ocv(soc[k + 1], soh)
        v_term[k + 1] = ocv_next - R0_next * current[k + 1] - v1[k + 1] - v2[k + 1]

    return v_term, soc, v1, v2


# ============================================================
# Main: validate ECM against PyBaMM drive cycle data
# ============================================================
def main():
    # Load parameters
    print("Loading ECM parameter tables...")
    params = ECMParameters(R0_FILE, RC_FILE, OCV_FOLDER, SOH_LIST)

    # Run ECM on each drive cycle / SOH combination and compare to PyBaMM
    cycles = ["UDDS", "HWFET"]
    results = {}

    print("\nRunning ECM simulations...")
    for cycle_name in cycles:
        for soh in SOH_LIST:
            # Load PyBaMM drive cycle CSV
            fname = os.path.join(DRIVE_FOLDER, f"{cycle_name}_SOH{soh}.csv")
            df = pd.read_csv(fname)
            t = df["Time [s]"].values
            i = df["Current [A]"].values
            v_pybamm = df["Voltage [V]"].values
            soc_pybamm = df["True SOC"].values

            # Actual capacity at this SOH
            actual_cap = NOM_CAP * soh / 100.0

            # Run ECM
            v_ecm, soc_ecm, v1, v2 = simulate_ecm(
                t, i, soc_init=1.0, soh=soh,
                params=params, capacity=actual_cap
            )

            # Compute RMSE
            rmse_v = np.sqrt(np.mean((v_ecm - v_pybamm) ** 2)) * 1000  # mV
            rmse_soc = np.sqrt(np.mean((soc_ecm - soc_pybamm) ** 2)) * 100  # %

            results[(cycle_name, soh)] = {
                "t": t, "i": i,
                "v_pybamm": v_pybamm, "v_ecm": v_ecm,
                "soc_pybamm": soc_pybamm, "soc_ecm": soc_ecm,
                "rmse_v_mV": rmse_v, "rmse_soc_pct": rmse_soc,
            }

            print(f"  {cycle_name} SOH {soh}%: "
                  f"V RMSE = {rmse_v:6.1f} mV, "
                  f"SOC RMSE = {rmse_soc:5.2f} %")

    # ============================================================
    # Plot comparisons
    # ============================================================
    soh_colors = {100: '#1f77b4', 95: '#ff7f0e', 90: '#2ca02c',
                  85: '#d62728', 80: '#9467bd'}

    for cycle_name in cycles:
        # Voltage comparison: ECM vs PyBaMM
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
        fig.suptitle(f"{cycle_name}: ECM Voltage vs PyBaMM (Truth)",
                     fontsize=13, fontweight='bold')

        for ax, soh in zip(axes, SOH_LIST):
            r = results[(cycle_name, soh)]
            ax.plot(r["t"], r["v_pybamm"], 'k-', linewidth=1.0,
                    label='PyBaMM (truth)')
            ax.plot(r["t"], r["v_ecm"], color=soh_colors[soh],
                    linewidth=1.0, linestyle='--',
                    label=f'ECM (RMSE={r["rmse_v_mV"]:.1f} mV)')
            ax.set_ylabel(f"SOH {soh}%\nV [V]")
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time [s]")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # SOC comparison
        plt.figure(figsize=(11, 5))
        for soh in SOH_LIST:
            r = results[(cycle_name, soh)]
            plt.plot(r["t"], r["soc_pybamm"], color=soh_colors[soh],
                     linewidth=1.5, label=f'PyBaMM SOH {soh}%')
            plt.plot(r["t"], r["soc_ecm"], color=soh_colors[soh],
                     linewidth=1.0, linestyle='--')
        plt.title(f"{cycle_name}: SOC - ECM (dashed) vs PyBaMM (solid)")
        plt.xlabel("Time [s]"); plt.ylabel("SOC [-]")
        plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.show()

    print("\nECM plant model validation complete!")


if __name__ == "__main__":
    main()