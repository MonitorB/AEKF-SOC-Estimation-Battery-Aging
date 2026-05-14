"""
drive_cycle_simulation.py
==========================
Task 5 - Stage 1: Generate drive-cycle datasets for AEKF validation.

Pipeline:
  1) Run 100-cycle aging simulation (same as HPPC script)
  2) Load UDDS and HWFET speed profiles (mph vs time)
  3) Convert speed -> cell current using vehicle dynamics:
         P_wheel = v * (m*g*Cr + 0.5*rho*Cd*A*v^2 + m*a)
         P_bat   = P_wheel / eta_dt   (discharge)
                 = P_wheel * eta_rg   (regenerative braking)
     Scaled so peak |current| = PEAK_C * cell_capacity.
  4) For each SOH state (100/95/90/85/80%), run the drive cycle current
     through PyBaMM starting from fully charged aged cell.
  5) Save CSVs with:  Time [s], Current [A], Voltage [V], True SOC [-]

Outputs (drive_cycle_data/):
    UDDS_SOH100.csv, UDDS_SOH95.csv, ..., UDDS_SOH80.csv
    HWFET_SOH100.csv, HWFET_SOH95.csv, ..., HWFET_SOH80.csv
"""

import pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ============================================================
# Configuration
# ============================================================
UDDS_FILE  = "Urban_Dynamometer_Driving.txt"
HWFET_FILE = "Highway_Fuel_Economy_Tes.txt"
OUT_DIR    = "drive_cycle_data"

PEAK_C = 2.0       # peak C-rate for scaling the current profile
NOM_CAP = 0.6806   # Ah, Marquis2019 nominal capacity

# Vehicle parameters (small EV)
VEHICLE = {
    'mass'      : 1500.0,   # kg
    'Cr'        : 0.01,     # rolling resistance coefficient
    'Cd'        : 0.3,      # drag coefficient
    'A'         : 2.5,      # m^2 frontal area
    'rho'       : 1.225,    # kg/m^3 air density
    'g'         : 9.81,     # m/s^2
    'eta_dt'    : 0.85,     # drivetrain efficiency
    'eta_rg'    : 0.50,     # regenerative braking efficiency
}

MPH_TO_MPS = 0.44704

# ============================================================
# Helpers
# ============================================================
def load_drive_cycle(path):
    """Load EPA drive cycle txt file. Returns (time_s, speed_mph)."""
    df = pd.read_csv(path, sep=r'\s+', skiprows=2, header=None,
                     names=['time', 'speed'], engine='python')
    return df['time'].values.astype(float), df['speed'].values.astype(float)


def speed_to_current(t, v_mph, peak_C=PEAK_C, cell_cap=NOM_CAP):
    """
    Convert drive-cycle speed (mph) to cell current (A).
    Positive = discharge, negative = regenerative charging.
    Scales so peak |current| == peak_C * cell_capacity.
    """
    v = v_mph * MPH_TO_MPS
    a = np.gradient(v, t)  # acceleration m/s^2

    m   = VEHICLE['mass']
    Cr  = VEHICLE['Cr']
    Cd  = VEHICLE['Cd']
    A   = VEHICLE['A']
    rho = VEHICLE['rho']
    g   = VEHICLE['g']
    eta_dt = VEHICLE['eta_dt']
    eta_rg = VEHICLE['eta_rg']

    F_total = m * g * Cr + 0.5 * rho * Cd * A * v**2 + m * a
    P_wheel = v * F_total  # W

    # Efficiency-adjusted battery power
    P_bat = np.where(P_wheel >= 0, P_wheel / eta_dt, P_wheel * eta_rg)

    # Scale so peak |current| = peak_C * cell_cap
    peak_target = peak_C * cell_cap
    max_abs = np.max(np.abs(P_bat))
    if max_abs < 1e-9:
        return np.zeros_like(P_bat)
    scale = peak_target / max_abs
    return P_bat * scale


# ============================================================
# STEP 1: Aging simulation
# ============================================================
print("Loading model and running 100-cycle aging simulation...")
model = pybamm.lithium_ion.DFN(options={"SEI": "ec reaction limited"})
param = pybamm.ParameterValues("Marquis2019")

aging_experiment = pybamm.Experiment(
    [
        (
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
            "Discharge at 1C until 3.2V",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
        )
    ]
    +
    [
        (
            "Discharge at 1C until 3.2V",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
        )
    ] * 99
)

aging_sim = pybamm.Simulation(model, parameter_values=param, experiment=aging_experiment)
aging_sim.solve()
aging_solution = aging_sim.solution
print("Aging simulation complete.")

# Determine SOH cycle numbers
capacities = []
for i, cycle in enumerate(aging_solution.cycles):
    discharge = cycle.steps[2] if i == 0 else cycle.steps[0]
    capacities.append(abs(discharge["Discharge capacity [A.h]"].entries[-1]))

capacities = np.array(capacities)
initial_capacity = capacities[0]
soh = (capacities / initial_capacity) * 100

soh_cycles = {"100": 1}
for threshold in [95, 90, 85, 80]:
    below = np.where(soh <= threshold)[0]
    if len(below) > 0:
        soh_cycles[str(threshold)] = int(below[0]) + 1
        print(f"  SOH {threshold}% reached at cycle {soh_cycles[str(threshold)]}")

print(f"\nSOH cycle map: {soh_cycles}")

# Actual capacity at each SOH state for Coulomb counting
print("Actual capacities for Coulomb counting:")
for soh_label in soh_cycles:
    actual_cap = NOM_CAP * float(soh_label) / 100.0
    print(f"  SOH {soh_label}%: {actual_cap:.4f} Ah")

# ============================================================
# STEP 2: Load drive cycles and compute currents
# ============================================================
print("\nLoading drive cycles...")
t_udds, v_udds = load_drive_cycle(UDDS_FILE)
t_hwfet, v_hwfet = load_drive_cycle(HWFET_FILE)

I_udds = speed_to_current(t_udds, v_udds)
I_hwfet = speed_to_current(t_hwfet, v_hwfet)

# Verify: compute expected SOC drop from Coulomb counting on input current
for name, t_arr, i_arr in [("UDDS", t_udds, I_udds), ("HWFET", t_hwfet, I_hwfet)]:
    total_ah = np.trapezoid(i_arr, t_arr) / 3600
    soc_drop_pct = total_ah / NOM_CAP * 100
    print(f"  {name}: {len(t_arr)} points, {t_arr[-1]:.0f}s, "
          f"peak I = {np.max(np.abs(i_arr)):.3f}A, "
          f"mean I = {np.mean(i_arr):.3f}A, "
          f"net Ah = {total_ah:.4f}, expected SOC drop = {soc_drop_pct:.1f}%")

# Plot drive cycle speed and current profiles (verification figure)
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex='col')
axes[0, 0].plot(t_udds, v_udds, 'b-', linewidth=0.8)
axes[0, 0].set_ylabel("Speed [mph]")
axes[0, 0].set_title("UDDS Drive Cycle")
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(t_udds, I_udds, 'r-', linewidth=0.8)
axes[1, 0].set_ylabel("Cell Current [A]")
axes[1, 0].set_xlabel("Time [s]")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(0, color='k', linewidth=0.5)

axes[0, 1].plot(t_hwfet, v_hwfet, 'b-', linewidth=0.8)
axes[0, 1].set_ylabel("Speed [mph]")
axes[0, 1].set_title("HWFET Drive Cycle")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(t_hwfet, I_hwfet, 'r-', linewidth=0.8)
axes[1, 1].set_ylabel("Cell Current [A]")
axes[1, 1].set_xlabel("Time [s]")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show(block=False)

# ============================================================
# STEP 3: Run drive cycles through PyBaMM at each SOH
# ============================================================
os.makedirs(OUT_DIR, exist_ok=True)

def run_drive_cycle(t_data, i_data, starting_solution, soh_label, cycle_name,
                    actual_cap):
    """
    Run a drive-cycle current profile through PyBaMM from a given
    aged starting state. Save CSV with time/current/voltage/true SOC.
    
    SOC is computed from Coulomb counting on the INPUT current profile.
    Voltage is extracted from the DRIVE CYCLE portion of PyBaMM's output
    (skipping the starting solution data that precedes it).
    """
    # Build drive-cycle step: 2D array [time, current] wrapped in an Experiment
    drive_cycle_array = np.column_stack([t_data, i_data])
    drive_step = pybamm.step.current(drive_cycle_array)
    experiment = pybamm.Experiment([drive_step])

    sim = pybamm.Simulation(
        model, parameter_values=param, experiment=experiment
    )
    sim.solve(starting_solution=starting_solution)
    sol = sim.solution

    # Extract PyBaMM's time and voltage
    time_h_raw = sol["Time [h]"].entries
    time_s_raw = (time_h_raw - time_h_raw[0]) * 3600
    voltage_raw = sol["Voltage [V]"].entries

    # The solution includes the starting solution PLUS the drive cycle.
    # The drive cycle occupies the LAST portion of the time array.
    # Shift time so the drive cycle maps to 0 -> drive_duration.
    drive_duration = t_data[-1] - t_data[0]
    drive_start = time_s_raw[-1] - drive_duration
    time_s_shifted = time_s_raw - drive_start  # now drive cycle is at 0..duration

    print(f"    [DEBUG] PyBaMM total: {len(time_s_raw)} pts, "
          f"0 to {time_s_raw[-1]:.1f}s. "
          f"Drive cycle starts at {drive_start:.1f}s")

    # Interpolate voltage onto our input time grid (0..duration)
    v_interp = interp1d(time_s_shifted, voltage_raw, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    voltage_out = v_interp(t_data)

    print(f"    [DEBUG] Interpolated voltage: {voltage_out.min():.4f} to "
          f"{voltage_out.max():.4f} V")

    # SOC via Coulomb counting on INPUT current (which we trust)
    # Positive current = discharge -> SOC decreases
    dt = np.diff(t_data)
    charge_ah = np.cumsum(i_data[:-1] * dt) / 3600
    charge_ah = np.insert(charge_ah, 0, 0.0)
    true_soc = 1.0 - charge_ah / actual_cap

    df_out = pd.DataFrame({
        "Time [s]":    t_data,
        "Current [A]": i_data,
        "Voltage [V]": voltage_out,
        "True SOC":    true_soc,
    })
    fname = os.path.join(OUT_DIR, f"{cycle_name}_SOH{soh_label}.csv")
    df_out.to_csv(fname, index=False)
    print(f"    Saved {fname}  (SOC: {true_soc[0]:.3f} -> {true_soc[-1]:.3f})")
    return df_out


# Loop over SOH states and drive cycles
print("\nRunning drive-cycle simulations at each SOH state...")
results = {}

for soh_label, cycle_num in soh_cycles.items():
    print(f"\n--- SOH {soh_label}% (from aging cycle {cycle_num}) ---")
    aged_cycle = aging_solution.cycles[cycle_num - 1]
    # Fully charged aged state (end of CV hold)
    if cycle_num == 1:
        charged_state = aged_cycle.steps[1]
    else:
        charged_state = aged_cycle.steps[2]

    # Actual capacity at this SOH: Q_nominal * SOH/100
    actual_cap = NOM_CAP * float(soh_label) / 100.0

    print("  Running UDDS...")
    df_udds = run_drive_cycle(t_udds, I_udds, charged_state, soh_label, "UDDS",
                              actual_cap)
    results[f"UDDS_{soh_label}"] = df_udds

    print("  Running HWFET...")
    df_hwfet = run_drive_cycle(t_hwfet, I_hwfet, charged_state, soh_label, "HWFET",
                               actual_cap)
    results[f"HWFET_{soh_label}"] = df_hwfet

# ============================================================
# STEP 4: Summary plots - separate figures
# ============================================================
soh_colors = {"100": '#1f77b4', "95": '#ff7f0e', "90": '#2ca02c',
              "85": '#d62728', "80": '#9467bd'}

# Figure 1: UDDS Voltage
plt.figure(figsize=(10, 5))
for soh_label, color in soh_colors.items():
    if f"UDDS_{soh_label}" in results:
        df = results[f"UDDS_{soh_label}"]
        plt.plot(df["Time [s]"], df["Voltage [V]"], color=color,
                 linewidth=1, label=f"SOH {soh_label}%")
plt.title("UDDS Voltage Response Across SOH States")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

# Figure 2: UDDS SOC
plt.figure(figsize=(10, 5))
for soh_label, color in soh_colors.items():
    if f"UDDS_{soh_label}" in results:
        df = results[f"UDDS_{soh_label}"]
        plt.plot(df["Time [s]"], df["True SOC"], color=color,
                 linewidth=1, label=f"SOH {soh_label}%")
plt.title("UDDS True SOC Across SOH States")
plt.xlabel("Time [s]"); plt.ylabel("SOC [-]")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

# Figure 3: HWFET Voltage
plt.figure(figsize=(10, 5))
for soh_label, color in soh_colors.items():
    if f"HWFET_{soh_label}" in results:
        df = results[f"HWFET_{soh_label}"]
        plt.plot(df["Time [s]"], df["Voltage [V]"], color=color,
                 linewidth=1, label=f"SOH {soh_label}%")
plt.title("HWFET Voltage Response Across SOH States")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

# Figure 4: HWFET SOC
plt.figure(figsize=(10, 5))
for soh_label, color in soh_colors.items():
    if f"HWFET_{soh_label}" in results:
        df = results[f"HWFET_{soh_label}"]
        plt.plot(df["Time [s]"], df["True SOC"], color=color,
                 linewidth=1, label=f"SOH {soh_label}%")
plt.title("HWFET True SOC Across SOH States")
plt.xlabel("Time [s]"); plt.ylabel("SOC [-]")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

plt.show()

print("\nDrive-cycle data generation complete!")
print(f"CSVs saved in: {OUT_DIR}/")