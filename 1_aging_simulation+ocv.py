

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================
# AGING SIMULATION + OCV TEST
# Battery Model: Marquis2019 parameter set
# Cell chemistry: NMC cathode / Graphite anode
# Nominal capacity: 0.68 Ah
# Voltage range: 3.2V (empty) to 4.2V (full)
# ============================================

# Load model and parameters
model = pybamm.lithium_ion.DFN(options={"SEI": "ec reaction limited"})
param = pybamm.ParameterValues("Marquis2019")

# ============================================
# STEP 1: RUN AGING SIMULATION
# ============================================

print("Running 100 cycle aging simulation...")
print("This will take several minutes — please wait!")

experiment = pybamm.Experiment(
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

sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
sim.solve()
solution = sim.solution
print("Aging simulation complete!")

# ============================================
# STEP 2: TRACK SOH OVER CYCLES
# ============================================

capacities = []
cycle_numbers = []

for i, cycle in enumerate(solution.cycles):
    if i == 0:
        discharge = cycle.steps[2]
    else:
        discharge = cycle.steps[0]

    cap = abs(discharge["Discharge capacity [A.h]"].entries[-1])
    capacities.append(cap)
    cycle_numbers.append(i + 1)

capacities = np.array(capacities)
initial_capacity = capacities[0]
soh = (capacities / initial_capacity) * 100

# Find SOH threshold cycles
soh_cycles = {"100": 1}
thresholds = [95, 90, 85, 80]
for threshold in thresholds:
    below = np.where(soh <= threshold)[0]
    if len(below) > 0:
        cycle_num = cycle_numbers[below[0]]
        soh_cycles[str(threshold)] = cycle_num
        print(f"SOH {threshold}% reached at cycle {cycle_num}")
    else:
        print(f"SOH {threshold}% not reached in 100 cycles")

print(f"\nSOH cycle mapping: {soh_cycles}")

# Plot SOH over cycles
plt.figure()
plt.plot(cycle_numbers, soh, 'b-')
plt.xlabel("Cycle number")
plt.ylabel("SOH (%)")
plt.title("Battery State of Health vs Cycle Number")
plt.axhline(y=80, color='r', linestyle='--', label='End of life (80%)')
plt.legend()
plt.grid(True, which='major', linestyle='-', linewidth=0.8)
plt.grid(True, which='minor', linestyle=':', linewidth=0.4)
plt.minorticks_on()
plt.xticks(range(0, 101, 5))
plt.yticks(range(0, 101, 5))
plt.xlim(0, 100)
plt.ylim(78, 101)
plt.show()

# ============================================
# STEP 3: EXTRACT DISCHARGE DATA AT EACH SOH
# ============================================

save_folder = "soh_data"
os.makedirs(save_folder, exist_ok=True)

extracted = {}

for soh_label, cycle_num in soh_cycles.items():
    cycle = solution.cycles[cycle_num - 1]

    if cycle_num == 1:
        discharge = cycle.steps[2]
    else:
        discharge = cycle.steps[0]

    time = discharge["Time [h]"].entries
    voltage = discharge["Voltage [V]"].entries
    current = discharge["Current [A]"].entries

    time = time - time[0]
    dt = np.diff(time, prepend=0)
    capacity = np.cumsum(abs(current) * dt)

    peak_idx = np.argmax(voltage)
    time = time[peak_idx:]
    voltage = voltage[peak_idx:]
    current = current[peak_idx:]
    capacity = capacity[peak_idx:]

    capacity = capacity - capacity[0]
    time = time - time[0]

    mask = voltage >= 3.2
    time = time[mask]
    voltage = voltage[mask]
    current = current[mask]
    capacity = capacity[mask]

    extracted[soh_label] = {
        "time": time,
        "voltage": voltage,
        "current": current,
        "capacity": capacity
    }

    df = pd.DataFrame({
        "Time [h]": time,
        "Voltage [V]": voltage,
        "Current [A]": current,
        "Discharge capacity [A.h]": capacity,
    })
    filename = os.path.join(save_folder, f"SOH_{soh_label}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved discharge data: {filename}")

# ============================================
# STEP 4: OCV TEST AT EACH SOH STATE
# Simulate aging effect on OCV by scaling
# negative electrode active material volume
# fraction — this models Loss of Lithium
# Inventory (LLI) caused by SEI growth
# ============================================

print("\n--- Running OCV tests at C/25 for each SOH state ---")

ocv_experiment = pybamm.Experiment(
    [
        "Charge at 1C until 4.2V",
        "Hold at 4.2V until C/50",
        "Discharge at C/25 until 3.2V",
        "Rest for 3600 seconds",
        "Charge at C/25 until 4.2V",
        "Rest for 3600 seconds",
    ]
)

ocv_folder = "ocv_data"
os.makedirs(ocv_folder, exist_ok=True)

ocv_data = {}
colors = ['blue', 'green', 'orange', 'red', 'black']

# Get original active material volume fraction
orig_neg_vol_frac = param["Negative electrode active material volume fraction"]
print(f"Original negative electrode active material volume fraction: {orig_neg_vol_frac:.4f}")

plt.figure(figsize=(10, 6))

for (soh_label, cycle_num), color in zip(soh_cycles.items(), colors):
    print(f"\nRunning OCV test for SOH {soh_label}%...")

    soh_factor = int(soh_label) / 100

    # Create aged parameters
    # Scale negative electrode active material to simulate LLI from SEI
    aged_param = param.copy()
    aged_param["Negative electrode active material volume fraction"] = (
        orig_neg_vol_frac * soh_factor
    )

    # Run OCV simulation with aged parameters
    ocv_sim = pybamm.Simulation(
        model,
        parameter_values=aged_param,
        experiment=ocv_experiment
    )
    ocv_sim.solve()
    ocv_solution = ocv_sim.solution
    cycles_ocv = ocv_solution.cycles

    # Find discharge and charge steps automatically
    discharge_step = None
    charge_step = None

    for ci, cyc in enumerate(cycles_ocv):
        for si, step in enumerate(cyc.steps):
            v = step["Voltage [V]"].entries
            i = step["Current [A]"].entries
            duration = step["Time [h]"].entries[-1] - step["Time [h]"].entries[0]

            if duration > 10 and v[-1] < v[0] and i.mean() > 0 and discharge_step is None:
                discharge_step = step

            if duration > 10 and v[-1] > v[0] and i.mean() < 0 and charge_step is None:
                charge_step = step

    # Extract discharge curve
    t_dis = discharge_step["Time [h]"].entries
    v_dis = discharge_step["Voltage [V]"].entries
    i_dis = discharge_step["Current [A]"].entries
    t_dis = t_dis - t_dis[0]
    dt_dis = np.diff(t_dis, prepend=0)
    cap_dis = np.cumsum(abs(i_dis) * dt_dis)
    max_cap = cap_dis[-1]
    soc_dis = 1 - (cap_dis / max_cap)

    # Extract charge curve
    t_chg = charge_step["Time [h]"].entries
    v_chg = charge_step["Voltage [V]"].entries
    i_chg = charge_step["Current [A]"].entries
    t_chg = t_chg - t_chg[0]
    dt_chg = np.diff(t_chg, prepend=0)
    cap_chg = np.cumsum(abs(i_chg) * dt_chg)
    soc_chg = cap_chg / max_cap

    # Average to get true OCV
    soc_grid = np.linspace(0, 1, 200)
    v_dis_interp = np.interp(soc_grid, soc_dis[::-1], v_dis[::-1])
    v_chg_interp = np.interp(soc_grid, soc_chg, v_chg)
    ocv_avg = (v_dis_interp + v_chg_interp) / 2

    ocv_data[soh_label] = {
        "soc": soc_grid,
        "ocv": ocv_avg,
        "v_discharge": v_dis_interp,
        "v_charge": v_chg_interp,
        "max_cap": max_cap
    }

    # Print key OCV values
    print(f"  Max capacity: {max_cap:.4f} Ah")
    print(f"  SOC 100% -> OCV = {ocv_avg[-1]:.4f} V")
    print(f"  SOC  50% -> OCV = {np.interp(0.50, soc_grid, ocv_avg):.4f} V")
    print(f"  SOC   0% -> OCV = {ocv_avg[0]:.4f} V")

    # Plot OCV curve
    plt.plot(soc_grid, ocv_avg, color=color,
             label=f"SOH {soh_label}%", linewidth=2)

    # Save OCV CSV
    df_ocv = pd.DataFrame({
        "SOC": soc_grid,
        "OCV [V]": ocv_avg,
        "Discharge voltage [V]": v_dis_interp,
        "Charge voltage [V]": v_chg_interp,
        "Max capacity [A.h]": max_cap
    })
    filename = os.path.join(ocv_folder, f"OCV_SOH{soh_label}.csv")
    df_ocv.to_csv(filename, index=False)
    print(f"  Saved to {filename}")

plt.xlabel("SOC")
plt.ylabel("OCV [V]")
plt.title("OCV vs SOC at Different SOH States (C/25 test)")
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.show()

print("\nAging + OCV complete!")

