import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================
# AGING SIMULATION + HPPC TEST
# Battery Model: Marquis2019 parameter set
# Cell chemistry: NMC cathode / Graphite anode
# Nominal capacity: 0.68 Ah
# Voltage range: 3.2V (empty) to 4.2V (full)
#
# CHANGE LOG:
#   - Rest periods: 600s (was 300s)
#   - Added charge pulses (0.5C, 1C) after discharge pulses
#   - Full pulse sequence per SOC level:
#       0.5C dis 30s → rest 600s →
#       1C dis 30s   → rest 600s →
#       2C dis 30s   → rest 600s →
#       0.5C chg 30s → rest 600s →
#       1C chg 30s   → rest 600s →
#       1C dis 360s  (step-down to next SOC)
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
# STEP 3: HPPC TEST AT EACH SOH STATE
#   Discharge + Charge pulses, 600s rests
# ============================================

print("\n--- Running HPPC simulations ---")

hppc_experiment = pybamm.Experiment(
    [
        # Discharge pulses
        "Discharge at 0.5C for 30 seconds",
        "Rest for 600 seconds",
        "Discharge at 1C for 30 seconds",
        "Rest for 600 seconds",
        "Discharge at 2C for 30 seconds",
        "Rest for 600 seconds",
        # Charge pulses
        "Charge at 0.5C for 30 seconds",
        "Rest for 600 seconds",
        "Charge at 1C for 30 seconds",
        "Rest for 600 seconds",
        # Step-down to next SOC
        "Discharge at 1C for 360 seconds",
    ] * 9
)
hppc_folder = "hppc_data"
os.makedirs(hppc_folder, exist_ok=True)

for soh_label, cycle_num in soh_cycles.items():
    print(f"\nRunning HPPC for SOH {soh_label}% (from cycle {cycle_num})...")

    aged_cycle = solution.cycles[cycle_num - 1]

    if cycle_num == 1:
        starting_solution = aged_cycle.steps[1]
    else:
        starting_solution = aged_cycle.steps[2]

    hppc_sim = pybamm.Simulation(
        model,
        parameter_values=param,
        experiment=hppc_experiment
    )

    hppc_sim.solve(starting_solution=starting_solution)
    hppc_solution = hppc_sim.solution

    time = hppc_solution["Time [h]"].entries
    voltage = hppc_solution["Voltage [V]"].entries
    current = hppc_solution["Current [A]"].entries

    time = time - time[0]

    # Save to CSV
    df = pd.DataFrame({
        "Time [h]": time,
        "Voltage [V]": voltage,
        "Current [A]": current
    })

    filename = os.path.join(hppc_folder, f"HPPC_SOH{soh_label}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved HPPC SOH {soh_label}% to {filename}")

    # Plot voltage and current on same figure with twin axes
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(time * 3600, voltage, 'b-', label="Voltage")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time * 3600, current, 'r-', label="Current")
    ax2.set_ylabel("Current [A]", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f"HPPC Test - SOH {soh_label}%")
    plt.tight_layout()
    plt.show(block=False)

plt.show()
print("\nAging + HPPC complete!")