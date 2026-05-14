# AEKF SOC Estimation Across Battery Aging States

Complete pipeline for State-of-Charge (SOC) estimation of lithium-ion cells 
across five aging states (100%–80% SOH) using electrochemical modeling and 
adaptive state estimation.

## Authors
- Aditya Bende — University of Michigan-Dearborn
- Yash Pantahri — University of Michigan-Dearborn

## Project Overview
This project implements a five-task pipeline:
1. DFN aging simulation (100 cycles, SEI growth, lithium plating, LAM)
2. OCV–SOC characterisation at each SOH state
3. HPPC testing and 2RC ECM parameter identification
4. Drive cycle simulation (EPA UDDS + HWFET)
5. AEKF implementation with PSO-based noise optimisation

## Results
- Baseline AEKF: SOC RMSE < 2.5% across all 10 test cases
- PSO-optimised AEKF: SOC RMSE < 1% across all 10 test cases
- Convergence under 1 second from a 15% initial error
## Execution Order

> **NOTE:** Scripts 1 and 2 run full 100-cycle DFN simulations and take several hours. The output data folders are already included so you can skip directly to Step 6.

**Step 1:**  `py -3.12 1_aging_simulation+ocv.py`  
→ Generates soh_data/ and ocv_data/

**Step 2:**  `py -3.12 2_aging_simulation+hppc_v2.py`  
→ Generates hppc_data/

**Step 3:**  `py -3.12 3_parameter_identification_R0_v2.py`  
→ Generates 2rc_parameters/ecm_r0_2d.csv

**Step 4:**  `py -3.12 4_parameter_identification_r1c1r2c2_v2.py`  
→ Generates 2rc_parameters/ecm_r1c1r2c2_2d.csv

**Step 5:**  `py -3.12 5_drive_cycle_simulation.py`  
→ Generates drive_cycle_data/ (10 CSVs)

**Step 6:**  `py -3.12 9_unoptimized_aekf_estimator.py`  
→ Runs baseline AEKF, generates plots and aekf_excel_output/

**Step 7 (OPTIONAL):**  `py -3.12 7_pso_tune_aekf.py --particles 30 --iters 50`  
→ PSO noise parameter optimisation (takes ~30-60 min)  
→ NOTE: aekf_pso_best.json is already included. Only run this if you want to re-tune the parameters from scratch.

**Step 8:**  `py -3.12 8_optimized_aekf_estimator.py`  
→ Runs PSO-tuned AEKF, generates final plots

## Requirements
- Python 3.12.10
- PyBaMM 25.12.2
- numpy, pandas, matplotlib, scipy
