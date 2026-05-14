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
Run scripts in numerical order (1 through 9).
NOTE: Scripts 1 and 2 take several hours. Pre-generated data is not 
included due to file size limits.

## Requirements
- Python 3.12.10
- PyBaMM 25.12.2
- numpy, pandas, matplotlib, scipy
