ME-576 Battery Systems Modelling and Control — Final Project
Adaptive Extended Kalman Filter for SOC Estimation Across Battery Aging States

Authors: Aditya Bende, Yash Pantahri, Farook Algehlani, Rishitha Palalparthi
University of Michigan-Dearborn | Prof. Youngki Kim

=======================================================================
EXECUTION ORDER
=======================================================================

NOTE: Scripts 1 and 2 run full 100-cycle DFN simulations and take
several hours. The output data folders are already included so you
can skip directly to Step 6.

Step 1:  py -3.12 1_aging_simulation+ocv.py
         → Generates soh_data/ and ocv_data/

Step 2:  py -3.12 2_aging_simulation+hppc_v2.py
         → Generates hppc_data/

Step 3:  py -3.12 3_parameter_identification_R0_v2.py
         → Generates 2rc_parameters/ecm_r0_2d.csv

Step 4:  py -3.12 4_parameter_identification_r1c1r2c2_v2.py
         → Generates 2rc_parameters/ecm_r1c1r2c2_2d.csv

Step 5:  py -3.12 5_drive_cycle_simulation.py
         → Generates drive_cycle_data/ (10 CSVs)

Step 6:  py -3.12 9_unoptimized_aekf_estimator.py
         → Runs baseline AEKF, generates plots and aekf_excel_output/

Step 7:  [OPTIONAL] py -3.12 7_pso_tune_aekf.py --particles 30 --iters 50
         → PSO noise parameter optimisation (takes ~30-60 min)
         → NOTE: aekf_pso_best.json is already included.
           8_optimized_aekf_estimator.py reads directly from this file.
           Only run this if you want to re-tune the parameters from scratch.

Step 8:  py -3.12 8_optimized_aekf_estimator.py
         → Runs PSO-tuned AEKF, generates final plots

=======================================================================
REQUIREMENTS
=======================================================================
Python 3.12.10
PyBaMM 25.12.2
numpy, pandas, matplotlib, scipy

Install dependencies:
  python -m pip install pybamm numpy pandas matplotlib scipy

=======================================================================
REPORT
=======================================================================
Final_Report.pdf — compiled report