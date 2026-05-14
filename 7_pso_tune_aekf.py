"""
pso_tune_aekf.py
================
Particle Swarm Optimization (PSO) tuner for AEKF noise parameters.

This script DOES NOT modify the math in aekf_estimator.py. It imports the same
AEKF class and runs it with candidate tuning parameters to minimize a cost.

Optimized parameters (in variance space, searched in log10):
  - R_INIT       (initial measurement variance)
  - R_MIN        (minimum measurement variance during adaptation)
  - Q_SOC_INIT   (initial SOC process variance, Q[0,0])
  - Q_SOC_MAX    (maximum adaptive SOC process variance)

Objective:
  J = mean(SOC_RMSE_pct) + lambda_v * mean(V_RMSE_mV)
across all (cycle, SOH) combinations.

Run from battery_project/:
  py -3 pso_tune_aekf.py
  py -3 pso_tune_aekf.py --particles 20 --iters 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import numpy as np
import pandas as pd

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
ae = importlib.import_module("8_aekf_estimator")
ecm_module = importlib.import_module("6_ecm_plant_model")
ECMParameters = ecm_module.ECMParameters


@dataclass
class TuneResult:
    cost: float
    mean_soc_rmse_pct: float
    mean_v_rmse_mV: float
    per_case: Dict[Tuple[str, int], Dict[str, float]]


def run_case_with_params(
    cycle_name: str,
    soh: int,
    params: ECMParameters,
    R_init: float,
    R_min: float,
    q_soc_init: float,
    q_soc_max: float,
) -> Dict[str, float]:
    """Run one cycle/SOH case with explicit AEKF tuning values."""
    fname = os.path.join(ae.DRIVE_FOLDER, f"{cycle_name}_SOH{soh}.csv")
    df = pd.read_csv(fname)

    t = df["Time [s]"].values
    current = df["Current [A]"].values
    voltage = df["Voltage [V]"].values
    soc_true = df["True SOC"].values

    capacity = ae.NOM_CAP * soh / 100.0

    # Build Q from baseline but override SOC channel
    Q_init = ae.Q_INIT.copy()
    Q_init[0, 0] = q_soc_init

    kf = ae.AEKF(
        params=params,
        soh=soh,
        capacity=capacity,
        soc_init=ae.SOC_INIT_GUESS,
        P_init=ae.P_INIT,
        R_init=R_init,
        Q_init=Q_init,
        adapt_window=ae.ADAPT_WINDOW,
        adapt=ae.ADAPT_ENABLED,
    )

    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        kf.step(current[k - 1], current[k], voltage[k], dt)

    soc_est = np.array([ae.SOC_INIT_GUESS] + kf.history["soc"])
    v_pred = np.array([voltage[0]] + kf.history["v_pred"])

    soc_rmse_pct = float(np.sqrt(np.mean((soc_est - soc_true) ** 2)) * 100.0)
    v_rmse_mV = float(np.sqrt(np.mean((v_pred - voltage) ** 2)) * 1000.0)

    return {"soc_rmse_pct": soc_rmse_pct, "v_rmse_mV": v_rmse_mV}


def evaluate_candidate(
    x_log10: np.ndarray,
    params: ECMParameters,
    cycles: List[str],
    soh_list: List[int],
    lambda_v: float,
) -> TuneResult:
    """
    Evaluate one particle.
    x_log10 = [log10(R_INIT), log10(R_MIN), log10(Q_SOC_INIT), log10(Q_SOC_MAX)]
    """
    R_init = 10.0 ** float(x_log10[0])
    R_min = 10.0 ** float(x_log10[1])
    q_soc_init = 10.0 ** float(x_log10[2])
    q_soc_max = 10.0 ** float(x_log10[3])

    # Feasibility
    if R_min > R_init or q_soc_init > q_soc_max:
        return TuneResult(cost=1e9, mean_soc_rmse_pct=1e9, mean_v_rmse_mV=1e9, per_case={})

    # Temporarily patch module-level bounds used inside AEKF.step()
    old_R_min = ae.R_MIN
    old_Q_soc_max = ae.Q_SOC_MAX
    ae.R_MIN = R_min
    ae.Q_SOC_MAX = q_soc_max

    per_case: Dict[Tuple[str, int], Dict[str, float]] = {}
    try:
        for cycle_name in cycles:
            for soh in soh_list:
                per_case[(cycle_name, soh)] = run_case_with_params(
                    cycle_name, soh, params, R_init, R_min, q_soc_init, q_soc_max,
                )
    finally:
        ae.R_MIN = old_R_min
        ae.Q_SOC_MAX = old_Q_soc_max

    soc_vals = [v["soc_rmse_pct"] for v in per_case.values()]
    v_vals = [v["v_rmse_mV"] for v in per_case.values()]
    mean_soc = float(np.mean(soc_vals))
    mean_v = float(np.mean(v_vals))
    cost = mean_soc + lambda_v * mean_v

    return TuneResult(cost=cost, mean_soc_rmse_pct=mean_soc, mean_v_rmse_mV=mean_v, per_case=per_case)


def pso_optimize(
    params: ECMParameters,
    cycles: List[str],
    soh_list: List[int],
    lambda_v: float,
    n_particles: int,
    n_iters: int,
    seed: int,
) -> Tuple[np.ndarray, TuneResult]:
    """PSO in log10-variance space."""
    rng = np.random.default_rng(seed)

    # Search bounds in log10(variance) space — wide enough to cover the
    # physically-motivated starting point and let PSO explore freely.
    #   R_INIT / R_MIN : 1 mV stdev  →  200 mV stdev
    #   Q_SOC_INIT     : 1e-8 SOC/√step  →  1e-4 SOC/√step
    #   Q_SOC_MAX      : 1e-7 SOC/√step  →  5e-3 SOC/√step
    lb = np.array([
        math.log10((1e-3) ** 2),    # R_INIT min  (1 mV)
        math.log10((5e-4) ** 2),    # R_MIN min   (0.5 mV)
        math.log10((1e-8) ** 2),    # Q_SOC_INIT min
        math.log10((1e-7) ** 2),    # Q_SOC_MAX min
    ])
    ub = np.array([
        math.log10((200e-3) ** 2),  # R_INIT max  (200 mV)
        math.log10((100e-3) ** 2),  # R_MIN max   (100 mV)
        math.log10((1e-4) ** 2),    # Q_SOC_INIT max
        math.log10((5e-3) ** 2),    # Q_SOC_MAX max
    ])

    dim = len(lb)
    X = rng.uniform(lb, ub, size=(n_particles, dim))
    V = np.zeros_like(X)

    pbest_X = X.copy()
    pbest_cost = np.full(n_particles, np.inf)

    gbest_x = X[0].copy()
    gbest_cost = np.inf
    gbest_res = None

    w, c1, c2 = 0.72, 1.49, 1.49  # standard constriction PSO

    for it in range(1, n_iters + 1):
        for i in range(n_particles):
            res = evaluate_candidate(X[i], params, cycles, soh_list, lambda_v)

            if res.cost < pbest_cost[i]:
                pbest_cost[i] = res.cost
                pbest_X[i] = X[i].copy()

            if res.cost < gbest_cost:
                gbest_cost = res.cost
                gbest_x = X[i].copy()
                gbest_res = res

        print(
            f"[Iter {it:02d}/{n_iters}]  "
            f"best J={gbest_cost:.4f}  "
            f"SOC={gbest_res.mean_soc_rmse_pct:.4f}%  "
            f"V={gbest_res.mean_v_rmse_mV:.2f}mV"
        )

        r1 = rng.random(size=(n_particles, dim))
        r2 = rng.random(size=(n_particles, dim))
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_x - X)
        X = np.clip(X + V, lb, ub)

    return gbest_x, gbest_res


def decode(x_log10: np.ndarray) -> Dict[str, float]:
    return {
        "R_INIT": float(10.0 ** x_log10[0]),
        "R_MIN": float(10.0 ** x_log10[1]),
        "Q_SOC_INIT": float(10.0 ** x_log10[2]),
        "Q_SOC_MAX": float(10.0 ** x_log10[3]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PSO tuner for AEKF noise parameters.")
    ap.add_argument("--particles", type=int, default=30)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lambda-v", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="aekf_pso_best.json")
    args = ap.parse_args()

    print("Loading parameter tables...")
    ecm = ECMParameters(ae.R0_FILE, ae.RC_FILE, ae.OCV_FOLDER, ae.SOH_LIST)
    cycles = ["UDDS", "HWFET"]

    best_x, best_res = pso_optimize(
        params=ecm,
        cycles=cycles,
        soh_list=ae.SOH_LIST,
        lambda_v=args.lambda_v,
        n_particles=args.particles,
        n_iters=args.iters,
        seed=args.seed,
    )

    best = decode(best_x)
    print("\n" + "=" * 60)
    print("BEST TUNING PARAMETERS")
    print("=" * 60)
    for k, v in best.items():
        print(f"  {k:<14} = {v:.6e}   (stdev = {v**0.5:.6e})")

    print(
        f"\nObjective J = {best_res.cost:.6f}  "
        f"(mean SOC RMSE = {best_res.mean_soc_rmse_pct:.4f}%, "
        f"mean V RMSE = {best_res.mean_v_rmse_mV:.2f} mV)"
    )

    print("\nPer-case breakdown:")
    print(f"  {'Case':<20} {'SOC RMSE %':<14} {'V RMSE mV':<14}")
    print("  " + "-" * 48)
    for (cycle, soh), m in sorted(best_res.per_case.items()):
        print(f"  {cycle}_SOH{soh:<5}      {m['soc_rmse_pct']:<14.4f} {m['v_rmse_mV']:<14.2f}")

    # Save to JSON
    payload = {
        "objective": {"formula": "J = mean(SOC_RMSE_pct) + lambda_v * mean(V_RMSE_mV)", "lambda_v": args.lambda_v},
        "best_params": best,
        "best_metrics": {"J": best_res.cost, "mean_soc_rmse_pct": best_res.mean_soc_rmse_pct, "mean_v_rmse_mV": best_res.mean_v_rmse_mV},
        "per_case": {f"{c}_SOH{s}": m for (c, s), m in best_res.per_case.items()},
        "pso": {"particles": args.particles, "iters": args.iters, "seed": args.seed},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
