"""Stage 1: exact two-agent equilibria over a grid of (R/K, D, rho).

For each grid point we solve the two-agent standoff game by backward induction with
nashpy support enumeration at every stage, and report:
    h*            holding time at which Yield first enters the equilibrium support
    yield hazard  on-path P(Yield) at the standoff for h = 1..5
    delta, lambda, completion, social cost C, C*, PoA-hat (Monte Carlo + exact)
    PoA bounds    cost of the welfare-min / welfare-max equilibrium selections over C*
    epsilon       exploitability of the computed equilibrium (sanity check: ~0)

    python experiments/stage1_two_agent.py [--quick] [--out results/stage1]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deadlock_game.config import GRID_D, GRID_RHO, GRID_RK, params_from_ratio  # noqa: E402
from deadlock_game.exact import centralized_optimum, evaluate  # noqa: E402
from deadlock_game.experiments import thresholds  # noqa: E402
from deadlock_game.metrics import evaluate_mc  # noqa: E402
from deadlock_game.two_agent import solve_two_agent, stage_equilibria, stage_matrix, verify_spe  # noqa: E402


def one_shot_report(params):
    """Equilibria of the plain one-shot matrix M_rho(h) (Equation 11), for reference."""
    M, names = stage_matrix(params, 1)
    eqs = stage_equilibria(M, M.T)
    return "; ".join(
        "(" + ",".join(f"{names[k][0]}:{x[k]:.2f}" for k in range(3) if x[k] > 1e-9) + " | "
        + ",".join(f"{names[k][0]}:{y[k]:.2f}" for k in range(3) if y[k] > 1e-9) + ")" for x, y in eqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid for a fast check")
    ap.add_argument("--out", default="results/stage1")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--no-eps", action="store_true", help="skip the exploitability check")
    args = ap.parse_args()
    rks = GRID_RK
    Ds = (0.0, 1.0, 10.0) if args.quick else GRID_D
    rhos = (0.0, 0.2) if args.quick else GRID_RHO
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for rk, D, rho in itertools.product(rks, Ds, rhos):
        p = params_from_ratio(rk, D, rho, H=args.H)
        row = {"rk": rk, "D": D, "rho": rho, "topology": "standoff", "regime": "full", "method": "spe"}
        costs = {}
        for sel in ("symmetric", "welfare_min", "welfare_max"):
            sol = solve_two_agent(p, selection=sel)
            costs[sel] = evaluate(sol.env, sol.profile).social_cost
            if sel == "symmetric":
                main_sol = sol
        sol = main_sol
        c_star = centralized_optimum(sol.env).c_star
        s = evaluate_mc(sol.env, sol.profile, args.episodes, seed=1)
        haz = sol.yield_hazard()
        row.update(
            h_star=sol.h_star(),
            **{f"p_yield_h{h}": haz.get(h, float("nan")) for h in range(1, 6)},
            deadlock=s.deadlock, livelock=s.livelock, stall=s.stall, completion=s.completion, latency=s.latency,
            social_cost=s.social_cost, exact_cost=costs["symmetric"], c_star=c_star,
            poa_hat=s.social_cost / c_star,
            poa_worst=costs["welfare_min"] / c_star, poa_best=costs["welfare_max"] / c_star,
            n_stage_eq_max=max(sol.n_equilibria.values()),
            epsilon=float("nan") if args.no_eps else verify_spe(sol),
            one_shot_equilibria=one_shot_report(p),
        )
        rows.append(row)
        print(f"R/K={rk:<4} D={D:<7} rho={rho:<5} h*={row['h_star']} delta={s.deadlock:.3f} "
              f"lambda={s.livelock:.3f} C={s.social_cost:.2f} C*={c_star:.2f} "
              f"PoA in [{row['poa_best']:.2f},{row['poa_worst']:.2f}] eps={row['epsilon']:.1e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "stage1_two_agent.csv"), index=False)
    th = thresholds(df)
    th.to_csv(os.path.join(args.out, "stage1_thresholds.csv"), index=False)
    print("\nthreshold theta (R/K where delta crosses 1/2), rho = 0:\n", th.to_string(index=False))
    print(f"\nwrote {args.out}/stage1_two_agent.csv")


if __name__ == "__main__":
    main()
