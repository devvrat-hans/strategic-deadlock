"""Stages 2-4 and baselines on the (R/K, D, rho, I, topology) grid.

Presets
-------
    --preset stage2      best-response dynamics, rho = 0                  (RQ1 regime map)
    --preset stage3      IQL / FP / shared-Q, rho = 0, 10 seeds            (RQ1 with learners)
    --preset stage4      learners + BRD on rho in {0, .05, .1, .2, .5}     (RQ2, g(rho))
    --preset baselines   always-request, Banker's, ordering, back-off, C*
    --preset all         everything above

Add ``--quick`` for a reduced grid (minutes instead of hours). Any grid axis can be
overridden, e.g. ``--rk 1 4 --D 0 10 --methods iql sql``.

Outputs (in --out): raw.csv (one row per seed), summary.csv (mean and 95% CI per cell,
g(rho)), thresholds.csv (theta for RQ1).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deadlock_game.config import GRID_D, GRID_INFO, GRID_RHO, GRID_RK, GRID_TOPOLOGY  # noqa: E402
from deadlock_game.experiments import BASELINES, aggregate, grid_cells, run_grid, thresholds  # noqa: E402

PRESETS = {
    "stage2": dict(methods=["brd"], rhos=[0.0]),
    "stage3": dict(methods=["iql", "fp", "sql"], rhos=[0.0]),
    "stage4": dict(methods=["brd", "iql", "fp", "sql"], rhos=list(GRID_RHO)),
    "baselines": dict(methods=list(BASELINES), rhos=[0.0]),
    "all": dict(methods=["brd", "iql", "fp", "sql", *BASELINES], rhos=list(GRID_RHO)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--topologies", nargs="+", default=list(GRID_TOPOLOGY))
    ap.add_argument("--rk", nargs="+", type=float)
    ap.add_argument("--D", nargs="+", type=float)
    ap.add_argument("--rho", nargs="+", type=float)
    ap.add_argument("--info", nargs="+", default=list(GRID_INFO))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=500, help="MC evaluation episodes per cell")
    ap.add_argument("--max-episodes", type=int, default=5000, help="Q-learning training cap")
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--K", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pre = PRESETS[args.preset]
    methods = args.methods or pre["methods"]
    rks = args.rk or ([0.5, 2.0, 8.0] if args.quick else list(GRID_RK))
    Ds = args.D or ([0.0, 3.162, 100.0] if args.quick else list(GRID_D))
    rhos = args.rho or ([0.0, 0.2, 0.5] if args.quick and len(pre["rhos"]) > 1 else pre["rhos"])
    seeds = list(range(3 if args.quick and args.seeds == 10 else args.seeds))
    out = args.out or os.path.join("results", args.preset + ("_quick" if args.quick else ""))

    cells = grid_cells(args.topologies, rks, Ds, rhos, args.info, methods, seeds,
                       episodes=args.episodes, max_episodes=args.max_episodes, H=args.H, K=args.K)
    print(f"{len(cells)} cells -> {out}", flush=True)
    raw = run_grid(cells, workers=args.workers, out_csv=os.path.join(out, "raw.csv"))
    agg = aggregate(raw)
    agg.to_csv(os.path.join(out, "summary.csv"), index=False)
    th = thresholds(agg)
    th.to_csv(os.path.join(out, "thresholds.csv"), index=False)
    cols = ["topology", "method", "regime", "rk", "D", "rho", "deadlock", "livelock", "stall", "completion",
            "social_cost", "poa_hat", "epsilon", "g_rho"]
    print(agg[[c for c in cols if c in agg]].round(3).to_string(index=False, max_rows=60))
    print(f"\nwrote {out}/raw.csv, summary.csv, thresholds.csv")


if __name__ == "__main__":
    main()
