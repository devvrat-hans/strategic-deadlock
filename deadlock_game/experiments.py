"""Experiment runner for Stages 2-4 and the baselines.

One *cell* = (topology, R/K, D/c0, rho, information regime, method, seed). For each cell we
solve / train the profile, then report

    delta (deadlock episodes), lambda (livelock episodes), stall episodes, completion rate, latency,
    social cost C (Monte Carlo, base payoff), exact C, C*, PoA-hat, exploitability epsilon.

:func:`aggregate` turns per-seed rows into mean and 95% confidence interval per cell and adds
g(rho) (Equation 12) relative to the rho = 0 row of the same cell.
"""
from __future__ import annotations

import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .baselines import bankers_policy, resource_ordering
from .best_response import best_response_dynamics
from .config import GameParams, make_topology, params_from_ratio
from .env import DeadlockEnv
from .exact import centralized_optimum, evaluate, exploitability
from .learning import QConfig, train
from .metrics import evaluate_mc, gap_closed, mean_ci, price_of_anarchy
from .policies import always_request, backoff_profile

LEARNED = ("brd", "iql", "sql", "fp", "spe")
BASELINES = ("always-request", "bankers", "resource-ordering", "exp-backoff", "centralized")
METRICS = ("deadlock", "livelock", "stall", "completion", "latency", "makespan", "social_cost",
           "exact_cost", "poa_hat", "epsilon")


@dataclass(frozen=True)
class Cell:
    topology: str
    rk: float
    D: float
    rho: float
    regime: str
    method: str
    seed: int
    n: int = 3
    m: int = 3
    topology_seed: int = 0
    K: float = 4.0
    H: int = 20
    tau_max: int = 3
    L: int = 4
    gamma: float = 0.95
    h_bar: int = 4
    episodes: int = 500
    max_episodes: int = 5000
    fp_iterations: int = 15


def make_env(cell: Cell) -> DeadlockEnv:
    top = make_topology(cell.topology, cell.n, cell.m, cell.topology_seed)
    params = params_from_ratio(cell.rk, cell.D, cell.rho, K=cell.K, H=cell.H, tau_max=cell.tau_max,
                               L=cell.L, gamma=cell.gamma, h_bar=cell.h_bar)
    return DeadlockEnv(top, params)


@lru_cache(maxsize=256)
def _c_star(topology: str, n: int, m: int, tseed: int, params: GameParams) -> float:
    env = DeadlockEnv(make_topology(topology, n, m, tseed), params.base)
    return centralized_optimum(env).c_star


def run_cell(cell: Cell) -> dict:
    t0 = time.time()
    env = make_env(cell)
    c_star = _c_star(cell.topology, env.n, env.m, cell.topology_seed, env.params.base)
    eval_env = env
    eps = float("nan")
    extra: dict = {}
    m = cell.method
    if m == "brd":
        res = best_response_dynamics(env, cell.regime)
        policy, eps = res.profile, res.epsilon
        extra = {"rounds": res.rounds, "converged": res.converged, "cycled": res.cycled}
    elif m in ("iql", "sql", "fp"):
        res = train(m, env, cell.regime, seed=cell.seed, cfg=QConfig(max_episodes=cell.max_episodes),
                    fp_iterations=cell.fp_iterations)
        policy, eps = res.profile, res.epsilon
        extra = {"rounds": res.episodes, "converged": res.converged}
    elif m == "spe":
        from .two_agent import solve_two_agent

        sol = solve_two_agent(env.params, env.topology)
        policy, eps = sol.profile, 0.0
        extra = {"h_star": sol.h_star()}
    elif m == "always-request":
        policy = always_request(env.n)
        eps = exploitability(env, policy)[0]
    elif m == "bankers":
        policy = bankers_policy()
    elif m == "resource-ordering":
        eval_env, policy = resource_ordering(env)
    elif m == "exp-backoff":
        policy = backoff_profile(env.n)
    elif m == "centralized":
        policy = centralized_optimum(env).policy
    else:
        raise ValueError(m)
    s = evaluate_mc(eval_env, policy, cell.episodes, seed=10_000 + cell.seed)
    exact_cost = evaluate(eval_env, policy).social_cost if policy.markov else float("nan")
    row = dict(cell.__dict__)
    row.update(
        deadlock=s.deadlock, livelock=s.livelock, stall=s.stall, completion=s.completion, latency=s.latency,
        makespan=s.makespan, social_cost=s.social_cost, exact_cost=exact_cost, c_star=c_star,
        poa_hat=price_of_anarchy(s.social_cost, c_star), epsilon=eps, seconds=time.time() - t0,
    )
    row.update(extra)
    return row


def grid_cells(topologies, rks, Ds, rhos, regimes, methods, seeds, **kw) -> list[Cell]:
    cells = []
    for top, rk, D, rho, reg, meth, seed in itertools.product(topologies, rks, Ds, rhos, regimes, methods, seeds):
        if meth in BASELINES and reg != regimes[0]:
            continue  # baselines do not depend on the information regime
        if meth in ("bankers", "resource-ordering", "exp-backoff", "centralized", "always-request") and rho != rhos[0]:
            continue  # nor on the agents' private holding-cost escalation
        if meth in ("brd", "fp", "spe", "always-request", "bankers", "resource-ordering", "centralized") \
                and seed != seeds[0]:
            continue  # deterministic solvers: one run, MC seed only matters for estimation
        if meth == "spe" and top != "standoff":
            continue
        cells.append(Cell(top, float(rk), float(D), float(rho), reg, meth, int(seed), **kw))
    return cells


def run_grid(cells: list[Cell], workers: int | None = None, out_csv: str | None = None,
             progress: bool = True) -> pd.DataFrame:
    """Run all cells. With ``out_csv``, finished rows are checkpointed to ``<out_csv>.partial``
    and an interrupted run resumes from it (cells already done are skipped)."""
    rows: list[dict] = []
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    partial = out_csv + ".partial" if out_csv else None
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    if partial and os.path.exists(partial):
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        fields = list(Cell.__dataclass_fields__)
        done = {tuple(r[f] for f in fields) for r in rows}
        before = len(cells)
        cells = [c for c in cells if tuple(getattr(c, f) for f in fields) not in done]
        print(f"resuming: {before - len(cells)} cells already done", flush=True)
    total = len(cells)
    every = max(1, total // 100)

    def checkpoint():
        if partial:
            pd.DataFrame(rows).to_csv(partial, index=False)

    if workers == 1:
        for k, c in enumerate(cells):
            rows.append(run_cell(c))
            if progress:
                print(f"[{k + 1}/{total}] {c.method} {c.topology} R/K={c.rk} D={c.D} rho={c.rho} {c.regime}",
                      flush=True)
            if (k + 1) % every == 0:
                checkpoint()
    else:
        # slowest methods first so the tail of the run is not one long cell
        order = {"fp": 0, "brd": 1, "sql": 2, "iql": 3}
        cells = sorted(cells, key=lambda c: (order.get(c.method, 9), -c.rho))
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_cell, c): c for c in cells}
            for k, f in enumerate(as_completed(futs)):
                rows.append(f.result())
                if (k + 1) % every == 0:
                    checkpoint()
                    if progress:
                        print(f"[{k + 1}/{total}] cells done", flush=True)
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        if partial and os.path.exists(partial):
            os.remove(partial)
    return df


KEYS = ["topology", "rk", "D", "rho", "regime", "method"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over seeds per cell, plus g(rho) w.r.t. the rho = 0 cell."""
    out = []
    for key, g in df.groupby(KEYS):
        row = dict(zip(KEYS, key))
        row["seeds"] = len(g)
        row["c_star"] = float(g["c_star"].iloc[0])
        for mname in METRICS:
            if mname in g:
                mu, ci = mean_ci(g[mname].astype(float))
                row[mname] = mu
                row[mname + "_ci"] = ci
        out.append(row)
    agg = pd.DataFrame(out)
    if agg.empty:
        return agg
    base = agg[agg["rho"] == 0.0].set_index(["topology", "rk", "D", "regime", "method"])["social_cost"]
    gs = []
    for _, r in agg.iterrows():
        k = (r["topology"], r["rk"], r["D"], r["regime"], r["method"])
        gs.append(gap_closed(base[k], r["social_cost"], r["c_star"]) if k in base.index and r["rho"] > 0 else np.nan)
    agg["g_rho"] = gs
    return agg


def thresholds(agg: pd.DataFrame, level: float = 0.5) -> pd.DataFrame:
    """theta(n, T, I, D): R/K at which delta crosses 1/2 (RQ1)."""
    from .two_agent import find_threshold

    rows = []
    for key, g in agg[agg["rho"] == 0.0].groupby(["topology", "D", "regime", "method"]):
        g = g.sort_values("rk")
        rows.append(dict(zip(["topology", "D", "regime", "method"], key),
                         theta=find_threshold(g["rk"], g["deadlock"], level)))
    return pd.DataFrame(rows)


def regime_label(deadlock: float, livelock: float, stall: float = 0.0, hi: float = 0.5) -> str:
    """Classify a cell: deadlock- / livelock- / stall-dominated, strategic, or mixed."""
    if deadlock >= hi:
        return "deadlock"
    if livelock >= hi:
        return "livelock"
    if stall >= hi:
        return "stall"
    if deadlock < 1 - hi and livelock < 1 - hi and stall < 1 - hi:
        return "strategic"
    return "mixed"
