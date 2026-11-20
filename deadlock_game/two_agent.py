"""Stage 1: exact two-agent equilibria.

For n = 2 we compute subgame-perfect equilibria of the finite-horizon stochastic game
Gamma_rho by backward induction. At every (t, s) the agents play the bimatrix stage game

    M_k[a1, a2] = E[ u^rho_k(s, a, s') + gamma * V_k(t+1, s') ]

(i.e. M_rho(h) of Equation 11 augmented with continuation values); all its Nash
equilibria are enumerated with nashpy's support enumeration (plus an explicit pure-strategy
check, which is robust to degenerate games). One equilibrium is selected by a rule
(``symmetric``, ``welfare_max`` or ``welfare_min``) and its values are propagated.

At the standoff states (each agent holds its first lock and needs the other's) this is
exactly the war of attrition of Section 3; the time at which Yield first enters the
equilibrium support gives the threshold holding time h*.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import nashpy as nash
import numpy as np

from .config import YIELD, GameParams, Topology, standoff
from .env import DeadlockEnv, State
from .exact import _reachable, evaluate, exploitability
from .policies import Profile, StateDistPolicy

TOL = 1e-9


def stage_equilibria(M1: np.ndarray, M2: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """All (found) Nash equilibria of the bimatrix game (M1, M2)."""
    r, c = M1.shape
    eqs: list[tuple[np.ndarray, np.ndarray]] = []

    def add(x, y):
        for (a, b) in eqs:
            if np.allclose(a, x, atol=1e-7) and np.allclose(b, y, atol=1e-7):
                return
        eqs.append((x, y))

    # pure equilibria (exact, handles degeneracy)
    for i in range(r):
        for j in range(c):
            if M1[i, j] >= M1[:, j].max() - TOL and M2[i, j] >= M2[i, :].max() - TOL:
                add(np.eye(r)[i], np.eye(c)[j])
    if r > 1 and c > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                for x, y in nash.Game(M1, M2).support_enumeration(tol=1e-12):
                    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and _is_ne(M1, M2, x, y):
                        add(np.clip(x, 0, None) / np.clip(x, 0, None).sum(),
                            np.clip(y, 0, None) / np.clip(y, 0, None).sum())
            except Exception:  # pragma: no cover - nashpy edge cases
                pass
        if not eqs:  # pragma: no cover - fall back to Lemke-Howson
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, y = nash.Game(M1, M2).lemke_howson(initial_dropped_label=0)
                add(x, y)
    return eqs


def _is_ne(M1, M2, x, y, tol: float = 1e-6) -> bool:
    v1 = x @ M1 @ y
    v2 = x @ M2 @ y
    return (M1 @ y).max() <= v1 + tol and (x @ M2).max() <= v2 + tol


def select(eqs, M1, M2, rule: str):
    vals = [(float(x @ M1 @ y), float(x @ M2 @ y)) for x, y in eqs]
    if rule == "welfare_max":
        k = int(np.argmax([a + b for a, b in vals]))
    elif rule == "welfare_min":
        k = int(np.argmin([a + b for a, b in vals]))
    elif rule == "symmetric":
        # prefer the equilibrium in which both agents use the same mixed action
        # (agents are exchangeable and cannot coordinate who yields), then welfare
        def key(idx):
            x, y = eqs[idx]
            asym = np.abs(x - y).sum() if len(x) == len(y) else 1.0
            return (round(asym, 6), -(vals[idx][0] + vals[idx][1]))

        k = min(range(len(eqs)), key=key)
    else:
        raise ValueError(rule)
    return eqs[k], vals[k]


@dataclass
class TwoAgentSolution:
    env: DeadlockEnv
    selection: str
    profile: Profile                              # SPE as state-dependent mixed policies
    values: tuple[float, float]                   # V_k(0, s0)
    n_equilibria: dict[tuple[int, State], int]    # number of stage equilibria per (t, s)
    strategies: dict[tuple[int, State], tuple[dict, dict]]

    def standoff_states(self):
        """(t, s) where both agents hold their first lock, need the other's, and can act."""
        out = []
        for (t, s) in self.strategies:
            p, h, kappa, d, _ = s
            if p == (1, 1) and kappa == (0, 0) and d == (0, 0):
                out.append((t, s))
        return sorted(out, key=lambda x: x[0])

    def yield_hazard(self) -> dict[int, float]:
        """On-path probability of Yield (agent 0) at the standoff, by holding time h = t."""
        ev = evaluate(self.env, self.profile)
        acc: dict[int, list[float]] = {}
        for (t, s) in self.standoff_states():
            w = ev.occupancy[t].get(s, 0.0)
            py = self.strategies[(t, s)][0].get(YIELD, 0.0)
            num, den = acc.get(t, [0.0, 0.0])
            acc[t] = [num + w * py, den + w]
        haz = {}
        for t, (num, den) in acc.items():
            if den > 1e-12:
                haz[t] = num / den
        return haz

    def h_star(self) -> int | None:
        """Smallest holding time at which Yield is in the on-path equilibrium support."""
        for t, py in sorted(self.yield_hazard().items()):
            if py > 1e-9:
                return t
        # off-path fallback: any standoff state with Yield in the support
        hs = [s[1][0] for (t, s) in self.standoff_states() if self.strategies[(t, s)][0].get(YIELD, 0) > 1e-9]
        return min(hs) if hs else None


def solve_two_agent(params: GameParams, topology: Topology | None = None,
                    selection: str = "symmetric") -> TwoAgentSolution:
    top = topology or standoff()
    if top.n_agents != 2:
        raise ValueError("exact equilibria are computed for n = 2 only")
    env = DeadlockEnv(top, params)
    g = params.gamma
    layers = _reachable(env, None, None)
    V_next: dict[State, tuple[float, float]] = {}
    strategies: dict[tuple[int, State], tuple[dict, dict]] = {}
    n_eq: dict[tuple[int, State], int] = {}
    for t in range(params.H - 1, -1, -1):
        V_t: dict[State, tuple[float, float]] = {}
        for s in layers[t]:
            if env.is_terminal(s):
                V_t[s] = (0.0, 0.0)
                continue
            A1 = env.available_actions(s, 0)
            A2 = env.available_actions(s, 1)
            M1 = np.zeros((len(A1), len(A2)))
            M2 = np.zeros((len(A1), len(A2)))
            for (i, a1), (j, a2) in itertools.product(enumerate(A1), enumerate(A2)):
                for o in env.transitions(s, (a1, a2)):
                    v = V_next.get(o.next_state, (0.0, 0.0))
                    M1[i, j] += o.prob * (o.rewards[0] + g * v[0])
                    M2[i, j] += o.prob * (o.rewards[1] + g * v[1])
            eqs = stage_equilibria(M1, M2)
            (x, y), val = select(eqs, M1, M2, selection)
            strategies[(t, s)] = (
                {a: float(x[k]) for k, a in enumerate(A1) if x[k] > 1e-12},
                {a: float(y[k]) for k, a in enumerate(A2) if y[k] > 1e-12},
            )
            n_eq[(t, s)] = len(eqs)
            V_t[s] = val
        V_next = V_t
    d0 = {k: v[0] for k, v in strategies.items()}
    d1 = {k: v[1] for k, v in strategies.items()}
    profile = Profile([StateDistPolicy(d0), StateDistPolicy(d1)], f"spe-{selection}")
    return TwoAgentSolution(env, selection, profile, V_next[env.initial_state()], n_eq, strategies)


def stage_matrix(params: GameParams, h: int) -> tuple[np.ndarray, list[str]]:
    """The one-shot standoff matrix M_rho(h) of Equation 11 (row player's payoff)."""
    phi = params.phi(h)
    M = np.array([
        [-phi - params.D, -phi, params.R],
        [-phi, -phi, -phi],
        [-params.K, -params.K, -params.K],
    ])
    return M, ["Request", "Hold", "Yield"]


def verify_spe(sol: TwoAgentSolution) -> float:
    """Exploitability of the computed SPE (should be ~0: ground-truth check)."""
    eps, _, _ = exploitability(sol.env, sol.profile)
    return eps


def find_threshold(xs, ys, level: float = 0.5) -> float | None:
    """Linear interpolation of the first crossing of ``level`` (e.g. delta = 1/2 vs R/K)."""
    xs = list(xs)
    ys = list(ys)
    for k in range(len(xs) - 1):
        a, b = ys[k] - level, ys[k + 1] - level
        if a == 0:
            return float(xs[k])
        if a * b < 0:
            return float(xs[k] + (xs[k + 1] - xs[k]) * (-a) / (b - a))
    return None
