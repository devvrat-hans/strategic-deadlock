"""Monte-Carlo rollouts and episode metrics (Definitions 4-5).

Per episode we log every step (actions, cycle set, yields, completions, rewards) and
derive: deadlock / livelock / stall flags, completion rate, latency, social cost (base payoff).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .env import DeadlockEnv
from .policies import JointPolicy


@dataclass
class EpisodeLog:
    steps: int = 0
    cycle_nonempty: list[bool] = field(default_factory=list)
    yields_per_step: list[int] = field(default_factory=list)
    actions: list[tuple[int, ...]] = field(default_factory=list)
    completion_time: dict[int, int] = field(default_factory=dict)
    returns: np.ndarray | None = None        # discounted, rho payoff (agents' objective)
    base_returns: np.ndarray | None = None   # undiscounted, rho = 0 (Equation 1)
    states: list = field(default_factory=list)


def is_deadlock_episode(cycle_nonempty: list[bool], L: int) -> bool:
    run = 0
    for c in cycle_nonempty:
        run = run + 1 if c else 0
        if run >= L:
            return True
    return False


def is_livelock_episode(log: EpisodeLog, n: int, H: int, L: int) -> bool:
    if is_deadlock_episode(log.cycle_nonempty, L):
        return False
    if len(log.completion_time) == n or log.steps < H:
        return False  # everyone finished (or episode ended early)
    y = log.yields_per_step
    if len(y) < L:
        return False
    return all(sum(y[t : t + L]) > 0 for t in range(0, len(y) - L + 1))


def is_stall_episode(log: EpisodeLog, n: int, H: int, L: int) -> bool:
    """Neither deadlock nor livelock, yet the horizon is reached with some agent incomplete
    (e.g. agents that Hold forever without forming a circular wait or yielding)."""
    if len(log.completion_time) == n or log.steps < H:
        return False
    return not is_deadlock_episode(log.cycle_nonempty, L) and not is_livelock_episode(log, n, H, L)


def run_episode(env: DeadlockEnv, policy: JointPolicy, rng: np.random.Generator,
                record_states: bool = False) -> EpisodeLog:
    P = env.params
    s = env.initial_state()
    policy.reset(rng)
    log = EpisodeLog()
    ret = np.zeros(env.n)
    base = np.zeros(env.n)
    disc = 1.0
    for t in range(P.H):
        if env.is_terminal(s):
            break
        if record_states:
            log.states.append(s)
        a = policy.act(env, s, t, rng)
        s, r, info = env.step(s, a, rng)
        ret += disc * np.asarray(r)
        base += np.asarray(info["base_rewards"])
        disc *= P.gamma
        log.cycle_nonempty.append(bool(info["cycle"]))
        log.yields_per_step.append(len(info["yielded"]))
        log.actions.append(info["actions"])
        for i in info["completed"]:
            log.completion_time[i] = t + 1
        log.steps = t + 1
    if record_states:
        log.states.append(s)
    log.returns, log.base_returns = ret, base
    return log


@dataclass
class Summary:
    deadlock: float
    livelock: float
    stall: float
    completion: float
    latency: float
    makespan: float
    social_cost: float
    returns: np.ndarray
    episodes: int

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "returns"}
        for i, r in enumerate(self.returns):
            d[f"return_{i}"] = float(r)
        return d


def evaluate_mc(env: DeadlockEnv, policy: JointPolicy, episodes: int = 500, seed: int = 0) -> Summary:
    """Estimate delta, lambda, completion rate, latency and C(pi) by simulation."""
    rng = np.random.default_rng(seed)
    P = env.params
    dl = ll = st = 0
    comp, lat, mk, cost = [], [], [], []
    rets = np.zeros(env.n)
    for _ in range(episodes):
        log = run_episode(env, policy, rng)
        dl += is_deadlock_episode(log.cycle_nonempty, P.L)
        ll += is_livelock_episode(log, env.n, P.H, P.L)
        st += is_stall_episode(log, env.n, P.H, P.L)
        comp.append(len(log.completion_time) / env.n)
        if log.completion_time:
            lat.append(np.mean(list(log.completion_time.values())))
        mk.append(max(log.completion_time.values()) if len(log.completion_time) == env.n else P.H)
        cost.append(float(np.sum(P.R - log.base_returns)))
        rets += log.returns
    return Summary(
        deadlock=dl / episodes,
        livelock=ll / episodes,
        stall=st / episodes,
        completion=float(np.mean(comp)),
        latency=float(np.mean(lat)) if lat else float("nan"),
        makespan=float(np.mean(mk)),
        social_cost=float(np.mean(cost)),
        returns=rets / episodes,
        episodes=episodes,
    )


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a Student-t confidence interval."""
    x = np.asarray([v for v in values if v == v], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    half = stats.t.ppf(0.5 + confidence / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)


def price_of_anarchy(cost: float, c_star: float) -> float:
    """Empirical PoA-hat = C(pi) / C* (Equation 2)."""
    return float("inf") if c_star <= 1e-12 else cost / c_star


def gap_closed(c0: float, c_rho: float, c_star: float) -> float:
    """g(rho) of Equation 12."""
    den = c0 - c_star
    return float("nan") if abs(den) < 1e-12 else (c0 - c_rho) / den
