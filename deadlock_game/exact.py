"""Exact solvers over the joint state space (finite horizon, backward induction).

* :func:`evaluate`              exact expected returns U_i(pi) and social cost C(pi)
* :func:`best_response`         exact best response of agent i to pi_{-i}
                                (full-state, time-dependent deviator)
* :func:`exploitability`        epsilon(pi) = max_i [ max U_i(pi'_i, pi_-i) - U_i(pi) ]
* :func:`centralized_optimum`   C* by dynamic programming (denominator of the PoA)
* :func:`project_to_observations`  turn a full-state best response into an
                                observation-based policy (used by BR dynamics / FP)

Because the deviator in :func:`best_response` sees the full state and the clock, the
reported exploitability is an *upper bound* on the exploitability within the class of
observation-based policies; this is the conservative direction for "equilibrium" claims.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import ACTIONS, HOLD
from .env import DeadlockEnv, State
from .policies import FunctionJointPolicy, JointPolicy, StatePolicy, TabularPolicy

NEG = -1e18


def _joint(dists):
    """Enumerate joint actions with probabilities from independent per-agent dists."""
    items = [list(d.items()) for d in dists]
    for combo in itertools.product(*items):
        a = tuple(x[0] for x in combo)
        pr = 1.0
        for x in combo:
            pr *= x[1]
        if pr > 0:
            yield a, pr


# evaluation
@dataclass
class ExactValue:
    returns: np.ndarray        # discounted (gamma) returns with the rho payoff
    base_returns: np.ndarray   # undiscounted base (rho = 0) returns
    social_cost: float         # sum_i (R - U^0_i)
    p_cycle: np.ndarray        # Pr[C(G_t) != empty] for each t
    p_all_done: float          # Pr[all agents complete by H]
    occupancy: list[dict]      # distribution over states at each t


def evaluate(env: DeadlockEnv, policy: JointPolicy, gamma: float | None = None) -> ExactValue:
    if not policy.markov:
        raise TypeError(f"{policy.name} is not Markov; use metrics.evaluate_mc")
    P = env.params
    g = P.gamma if gamma is None else gamma
    dist: dict[State, float] = {env.initial_state(): 1.0}
    ret = np.zeros(env.n)
    base = np.zeros(env.n)
    p_cyc = np.zeros(P.H)
    occ = []
    done_mass = 0.0
    for t in range(P.H):
        occ.append(dist)
        new: dict[State, float] = defaultdict(float)
        for s, ps in dist.items():
            if env.is_terminal(s):
                done_mass += ps
                continue
            for a, pa in _joint(policy.dist(env, s, t)):
                for o in env.transitions(s, a):
                    w = ps * pa * o.prob
                    ret += w * (g ** t) * np.asarray(o.rewards)
                    base += w * np.asarray(o.base_rewards)
                    if o.cycle:
                        p_cyc[t] += w
                    new[o.next_state] += w
        dist = new
    occ.append(dist)
    done_mass += sum(ps for s, ps in dist.items() if env.is_terminal(s))
    cost = float(np.sum(P.R - base))
    return ExactValue(ret, base, cost, p_cyc, done_mass, occ)


# best response
@dataclass
class BestResponse:
    agent: int
    value: float                               # V_0(s0)
    policy: StatePolicy                        # full-state, time-dependent
    Q: list[dict[State, np.ndarray]]           # Q_t(s, a_i)


def _reachable(env: DeadlockEnv, i: int | None, policy: JointPolicy | None):
    """Per-time reachable state sets when agent i (or everybody if i is None) may do anything."""
    H = env.params.H
    layers = [{env.initial_state()}]
    for t in range(H):
        nxt = set()
        for s in layers[t]:
            if env.is_terminal(s):
                continue
            for a in _actions_for(env, s, t, i, policy):
                for o in env.transitions(s, a[0]):
                    nxt.add(o.next_state)
        layers.append(nxt)
    return layers


def _actions_for(env, s, t, i, policy):
    """Joint actions (with probability, ignoring agent i's own probability) to expand."""
    if i is None:
        avail = [env.available_actions(s, j) for j in range(env.n)]
        return [(a, 1.0) for a in itertools.product(*avail)]
    dists = policy.dist(env, s, t)
    dists[i] = {a: 1.0 for a in env.available_actions(s, i)}
    return list(_joint(dists))


def best_response(env: DeadlockEnv, i: int, policy: JointPolicy, gamma: float | None = None) -> BestResponse:
    """Exact best response of agent i (maximizes its discounted rho-payoff return)."""
    P = env.params
    g = P.gamma if gamma is None else gamma
    layers = _reachable(env, i, policy)
    V_next: dict[State, float] = {}
    Qs: list[dict[State, np.ndarray]] = [None] * P.H  # type: ignore[list-item]
    acts: dict[tuple[int, State], int] = {}
    for t in range(P.H - 1, -1, -1):
        V_t: dict[State, float] = {}
        Q_t: dict[State, np.ndarray] = {}
        for s in layers[t]:
            if env.is_terminal(s):
                V_t[s] = 0.0
                continue
            others = policy.dist(env, s, t)
            q = np.full(3, NEG)
            for ai in env.available_actions(s, i):
                others[i] = {ai: 1.0}
                val = 0.0
                for a, pa in _joint(others):
                    for o in env.transitions(s, a):
                        val += pa * o.prob * (o.rewards[i] + g * V_next.get(o.next_state, 0.0))
                q[ai] = val
            best = int(np.argmax(q))
            Q_t[s] = q
            V_t[s] = float(q[best])
            acts[(t, s)] = best
        Qs[t] = Q_t
        V_next = V_t
    v0 = V_next.get(env.initial_state(), 0.0)
    return BestResponse(i, v0, StatePolicy(acts), Qs)


def exploitability(env: DeadlockEnv, policy: JointPolicy, gamma: float | None = None):
    """Return (epsilon, per-agent gains, per-agent values U_i(pi))."""
    ev = evaluate(env, policy, gamma)
    gains = []
    for i in range(env.n):
        br = best_response(env, i, policy, gamma)
        gains.append(max(0.0, br.value - float(ev.returns[i])))
    return max(gains), np.array(gains), ev.returns


def project_to_observations(env: DeadlockEnv, br: BestResponse, policy: JointPolicy, regime: str,
                            previous: TabularPolicy | None = None) -> TabularPolicy:
    """Observation-based policy: for each o, the action maximizing the occupancy-weighted Q.

    Occupancy is that of the profile in which agent i plays its full-state best response.
    """
    i = br.agent
    from .policies import Profile  # local import to avoid a cycle

    if isinstance(policy, Profile):
        dev = policy.replace(i, br.policy)
    else:  # pragma: no cover
        raise TypeError("projection requires a decentralized Profile")
    ev = evaluate(env, dev)
    agg: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
    mass: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
    for t in range(env.params.H):
        for s, w in ev.occupancy[t].items():
            q = br.Q[t].get(s)
            if q is None or w <= 0 or env.available_actions(s, i) == (HOLD,):
                continue
            o = env.observe(s, i, regime)
            for a in ACTIONS:
                if q[a] > NEG / 2:
                    agg[o][a] += w * q[a]
                    mass[o][a] += w
    table = {} if previous is None else {k: v.copy() for k, v in previous.table.items()}
    for o in agg:
        valid = mass[o] > 0
        avg = np.where(valid, agg[o] / np.maximum(mass[o], 1e-300), NEG)
        table[o] = np.eye(3)[int(np.argmax(avg))]
    default = previous.default if previous is not None else np.eye(3)[0]
    return TabularPolicy(regime, table, default)


# centralized optimum
@dataclass
class CentralizedResult:
    c_star: float
    welfare: float
    policy: FunctionJointPolicy


def _strip(state: State) -> State:
    p, h, kappa, d, cyc = state
    z = (0,) * len(p)
    return (p, z, kappa, d, z)


def centralized_optimum(env: DeadlockEnv) -> CentralizedResult:
    """C* = min over centralized joint policies of the social cost (base payoff, Eq. 1)."""
    P = env.params.base
    denv = DeadlockEnv(env.topology, P, track_obs=False)
    layers = _reachable(denv, None, None)
    V_next: dict[State, float] = {}
    acts: dict[tuple[int, State], tuple[int, ...]] = {}
    for t in range(P.H - 1, -1, -1):
        V_t = {}
        for s in layers[t]:
            if denv.is_terminal(s):
                V_t[s] = 0.0
                continue
            best, best_a = NEG, None
            for a, _ in _actions_for(denv, s, t, None, None):
                val = 0.0
                for o in denv.transitions(s, a):
                    val += o.prob * (sum(o.base_rewards) + V_next.get(o.next_state, 0.0))
                if val > best + 1e-12:
                    best, best_a = val, a
            V_t[s] = best
            acts[(t, s)] = best_a
        V_next = V_t
    welfare = V_next[denv.initial_state()]
    c_star = env.n * P.R - welfare

    def fn(e, s, t):
        a = acts.get((t, _strip(s)))
        return a if a is not None else tuple(HOLD for _ in range(e.n))

    return CentralizedResult(c_star, welfare, FunctionJointPolicy(fn, "centralized-optimum"))
