"""Policies: decentralized per-agent policies, profiles, and centralized joint policies.

Every joint policy exposes

    reset(rng)                         start of an episode
    act(env, state, t, rng) -> actions sample a joint action

and Markov policies (the distribution depends only on (state, t)) additionally expose

    dist(env, state, t) -> list[dict[action, prob]]   (independent across agents)

which the exact evaluators in :mod:`deadlock_game.exact` use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np

from .config import ACTIONS, HOLD, REQUEST, YIELD
from .env import DeadlockEnv, State

Dist = dict[int, float]
FORCED: Dist = {HOLD: 1.0}


def _forced(env: DeadlockEnv, state: State, i: int) -> bool:
    return state[3][i] == 1 or state[2][i] > 0


def _sample(dist: Dist, rng: np.random.Generator) -> int:
    if len(dist) == 1:
        return next(iter(dist))
    acts = list(dist.keys())
    pr = np.array([dist[a] for a in acts], dtype=float)
    return int(acts[int(rng.choice(len(acts), p=pr / pr.sum()))])


# agent policies
class AgentPolicy:
    markov = True

    def reset(self, rng: np.random.Generator | None = None) -> None:  # noqa: B027
        pass

    def dist(self, env: DeadlockEnv, state: State, t: int, i: int) -> Dist:
        raise NotImplementedError

    def act(self, env: DeadlockEnv, state: State, t: int, i: int, rng: np.random.Generator) -> int:
        return _sample(self.dist(env, state, t, i), rng)


class AlwaysRequest(AgentPolicy):
    """Greedy: always request the next lock, never yield."""

    def dist(self, env, state, t, i):
        return FORCED if _forced(env, state, i) else {REQUEST: 1.0}


class UniformRandom(AgentPolicy):
    def dist(self, env, state, t, i):
        return FORCED if _forced(env, state, i) else {a: 1 / 3 for a in ACTIONS}


@dataclass
class TabularPolicy(AgentPolicy):
    """Observation-based policy pi_i : O_i -> Delta(A) (Equation 10)."""

    regime: str = "full"
    table: dict[Hashable, np.ndarray] = field(default_factory=dict)
    default: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def probs(self, obs) -> np.ndarray:
        return self.table.get(obs, self.default)

    def dist(self, env, state, t, i):
        if _forced(env, state, i):
            return FORCED
        pr = self.probs(env.observe(state, i, self.regime))
        return {a: float(pr[a]) for a in ACTIONS if pr[a] > 0}

    def greedy_actions(self) -> dict:
        return {o: int(np.argmax(v)) for o, v in self.table.items()}

    def copy(self) -> "TabularPolicy":
        return TabularPolicy(self.regime, {k: v.copy() for k, v in self.table.items()}, self.default.copy())

    @staticmethod
    def deterministic(regime: str, actions: dict, default_action: int = REQUEST) -> "TabularPolicy":
        tab = {o: np.eye(3)[a] for o, a in actions.items()}
        return TabularPolicy(regime, tab, np.eye(3)[default_action])


@dataclass
class StatePolicy(AgentPolicy):
    """Full-state, time-dependent deterministic policy (output of exact best responses)."""

    actions: dict[tuple[int, State], int]
    fallback: AgentPolicy = field(default_factory=AlwaysRequest)

    def dist(self, env, state, t, i):
        if _forced(env, state, i):
            return FORCED
        a = self.actions.get((t, state))
        if a is None:
            return self.fallback.dist(env, state, t, i)
        return {a: 1.0}


@dataclass
class StateDistPolicy(AgentPolicy):
    """Full-state, time-dependent mixed policy (e.g. Stage 1 equilibrium strategies)."""

    dists: dict[tuple[int, State], Dist]
    fallback: AgentPolicy = field(default_factory=AlwaysRequest)

    def dist(self, env, state, t, i):
        if _forced(env, state, i):
            return FORCED
        d = self.dists.get((t, state))
        return d if d is not None else self.fallback.dist(env, state, t, i)


class ExponentialBackoff(AgentPolicy):
    """Randomized exponential back-off [Metcalfe & Boggs].

    Request; after ``w`` consecutive failed requests Yield; after the j-th consecutive
    yield additionally wait Unif{1..2^min(j, j_max)} steps (on top of the env cooldown).
    The counter j resets whenever a lock is acquired.
    """

    markov = False

    def __init__(self, w: int = 2, j_max: int = 4):
        self.w, self.j_max = w, j_max
        self.reset()

    def reset(self, rng=None):
        self.fails = 0
        self.j = 0
        self.wait = 0
        self.last_p = 0
        self.last_a = HOLD
        self.pending_wait = False

    def act(self, env, state, t, i, rng):
        p = state[0][i]
        if self.last_a == REQUEST:
            if p > self.last_p:
                self.fails, self.j = 0, 0
            else:
                self.fails += 1
        self.last_p = p
        if _forced(env, state, i):
            self.last_a = HOLD
            return HOLD
        if self.pending_wait:
            self.wait = int(rng.integers(1, 2 ** min(self.j, self.j_max) + 1))
            self.pending_wait = False
        if self.wait > 0:
            self.wait -= 1
            self.last_a = HOLD
            return HOLD
        if self.fails >= self.w and p > 0:
            self.fails = 0
            self.j += 1
            self.pending_wait = True
            self.last_a = YIELD
            return YIELD
        self.last_a = REQUEST
        return REQUEST

    def dist(self, env, state, t, i):  # pragma: no cover - stateful
        raise TypeError("ExponentialBackoff is not Markov")


# joint policies
class JointPolicy:
    markov = True
    name = "joint"

    def reset(self, rng: np.random.Generator | None = None) -> None:  # noqa: B027
        pass

    def act(self, env: DeadlockEnv, state: State, t: int, rng: np.random.Generator) -> tuple[int, ...]:
        return tuple(_sample(d, rng) for d in self.dist(env, state, t))

    def dist(self, env: DeadlockEnv, state: State, t: int) -> list[Dist]:
        raise NotImplementedError


class Profile(JointPolicy):
    """Decentralized profile pi = (pi_i): each agent acts on its own information."""

    def __init__(self, agents: Sequence[AgentPolicy], name: str = "profile"):
        self.agents = list(agents)
        self.name = name
        self.markov = all(a.markov for a in self.agents)

    def reset(self, rng=None):
        for a in self.agents:
            a.reset(rng)

    def act(self, env, state, t, rng):
        return tuple(a.act(env, state, t, i, rng) for i, a in enumerate(self.agents))

    def dist(self, env, state, t):
        return [a.dist(env, state, t, i) for i, a in enumerate(self.agents)]

    def replace(self, i: int, pol: AgentPolicy) -> "Profile":
        ag = list(self.agents)
        ag[i] = pol
        return Profile(ag, self.name)


class FunctionJointPolicy(JointPolicy):
    """Deterministic centralized policy given by a function (state, t) -> joint action."""

    def __init__(self, fn: Callable[[DeadlockEnv, State, int], tuple[int, ...]], name: str):
        self.fn = fn
        self.name = name

    def dist(self, env, state, t):
        return [{a: 1.0} for a in self.fn(env, state, t)]


def always_request(n: int) -> Profile:
    return Profile([AlwaysRequest() for _ in range(n)], "always-request")


def backoff_profile(n: int, w: int = 2, j_max: int = 4) -> Profile:
    return Profile([ExponentialBackoff(w, j_max) for _ in range(n)], "exp-backoff")
