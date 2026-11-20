"""Stage 0: the decentralized lock-acquisition simulator (Equations 5-9).

State representation
--------------------
Holdings are always a prefix of o_i, so the allocation sigma is fully described by
``p[i]`` = number of resources agent i currently holds (sigma^{-1}(i) = o_i[:p[i]]).
A state is the hashable tuple

    (p, h, kappa, d, cyc)

each a tuple of length n:
    p[i]     resources held (prefix length of o_i)
    h[i]     consecutive holding time (capped at ``h_cap``; exact when rho > 0)
    kappa[i] remaining cooldown
    d[i]     completion flag
    cyc[i]   1 iff i was on a directed cycle of the previous Wait-For Graph
             (needed for the full-information observation, Equation 10)

The frozen interface is ``step(state, actions, rng) -> (next_state, rewards, info)``.
``transitions(state, actions)`` enumerates the *exact* distribution over next states
(tie-breaking and cooldown draws), which is used by all exact solvers.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple, Sequence

import numpy as np

from .config import HOLD, REQUEST, YIELD, GameParams, Topology
from .wfg import cycle_members

State = tuple  # (p, h, kappa, d, cyc)


class Outcome(NamedTuple):
    """One deterministic resolution of a step."""

    prob: float
    next_state: State
    rewards: tuple[float, ...]       # u^rho_i (Equation 9)
    base_rewards: tuple[float, ...]  # u^0_i (rho = 0), for the social cost
    cycle: frozenset[int]            # C(G_t)
    edges: tuple[tuple[int, int], ...]
    completed: tuple[int, ...]       # agents completing at this step
    yielded: tuple[int, ...]


@dataclass
class DeadlockEnv:
    topology: Topology
    params: GameParams
    track_obs: bool = True  # False drops h (if rho == 0) and cyc from the state (centralized DP)
    _cache_size: int = field(default=500_000, repr=False)

    def __post_init__(self) -> None:
        self.n = self.topology.n_agents
        self.m = self.topology.n_resources
        self.orders = self.topology.orders
        self.sizes = tuple(len(o) for o in self.orders)
        if self.params.rho > 0:
            self.h_cap = self.params.H
        else:
            self.h_cap = self.params.h_bar if self.track_obs else 0
        self.transitions = lru_cache(maxsize=self._cache_size)(self._transitions)

    # basics
    def initial_state(self) -> State:
        z = (0,) * self.n
        return (z, z, z, z, z)

    def owner(self, p: Sequence[int]) -> list[int | None]:
        own: list[int | None] = [None] * self.m
        for i, k in enumerate(p):
            for r in self.orders[i][:k]:
                own[r] = i
        return own

    def next_resource(self, i: int, p: Sequence[int]) -> int | None:
        return self.orders[i][p[i]] if p[i] < self.sizes[i] else None

    def missing(self, state: State, i: int) -> int:
        """k_i = |R_i \\ sigma^{-1}(i)|."""
        return self.sizes[i] - state[0][i]

    def available_actions(self, state: State, i: int) -> tuple[int, ...]:
        """Equation 4."""
        _, _, kappa, d, _ = state
        if d[i] or kappa[i] > 0:
            return (HOLD,)
        return (REQUEST, HOLD, YIELD)

    def is_terminal(self, state: State) -> bool:
        return all(state[3])

    def held_sets(self, state: State) -> list[tuple[int, ...]]:
        return [self.orders[i][: state[0][i]] for i in range(self.n)]

    # observations
    def observe(self, state: State, i: int, regime: str = "full") -> tuple[int, int, int, int]:
        """Compressed local observation o_i (Equation 10)."""
        p, h, kappa, d, cyc = state
        k = self.sizes[i] - p[i]
        hh = min(h[i], self.params.h_bar)
        cool = int(kappa[i] > 0)
        if regime == "full":
            flag = int(cyc[i])
        elif regime == "part":
            r = self.next_resource(i, p)
            own = self.owner(p)
            flag = int(r is not None and own[r] is not None and own[r] != i)
        else:
            raise ValueError(regime)
        return (k, hh, cool, flag)

    # dynamics
    def _choices(self, state: State, actions: tuple[int, ...]):
        """Random choices of a step: contested resources (with their requester sets) and yielders."""
        p, _, kappa, d, _ = state
        acts = self._sanitize(state, actions)
        p_after = list(p)
        yielders = [i for i in range(self.n) if acts[i] == YIELD]
        for i in yielders:
            p_after[i] = 0
        own = self.owner(p_after)
        queues: dict[int, list[int]] = {}
        for i in range(self.n):
            if acts[i] == REQUEST and not d[i]:
                r = self.next_resource(i, p_after)
                if r is not None and own[r] is None:
                    queues.setdefault(r, []).append(i)
        return acts, yielders, queues

    def _sanitize(self, state: State, actions: Sequence[int]) -> tuple[int, ...]:
        """Illegal actions (Request/Yield while done or cooling down) are coerced to Hold."""
        _, _, kappa, d, _ = state
        return tuple(HOLD if (d[i] or kappa[i] > 0) else int(a) for i, a in enumerate(actions))

    def _resolve(self, state: State, acts: tuple[int, ...], yielders: list[int],
                 winners: dict[int, int], cooldowns: dict[int, int], prob: float) -> Outcome:
        P = self.params
        p0, h0, k0, d0, _ = state
        p, h, kap, d = list(p0), list(h0), list(k0), list(d0)
        # (T1) release
        for i in yielders:
            p[i] = 0
            h[i] = 0
            kap[i] = cooldowns[i]
        # (T2) acquire (winners pre-drawn uniformly from each queue Q_r)
        acquired = set()
        for _, w in winners.items():
            p[w] += 1
            acquired.add(w)
        # (T3) complete
        completed = []
        for i in range(self.n):
            if not d[i] and p[i] == self.sizes[i]:
                d[i] = 1
                p[i] = 0
                completed.append(i)
        # Wait-For Graph G_t on sigma_{t+1}: pending, unsatisfied requests
        own = self.owner(p)
        edges = []
        for i in range(self.n):
            if not d[i] and acts[i] == REQUEST and i not in acquired:
                r = self.next_resource(i, p)
                if r is not None and own[r] is not None and own[r] != i:
                    edges.append((i, own[r]))
        cyc_set = cycle_members(self.n, edges)
        # payoffs (Equation 9), computed with h_{i,t} (before the update)
        rew, base = [], []
        for i in range(self.n):
            r_i = 0.0
            b_i = 0.0
            if i in completed:
                r_i += P.R
                b_i += P.R
            if not d[i] and p[i] > 0:
                r_i -= P.phi(h0[i])
                b_i -= P.c0
            if acts[i] == YIELD:
                r_i -= P.K
                b_i -= P.K
            if i in cyc_set:
                r_i -= P.D
                b_i -= P.D
            rew.append(r_i)
            base.append(b_i)
        # (T4) counters
        for i in range(self.n):
            h[i] = min(h0[i] + 1, self.h_cap) if (not d[i] and p[i] > 0) else 0
            if acts[i] != YIELD:
                kap[i] = max(k0[i] - 1, 0)
            if d[i]:
                kap[i] = 0
        if self.track_obs:
            cyc = tuple(int(i in cyc_set) for i in range(self.n))
        else:
            cyc = (0,) * self.n
        nxt = (tuple(p), tuple(h), tuple(kap), tuple(d), cyc)
        return Outcome(prob, nxt, tuple(rew), tuple(base), cyc_set, tuple(edges),
                       tuple(completed), tuple(yielders))

    def _transitions(self, state: State, actions: tuple[int, ...]) -> tuple[Outcome, ...]:
        """Exact distribution over outcomes of ``step``."""
        acts, yielders, queues = self._choices(state, tuple(actions))
        res = list(queues.keys())
        tie_opts = [queues[r] for r in res]
        cd_opts = [range(1, self.params.tau_max + 1)] * len(yielders)
        n_combo = 1
        for q in tie_opts:
            n_combo *= len(q)
        n_combo *= self.params.tau_max ** len(yielders)
        prob = 1.0 / n_combo
        out = []
        for ties in itertools.product(*tie_opts):
            winners = dict(zip(res, ties))
            for cds in itertools.product(*cd_opts):
                out.append(self._resolve(state, acts, yielders, winners, dict(zip(yielders, cds)), prob))
        return tuple(out)

    def step(self, state: State, actions: Sequence[int], rng: np.random.Generator):
        """Sampled step: returns (next_state, rewards, info)."""
        acts, yielders, queues = self._choices(state, tuple(int(a) for a in actions))
        winners = {r: q[int(rng.integers(len(q)))] for r, q in queues.items()}
        cds = {i: int(rng.integers(1, self.params.tau_max + 1)) for i in yielders}
        o = self._resolve(state, acts, yielders, winners, cds, 1.0)
        info = {
            "base_rewards": o.base_rewards,
            "cycle": o.cycle,
            "edges": o.edges,
            "completed": o.completed,
            "yielded": o.yielded,
            "actions": acts,
            "done": self.is_terminal(o.next_state),
        }
        return o.next_state, o.rewards, info

    def describe(self, state: State) -> str:
        p, h, kappa, d, cyc = state
        parts = []
        for i in range(self.n):
            held = ",".join(f"R{r + 1}" for r in self.orders[i][: p[i]]) or "-"
            parts.append(f"A{i + 1}[held={held} h={h[i]} k={kappa[i]} done={d[i]} cyc={cyc[i]}]")
        return " ".join(parts)
