"""Game parameters and contention topologies (Table 1 of the proposal)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

# Action encoding (A = {Request, Hold, Yield})
REQUEST, HOLD, YIELD = 0, 1, 2
ACTIONS = (REQUEST, HOLD, YIELD)
ACTION_NAMES = {REQUEST: "Request", HOLD: "Hold", YIELD: "Yield"}


@dataclass(frozen=True)
class Topology:
    """Who needs which resources, and in what order (R_i, o_i)."""

    name: str
    n_resources: int
    orders: tuple[tuple[int, ...], ...]  # orders[i] = o_i (resource indices)

    @property
    def n_agents(self) -> int:
        return len(self.orders)

    def needs(self, i: int) -> frozenset[int]:
        return frozenset(self.orders[i])

    def __post_init__(self) -> None:
        for o in self.orders:
            if len(set(o)) != len(o) or not o:
                raise ValueError(f"order {o} must be non-empty without repeats")
            if any(r < 0 or r >= self.n_resources for r in o):
                raise ValueError(f"order {o} references unknown resource")


def ring(n: int = 3) -> Topology:
    """R_i = {R_i, R_{i+1 mod n}}, acquired in that order (Figure 1)."""
    return Topology("ring", n, tuple((i, (i + 1) % n) for i in range(n)))


def _pairwise_deadlock_possible(orders) -> bool:
    """True if two agents can each hold a prefix containing the other's next resource."""
    for a in range(len(orders)):
        for b in range(a + 1, len(orders)):
            oa, ob = orders[a], orders[b]
            for pa in range(1, len(oa)):
                for pb in range(1, len(ob)):
                    Pa, Pb = set(oa[:pa]), set(ob[:pb])
                    if not Pa & Pb and oa[pa] in Pb and ob[pb] in Pa:
                        return True
    return False


def dense(n: int = 3, m: int = 3, seed: int = 0) -> Topology:
    """Every agent needs all m resources, each in its own random order.

    Orders are redrawn (deterministically from ``seed``) until a circular wait is
    reachable, so that the topology actually has contention worth studying.
    """
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        orders = tuple(tuple(int(r) for r in rng.permutation(m)) for _ in range(n))
        if _pairwise_deadlock_possible(orders):
            break
    return Topology(f"dense(seed={seed})", m, orders)


def standoff() -> Topology:
    """Two agents, two resources, opposite orders (Equation 11)."""
    return Topology("standoff", 2, ((0, 1), (1, 0)))


def ordered(top: Topology) -> Topology:
    """Resource-ordering baseline [Havender]: everyone acquires in one global order."""
    return Topology(top.name + "+ordered", top.n_resources, tuple(tuple(sorted(o)) for o in top.orders))


def custom(orders: Sequence[Sequence[int]], n_resources: int | None = None, name: str = "custom") -> Topology:
    orders_t = tuple(tuple(int(r) for r in o) for o in orders)
    m = n_resources if n_resources is not None else 1 + max(max(o) for o in orders_t)
    return Topology(name, m, orders_t)


def make_topology(name: str, n: int = 3, m: int = 3, seed: int = 0) -> Topology:
    if name == "ring":
        return ring(n)
    if name == "dense":
        return dense(n, m, seed)
    if name == "standoff":
        return standoff()
    raise ValueError(f"unknown topology {name!r}")


@dataclass(frozen=True)
class GameParams:
    """Payoff and dynamics parameters (R, K, D, c0, rho, tau_max, H, L, gamma, h_bar)."""

    R: float = 8.0          # completion reward
    K: float = 2.0          # abort (yield) cost
    D: float = 5.0          # per-step circular-wait (deadlock) penalty
    c0: float = 1.0         # base holding cost
    rho: float = 0.0        # escalation rate of the holding cost (0 = no mechanism)
    tau_max: int = 3        # cooldown after Yield ~ Unif{1..tau_max}
    H: int = 20             # episode horizon
    L: int = 4              # window for deadlock / livelock episodes
    gamma: float = 0.95     # discount used by agents' objectives / learning
    h_bar: int = 4          # cap of h in observations

    def phi(self, h: int) -> float:
        """Escalating per-step holding cost phi_rho(h) = c0 (1 + rho)^h."""
        return self.c0 * (1.0 + self.rho) ** h

    def with_(self, **kw) -> "GameParams":
        return replace(self, **kw)

    @property
    def base(self) -> "GameParams":
        """Same game with rho = 0 (used for social cost, Equation 1)."""
        return replace(self, rho=0.0)


def params_from_ratio(ratio_RK: float, D_over_c0: float, rho: float = 0.0, K: float = 4.0,
                      c0: float = 1.0, **kw) -> GameParams:
    """Grid helper: fix K and c0, set R = ratio * K and D = (D/c0) * c0."""
    return GameParams(R=ratio_RK * K, K=K, D=D_over_c0 * c0, c0=c0, rho=rho, **kw)


# Default experiment grid (Section 4: "Baselines, metrics and grid")
GRID_RK = (0.5, 1.0, 2.0, 4.0, 8.0)
GRID_D = (0.0,) + tuple(float(x) for x in np.round(np.logspace(-1, 2, 5), 3))  # 0, 0.1, .., 100
GRID_RHO = (0.0, 0.05, 0.1, 0.2, 0.5)
GRID_INFO = ("full", "part")
GRID_TOPOLOGY = ("ring", "dense")
