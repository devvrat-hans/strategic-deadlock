"""Stage 0 deliverable: hand-checkable traces of the three-agent ring (Figure 1).

    python experiments/stage0_ring_trace.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deadlock_game import ACTION_NAMES, HOLD, REQUEST, YIELD, DeadlockEnv, GameParams, ring  # noqa: E402


def show(env, s, actions, rng):
    s2, r, info = env.step(s, actions, rng)
    print("  actions :", [ACTION_NAMES[a] for a in info["actions"]])
    print("  WFG     :", [(f"{i + 1}->{j + 1}") for i, j in info["edges"]], " cycle C(G_t) =",
          sorted(i + 1 for i in info["cycle"]))
    print("  rewards :", [round(x, 3) for x in r], " completed:", [i + 1 for i in info["completed"]])
    print("  next    :", env.describe(s2))
    return s2


def main():
    params = GameParams(R=8, K=2, D=5, c0=1, rho=0.1, tau_max=3, H=20)
    env = DeadlockEnv(ring(3), params)
    rng = np.random.default_rng(0)
    print("Three-agent ring, o_1=(R1,R2), o_2=(R2,R3), o_3=(R3,R1);", params)
    s = env.initial_state()
    print("\nt=0  everyone requests its first lock (no contention: each acquires it)")
    print("  state   :", env.describe(s))
    s = show(env, s, (REQUEST,) * 3, rng)
    print("\nt=1  everyone requests its second lock -> circular wait 1->2->3->1")
    print("     expected reward per agent: -phi(1) - D = -(1*1.1) - 5 = -6.1")
    s = show(env, s, (REQUEST,) * 3, rng)
    print("\nt=2  agent 1 yields (pays K=2, releases R1, cooldown ~ U{1..3}); agent 3 requests R1")
    s = show(env, s, (YIELD, HOLD, REQUEST), rng)
    print("     (agent 3 acquires R1, completes at T3 and releases R3,R1)")
    print("\nt=3  agent 2 requests the now-free R3 and completes; agent 1 is still cooling down")
    s = show(env, s, (HOLD, REQUEST, HOLD), rng)
    print("\nObservations (full / partial) of agent 1:", env.observe(s, 0, "full"), env.observe(s, 0, "part"))


if __name__ == "__main__":
    main()
