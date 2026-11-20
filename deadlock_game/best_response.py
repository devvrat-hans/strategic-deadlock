"""Stage 2: n-agent best-response dynamics.

Each round every agent switches to a best response against the others' current
policies. The best response is computed exactly by value iteration (backward induction)
on the single-agent MDP induced by the others' policies, then projected onto the agent's
observation space (Equation 10) so that the resulting profile is decentralized.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import DeadlockEnv
from .exact import best_response, exploitability, project_to_observations
from .policies import Profile, TabularPolicy


def greedy_signature(profile: Profile) -> tuple:
    sig = []
    for a in profile.agents:
        if isinstance(a, TabularPolicy):
            sig.append(tuple(sorted((o, int(np.argmax(v))) for o, v in a.table.items())))
        else:
            sig.append(repr(a))
    return tuple(sig)


def initial_profile(n: int, regime: str) -> Profile:
    """Start from the greedy profile: always Request."""
    return Profile([TabularPolicy(regime) for _ in range(n)], "always-request")


@dataclass
class BRDResult:
    profile: Profile
    rounds: int
    converged: bool
    cycled: bool
    epsilon: float
    history: list[dict] = field(default_factory=list)


def best_response_dynamics(env: DeadlockEnv, regime: str = "full", max_rounds: int = 20,
                           mode: str = "simultaneous", init: Profile | None = None,
                           verbose: bool = False) -> BRDResult:
    """Run BR dynamics until the profile is a fixed point, revisits an earlier profile, or max_rounds."""
    profile = init or initial_profile(env.n, regime)
    seen = {greedy_signature(profile): 0}
    history = []
    converged = cycled = False
    k = 0
    for k in range(1, max_rounds + 1):
        if mode == "simultaneous":
            new = []
            for i in range(env.n):
                br = best_response(env, i, profile)
                new.append(project_to_observations(env, br, profile, regime, profile.agents[i]))
            nxt = Profile(new, f"brd-{regime}")
        elif mode == "sequential":
            nxt = profile
            for i in range(env.n):
                br = best_response(env, i, nxt)
                nxt = nxt.replace(i, project_to_observations(env, br, nxt, regime, nxt.agents[i]))
            nxt.name = f"brd-{regime}"
        else:
            raise ValueError(mode)
        sig = greedy_signature(nxt)
        history.append({"round": k})
        if verbose:
            print(f"[BRD] round {k}")
        if sig == greedy_signature(profile):
            profile = nxt
            converged = True
            break
        if sig in seen:
            profile = nxt
            cycled = True
            break
        seen[sig] = k
        profile = nxt
    eps, _, _ = exploitability(env, profile)
    return BRDResult(profile, k, converged, cycled, eps, history)
