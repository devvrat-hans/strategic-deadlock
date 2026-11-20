"""Classical baselines: Banker's Algorithm, resource ordering, exponential back-off."""
from __future__ import annotations

from .config import HOLD, REQUEST, ordered
from .env import DeadlockEnv, State
from .policies import FunctionJointPolicy, Profile, always_request, backoff_profile


def is_safe(env: DeadlockEnv, p: list[int], d: list[int]) -> bool:
    """Banker's safety check with single-unit resources and maximum claims R_i."""
    own = env.owner(p)
    free = {r for r in range(env.m) if own[r] is None}
    pending = [i for i in range(env.n) if not d[i]]
    progress = True
    while pending and progress:
        progress = False
        for i in list(pending):
            need = set(env.orders[i][p[i]:])
            if need <= free:
                free |= set(env.orders[i][: p[i]])  # i finishes and releases everything
                free |= need
                pending.remove(i)
                progress = True
    return not pending


def bankers_action(env: DeadlockEnv, state: State, t: int) -> tuple[int, ...]:
    """Centralized Banker: grant a request only if the resulting state is safe."""
    p0, _, kappa, d, _ = state
    p = list(p0)
    own = env.owner(p)  # ownership at the start of the step (before this step's grants)
    granted: set[int] = set()
    acts = []
    for i in range(env.n):
        if d[i] or kappa[i] > 0:
            acts.append(HOLD)
            continue
        r = env.next_resource(i, p)
        if r is None:
            acts.append(HOLD)
            continue
        if own[r] is not None:  # held by someone: pending request (env will not grant it)
            acts.append(REQUEST)
            continue
        if r in granted:  # already granted to another agent this step: wait
            acts.append(HOLD)
            continue
        p_try = list(p)
        p_try[i] += 1
        d_try = list(d)
        if p_try[i] == env.sizes[i]:
            d_try[i] = 1
            p_try[i] = 0
        if is_safe(env, p_try, d_try):
            acts.append(REQUEST)
            # tentatively granted (one requester per resource). A completing agent only
            # releases at (T3), after this step's acquisitions, so it keeps its locks here.
            p[i] += 1
            granted.add(r)
        else:
            acts.append(HOLD)
    return tuple(acts)


def bankers_policy() -> FunctionJointPolicy:
    return FunctionJointPolicy(bankers_action, "bankers")


def resource_ordering(env: DeadlockEnv) -> tuple[DeadlockEnv, Profile]:
    """Everyone acquires in one global order; agents always request."""
    oenv = DeadlockEnv(ordered(env.topology), env.params, env.track_obs)
    return oenv, always_request(env.n)


def all_baselines(env: DeadlockEnv, w: int = 2, j_max: int = 4):
    """List of (name, env, policy) for the classical baselines."""
    oenv, opol = resource_ordering(env)
    return [
        ("always-request", env, always_request(env.n)),
        ("bankers", env, bankers_policy()),
        ("resource-ordering", oenv, opol),
        ("exp-backoff", env, backoff_profile(env.n, w, j_max)),
    ]
