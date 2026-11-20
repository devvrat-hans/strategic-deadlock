"""Stage 3: learned policies over the compressed observations (Equation 10).

(a) independent tabular Q-learning [Watkins & Dayan]  - one table per agent
(b) fictitious play [Brown]                           - best response to the others' average policy
(c) parameter-shared Q-learning                       - one table trained on all agents' experience,
                                                       decaying learning rate

Training stops after ``max_episodes`` (5000) or when the greedy policy is unchanged over a
``window`` (200) episode window. Every learned profile can be checked with
:func:`deadlock_game.exact.exploitability`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import ACTIONS, HOLD
from .env import DeadlockEnv
from .exact import best_response, exploitability, project_to_observations
from .policies import Profile, TabularPolicy


@dataclass
class QConfig:
    max_episodes: int = 5000
    window: int = 200
    min_episodes: int = 500
    alpha: float = 0.1            # (a) constant learning rate
    alpha_decay: float = 1000.0   # (c) alpha_e = alpha / (1 + e / alpha_decay)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000
    q_init: float = 0.0


@dataclass
class LearnResult:
    profile: Profile
    episodes: int
    converged: bool
    curve: list[float] = field(default_factory=list)  # mean social cost of training episodes (per 100)
    epsilon: float | None = None


def _epsilon(cfg: QConfig, e: int) -> float:
    frac = min(1.0, e / max(1, cfg.eps_decay_episodes))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _greedy(tables: list[dict]) -> tuple:
    return tuple(tuple(sorted((o, int(np.argmax(q))) for o, q in tab.items())) for tab in tables)


def q_learning(env: DeadlockEnv, regime: str = "full", shared: bool = False, seed: int = 0,
               cfg: QConfig | None = None) -> LearnResult:
    """Independent (shared=False) or parameter-shared (shared=True) tabular Q-learning."""
    cfg = cfg or QConfig()
    P = env.params
    rng = np.random.default_rng(seed)
    n_tables = 1 if shared else env.n
    tables: list[dict] = [defaultdict(lambda: np.full(3, cfg.q_init)) for _ in range(n_tables)]
    tab = (lambda i: tables[0]) if shared else (lambda i: tables[i])
    last_sig, stable_since = None, 0
    curve, costs = [], []
    converged = False
    e = 0
    for e in range(1, cfg.max_episodes + 1):
        eps = _epsilon(cfg, e)
        alpha = cfg.alpha / (1 + e / cfg.alpha_decay) if shared else cfg.alpha
        s = env.initial_state()
        base = np.zeros(env.n)
        for t in range(P.H):
            if env.is_terminal(s):
                break
            obs = [env.observe(s, i, regime) for i in range(env.n)]
            acts = []
            for i in range(env.n):
                av = env.available_actions(s, i)
                if len(av) == 1:
                    acts.append(av[0])
                elif rng.random() < eps:
                    acts.append(int(rng.choice(av)))
                else:
                    q = tab(i)[obs[i]]
                    best = np.flatnonzero(q == q.max())
                    acts.append(int(rng.choice(best)))
            s2, r, info = env.step(s, acts, rng)
            base += np.asarray(info["base_rewards"])
            last = t == P.H - 1
            for i in range(env.n):
                if s[3][i]:  # already done: absorbing, no learning
                    continue
                target = r[i]
                if not s2[3][i] and not last:
                    o2 = env.observe(s2, i, regime)
                    av2 = env.available_actions(s2, i)
                    q2 = tab(i)[o2]
                    target += P.gamma * max(q2[a] for a in av2)
                q = tab(i)[obs[i]]
                q[acts[i]] += alpha * (target - q[acts[i]])
            s = s2
        costs.append(float(np.sum(P.R - base)))
        if e % 100 == 0:
            curve.append(float(np.mean(costs[-100:])))
        sig = _greedy(tables)
        if sig != last_sig:
            last_sig, stable_since = sig, e
        elif e >= cfg.min_episodes and e - stable_since >= cfg.window:
            converged = True
            break
    agents = []
    for i in range(env.n):
        t_i = tab(i)
        agents.append(TabularPolicy.deterministic(regime, {o: int(np.argmax(q)) for o, q in t_i.items()}))
    name = "shared-q" if shared else "independent-q"
    return LearnResult(Profile(agents, f"{name}-{regime}"), e, converged, curve)


def fictitious_play(env: DeadlockEnv, regime: str = "full", iterations: int = 30,
                    tol: float = 1e-3) -> LearnResult:
    """Fictitious play in policy space.

    Each iteration every agent computes an exact best response to the others' *average*
    policies (projected onto its observations), and the averages are updated with step 1/(k+1).
    """
    avg = [TabularPolicy(regime) for _ in range(env.n)]
    profile = Profile(avg, f"fictitious-play-{regime}")
    curve = []
    converged = False
    k = 0
    for k in range(1, iterations + 1):
        brs = []
        for i in range(env.n):
            br = best_response(env, i, profile)
            brs.append(project_to_observations(env, br, profile, regime, None))
        delta = 0.0
        new_avg = []
        w = 1.0 / (k + 1)
        for i in range(env.n):
            old = profile.agents[i]
            keys = set(old.table) | set(brs[i].table)
            tab = {}
            for o in keys:
                a = old.probs(o)
                b = brs[i].probs(o)
                tab[o] = (1 - w) * a + w * b
                delta = max(delta, float(np.abs(tab[o] - a).max()))
            new_avg.append(TabularPolicy(regime, tab, old.default.copy()))
        profile = Profile(new_avg, f"fictitious-play-{regime}")
        curve.append(delta)
        if delta < tol:
            converged = True
            break
    return LearnResult(profile, k, converged, curve)


def train(method: str, env: DeadlockEnv, regime: str = "full", seed: int = 0,
          cfg: QConfig | None = None, fp_iterations: int = 30, check_eps: bool = True) -> LearnResult:
    if method in ("iql", "independent-q"):
        res = q_learning(env, regime, shared=False, seed=seed, cfg=cfg)
    elif method in ("sql", "shared-q"):
        res = q_learning(env, regime, shared=True, seed=seed, cfg=cfg)
    elif method in ("fp", "fictitious-play"):
        res = fictitious_play(env, regime, fp_iterations)
    else:
        raise ValueError(method)
    if check_eps:
        res.epsilon = exploitability(env, res.profile)[0]
    return res


def greedy_actions_table(profile: Profile) -> dict:
    """Human-readable table: agent -> {obs: action}."""
    from .config import ACTION_NAMES

    out = {}
    for i, a in enumerate(profile.agents):
        if isinstance(a, TabularPolicy):
            out[i] = {o: ACTION_NAMES[int(np.argmax(v))] for o, v in sorted(a.table.items())}
    return out


__all__ = ["QConfig", "LearnResult", "q_learning", "fictitious_play", "train", "greedy_actions_table",
           "ACTIONS", "HOLD"]
