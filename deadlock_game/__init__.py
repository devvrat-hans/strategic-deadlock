"""Strategic Resource Contention and Deadlock in Multi-Agent Systems (CS329 project).

Package layout (one module per pipeline stage of the proposal):

    config        parameters, topologies (ring, dense, standoff), default grid
    env           Stage 0: simulator, Equations (5)-(9), step() interface
    wfg           Wait-For Graph cycle detection (Definition 3)
    policies      decentralized / centralized policy classes
    metrics       Monte-Carlo rollouts, deadlock/livelock episodes, PoA, g(rho), CIs
    exact         exact evaluation, best responses, exploitability, centralized optimum C*
    baselines     Banker's Algorithm, resource ordering, exponential back-off
    two_agent     Stage 1: exact two-agent equilibria with nashpy
    best_response Stage 2: n-agent best-response dynamics
    learning      Stage 3: independent / shared Q-learning, fictitious play
    experiments   Stages 2-4 grid runner and aggregation
"""
from .config import (ACTION_NAMES, HOLD, REQUEST, YIELD, GameParams, Topology, dense,
                     make_topology, ordered, params_from_ratio, ring, standoff)
from .env import DeadlockEnv

__all__ = [
    "ACTION_NAMES", "HOLD", "REQUEST", "YIELD", "GameParams", "Topology", "DeadlockEnv",
    "dense", "make_topology", "ordered", "params_from_ratio", "ring", "standoff",
]
__version__ = "0.1.0"
