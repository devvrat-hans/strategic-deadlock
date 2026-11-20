"""Exact solvers, baselines, Stage 1-3 sanity checks."""
import numpy as np
import pytest

from deadlock_game import DeadlockEnv, GameParams, dense, ring, standoff
from deadlock_game.baselines import all_baselines, bankers_policy, is_safe
from deadlock_game.best_response import best_response_dynamics
from deadlock_game.exact import best_response, centralized_optimum, evaluate, exploitability
from deadlock_game.experiments import Cell, aggregate, run_cell
from deadlock_game.learning import QConfig, train
from deadlock_game.metrics import EpisodeLog, evaluate_mc, gap_closed, is_deadlock_episode, is_livelock_episode, is_stall_episode
from deadlock_game.policies import always_request
from deadlock_game.two_agent import find_threshold, solve_two_agent, stage_equilibria

P = GameParams(R=16, K=4, D=5, c0=1, rho=0.0, tau_max=2, H=12, L=3)


def test_deadlock_livelock_definitions():
    assert is_deadlock_episode([0, 1, 1, 1, 0], 3)
    assert not is_deadlock_episode([1, 1, 0, 1, 1], 3)
    log = EpisodeLog(steps=6, cycle_nonempty=[0] * 6, yields_per_step=[1, 0, 1, 0, 0, 1])
    assert is_livelock_episode(log, 2, 6, 3)
    log.yields_per_step = [1, 0, 0, 0, 0, 1]
    assert not is_livelock_episode(log, 2, 6, 3)
    assert is_stall_episode(log, 2, 6, 3)          # incomplete, no cycle, not a livelock
    log.completion_time = {0: 3, 1: 5}
    assert not is_stall_episode(log, 2, 6, 3)      # everyone finished


def test_all_hold_is_a_stall():
    from deadlock_game.policies import FunctionJointPolicy
    env = DeadlockEnv(ring(3), P)
    s = evaluate_mc(env, FunctionJointPolicy(lambda env, st, t: (1, 1, 1), "all-hold"), 50)
    assert s.stall == 1.0 and s.deadlock == 0.0 and s.livelock == 0.0


def test_always_request_deadlocks_on_ring():
    env = DeadlockEnv(ring(3), P)
    s = evaluate_mc(env, always_request(3), 100)
    assert s.deadlock == 1.0 and s.completion == 0.0


def test_exact_matches_monte_carlo():
    env = DeadlockEnv(dense(3, 3, 0), P)
    pol = bankers_policy()
    ev = evaluate(env, pol)
    mc = evaluate_mc(env, pol, 2000, seed=2)
    assert mc.social_cost == pytest.approx(ev.social_cost, rel=0.05, abs=0.3)


@pytest.mark.parametrize("top", [ring(3), dense(3, 3, 0), dense(3, 3, 4)])
def test_baselines_deadlock_free_and_optimum_is_lower_bound(top):
    env = DeadlockEnv(top, P)
    c_star = centralized_optimum(env).c_star
    for name, e, pol in all_baselines(env):
        s = evaluate_mc(e, pol, 200, seed=1)
        if name in ("bankers", "resource-ordering"):
            assert s.deadlock == 0.0, name
            assert s.completion == 1.0, name
        if name != "resource-ordering":  # ordering changes o_i, i.e. a different game
            assert s.social_cost >= c_star - 1e-6, name
    cen = centralized_optimum(env)
    assert evaluate(env, cen.policy).social_cost == pytest.approx(cen.c_star)


def test_bankers_safety_check():
    env = DeadlockEnv(ring(3), P)
    assert is_safe(env, [1, 1, 0], [0, 0, 0])
    assert not is_safe(env, [1, 1, 1], [0, 0, 0])


def test_best_response_improves():
    env = DeadlockEnv(ring(3), P)
    pol = always_request(3)
    ev = evaluate(env, pol)
    br = best_response(env, 0, pol)
    assert br.value >= ev.returns[0] - 1e-9
    eps, gains, _ = exploitability(env, pol)
    assert eps > 0  # always-request on a ring is not an equilibrium


def test_stage_equilibria_chicken():
    M1 = np.array([[-10.0, 2.0], [0.0, 1.0]])
    eqs = stage_equilibria(M1, M1.T)
    assert len(eqs) == 3  # two pure + one mixed


def test_two_agent_spe_has_zero_exploitability():
    p = P.with_(H=8)
    sol = solve_two_agent(p, standoff())
    eps, _, _ = exploitability(sol.env, sol.profile)
    assert eps < 1e-6


def test_threshold_interpolation():
    assert find_threshold([1, 2, 4], [0.0, 0.4, 0.8]) == pytest.approx(2.5)
    assert find_threshold([1, 2], [0.1, 0.2]) is None
    assert gap_closed(10, 6, 2) == pytest.approx(0.5)


def test_brd_and_learning_run():
    env = DeadlockEnv(ring(3), P)
    r = best_response_dynamics(env, "part", max_rounds=5)
    assert r.epsilon >= 0
    for m in ("iql", "sql", "fp"):
        res = train(m, env, "full", seed=0, cfg=QConfig(max_episodes=300, min_episodes=100), fp_iterations=3)
        assert res.epsilon is not None and res.epsilon >= 0


def test_run_cell_and_aggregate():
    rows = [run_cell(Cell("ring", 4.0, 1.0, rho, "full", "iql", seed, H=10, episodes=50, max_episodes=200))
            for rho in (0.0, 0.2) for seed in (0, 1)]
    import pandas as pd

    agg = aggregate(pd.DataFrame(rows))
    assert len(agg) == 2 and "g_rho" in agg
