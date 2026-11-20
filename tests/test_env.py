"""Stage 0 deliverable: unit tests of the simulator, incl. hand-checked ring traces."""
import itertools

import numpy as np
import pytest

from deadlock_game import HOLD, REQUEST, YIELD, DeadlockEnv, GameParams, ring, standoff
from deadlock_game.config import custom, dense
from deadlock_game.wfg import cycle_members

P = GameParams(R=8, K=2, D=5, c0=1, rho=0.1, tau_max=3, H=20, L=3)


def rng():
    return np.random.default_rng(0)


def test_cycle_detection():
    assert cycle_members(3, [(0, 1), (1, 2), (2, 0)]) == {0, 1, 2}
    assert cycle_members(3, [(0, 1), (1, 2)]) == frozenset()
    assert cycle_members(4, [(0, 1), (1, 0), (2, 0), (3, 3)]) == {0, 1, 3}


def test_ring_trace_circular_wait():
    env = DeadlockEnv(ring(3), P)
    s = env.initial_state()
    s, r, info = env.step(s, (REQUEST,) * 3, rng())
    assert s[0] == (1, 1, 1)                     # each acquired its first lock
    assert r == pytest.approx((-1.0, -1.0, -1.0))  # phi(0) = c0
    assert not info["cycle"]
    s, r, info = env.step(s, (REQUEST,) * 3, rng())
    assert set(info["edges"]) == {(0, 1), (1, 2), (2, 0)}
    assert info["cycle"] == {0, 1, 2}
    assert r == pytest.approx((-1.1 - 5,) * 3)   # phi(1) + D
    assert s[1] == (2, 2, 2) and s[4] == (1, 1, 1)
    assert env.observe(s, 0, "full") == (1, 2, 0, 1)
    assert env.observe(s, 0, "part") == (1, 2, 0, 1)


def test_ring_trace_yield_breaks_cycle():
    env = DeadlockEnv(ring(3), P)
    s = ((1, 1, 1), (2, 2, 2), (0, 0, 0), (0, 0, 0), (1, 1, 1))
    s, r, info = env.step(s, (YIELD, HOLD, REQUEST), rng())
    assert s[0] == (0, 1, 0) and s[3] == (0, 0, 1)  # agent 3 got R1 and completed
    assert 1 <= s[2][0] <= 3                        # cooldown drawn
    assert s[1][0] == 0                              # holding time reset
    assert r[0] == pytest.approx(-2.0)              # K
    assert r[1] == pytest.approx(-1.0 * 1.1 ** 2)   # holding at h = 2
    assert r[2] == pytest.approx(8.0)               # R, no holding cost on completion
    assert info["completed"] == (2,)
    # cooled-down agent can only Hold; Request is coerced
    assert env.available_actions(s, 0) == (HOLD,)
    s2, _, info = env.step(s, (REQUEST, HOLD, HOLD), rng())
    assert info["actions"][0] == HOLD
    assert s2[2][0] == s[2][0] - 1


def test_tie_breaking_uniform_and_distribution_sums_to_one():
    env = DeadlockEnv(custom([(0,), (0,), (0,)]), P)
    outs = env.transitions(env.initial_state(), (REQUEST,) * 3)
    assert len(outs) == 3
    assert sum(o.prob for o in outs) == pytest.approx(1.0)
    winners = sorted(o.completed[0] for o in outs)
    assert winners == [0, 1, 2]
    # sampled frequencies
    g = np.random.default_rng(1)
    counts = np.zeros(3)
    for _ in range(3000):
        _, _, info = env.step(env.initial_state(), (REQUEST,) * 3, g)
        counts[info["completed"][0]] += 1
    assert np.all(np.abs(counts / 3000 - 1 / 3) < 0.04)


def test_cooldown_distribution():
    env = DeadlockEnv(standoff(), P)
    s = ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0))
    outs = env.transitions(s, (YIELD, YIELD))
    assert len(outs) == 9
    assert {o.next_state[2] for o in outs} == set(itertools.product(range(1, 4), repeat=2))


def test_standoff_reproduces_matrix_M():
    """Equation 11: env payoffs at the standoff equal M_rho(h) for every action pair."""
    from deadlock_game.two_agent import stage_matrix

    for h in (0, 1, 3):
        env = DeadlockEnv(standoff(), P)
        s = ((1, 1), (h, h), (0, 0), (0, 0), (0, 0))
        M, _ = stage_matrix(P, h)
        for a1, a2 in itertools.product((REQUEST, HOLD, YIELD), repeat=2):
            outs = env.transitions(s, (a1, a2))
            exp_r1 = sum(o.prob * o.rewards[0] for o in outs)
            assert exp_r1 == pytest.approx(M[a1, a2]), (h, a1, a2)


def test_transition_matches_step():
    env = DeadlockEnv(dense(3, 3, 0), P)
    g = np.random.default_rng(3)
    s = env.initial_state()
    for _ in range(15):
        a = tuple(int(x) for x in g.integers(0, 3, size=3))
        support = {o.next_state for o in env.transitions(s, a)}
        s2, _, _ = env.step(s, a, g)
        assert s2 in support
        s = s2 if not env.is_terminal(s2) else env.initial_state()


def test_holdings_are_prefixes_and_resources_exclusive():
    env = DeadlockEnv(dense(3, 3, 1), P)
    g = np.random.default_rng(5)
    s = env.initial_state()
    for _ in range(300):
        a = tuple(int(x) for x in g.integers(0, 3, size=3))
        s, _, _ = env.step(s, a, g)
        held = env.held_sets(s)
        flat = [r for h in held for r in h]
        assert len(flat) == len(set(flat))
        for i in range(3):
            if s[3][i]:
                assert s[0][i] == 0
        if env.is_terminal(s):
            s = env.initial_state()


def test_escalating_holding_cost():
    p = P.with_(rho=0.5)
    assert p.phi(0) == 1.0 and p.phi(2) == pytest.approx(2.25)
    assert P.base.rho == 0.0
