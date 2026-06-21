"""The Phase-1 insurance policy: the sim is reproducible and snapshot-able.

These protect the RL-ready claim. If any of them ever fails, determinism (and
therefore replay validity) has regressed.
"""

from __future__ import annotations

import math

from momentum_lab.core.action import Action
from momentum_lab.core.sim import Simulation


def _scripted_actions(n: int) -> list[Action]:
    """A varied but deterministic action stream that exercises drift + steering."""
    acts = []
    for i in range(n):
        acts.append(
            Action(
                throttle=1.0 if i % 5 != 0 else 0.0,
                brake=1.0 if i % 17 == 0 else 0.0,
                steer=math.sin(i * 0.05),
                drift=(i % 3 == 0),
            )
        )
    return acts


def _run(actions: list[Action]) -> str:
    sim = Simulation()
    sim.reset(spawn=(200.0, 360.0), heading=0.0, seed=0)
    for a in actions:
        sim.step(a)
    return sim.state_hash()


def test_same_inputs_same_state():
    actions = _scripted_actions(600)
    assert _run(actions) == _run(actions)


def test_seedless_reset_is_still_deterministic():
    sim_a, sim_b = Simulation(), Simulation()
    sim_a.reset()
    sim_b.reset()
    for a in _scripted_actions(300):
        sim_a.step(a)
        sim_b.step(a)
    assert sim_a.state_hash() == sim_b.state_hash()


def test_snapshot_restore_roundtrip():
    sim = Simulation()
    sim.reset(seed=0)
    for a in _scripted_actions(120):
        sim.step(a)

    snap = sim.snapshot()
    expected = sim.state_hash()

    # Diverge hard, then restore and confirm we are bit-identical again.
    for _ in range(200):
        sim.step(Action(throttle=1.0, steer=-1.0, drift=True))
    assert sim.state_hash() != expected

    sim.restore(snap)
    assert sim.state_hash() == expected


def test_restored_state_continues_identically():
    actions = _scripted_actions(400)

    full = Simulation()
    full.reset(seed=0)
    for a in actions:
        full.step(a)

    branched = Simulation()
    branched.reset(seed=0)
    for a in actions[:150]:
        branched.step(a)
    snap = branched.snapshot()
    branched.restore(snap)  # restore is a no-op identity here, but exercises the path
    for a in actions[150:]:
        branched.step(a)

    assert full.state_hash() == branched.state_hash()


def test_car_actually_moves_under_throttle():
    sim = Simulation()
    sim.reset(spawn=(200.0, 360.0), heading=0.0)
    start = sim.world.car.px
    for _ in range(60):
        sim.step(Action(throttle=1.0))
    assert sim.world.car.px > start + 50.0  # accelerated to the right (+x)
