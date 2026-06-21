"""Executable drift acceptance criteria.

These turn the drift "feel" bar into numbers so it can't silently regress and so a
coding LLM tuning ``CarPhysics`` gets an immediate PASS/FAIL. The experiments live
in ``momentum_lab/eval/harness.py``; ``python -m momentum_lab.eval`` prints the
full human/agent-readable report.

These assert the car-feel bar: controlled drift beats braking, drift-and-catch
takes a tighter line than high-speed grip understeer, and over-rotation costs speed.
"""

from __future__ import annotations

from momentum_lab.config import CAR
from momentum_lab.core.action import Action
from momentum_lab.core.sim import Simulation
from momentum_lab.eval.harness import corner_comparison, over_rotation


def test_controlled_drift_is_faster_than_braking_through_a_corner():
    # With the handbrake (physics_v3) holding drift the WHOLE corner just scrubs
    # speed; the controlled technique is drift-and-catch (rotate, then release to
    # hook the velocity onto the new heading and fire out). That must beat braking.
    _, results = corner_comparison(CAR)
    assert results["brake"].completed and results["drift_catch"].completed
    assert results["drift_catch"].exit_speed > results["brake"].exit_speed


def test_handbrake_cannot_reach_max_speed():
    # Holding drift (the handbrake) cuts drive + scrubs forward speed, so
    # full throttle while drifting cannot hold anywhere near max_speed.
    sim = Simulation()
    sim.reset(seed=0)
    for _ in range(600):
        sim.step(Action(throttle=1.0, drift=True))
    assert sim.world.car.speed < 0.7 * CAR.max_speed


def test_handbrake_scrubs_speed_from_top():
    sim = Simulation()
    sim.reset(seed=0)
    for _ in range(180):
        sim.step(Action(throttle=1.0))
    fast = sim.world.car.speed
    for _ in range(45):  # 0.75 s on the handbrake
        sim.step(Action(throttle=1.0, drift=True))
    assert sim.world.car.speed < fast - 150.0


def test_drift_takes_a_tighter_line_than_grip_through_a_tight_corner():
    # The point of high-speed understeer: at top entry speed a grip turn
    # pushes wide, so the drift-and-catch line gets through a tight corner in much
    # less distance and time. This is what gives drift a purpose / makes it the core
    # technique, rather than being strictly dominated by gripping.
    _, results = corner_comparison(CAR, target_turn_deg=120.0)
    assert results["drift_catch"].distance < results["grip"].distance
    assert results["drift_catch"].time < results["grip"].time


def test_over_rotation_loses_speed():
    _, tidy, over = over_rotation(CAR)
    # Steering harder while drifting reaches a larger slip angle...
    assert over.peak_slip_deg > tidy.peak_slip_deg
    # ...and the extra lateral scrub costs exit speed (same duration => same drag).
    assert over.exit_speed < tidy.exit_speed


def test_eval_is_deterministic():
    # The measurement layer must itself be bit-reproducible, or tuning-by-numbers
    # is built on sand. Same cfg -> same numbers, every time.
    a = corner_comparison(CAR)[1]["drift"].exit_speed
    b = corner_comparison(CAR)[1]["drift"].exit_speed
    assert a == b


def test_drift_metric_counts_only_while_drifting():
    # The drift metric: drift_time accrues only when the
    # handbrake is engaged above drift_min_speed, and a slide builds a peak slip.
    sim = Simulation()
    sim.reset(seed=0)
    for _ in range(180):  # spin up well past drift_min_speed, no drift held
        sim.step(Action(throttle=1.0))
    assert sim.world.drift_time == 0.0 and sim.world.peak_slip == 0.0
    for _ in range(30):  # hold the handbrake + steer for a 0.5 s window
        sim.step(Action(throttle=1.0, steer=1.0, drift=True))
    # Accrues while sliding, but stops once the handbrake bleeds speed below
    # drift_min_speed, so it can never exceed the elapsed window (30 * CONTROL_DT).
    assert 0.3 < sim.world.drift_time <= 30 * (1.0 / 60.0) + 1e-9
    assert sim.world.peak_slip > 0.05  # steering while drifting builds a slip angle


def test_drift_metric_survives_snapshot_restore():
    # The metric rides in World, so branching/eval (and M6 replay) must round-trip it.
    sim = Simulation()
    sim.reset(seed=0)
    for _ in range(120):
        sim.step(Action(throttle=1.0, steer=1.0, drift=True))
    snap = sim.snapshot()
    saved = (snap.drift_time, snap.peak_slip)
    for _ in range(60):  # diverge hard
        sim.step(Action(throttle=1.0, steer=1.0, drift=True))
    sim.restore(snap)
    assert (sim.world.drift_time, sim.world.peak_slip) == saved
