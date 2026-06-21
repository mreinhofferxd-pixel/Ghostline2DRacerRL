"""Checkpoint and lap-timing acceptance tests.

Three layers: the directed gate-crossing geometry, the lap/timer state machine
(its headline guarantee is that out-of-order crossings are rejected), and an
end-to-end scripted lap that must report a valid time.
"""

from __future__ import annotations

import pytest

from momentum_lab.core.action import Action
from momentum_lab.core.checkpoints import Gate
from momentum_lab.core.sim import Simulation
from momentum_lab.core.timing import RunState
from momentum_lab.eval.harness import time_trial
from momentum_lab.tracks import load_track_by_id

TRACK = load_track_by_id("track_01_easy_loop")


# --- gate geometry -----------------------------------------------------------
def _vertical_gate() -> Gate:
    # Endpoints ordered bottom->top so the forward normal points +x (see
    # Gate.from_endpoints: n = tangent rotated +90deg).
    return Gate.from_endpoints(100.0, 100.0, 100.0, -100.0)


def test_forward_crossing_matches_normal():
    g = _vertical_gate()
    assert g.nx > 0.99 and abs(g.ny) < 1e-9  # normal is +x
    assert g.crossing(90.0, 0.0, 110.0, 0.0) == +1  # moving +x through the gate


def test_reverse_crossing_is_negative():
    g = _vertical_gate()
    assert g.crossing(110.0, 0.0, 90.0, 0.0) == -1  # moving -x, against the normal


def test_no_crossing_when_not_reaching_line():
    g = _vertical_gate()
    assert g.crossing(60.0, 0.0, 90.0, 0.0) == 0  # stops short of x=100


def test_crossing_outside_gate_extent_is_ignored():
    g = _vertical_gate()  # gate spans y in [-100, 100]
    assert g.crossing(90.0, 500.0, 110.0, 500.0) == 0  # crosses the line far above


def test_parallel_motion_does_not_cross():
    g = _vertical_gate()
    assert g.crossing(90.0, -50.0, 90.0, 50.0) == 0  # moves along, never reaches x=100


# --- lap state machine -------------------------------------------------------
def _gates() -> tuple[Gate, Gate, Gate]:
    """Three +x gates at x=100/200/300 spanning a wide vertical band."""
    return (
        Gate.from_endpoints(100.0, 100.0, 100.0, -100.0),
        Gate.from_endpoints(200.0, 100.0, 200.0, -100.0),
        Gate.from_endpoints(300.0, 100.0, 300.0, -100.0),
    )


def _cross(run: RunState, cps, finish, x0, x1, *, throttle=True, tick=1, y=0.0):
    run.update(
        throttle_on=throttle,
        prev=(x0, y),
        curr=(x1, y),
        checkpoints=cps,
        finish=finish,
        tick=tick,
    )


def test_timer_does_not_start_without_throttle():
    cps = _gates()
    run = RunState()
    # Cross the first gate, but with no throttle -> timer never armed, nothing counts.
    _cross(run, cps, None, 90.0, 110.0, throttle=False, tick=1)
    assert not run.started
    assert run.next_cp == 0


def test_timer_starts_on_first_throttle():
    run = RunState()
    _cross(run, (), None, 0.0, 1.0, throttle=True, tick=7)
    assert run.started and run.start_tick == 7


def test_checkpoints_must_be_passed_in_order():
    cp1, cp2, cp3 = _gates()
    cps = (cp1, cp2, cp3)
    run = RunState()
    _cross(run, cps, None, -10.0, 1.0, tick=1)  # arm the timer behind x=100
    assert run.next_cp == 0
    _cross(run, cps, None, 90.0, 110.0, tick=2)  # cross cp1
    assert run.next_cp == 1
    _cross(run, cps, None, 190.0, 210.0, tick=3)  # cross cp2
    assert run.next_cp == 2


def test_out_of_order_crossing_is_rejected():
    """The §17.5 acceptance: crossing a later gate early must not advance progress."""
    cp1, cp2, cp3 = _gates()
    cps = (cp1, cp2, cp3)
    run = RunState()
    _cross(run, cps, None, -10.0, 1.0, tick=1)  # arm timer
    # Skip cp1; cross cp2 (x=200) and cp3 (x=300) early — neither is the expected next.
    _cross(run, cps, None, 190.0, 210.0, tick=2)
    _cross(run, cps, None, 290.0, 310.0, tick=3)
    assert run.next_cp == 0  # still waiting on cp1 — out-of-order crossings ignored


def test_reverse_crossing_does_not_advance():
    cp1, cp2, cp3 = _gates()
    cps = (cp1, cp2, cp3)
    run = RunState()
    _cross(run, cps, None, -10.0, 1.0, tick=1)
    _cross(run, cps, None, 110.0, 90.0, tick=2)  # cross cp1 the wrong way (-x)
    assert run.next_cp == 0


def test_finish_only_closes_after_all_checkpoints():
    cp1, cp2, cp3 = _gates()
    cps = (cp1, cp2, cp3)
    finish = Gate.from_endpoints(400.0, 100.0, 400.0, -100.0)
    run = RunState()
    _cross(run, cps, finish, -10.0, 1.0, tick=1)
    # Cross the finish line before any checkpoint: must NOT close the lap.
    _cross(run, cps, finish, 390.0, 410.0, tick=2)
    assert not run.finished


def test_full_valid_lap_reports_a_time():
    cp1, cp2, cp3 = _gates()
    cps = (cp1, cp2, cp3)
    finish = Gate.from_endpoints(400.0, 100.0, 400.0, -100.0)
    run = RunState()
    _cross(run, cps, finish, -10.0, 1.0, tick=10)  # timer starts at tick 10
    _cross(run, cps, finish, 90.0, 110.0, tick=20)  # cp1
    _cross(run, cps, finish, 190.0, 210.0, tick=30)  # cp2
    _cross(run, cps, finish, 290.0, 310.0, tick=40)  # cp3
    assert not run.finished  # still need the finish line
    _cross(run, cps, finish, 390.0, 410.0, tick=70)  # finish: crosses x=400 mid-step
    assert run.finished and run.valid
    assert run.lap_ticks == 60  # canonical integer tick count is unchanged
    # Sub-tick (B9): the move 390->410 crosses the line at the midpoint, so the true
    # crossing is half a tick before the closing tick -> lap_time is (60 - 0.5) ticks.
    assert run.finish_fraction == pytest.approx(0.5)
    assert run.lap_time(70, 1.0 / 60.0) == pytest.approx((60 - 0.5) * (1.0 / 60.0))
    assert run.split_times(1.0 / 60.0) == [10 / 60.0, 20 / 60.0, 30 / 60.0]


def test_finish_line_crossing_is_sub_tick():
    """B9: two laps that close on the *same* control tick but cross the finish line at
    different points within that step report different (sub-tick) times, while the
    canonical integer ``lap_ticks`` stays equal for both."""
    finish = Gate.from_endpoints(400.0, 100.0, 400.0, -100.0)  # +x normal
    dt = 1.0 / 60.0

    def lap(x0: float, x1: float) -> RunState:
        run = RunState()
        _cross(run, (), finish, -10.0, 1.0, tick=10)  # arm the timer at tick 10
        _cross(run, (), finish, x0, x1, tick=70)  # close the lap on tick 70
        assert run.finished and run.valid and run.lap_ticks == 60
        return run

    early = lap(399.0, 409.0)  # crosses x=400 early in the step (t=0.1)
    late = lap(391.0, 401.0)  # crosses x=400 late in the step (t=0.9)

    assert early.finish_fraction == pytest.approx(0.1)
    assert late.finish_fraction == pytest.approx(0.9)
    t_early = early.lap_time(70, dt)
    t_late = late.lap_time(70, dt)
    assert t_early != t_late  # the staircase is broken: same tick, different time
    assert t_early == pytest.approx((60 - 0.9) * dt)
    assert t_late == pytest.approx((60 - 0.1) * dt)
    assert t_early < t_late  # crossing earlier within the step is the faster lap


# --- end-to-end: a scripted lap of the real track ----------------------------
def test_scripted_lap_on_track_01_is_valid_and_timed():
    """A valid lap reports a time. Drive the real track
    headless with the pure-pursuit autopilot and check the timed result."""
    assert len(TRACK.checkpoints) == 4 and TRACK.finish is not None
    result = time_trial(TRACK)
    assert result.completed and result.valid
    assert result.cp_reached == 4
    assert 0.0 < result.lap_time < 60.0
    assert len(result.splits) == 4
    # Splits are monotonic and within the final lap time.
    assert list(result.splits) == sorted(result.splits)
    assert result.splits[-1] < result.lap_time


def test_scripted_lap_is_deterministic():
    a = time_trial(TRACK)
    b = time_trial(TRACK)
    assert a == b


def test_track_01_finish_catches_the_inside_line():
    """The start/finish spans the main straight wall-to-wall, so no line can skip it.
    A forward move down the straight crosses it +1; the reverse crosses -1."""
    assert TRACK.finish is not None
    g = TRACK.finish
    cx, cy = g.center
    fwd = g.crossing(cx - g.nx * 8, cy - g.ny * 8, cx + g.nx * 8, cy + g.ny * 8)
    rev = g.crossing(cx + g.nx * 8, cy + g.ny * 8, cx - g.nx * 8, cy - g.ny * 8)
    assert fwd == +1 and rev == -1


# --- snapshot/restore: lap progress is part of the round-trip -----------------
def test_runstate_copy_is_independent():
    run = RunState(started=True, start_tick=5, next_cp=2, cp_ticks=(10, 20))
    clone = run.copy()
    run.next_cp = 99
    run.cp_ticks = (1,)
    assert clone.next_cp == 2 and clone.cp_ticks == (10, 20) and clone.start_tick == 5


def test_sim_snapshot_restore_preserves_lap_progress():
    """Lap progress rides in ``World`` and must survive snapshot/restore (the basis
    of M6 replay + branching), not just the physics vector."""
    sim = Simulation()
    sim.reset(track=TRACK, seed=0)
    for _ in range(30):
        sim.step(Action(throttle=1.0))  # arms the timer
    assert sim.world.run.started
    snap = sim.snapshot()
    start_tick = sim.world.run.start_tick

    for _ in range(120):  # diverge hard
        sim.step(Action(throttle=1.0, steer=1.0, drift=True))

    sim.restore(snap)
    assert sim.world.run.started
    assert sim.world.run.start_tick == start_tick
    assert sim.world.tick == snap.tick
