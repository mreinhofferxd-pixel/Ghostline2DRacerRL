"""Ghost playback acceptance."""

from __future__ import annotations

import math

import pytest

from momentum_lab.config import CONTROL_DT, CONTROL_HZ
from momentum_lab.core.action import Action
from momentum_lab.replay import Frame, GhostPlayback, InitialState, ReplayData


def _initial_state() -> InitialState:
    return InitialState(
        car=(0.0, 0.0, 0.0, 0.0, 0.0),
        prev=(0.0, 0.0),
        tick=0,
        sim_time=0.0,
        wall_hits=0,
        wall_scrape_time=0.0,
        largest_impact_speed=0.0,
        in_wall_contact=False,
        drift_time=0.0,
        peak_slip=0.0,
        boost_time=0.0,
        boost_cooldowns=(),
        boosts_used=0,
        run_started=False,
        run_start_tick=0,
        run_next_cp=0,
        run_finished=False,
        run_valid=False,
        run_lap_ticks=0,
        run_cp_ticks=(),
    )


def _frame(i: int, x: float, *, angle: float = 0.0, cp: int = 0) -> Frame:
    return Frame(
        t=(i + 1) * CONTROL_DT,
        x=x,
        y=0.0,
        angle=angle,
        speed=x,
        drift=False,
        cp=cp,
        wall=False,
        boost=False,
    )


def _replay(frames: tuple[Frame, ...]) -> ReplayData:
    return ReplayData(
        track_id="ghost_test",
        physics_version="physics_test",
        seed=0,
        initial_state=_initial_state(),
        lap_time=(len(frames) - 1) * CONTROL_DT,
        valid=True,
        actions=tuple(Action(throttle=1.0) for _ in frames),
        frames=frames,
        control_hz=CONTROL_HZ,
    )


def test_ghost_samples_are_aligned_to_lap_timer_start():
    ghost = GhostPlayback(
        _replay(
            (
                _frame(0, 0.0),
                _frame(1, 10.0),
                _frame(2, 20.0),
            )
        )
    )

    assert ghost.sample(0.0).x == 0.0
    assert ghost.sample(CONTROL_DT).x == 10.0


def test_ghost_interpolates_between_cached_frames():
    ghost = GhostPlayback(
        _replay(
            (
                _frame(0, 0.0, angle=0.0),
                _frame(1, 10.0, angle=math.pi / 2.0),
                _frame(2, 20.0, angle=math.pi / 2.0),
            )
        )
    )

    pose = ghost.sample(CONTROL_DT / 2.0)

    assert pose.x == pytest.approx(5.0)
    assert pose.angle == pytest.approx(math.pi / 4.0)


def test_ghost_delta_uses_matching_checkpoint_stage_when_available():
    ghost = GhostPlayback(
        _replay(
            (
                _frame(0, 0.0, cp=0),
                _frame(1, 100.0, cp=0),
                _frame(2, 0.0, cp=1),
            )
        )
    )

    delta = ghost.delta_to_position(0.0, 0.0, 1.0, checkpoint=1)

    assert delta == pytest.approx(1.0 - 2.0 * CONTROL_DT)
