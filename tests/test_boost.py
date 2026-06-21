"""Boost pad acceptance."""

from __future__ import annotations

from momentum_lab.config import CAR, PHYSICS_DT
from momentum_lab.core import physics
from momentum_lab.core.action import Action
from momentum_lab.core.car import Car
from momentum_lab.core.sim import Simulation
from momentum_lab.core.track import BoostPad, Track
from momentum_lab.eval.harness import boost_sprint


def _boost_track() -> Track:
    return Track(
        track_id="boost_test",
        spawn=(0.0, 0.0),
        spawn_heading=0.0,
        boost_pads=(BoostPad(-100.0, -100.0, 2000.0, 100.0),),
    )


def test_boost_pad_can_push_past_normal_max_speed():
    sim = Simulation()
    sim.reset(track=_boost_track(), seed=0)
    top_speed = 0.0
    for _ in range(240):
        sim.step(Action(throttle=1.0))
        top_speed = max(top_speed, sim.world.car.speed)

    assert sim.world.boosts_used > 0
    assert top_speed > CAR.max_speed + 25.0
    assert top_speed <= CAR.max_boost_speed + 1e-9


def test_pad_cooldown_prevents_retrigger_every_frame():
    sim = Simulation()
    sim.reset(track=_boost_track(), seed=0)
    sim.step()
    assert sim.world.boosts_used == 1

    for _ in range(10):
        sim.step()
    assert sim.world.boosts_used == 1
    assert sim.world.boost_cooldowns[0] > 0.0


def test_boost_state_survives_snapshot_restore():
    sim = Simulation()
    sim.reset(track=_boost_track(), seed=0)
    sim.step()
    snap = sim.snapshot()
    saved = (snap.boost_time, snap.boost_cooldowns, snap.boosts_used, sim.state_hash())

    for _ in range(45):
        sim.step(Action(throttle=1.0))

    sim.restore(snap)
    assert (sim.world.boost_time, sim.world.boost_cooldowns, sim.world.boosts_used) == saved[:3]
    assert sim.state_hash() == saved[3]


def test_boost_overspeed_decays_after_active_window():
    sim = Simulation()
    sim.reset(seed=0)
    sim.world.car.vx = 900.0
    start = sim.world.car.speed

    for _ in range(30):
        sim.step(Action(throttle=1.0))

    assert sim.world.car.speed < start


def test_large_slip_boosts_along_velocity_not_heading():
    base = Car(px=0.0, py=0.0, vx=0.0, vy=500.0, heading=0.0)
    plain = base.copy()
    boosted = base.copy()

    physics.integrate(plain, Action(), PHYSICS_DT, CAR)
    physics.integrate(boosted, Action(), PHYSICS_DT, CAR, boost_active=True)

    assert boosted.vy > plain.vy + 0.5 * CAR.boost_accel * PHYSICS_DT
    assert abs(boosted.vx - plain.vx) < 1.0


def test_boosted_line_beats_same_line_without_pad():
    sprint = boost_sprint(CAR)
    plain, boosted = sprint["plain"], sprint["boosted"]

    assert plain.completed and boosted.completed
    assert boosted.boosts_used > 0
    assert boosted.time < plain.time
    assert boosted.top_speed > plain.top_speed
