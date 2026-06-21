"""Collision acceptance: the circle stays inside the track, never
tunnels, impacts slow it hard, scrapes slide, and it is all deterministic."""

from __future__ import annotations

import math

from momentum_lab.config import CAR
from momentum_lab.core import collision
from momentum_lab.core.action import Action
from momentum_lab.core.car import Car
from momentum_lab.core.collision import Segment
from momentum_lab.core.sim import Simulation
from momentum_lab.tracks import load_track_by_id

TRACK = load_track_by_id("track_01_easy_loop")


def _min_clearance(px: float, py: float, walls) -> float:
    return min(
        math.hypot(px - cx, py - cy)
        for cx, cy in (
            collision.closest_point_on_segment(px, py, s.x1, s.y1, s.x2, s.y2)
            for s in walls
        )
    )


def test_circle_never_ends_a_step_inside_a_wall():
    """The headline M2 acceptance: drive hard (throttle + sweeping steer + drift)
    for 1500 control steps and assert the car is outside every wall every step."""
    sim = Simulation()
    sim.reset(track=TRACK, seed=0)
    for i in range(1500):
        sim.step(Action(throttle=1.0, steer=math.sin(i * 0.05), drift=(i % 3 == 0)))
        clear = _min_clearance(sim.world.car.px, sim.world.car.py, TRACK.walls)
        assert clear >= CAR.radius - 0.5, f"penetrated at step {i}: clearance {clear}"


def test_no_tunneling_at_extreme_speed():
    # Fire the circle straight through a wall far faster than any in-game speed.
    wall = Segment(0, 200, 400, 200)
    car = Car(px=200, py=100, vx=0.0, vy=50000.0)
    collision.advance(car, 1.0 / 120.0, (wall,), CAR)
    assert car.py <= 200 - CAR.radius + 0.5  # stopped on the near side
    assert _min_clearance(car.px, car.py, (wall,)) >= CAR.radius - 0.5


def test_head_on_impact_kills_most_speed():
    wall = Segment(0, 200, 400, 200)
    car = Car(px=200, py=185, vx=0.0, vy=400.0)  # straight into the wall
    contact = collision.advance(car, 1.0 / 120.0, (wall,), CAR)
    assert contact is not None and contact.is_impact
    assert car.speed < 0.5 * 400.0  # big speed loss


def test_scrape_keeps_more_speed_than_impact():
    wall = Segment(0, 200, 400, 200)
    car = Car(px=200, py=185, vx=400.0, vy=150.0)  # mostly tangential -> scrape
    contact = collision.advance(car, 1.0 / 120.0, (wall,), CAR)
    assert contact is not None and not contact.is_impact
    assert car.speed > 0.7 * 400.0  # tangential speed mostly preserved


def test_tangential_velocity_is_preserved_through_resolution():
    # Pure along-wall motion with a tiny nudge in: direction stays ~tangential.
    wall = Segment(0, 200, 400, 200)
    car = Car(px=200, py=185, vx=300.0, vy=120.0)
    collision.advance(car, 1.0 / 120.0, (wall,), CAR)
    # after resolution the into-wall (+y) component is gone; motion is along -x/+x
    assert abs(car.vy) < 1e-6
    assert car.vx > 0.0


def test_collision_is_deterministic():
    def run():
        sim = Simulation()
        sim.reset(track=TRACK, seed=0)
        for i in range(400):
            sim.step(Action(throttle=1.0, steer=math.sin(i * 0.05), drift=(i % 3 == 0)))
        return (
            sim.state_hash(),
            sim.world.wall_hits,
            round(sim.world.wall_scrape_time, 9),
            round(sim.world.largest_impact_speed, 6),
        )

    assert run() == run()
