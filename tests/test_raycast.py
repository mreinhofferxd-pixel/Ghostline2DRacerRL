"""Raycast distance-to-wall infrastructure for debug view and RL observation."""

from __future__ import annotations

import math

from momentum_lab.core import collision
from momentum_lab.core.collision import Segment


def test_ray_hits_wall_ahead():
    wall = Segment(100, -50, 100, 50)  # vertical wall at x=100
    d = collision.raycast(0, 0, 1.0, 0.0, (wall,), 1000.0)
    assert abs(d - 100.0) < 1e-6


def test_ray_pointing_away_returns_max():
    wall = Segment(100, -50, 100, 50)
    d = collision.raycast(0, 0, -1.0, 0.0, (wall,), 1000.0)
    assert d == 1000.0


def test_ray_past_segment_end_misses():
    wall = Segment(100, 50, 100, 150)  # wall entirely above the +x axis
    d = collision.raycast(0, 0, 1.0, 0.0, (wall,), 1000.0)
    assert d == 1000.0  # ray along y=0 never crosses the segment's y-span


def test_ray_distance_matches_geometry():
    wall = Segment(300, -100, 300, 100)  # vertical wall at x=300
    for ang in (0.0, 0.3, -0.3):
        d = collision.raycast(0, 0, math.cos(ang), math.sin(ang), (wall,), 9999.0)
        expect = 300.0 / math.cos(ang)
        if abs(expect * math.sin(ang)) <= 100.0:  # hit point within the wall span
            assert abs(d - expect) < 1e-6


def test_ray_fan_count_and_cap_with_no_walls():
    fan = collision.ray_fan(0, 0, 0.0, (), 16, 500.0)
    assert len(fan) == 16
    assert all(dist == 500.0 for _, dist in fan)


def test_ray_fan_finds_nearest_wall_inside_a_box():
    track = []
    # 200x200 box centered on origin; nearest wall is 100 away in each cardinal dir
    track = (
        Segment(-100, -100, 100, -100),
        Segment(100, -100, 100, 100),
        Segment(100, 100, -100, 100),
        Segment(-100, 100, -100, -100),
    )
    fan = collision.ray_fan(0, 0, 0.0, track, 4, 1000.0)  # rays at 0, 90, 180, 270
    for _, dist in fan:
        assert abs(dist - 100.0) < 1e-6
