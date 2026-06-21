"""Swept circle-vs-segment collision, wall resolution, and wall raycasts.

The car is a circle (``cfg.radius``); walls are line ``Segment``s. Motion is
resolved by a **swept** test so the circle cannot tunnel through a thin wall at
high speed (it finds the time-of-impact along the substep move, not just a
discrete end-of-step overlap). Resolution rules:

  * push the circle out along the contact normal to rest at distance ``radius``;
  * remove only the *into-wall* (normal) velocity component, keep the tangential;
  * scale the retained tangential speed by a friction factor — a hard
    ``wall_speed_loss_factor`` for an **impact** (large normal speed) or a gentler
    ``wall_scrape_friction`` for a **scrape** (mostly tangential).

Pure module: no pygame, no wall-clock, no ``dt`` leakage beyond the fixed substep
``dt`` the sim passes in. Raycasts live here too because both the F1 debug view and
the future RL observation need cheap distance-to-wall queries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import CarPhysics

_EPS = 1e-9
_SKIN = 0.01  # tiny outward nudge after resolving, so we don't immediately re-hit
_MAX_ITERS = 4  # collision passes per substep (lets the circle slide into a corner)


@dataclass(frozen=True)
class Segment:
    """A wall line segment in world units. Immutable static geometry."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Contact:
    """What happened against the walls during one physics substep."""

    max_normal_speed: float  # largest into-wall speed removed (px/s)
    is_impact: bool  # True if that exceeded cfg.wall_impact_speed


# --- geometry helpers --------------------------------------------------------
def closest_point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Closest point to (px,py) on segment (x1,y1)-(x2,y2)."""
    ex, ey = x2 - x1, y2 - y1
    len_sq = ex * ex + ey * ey
    if len_sq < _EPS:
        return x1, y1
    t = ((px - x1) * ex + (py - y1) * ey) / len_sq
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return x1 + t * ex, y1 + t * ey


def _swept_first_hit(
    ax: float, ay: float, dx: float, dy: float, r: float, seg: Segment
) -> tuple[float, float, float] | None:
    """Earliest time-of-impact of a circle of radius ``r`` whose center moves from
    (ax,ay) by (dx,dy), against ``seg``. Returns ``(t, nx, ny)`` with t in [0,1] and
    a unit outward normal (wall -> circle), or None if it stays clear.

    Assumes the circle starts outside the segment's radius-r capsule (the sim runs a
    depenetration pass first), so this only needs the *entering* crossing.
    """
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
    best_t = math.inf
    best_n = (0.0, 0.0)

    # (a) Against the segment body: the two lines offset by +/- r, clipped to span.
    ex, ey = x2 - x1, y2 - y1
    elen = math.hypot(ex, ey)
    if elen > _EPS:
        ux, uy = ex / elen, ey / elen
        nx, ny = -uy, ux  # unit normal to the segment line
        s_a = (ax - x1) * nx + (ay - y1) * ny  # signed dist of start from the line
        s_d = dx * nx + dy * ny  # rate of change of that signed dist
        if abs(s_d) > _EPS:
            side = 1.0 if s_a >= 0.0 else -1.0
            t = (side * r - s_a) / s_d
            if 0.0 <= t <= 1.0:
                cx, cy = ax + dx * t, ay + dy * t
                proj = (cx - x1) * ux + (cy - y1) * uy  # along-segment distance
                if 0.0 <= proj <= elen and t < best_t:
                    best_t, best_n = t, (side * nx, side * ny)

    # (b)/(c) Against each endpoint, treated as a disk of radius r.
    for ex0, ey0 in ((x1, y1), (x2, y2)):
        fx, fy = ax - ex0, ay - ey0
        aq = dx * dx + dy * dy
        if aq < _EPS:
            continue
        bq = 2.0 * (fx * dx + fy * dy)
        cq = fx * fx + fy * fy - r * r
        disc = bq * bq - 4.0 * aq * cq
        if disc < 0.0:
            continue
        t = (-bq - math.sqrt(disc)) / (2.0 * aq)  # earliest root
        if 0.0 <= t <= 1.0 and t < best_t:
            cx, cy = ax + dx * t, ay + dy * t
            nlen = math.hypot(cx - ex0, cy - ey0)
            if nlen > _EPS:
                best_t, best_n = t, ((cx - ex0) / nlen, (cy - ey0) / nlen)

    if best_t is math.inf or best_t == math.inf:
        return None
    return best_t, best_n[0], best_n[1]


def _depenetrate(
    px: float, py: float, vx: float, vy: float, r: float, walls, cfg: CarPhysics
) -> tuple[float, float, float, float, bool, float]:
    """Push the circle out of any wall it already overlaps and strip into-wall
    velocity. Returns (px, py, vx, vy, touched, max_normal_speed)."""
    touched = False
    max_ns = 0.0
    for seg in walls:
        cx, cy = closest_point_on_segment(px, py, seg.x1, seg.y1, seg.x2, seg.y2)
        ox, oy = px - cx, py - cy
        d = math.hypot(ox, oy)
        if d >= r:
            continue
        if d > _EPS:
            nx, ny = ox / d, oy / d
        else:  # center exactly on the wall: push along the segment's normal
            ex, ey = seg.x2 - seg.x1, seg.y2 - seg.y1
            el = math.hypot(ex, ey) or 1.0
            nx, ny = -ey / el, ex / el
        px += nx * (r - d)
        py += ny * (r - d)
        v_n = vx * nx + vy * ny
        if v_n < 0.0:  # moving into the wall -> remove that component
            touched = True
            ns = -v_n
            if ns > max_ns:
                max_ns = ns
            tx, ty = vx - v_n * nx, vy - v_n * ny
            friction = (
                cfg.wall_speed_loss_factor
                if ns > cfg.wall_impact_speed
                else cfg.wall_scrape_friction
            )
            vx, vy = tx * friction, ty * friction
    return px, py, vx, vy, touched, max_ns


def advance(car, dt: float, walls, cfg: CarPhysics) -> Contact | None:
    """Move ``car`` by its velocity over ``dt`` with swept wall collision, in place.

    Returns a ``Contact`` summarizing the worst
    wall interaction this substep, or None if the car never touched a wall.
    """
    r = cfg.radius
    px, py = car.px, car.py
    vx, vy = car.vx, car.vy

    px, py, vx, vy, touched, max_ns = _depenetrate(px, py, vx, vy, r, walls, cfg)

    remaining = dt
    for _ in range(_MAX_ITERS):
        dx, dy = vx * remaining, vy * remaining
        # earliest hit across all walls
        hit_t = math.inf
        hit_n = (0.0, 0.0)
        for seg in walls:
            h = _swept_first_hit(px, py, dx, dy, r, seg)
            if h is not None and h[0] < hit_t:
                hit_t, hit_n = h[0], (h[1], h[2])
        if hit_t is math.inf or hit_t == math.inf:
            px += dx
            py += dy
            break
        px += dx * hit_t
        py += dy * hit_t
        nx, ny = hit_n
        v_n = vx * nx + vy * ny
        if v_n < 0.0:
            touched = True
            ns = -v_n
            if ns > max_ns:
                max_ns = ns
            tx, ty = vx - v_n * nx, vy - v_n * ny
            friction = (
                cfg.wall_speed_loss_factor
                if ns > cfg.wall_impact_speed
                else cfg.wall_scrape_friction
            )
            vx, vy = tx * friction, ty * friction
        px += nx * _SKIN
        py += ny * _SKIN
        remaining *= 1.0 - hit_t
        if remaining <= _EPS or (vx * vx + vy * vy) < _EPS:
            break

    car.px, car.py, car.vx, car.vy = px, py, vx, vy
    if not touched:
        return None
    return Contact(max_normal_speed=max_ns, is_impact=max_ns > cfg.wall_impact_speed)


# --- raycasts ----------------------------------------------------------------
def raycast(ox: float, oy: float, dx: float, dy: float, walls, max_dist: float) -> float:
    """Distance from (ox,oy) along unit ray (dx,dy) to the nearest wall, capped at
    ``max_dist``. (dx,dy) must be unit length."""
    nearest = max_dist
    for seg in walls:
        ex, ey = seg.x2 - seg.x1, seg.y2 - seg.y1
        det = ex * dy - dx * ey  # cross(D, e) with sign per the 2x2 solve below
        if abs(det) < _EPS:
            continue  # ray parallel to segment
        wx, wy = seg.x1 - ox, seg.y1 - oy
        t = (ex * wy - wx * ey) / det  # distance along the (unit) ray
        s = (dx * wy - dy * wx) / det  # parameter along the segment
        if t >= 0.0 and 0.0 <= s <= 1.0 and t < nearest:
            nearest = t
    return nearest


def ray_fan(
    ox: float, oy: float, heading: float, walls, count: int, max_dist: float
) -> list[tuple[float, float]]:
    """``count`` rays evenly around the car (starting at ``heading``). Returns a list
    of (absolute_angle, distance) — the raw material for the F1 view and the future
    RL observation."""
    out = []
    for i in range(count):
        a = heading + (2.0 * math.pi) * i / count
        out.append((a, raycast(ox, oy, math.cos(a), math.sin(a), walls, max_dist)))
    return out
