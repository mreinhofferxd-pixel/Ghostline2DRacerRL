"""Numeric observations for the headless RL wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import config
from ..core import collision
from ..core.checkpoints import Gate
from ..core.sim import World


BASE_OBSERVATION_FIELDS: tuple[str, ...] = (
    "car_x",
    "car_y",
    "vel_x",
    "vel_y",
    "heading_cos",
    "heading_sin",
    "speed",
    "forward_speed",
    "slip",
    "target_dx",
    "target_dy",
    "target_distance",
    "target_angle_sin",
    "target_angle_cos",
    "checkpoint_progress",
    "next_checkpoint",
    "target_is_finish",
    "boost_active",
    "boost_time",
    "boost_cooldown_min",
    "boosts_used",
    "wall_hits",
    "wall_scrape_time",
    "wall_contact",
    "largest_impact",
)

# Optional one-corner-lookahead fields (B7.7). The base observation only exposes the
# *next* gate, so the policy cannot plan an apex/exit for the corner after it -- the
# structural ceiling that froze the lap at exactly 4.100 s across many seeds. These
# add the target gate's orientation (its forward normal, relative to car heading) and
# the gate *after* the target (relative position/bearing), so the policy can set up
# the slow CP2->CP3 corner it currently over-slows. Off by default so existing models
# keep their observation dimension; new training opts in.
LOOKAHEAD_OBSERVATION_FIELDS: tuple[str, ...] = (
    "target_normal_sin",
    "target_normal_cos",
    "next2_dx",
    "next2_dy",
    "next2_distance",
    "next2_angle_sin",
    "next2_angle_cos",
    "has_next2",
)

# Optional inside-wall apex sensors (B7.7 strict beat / sub-4 s). The heading-relative
# raycast fan already exposes wall distances, but it rotates with the *car body*, so
# while the car yaws/drifts into a corner the "inside wall" ray keeps moving and the
# 22.5 deg spacing is coarse for judging exact apex clearance. These three rays are
# cast in the *target gate's* frame instead -- forward = the gate's through-direction
# (its normal), plus the two lateral sides -- so they measure corridor room across the
# corner the car is approaching, independent of how the car is rotated. That gives the
# policy a stable signal to perceive and commit to the slow CP2->CP3 apex at speed
# (the persistent ~2-tick residual). Off by default so existing models keep their dim.
WALL_SENSOR_OBSERVATION_FIELDS: tuple[str, ...] = (
    "gate_fwd_wall_dist",
    "gate_left_wall_dist",
    "gate_right_wall_dist",
)

OBSERVATION_FIELDS: tuple[str, ...] = BASE_OBSERVATION_FIELDS + tuple(
    f"raycast_{i:02d}" for i in range(config.RAYCAST_COUNT)
)

_WORLD_DIAGONAL = math.hypot(config.WORLD_WIDTH, config.WORLD_HEIGHT)


def _clamp_unit(x: float) -> float:
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x


def target_gate(world: World) -> tuple[Gate | None, bool]:
    """Return the current target gate and whether it is the finish."""
    checkpoints = world.track.checkpoints
    if world.run.next_cp < len(checkpoints):
        return checkpoints[world.run.next_cp], False
    return world.track.finish, True


@dataclass(frozen=True)
class ObservationConfig:
    raycast_count: int = config.RAYCAST_COUNT
    raycast_max_dist: float = config.RAYCAST_MAX_DIST
    include_lookahead: bool = False
    include_wall_sensors: bool = False

    @property
    def fields(self) -> tuple[str, ...]:
        lookahead = LOOKAHEAD_OBSERVATION_FIELDS if self.include_lookahead else ()
        wall_sensors = WALL_SENSOR_OBSERVATION_FIELDS if self.include_wall_sensors else ()
        return (
            BASE_OBSERVATION_FIELDS
            + lookahead
            + wall_sensors
            + tuple(f"raycast_{i:02d}" for i in range(self.raycast_count))
        )


def _gate_at(world: World, index: int) -> Gate | None:
    """The ordered gate at ``index`` over ``checkpoints + [finish]``, else None."""
    checkpoints = world.track.checkpoints
    if 0 <= index < len(checkpoints):
        return checkpoints[index]
    if index == len(checkpoints):
        return world.track.finish
    return None


def _lookahead_values(world: World, gate: Gate | None) -> list[float]:
    """One-corner-lookahead features, ordered as ``LOOKAHEAD_OBSERVATION_FIELDS``.

    ``gate`` is the current target gate (already resolved by the caller). The
    next-next gate is the one after the target in ``checkpoints + [finish]``; when it
    does not exist (the target is the finish) the features fall back to neutral
    sentinels and ``has_next2`` is 0.
    """
    car = world.car
    if gate is None:
        target_normal_sin, target_normal_cos = 0.0, 1.0
    else:
        normal_rel = math.atan2(gate.ny, gate.nx) - car.heading
        target_normal_sin = math.sin(normal_rel)
        target_normal_cos = math.cos(normal_rel)

    next2 = _gate_at(world, world.run.next_cp + 1)
    if next2 is None:
        return [target_normal_sin, target_normal_cos, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    n2x, n2y = next2.center
    d2x, d2y = n2x - car.px, n2y - car.py
    dist2 = math.hypot(d2x, d2y)
    angle2 = math.atan2(d2y, d2x)
    rel2 = (angle2 - car.heading + math.pi) % (2.0 * math.pi) - math.pi
    return [
        target_normal_sin,
        target_normal_cos,
        d2x / _WORLD_DIAGONAL,
        d2y / _WORLD_DIAGONAL,
        dist2 / _WORLD_DIAGONAL,
        math.sin(rel2),
        math.cos(rel2),
        1.0,
    ]


def _wall_sensor_values(world: World, gate: Gate | None, max_dist: float) -> list[float]:
    """Inside-wall apex sensors in the target-gate frame, ordered as
    ``WALL_SENSOR_OBSERVATION_FIELDS``.

    Three rays cast from the car: forward along the gate's unit normal (its
    through-gate racing direction) and the two lateral sides (the gate tangent,
    left = normal rotated +90deg). Distances are normalized by ``max_dist`` exactly
    like the heading-relative fan. Unlike the fan these are anchored to the gate, not
    the car heading, so they stay stable while the car yaws/drifts through the corner.
    With no target gate (a degenerate track with no gates) they fall back to "open"
    (1.0) -- the same neutral the fan reports when nothing is in range.
    """
    if gate is None:
        return [1.0, 1.0, 1.0]
    car = world.car
    walls = world.track.walls
    fx, fy = gate.nx, gate.ny  # gate forward = its unit through-gate normal
    fwd = collision.raycast(car.px, car.py, fx, fy, walls, max_dist)
    left = collision.raycast(car.px, car.py, -fy, fx, walls, max_dist)  # +90deg
    right = collision.raycast(car.px, car.py, fy, -fx, walls, max_dist)  # -90deg
    return [fwd / max_dist, left / max_dist, right / max_dist]


def observe(world: World, obs_cfg: ObservationConfig = ObservationConfig()) -> tuple[float, ...]:
    """Build a fixed-size, normalized numeric observation.

    This deliberately does not read ``Track.racing_line``. The agent sees physical
    state, run progress, target gates, boost/wall state, and raycasts.
    """
    car = world.car
    gate, is_finish = target_gate(world)
    if gate is None:
        tx, ty = car.px, car.py
    else:
        tx, ty = gate.center
    dx, dy = tx - car.px, ty - car.py
    target_angle = math.atan2(dy, dx)
    rel_angle = (target_angle - car.heading + math.pi) % (2.0 * math.pi) - math.pi
    dist = math.hypot(dx, dy)
    total_gates = len(world.track.checkpoints) + (1 if world.track.finish is not None else 0)
    progress = 1.0 if total_gates == 0 else world.run.next_cp / total_gates
    next_cp = 1.0 if len(world.track.checkpoints) == 0 else world.run.next_cp / len(world.track.checkpoints)

    cooldowns = world.boost_cooldowns
    if cooldowns:
        min_cooldown = min(cooldowns)
        max_cooldown = max(world.track.boost_pads and world.boost_cooldowns or (1.0,))
    else:
        min_cooldown = 0.0
        max_cooldown = 1.0

    values = [
        (car.px / config.WORLD_WIDTH) * 2.0 - 1.0,
        (car.py / config.WORLD_HEIGHT) * 2.0 - 1.0,
        car.vx / world_max_speed(world),
        car.vy / world_max_speed(world),
        math.cos(car.heading),
        math.sin(car.heading),
        car.speed / world_max_speed(world),
        car.forward_speed / world_max_speed(world),
        car.slip_angle / math.pi,
        dx / _WORLD_DIAGONAL,
        dy / _WORLD_DIAGONAL,
        dist / _WORLD_DIAGONAL,
        math.sin(rel_angle),
        math.cos(rel_angle),
        progress,
        next_cp,
        1.0 if is_finish else 0.0,
        1.0 if world.boost_active else 0.0,
        world.boost_time / max(world_max_boost_duration(world), 1e-9),
        min_cooldown / max(max_cooldown, 1e-9),
        world.boosts_used / max(len(world.track.boost_pads), 1),
        world.wall_hits / 10.0,
        world.wall_scrape_time / 10.0,
        1.0 if world.in_wall_contact else 0.0,
        world.largest_impact_speed / world_max_speed(world),
    ]

    if obs_cfg.include_lookahead:
        values.extend(_lookahead_values(world, gate))

    if obs_cfg.include_wall_sensors:
        values.extend(_wall_sensor_values(world, gate, obs_cfg.raycast_max_dist))

    rays = collision.ray_fan(
        car.px,
        car.py,
        car.heading,
        world.track.walls,
        obs_cfg.raycast_count,
        obs_cfg.raycast_max_dist,
    )
    values.extend(dist / obs_cfg.raycast_max_dist for _, dist in rays)
    return tuple(_clamp_unit(float(v)) for v in values)


def world_max_speed(world: World) -> float:
    # The env owns the active Simulation, whose cfg can differ from config.CAR during
    # tests/tuning. ``World`` does not carry cfg, so use the default physics scale.
    return config.CAR.max_boost_speed


def world_max_boost_duration(world: World) -> float:
    return config.CAR.boost_duration
