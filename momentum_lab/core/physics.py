"""Pure physics integration for one fixed substep.

Drift is *emergent*: velocity is decomposed into the car's local frame and
lateral velocity is partially retained (``grip``). Reducing grip (the drift
modifier) lets the velocity lag behind the heading, which is exactly a slide.
There is no drift-angle lookup table anywhere.

This module imports nothing from pygame, input, or wall-clock — it is the part of
the program a future RL environment runs thousands of times per second.
"""

from __future__ import annotations

import math

from ..config import CarPhysics
from . import collision
from .action import Action
from .car import Car


def integrate(
    car: Car,
    action: Action,
    dt: float,
    cfg: CarPhysics,
    walls=(),
    boost_active: bool = False,
) -> "collision.Contact | None":
    """Advance ``car`` by one physics substep ``dt`` in place.

    ``dt`` is the *fixed* physics timestep, supplied by the Simulation — never a
    variable per-frame delta. Returns the
    wall ``Contact`` for this substep (or None) so the sim can tally wall stats.
    """
    cos_h = math.cos(car.heading)
    sin_h = math.sin(car.heading)
    fwd_speed = car.vx * cos_h + car.vy * sin_h  # signed speed along heading
    speed0 = math.hypot(car.vx, car.vy)
    # One drift predicate for the whole substep: handbrake longitudinal, steering
    # authority, and lateral grip all engage/disengage together.
    drifting = action.drift and speed0 >= cfg.drift_min_speed

    # 1. Longitudinal: throttle / brake / reverse, along the heading axis. While
    #    drifting the handbrake cuts engine drive (drift_throttle_factor) and scrubs
    #    forward speed (drift_brake_accel) — you cannot hold max_speed on the brake.
    # Boost overspeed should decay by drag once the boost ends. While above the
    # normal top speed, forward throttle is withheld so the engine cannot sustain
    # a boosted velocity forever.
    throttle_blocked_by_overspeed = speed0 > cfg.max_speed and fwd_speed > 0.0
    if action.throttle > 0.0 and not throttle_blocked_by_overspeed:
        accel = action.throttle * cfg.engine_accel
        if drifting:
            accel *= cfg.drift_throttle_factor
        car.vx += cos_h * accel * dt
        car.vy += sin_h * accel * dt
    if action.brake > 0.0:
        if fwd_speed > 1.0:
            # Braking opposes forward motion; clamp so we don't snap into reverse.
            dv = min(action.brake * cfg.brake_accel * dt, fwd_speed)
            car.vx -= cos_h * dv
            car.vy -= sin_h * dv
        else:
            # At rest or already reversing: accelerate backwards.
            accel = action.brake * cfg.reverse_accel
            car.vx -= cos_h * accel * dt
            car.vy -= sin_h * accel * dt
    if drifting:
        # Handbrake scrub on the (recomputed) forward component; never reverses.
        fwd_now = car.vx * cos_h + car.vy * sin_h
        if fwd_now > 0.0:
            dv = min(cfg.drift_brake_accel * dt, fwd_now)
            car.vx -= cos_h * dv
            car.vy -= sin_h * dv

    # 2. Steering: kinematic, speed-scaled, reverse-aware. Above turn_full_speed a
    #    grip turn understeers — authority fades toward grip_turn_falloff at
    #    max_speed, so it pushes wide. Holding drift bypasses the falloff, so drift
    #    is how you rotate the car through a tight corner at speed.
    speed = math.hypot(car.vx, car.vy)
    if speed > 1.0:
        base = min(speed / cfg.turn_full_speed, 1.0)
        if drifting:
            authority = base * cfg.drift_turn_authority
        else:
            span = max(1.0, cfg.max_speed - cfg.turn_full_speed)
            over = min(max((speed - cfg.turn_full_speed) / span, 0.0), 1.0)
            authority = base * (1.0 - (1.0 - cfg.grip_turn_falloff) * over)
        direction = 1.0 if fwd_speed >= 0.0 else -1.0
        car.heading += action.steer * cfg.turn_rate * authority * direction * dt
        # Keep heading in [-pi, pi] for clean slip-angle math downstream.
        if car.heading > math.pi:
            car.heading -= 2.0 * math.pi
        elif car.heading < -math.pi:
            car.heading += 2.0 * math.pi

    # 3. Lateral grip in the local frame of the NEW heading (drift emerges here).
    cos_h = math.cos(car.heading)
    sin_h = math.sin(car.heading)
    v_fwd = car.vx * cos_h + car.vy * sin_h
    v_lat = -car.vx * sin_h + car.vy * cos_h  # component along the right axis
    grip = cfg.grip_drift if drifting else cfg.grip_normal
    v_lat *= grip
    car.vx = cos_h * v_fwd - sin_h * v_lat
    car.vy = sin_h * v_fwd + cos_h * v_lat

    # 4. Drag (always) + rolling resistance (only while coasting).
    speed = math.hypot(car.vx, car.vy)
    if speed > 0.0:
        drag_factor = max(0.0, 1.0 - cfg.drag * dt)
        car.vx *= drag_factor
        car.vy *= drag_factor
        if action.throttle == 0.0 and action.brake == 0.0:
            rr = min(cfg.rolling_resistance * dt, speed)
            car.vx -= (car.vx / speed) * rr
            car.vy -= (car.vy / speed) * rr

    # 5. Clamp to the current speed budget.
    speed = math.hypot(car.vx, car.vy)
    limit = cfg.max_boost_speed if (boost_active or speed0 > cfg.max_speed) else cfg.max_speed
    if speed > limit:
        scale = limit / speed
        car.vx *= scale
        car.vy *= scale

    # 6. Integrate position with swept wall collision.
    if boost_active:
        speed = math.hypot(car.vx, car.vy)
        if speed > 1.0 and abs(car.slip_angle) >= cfg.boost_velocity_slip_angle:
            bx, by = car.vx / speed, car.vy / speed
        else:
            bx, by = math.cos(car.heading), math.sin(car.heading)
        car.vx += bx * cfg.boost_accel * dt
        car.vy += by * cfg.boost_accel * dt
        speed = math.hypot(car.vx, car.vy)
        if speed > cfg.max_boost_speed:
            scale = cfg.max_boost_speed / speed
            car.vx *= scale
            car.vy *= scale

    if walls:
        return collision.advance(car, dt, walls, cfg)
    car.px += car.vx * dt
    car.py += car.vy * dt
    return None
