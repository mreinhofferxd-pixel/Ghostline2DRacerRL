"""The deterministic, fixed-timestep simulation.

``step(action)`` advances exactly one control step (``CONTROL_DT`` of sim time)
by running ``SUBSTEPS`` physics integrations with the action held constant. There
is no ``dt`` argument, no wall-clock, and no rendering — so the entire trajectory
is reproducible from ``(seed, initial state, action stream)``.

The ``snapshot``/``restore``/``state_hash`` trio is both the render-interpolation
mechanism and the basis of the determinism tests, and is exactly what a future RL
environment needs for resets and branching.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

from ..config import CAR, CONTROL_DT, PHYSICS_DT, SUBSTEPS, CarPhysics
from . import collision, physics
from .action import NEUTRAL, Action
from .car import Car
from .timing import RunState
from .track import EMPTY_TRACK, Track


@dataclass
class World:
    """Everything the sim owns. Grows with checkpoints/pads in later milestones."""

    car: Car
    track: Track = EMPTY_TRACK  # immutable static geometry (walls); shared by copy()
    prev_px: float = 0.0  # position before the last control step (for swept tests)
    prev_py: float = 0.0
    tick: int = 0  # control-step counter; the single source of sim time
    sim_time: float = 0.0  # tick * CONTROL_DT
    # Wall-interaction stats: deterministic, exposed for HUD/metrics/RL.
    wall_hits: int = 0
    wall_scrape_time: float = 0.0
    largest_impact_speed: float = 0.0
    in_wall_contact: bool = False  # edge-detect so a sustained crash counts once
    path_distance: float = 0.0  # descriptive lap-efficiency metric; never affects physics
    # Drift-feedback metrics: descriptive, tick-driven, and
    # snapshot-able — read off the sim for the HUD/metrics, never fed back into physics.
    drift_time: float = 0.0  # seconds spent drifting (handbrake on, above drift_min_speed)
    peak_slip: float = 0.0  # largest |slip angle| reached this run (radians)
    # Boost-pad runtime state (M5). Pad geometry is immutable Track data; these
    # timers are run state because they affect future physics and must branch.
    boost_time: float = 0.0  # remaining active boost duration, seconds
    boost_cooldowns: tuple[float, ...] = field(default_factory=tuple)  # per-pad seconds
    boosts_used: int = 0  # descriptive run metric; does not feed physics back
    # Lap/checkpoint progress: run-layer, tick-driven, snapshot-able.
    run: RunState = field(default_factory=RunState)

    @property
    def boost_active(self) -> bool:
        return self.boost_time > 0.0

    def copy(self) -> "World":
        return World(
            car=self.car.copy(),
            track=self.track,  # immutable: share the reference, don't deep-copy
            prev_px=self.prev_px,
            prev_py=self.prev_py,
            tick=self.tick,
            sim_time=self.sim_time,
            wall_hits=self.wall_hits,
            wall_scrape_time=self.wall_scrape_time,
            largest_impact_speed=self.largest_impact_speed,
            in_wall_contact=self.in_wall_contact,
            path_distance=self.path_distance,
            drift_time=self.drift_time,
            peak_slip=self.peak_slip,
            boost_time=self.boost_time,
            boost_cooldowns=self.boost_cooldowns,
            boosts_used=self.boosts_used,
            run=self.run.copy(),
        )

    def register_contact(self, contact: collision.Contact | None, dt: float) -> None:
        """Fold one substep's wall contact into the run stats."""
        if contact is None:
            self.in_wall_contact = False
            return
        if contact.max_normal_speed > self.largest_impact_speed:
            self.largest_impact_speed = contact.max_normal_speed
        if contact.is_impact and not self.in_wall_contact:
            self.wall_hits += 1  # rising edge of a hard contact: one crash = one hit
        else:
            self.wall_scrape_time += dt  # sustained/glancing contact = scraping time
        self.in_wall_contact = True


class Simulation:
    def __init__(self, cfg: CarPhysics = CAR) -> None:
        self.cfg = cfg
        self._seed: int | None = None
        self.world: World = World(car=Car())
        self.reset()

    @property
    def seed(self) -> int | None:
        """Seed from the latest reset; recorded with action-stream replays."""
        return self._seed

    def reset(
        self,
        track: Track | None = None,
        spawn: tuple[float, float] | None = None,
        heading: float | None = None,
        seed: int | None = None,
    ) -> World:
        """Deterministically (re)initialize the world.

        With a ``track``, the spawn pose and walls come from it (``spawn``/``heading``
        still override if given). With no track, the world has no walls — identical
        to the Milestone-1 behavior — so the determinism tests are unaffected.
        """
        self._seed = seed
        tr = track if track is not None else EMPTY_TRACK
        sx, sy = spawn if spawn is not None else tr.spawn
        hd = heading if heading is not None else tr.spawn_heading
        car = Car(px=sx, py=sy, heading=hd)
        self.world = World(
            car=car,
            track=tr,
            prev_px=sx,
            prev_py=sy,
            boost_cooldowns=tuple(0.0 for _ in tr.boost_pads),
        )
        return self.world

    def _trigger_boost_pads(self) -> None:
        pads = self.world.track.boost_pads
        if not pads:
            return
        cooldowns = list(self.world.boost_cooldowns)
        changed = False
        if len(cooldowns) != len(pads):
            cooldowns = [0.0 for _ in pads]
            changed = True
        car = self.world.car
        for i, pad in enumerate(pads):
            if cooldowns[i] <= 0.0 and pad.overlaps_circle(car.px, car.py, self.cfg.radius):
                cooldowns[i] = self.cfg.boost_pad_cooldown
                self.world.boost_time = max(self.world.boost_time, self.cfg.boost_duration)
                self.world.boosts_used += 1
                changed = True
        if changed:
            self.world.boost_cooldowns = tuple(cooldowns)

    def _tick_boost_timers(self, dt: float) -> None:
        if self.world.boost_time > 0.0:
            self.world.boost_time = max(0.0, self.world.boost_time - dt)
        if self.world.boost_cooldowns:
            self.world.boost_cooldowns = tuple(
                max(0.0, cooldown - dt) for cooldown in self.world.boost_cooldowns
            )

    def step(self, action: Action = NEUTRAL) -> World:
        """Advance one control step. Action is clamped then held across substeps."""
        action = action.clamped()
        car = self.world.car
        self.world.prev_px = car.px
        self.world.prev_py = car.py
        walls = self.world.track.walls
        count_path_distance = self.world.run.started or action.throttle > 0.0
        for _ in range(SUBSTEPS):
            before_px, before_py = car.px, car.py
            self._trigger_boost_pads()
            contact = physics.integrate(
                car,
                action,
                PHYSICS_DT,
                self.cfg,
                walls,
                boost_active=self.world.boost_active,
            )
            if count_path_distance:
                self.world.path_distance += math.hypot(car.px - before_px, car.py - before_py)
            self.world.register_contact(contact, PHYSICS_DT)
            self._tick_boost_timers(PHYSICS_DT)
        self.world.tick += 1
        self.world.sim_time = self.world.tick * CONTROL_DT
        # Lap/checkpoint progress over the control-step move segment. The
        # timer starts on the first non-zero throttle; gate crossings are directional
        # and ordered. This reads positions/action only — it never affects physics.
        self.world.run.update(
            throttle_on=action.throttle > 0.0,
            prev=(self.world.prev_px, self.world.prev_py),
            curr=(car.px, car.py),
            checkpoints=self.world.track.checkpoints,
            finish=self.world.track.finish,
            tick=self.world.tick,
        )
        # Drift-feedback metrics (descriptive only; never affects physics). Counted at
        # control-step granularity on the same predicate the physics substeps use
        # (handbrake engaged + above drift_min_speed).
        if action.drift and car.speed >= self.cfg.drift_min_speed:
            self.world.drift_time += CONTROL_DT
        slip = abs(car.slip_angle)
        if slip > self.world.peak_slip:
            self.world.peak_slip = slip
        return self.world

    # --- determinism / RL plumbing -------------------------------------------
    def snapshot(self) -> World:
        return self.world.copy()

    def restore(self, snap: World) -> None:
        self.world = snap.copy()

    def state_hash(self) -> str:
        """Stable digest of physics-affecting state, for determinism tests."""
        c = self.world.car
        data = bytearray(
            struct.pack(
                "<6dI",
                c.px,
                c.py,
                c.vx,
                c.vy,
                c.heading,
                self.world.boost_time,
                len(self.world.boost_cooldowns),
            )
        )
        for cooldown in self.world.boost_cooldowns:
            data.extend(struct.pack("<d", cooldown))
        return hashlib.sha256(data).hexdigest()[:16]
