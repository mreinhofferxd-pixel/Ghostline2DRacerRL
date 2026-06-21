"""Replay recording and playback for deterministic action streams.

The canonical replay is the seed + initial state + control-rate ``Action`` stream.
Frames are recorded alongside as a derived cache for cheap ghost rendering later;
they are not the source of truth. This module lives outside ``core/`` because it
serializes data and constructs playback sims, while the sim itself stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import PHYSICS_VERSION, TIMING_VERSION
from ..config import CAR, CONTROL_DT, CONTROL_HZ, CarPhysics
from ..core.action import Action
from ..core.car import Car
from ..core.sim import Simulation, World
from ..core.timing import RunState
from ..core.track import Track
from ..physics_identity import physics_config_fingerprint, physics_config_payload


class ReplayError(ValueError):
    """A replay file is malformed or incompatible with the requested playback."""


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{field}: expected a number, got {value!r}")
    return float(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(f"{field}: expected an integer, got {value!r}")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReplayError(f"{field}: expected a boolean, got {value!r}")
    return value


def _str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError(f"{field}: expected a non-empty string, got {value!r}")
    return value


def _optional_physics_config(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReplayError(f"physics_config: expected an object, got {value!r}")
    payload: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ReplayError(f"physics_config: expected string keys, got {key!r}")
        payload[key] = _number(raw, f"physics_config.{key}")
    return payload


def _action_to_json(action: Action) -> list[float | bool]:
    throttle, brake, steer, drift = action.clamped().as_tuple()
    return [throttle, brake, steer, drift]


def action_from_json(raw: Any, *, field: str = "action") -> Action:
    """Parse one serialized action tuple/list and clamp it to the input contract."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ReplayError(f"{field}: expected [throttle, brake, steer, drift], got {raw!r}")
    return Action(
        throttle=_number(raw[0], f"{field}[0]"),
        brake=_number(raw[1], f"{field}[1]"),
        steer=_number(raw[2], f"{field}[2]"),
        drift=_bool(raw[3], f"{field}[3]"),
    ).clamped()


@dataclass(frozen=True)
class InitialState:
    """Serializable dynamic state captured before the first recorded action."""

    car: tuple[float, float, float, float, float]
    prev: tuple[float, float]
    tick: int
    sim_time: float
    wall_hits: int
    wall_scrape_time: float
    largest_impact_speed: float
    in_wall_contact: bool
    drift_time: float
    peak_slip: float
    boost_time: float
    boost_cooldowns: tuple[float, ...]
    boosts_used: int
    run_started: bool
    run_start_tick: int
    run_next_cp: int
    run_finished: bool
    run_valid: bool
    run_lap_ticks: int
    run_cp_ticks: tuple[int, ...]
    path_distance: float = 0.0

    @classmethod
    def from_world(cls, world) -> "InitialState":
        car = world.car
        run = world.run
        return cls(
            car=(car.px, car.py, car.vx, car.vy, car.heading),
            prev=(world.prev_px, world.prev_py),
            tick=world.tick,
            sim_time=world.sim_time,
            wall_hits=world.wall_hits,
            wall_scrape_time=world.wall_scrape_time,
            largest_impact_speed=world.largest_impact_speed,
            in_wall_contact=world.in_wall_contact,
            path_distance=world.path_distance,
            drift_time=world.drift_time,
            peak_slip=world.peak_slip,
            boost_time=world.boost_time,
            boost_cooldowns=world.boost_cooldowns,
            boosts_used=world.boosts_used,
            run_started=run.started,
            run_start_tick=run.start_tick,
            run_next_cp=run.next_cp,
            run_finished=run.finished,
            run_valid=run.valid,
            run_lap_ticks=run.lap_ticks,
            run_cp_ticks=run.cp_ticks,
        )

    def apply_to(self, sim: Simulation) -> None:
        """Restore this state into a sim that has already been reset with its track."""
        x, y, vx, vy, heading = self.car
        px, py = self.prev
        world = sim.world
        world.car = Car(px=x, py=y, vx=vx, vy=vy, heading=heading)
        world.prev_px = px
        world.prev_py = py
        world.tick = self.tick
        world.sim_time = self.sim_time
        world.wall_hits = self.wall_hits
        world.wall_scrape_time = self.wall_scrape_time
        world.largest_impact_speed = self.largest_impact_speed
        world.in_wall_contact = self.in_wall_contact
        world.path_distance = self.path_distance
        world.drift_time = self.drift_time
        world.peak_slip = self.peak_slip
        world.boost_time = self.boost_time
        world.boost_cooldowns = self.boost_cooldowns
        world.boosts_used = self.boosts_used
        world.run = RunState(
            started=self.run_started,
            start_tick=self.run_start_tick,
            next_cp=self.run_next_cp,
            finished=self.run_finished,
            valid=self.run_valid,
            lap_ticks=self.run_lap_ticks,
            cp_ticks=self.run_cp_ticks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "car": {
                "x": self.car[0],
                "y": self.car[1],
                "vx": self.car[2],
                "vy": self.car[3],
                "heading": self.car[4],
            },
            "prev": [self.prev[0], self.prev[1]],
            "tick": self.tick,
            "sim_time": self.sim_time,
            "wall": {
                "hits": self.wall_hits,
                "scrape_time": self.wall_scrape_time,
                "largest_impact_speed": self.largest_impact_speed,
                "in_contact": self.in_wall_contact,
            },
            "path_distance": self.path_distance,
            "drift_time": self.drift_time,
            "peak_slip": self.peak_slip,
            "boost": {
                "time": self.boost_time,
                "cooldowns": list(self.boost_cooldowns),
                "used": self.boosts_used,
            },
            "run": {
                "started": self.run_started,
                "start_tick": self.run_start_tick,
                "next_cp": self.run_next_cp,
                "finished": self.run_finished,
                "valid": self.run_valid,
                "lap_ticks": self.run_lap_ticks,
                "cp_ticks": list(self.run_cp_ticks),
            },
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "InitialState":
        if not isinstance(raw, dict):
            raise ReplayError("initial_state: expected an object")
        car = raw.get("car")
        if not isinstance(car, dict):
            raise ReplayError("initial_state.car: expected an object")
        prev = raw.get("prev")
        if not isinstance(prev, list) or len(prev) != 2:
            raise ReplayError("initial_state.prev: expected [x, y]")
        wall = raw.get("wall")
        if not isinstance(wall, dict):
            raise ReplayError("initial_state.wall: expected an object")
        boost = raw.get("boost")
        if not isinstance(boost, dict):
            raise ReplayError("initial_state.boost: expected an object")
        run = raw.get("run")
        if not isinstance(run, dict):
            raise ReplayError("initial_state.run: expected an object")

        cooldowns = boost.get("cooldowns", [])
        if not isinstance(cooldowns, list):
            raise ReplayError("initial_state.boost.cooldowns: expected a list")
        cp_ticks = run.get("cp_ticks", [])
        if not isinstance(cp_ticks, list):
            raise ReplayError("initial_state.run.cp_ticks: expected a list")

        return cls(
            car=(
                _number(car.get("x"), "initial_state.car.x"),
                _number(car.get("y"), "initial_state.car.y"),
                _number(car.get("vx"), "initial_state.car.vx"),
                _number(car.get("vy"), "initial_state.car.vy"),
                _number(car.get("heading"), "initial_state.car.heading"),
            ),
            prev=(
                _number(prev[0], "initial_state.prev[0]"),
                _number(prev[1], "initial_state.prev[1]"),
            ),
            tick=_int(raw.get("tick"), "initial_state.tick"),
            sim_time=_number(raw.get("sim_time"), "initial_state.sim_time"),
            wall_hits=_int(wall.get("hits"), "initial_state.wall.hits"),
            wall_scrape_time=_number(
                wall.get("scrape_time"), "initial_state.wall.scrape_time"
            ),
            largest_impact_speed=_number(
                wall.get("largest_impact_speed"),
                "initial_state.wall.largest_impact_speed",
            ),
            in_wall_contact=_bool(wall.get("in_contact"), "initial_state.wall.in_contact"),
            path_distance=_number(raw.get("path_distance", 0.0), "initial_state.path_distance"),
            drift_time=_number(raw.get("drift_time"), "initial_state.drift_time"),
            peak_slip=_number(raw.get("peak_slip"), "initial_state.peak_slip"),
            boost_time=_number(boost.get("time"), "initial_state.boost.time"),
            boost_cooldowns=tuple(
                _number(v, f"initial_state.boost.cooldowns[{i}]")
                for i, v in enumerate(cooldowns)
            ),
            boosts_used=_int(boost.get("used"), "initial_state.boost.used"),
            run_started=_bool(run.get("started"), "initial_state.run.started"),
            run_start_tick=_int(run.get("start_tick"), "initial_state.run.start_tick"),
            run_next_cp=_int(run.get("next_cp"), "initial_state.run.next_cp"),
            run_finished=_bool(run.get("finished"), "initial_state.run.finished"),
            run_valid=_bool(run.get("valid"), "initial_state.run.valid"),
            run_lap_ticks=_int(run.get("lap_ticks"), "initial_state.run.lap_ticks"),
            run_cp_ticks=tuple(
                _int(v, f"initial_state.run.cp_ticks[{i}]")
                for i, v in enumerate(cp_ticks)
            ),
        )


@dataclass(frozen=True)
class Frame:
    """One derived control-rate pose sample for ghost rendering/validation."""

    t: float
    x: float
    y: float
    angle: float
    speed: float
    drift: bool
    cp: int
    wall: bool
    boost: bool

    @classmethod
    def from_world(cls, world, action: Action, cfg: CarPhysics) -> "Frame":
        car = world.car
        return cls(
            t=world.sim_time,
            x=car.px,
            y=car.py,
            angle=car.heading,
            speed=car.speed,
            drift=bool(action.drift) and car.speed >= cfg.drift_min_speed,
            cp=world.run.next_cp,
            wall=world.in_wall_contact,
            boost=world.boost_active,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "speed": self.speed,
            "drift": self.drift,
            "cp": self.cp,
            "wall": self.wall,
            "boost": self.boost,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Frame":
        if not isinstance(raw, dict):
            raise ReplayError("frame: expected an object")
        return cls(
            t=_number(raw.get("t"), "frame.t"),
            x=_number(raw.get("x"), "frame.x"),
            y=_number(raw.get("y"), "frame.y"),
            angle=_number(raw.get("angle"), "frame.angle"),
            speed=_number(raw.get("speed"), "frame.speed"),
            drift=_bool(raw.get("drift"), "frame.drift"),
            cp=_int(raw.get("cp"), "frame.cp"),
            wall=_bool(raw.get("wall"), "frame.wall"),
            boost=_bool(raw.get("boost"), "frame.boost"),
        )


@dataclass(frozen=True)
class ReplayData:
    """A complete replay document, ready to serialize as JSON."""

    track_id: str
    physics_version: str
    seed: int | None
    initial_state: InitialState
    lap_time: float
    valid: bool
    physics_config: dict[str, float] | None = None
    physics_fingerprint: str | None = None
    actions: tuple[Action, ...] = ()
    frames: tuple[Frame, ...] = ()
    control_hz: int = CONTROL_HZ
    schema_version: int = 2
    # Sub-tick lap-time marker (B9): a fresh replay's ``lap_time`` is the interpolated
    # finish-line crossing time. Legacy replays predate this and read back as
    # "timing_v1" (their stored ``lap_time`` is rounded up to the whole closing tick).
    timing_version: str = TIMING_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "physics_version": self.physics_version,
            "timing_version": self.timing_version,
            "control_hz": self.control_hz,
            "seed": self.seed,
            "initial_state": self.initial_state.to_dict(),
            "lap_time": self.lap_time,
            "valid": self.valid,
            "actions": [_action_to_json(action) for action in self.actions],
            "frames": [frame.to_dict() for frame in self.frames],
        }
        if self.physics_config is not None:
            payload["physics_config"] = {
                key: self.physics_config[key] for key in sorted(self.physics_config)
            }
        if self.physics_fingerprint is not None:
            payload["physics_fingerprint"] = self.physics_fingerprint
        return payload

    @classmethod
    def from_dict(cls, raw: Any) -> "ReplayData":
        if not isinstance(raw, dict):
            raise ReplayError("replay: expected an object")
        seed = raw.get("seed")
        if seed is not None:
            seed = _int(seed, "seed")
        actions = raw.get("actions", [])
        frames = raw.get("frames", [])
        if not isinstance(actions, list):
            raise ReplayError("actions: expected a list")
        if not isinstance(frames, list):
            raise ReplayError("frames: expected a list")
        physics_fingerprint = raw.get("physics_fingerprint")
        if physics_fingerprint is not None:
            physics_fingerprint = _str(physics_fingerprint, "physics_fingerprint")
        return cls(
            schema_version=_int(raw.get("schema_version", 1), "schema_version"),
            track_id=_str(raw.get("track_id"), "track_id"),
            physics_version=_str(raw.get("physics_version"), "physics_version"),
            # Replays written before sub-tick timing carry no marker -> timing_v1.
            timing_version=_str(raw.get("timing_version", "timing_v1"), "timing_version"),
            physics_config=_optional_physics_config(raw.get("physics_config")),
            physics_fingerprint=physics_fingerprint,
            control_hz=_int(raw.get("control_hz", CONTROL_HZ), "control_hz"),
            seed=seed,
            initial_state=InitialState.from_dict(raw.get("initial_state")),
            lap_time=_number(raw.get("lap_time"), "lap_time"),
            valid=_bool(raw.get("valid"), "valid"),
            actions=tuple(
                action_from_json(action, field=f"actions[{i}]")
                for i, action in enumerate(actions)
            ),
            frames=tuple(Frame.from_dict(frame) for frame in frames),
        )


@dataclass
class ReplayRecorder:
    """Collect actions and derived frames while an existing sim is stepped."""

    track_id: str
    physics_version: str
    physics_config: dict[str, float]
    physics_fingerprint: str
    seed: int | None
    initial_state: InitialState
    actions: list[Action] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)

    @classmethod
    def start(cls, sim: Simulation, *, physics_version: str = PHYSICS_VERSION) -> "ReplayRecorder":
        return cls(
            track_id=sim.world.track.track_id,
            physics_version=physics_version,
            physics_config=physics_config_payload(sim.cfg),
            physics_fingerprint=physics_config_fingerprint(sim.cfg),
            seed=sim.seed,
            initial_state=InitialState.from_world(sim.world),
        )

    def record_after_step(self, sim: Simulation, action: Action) -> None:
        action = action.clamped()
        self.actions.append(action)
        self.frames.append(Frame.from_world(sim.world, action, sim.cfg))

    def step(self, sim: Simulation, action: Action):
        """Advance ``sim`` by one recorded control step."""
        action = action.clamped()
        world = sim.step(action)
        self.record_after_step(sim, action)
        return world

    def to_replay(self, sim: Simulation) -> ReplayData:
        run = sim.world.run
        return ReplayData(
            track_id=self.track_id,
            physics_version=self.physics_version,
            physics_config=self.physics_config,
            physics_fingerprint=self.physics_fingerprint,
            seed=self.seed,
            initial_state=self.initial_state,
            lap_time=run.lap_time(sim.world.tick, CONTROL_DT),
            valid=run.valid,
            actions=tuple(self.actions),
            frames=tuple(self.frames),
        )


@dataclass(frozen=True)
class ReplayResult:
    frames: tuple[Frame, ...]
    final_hash: str
    world: World


def frames_match(a: Frame, b: Frame, *, tolerance: float = 1e-9) -> bool:
    return (
        abs(a.t - b.t) <= tolerance
        and abs(a.x - b.x) <= tolerance
        and abs(a.y - b.y) <= tolerance
        and abs(a.angle - b.angle) <= tolerance
        and abs(a.speed - b.speed) <= tolerance
        and a.drift == b.drift
        and a.cp == b.cp
        and a.wall == b.wall
        and a.boost == b.boost
    )


def play_replay(
    replay: ReplayData,
    track: Track,
    cfg: CarPhysics = CAR,
    *,
    strict_version: bool = True,
) -> ReplayResult:
    """Replay the canonical action stream against ``track`` and return new frames."""
    if replay.control_hz != CONTROL_HZ:
        raise ReplayError(
            f"control_hz mismatch: replay={replay.control_hz}, sim={CONTROL_HZ}"
        )
    if track.track_id != replay.track_id:
        raise ReplayError(f"track mismatch: replay={replay.track_id}, track={track.track_id}")
    if strict_version and replay.physics_version != PHYSICS_VERSION:
        raise ReplayError(
            f"physics_version mismatch: replay={replay.physics_version}, current={PHYSICS_VERSION}"
        )
    if strict_version:
        if replay.physics_fingerprint is None:
            raise ReplayError("physics_fingerprint missing: legacy replay is incompatible")
        current_fingerprint = physics_config_fingerprint(cfg)
        if replay.physics_fingerprint != current_fingerprint:
            raise ReplayError(
                "physics_fingerprint mismatch: "
                f"replay={replay.physics_fingerprint}, current={current_fingerprint}"
            )

    sim = Simulation(cfg)
    sim.reset(track=track, seed=replay.seed)
    replay.initial_state.apply_to(sim)

    frames: list[Frame] = []
    for action in replay.actions:
        sim.step(action)
        frames.append(Frame.from_world(sim.world, action, sim.cfg))
    return ReplayResult(frames=tuple(frames), final_hash=sim.state_hash(), world=sim.snapshot())


def trajectory_matches(
    replay: ReplayData,
    track: Track,
    cfg: CarPhysics = CAR,
    *,
    tolerance: float = 1e-9,
    strict_version: bool = True,
) -> bool:
    """Return True when replayed actions reproduce every stored derived frame."""
    result = play_replay(replay, track, cfg, strict_version=strict_version)
    if len(result.frames) != len(replay.frames):
        return False
    return all(
        frames_match(expected, actual, tolerance=tolerance)
        for expected, actual in zip(replay.frames, result.frames)
    )
