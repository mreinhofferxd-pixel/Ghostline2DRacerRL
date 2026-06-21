"""Versioned reward calculation for the RL wrapper."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ..config import CAR, CONTROL_DT, WORLD_HEIGHT, WORLD_WIDTH
from ..core import collision
from ..core.sim import World
from .observations import target_gate


REWARD_VERSION = "reward_v1"
_WORLD_DIAGONAL = math.hypot(WORLD_WIDTH, WORLD_HEIGHT)


@dataclass(frozen=True)
class RewardConfig:
    version: str = REWARD_VERSION
    progress_scale: float = 2.0
    checkpoint_bonus: float = 5.0
    finish_bonus: float = 25.0
    time_penalty: float = -0.01
    wall_hit_penalty: float = -2.0
    wall_scrape_penalty_per_second: float = -0.5
    wall_proximity_penalty: float = 0.0
    wall_proximity_threshold: float = 0.0
    target_speed_scale: float = 0.0
    heading_alignment_scale: float = 0.0
    # Direct lap-time reward, paid once when the lap closes: the agent earns
    # ``finish_time_bonus_scale`` per control step the lap came in *under*
    # ``finish_time_reference_steps``. This optimizes the real objective (total lap
    # time) rather than relying only on the per-step ``time_penalty`` proxy, which is
    # a weak, noisy gradient. ``finish_time_reference_steps == 0`` disables it.
    finish_time_bonus_scale: float = 0.0
    finish_time_reference_steps: float = 0.0
    # Finish-gated efficiency terms. These normalize around reference values from
    # the current best laps, rewarding the sweet spot of high average speed and a
    # compact route without making either one dominate lap time.
    avg_speed_bonus_scale: float = 0.0
    avg_speed_reference: float = 0.0
    path_efficiency_bonus_scale: float = 0.0
    path_distance_reference: float = 0.0
    # Per-step drift cost (added in v5): discourages gratuitous handbrake drift that
    # does not buy lap time. Mirrors ``wall_scrape`` -- charged per second of newly
    # accumulated drift time (``world.drift_time``). 0 disables it (v1-v4 behavior).
    # Telemetry (the human vs b7_19 sector trace) showed the policy over-drifts into
    # CP2, running wide and slow; a small cost nudges it toward the human's tighter,
    # less-drifty line without killing drift-and-catch -- physics still rewards a good
    # drift via the carried speed that shortens the lap (and thus the finish-time
    # bonus), so only drift that fails to pay for itself is discouraged.
    drift_penalty_per_second: float = 0.0

    def payload(self) -> dict[str, float | str]:
        data = asdict(self)
        return {key: data[key] for key in sorted(data)}


@dataclass(frozen=True)
class RewardState:
    target: tuple[float, float] | None
    next_cp: int
    finished: bool
    wall_hits: int
    wall_scrape_time: float
    drift_time: float = 0.0

    @classmethod
    def from_world(cls, world: World) -> "RewardState":
        gate, _is_finish = target_gate(world)
        return cls(
            target=None if gate is None else gate.center,
            next_cp=world.run.next_cp,
            finished=world.run.finished,
            wall_hits=world.wall_hits,
            wall_scrape_time=world.wall_scrape_time,
            drift_time=world.drift_time,
        )


@dataclass(frozen=True)
class RewardBreakdown:
    progress: float
    target_speed: float
    heading_alignment: float
    checkpoint: float
    finish: float
    finish_time: float
    avg_speed: float
    path_efficiency: float
    time: float
    wall_hit: float
    wall_scrape: float
    wall_proximity: float
    drift: float

    @property
    def total(self) -> float:
        return (
            self.progress
            + self.target_speed
            + self.heading_alignment
            + self.checkpoint
            + self.finish
            + self.finish_time
            + self.avg_speed
            + self.path_efficiency
            + self.time
            + self.wall_hit
            + self.wall_scrape
            + self.wall_proximity
            + self.drift
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "progress": self.progress,
            "target_speed": self.target_speed,
            "heading_alignment": self.heading_alignment,
            "checkpoint": self.checkpoint,
            "finish": self.finish,
            "finish_time": self.finish_time,
            "avg_speed": self.avg_speed,
            "path_efficiency": self.path_efficiency,
            "time": self.time,
            "wall_hit": self.wall_hit,
            "wall_scrape": self.wall_scrape,
            "wall_proximity": self.wall_proximity,
            "drift": self.drift,
            "total": self.total,
        }


def _distance_to_target(world: World, target: tuple[float, float] | None) -> float:
    if target is None:
        return 0.0
    car = world.car
    return math.hypot(target[0] - car.px, target[1] - car.py)


def _wall_clearance(world: World) -> float:
    """Shortest free-space distance from the car shell to any wall segment."""
    if not world.track.walls:
        return math.inf
    car = world.car
    best = math.inf
    for seg in world.track.walls:
        cx, cy = collision.closest_point_on_segment(
            car.px,
            car.py,
            seg.x1,
            seg.y1,
            seg.x2,
            seg.y2,
        )
        clearance = math.hypot(car.px - cx, car.py - cy) - CAR.radius
        if clearance < best:
            best = clearance
    return best


def _wall_proximity_penalty(world: World, cfg: RewardConfig) -> float:
    if cfg.wall_proximity_threshold <= 0.0 or cfg.wall_proximity_penalty == 0.0:
        return 0.0
    clearance = _wall_clearance(world)
    if clearance >= cfg.wall_proximity_threshold:
        return 0.0
    danger = (cfg.wall_proximity_threshold - max(0.0, clearance)) / cfg.wall_proximity_threshold
    return danger * danger * cfg.wall_proximity_penalty


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def compute_reward(
    before: RewardState,
    before_world: World,
    after_world: World,
    cfg: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    """Reward the latest env step using only sim/run state.

    Distance progress is measured toward the target gate that was active before
    the step, so crossing a checkpoint cannot create a large negative distance
    jump when the target advances to the next gate.
    """
    prev_dist = _distance_to_target(before_world, before.target)
    curr_dist = _distance_to_target(after_world, before.target)
    progress = ((prev_dist - curr_dist) / _WORLD_DIAGONAL) * cfg.progress_scale
    target_speed = 0.0
    heading_alignment = 0.0
    if before.target is not None:
        car = after_world.car
        dx, dy = before.target[0] - car.px, before.target[1] - car.py
        dist = math.hypot(dx, dy)
        if dist > 1e-9:
            ux, uy = dx / dist, dy / dist
            speed_toward_target = (car.vx * ux + car.vy * uy) / CAR.max_boost_speed
            target_speed = speed_toward_target * cfg.target_speed_scale
            heading_alignment = (
                (math.cos(car.heading) * ux + math.sin(car.heading) * uy)
                * cfg.heading_alignment_scale
            )
    cp_delta = max(0, after_world.run.next_cp - before.next_cp)
    checkpoint = cp_delta * cfg.checkpoint_bonus
    just_finished = after_world.run.finished and not before.finished
    finish = cfg.finish_bonus if just_finished else 0.0
    finish_time = 0.0
    if just_finished and cfg.finish_time_reference_steps > 0.0:
        # Paid once on the closing step: reward each control step the lap beat the
        # reference. ``lap_ticks`` is the start->finish control-step count (timing.py).
        steps_saved = cfg.finish_time_reference_steps - after_world.run.lap_ticks
        finish_time = cfg.finish_time_bonus_scale * max(0.0, steps_saved)
    avg_speed = 0.0
    if just_finished and cfg.avg_speed_bonus_scale != 0.0 and cfg.avg_speed_reference > 0.0:
        lap_seconds = max(CONTROL_DT, after_world.run.lap_ticks * CONTROL_DT)
        lap_avg_speed = after_world.path_distance / lap_seconds
        avg_speed = cfg.avg_speed_bonus_scale * _clamp_unit(
            (lap_avg_speed - cfg.avg_speed_reference) / cfg.avg_speed_reference
        )
    path_efficiency = 0.0
    if (
        just_finished
        and cfg.path_efficiency_bonus_scale != 0.0
        and cfg.path_distance_reference > 0.0
    ):
        path_efficiency = cfg.path_efficiency_bonus_scale * _clamp_unit(
            (cfg.path_distance_reference - after_world.path_distance)
            / cfg.path_distance_reference
        )
    wall_hit = max(0, after_world.wall_hits - before.wall_hits) * cfg.wall_hit_penalty
    scrape_delta = max(0.0, after_world.wall_scrape_time - before.wall_scrape_time)
    wall_scrape = scrape_delta * cfg.wall_scrape_penalty_per_second
    wall_proximity = _wall_proximity_penalty(after_world, cfg)
    drift_delta = max(0.0, after_world.drift_time - before.drift_time)
    drift = drift_delta * cfg.drift_penalty_per_second
    return RewardBreakdown(
        progress=progress,
        target_speed=target_speed,
        heading_alignment=heading_alignment,
        checkpoint=checkpoint,
        finish=finish,
        finish_time=finish_time,
        avg_speed=avg_speed,
        path_efficiency=path_efficiency,
        time=cfg.time_penalty,
        wall_hit=wall_hit,
        wall_scrape=wall_scrape,
        wall_proximity=wall_proximity,
        drift=drift,
    )


def reward_info(cfg: RewardConfig, breakdown: RewardBreakdown) -> dict[str, Any]:
    return {
        "reward_version": cfg.version,
        "reward_config": cfg.payload(),
        "reward_breakdown": breakdown.to_dict(),
    }
