"""Completed-run metrics derived from sim state and replay frames."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from .. import PHYSICS_VERSION, TIMING_VERSION
from ..config import CAR, CONTROL_DT, CarPhysics
from ..core.sim import World
from ..physics_identity import physics_config_fingerprint, physics_config_payload
from ..replay.recorder import Frame


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RunSummary:
    track_id: str
    lap_time: float
    valid: bool
    wall_hits: int
    boosts_used: int
    drift_time: float
    max_speed: float
    avg_speed: float
    checkpoint_times: tuple[float, ...]
    physics_version: str
    physics_fingerprint: str
    physics_config: dict[str, float]
    timestamp: str
    timing_version: str = TIMING_VERSION

    @classmethod
    def from_world(
        cls,
        world: World,
        frames: Sequence[Frame],
        *,
        physics_version: str = PHYSICS_VERSION,
        physics_cfg: CarPhysics = CAR,
        timestamp: str | None = None,
    ) -> "RunSummary":
        speeds = [frame.speed for frame in frames]
        if speeds:
            max_speed = max(speeds)
            avg_speed = sum(speeds) / len(speeds)
        else:
            max_speed = world.car.speed
            avg_speed = world.car.speed
        return cls(
            track_id=world.track.track_id,
            lap_time=world.run.lap_time(world.tick, CONTROL_DT),
            valid=world.run.valid,
            wall_hits=world.wall_hits,
            boosts_used=world.boosts_used,
            drift_time=world.drift_time,
            max_speed=max_speed,
            avg_speed=avg_speed,
            checkpoint_times=tuple(world.run.split_times(CONTROL_DT)),
            physics_version=physics_version,
            physics_fingerprint=physics_config_fingerprint(physics_cfg),
            physics_config=physics_config_payload(physics_cfg),
            timestamp=timestamp if timestamp is not None else _utc_timestamp(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "lap_time": self.lap_time,
            "valid": self.valid,
            "wall_hits": self.wall_hits,
            "boosts_used": self.boosts_used,
            "drift_time": self.drift_time,
            "max_speed": self.max_speed,
            "avg_speed": self.avg_speed,
            "checkpoint_times": list(self.checkpoint_times),
            "physics_version": self.physics_version,
            "physics_fingerprint": self.physics_fingerprint,
            "physics_config": {
                key: self.physics_config[key] for key in sorted(self.physics_config)
            },
            "timestamp": self.timestamp,
            "timing_version": self.timing_version,
        }
