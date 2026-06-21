"""Derived-frame ghost playback helpers.

The canonical replay remains the action stream. This module consumes the cached
control-rate frames saved beside that stream so rendering can show a non-colliding
best ghost without re-simulating it every frame.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .recorder import Frame, ReplayData, ReplayError


@dataclass(frozen=True)
class GhostPose:
    """Interpolated ghost pose at one lap-timer instant."""

    t: float
    x: float
    y: float
    angle: float
    speed: float
    drift: bool
    cp: int
    wall: bool
    boost: bool


@dataclass(frozen=True)
class _GhostSample:
    t: float
    frame: Frame


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_angle(a: float, b: float, t: float) -> float:
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * t


def _pose_from_sample(sample: _GhostSample) -> GhostPose:
    f = sample.frame
    return GhostPose(
        t=sample.t,
        x=f.x,
        y=f.y,
        angle=f.angle,
        speed=f.speed,
        drift=f.drift,
        cp=f.cp,
        wall=f.wall,
        boost=f.boost,
    )


def _run_start_tick(replay: ReplayData) -> int | None:
    if replay.initial_state.run_started:
        return replay.initial_state.run_start_tick
    for i, action in enumerate(replay.actions):
        if action.throttle > 0.0:
            return replay.initial_state.tick + i + 1
    return None


class GhostPlayback:
    """Time-aligned, interpolated view of a valid best replay."""

    def __init__(self, replay: ReplayData) -> None:
        if not replay.valid:
            raise ReplayError("ghost: replay is not a valid lap")
        if replay.control_hz <= 0:
            raise ReplayError(f"ghost: invalid control_hz {replay.control_hz}")
        if not replay.frames:
            raise ReplayError("ghost: replay has no derived frames")
        if len(replay.frames) != len(replay.actions):
            raise ReplayError("ghost: replay frame/action count mismatch")
        start_tick = _run_start_tick(replay)
        if start_tick is None:
            raise ReplayError("ghost: replay never starts a timed run")

        control_dt = 1.0 / replay.control_hz
        samples: list[_GhostSample] = []
        for i, frame in enumerate(replay.frames):
            frame_tick = replay.initial_state.tick + i + 1
            lap_t = max(0.0, (frame_tick - start_tick) * control_dt)
            samples.append(_GhostSample(lap_t, frame))

        self.replay = replay
        self.lap_time = replay.lap_time
        self._samples = tuple(samples)
        self._times = tuple(sample.t for sample in samples)
        by_cp: dict[int, list[_GhostSample]] = {}
        for sample in samples:
            by_cp.setdefault(sample.frame.cp, []).append(sample)
        self._samples_by_cp = {cp: tuple(items) for cp, items in by_cp.items()}

    def sample(self, lap_time: float) -> GhostPose:
        """Return the ghost pose interpolated at ``lap_time`` seconds."""
        if lap_time <= self._times[0]:
            return _pose_from_sample(self._samples[0])
        if lap_time >= self._times[-1]:
            return _pose_from_sample(self._samples[-1])

        idx = bisect.bisect_right(self._times, lap_time)
        before = self._samples[idx - 1]
        after = self._samples[idx]
        span = after.t - before.t
        if span <= 0.0:
            return _pose_from_sample(after)
        amount = (lap_time - before.t) / span
        a = before.frame
        b = after.frame
        discrete = b if amount >= 0.5 else a
        return GhostPose(
            t=lap_time,
            x=_lerp(a.x, b.x, amount),
            y=_lerp(a.y, b.y, amount),
            angle=_lerp_angle(a.angle, b.angle, amount),
            speed=_lerp(a.speed, b.speed, amount),
            drift=discrete.drift,
            cp=discrete.cp,
            wall=discrete.wall,
            boost=discrete.boost,
        )

    def delta_to_position(
        self,
        x: float,
        y: float,
        lap_time: float,
        *,
        checkpoint: int | None = None,
    ) -> float:
        """Compare current lap time with the nearest ghost frame on the same stage.

        Negative means the live car reached this part of the route earlier than
        the ghost; positive means it is behind the ghost.
        """
        samples = self._samples
        if checkpoint is not None:
            samples = self._samples_by_cp.get(checkpoint, samples)
        nearest = min(
            samples,
            key=lambda sample: (sample.frame.x - x) ** 2 + (sample.frame.y - y) ** 2,
        )
        return lap_time - nearest.t

    def trail(
        self,
        lap_time: float,
        *,
        seconds: float = 1.0,
        max_points: int = 28,
    ) -> tuple[GhostPose, ...]:
        """Recent sampled ghost poses for a faint render-only trail."""
        start = max(0.0, lap_time - seconds)
        end_idx = bisect.bisect_right(self._times, lap_time)
        start_idx = bisect.bisect_left(self._times, start)
        samples = self._samples[start_idx:end_idx]
        if len(samples) > max_points:
            stride = max(1, math.ceil(len(samples) / max_points))
            samples = samples[::stride]
        return tuple(_pose_from_sample(sample) for sample in samples)
