"""The Action seam.

Every input path — keyboard now, an RL policy later — produces an ``Action``.
The simulation knows nothing else about where input comes from.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class Action:
    throttle: float = 0.0  # [0, 1]
    brake: float = 0.0  # [0, 1]  (also reverse from a standstill)
    steer: float = 0.0  # [-1, 1] (negative = left, positive = right)
    drift: bool = False

    def clamped(self) -> "Action":
        """Return a copy with continuous axes clamped to valid ranges."""
        return Action(
            throttle=_clamp(self.throttle, 0.0, 1.0),
            brake=_clamp(self.brake, 0.0, 1.0),
            steer=_clamp(self.steer, -1.0, 1.0),
            drift=bool(self.drift),
        )

    def as_tuple(self) -> tuple[float, float, float, bool]:
        """Serialization form for the replay action stream."""
        return (self.throttle, self.brake, self.steer, self.drift)


NEUTRAL = Action()
