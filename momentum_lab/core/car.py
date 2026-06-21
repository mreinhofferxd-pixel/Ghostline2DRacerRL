"""Car state — the snapshot-able physics vector.

Deliberately minimal: position, velocity, heading. Timing and checkpoint progress
live in the run/session layer, not here, so this stays a clean copyable state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class Car:
    px: float = 0.0
    py: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0  # radians; 0 = +x (right)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def forward_speed(self) -> float:
        """Signed speed along the heading direction (negative when reversing)."""
        return self.vx * math.cos(self.heading) + self.vy * math.sin(self.heading)

    @property
    def slip_angle(self) -> float:
        """Signed angle between heading and velocity (the drift angle), radians.

        Descriptive only: read off the sim for HUD/metrics; it never drives the
        physics.
        """
        if self.speed < 1e-6:
            return 0.0
        vel_angle = math.atan2(self.vy, self.vx)
        d = (vel_angle - self.heading + math.pi) % (2.0 * math.pi) - math.pi
        return d

    def copy(self) -> "Car":
        return replace(self)
