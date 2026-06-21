"""Ordered checkpoint gates: directed line-segment crossing.

A ``Gate`` is a wall-like line segment plus a **forward normal**. A gate is
"passed" only when the control-step move segment ``prev -> curr`` crosses it
*within the gate's extent* and *in the +normal direction* — the same swept idea as
collision (the sim already owns ``prev_px/prev_py``), so a gate can't be skipped by
flying over it between frames, and crossing it backwards never counts.

Pure geometry: no pygame, no wall-clock, no ``dt``. The lap/timer state machine that
*uses* these crossings lives in ``timing.py``; together they are the run/session
layer: this progress stays out of the physics ``Car`` state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class Gate:
    """A directed gate. ``(nx, ny)`` is the unit forward normal; a crossing only
    counts when the car moves across the segment in that direction.

    Build with :meth:`from_endpoints`, which fixes the normal by a documented rule
    from the endpoint order, so tracks author a gate exactly like a wall (four
    numbers) and choose the direction by which endpoint comes first.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    nx: float  # unit forward normal x
    ny: float  # unit forward normal y

    @classmethod
    def from_endpoints(cls, x1: float, y1: float, x2: float, y2: float) -> "Gate":
        """Forward normal = the segment tangent ``(x1,y1)->(x2,y2)`` rotated +90deg,
        i.e. ``n = (-uy, ux)``. Order the endpoints so this points the way the car
        should travel through the gate. Raises on a zero-length segment."""
        ex, ey = x2 - x1, y2 - y1
        length = math.hypot(ex, ey)
        if length < _EPS:
            raise ValueError("gate segment is zero-length")
        ux, uy = ex / length, ey / length
        return cls(x1, y1, x2, y2, nx=-uy, ny=ux)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def crossing(self, x0: float, y0: float, x1: float, y1: float) -> int:
        """Classify the move segment ``(x0,y0)->(x1,y1)`` against this gate.

        Returns ``+1`` for a forward crossing (in the +normal direction), ``-1`` for
        a reverse crossing, and ``0`` if the move does not cross the gate within its
        extent (parallel motion, or it crosses the infinite line off the segment's
        ends). Starting exactly on the line counts as not-yet-crossed.
        """
        return self.crossing_with_fraction(x0, y0, x1, y1)[0]

    def crossing_with_fraction(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, float]:
        """Like :meth:`crossing`, but also return ``t`` — the fraction along the move
        segment at which it meets the gate line.

        ``t`` lies in ``(0, 1]`` and is meaningful only when the returned direction is
        non-zero (a non-crossing returns ``(0, 0.0)``). It is the sub-tick crossing
        position the timing layer interpolates lap time with; the boolean lap state
        machine keeps using :meth:`crossing`, so the ``crossing() -> int`` contract is
        unchanged.
        """
        nx, ny = self.nx, self.ny
        # Signed distances of the two endpoints from the gate's infinite line.
        s0 = (x0 - self.x1) * nx + (y0 - self.y1) * ny
        s1 = (x1 - self.x1) * nx + (y1 - self.y1) * ny
        if s0 < 0.0 <= s1:
            direction = 1
        elif s1 < 0.0 <= s0:
            direction = -1
        else:
            return 0, 0.0  # same side (or grazing): no through-crossing

        # Fraction of the move where it meets the line, then verify the contact
        # point lies on the finite gate segment (0 <= proj <= length).
        t = s0 / (s0 - s1)  # denominator non-zero: s0, s1 straddle the line
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        ex, ey = self.x2 - self.x1, self.y2 - self.y1
        elen = math.hypot(ex, ey)
        proj = ((cx - self.x1) * ex + (cy - self.y1) * ey) / elen
        if -_EPS <= proj <= elen + _EPS:
            return direction, t
        return 0, 0.0
