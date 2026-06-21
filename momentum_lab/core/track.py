"""Track data the sim owns: spawn pose, walls, boost pads, checkpoints + finish.

Pure, immutable static geometry: no pygame, no file I/O. The JSON loader that
*builds* a Track lives at the boundary in ``momentum_lab/tracks/`` so that ``core/``
never reads files. The sim holds a Track by reference and shares it across
``snapshot``/``restore`` (it never mutates).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoints import Gate
from .collision import Segment


@dataclass(frozen=True)
class BoostPad:
    """An axis-aligned boost hitbox in world units. Immutable static geometry."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        return left, top, right, bottom

    @property
    def center(self) -> tuple[float, float]:
        left, top, right, bottom = self.bounds
        return (0.5 * (left + right), 0.5 * (top + bottom))

    def overlaps_circle(self, px: float, py: float, radius: float) -> bool:
        """True if the car's collision circle touches the pad rectangle."""
        left, top, right, bottom = self.bounds
        cx = min(max(px, left), right)
        cy = min(max(py, top), bottom)
        dx, dy = px - cx, py - cy
        return dx * dx + dy * dy <= radius * radius


@dataclass(frozen=True)
class Track:
    track_id: str
    spawn: tuple[float, float]
    spawn_heading: float
    walls: tuple[Segment, ...] = field(default_factory=tuple)
    # Lap layout (M3): ordered checkpoint gates the car clears in sequence, then the
    # finish line closes the lap. A track may have neither (a free-drive sandbox).
    checkpoints: tuple[Gate, ...] = field(default_factory=tuple)
    finish: Gate | None = None
    # Boost pads (M5): immutable hitboxes. Runtime cooldown/active timers live in
    # World, because they are run state and must snapshot/restore.
    boost_pads: tuple[BoostPad, ...] = field(default_factory=tuple)
    # --- Non-physics authoring hints (NEVER read by the sim/physics) -----------
    # Closed boundary loops (outer ring + inner infield) used only to FILL the track
    # surface in the renderer, so the track reads as asphalt + infield instead of a
    # wireframe. Physics uses `walls`; these are presentation. Each is a tuple of
    # (x, y) world-space points.
    surface_outer: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    surface_inner: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # Authored centerline / ideal line. Used by the scripted pure-pursuit eval
    # (eval/harness.time_trial) and as an optional faint guide in debug; it does not
    # feed physics, timing, or determinism.
    racing_line: tuple[tuple[float, float], ...] = field(default_factory=tuple)


EMPTY_TRACK = Track(track_id="<empty>", spawn=(200.0, 360.0), spawn_heading=0.0, walls=())
