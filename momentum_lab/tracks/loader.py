"""Load + validate JSON track files into ``core.track.Track`` objects.

Schema (v3, adds M5 boost pads):

    {
      "track_id": "track_01_easy_loop",
      "spawn": [180, 560],
      "spawn_heading": 0.0,
      "walls": [[80, 80, 1200, 80], ...],        // each: [x1, y1, x2, y2]
      "checkpoints": [[1200, 360, 1000, 360], ...], // ordered gates: [x1,y1,x2,y2]
      "finish": [200, 640, 200, 480],            // gate that closes the lap
      "boost_pads": [[520, 520, 700, 600], ...], // axis rects: [x1,y1,x2,y2]
      "surface_outer": [[x, y], ...],            // render-only: outer fill loop
      "surface_inner": [[x, y], ...],            // render-only: infield fill loop
      "racing_line":   [[x, y], ...]             // authoring: ideal line for the eval
    }

The last three keys are optional, non-physics authoring/render hints (see
``core.track.Track``): the sim never reads them. ``tools/build_track_01.py``
generates Track 1 — its geometry, gates, pads, and these hints — in one pass.

A gate's *direction* comes from its endpoint order (forward normal = tangent
rotated +90deg; see ``core.checkpoints.Gate.from_endpoints``): order the points so
the normal points the way the car travels through the gate.

Validation is strict and the errors name the offending field, because a bad track
should fail loudly at load time, not produce a silently broken sim.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.checkpoints import Gate
from ..core.collision import Segment
from ..core.track import BoostPad, Track


class TrackError(ValueError):
    """A track file is missing, malformed, or fails schema validation."""


def tracks_dir() -> Path:
    """The repo's top-level ``tracks/`` directory (sibling of ``momentum_lab/``)."""
    return Path(__file__).resolve().parents[2] / "tracks"


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackError(f"{field}: expected a number, got {value!r}")
    return float(value)


def _parse_walls(raw, *, track_id: str) -> tuple[Segment, ...]:
    if not isinstance(raw, list):
        raise TrackError(f"{track_id}: 'walls' must be a list")
    walls: list[Segment] = []
    for i, w in enumerate(raw):
        if not isinstance(w, (list, tuple)) or len(w) != 4:
            raise TrackError(
                f"{track_id}: walls[{i}] must be [x1, y1, x2, y2], got {w!r}"
            )
        x1, y1, x2, y2 = (_number(v, f"walls[{i}]") for v in w)
        if x1 == x2 and y1 == y2:
            raise TrackError(f"{track_id}: walls[{i}] is zero-length")
        walls.append(Segment(x1, y1, x2, y2))
    return tuple(walls)


def _parse_gate(raw, *, field: str) -> Gate:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise TrackError(f"{field} must be [x1, y1, x2, y2], got {raw!r}")
    x1, y1, x2, y2 = (_number(v, field) for v in raw)
    try:
        return Gate.from_endpoints(x1, y1, x2, y2)
    except ValueError as e:
        raise TrackError(f"{field}: {e}") from e


def _parse_checkpoints(raw, *, track_id: str) -> tuple[Gate, ...]:
    if not isinstance(raw, list):
        raise TrackError(f"{track_id}: 'checkpoints' must be a list")
    return tuple(
        _parse_gate(cp, field=f"{track_id}: checkpoints[{i}]")
        for i, cp in enumerate(raw)
    )


def _parse_point_loop(raw, *, field: str) -> tuple[tuple[float, float], ...]:
    """Parse an optional list of ``[x, y]`` points (a render/authoring hint).

    Lenient on emptiness (the key is optional) but strict on shape: a present value
    must be a list of 2-number points, so a typo fails loudly like the rest of the
    schema rather than silently dropping render geometry.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TrackError(f"{field} must be a list of [x, y] points")
    pts: list[tuple[float, float]] = []
    for i, p in enumerate(raw):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise TrackError(f"{field}[{i}] must be [x, y], got {p!r}")
        pts.append((_number(p[0], f"{field}[{i}]"), _number(p[1], f"{field}[{i}]")))
    return tuple(pts)


def _parse_boost_pads(raw, *, track_id: str) -> tuple[BoostPad, ...]:
    if not isinstance(raw, list):
        raise TrackError(f"{track_id}: 'boost_pads' must be a list")
    pads: list[BoostPad] = []
    for i, pad in enumerate(raw):
        if not isinstance(pad, (list, tuple)) or len(pad) != 4:
            raise TrackError(
                f"{track_id}: boost_pads[{i}] must be [x1, y1, x2, y2], got {pad!r}"
            )
        x1, y1, x2, y2 = (_number(v, f"boost_pads[{i}]") for v in pad)
        if x1 == x2 or y1 == y2:
            raise TrackError(f"{track_id}: boost_pads[{i}] has zero area")
        pads.append(BoostPad(x1, y1, x2, y2))
    return tuple(pads)


def load_track(path: str | Path) -> Track:
    """Read, validate, and build a ``Track`` from a JSON file at ``path``."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise TrackError(f"track file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise TrackError(f"{path}: invalid JSON ({e})") from e

    if not isinstance(data, dict):
        raise TrackError(f"{path}: top level must be a JSON object")

    track_id = data.get("track_id")
    if not isinstance(track_id, str) or not track_id:
        raise TrackError(f"{path}: 'track_id' must be a non-empty string")

    spawn = data.get("spawn")
    if not isinstance(spawn, (list, tuple)) or len(spawn) != 2:
        raise TrackError(f"{track_id}: 'spawn' must be [x, y]")
    spawn_xy = (_number(spawn[0], "spawn[0]"), _number(spawn[1], "spawn[1]"))

    spawn_heading = _number(data.get("spawn_heading", 0.0), "spawn_heading")
    walls = _parse_walls(data.get("walls", []), track_id=track_id)
    checkpoints = _parse_checkpoints(data.get("checkpoints", []), track_id=track_id)
    finish_raw = data.get("finish")
    finish = (
        None if finish_raw is None else _parse_gate(finish_raw, field=f"{track_id}: finish")
    )
    boost_pads = _parse_boost_pads(data.get("boost_pads", []), track_id=track_id)
    surface_outer = _parse_point_loop(data.get("surface_outer"), field=f"{track_id}: surface_outer")
    surface_inner = _parse_point_loop(data.get("surface_inner"), field=f"{track_id}: surface_inner")
    racing_line = _parse_point_loop(data.get("racing_line"), field=f"{track_id}: racing_line")

    return Track(
        track_id=track_id,
        spawn=spawn_xy,
        spawn_heading=spawn_heading,
        walls=walls,
        checkpoints=checkpoints,
        finish=finish,
        boost_pads=boost_pads,
        surface_outer=surface_outer,
        surface_inner=surface_inner,
        racing_line=racing_line,
    )


def load_track_by_id(track_id: str) -> Track:
    """Load ``tracks/<track_id>.json`` from the repo's tracks directory."""
    return load_track(tracks_dir() / f"{track_id}.json")
