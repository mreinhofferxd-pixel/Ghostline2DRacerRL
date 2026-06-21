"""Generate ``tracks/track_01_easy_loop.json`` — the source of Track 1's geometry.

Track 1 used to be a rectangular ring (four identical 90deg corners) that read like
a dev wireframe. This tool builds a *real* flowing circuit instead: a smooth closed
centerline (centripetal Catmull-Rom through the hand-placed ANCHORS below) offset by
a constant half-width into outer + inner wall loops. Edit ANCHORS / HALF_WIDTH and
re-run to reshape the track; the script regenerates walls, evenly arc-spaced
checkpoints, the finish, boost pads, spawn pose, and the render-only surface polygons
in one pass, and validates corridor width / curvature / bounds before writing.

This is a dev tool (lives outside the package). It imports ``momentum_lab`` only to
reuse the real ``Gate`` so generated gate directions match the sim exactly.

    python tools/build_track_01.py            # write the JSON + an SVG preview
    python tools/build_track_01.py --check    # validate + print diagnostics only

Keep determinism + the Action seam untouched: this only changes track DATA and
render hints; physics constants (the physics_version) are not touched here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Import the real Gate so generated gate orientation matches the sim's crossing rule.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from momentum_lab.core.checkpoints import Gate  # noqa: E402

TRACK_ID = "track_01_easy_loop"

# Centerline anchors in TRAVEL ORDER (counter-clockwise on screen, y-down). The car
# spawns on the bottom "main straight" just behind the finish and drives +x first.
# A flowing circuit whose interest comes from VARIED corner radii (what drift-and-catch
# rewards), not from a wireframe's identical 90deg boxes:
#   main straight -> T1 fast bottom-right sweeper -> short right straight ->
#   T2 easy top-right -> long top straight -> T3 tight top-left -> left straight ->
#   T4 bottom-left sweeper -> back onto the main straight.
ANCHORS: tuple[tuple[float, float], ...] = (
    (360.0, 562.0),   # 0  main straight (start/finish sits here)
    (740.0, 570.0),   # 1  long main straight
    (986.0, 552.0),   # 2  T1 turn-in
    (1086.0, 468.0),  # 3  T1 fast bottom-right sweeper
    (1110.0, 320.0),  # 4  right chute (drawn out -> the longest, fastest section)
    (1066.0, 220.0),  # 5  T2 top-right
    (936.0, 176.0),   # 6  onto the top straight
    (628.0, 160.0),   # 7  top straight
    (336.0, 182.0),   # 8  T3 entry (tighter, top-left)
    (190.0, 312.0),   # 9  T3 top-left
    (182.0, 452.0),   # 10 left side
    (252.0, 540.0),   # 11 T4 bottom-left sweeper back to the straight
)

HALF_WIDTH = 82.0          # corridor half-width (full corridor ~164 px)
SAMPLES_PER_SEG = 24       # centerline resolution per anchor span
WALL_STEP = 8              # downsample factor: every Nth sample becomes a wall vertex
GATE_FRACS = (0.2, 0.4, 0.6, 0.8)   # checkpoint arc-length fractions (finish = 0.0)
SPAWN_BACK_FRAC = 0.035    # how far behind the finish the car spawns (arc fraction)
RACING_LINE_POINTS = 18    # centerline samples exported for the scripted autopilot

# Render-only palette for the SVG preview (mirrors the in-game renderer intent).
_BG = "#181a1e"
_ASPHALT = "#2f323a"
_INFIELD = "#1c1f25"
_WALL = "#c2cad8"
_CP = "#5a6e8c"
_FINISH = "#eef5ff"
_BOOST = "#5ae6d2"


# --- vector helpers ----------------------------------------------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _mul(a, s):
    return (a[0] * s, a[1] * s)


def _hypot(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-12 else (0.0, 0.0)


# --- centripetal Catmull-Rom (closed) ---------------------------------------
def _cr_segment(p0, p1, p2, p3, samples):
    """Centripetal (alpha=0.5) Catmull-Rom samples for the p1->p2 span [t in [t1,t2))."""
    def tj(ti, a, b):
        return ti + _hypot(a, b) ** 0.5

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    out = []
    for i in range(samples):
        t = t1 + (t2 - t1) * (i / samples)
        a1 = _add(_mul(p0, (t1 - t) / (t1 - t0)), _mul(p1, (t - t0) / (t1 - t0)))
        a2 = _add(_mul(p1, (t2 - t) / (t2 - t1)), _mul(p2, (t - t1) / (t2 - t1)))
        a3 = _add(_mul(p2, (t3 - t) / (t3 - t2)), _mul(p3, (t - t2) / (t3 - t2)))
        b1 = _add(_mul(a1, (t2 - t) / (t2 - t0)), _mul(a2, (t - t0) / (t2 - t0)))
        b2 = _add(_mul(a2, (t3 - t) / (t3 - t1)), _mul(a3, (t - t1) / (t3 - t1)))
        c = _add(_mul(b1, (t2 - t) / (t2 - t1)), _mul(b2, (t - t1) / (t2 - t1)))
        out.append(c)
    return out


def _centerline(anchors, samples_per_seg):
    n = len(anchors)
    pts = []
    for i in range(n):
        p0 = anchors[(i - 1) % n]
        p1 = anchors[i]
        p2 = anchors[(i + 1) % n]
        p3 = anchors[(i + 2) % n]
        pts.extend(_cr_segment(p0, p1, p2, p3, samples_per_seg))
    return pts


# --- offset geometry ---------------------------------------------------------
def _tangents(pts):
    n = len(pts)
    return [_norm(_sub(pts[(i + 1) % n], pts[(i - 1) % n])) for i in range(n)]


def _outward_sign(pts, normals):
    """Single global sign so ``sign * normal`` points away from the loop centroid."""
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    votes = 0
    for p, nrm in zip(pts, normals):
        votes += 1 if (nrm[0] * (p[0] - cx) + nrm[1] * (p[1] - cy)) > 0 else -1
    return 1.0 if votes >= 0 else -1.0


def _build():
    pts = _centerline(ANCHORS, SAMPLES_PER_SEG)
    n = len(pts)
    tans = _tangents(pts)
    # Left normal of the tangent (rot +90 in y-down screen space): (-ty, tx).
    normals = [(-t[1], t[0]) for t in tans]
    s = _outward_sign(pts, normals)
    out_n = [_mul(nrm, s) for nrm in normals]  # unit outward normal per sample

    outer = [_add(p, _mul(nrm, HALF_WIDTH)) for p, nrm in zip(pts, out_n)]
    inner = [_sub(p, _mul(nrm, HALF_WIDTH)) for p, nrm in zip(pts, out_n)]

    # Cumulative arc length (closed) for station placement.
    seglen = [_hypot(pts[i], pts[(i + 1) % n]) for i in range(n)]
    total = sum(seglen)
    cum = [0.0]
    for L in seglen[:-1]:
        cum.append(cum[-1] + L)

    def station_index(frac):
        target = (frac % 1.0) * total
        # nearest sample to the target arc length
        best, bi = 1e18, 0
        for i, c in enumerate(cum):
            d = abs(c - target)
            if d < best:
                best, bi = d, i
        return bi

    def gate_at(frac):
        k = station_index(frac)
        c, nrm, t = pts[k], out_n[k], tans[k]
        a = _add(c, _mul(nrm, HALF_WIDTH + 4.0))   # outer wall end
        b = _sub(c, _mul(nrm, HALF_WIDTH + 4.0))   # inner wall end
        # Order endpoints so the forward normal (tangent rot +90) == travel tangent t.
        g = Gate.from_endpoints(a[0], a[1], b[0], b[1])
        if g.nx * t[0] + g.ny * t[1] < 0:
            a, b = b, a
        return [round(a[0], 1), round(a[1], 1), round(b[0], 1), round(b[1], 1)]

    checkpoints = [gate_at(f) for f in GATE_FRACS]
    finish = gate_at(0.0)

    # Spawn just behind the finish, facing along travel.
    sk = station_index(1.0 - SPAWN_BACK_FRAC)
    spawn = pts[sk]
    spawn_heading = math.atan2(tans[sk][1], tans[sk][0])

    # Boost pads: axis-aligned rects centered on the centerline where it runs roughly
    # straight (main straight after the line, and the right-side straight).
    def pad_at(frac, half_w, half_h):
        k = station_index(frac)
        c = pts[k]
        return [
            round(c[0] - half_w, 1), round(c[1] - half_h, 1),
            round(c[0] + half_w, 1), round(c[1] + half_h, 1),
        ]

    boost_pads = [
        pad_at(0.07, 70.0, 34.0),   # main straight, just past the line
        pad_at(0.31, 40.0, 60.0),   # turn-in to the fast right sweeper (between gates)
    ]

    def loop_round(seq, step):
        idx = list(range(0, len(seq), step))
        return [[round(seq[i][0], 1), round(seq[i][1], 1)] for i in idx]

    walls = []
    for loop in (loop_round(outer, WALL_STEP), loop_round(inner, WALL_STEP)):
        m = len(loop)
        for i in range(m):
            a, b = loop[i], loop[(i + 1) % m]
            walls.append([a[0], a[1], b[0], b[1]])

    rl_step = max(1, n // RACING_LINE_POINTS)
    racing_line = [[round(pts[i][0], 1), round(pts[i][1], 1)] for i in range(0, n, rl_step)]

    track = {
        "track_id": TRACK_ID,
        "spawn": [round(spawn[0], 1), round(spawn[1], 1)],
        "spawn_heading": round(spawn_heading, 4),
        "walls": walls,
        "checkpoints": checkpoints,
        "finish": finish,
        "boost_pads": boost_pads,
        # --- render / authoring hints (ignored by physics) ---
        "surface_outer": loop_round(outer, WALL_STEP),
        "surface_inner": loop_round(inner, WALL_STEP),
        "racing_line": racing_line,
    }
    diag = _diagnostics(pts, outer, inner, out_n, total, spawn, walls)
    return track, diag


# --- validation --------------------------------------------------------------
def _diagnostics(pts, outer, inner, out_n, total, spawn, walls):
    n = len(pts)
    widths = [_hypot(outer[i], inner[i]) for i in range(n)]
    # Local radius of curvature ~ ds / dtheta; flag where it pinches below the offset.
    min_R, min_R_at = 1e18, None
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        t1 = _norm(_sub(b, a))
        t2 = _norm(_sub(c, b))
        dot = max(-1.0, min(1.0, t1[0] * t2[0] + t1[1] * t2[1]))
        dtheta = math.acos(dot)
        ds = 0.5 * (_hypot(a, b) + _hypot(b, c))
        R = ds / dtheta if dtheta > 1e-9 else 1e18
        if R < min_R:
            min_R, min_R_at = R, (round(b[0]), round(b[1]))
    xs = [w[0] for w in walls] + [w[2] for w in walls]
    ys = [w[1] for w in walls] + [w[3] for w in walls]
    return {
        "centerline_samples": n,
        "wall_segments": len(walls),
        "perimeter": round(total, 1),
        "min_width": round(min(widths), 1),
        "max_width": round(max(widths), 1),
        "min_radius": round(min_R, 1),
        "min_radius_at": min_R_at,
        "bounds": (round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)),
        "spawn": (round(spawn[0], 1), round(spawn[1], 1)),
    }


def _check(diag):
    ok = True
    msgs = []
    if diag["min_width"] < 60.0:
        ok = False
        msgs.append(f"corridor too narrow: min_width={diag['min_width']}")
    if diag["min_radius"] < HALF_WIDTH * 1.1:
        ok = False
        msgs.append(
            f"curvature too tight: min_radius={diag['min_radius']} at {diag['min_radius_at']} "
            f"(needs > {HALF_WIDTH * 1.1:.0f} to keep the inner wall from pinching)"
        )
    bx0, by0, bx1, by1 = diag["bounds"]
    if bx0 < 50 or by0 < 40 or bx1 > 1230 or by1 > 680:
        ok = False
        msgs.append(f"out of bounds: walls span {diag['bounds']} (world is 1280x720)")
    return ok, msgs


# --- SVG preview -------------------------------------------------------------
def _poly(seq):
    return " ".join(f"{x},{y}" for x, y in seq)


def _svg(track) -> str:
    w, h = 1280, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="{_BG}"/>',
        f'<polygon points="{_poly(track["surface_outer"])}" fill="{_ASPHALT}"/>',
        f'<polygon points="{_poly(track["surface_inner"])}" fill="{_INFIELD}"/>',
    ]
    for x1, y1, x2, y2 in track["walls"]:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_WALL}" stroke-width="3"/>')
    for p in track["boost_pads"]:
        x, y = min(p[0], p[2]), min(p[1], p[3])
        parts.append(
            f'<rect x="{x}" y="{y}" width="{abs(p[2]-p[0])}" height="{abs(p[3]-p[1])}" '
            f'fill="none" stroke="{_BOOST}" stroke-width="2"/>'
        )
    for i, g in enumerate(track["checkpoints"]):
        parts.append(f'<line x1="{g[0]}" y1="{g[1]}" x2="{g[2]}" y2="{g[3]}" stroke="{_CP}" stroke-width="4"/>')
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        parts.append(f'<text x="{cx}" y="{cy}" fill="{_CP}" font-size="20" text-anchor="middle">{i+1}</text>')
    f = track["finish"]
    parts.append(f'<line x1="{f[0]}" y1="{f[1]}" x2="{f[2]}" y2="{f[3]}" stroke="{_FINISH}" stroke-width="6"/>')
    sx, sy = track["spawn"]
    parts.append(f'<circle cx="{sx}" cy="{sy}" r="8" fill="#5ab0ff"/>')
    parts.append("</svg>")
    return "\n".join(parts)


# --- entry -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Build Track 1 geometry.")
    ap.add_argument("--check", action="store_true", help="validate + print diagnostics only")
    args = ap.parse_args()

    track, diag = _build()
    print("Track 1 geometry diagnostics:")
    for k, v in diag.items():
        print(f"  {k:18} {v}")
    ok, msgs = _check(diag)
    for m in msgs:
        print(f"  ! {m}")
    print(f"  verdict: {'OK' if ok else 'PROBLEM'}")
    if not ok:
        print("Not writing files (fix ANCHORS/HALF_WIDTH and re-run).")
        return 1
    if args.check:
        return 0

    root = Path(__file__).resolve().parents[1]
    json_path = root / "tracks" / f"{TRACK_ID}.json"
    json_path.write_text(json.dumps(track, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")

    svg_path = root / "runs" / "track_previews" / f"{TRACK_ID}_preview.svg"
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(_svg(track), encoding="utf-8")
    print(f"wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
