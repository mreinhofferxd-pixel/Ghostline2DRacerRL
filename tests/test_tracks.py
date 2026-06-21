"""Track loader + schema validation: a bad track must fail loudly at load time."""

from __future__ import annotations

import pytest

from momentum_lab import config
from momentum_lab.core.checkpoints import Gate
from momentum_lab.core.collision import Segment, closest_point_on_segment
from momentum_lab.core.track import BoostPad
from momentum_lab.tracks import TrackError, load_track, load_track_by_id


def test_loads_sample_track():
    t = load_track_by_id("track_01_easy_loop")
    assert t.track_id == "track_01_easy_loop"
    assert t.spawn == (278.6, 551.5)
    # The reworked Track 1 is a flowing circuit (outer + inner wall loops), not a
    # 4-wall box; both loops downsample to the same count.
    assert len(t.walls) == 72
    assert all(isinstance(w, Segment) for w in t.walls)


def test_loads_checkpoints_and_finish():
    t = load_track_by_id("track_01_easy_loop")
    assert len(t.checkpoints) == 4
    assert all(isinstance(g, Gate) for g in t.checkpoints)
    assert isinstance(t.finish, Gate)


def test_loads_boost_pads():
    t = load_track_by_id("track_01_easy_loop")
    assert len(t.boost_pads) == 2
    assert all(isinstance(p, BoostPad) for p in t.boost_pads)


def test_loads_render_and_authoring_hints():
    """The reworked track carries render-only surface fills + an authored racing line
    (non-physics). They must load as point loops."""
    t = load_track_by_id("track_01_easy_loop")
    assert len(t.surface_outer) >= 3 and len(t.surface_inner) >= 3
    assert all(len(p) == 2 for p in t.surface_outer)
    assert len(t.racing_line) >= 4
    assert all(len(p) == 2 for p in t.racing_line)


def test_mvp_track_set_loads():
    # Tracks 2 and 3 were removed (poor layout); Easy Loop is the only playable
    # track until the Track 1 quality pass. Each MVP track must be a full lap layout.
    assert config.MVP_TRACKS == ("track_01_easy_loop",)
    for track_id in config.MVP_TRACKS:
        t = load_track_by_id(track_id)
        assert t.track_id == track_id
        assert t.walls
        assert len(t.checkpoints) >= 3
        assert t.finish is not None
        assert t.boost_pads


def test_m8_spawns_start_clear_of_walls():
    radius = config.CAR.radius
    for track_id in config.MVP_TRACKS:
        t = load_track_by_id(track_id)
        sx, sy = t.spawn
        nearest = min(
            (
                (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy)
                for cx, cy in (
                    closest_point_on_segment(sx, sy, w.x1, w.y1, w.x2, w.y2)
                    for w in t.walls
                )
            ),
            default=float("inf"),
        )
        assert nearest > (radius + 1.0) * (radius + 1.0)


def test_mvp_gate_directions_match_intended_lap_flow():
    """Every checkpoint + the finish must FACE the way the car travels: the gate's
    forward normal should align with the local racing-line tangent. Derived from the
    authored line, so it stays correct as the geometry is retuned."""
    for track_id in config.MVP_TRACKS:
        t = load_track_by_id(track_id)
        rl = t.racing_line
        assert len(rl) >= 4

        def local_tangent(cx, cy):
            i = min(range(len(rl)), key=lambda k: (rl[k][0] - cx) ** 2 + (rl[k][1] - cy) ** 2)
            ax, ay = rl[i]
            bx, by = rl[(i + 1) % len(rl)]
            return (bx - ax, by - ay)

        assert t.finish is not None
        for gate in (*t.checkpoints, t.finish):
            cx, cy = gate.center
            tx, ty = local_tangent(cx, cy)
            assert gate.nx * tx + gate.ny * ty > 0  # gate faces travel direction


def test_mvp_gates_span_wall_to_wall():
    """Gates must anchor across the full corridor (outer wall to inner wall) so the
    inside line can't go around them (the start/finish skip regression)."""
    for track_id in config.MVP_TRACKS:
        t = load_track_by_id(track_id)
        # The corridor is ~2*offset wide; each gate's endpoints should be at least a
        # car's width short of touching, i.e. it really crosses the whole track.
        for gate in (*t.checkpoints, t.finish):
            span = ((gate.x1 - gate.x2) ** 2 + (gate.y1 - gate.y2) ** 2) ** 0.5
            assert span > 8 * config.CAR.radius  # >112 px: spans the full ~164 corridor


def test_track_without_lap_layout_is_valid(tmp_path):
    p = tmp_path / "sandbox.json"
    p.write_text('{"track_id":"s","spawn":[0,0],"walls":[]}', encoding="utf-8")
    t = load_track(p)
    assert t.checkpoints == () and t.finish is None and t.boost_pads == ()


def test_bad_checkpoint_shape_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"track_id":"x","spawn":[0,0],"checkpoints":[[1,2,3]]}', encoding="utf-8"
    )
    with pytest.raises(TrackError):
        load_track(p)


def test_zero_length_gate_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"track_id":"x","spawn":[0,0],"finish":[5,5,5,5]}', encoding="utf-8"
    )
    with pytest.raises(TrackError):
        load_track(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(TrackError):
        load_track(tmp_path / "nope.json")


def test_wall_wrong_length_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"track_id":"x","spawn":[1,2],"walls":[[1,2,3]]}', encoding="utf-8")
    with pytest.raises(TrackError):
        load_track(p)


def test_zero_length_wall_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"track_id":"x","spawn":[0,0],"walls":[[5,5,5,5]]}', encoding="utf-8")
    with pytest.raises(TrackError):
        load_track(p)


def test_bad_boost_pad_shape_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"track_id":"x","spawn":[0,0],"boost_pads":[[1,2,3]]}', encoding="utf-8"
    )
    with pytest.raises(TrackError):
        load_track(p)


def test_zero_area_boost_pad_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(
        '{"track_id":"x","spawn":[0,0],"boost_pads":[[1,2,1,5]]}', encoding="utf-8"
    )
    with pytest.raises(TrackError):
        load_track(p)


def test_bad_spawn_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"track_id":"x","spawn":[0],"walls":[]}', encoding="utf-8")
    with pytest.raises(TrackError):
        load_track(p)


def test_missing_track_id_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"spawn":[0,0],"walls":[]}', encoding="utf-8")
    with pytest.raises(TrackError):
        load_track(p)
