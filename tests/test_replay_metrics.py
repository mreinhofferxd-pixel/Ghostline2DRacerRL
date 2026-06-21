"""Replay + metrics acceptance."""

from __future__ import annotations

import dataclasses
import json

import pytest

from momentum_lab.config import CAR
from momentum_lab.core.action import Action
from momentum_lab.core.checkpoints import Gate
from momentum_lab.core.sim import Simulation
from momentum_lab.core.track import Track
from momentum_lab.main import _should_save_best_replay
from momentum_lab.metrics import RunSummary, append_run_summary
from momentum_lab.physics_identity import (
    physics_config_fingerprint,
    physics_config_payload,
)
from momentum_lab.replay import (
    ReplayData,
    ReplayError,
    ReplayRecorder,
    load_replay,
    play_replay,
    save_replay,
    trajectory_matches,
)


def _straight_lap_track() -> Track:
    return Track(
        track_id="m6_straight_lap",
        spawn=(0.0, 0.0),
        spawn_heading=0.0,
        finish=Gate.from_endpoints(300.0, 100.0, 300.0, -100.0),
    )


def _record_straight_lap(cfg=CAR):
    track = _straight_lap_track()
    sim = Simulation(cfg)
    sim.reset(track=track, seed=123)
    recorder = ReplayRecorder.start(sim)
    for _ in range(180):
        recorder.step(sim, Action(throttle=1.0))
        if sim.world.run.finished:
            break
    assert sim.world.run.finished and sim.world.run.valid
    return track, sim, recorder, recorder.to_replay(sim)


def test_replay_round_trip_reproduces_trajectory():
    track, sim, _recorder, replay = _record_straight_lap()

    loaded = ReplayData.from_dict(replay.to_dict())
    result = play_replay(loaded, track)

    assert replay.valid
    assert replay.lap_time > 0.0
    assert replay.schema_version == 2
    assert replay.physics_config == physics_config_payload(CAR)
    assert replay.physics_fingerprint == physics_config_fingerprint(CAR)
    assert len(replay.actions) == len(replay.frames)
    assert result.final_hash == sim.state_hash()
    assert trajectory_matches(loaded, track)


def test_replay_json_storage_round_trips(tmp_path):
    track, _sim, _recorder, replay = _record_straight_lap()
    path = tmp_path / "last_run.json"

    save_replay(replay, path)
    loaded = load_replay(path)

    assert loaded.to_dict() == replay.to_dict()
    assert loaded.physics_config == physics_config_payload(CAR)
    assert loaded.physics_fingerprint == physics_config_fingerprint(CAR)
    assert trajectory_matches(loaded, track)


def test_run_summary_jsonl_uses_replay_frames(tmp_path):
    tuned = dataclasses.replace(CAR, grip_normal=0.905)
    _track, sim, recorder, replay = _record_straight_lap(tuned)
    summary = RunSummary.from_world(
        sim.world,
        replay.frames,
        physics_cfg=tuned,
        timestamp="2026-06-20T12:00:00Z",
    )
    path = append_run_summary(summary, tmp_path / "runs.jsonl")

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["track_id"] == "m6_straight_lap"
    assert row["valid"] is True
    assert row["lap_time"] == replay.lap_time
    assert row["timing_version"] == "timing_v2"  # B9: sub-tick lap-time marker
    assert row["max_speed"] == max(frame.speed for frame in recorder.frames)
    assert row["avg_speed"] > 0.0
    assert row["checkpoint_times"] == []
    assert row["physics_fingerprint"] == physics_config_fingerprint(tuned)
    assert row["physics_config"]["grip_normal"] == tuned.grip_normal
    assert "length" not in row["physics_config"]
    assert "width" not in row["physics_config"]
    assert "radius" in row["physics_config"]
    assert row["timestamp"] == "2026-06-20T12:00:00Z"


def test_tuned_replay_requires_matching_physics_config():
    tuned = dataclasses.replace(CAR, grip_normal=0.905)
    track, sim, _recorder, replay = _record_straight_lap(tuned)

    with pytest.raises(ReplayError, match="physics_fingerprint mismatch"):
        play_replay(replay, track)

    result = play_replay(replay, track, cfg=tuned)

    assert result.final_hash == sim.state_hash()
    assert trajectory_matches(replay, track, cfg=tuned)


def test_legacy_replay_without_fingerprint_loads_but_is_strictly_stale():
    track, _sim, _recorder, replay = _record_straight_lap()
    raw = replay.to_dict()
    raw.pop("physics_config")
    raw.pop("physics_fingerprint")
    raw["schema_version"] = 1
    legacy = ReplayData.from_dict(raw)

    assert legacy.physics_config is None
    assert legacy.physics_fingerprint is None
    with pytest.raises(ReplayError, match="physics_fingerprint missing"):
        play_replay(legacy, track)
    assert trajectory_matches(legacy, track, strict_version=False)


def test_best_save_guard_allows_faster_same_fingerprint():
    _track, _sim, _recorder, replay = _record_straight_lap()
    existing = dataclasses.replace(replay, lap_time=10.0)
    faster = dataclasses.replace(replay, lap_time=9.0)
    slower = dataclasses.replace(replay, lap_time=11.0)

    assert _should_save_best_replay(faster, existing)
    assert not _should_save_best_replay(slower, existing)


def test_best_save_guard_rejects_different_trusted_fingerprint():
    tuned = dataclasses.replace(CAR, grip_normal=0.905)
    _track, _sim, _recorder, default_replay = _record_straight_lap()
    _track, _sim, _recorder, tuned_replay = _record_straight_lap(tuned)
    faster_tuned = dataclasses.replace(tuned_replay, lap_time=default_replay.lap_time - 1.0)

    assert not _should_save_best_replay(faster_tuned, default_replay)


def test_best_save_guard_replaces_legacy_missing_fingerprint():
    _track, _sim, _recorder, replay = _record_straight_lap()
    legacy = dataclasses.replace(replay, physics_config=None, physics_fingerprint=None)

    assert _should_save_best_replay(replay, legacy)
