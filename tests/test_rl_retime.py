"""B9 migration: retroactive sub-tick re-timing of saved runs."""

from __future__ import annotations

import json
import math

import pytest

from momentum_lab.config import CONTROL_DT
from momentum_lab.rl import EnvConfig, GhostlineEnv
from momentum_lab.rl.rollout import run_rollout
from momentum_lab.rl.retime import (
    finish_fraction_from_frames,
    retime_artifact_dict,
    retime_tree,
    subtick_lap_time,
)
from momentum_lab.tracks import load_track_by_id

from test_rl_rollout import _scripted_lap_action


def _completed_lap_artifact():
    """A real completed-lap rollout artifact on the loadable Track 1."""
    env = GhostlineEnv(EnvConfig(max_episode_steps=4000))
    env.reset(seed=0)
    setattr(_scripted_lap_action, "wp", 0)
    actions = []
    for _ in range(4000):
        actions.append(_scripted_lap_action(env.sim.world))
        env.step(actions[-1])
        if env.sim.world.run.finished:
            break
    artifact = run_rollout(GhostlineEnv(EnvConfig(max_episode_steps=4000)), actions, seed=0)
    assert artifact.summary.valid, "fixture must produce a valid lap"
    return artifact


def _integer_tick_lap_time(sub_tick: float) -> float:
    """The pre-B9 (tick-rounded) lap_time a run with this sub-tick time would have stored."""
    return math.ceil(round(sub_tick / CONTROL_DT, 9)) * CONTROL_DT


def test_finish_fraction_reproduces_live_subtick_lap_time():
    artifact = _completed_lap_artifact()
    track = load_track_by_id(artifact.summary.track_id)
    # Frame objects exercise the object (not dict) branch of the recovery helper.
    fraction = finish_fraction_from_frames(
        track.finish, artifact.replay.frames, len(track.checkpoints)
    )
    assert fraction is not None
    assert 0.0 < fraction <= 1.0

    integer = _integer_tick_lap_time(artifact.summary.lap_time)
    assert integer >= artifact.summary.lap_time
    # Re-timing the integer-tick value recovers the live sub-tick lap_time exactly.
    assert subtick_lap_time(integer, fraction) == pytest.approx(
        artifact.summary.lap_time, abs=1e-9
    )


def test_retime_artifact_dict_recovers_value_and_is_idempotent():
    artifact = _completed_lap_artifact()
    sub = artifact.summary.lap_time
    integer = _integer_tick_lap_time(sub)

    d = artifact.to_dict()
    # Fabricate a pre-B9 record: tick-rounded lap_time, no sub-tick marker.
    d["summary"]["lap_time"] = integer
    d["summary"]["timing_version"] = "timing_v1"
    d["replay"]["lap_time"] = integer
    d["replay"].pop("timing_version", None)

    assert retime_artifact_dict(d) is True
    assert d["summary"]["timing_version"] == "timing_v2"
    assert d["summary"]["lap_time"] == pytest.approx(sub, abs=1e-9)
    assert d["replay"]["lap_time"] == pytest.approx(sub, abs=1e-9)
    assert d["replay"]["timing_version"] == "timing_v2"

    # Already migrated -> a second pass changes nothing (no double subtraction).
    assert retime_artifact_dict(d) is False
    assert d["summary"]["lap_time"] == pytest.approx(sub, abs=1e-9)


def test_retime_artifact_dict_skips_invalid_runs():
    env = GhostlineEnv(EnvConfig(max_episode_steps=1))
    artifact = run_rollout(env, [0], seed=3)  # one idle step: not a valid lap
    d = artifact.to_dict()
    d["summary"]["timing_version"] = "timing_v1"
    assert d["summary"]["valid"] is False
    assert retime_artifact_dict(d) is False


def test_retime_tree_migrates_artifact_row_and_replay(tmp_path):
    artifact = _completed_lap_artifact()
    sub = artifact.summary.lap_time
    integer = _integer_tick_lap_time(sub)

    d = artifact.to_dict()
    d["summary"]["lap_time"] = integer
    d["summary"]["timing_version"] = "timing_v1"
    d["replay"]["lap_time"] = integer
    d["replay"].pop("timing_version", None)

    root = tmp_path / "rl"
    eval_dir = root / "evals" / "m"
    eval_dir.mkdir(parents=True)
    art_path = eval_dir / "rollout.json"
    art_path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # A summary row that points at the artifact by sibling filename (no frames of its own).
    row = {"valid": True, "lap_time": integer, "artifact_path": "rollout.json"}
    jsonl_path = eval_dir / "m.jsonl"
    jsonl_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    # A bare in-game replay file (best lap / ghost source).
    rdir = tmp_path / "replays"
    rdir.mkdir()
    replay = artifact.replay.to_dict()
    replay["lap_time"] = integer
    replay.pop("timing_version", None)
    (rdir / "best.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = retime_tree(root, replays_dir=rdir)
    assert report.artifacts_changed == 1
    assert report.rows_changed == 1
    assert report.replays_changed == 1

    got_art = json.loads(art_path.read_text(encoding="utf-8"))
    assert got_art["summary"]["lap_time"] == pytest.approx(sub, abs=1e-9)
    assert got_art["summary"]["timing_version"] == "timing_v2"

    got_row = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert got_row["lap_time"] == pytest.approx(sub, abs=1e-9)
    assert got_row["timing_version"] == "timing_v2"

    got_rep = json.loads((rdir / "best.json").read_text(encoding="utf-8"))
    assert got_rep["lap_time"] == pytest.approx(sub, abs=1e-9)
    assert got_rep["timing_version"] == "timing_v2"

    # A full second pass is a no-op (idempotent migration).
    again = retime_tree(root, replays_dir=rdir)
    assert again.artifacts_changed == 0
    assert again.rows_changed == 0
    assert again.replays_changed == 0
