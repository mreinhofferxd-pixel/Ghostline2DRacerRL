"""B7.2 RL rollout artifacts."""

from __future__ import annotations

import json
import math
import subprocess
import sys

from momentum_lab.core.action import Action
from momentum_lab.rl import EnvConfig, GhostlineEnv
from momentum_lab.rl.rollout import (
    RolloutSummary,
    load_rollout,
    run_batch,
    run_rollout,
    save_rollout,
    validate_rollout,
)


def _scripted_lap_action(world) -> Action:
    line = world.track.racing_line
    wp = getattr(_scripted_lap_action, "wp", 0)
    target = line[wp % len(line)]
    car = world.car
    if math.hypot(target[0] - car.px, target[1] - car.py) <= 90.0:
        wp += 1
        target = line[wp % len(line)]
    setattr(_scripted_lap_action, "wp", wp)

    desired = math.atan2(target[1] - car.py, target[0] - car.px)
    err = (desired - car.heading + math.pi) % (2.0 * math.pi) - math.pi
    steer = max(-1.0, min(1.0, 2.5 * err))
    if abs(err) > 0.9:
        return Action(brake=0.4, steer=steer)
    if car.speed > 320.0:
        return Action(steer=steer)
    return Action(throttle=1.0, steer=steer)


def test_truncated_rollout_saves_loads_and_replay_validates(tmp_path):
    env = GhostlineEnv(EnvConfig(max_episode_steps=12))
    actions = [1, 2, 3, 9, 10, 1, 1, 4, 0, 3, 2, 1, 1, 1]
    artifact = run_rollout(env, actions, seed=7)

    assert artifact.summary.truncated is True
    assert artifact.summary.terminated is False
    assert artifact.summary.final_reason == "time_limit"
    assert artifact.summary.episode_steps == 12
    assert len(artifact.replay.actions) == 12
    assert len(artifact.replay.frames) == 12
    assert artifact.summary.reward_version == "reward_v1"
    assert artifact.summary.physics_fingerprint == artifact.replay.physics_fingerprint
    assert artifact.summary.reward_config["version"] == "reward_v1"
    assert validate_rollout(artifact)

    path = save_rollout(artifact, tmp_path / "rollout.json")
    loaded = load_rollout(path)
    assert loaded.summary == artifact.summary
    assert loaded.replay.to_dict() == artifact.replay.to_dict()
    assert validate_rollout(loaded)


def test_completed_lap_rollout_records_valid_episode():
    env = GhostlineEnv(EnvConfig(max_episode_steps=4000))
    env.reset(seed=0)
    setattr(_scripted_lap_action, "wp", 0)
    actions: list[Action] = []
    for _ in range(4000):
        actions.append(_scripted_lap_action(env.sim.world))
        # Mirror the action into the planning sim so the next scripted action sees
        # the same state run_rollout will see after reset.
        env.step(actions[-1])
        if env.sim.world.run.finished:
            break

    artifact = run_rollout(GhostlineEnv(EnvConfig(max_episode_steps=4000)), actions, seed=0)

    assert artifact.summary.terminated is True
    assert artifact.summary.truncated is False
    assert artifact.summary.final_reason == "lap_complete"
    assert artifact.summary.valid is True
    assert artifact.summary.lap_time > 0.0
    assert artifact.summary.checkpoint_index == artifact.summary.checkpoint_count
    assert artifact.replay.valid is True
    # B9: the sub-tick lap-time marker is stamped and survives a serialize round-trip.
    assert artifact.summary.timing_version == "timing_v2"
    assert RolloutSummary.from_dict(artifact.summary.to_dict()).timing_version == "timing_v2"
    assert validate_rollout(artifact)


def test_legacy_rollout_summary_defaults_to_integer_tick_timing():
    """B9: rollout summaries written before sub-tick timing have no `timing_version`;
    they must load as the integer-tick marker, not crash."""
    env = GhostlineEnv(EnvConfig(max_episode_steps=1))
    artifact = run_rollout(env, [0], seed=3)
    raw = artifact.summary.to_dict()
    raw.pop("timing_version")
    assert RolloutSummary.from_dict(raw).timing_version == "timing_v1"


def test_rollout_default_path_is_under_runs_rl_rollouts():
    env = GhostlineEnv(EnvConfig(max_episode_steps=1))
    artifact = run_rollout(env, [0], seed=3)
    path = save_rollout(artifact)
    try:
        assert "runs" in path.parts
        assert "rl" in path.parts
        assert "rollouts" in path.parts
        assert path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_run_batch_writes_artifacts_and_summary_jsonl(tmp_path):
    output_dir = tmp_path / "artifacts"
    summary_path = tmp_path / "summaries.jsonl"
    artifacts = run_batch(
        episodes=3,
        policy="cycle",
        seed=10,
        env_config=EnvConfig(max_episode_steps=5),
        output_dir=output_dir,
        summary_path=summary_path,
    )

    assert len(artifacts) == 3
    files = sorted(output_dir.glob("*.json"))
    assert len(files) == 3
    assert all(validate_rollout(load_rollout(path)) for path in files)

    rows = [
        json.loads(line)
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 3
    assert [row["episode"] for row in rows] == [0, 1, 2]
    assert all(row["policy"] == "cycle" for row in rows)
    assert all(row["artifact_path"] for row in rows)


def test_append_rollout_summary_accepts_model_metadata(tmp_path):
    from momentum_lab.rl.rollout import append_rollout_summary

    artifact = run_rollout(GhostlineEnv(EnvConfig(max_episode_steps=1)), [0], seed=3)
    summary_path = tmp_path / "summary.jsonl"
    artifact_path = tmp_path / "artifact.json"
    append_rollout_summary(
        artifact.summary,
        summary_path,
        artifact_path=artifact_path,
        policy="ppo_eval",
        episode=4,
        model="model_name",
        extra={"model_path": "runs/rl/models/model_name.zip", "deterministic": True},
    )

    row = json.loads(summary_path.read_text(encoding="utf-8"))
    assert row["model"] == "model_name"
    assert row["model_path"] == "runs/rl/models/model_name.zip"
    assert row["deterministic"] is True


def test_rollout_module_cli_runs_batch(tmp_path):
    output_dir = tmp_path / "cli_artifacts"
    summary_path = tmp_path / "cli_summaries.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "momentum_lab.rl.rollout",
            "--episodes",
            "2",
            "--policy",
            "throttle",
            "--seed",
            "20",
            "--max-episode-steps",
            "4",
            "--output-dir",
            str(output_dir),
            "--summary-path",
            str(summary_path),
        ],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["episodes"] == 2
    assert payload["policy"] == "throttle"
    assert payload["summary_path"] == str(summary_path)
    assert len(list(output_dir.glob("*.json"))) == 2
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 2
