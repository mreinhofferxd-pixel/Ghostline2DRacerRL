"""B7.6 interactive RL replay/story viewer contracts.

These exercise the read-only viewer layer over saved RL artifacts: selection,
the curated progression story, the JSON view-model, and the self-contained HTML.
They never train, re-evaluate, or mutate sim state.
"""

from __future__ import annotations

import json
import subprocess
import sys

from momentum_lab import PHYSICS_VERSION
from momentum_lab.config import CAR
from momentum_lab.core.action import Action
from momentum_lab.physics_identity import physics_config_fingerprint, physics_config_payload
from momentum_lab.replay import Frame, InitialState, ReplayData
from momentum_lab.rl.rollout import (
    RolloutArtifact,
    RolloutSummary,
    append_rollout_summary,
    save_rollout,
)
from momentum_lab.rl.analytics import scan_runs
from momentum_lab.rl.visualizer import (
    REFERENCE_COLOR,
    _sample_indices,
    build_progression_story,
    build_view_model,
    render_viewer_html,
    select_runs,
    write_replay_viewer,
)

TRACK_ID = "track_01_easy_loop"


def _initial_state() -> InitialState:
    return InitialState(
        car=(200.0, 560.0, 0.0, 0.0, 0.0),
        prev=(200.0, 560.0),
        tick=0,
        sim_time=0.0,
        wall_hits=0,
        wall_scrape_time=0.0,
        largest_impact_speed=0.0,
        in_wall_contact=False,
        drift_time=0.0,
        peak_slip=0.0,
        boost_time=0.0,
        boost_cooldowns=(0.0, 0.0),
        boosts_used=0,
        run_started=False,
        run_start_tick=0,
        run_next_cp=0,
        run_finished=False,
        run_valid=False,
        run_lap_ticks=0,
        run_cp_ticks=(),
    )


def _artifact(
    *,
    seed: int,
    frames: tuple[Frame, ...],
    valid: bool,
    lap_time: float,
    checkpoint_index: int,
    wall_hits: int = 0,
    with_actions: bool = True,
) -> RolloutArtifact:
    summary = RolloutSummary(
        track_id=TRACK_ID,
        seed=seed,
        terminated=valid,
        truncated=not valid,
        final_reason="lap_complete" if valid else "time_limit",
        episode_steps=len(frames),
        total_reward=10.0 if valid else -3.0,
        lap_time=lap_time,
        valid=valid,
        checkpoint_index=checkpoint_index,
        checkpoint_count=4,
        wall_hits=wall_hits,
        wall_scrape_time=0.0,
        boosts_used=1,
        drift_time=0.1,
        peak_slip=0.2,
        physics_version=PHYSICS_VERSION,
        physics_fingerprint=physics_config_fingerprint(CAR),
        physics_config=physics_config_payload(CAR),
        reward_version="reward_test",
        reward_config={"version": "reward_test"},
        action_adapter={"kind": "drive_discrete"},
        path_distance=600.0,
    )
    actions = (
        tuple(Action(throttle=1.0, steer=0.1, drift=bool(f.drift)) for f in frames)
        if with_actions
        else ()
    )
    replay = ReplayData(
        track_id=TRACK_ID,
        physics_version=PHYSICS_VERSION,
        seed=seed,
        initial_state=_initial_state(),
        lap_time=lap_time,
        valid=valid,
        physics_config=physics_config_payload(CAR),
        physics_fingerprint=physics_config_fingerprint(CAR),
        actions=actions,
        frames=frames,
    )
    return RolloutArtifact(summary=summary, replay=replay)


def _lap_frames(n: int = 6) -> tuple[Frame, ...]:
    # A short trajectory that advances through all four checkpoints to the finish.
    return tuple(
        Frame(
            t=0.1 * (i + 1),
            x=100.0 + 100.0 * i,
            y=560.0,
            angle=0.0,
            speed=200.0,
            drift=(i == 3),
            cp=min(4, i),
            wall=False,
            boost=(i == 1),
        )
        for i in range(n)
    )


def _stall_frames() -> tuple[Frame, ...]:
    return (
        Frame(0.1, 200.0, 560.0, 0.0, 120.0, False, 0, False, False),
        Frame(0.2, 250.0, 560.0, 0.0, 10.0, False, 0, True, False),
    )


def _write(tmp_path, *, name, model, policy, artifact, seed):
    art_path = save_rollout(
        artifact,
        tmp_path / "evals" / model / f"{TRACK_ID}_{policy}_{name}_seed_{seed}.json",
    )
    append_rollout_summary(
        artifact.summary,
        tmp_path / "evals" / f"{model}.jsonl",
        artifact_path=art_path,
        policy=policy,
        episode=0,
        model=model,
    )


def _portfolio(tmp_path):
    """Stall + two model laps + a faster human lap -> 4 distinct visualizable runs."""
    _write(
        tmp_path,
        name="stall",
        model="b7_4",
        policy="ppo_eval",
        seed=1,
        artifact=_artifact(seed=1, frames=_stall_frames(), valid=False, lap_time=0.0, checkpoint_index=0),
    )
    _write(
        tmp_path,
        name="firstlap",
        model="b7_8",
        policy="ppo_eval",
        seed=2,
        artifact=_artifact(seed=2, frames=_lap_frames(), valid=True, lap_time=6.0, checkpoint_index=4),
    )
    _write(
        tmp_path,
        name="best",
        model="b7_15",
        policy="ppo_eval",
        seed=3,
        artifact=_artifact(seed=3, frames=_lap_frames(), valid=True, lap_time=4.5, checkpoint_index=4),
    )
    _write(
        tmp_path,
        name="human",
        model="Michi",
        policy="human_keyboard",
        seed=4,
        artifact=_artifact(seed=4, frames=_lap_frames(7), valid=True, lap_time=4.0, checkpoint_index=4),
    )
    return scan_runs(tmp_path)


def test_select_runs_filters_dedupe_and_rank(tmp_path):
    records = _portfolio(tmp_path)
    assert len(records) == 4

    top2 = select_runs(records, top_n=2)
    assert len(top2) == 2
    assert all(r.valid for r in top2)
    assert top2[0].lap_time <= top2[1].lap_time  # ranked best-first

    laps = select_runs(records, statuses=["lap_complete"])
    assert {r.model for r in laps} == {"b7_8", "b7_15", "Michi"}

    family = select_runs(records, models=["b7_1"])  # substring -> b7_15 only here
    assert {r.model for r in family} == {"b7_15"}

    best = next(r for r in records if r.model == "b7_15")
    picked = select_runs(records, run_ids=[best.run_id])
    assert len(picked) == 1 and picked[0].run_id == best.run_id


def test_progression_story_is_ordered_and_captioned(tmp_path):
    records = _portfolio(tmp_path)
    story = build_progression_story(records)

    assert story.name == "progression"
    ids = [s.run_id for s in story.steps]
    assert len(ids) == len(set(ids))  # no repeats
    assert all(s.caption for s in story.steps)

    stall = next(r for r in records if not r.valid)
    human = next(r for r in records if r.model == "Michi")
    assert story.steps[0].run_id == stall.run_id  # early stall leads the story
    assert story.steps[-1].run_id == human.run_id  # human reference closes it

    # The "Current best" step is the best *model* lap, never the (faster) human.
    best_step = next(s for s in story.steps if s.caption.startswith("Current best"))
    assert best_step.run_id != human.run_id
    best_model = min(
        (r for r in records if r.valid and r.model != "Michi"), key=lambda r: r.lap_time
    )
    assert best_step.run_id == best_model.run_id


def test_progression_story_breaks_lap_ties_toward_clean_lap(tmp_path):
    """Two equally fast laps -> the 0-wall one is the "Current best", not the crash.

    Mirrors keep-best's lap_time->wall_hits ordering so the story never advertises a
    wall-hit lap as the best when an equally fast clean lap exists.
    """
    _write(
        tmp_path,
        name="tie_clean",
        model="b7_27",
        policy="ppo_eval",
        seed=26,
        artifact=_artifact(
            seed=26, frames=_lap_frames(), valid=True, lap_time=4.1, checkpoint_index=4, wall_hits=0
        ),
    )
    _write(
        tmp_path,
        name="tie_crash",
        model="b7_25",
        policy="ppo_eval",
        seed=24,
        artifact=_artifact(
            seed=24, frames=_lap_frames(), valid=True, lap_time=4.1, checkpoint_index=4, wall_hits=1
        ),
    )
    records = scan_runs(tmp_path)
    story = build_progression_story(records)

    best_step = next(s for s in story.steps if s.caption.startswith("Current best"))
    clean = next(r for r in records if r.model == "b7_27")
    assert best_step.run_id == clean.run_id
    assert "0 wall hits" in best_step.caption


def test_view_model_has_aligned_frame_and_action_columns(tmp_path):
    records = _portfolio(tmp_path)
    best = next(r for r in records if r.model == "b7_15")
    selected = select_runs(records, top_n=3)
    vm = build_view_model(selected, reference_id=best.run_id)

    assert vm["track"]["track_id"] == TRACK_ID
    assert vm["track"]["checkpoints"] and vm["track"]["finish"]

    for run in vm["runs"]:
        f = run["frames"]
        lengths = {len(col) for col in f.values()}
        assert len(lengths) == 1  # every frame column is the same length
        # actions are parallel-indexed to frames
        assert len(run["actions"]["throttle"]) == len(f["t"])

    valid_run = next(r for r in vm["runs"] if r["valid"])
    assert valid_run["sectors"]  # sector splits derived for a completed lap

    ref = next(r for r in vm["runs"] if r["is_reference"])
    assert ref["run_id"] == best.run_id
    assert ref["color"] == REFERENCE_COLOR


def test_sample_indices_downsamples_and_keeps_last():
    assert _sample_indices(5, None) == [0, 1, 2, 3, 4]
    idxs = _sample_indices(100, 10)
    assert idxs[0] == 0 and idxs[-1] == 99
    assert len(idxs) <= 11
    assert idxs == sorted(set(idxs))


def test_summary_only_runs_excluded_unless_requested(tmp_path):
    _portfolio(tmp_path)
    # A summary row with no artifact on disk -> degraded, frame-less.
    append_rollout_summary(
        _artifact(seed=9, frames=_lap_frames(), valid=True, lap_time=9.9, checkpoint_index=4).summary,
        tmp_path / "evals" / "ghost_only.jsonl",
        artifact_path=tmp_path / "evals" / "missing.json",
        policy="ppo_eval",
        episode=0,
        model="ghost_only",
    )
    records = scan_runs(tmp_path)

    assert not any(r.model == "ghost_only" for r in select_runs(records))
    assert any(r.model == "ghost_only" for r in select_runs(records, include_summary_only=True))


def test_write_replay_viewer_is_self_contained(tmp_path):
    records = _portfolio(tmp_path)
    selected = select_runs(records, top_n=4)
    story = build_progression_story(records)
    best = next(r for r in records if r.model == "b7_15")
    vm = build_view_model(selected, reference_id=best.run_id, story=story)

    index_path = write_replay_viewer(vm, tmp_path / "viewer")
    assert index_path.exists()
    html = index_path.read_text(encoding="utf-8")

    # Single self-contained file: data is embedded, not fetched from a sidecar.
    assert 'id="ghostline-data"' in html
    assert "fetch(" not in html
    assert "data.json" not in html
    # Interactive controls and the decision telemetry are present.
    for marker in ('id="scrub"', 'id="align"', 'id="play"', "Decision timeline", "Live readout"):
        assert marker in html
    # The selected runs and the story show up.
    for run in vm["runs"]:
        assert run["run_id"] in html
    assert story.title in html


def test_env_config_from_manifest_round_trips():
    """B7.6b: re-eval must rebuild the trained observation/action/reward shape.

    Importing ``train`` does not require Stable-Baselines3 (the SB3 import is lazy),
    so this pure reconstruction is exercised even where the training extra is absent.
    """
    from momentum_lab.rl.train import (
        _env_config_payload,
        env_config_from_manifest,
        time_attack_env_config,
    )

    original = time_attack_env_config(max_episode_steps=321)
    manifest = {"env_config": _env_config_payload(original)}
    rebuilt = env_config_from_manifest(manifest)

    assert rebuilt.track_id == original.track_id
    assert rebuilt.max_episode_steps == original.max_episode_steps
    assert rebuilt.action_adapter == original.action_adapter
    assert rebuilt.observation == original.observation
    assert rebuilt.reward == original.reward

    # A manifest with no env_config degrades to the default first-lap config
    # (drive_discrete), not the bare EnvConfig discrete adapter.
    assert env_config_from_manifest({}).action_adapter == "drive_discrete"


def test_best_sidecar_env_config_uses_parent_manifest(tmp_path):
    from momentum_lab.rl.train import (
        _env_config_payload,
        infer_env_config_for_model,
        time_attack_env_config,
    )

    base_model = tmp_path / "wall_sensor_model.zip"
    best_model = tmp_path / "wall_sensor_model.best.zip"
    manifest = {
        "env_config": _env_config_payload(
            time_attack_env_config(max_episode_steps=321, include_wall_sensors=True)
        )
    }
    base_model.with_suffix(".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    rebuilt = infer_env_config_for_model(best_model)

    assert rebuilt.max_episode_steps == 321
    assert rebuilt.observation.include_lookahead is True
    assert rebuilt.observation.include_wall_sensors is True


def test_cli_builds_viewer_with_story(tmp_path):
    _portfolio(tmp_path)
    out_dir = tmp_path / "viewer"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "momentum_lab.rl.visualizer",
            "--root",
            str(tmp_path),
            "--story",
            "progression",
            "--out-dir",
            str(out_dir),
        ],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "index.html").exists()
    assert '"story": "progression"' in result.stdout
