"""B7.4a RL rollout analytics and static visual triage."""

from __future__ import annotations

import subprocess
import sys

from momentum_lab import PHYSICS_VERSION
from momentum_lab.config import CAR
from momentum_lab.physics_identity import physics_config_fingerprint, physics_config_payload
from momentum_lab.replay import Frame, InitialState, ReplayData
from momentum_lab.rl import EnvConfig, GhostlineEnv
from momentum_lab.rl.analytics import (
    compare_run_traces,
    find_run_by_id,
    format_markdown_report,
    format_trace_comparison_report,
    parse_group_by,
    rank_runs,
    scan_runs,
    write_visual_report,
)
from momentum_lab.rl.rewards import RewardConfig
from momentum_lab.rl.rollout import (
    RolloutArtifact,
    RolloutSummary,
    append_rollout_summary,
    run_rollout,
    save_rollout,
)


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
    checkpoint_index: int,
    total_reward: float,
    frames: tuple[Frame, ...],
    wall_hits: int = 0,
    valid: bool = False,
    lap_time: float | None = None,
    boosts_used: int = 1,
    drift_time: float = 0.10,
    path_distance: float = 0.0,
) -> RolloutArtifact:
    lap_time = lap_time if lap_time is not None else len(frames) / 60.0
    summary = RolloutSummary(
        track_id="track_01_easy_loop",
        seed=seed,
        terminated=valid,
        truncated=not valid,
        final_reason="lap_complete" if valid else "time_limit",
        episode_steps=len(frames),
        total_reward=total_reward,
        lap_time=lap_time,
        valid=valid,
        checkpoint_index=checkpoint_index,
        checkpoint_count=4,
        wall_hits=wall_hits,
        wall_scrape_time=0.0,
        boosts_used=boosts_used,
        drift_time=drift_time,
        peak_slip=0.2,
        physics_version=PHYSICS_VERSION,
        physics_fingerprint=physics_config_fingerprint(CAR),
        physics_config=physics_config_payload(CAR),
        reward_version="reward_test",
        reward_config={"version": "reward_test"},
        action_adapter={"kind": "drive_discrete"},
        path_distance=path_distance,
    )
    replay = ReplayData(
        track_id="track_01_easy_loop",
        physics_version=PHYSICS_VERSION,
        seed=seed,
        initial_state=_initial_state(),
        lap_time=summary.lap_time,
        valid=valid,
        physics_config=physics_config_payload(CAR),
        physics_fingerprint=physics_config_fingerprint(CAR),
        actions=(),
        frames=frames,
    )
    return RolloutArtifact(summary=summary, replay=replay)


def _write_artifacts(tmp_path):
    output_dir = tmp_path / "evals" / "model_a"
    summary_path = tmp_path / "evals" / "model_a.jsonl"
    early = _artifact(
        seed=1,
        checkpoint_index=0,
        total_reward=-10.0,
        frames=(
            Frame(0.0, 200.0, 560.0, 0.0, 120.0, False, 0, False, False),
            Frame(0.1, 340.0, 590.0, 0.0, 110.0, False, 0, False, False),
        ),
    )
    farther = _artifact(
        seed=2,
        checkpoint_index=1,
        total_reward=8.0,
        wall_hits=1,
        frames=(
            Frame(0.0, 740.0, 520.0, 0.0, 20.0, False, 1, False, False),
            Frame(0.1, 1030.0, 460.0, 0.0, 10.0, True, 1, False, True),
            Frame(0.2, 1170.0, 300.0, 0.0, 0.0, False, 1, True, False),
        ),
    )

    early_path = save_rollout(
        early,
        output_dir / "track_01_easy_loop_ppo_eval_ep0000_seed_1_000002.json",
    )
    farther_path = save_rollout(
        farther,
        output_dir / "track_01_easy_loop_ppo_eval_ep0001_seed_2_000003.json",
    )
    append_rollout_summary(
        early.summary,
        summary_path,
        artifact_path=early_path,
        policy="ppo_eval",
        episode=0,
    )
    append_rollout_summary(
        farther.summary,
        summary_path,
        artifact_path=farther_path,
        policy="ppo_eval",
        episode=1,
    )
    return summary_path, (early_path, farther_path)


def test_scan_and_rank_runs_prefers_checkpoint_progress(tmp_path):
    _write_artifacts(tmp_path)

    records = scan_runs(tmp_path)
    groups = rank_runs(records, top_n=30, group_by=parse_group_by("model,policy"))

    assert len(records) == 2
    assert len(groups) == 1
    assert groups[0].label == "model=model_a / policy=ppo_eval"
    assert groups[0].runs[0].seed == 2
    assert groups[0].runs[0].checkpoint_index == 1
    assert groups[0].runs[0].status == "stalled_on_wall"
    assert groups[0].runs[0].stage_progress is not None


def test_identical_rollouts_collapse_to_one_run_with_stable_id(tmp_path):
    """Behaviorally-identical rollouts (e.g. deterministic eval triplicates) must
    collapse to one ranked row keyed on a stable content id, not show up N times."""
    output_dir = tmp_path / "evals" / "model_dup"
    summary_path = tmp_path / "evals" / "model_dup.jsonl"
    frames = (
        Frame(0.0, 200.0, 560.0, 0.0, 120.0, False, 0, False, False),
        Frame(0.1, 360.0, 600.0, 0.1, 130.0, True, 1, False, False),
    )
    # Same trajectory, different (no-op) eval seeds and different file paths.
    for ep, seed in ((0, 10000), (1, 10001), (2, 10002)):
        art = _artifact(seed=seed, checkpoint_index=1, total_reward=5.0, frames=frames)
        path = save_rollout(
            art,
            output_dir / f"track_01_easy_loop_ppo_eval_ep{ep:04d}_seed_{seed}_000002.json",
        )
        append_rollout_summary(
            art.summary, summary_path, artifact_path=path, policy="ppo_eval", episode=ep
        )

    records = scan_runs(tmp_path)
    assert len(records) == 3  # all three artifacts are still scanned

    groups = rank_runs(records, top_n=30, group_by=parse_group_by("model,policy"))
    runs = groups[0].runs
    assert len(runs) == 1  # ...but collapse to a single ranked row
    assert runs[0].duplicate_count == 3
    assert runs[0].run_id.isdigit() and len(runs[0].run_id) == 6

    # The id is stable (same content -> same id every time).
    again = rank_runs(scan_runs(tmp_path), top_n=30, group_by=parse_group_by("model,policy"))
    assert again[0].runs[0].run_id == runs[0].run_id

    report = format_markdown_report(groups, total_records=len(records), root=tmp_path)
    assert "| run |" in report  # run id is a first-class column
    assert runs[0].run_id in report


def test_scan_runs_uses_explicit_public_run_id_when_present(tmp_path):
    output_dir = tmp_path / "evals" / "der_goat"
    summary_path = tmp_path / "evals" / "der_goat.jsonl"
    artifact = _artifact(
        seed=0,
        checkpoint_index=4,
        total_reward=0.0,
        frames=(
            Frame(0.0, 200.0, 560.0, 0.0, 120.0, False, 0, False, False),
            Frame(0.1, 360.0, 600.0, 0.1, 130.0, False, 4, False, False),
        ),
        valid=True,
        lap_time=3.959,
    )
    artifact_path = save_rollout(
        artifact,
        output_dir / "track_01_easy_loop_human_keyboard_ep0000_seed_0_000239.json",
    )
    append_rollout_summary(
        artifact.summary,
        summary_path,
        artifact_path=artifact_path,
        policy="human_keyboard",
        episode=0,
        model="DER GOAT",
        extra={"run_id": "23"},
    )

    records = scan_runs(tmp_path)
    run = find_run_by_id(records, "23")

    assert run.model == "DER GOAT"
    assert run.policy == "human_keyboard"
    assert run.lap_time == 3.959


def test_equal_lap_and_reward_outcomes_collapse_but_reward_differences_stay(tmp_path):
    """Portfolio leaderboard rows collapse only when time and reward both match."""
    output_dir = tmp_path / "evals" / "model_outcome"
    summary_path = tmp_path / "evals" / "model_outcome.jsonl"
    frames_a = (
        Frame(0.0, 200.0, 560.0, 0.0, 220.0, False, 0, False, False),
        Frame(0.1, 360.0, 570.0, 0.0, 300.0, False, 4, False, True),
    )
    frames_b = (
        Frame(0.0, 210.0, 550.0, 0.0, 220.0, False, 0, False, False),
        Frame(0.1, 370.0, 560.0, 0.0, 300.0, True, 4, False, True),
    )
    frames_c = (
        Frame(0.0, 220.0, 540.0, 0.0, 220.0, False, 0, False, False),
        Frame(0.1, 380.0, 550.0, 0.0, 300.0, False, 4, False, True),
    )
    for ep, seed, reward, frames in (
        (0, 1, 12.5, frames_a),
        (1, 2, 12.5, frames_b),
        (2, 3, 13.0, frames_c),
    ):
        art = _artifact(
            seed=seed,
            checkpoint_index=4,
            total_reward=reward,
            frames=frames,
            valid=True,
            lap_time=4.0204,
        )
        path = save_rollout(
            art,
            output_dir / f"track_01_easy_loop_ppo_eval_ep{ep:04d}_seed_{seed}_000241.json",
        )
        append_rollout_summary(
            art.summary, summary_path, artifact_path=path, policy="ppo_eval", episode=ep
        )

    groups = rank_runs(scan_runs(tmp_path), top_n=30, group_by=parse_group_by("all"))
    runs = groups[0].runs

    assert len(runs) == 2
    assert sorted(run.duplicate_count for run in runs) == [1, 2]
    assert sorted(run.total_reward for run in runs) == [12.5, 13.0]


def test_reward_component_totals_are_recomputed_from_artifact_actions(tmp_path):
    env = GhostlineEnv(
        EnvConfig(
            max_episode_steps=3,
            reward=RewardConfig(
                progress_scale=10.0,
                checkpoint_bonus=0.0,
                finish_bonus=0.0,
                time_penalty=-0.25,
                target_speed_scale=1.0,
                heading_alignment_scale=0.0,
            ),
        )
    )
    artifact = run_rollout(env, [1, 1, 1], seed=5)
    output_dir = tmp_path / "evals" / "model_reward"
    summary_path = tmp_path / "evals" / "model_reward.jsonl"
    artifact_path = save_rollout(artifact, output_dir / "rollout.json")
    append_rollout_summary(
        artifact.summary,
        summary_path,
        artifact_path=artifact_path,
        policy="ppo_eval",
        episode=0,
    )

    record = scan_runs(tmp_path)[0]

    assert record.reward_components
    assert record.reward_components["time"] == -0.75
    assert record.reward_components["progress"] > 0.0
    assert record.reward_components["target_speed"] > 0.0
    assert record.top_positive_reward is not None
    assert record.top_negative_reward == ("time", -0.75)
    report = format_markdown_report(
        rank_runs([record], top_n=1, group_by=parse_group_by("model,policy")),
        total_records=1,
        root=tmp_path,
    )
    assert "Reward components:" in report
    assert "target speed" in report


def test_markdown_and_visual_reports_show_actionable_feedback(tmp_path):
    _write_artifacts(tmp_path)
    records = scan_runs(tmp_path)
    groups = rank_runs(records, top_n=2, group_by=parse_group_by("model,policy"))

    report = format_markdown_report(groups, total_records=len(records), root=tmp_path)
    assert "stalled_on_wall" in report
    assert "1/4" in report
    assert "| lap |" in report  # lap time is a first-class column
    assert "top +" in report
    assert "top -" in report
    assert "model=model_a / policy=ppo_eval" in report

    visual = write_visual_report(groups, tmp_path / "analysis", limit=2)
    assert visual.index_path.exists()
    assert visual.overview_svg_path.exists()
    assert len(visual.run_svg_paths) == 2
    assert "<polyline" in visual.overview_svg_path.read_text(encoding="utf-8")
    run_svg = visual.run_svg_paths[0].read_text(encoding="utf-8")
    assert "#4d96ff" in run_svg  # drift segment
    assert "#32d583" in run_svg  # boost segment
    assert "#ff453a" in run_svg  # wall-contact segment


def test_trace_comparison_report_shows_sector_boost_wall_and_svg_links(tmp_path):
    left = _artifact(
        seed=1,
        checkpoint_index=4,
        total_reward=0.0,
        valid=True,
        lap_time=0.60,
        boosts_used=1,
        drift_time=0.10,
        path_distance=600.0,
        frames=(
            Frame(0.10, 100.0, 560.0, 0.0, 100.0, False, 0, False, False),
            Frame(0.20, 200.0, 560.0, 0.0, 200.0, False, 1, False, True),
            Frame(0.30, 300.0, 560.0, 0.0, 300.0, False, 1, False, True),
            Frame(0.40, 400.0, 560.0, 0.0, 250.0, True, 2, False, False),
            Frame(0.50, 500.0, 560.0, 0.0, 260.0, False, 3, False, False),
            Frame(0.60, 600.0, 560.0, 0.0, 270.0, False, 4, False, False),
        ),
    )
    right = _artifact(
        seed=2,
        checkpoint_index=4,
        total_reward=0.0,
        valid=True,
        lap_time=0.80,
        boosts_used=1,
        drift_time=0.20,
        wall_hits=1,
        path_distance=750.0,
        frames=(
            Frame(0.10, 100.0, 560.0, 0.0, 100.0, False, 0, False, False),
            Frame(0.30, 250.0, 560.0, 0.0, 180.0, False, 1, False, True),
            Frame(0.40, 370.0, 560.0, 0.0, 220.0, False, 1, False, True),
            Frame(0.55, 490.0, 560.0, 0.0, 180.0, True, 2, True, False),
            Frame(0.65, 610.0, 560.0, 0.0, 220.0, True, 3, False, False),
            Frame(0.80, 750.0, 560.0, 0.0, 240.0, False, 4, False, False),
        ),
    )
    left_path = save_rollout(left, tmp_path / "evals" / "left_model" / "left.json")
    right_path = save_rollout(right, tmp_path / "evals" / "right_model" / "right.json")
    append_rollout_summary(
        left.summary,
        tmp_path / "evals" / "left_model.jsonl",
        artifact_path=left_path,
        policy="human_keyboard",
        episode=0,
        model="Michi",
    )
    append_rollout_summary(
        right.summary,
        tmp_path / "evals" / "right_model.jsonl",
        artifact_path=right_path,
        policy="ppo_eval",
        episode=0,
        model="b7_test",
    )

    records = scan_runs(tmp_path)
    left_run = next(run for run in records if run.model == "Michi")
    right_run = find_run_by_id(records, next(run for run in records if run.model == "b7_test").run_id)
    visual_dir = tmp_path / "analysis"
    visual_dir.mkdir()
    (visual_dir / f"run_001_id{left_run.run_id}_Michi_human_keyboard.svg").write_text(
        "<svg/>", encoding="utf-8"
    )
    (visual_dir / f"run_002_id{right_run.run_id}_b7_test_ppo_eval.svg").write_text(
        "<svg/>", encoding="utf-8"
    )
    (visual_dir / "top_runs.svg").write_text("<svg/>", encoding="utf-8")

    comparison = compare_run_traces(left_run, right_run, visual_dir=visual_dir)
    report = format_trace_comparison_report(comparison, root=tmp_path)

    assert len(comparison.left.sectors) == 4
    assert comparison.left.sectors[0].label == "start->CP1"
    assert comparison.right.sectors[0].split_time == 0.30
    assert comparison.right.sectors[1].wall_frames == 1
    assert comparison.right.boosts[0].exit_speed == 180.0
    assert "Boost Windows" in report
    assert "Wall Contacts" in report
    assert "Left SVG" in report and "Right SVG" in report
    assert "start->CP1" in report
    assert "+150.0" in report  # headline path gap


def test_generate_training_reports_is_default_flow(tmp_path):
    """The training flow's report step writes one ranked table + overlay + per-run SVGs."""
    from pathlib import Path

    from momentum_lab.rl.train import generate_training_reports

    _write_artifacts(tmp_path)
    report = generate_training_reports(
        root=tmp_path,
        report_path=tmp_path / "analysis" / "ranked_runs.md",
        visual_dir=tmp_path / "analysis",
        top_n=20,
        visual_limit=20,
    )

    assert report["records"] == 2
    assert Path(report["ranked_runs"]).exists()
    assert Path(report["index_html"]).exists()
    assert Path(report["overview_svg"]).exists()
    # One clickable SVG per visualized run, in ranked order.
    assert len(report["run_svgs"]) == 2
    assert all(Path(p).exists() for p in report["run_svgs"])
    # Default group_by=() yields a single global best-N ranking, not per-model tables.
    assert "## all" in Path(report["ranked_runs"]).read_text(encoding="utf-8")


def test_analytics_module_cli_writes_report_and_visuals(tmp_path):
    _write_artifacts(tmp_path)
    report_path = tmp_path / "analysis" / "ranked.md"
    visual_dir = tmp_path / "analysis" / "visuals"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "momentum_lab.rl.analytics",
            "--root",
            str(tmp_path),
            "--top",
            "1",
            "--group-by",
            "model,policy",
            "--output",
            str(report_path),
            "--visual-dir",
            str(visual_dir),
        ],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "Ghostline RL Run Analytics" in result.stdout
    assert "stalled_on_wall" in result.stdout
    assert report_path.exists()
    assert (visual_dir / "index.html").exists()
    assert (visual_dir / "top_runs.svg").exists()
