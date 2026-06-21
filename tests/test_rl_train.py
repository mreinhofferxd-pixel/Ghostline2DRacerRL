"""B7.4 Stable-Baselines3 training/evaluation smoke tests."""

from __future__ import annotations

import importlib.util
import json

import pytest

from momentum_lab.rl import EnvConfig


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="Stable-Baselines3 training extra is not installed",
)


def test_ppo_train_smoke_saves_model_and_eval_artifact(tmp_path):
    from momentum_lab.rl.train import evaluate_model, train_ppo

    model_path = tmp_path / "model.zip"
    train_result = train_ppo(
        total_timesteps=16,
        seed=1,
        env_config=EnvConfig(max_episode_steps=20),
        model_path=model_path,
        n_steps=16,
        batch_size=8,
        verbose=0,
    )
    assert train_result.model_path == model_path
    assert model_path.exists()
    assert train_result.manifest_path is not None
    assert train_result.manifest_path.exists()
    manifest = json.loads(train_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kind"] == "ghostline_rl_training_manifest"
    assert manifest["model_path"] == str(model_path)
    assert manifest["total_timesteps"] == 16
    assert manifest["seed"] == 1
    assert manifest["algorithm"] == "PPO"
    assert manifest["hyperparameters"]["n_steps"] == 16
    assert manifest["hyperparameters"]["batch_size"] == 8
    assert manifest["env_config"]["reward"]["version"]
    assert manifest["physics"]["fingerprint"]

    summary_path = tmp_path / "eval_summaries.jsonl"
    eval_result = evaluate_model(
        model_path,
        episodes=1,
        seed=2,
        env_config=EnvConfig(max_episode_steps=20),
        output_dir=tmp_path / "evals",
        summary_path=summary_path,
    )
    assert eval_result.episodes == 1
    assert eval_result.summary_path.exists()
    assert len(list(eval_result.artifacts_dir.glob("*.json"))) == 1
    row = json.loads(summary_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["model"] == model_path.stem
    assert row["model_path"] == str(model_path)
    assert row["deterministic"] is True


def test_deterministic_eval_collapses_to_single_episode(tmp_path):
    """B7.4b: deterministic eval must not emit N identical rollouts.

    With a fixed spawn + deterministic sim + arg-max policy there is no source of
    variation, so asking for many episodes should still produce exactly one rollout
    and one summary row rather than duplicates.
    """
    from momentum_lab.rl.train import evaluate_model, train_ppo

    model_path = tmp_path / "model.zip"
    train_ppo(
        total_timesteps=16,
        seed=1,
        env_config=EnvConfig(max_episode_steps=20),
        model_path=model_path,
        n_steps=16,
        batch_size=8,
        verbose=0,
    )

    summary_path = tmp_path / "eval_summaries.jsonl"
    eval_result = evaluate_model(
        model_path,
        episodes=5,
        seed=2,
        env_config=EnvConfig(max_episode_steps=20),
        deterministic=True,
        output_dir=tmp_path / "evals",
        summary_path=summary_path,
    )
    assert eval_result.episodes == 1
    assert len(list(eval_result.artifacts_dir.glob("*.json"))) == 1
    assert len(summary_path.read_text(encoding="utf-8").splitlines()) == 1


def test_ppo_train_keep_best_writes_selected_checkpoint_metadata(tmp_path):
    from momentum_lab.rl.train import evaluate_model, train_ppo

    model_path = tmp_path / "selected.zip"
    best_model_path = tmp_path / "best_checkpoint.zip"
    train_result = train_ppo(
        total_timesteps=16,
        seed=3,
        env_config=EnvConfig(max_episode_steps=20),
        model_path=model_path,
        n_steps=16,
        batch_size=8,
        keep_best=True,
        best_eval_freq=8,
        best_model_path=best_model_path,
        verbose=0,
    )

    assert train_result.model_path == model_path
    assert model_path.exists()
    assert best_model_path.exists()
    assert train_result.best_eval is not None

    manifest = json.loads(train_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["strategy"] == "best_deterministic_eval"
    assert manifest["selection"]["best_eval_freq"] == 8
    assert manifest["selection"]["best_model_path"] == str(best_model_path)
    assert manifest["selection"]["best_eval"]["episode_steps"] > 0

    eval_result = evaluate_model(
        model_path,
        episodes=1,
        seed=4,
        env_config=EnvConfig(max_episode_steps=20),
        output_dir=tmp_path / "evals",
        summary_path=tmp_path / "summary.jsonl",
    )
    assert eval_result.episodes == 1


def test_train_save_trace_checkpoints_emits_consumable_artifacts(tmp_path):
    """B7.6b: opt-in trace recorder writes per-step rollout artifacts the viewer can select."""
    import json

    from momentum_lab.rl.analytics import scan_runs
    from momentum_lab.rl.train import train_ppo
    from momentum_lab.rl.visualizer import select_runs

    model_path = tmp_path / "model.zip"
    trace_dir = tmp_path / "trace"
    trace_summary = tmp_path / "trace.jsonl"
    train_result = train_ppo(
        total_timesteps=32,
        seed=1,
        env_config=EnvConfig(max_episode_steps=12),
        model_path=model_path,
        n_steps=16,
        batch_size=8,
        save_trace_checkpoints=True,
        trace_eval_freq=16,
        trace_output_dir=trace_dir,
        trace_summary_path=trace_summary,
        verbose=0,
    )

    # Step-0 trace plus at least one interval trace.
    trace_files = list(trace_dir.glob("*.json"))
    assert len(trace_files) >= 2
    assert trace_summary.exists()
    rows = [json.loads(line) for line in trace_summary.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows and all(row["policy"] == "ppo_trace" for row in rows)
    assert {row["trace_step"] for row in rows} >= {0}

    # The manifest records the trace provenance.
    manifest = json.loads(train_result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["trace"]["policy"] == "ppo_trace"
    assert manifest["trace"]["trace_eval_freq"] == 16

    # Consumable by the same read-only viewer selection layer.
    records = scan_runs(tmp_path)
    traced = select_runs(records, policies=["ppo_trace"])
    assert traced
    assert all(r.policy == "ppo_trace" for r in traced)


def test_visualizer_reevaluate_materializes_missing_rollout(tmp_path):
    """B7.6b: re-evaluation builds a standard rollout artifact for a model with none."""
    from momentum_lab.rl.analytics import scan_runs
    from momentum_lab.rl.train import train_ppo
    from momentum_lab.rl.visualizer import reevaluate_models, select_runs

    model_path = tmp_path / "models" / "m.zip"
    train_ppo(
        total_timesteps=16,
        seed=1,
        env_config=EnvConfig(max_episode_steps=12),
        model_path=model_path,
        n_steps=16,
        batch_size=8,
        verbose=0,
    )
    # No eval rollout artifact exists yet for this model.
    assert not list(tmp_path.glob("evals/**/*.json"))

    stems = reevaluate_models([model_path], root=tmp_path)
    assert stems == ["m"]

    records = scan_runs(tmp_path)
    materialized = [r for r in records if r.model == "m" and r.artifact is not None]
    assert materialized
    # And the materialized run is selectable/visualizable.
    assert any(r.model == "m" for r in select_runs(records, models=["m"]))
