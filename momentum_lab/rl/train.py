"""Stable-Baselines3 training/evaluation entry point for Ghostline.

B7.4 starts real model training, but keeps the training dependency optional and
outside the deterministic core. The env remains the same ``GhostlineEnv`` used by
the smoke/batch rollout harness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .. import PHYSICS_VERSION, config
from ..physics_identity import physics_config_fingerprint, physics_config_payload
from .env import EnvConfig, GhostlineEnv
from .observations import ObservationConfig
from .rewards import RewardConfig
from .rollout import (
    RolloutArtifact,
    RolloutRecorder,
    append_rollout_summary,
    episode_rollout_path,
    rl_runs_dir,
    rollouts_dir,
    save_rollout,
    trace_rollout_path,
)


def models_dir() -> Path:
    return rl_runs_dir() / "models"


def evals_dir() -> Path:
    return rl_runs_dir() / "evals"


def analysis_dir() -> Path:
    return rl_runs_dir() / "analysis"


def default_model_path(track_id: str = config.DEFAULT_TRACK, seed: int = 0) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in track_id)
    return models_dir() / f"ppo_{safe}_seed_{seed}.zip"


def training_manifest_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(".manifest.json")


def first_lap_reward() -> RewardConfig:
    """Denser reward for the first-learning milestone.

    It still uses only target-gate geometry and physics state. No authored racing
    line, ghost, or teacher path is exposed to the agent.

    ``reward_first_lap_v3`` is progress-dominant: continuous progress toward the
    next gate is scaled so completing a full stage is worth roughly a checkpoint
    bonus, which makes total reward track "how far around the track did the car
    get" instead of being dominated by wall penalties. v2's heavy continuous wall
    terms (``wall_scrape``/``wall_proximity``) inverted the ranking -- they
    punished the racing line, which necessarily hugs walls, so a timid agent that
    crawled to the first checkpoint and parked in open space out-scored agents
    that actually drove most of the way around. v3 keeps a discrete impact
    penalty (``wall_hit``) plus a light scrape cost and otherwise lets physics do
    the work: hitting a wall kills speed, which already costs progress reward. The
    continuous proximity penalty and the heading-alignment term (which punished
    drifting through corners) are disabled.
    """
    return RewardConfig(
        version="reward_first_lap_v3",
        progress_scale=140.0,
        checkpoint_bonus=25.0,
        finish_bonus=500.0,
        time_penalty=-0.002,
        wall_hit_penalty=-8.0,
        wall_scrape_penalty_per_second=-0.5,
        wall_proximity_penalty=0.0,
        wall_proximity_threshold=0.0,
        target_speed_scale=0.15,
        heading_alignment_scale=0.0,
    )


def time_attack_reward() -> RewardConfig:
    """Stage-2 curriculum reward: keep finishing, but optimize lap time.

    Intended for *warm-started* fine-tuning of a model that already completes a
    valid lap under ``reward_first_lap_v3``. The finish/progress structure is kept
    so the agent does not trade the lap away, but the per-step ``time_penalty`` and
    target-directed ``target_speed`` reward are raised so a faster lap (fewer steps,
    more carried speed) scores higher. The time penalty only does useful work once
    the agent reliably finishes: shorter episodes accrue less penalty, so it becomes
    a real gradient toward speed instead of a flat tax on every non-finisher.

    ``reward_time_attack_v2`` fixed a misalignment found in v1 (a -8 ``wall_hit``
    let a clean-but-slow 6.50s lap out-score the 5.15s best). It softened the wall
    penalty to -3 and raised time/speed pressure, which made reward monotonic in lap
    time *at equal wall hits*. But v2 still valued a wall hit at ~0.6s of lap time
    (3 / 0.08 = 37.5 steps), so it preferred a cleaner line up to ~0.6s slower and
    its fine-tunes clustered at ~5.2s clean laps without ever beating 5.15s. The
    per-step ``time_penalty`` is also a weak, noisy proxy for the real objective.

    ``reward_time_attack_v3`` chased lap time directly. It added a one-shot
    ``finish_time`` bonus paid when the lap closes: ``finish_time_bonus_scale`` per
    control step the lap came in under ``finish_time_reference_steps`` (450 steps =
    7.5s, the first valid lap). At scale 2.0 a 310-step (5.15s) lap earns +280 vs a
    344-step (5.72s) lap's +212 -- a 68-point gap that dwarfs any wall-hit term, so
    raw lap time, not wall cleanliness, drives the ranking. The discrete wall penalty
    is softened further to -1.5 (a brush now costs ~0.19s of equivalent bonus);
    genuine crashes are still discouraged mostly via physics (an impact kills speed,
    adding steps and shrinking the finish-time bonus). Progress/checkpoint/finish and
    the per-step time/speed pressure are kept so the agent still finishes reliably.

    ``reward_time_attack_v4`` keeps lap time as the main objective, but adds a
    finish-gated racing-efficiency bonus: normalized average speed plus normalized
    path efficiency. This targets the observed post-boost-2 failure where the best
    policies run too wide into the outer wall before checkpoint 2. The efficiency
    terms are small compared with the finish-time bonus, so the agent should seek
    the sweet spot (fast and compact), not a timid shortest path.

    ``reward_time_attack_v5`` acts on the b7_19-vs-human sector trace
    (``runs/rl/analysis/michi_vs_b7_19_trace.md``). At 4.300s the policy's whole
    ~0.27s deficit was in the three middle corners (CP1->CP2->CP3) and had one
    shared cause -- a consistently *wider* line (+33/+43/+36 px per corner) -- plus
    an over-drift specifically into CP2 (drift +0.083s, lower min/avg speed). v4's
    finish-gated path-efficiency term had almost no leverage there (its [-1,1] clamp
    against a loose 2500 px reference made the human's tighter line worth only ~28
    points more than the wide line). v5 keeps the finish-time bonus as the dominant
    objective but (a) strengthens path efficiency (higher scale, reference pulled in
    toward the achievable ~2200-2300 px range) so a wide lap is clearly costlier at
    the finish, and (b) adds a small per-step ``drift_penalty_per_second`` so the
    agent stops drifting when it does not buy speed. Both are deliberately kept
    smaller than the lap-time signal so the agent still chases the fast line, not a
    timid one. Constants were validated by re-scoring the saved human/RL rollouts
    under v5 (the ``analytics._reward_component_totals`` replay trick) before
    training, to confirm v5 prefers the tighter/faster line without inverting the
    real lap-time ranking.

    ``reward_time_attack_v6`` chases the last few ticks to the human lap. Under v5
    the chain plateaued at exactly 4.100 s (246 control ticks) across six seeds -- a
    structural ceiling, not a search problem. The b7_27 trace vs the human shows the
    gap is ~4 ticks in one place: CP2->CP3 apex speed (543 vs 603). b7_27's *total*
    path is already about human-tight, but distributed wrong (tighter than the human
    early, wider late through CP2->CP3). v6 sharpens two warm-start-safe levers toward
    a faster *inside* line through the slow corner: the path-efficiency gradient is
    strengthened (scale 800->1100, reference pulled in 2350->2250 toward the human's
    ~2214 px line, so each extra px off the inside costs more), and ``target_speed``
    (per-step speed toward the gate) is raised 0.35->0.50 to reward carrying apex
    speed. Validated with the same replay trick (human/tight runs still rank top,
    lap-time order preserved among the fast runs). If v6 also re-converges to 246
    ticks, the limit is informational (the policy sees only the next gate, not the
    corner after it) and the next lever is richer observations, not reward.
    """
    return RewardConfig(
        version="reward_time_attack_v6",
        progress_scale=140.0,
        checkpoint_bonus=25.0,
        finish_bonus=500.0,
        time_penalty=-0.08,
        wall_hit_penalty=-6.0,
        wall_scrape_penalty_per_second=-2.0,
        wall_proximity_penalty=0.0,
        wall_proximity_threshold=0.0,
        target_speed_scale=0.50,
        heading_alignment_scale=0.0,
        finish_time_bonus_scale=2.0,
        finish_time_reference_steps=450.0,
        avg_speed_bonus_scale=400.0,
        avg_speed_reference=520.0,
        path_efficiency_bonus_scale=1100.0,
        path_distance_reference=2250.0,
        drift_penalty_per_second=-12.0,
    )


def first_lap_env_config(
    max_episode_steps: int = EnvConfig().max_episode_steps,
    include_wall_sensors: bool = False,
) -> EnvConfig:
    return EnvConfig(
        max_episode_steps=max_episode_steps,
        action_adapter="drive_discrete",
        reward=first_lap_reward(),
        observation=ObservationConfig(
            include_lookahead=True, include_wall_sensors=include_wall_sensors
        ),
    )


def time_attack_env_config(
    max_episode_steps: int = EnvConfig().max_episode_steps,
    include_wall_sensors: bool = False,
) -> EnvConfig:
    return EnvConfig(
        max_episode_steps=max_episode_steps,
        action_adapter="drive_discrete",
        reward=time_attack_reward(),
        observation=ObservationConfig(
            include_lookahead=True, include_wall_sensors=include_wall_sensors
        ),
    )


def _require_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install with "
            "`python -m pip install -e .[train]`."
        ) from e
    return PPO, Monitor


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    total_timesteps: int
    seed: int
    manifest_path: Path | None = None
    best_eval: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "model_path": str(self.model_path),
            "total_timesteps": self.total_timesteps,
            "seed": self.seed,
        }
        if self.manifest_path is not None:
            data["manifest_path"] = str(self.manifest_path)
        if self.best_eval is not None:
            data["best_eval"] = self.best_eval
        return data


@dataclass(frozen=True)
class EvalResult:
    episodes: int
    completed: int
    best_lap_time: float | None
    best_checkpoint: int
    checkpoint_count: int
    artifacts_dir: Path
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "completed": self.completed,
            "best_lap_time": self.best_lap_time,
            "best_checkpoint": self.best_checkpoint,
            "checkpoint_count": self.checkpoint_count,
            "artifacts_dir": str(self.artifacts_dir),
            "summary_path": str(self.summary_path),
        }


@dataclass(frozen=True)
class PolicyEval:
    valid: bool
    lap_time: float | None
    checkpoint_index: int
    checkpoint_count: int
    total_reward: float
    episode_steps: int
    wall_hits: int
    final_reason: str | None
    path_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "lap_time": self.lap_time,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_count": self.checkpoint_count,
            "total_reward": self.total_reward,
            "episode_steps": self.episode_steps,
            "wall_hits": self.wall_hits,
            "final_reason": self.final_reason,
            "path_distance": self.path_distance,
        }


def make_sb3_env(env_config: EnvConfig):
    """Build a monitored env for SB3 without leaking SB3 imports elsewhere."""
    _PPO, Monitor = _require_sb3()
    return Monitor(GhostlineEnv(env_config))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _env_config_payload(env_config: EnvConfig) -> dict[str, Any]:
    return {
        "track_id": env_config.track_id,
        "max_episode_steps": env_config.max_episode_steps,
        "action_adapter": env_config.action_adapter,
        "observation": asdict(env_config.observation),
        "reward": env_config.reward.payload(),
    }


def env_config_from_manifest(manifest: dict[str, Any]) -> EnvConfig:
    """Reconstruct an ``EnvConfig`` from a training manifest payload.

    Inverse of :func:`_env_config_payload`. Used to re-evaluate a saved model with
    the same observation/action/reward shape it was trained under (the policy
    expects a matching observation vector). Unknown keys are ignored so older
    manifests still load; missing sections fall back to ``EnvConfig`` defaults.
    """
    env = manifest.get("env_config") if isinstance(manifest, dict) else None
    if not isinstance(env, dict):
        return first_lap_env_config()
    kwargs: dict[str, Any] = {}
    if isinstance(env.get("track_id"), str):
        kwargs["track_id"] = env["track_id"]
    if isinstance(env.get("max_episode_steps"), int):
        kwargs["max_episode_steps"] = env["max_episode_steps"]
    if isinstance(env.get("action_adapter"), str):
        kwargs["action_adapter"] = env["action_adapter"]
    obs = env.get("observation")
    if isinstance(obs, dict):
        obs_fields = ObservationConfig.__dataclass_fields__
        kwargs["observation"] = ObservationConfig(
            **{k: obs[k] for k in obs if k in obs_fields}
        )
    reward = env.get("reward")
    if isinstance(reward, dict):
        reward_fields = RewardConfig.__dataclass_fields__
        kwargs["reward"] = RewardConfig(
            **{k: reward[k] for k in reward if k in reward_fields}
        )
    return EnvConfig(**kwargs)


def _action_adapter_kind_for_space(action_space: Any) -> str | None:
    """Map a discrete action-space size back to its adapter kind, or None."""
    from .actions import DISCRETE_ACTIONS, DRIVE_DISCRETE_ACTIONS

    n = getattr(action_space, "n", None)
    if n is None:
        return None
    n = int(n)
    if n == len(DRIVE_DISCRETE_ACTIONS):
        return "drive_discrete"
    if n == len(DISCRETE_ACTIONS):
        return "discrete"
    return None


def infer_env_config_for_model(model_path: str | Path) -> EnvConfig:
    """Best-effort env config to re-evaluate a saved model under (B7.6b).

    Prefers the training manifest (exact observation/action/reward shape). When no
    manifest exists, falls back to the first-lap config but matches the saved
    policy's action space so ``model.predict`` outputs stay in-range (older models
    used the larger ``discrete`` adapter, not ``drive_discrete``). Requires SB3 only
    on the no-manifest path.
    """
    model_path = Path(model_path)
    manifest_path = training_manifest_path(model_path)
    if not manifest_path.exists() and model_path.name.endswith(".best.zip"):
        manifest_path = model_path.with_name(model_path.name.removesuffix(".best.zip") + ".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return env_config_from_manifest(manifest)
    PPO, _Monitor = _require_sb3()
    model = PPO.load(model_path)
    config = first_lap_env_config()
    adapter = _action_adapter_kind_for_space(model.action_space)
    if adapter is not None and adapter != config.action_adapter:
        config = replace(config, action_adapter=adapter)
    return config


def write_training_manifest(
    *,
    model_path: str | Path,
    total_timesteps: int,
    seed: int,
    env_config: EnvConfig,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    algo: str = "PPO",
    policy: str = "MlpPolicy",
    init_model_path: str | Path | None = None,
    keep_best: bool = False,
    best_eval_freq: int | None = None,
    best_model_path: str | Path | None = None,
    best_eval: PolicyEval | None = None,
    save_trace_checkpoints: bool = False,
    trace_eval_freq: int | None = None,
    trace_output_dir: str | Path | None = None,
    trace_summary_path: str | Path | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write machine-readable provenance for a saved training run."""
    model_path = Path(model_path)
    manifest_path = Path(path) if path is not None else training_manifest_path(model_path)
    payload = {
        "schema_version": 1,
        "kind": "ghostline_rl_training_manifest",
        "created_at": _utc_now(),
        "model_path": str(model_path),
        "init_model_path": None if init_model_path is None else str(init_model_path),
        "algorithm": algo,
        "policy": policy,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "ent_coef": ent_coef,
        },
        "env_config": _env_config_payload(env_config),
        "physics": {
            "version": PHYSICS_VERSION,
            "fingerprint": physics_config_fingerprint(config.CAR),
            "config": physics_config_payload(config.CAR),
        },
        "dependencies": {
            "gymnasium": _package_version("gymnasium"),
            "stable_baselines3": _package_version("stable_baselines3"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
        },
    }
    if keep_best:
        payload["selection"] = {
            "strategy": "best_deterministic_eval",
            "best_eval_freq": best_eval_freq,
            "best_model_path": None if best_model_path is None else str(best_model_path),
            "best_eval": None if best_eval is None else best_eval.to_dict(),
        }
    if save_trace_checkpoints:
        payload["trace"] = {
            "policy": "ppo_trace",
            "trace_eval_freq": trace_eval_freq,
            "trace_output_dir": None if trace_output_dir is None else str(trace_output_dir),
            "trace_summary_path": None if trace_summary_path is None else str(trace_summary_path),
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _evaluate_policy_snapshot(
    model: Any,
    env_config: EnvConfig,
    *,
    seed: int = 10_000,
    deterministic: bool = True,
) -> PolicyEval:
    """Evaluate a policy in-memory for checkpoint selection without artifacts."""
    env = GhostlineEnv(env_config)
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
    result = PolicyEval(
        valid=bool(terminated and info.get("final_reason") == "lap_complete"),
        lap_time=info.get("lap_time"),
        checkpoint_index=int(info.get("checkpoint_index", 0)),
        checkpoint_count=int(info.get("checkpoint_count", 0)),
        total_reward=total_reward,
        episode_steps=int(info.get("episode_steps", 0)),
        wall_hits=int(info.get("wall_hits", 0)),
        final_reason=info.get("final_reason"),
        path_distance=float(info.get("path_distance", 0.0)),
    )
    env.close()
    return result


def _record_policy_rollout(
    model: Any,
    env_config: EnvConfig,
    *,
    seed: int,
    deterministic: bool = True,
) -> RolloutArtifact:
    """Play a policy for one episode and capture a canonical rollout artifact.

    Like :func:`_evaluate_policy_snapshot`, but records the ``Action`` stream and
    derived frame cache so the result is a standard rollout artifact (consumable by
    ``analytics.scan_runs`` and the interactive viewer), not just outcome metrics.
    """
    env = GhostlineEnv(env_config)
    obs, _info = env.reset(seed=seed)
    recorder = RolloutRecorder.start(env)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, _info = recorder.step(env, action)
    artifact = recorder.to_artifact(env)
    env.close()
    return artifact


def _policy_eval_is_better(candidate: PolicyEval, incumbent: PolicyEval | None) -> bool:
    """Rank valid laps by lap time, then use progress/reward for non-finishers."""
    if incumbent is None:
        return True
    if candidate.valid != incumbent.valid:
        return candidate.valid
    if candidate.valid:
        if candidate.lap_time is None:
            return False
        if incumbent.lap_time is None:
            return True
        if abs(candidate.lap_time - incumbent.lap_time) > 1e-9:
            return candidate.lap_time < incumbent.lap_time
        if candidate.wall_hits != incumbent.wall_hits:
            return candidate.wall_hits < incumbent.wall_hits
        return candidate.total_reward > incumbent.total_reward
    if candidate.checkpoint_index != incumbent.checkpoint_index:
        return candidate.checkpoint_index > incumbent.checkpoint_index
    if abs(candidate.total_reward - incumbent.total_reward) > 1e-9:
        return candidate.total_reward > incumbent.total_reward
    return candidate.episode_steps < incumbent.episode_steps


def _keep_best_callback(
    *,
    env_config: EnvConfig,
    checkpoint_path: Path,
    eval_freq: int,
    eval_seed: int,
    best_eval_ref: list[PolicyEval | None],
):
    from stable_baselines3.common.callbacks import BaseCallback

    class KeepBestCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._last_eval_step = 0

        def _on_step(self) -> bool:
            if self.n_calls - self._last_eval_step < eval_freq:
                return True
            self._last_eval_step = self.n_calls
            candidate = _evaluate_policy_snapshot(
                self.model,
                env_config,
                seed=eval_seed,
                deterministic=True,
            )
            if _policy_eval_is_better(candidate, best_eval_ref[0]):
                best_eval_ref[0] = candidate
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(checkpoint_path)
            return True

    return KeepBestCallback()


def _trace_recorder_callback(*, record: Any, eval_freq: int):
    """Callback that records a checkpoint-eval trace every ``eval_freq`` steps."""
    from stable_baselines3.common.callbacks import BaseCallback

    class TraceRecorderCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._last_eval_step = 0

        def _on_step(self) -> bool:
            if self.n_calls - self._last_eval_step < eval_freq:
                return True
            self._last_eval_step = self.n_calls
            record(self.num_timesteps)
            return True

    return TraceRecorderCallback()


def train_ppo(
    *,
    total_timesteps: int,
    seed: int = 0,
    env_config: EnvConfig | None = None,
    model_path: str | Path | None = None,
    init_model_path: str | Path | None = None,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    gamma: float = 0.995,
    ent_coef: float = 0.01,
    keep_best: bool = False,
    best_eval_freq: int = 10_000,
    best_eval_seed: int = 10_000,
    best_model_path: str | Path | None = None,
    save_trace_checkpoints: bool = False,
    trace_eval_freq: int = 10_000,
    trace_eval_seed: int | None = None,
    trace_output_dir: str | Path | None = None,
    trace_summary_path: str | Path | None = None,
    verbose: int = 1,
) -> TrainResult:
    """Train a PPO model and save it.

    When ``init_model_path`` is given, training warm-starts from that saved model
    (continued/fine-tune training) instead of a fresh random policy. The new
    ``env_config`` still applies, so this is how a finishing policy is fine-tuned
    under a stage-2 reward (e.g. ``reward_time_attack_v1``) without re-running the
    seed lottery. The policy/action/observation shapes must match the saved model.

    With ``save_trace_checkpoints`` (off by default; extra I/O per eval), a
    deterministic eval rollout is recorded as a standard rollout artifact every
    ``trace_eval_freq`` steps (plus once at step 0), giving an intra-run "learning
    over steps" timeline the interactive viewer can select and overlay. The trace
    rollouts are tagged ``policy="ppo_trace"`` with a per-step model name.
    """
    PPO, _Monitor = _require_sb3()
    env_config = env_config or first_lap_env_config()
    env = make_sb3_env(env_config)
    if init_model_path is not None:
        model = PPO.load(init_model_path, env=env)
        model.ent_coef = ent_coef
        model.seed = seed
        model.set_random_seed(seed)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            ent_coef=ent_coef,
            seed=seed,
            verbose=verbose,
        )
    path = Path(model_path) if model_path is not None else default_model_path(env_config.track_id, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = (
        Path(best_model_path)
        if best_model_path is not None
        else path.with_name(f"{path.stem}.best{path.suffix}")
    )
    best_eval_ref: list[PolicyEval | None] = [None]
    callbacks: list[Any] = []
    if keep_best:
        initial_eval = _evaluate_policy_snapshot(model, env_config, seed=best_eval_seed, deterministic=True)
        best_eval_ref[0] = initial_eval
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(best_checkpoint_path)
        callbacks.append(
            _keep_best_callback(
                env_config=env_config,
                checkpoint_path=best_checkpoint_path,
                eval_freq=best_eval_freq,
                eval_seed=best_eval_seed,
                best_eval_ref=best_eval_ref,
            )
        )

    trace_output_path = (
        Path(trace_output_dir)
        if trace_output_dir is not None
        else evals_dir() / f"{path.stem}_trace"
    )
    trace_summary = (
        Path(trace_summary_path)
        if trace_summary_path is not None
        else evals_dir() / f"{path.stem}_trace.jsonl"
    )
    trace_seed = best_eval_seed if trace_eval_seed is None else trace_eval_seed
    trace_count = [0]

    def _record_trace(step: int) -> None:
        artifact = _record_policy_rollout(model, env_config, seed=trace_seed, deterministic=True)
        artifact_path = trace_rollout_path(
            trace_output_path,
            track_id=artifact.summary.track_id,
            policy="ppo_trace",
            step=step,
            seed=artifact.summary.seed,
            episode_steps=artifact.summary.episode_steps,
        )
        save_rollout(artifact, artifact_path)
        append_rollout_summary(
            artifact.summary,
            trace_summary,
            artifact_path=artifact_path,
            policy="ppo_trace",
            episode=trace_count[0],
            model=f"{path.stem}@{step:08d}",
            extra={"trace_step": step, "trace_source_model": path.stem},
        )
        trace_count[0] += 1

    if save_trace_checkpoints:
        # Capture the starting policy so the timeline begins before any updates.
        _record_trace(0)
        callbacks.append(
            _trace_recorder_callback(record=_record_trace, eval_freq=trace_eval_freq)
        )

    if not callbacks:
        callback = None
    elif len(callbacks) == 1:
        callback = callbacks[0]
    else:
        from stable_baselines3.common.callbacks import CallbackList

        callback = CallbackList(callbacks)

    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)
    if keep_best and best_checkpoint_path.exists():
        selected_model = PPO.load(best_checkpoint_path)
        selected_model.save(path)
    else:
        model.save(path)
    manifest_path = write_training_manifest(
        model_path=path,
        total_timesteps=total_timesteps,
        seed=seed,
        env_config=env_config,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        init_model_path=init_model_path,
        keep_best=keep_best,
        best_eval_freq=best_eval_freq if keep_best else None,
        best_model_path=best_checkpoint_path if keep_best else None,
        best_eval=best_eval_ref[0] if keep_best else None,
        save_trace_checkpoints=save_trace_checkpoints,
        trace_eval_freq=trace_eval_freq if save_trace_checkpoints else None,
        trace_output_dir=trace_output_path if save_trace_checkpoints else None,
        trace_summary_path=trace_summary if save_trace_checkpoints else None,
    )
    env.close()
    return TrainResult(
        model_path=path,
        total_timesteps=total_timesteps,
        seed=seed,
        manifest_path=manifest_path,
        best_eval=None if best_eval_ref[0] is None else best_eval_ref[0].to_dict(),
    )


def evaluate_model(
    model_path: str | Path,
    *,
    episodes: int = 5,
    seed: int = 10_000,
    env_config: EnvConfig | None = None,
    deterministic: bool = True,
    output_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> EvalResult:
    """Run a saved SB3 model and write rollout artifacts/summaries.

    Deterministic eval is collapsed to a single episode: with a fixed spawn, a
    deterministic sim, and an arg-max policy there is no source of variation, so N
    episodes would produce N byte-identical rollouts (B7.4b). Pass
    ``deterministic=False`` to sample the policy and run ``episodes`` distinct
    rollouts — that *is* informative, yielding a lap-time distribution for the same
    model. The per-episode reset seed (``seed + episode``) still has no effect until
    a seeded randomized reset exists, but the stochastic policy makes the episodes
    differ.
    """
    PPO, _Monitor = _require_sb3()
    env_config = env_config or first_lap_env_config()
    output_dir = Path(output_dir) if output_dir is not None else evals_dir()
    summary_path = Path(summary_path) if summary_path is not None else rl_runs_dir() / "eval_summaries.jsonl"
    model = PPO.load(model_path)

    # Deterministic rollouts are identical, so one is all the information there is.
    effective_episodes = 1 if deterministic else episodes

    completed = 0
    best_lap: float | None = None
    best_cp = 0
    checkpoint_count = 0
    for episode in range(effective_episodes):
        env = GhostlineEnv(env_config)
        obs, _info = env.reset(seed=seed + episode)
        recorder = RolloutRecorder.start(env)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, _info = recorder.step(env, action)
        artifact = recorder.to_artifact(env)
        checkpoint_count = artifact.summary.checkpoint_count
        best_cp = max(best_cp, artifact.summary.checkpoint_index)
        if artifact.summary.valid:
            completed += 1
            best_lap = (
                artifact.summary.lap_time
                if best_lap is None
                else min(best_lap, artifact.summary.lap_time)
            )
        artifact_path = episode_rollout_path(
            output_dir,
            track_id=artifact.summary.track_id,
            policy="ppo_eval",
            episode=episode,
            seed=artifact.summary.seed,
            episode_steps=artifact.summary.episode_steps,
        )
        save_rollout(artifact, artifact_path)
        append_rollout_summary(
            artifact.summary,
            summary_path,
            artifact_path=artifact_path,
            policy="ppo_eval",
            episode=episode,
            model=Path(model_path).stem,
            extra={
                "model_path": str(model_path),
                "deterministic": deterministic,
            },
        )
        env.close()

    return EvalResult(
        episodes=effective_episodes,
        completed=completed,
        best_lap_time=best_lap,
        best_checkpoint=best_cp,
        checkpoint_count=checkpoint_count,
        artifacts_dir=output_dir,
        summary_path=summary_path,
    )


def generate_training_reports(
    *,
    root: str | Path | None = None,
    report_path: str | Path | None = None,
    visual_dir: str | Path | None = None,
    top_n: int = 20,
    visual_limit: int = 20,
    group_by: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Scan saved RL runs and (re)write the ranked report + overlay/per-run SVGs.

    This is part of the *default* training flow: after every train/eval the run
    portfolio should be presentable as one ranked best-N table, an overlay SVG of
    the top trajectories, and a clickable per-run SVG each. ``group_by=()`` (the
    default) produces a single global best-N ranking rather than per-model tables.
    Returns the written paths (including ``run_svgs`` in ranked order) so callers
    can surface clickable links.
    """
    from .analytics import rank_runs, scan_runs, write_markdown_report, write_visual_report

    root = Path(root) if root is not None else rl_runs_dir()
    report_path = Path(report_path) if report_path is not None else analysis_dir() / "ranked_runs.md"
    visual_dir = Path(visual_dir) if visual_dir is not None else analysis_dir()
    records = scan_runs(root)
    groups = rank_runs(records, top_n=top_n, group_by=group_by)
    write_markdown_report(groups, report_path, total_records=len(records), root=root)
    visual = write_visual_report(groups, visual_dir, limit=visual_limit)
    return {
        "records": len(records),
        "ranked_runs": str(report_path),
        "index_html": str(visual.index_path),
        "overview_svg": str(visual.overview_svg_path),
        "run_svgs": [str(p) for p in visual.run_svg_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a Ghostline PPO model.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=EnvConfig().max_episode_steps)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--reward",
        choices=("first_lap", "time_attack"),
        default="first_lap",
        help="Reward profile: 'first_lap' (learn to finish) or 'time_attack' (fine-tune for speed).",
    )
    parser.add_argument(
        "--init-model-path",
        type=Path,
        default=None,
        help="Warm-start training from this saved model instead of a random init.",
    )
    parser.add_argument(
        "--include-wall-sensors",
        action="store_true",
        help="Add the target-gate-frame inside-wall apex sensors to the observation "
        "(B7.7 strict beat). Widens the obs dim, so models trained with it do NOT "
        "transfer to/from models trained without it -- needs a fresh curriculum. Pass "
        "the same flag when evaluating a model trained with it.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--eval-stochastic",
        action="store_true",
        help="Sample the policy during eval (deterministic=False) and run --eval-episodes "
        "distinct rollouts for a lap-time distribution. Default eval is deterministic and "
        "runs a single episode (identical rollouts add nothing).",
    )
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-output-dir", type=Path, default=evals_dir())
    parser.add_argument("--eval-summary-path", type=Path, default=rl_runs_dir() / "eval_summaries.jsonl")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Select the best deterministic eval checkpoint during training instead of final weights.",
    )
    parser.add_argument(
        "--best-eval-freq",
        type=int,
        default=10_000,
        help="Training steps between deterministic checkpoint-selection evals.",
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=None,
        help="Optional sidecar path for the best checkpoint saved during --keep-best training.",
    )
    parser.add_argument(
        "--save-trace-checkpoints",
        action="store_true",
        help="Record a deterministic eval rollout artifact every --trace-eval-freq steps "
        "(plus step 0) for an intra-run 'learning over steps' timeline the viewer can "
        "select (policy=ppo_trace). Off by default (extra I/O per eval).",
    )
    parser.add_argument(
        "--trace-eval-freq",
        type=int,
        default=10_000,
        help="Training steps between recorded trace-checkpoint evals.",
    )
    parser.add_argument(
        "--trace-output-dir",
        type=Path,
        default=None,
        help="Directory for trace rollout artifacts (default: evals/<model>_trace).",
    )
    parser.add_argument(
        "--trace-summary-path",
        type=Path,
        default=None,
        help="Summary JSONL for trace rollouts (default: evals/<model>_trace.jsonl).",
    )
    # Results presentation is part of the default flow: every train/eval refreshes
    # the ranked best-N table + overlay + per-run SVGs unless explicitly skipped.
    parser.add_argument("--no-report", action="store_true", help="Skip the post-run analytics report.")
    parser.add_argument("--report-top", type=int, default=20, help="Rows in the ranked table.")
    parser.add_argument("--report-visual-limit", type=int, default=20, help="Trajectories to render/overlay.")
    parser.add_argument("--report-root", type=Path, default=rl_runs_dir(), help="Where to scan for runs.")
    parser.add_argument("--report-output", type=Path, default=analysis_dir() / "ranked_runs.md")
    parser.add_argument("--report-visual-dir", type=Path, default=analysis_dir())
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.reward == "time_attack":
        env_config = time_attack_env_config(args.max_episode_steps, args.include_wall_sensors)
    else:
        env_config = first_lap_env_config(args.max_episode_steps, args.include_wall_sensors)
    model_path = args.model_path or default_model_path(env_config.track_id, args.seed)
    payload: dict[str, Any] = {}
    if not args.no_train:
        train_result = train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            env_config=env_config,
            model_path=model_path,
            init_model_path=args.init_model_path,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            keep_best=args.keep_best,
            best_eval_freq=args.best_eval_freq,
            best_eval_seed=args.eval_seed,
            best_model_path=args.best_model_path,
            save_trace_checkpoints=args.save_trace_checkpoints,
            trace_eval_freq=args.trace_eval_freq,
            trace_output_dir=args.trace_output_dir,
            trace_summary_path=args.trace_summary_path,
            verbose=0 if args.quiet else 1,
        )
        payload["train"] = train_result.to_dict()
    if not args.no_eval:
        eval_result = evaluate_model(
            model_path,
            episodes=args.eval_episodes,
            seed=args.eval_seed,
            env_config=env_config,
            deterministic=not args.eval_stochastic,
            output_dir=args.eval_output_dir,
            summary_path=args.eval_summary_path,
        )
        payload["eval"] = eval_result.to_dict()
    if not args.no_report:
        payload["report"] = generate_training_reports(
            root=args.report_root,
            report_path=args.report_output,
            visual_dir=args.report_visual_dir,
            top_n=args.report_top,
            visual_limit=args.report_visual_limit,
        )
    if not args.quiet:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
