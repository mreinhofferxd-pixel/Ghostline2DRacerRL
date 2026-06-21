"""RL rollout artifacts built on the canonical replay action stream.

An RL rollout stores training/evaluation metadata next to a normal replay payload.
The replay remains the source of truth for trajectory reproduction; the rollout
summary is for filtering, plotting, and later multi-run visualization.
"""

from __future__ import annotations

import json
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .. import TIMING_VERSION
from ..config import CAR, CONTROL_DT, CarPhysics
from ..core.action import Action
from ..physics_identity import physics_config_fingerprint
from ..replay import ReplayData, ReplayRecorder, trajectory_matches
from ..tracks import load_track_by_id
from .env import EnvConfig, GhostlineEnv


ROLLOUT_SCHEMA_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rl_runs_dir() -> Path:
    return project_root() / "runs" / "rl"


def rollouts_dir() -> Path:
    return rl_runs_dir() / "rollouts"


def rollout_summaries_path() -> Path:
    return rl_runs_dir() / "rollout_summaries.jsonl"


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def rollout_path(track_id: str, seed: int | None, episode_steps: int) -> Path:
    seed_part = "seed_none" if seed is None else f"seed_{seed}"
    return rollouts_dir() / f"{_safe_name(track_id)}_{seed_part}_{episode_steps:06d}.json"


def episode_rollout_path(
    output_dir: str | Path,
    *,
    track_id: str,
    policy: str,
    episode: int,
    seed: int | None,
    episode_steps: int,
) -> Path:
    seed_part = "seed_none" if seed is None else f"seed_{seed}"
    return (
        Path(output_dir)
        / f"{_safe_name(track_id)}_{_safe_name(policy)}_ep{episode:04d}_"
        f"{seed_part}_{episode_steps:06d}.json"
    )


def trace_rollout_path(
    output_dir: str | Path,
    *,
    track_id: str,
    policy: str,
    step: int,
    seed: int | None,
    episode_steps: int,
) -> Path:
    """Path for a checkpoint-eval ("learning over steps") trace rollout.

    Keyed by the training ``step`` rather than an episode index so the artifacts
    sort into the intra-run timeline the viewer overlays.
    """
    seed_part = "seed_none" if seed is None else f"seed_{seed}"
    return (
        Path(output_dir)
        / f"{_safe_name(track_id)}_{_safe_name(policy)}_step{step:08d}_"
        f"{seed_part}_{episode_steps:06d}.json"
    )


@dataclass(frozen=True)
class RolloutSummary:
    track_id: str
    seed: int | None
    terminated: bool
    truncated: bool
    final_reason: str | None
    episode_steps: int
    total_reward: float
    lap_time: float
    valid: bool
    checkpoint_index: int
    checkpoint_count: int
    wall_hits: int
    wall_scrape_time: float
    boosts_used: int
    drift_time: float
    peak_slip: float
    physics_version: str
    physics_fingerprint: str
    physics_config: dict[str, float]
    reward_version: str
    reward_config: dict[str, float | str]
    action_adapter: dict[str, object]
    path_distance: float = 0.0
    timing_version: str = TIMING_VERSION

    @classmethod
    def from_env(
        cls,
        env: GhostlineEnv,
        *,
        total_reward: float,
        terminated: bool,
        truncated: bool,
        final_info: dict[str, Any],
    ) -> "RolloutSummary":
        world = env.sim.world
        return cls(
            track_id=world.track.track_id,
            seed=env.sim.seed,
            terminated=terminated,
            truncated=truncated,
            final_reason=final_info.get("final_reason"),
            episode_steps=final_info.get("episode_steps", world.tick),
            total_reward=float(total_reward),
            lap_time=world.run.lap_time(world.tick, CONTROL_DT),
            valid=world.run.valid,
            checkpoint_index=world.run.next_cp,
            checkpoint_count=len(world.track.checkpoints),
            wall_hits=world.wall_hits,
            wall_scrape_time=world.wall_scrape_time,
            boosts_used=world.boosts_used,
            drift_time=world.drift_time,
            peak_slip=world.peak_slip,
            physics_version=final_info["physics_version"],
            physics_fingerprint=final_info["physics_fingerprint"],
            physics_config=dict(final_info["physics_config"]),
            reward_version=final_info["reward_version"],
            reward_config=dict(final_info["reward_config"]),
            action_adapter=dict(final_info["action_adapter"]),
            path_distance=world.path_distance,
            timing_version=TIMING_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "seed": self.seed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_reason": self.final_reason,
            "episode_steps": self.episode_steps,
            "total_reward": self.total_reward,
            "lap_time": self.lap_time,
            "valid": self.valid,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_count": self.checkpoint_count,
            "wall_hits": self.wall_hits,
            "wall_scrape_time": self.wall_scrape_time,
            "boosts_used": self.boosts_used,
            "drift_time": self.drift_time,
            "peak_slip": self.peak_slip,
            "path_distance": self.path_distance,
            "physics_version": self.physics_version,
            "physics_fingerprint": self.physics_fingerprint,
            "physics_config": {
                key: self.physics_config[key] for key in sorted(self.physics_config)
            },
            "reward_version": self.reward_version,
            "reward_config": {
                key: self.reward_config[key] for key in sorted(self.reward_config)
            },
            "action_adapter": self.action_adapter,
            "timing_version": self.timing_version,
        }

    def compact_dict(self) -> dict[str, Any]:
        """Small JSONL-friendly row for scanning many episodes."""
        return {
            "track_id": self.track_id,
            "seed": self.seed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_reason": self.final_reason,
            "episode_steps": self.episode_steps,
            "total_reward": self.total_reward,
            "lap_time": self.lap_time,
            "valid": self.valid,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_count": self.checkpoint_count,
            "wall_hits": self.wall_hits,
            "boosts_used": self.boosts_used,
            "drift_time": self.drift_time,
            "path_distance": self.path_distance,
            "reward_version": self.reward_version,
            "physics_fingerprint": self.physics_fingerprint,
            "timing_version": self.timing_version,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RolloutSummary":
        if not isinstance(raw, dict):
            raise ValueError("rollout summary: expected an object")
        return cls(
            track_id=_str(raw.get("track_id"), "summary.track_id"),
            seed=_optional_int(raw.get("seed"), "summary.seed"),
            terminated=_bool(raw.get("terminated"), "summary.terminated"),
            truncated=_bool(raw.get("truncated"), "summary.truncated"),
            final_reason=_optional_str(raw.get("final_reason"), "summary.final_reason"),
            episode_steps=_int(raw.get("episode_steps"), "summary.episode_steps"),
            total_reward=_number(raw.get("total_reward"), "summary.total_reward"),
            lap_time=_number(raw.get("lap_time"), "summary.lap_time"),
            valid=_bool(raw.get("valid"), "summary.valid"),
            checkpoint_index=_int(raw.get("checkpoint_index"), "summary.checkpoint_index"),
            checkpoint_count=_int(raw.get("checkpoint_count"), "summary.checkpoint_count"),
            wall_hits=_int(raw.get("wall_hits"), "summary.wall_hits"),
            wall_scrape_time=_number(raw.get("wall_scrape_time"), "summary.wall_scrape_time"),
            boosts_used=_int(raw.get("boosts_used"), "summary.boosts_used"),
            drift_time=_number(raw.get("drift_time"), "summary.drift_time"),
            peak_slip=_number(raw.get("peak_slip"), "summary.peak_slip"),
            path_distance=_number(raw.get("path_distance", 0.0), "summary.path_distance"),
            physics_version=_str(raw.get("physics_version"), "summary.physics_version"),
            physics_fingerprint=_str(
                raw.get("physics_fingerprint"), "summary.physics_fingerprint"
            ),
            physics_config=_number_dict(raw.get("physics_config"), "summary.physics_config"),
            reward_version=_str(raw.get("reward_version"), "summary.reward_version"),
            reward_config=_reward_config(raw.get("reward_config"), "summary.reward_config"),
            action_adapter=_object_dict(raw.get("action_adapter"), "summary.action_adapter"),
            # Legacy rollouts predate sub-tick timing: their lap_time is integer-tick.
            timing_version=_str(
                raw.get("timing_version", "timing_v1"), "summary.timing_version"
            ),
        )


@dataclass(frozen=True)
class RolloutArtifact:
    summary: RolloutSummary
    replay: ReplayData
    schema_version: int = ROLLOUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "ghostline_rl_rollout",
            "summary": self.summary.to_dict(),
            "replay": self.replay.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RolloutArtifact":
        if not isinstance(raw, dict):
            raise ValueError("rollout artifact: expected an object")
        schema_version = _int(raw.get("schema_version"), "schema_version")
        if schema_version != ROLLOUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported rollout schema {schema_version}; "
                f"expected {ROLLOUT_SCHEMA_VERSION}"
            )
        if raw.get("kind") != "ghostline_rl_rollout":
            raise ValueError("rollout artifact: expected kind='ghostline_rl_rollout'")
        return cls(
            schema_version=schema_version,
            summary=RolloutSummary.from_dict(raw.get("summary")),
            replay=ReplayData.from_dict(raw.get("replay")),
        )


class RolloutRecorder:
    """Record an env episode as replay actions plus RL metadata."""

    def __init__(self, env: GhostlineEnv) -> None:
        self.recorder = ReplayRecorder.start(env.sim)
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.final_info: dict[str, Any] | None = None

    @classmethod
    def start(cls, env: GhostlineEnv) -> "RolloutRecorder":
        return cls(env)

    def record_after_step(
        self,
        env: GhostlineEnv,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self.total_reward += float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.final_info = dict(info)
        self.recorder.record_after_step(env.sim, env.last_action)

    def step(self, env: GhostlineEnv, action):
        obs, reward, terminated, truncated, info = env.step(action)
        self.record_after_step(
            env,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return obs, reward, terminated, truncated, info

    def to_artifact(self, env: GhostlineEnv) -> RolloutArtifact:
        final_info = self.final_info
        if final_info is None:
            final_info = env._info(final_reason=None)  # outer-layer artifact helper
        replay = self.recorder.to_replay(env.sim)
        return RolloutArtifact(
            summary=RolloutSummary.from_env(
                env,
                total_reward=self.total_reward,
                terminated=self.terminated,
                truncated=self.truncated,
                final_info=final_info,
            ),
            replay=replay,
        )


def run_rollout(env: GhostlineEnv, actions, *, seed: int | None = None) -> RolloutArtifact:
    """Reset ``env``, play ``actions`` until done/exhausted, and return an artifact."""
    env.reset(seed=seed)
    recorder = RolloutRecorder.start(env)
    for action in actions:
        _obs, _reward, terminated, truncated, _info = recorder.step(env, action)
        if terminated or truncated:
            break
    return recorder.to_artifact(env)


def save_rollout(artifact: RolloutArtifact, path: str | Path | None = None) -> Path:
    if path is None:
        path = rollout_path(
            artifact.summary.track_id,
            artifact.summary.seed,
            artifact.summary.episode_steps,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_rollout_summary(
    summary: RolloutSummary,
    path: str | Path | None = None,
    *,
    artifact_path: str | Path | None = None,
    policy: str | None = None,
    episode: int | None = None,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path) if path is not None else rollout_summaries_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summary.compact_dict()
    if artifact_path is not None:
        row["artifact_path"] = str(artifact_path)
    if policy is not None:
        row["policy"] = policy
    if episode is not None:
        row["episode"] = episode
    if model is not None:
        row["model"] = model
    if extra is not None:
        for key, value in extra.items():
            if not isinstance(key, str) or not key:
                raise ValueError("extra summary keys must be non-empty strings")
            if key in row:
                raise ValueError(f"extra summary key {key!r} would overwrite an existing field")
            row[key] = value
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def load_rollout(path: str | Path) -> RolloutArtifact:
    return RolloutArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_rollout(
    artifact: RolloutArtifact,
    cfg: CarPhysics = CAR,
    *,
    strict_version: bool = True,
) -> bool:
    track = load_track_by_id(artifact.replay.track_id)
    expected_fingerprint = physics_config_fingerprint(cfg)
    if strict_version and artifact.summary.physics_fingerprint != expected_fingerprint:
        return False
    return trajectory_matches(
        artifact.replay,
        track,
        cfg,
        strict_version=strict_version,
    )


def check_gymnasium_env(env_config: EnvConfig | None = None) -> None:
    """Run Gymnasium's env checker when the optional dependency is installed."""
    try:
        from gymnasium.utils.env_checker import check_env
    except ModuleNotFoundError as e:
        raise RuntimeError("Gymnasium is not installed; install the 'rl' extra") from e
    if env_config is not None and env_config.max_episode_steps < 2:
        env_config = replace(env_config, max_episode_steps=2)
    check_env(GhostlineEnv(env_config), skip_render_check=True)


def run_policy_episode(
    *,
    policy: str,
    seed: int | None,
    env_config: EnvConfig,
) -> RolloutArtifact:
    env = GhostlineEnv(env_config)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    recorder = RolloutRecorder.start(env)
    cycle = (1, 2, 3, 9, 10, 1, 1, 4, 0, 3)
    for step in range(env_config.max_episode_steps):
        if policy == "random":
            action = env.sample_action()
        elif policy == "throttle":
            action = 1
        elif policy == "cycle":
            action = cycle[step % len(cycle)]
        else:
            raise ValueError(f"unknown rollout policy {policy!r}")
        _obs, _reward, terminated, truncated, _info = recorder.step(env, action)
        if terminated or truncated:
            break
    return recorder.to_artifact(env)


def run_batch(
    *,
    episodes: int,
    policy: str = "random",
    seed: int | None = 0,
    env_config: EnvConfig | None = None,
    output_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    write_artifacts: bool = True,
) -> list[RolloutArtifact]:
    env_config = env_config or EnvConfig()
    output_dir = Path(output_dir) if output_dir is not None else rollouts_dir()
    artifacts: list[RolloutArtifact] = []
    for episode in range(episodes):
        episode_seed = None if seed is None else seed + episode
        artifact = run_policy_episode(
            policy=policy,
            seed=episode_seed,
            env_config=env_config,
        )
        artifact_path = None
        if write_artifacts:
            artifact_path = episode_rollout_path(
                output_dir,
                track_id=artifact.summary.track_id,
                policy=policy,
                episode=episode,
                seed=artifact.summary.seed,
                episode_steps=artifact.summary.episode_steps,
            )
            save_rollout(artifact, artifact_path)
        append_rollout_summary(
            artifact.summary,
            summary_path,
            artifact_path=artifact_path,
            policy=policy,
            episode=episode,
        )
        artifacts.append(artifact)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run headless Ghostline RL rollouts.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy", choices=("random", "throttle", "cycle"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=EnvConfig().max_episode_steps)
    parser.add_argument("--output-dir", type=Path, default=rollouts_dir())
    parser.add_argument("--summary-path", type=Path, default=rollout_summaries_path())
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    env_config = EnvConfig(max_episode_steps=args.max_episode_steps)
    if args.check_env:
        check_gymnasium_env(env_config)

    artifacts = run_batch(
        episodes=args.episodes,
        policy=args.policy,
        seed=args.seed,
        env_config=env_config,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        write_artifacts=not args.no_artifacts,
    )
    completed = sum(1 for artifact in artifacts if artifact.summary.terminated)
    best_cp = max(artifact.summary.checkpoint_index for artifact in artifacts)
    best_lap = min(
        (artifact.summary.lap_time for artifact in artifacts if artifact.summary.valid),
        default=None,
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "episodes": len(artifacts),
                    "policy": args.policy,
                    "completed": completed,
                    "best_checkpoint": best_cp,
                    "checkpoint_count": artifacts[0].summary.checkpoint_count,
                    "best_lap_time": best_lap,
                    "summary_path": str(args.summary_path),
                    "artifacts_dir": None if args.no_artifacts else str(args.output_dir),
                },
                sort_keys=True,
            )
        )
    return 0


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected a number, got {value!r}")
    return float(value)


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected an integer, got {value!r}")
    return value


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _int(value, field)


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field}: expected a boolean, got {value!r}")
    return value


def _str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}: expected a non-empty string, got {value!r}")
    return value


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _str(value, field)


def _object_dict(value: Any, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field}: expected an object, got {value!r}")
    if not all(isinstance(key, str) and key for key in value):
        raise ValueError(f"{field}: expected non-empty string keys")
    return dict(value)


def _number_dict(value: Any, field: str) -> dict[str, float]:
    raw = _object_dict(value, field)
    return {key: _number(raw[key], f"{field}.{key}") for key in raw}


def _reward_config(value: Any, field: str) -> dict[str, float | str]:
    raw = _object_dict(value, field)
    out: dict[str, float | str] = {}
    for key, val in raw.items():
        if isinstance(val, str):
            out[key] = val
        else:
            out[key] = _number(val, f"{field}.{key}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
