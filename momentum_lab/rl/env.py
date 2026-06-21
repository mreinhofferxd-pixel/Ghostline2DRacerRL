"""Gymnasium-style headless environment around the deterministic sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import PHYSICS_VERSION, config
from ..core.action import Action
from ..core.sim import Simulation
from ..physics_identity import physics_config_fingerprint, physics_config_payload
from ..tracks import load_track_by_id
from .actions import ActionAdapterError, make_action_adapter
from .observations import ObservationConfig, observe
from .rewards import RewardConfig, RewardState, compute_reward, reward_info
from .spaces import make_action_space, make_observation_space

try:
    from gymnasium import Env as _GymEnv
except ModuleNotFoundError:
    _GymEnv = object


@dataclass(frozen=True)
class EnvConfig:
    track_id: str = config.DEFAULT_TRACK
    max_episode_steps: int = 30 * config.CONTROL_HZ
    action_adapter: str = "discrete"
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class GhostlineEnv(_GymEnv):
    """Headless Track-1 RL wrapper.

    It conforms to the modern Gymnasium step/reset return shape, while remaining
    usable without Gymnasium installed.
    """

    metadata = {"render_modes": ()}

    def __init__(self, env_config: EnvConfig | None = None) -> None:
        if _GymEnv is not object:
            super().__init__()
        self.env_config = env_config or EnvConfig()
        self.track = load_track_by_id(self.env_config.track_id)
        self.sim = Simulation(config.CAR)
        self.action_adapter = make_action_adapter(self.env_config.action_adapter)
        self.action_space = make_action_space(self.action_adapter)
        self.observation_space = make_observation_space(len(self.env_config.observation.fields))
        self._episode_steps = 0
        self._done = False
        self._last_action = Action()
        self._last_breakdown = None
        self.reset(seed=0)

    @property
    def last_action(self) -> Action:
        return self._last_action

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Any]]:
        if _GymEnv is not object:
            super().reset(seed=seed)
        if options:
            track_id = options.get("track_id")
            if track_id is not None and track_id != self.track.track_id:
                raise ValueError(
                    "B7.1 supports a single configured track per env; "
                    f"got reset track_id={track_id!r}"
                )
        self._episode_steps = 0
        self._done = False
        self._last_action = Action()
        self._last_breakdown = None
        self.sim.reset(track=self.track, seed=seed)
        obs = self._observe()
        return obs, self._info(final_reason=None)

    def step(self, action):
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        before_world = self.sim.snapshot()
        before_reward = RewardState.from_world(before_world)
        core_action = self.action_adapter.to_action(action)
        self._last_action = core_action

        world = self.sim.step(core_action)
        self._episode_steps += 1
        breakdown = compute_reward(
            before_reward,
            before_world,
            world,
            self.env_config.reward,
        )
        self._last_breakdown = breakdown

        terminated = bool(world.run.finished and world.run.valid)
        truncated = not terminated and self._episode_steps >= self.env_config.max_episode_steps
        self._done = terminated or truncated
        final_reason = None
        if terminated:
            final_reason = "lap_complete"
        elif truncated:
            final_reason = "time_limit"

        obs = self._observe()
        return obs, float(breakdown.total), terminated, truncated, self._info(
            final_reason=final_reason,
            raw_action=action,
        )

    def sample_action(self):
        """Return one action-space sample without requiring Gymnasium."""
        return self.action_space.sample()

    def close(self) -> None:
        return None

    def _observe(self):
        obs = observe(self.sim.world, self.env_config.observation)
        if _GymEnv is object:
            return obs
        try:
            import numpy as np
        except ModuleNotFoundError:
            return obs
        return np.asarray(obs, dtype=np.float32)

    def _info(self, *, final_reason: str | None, raw_action=None) -> dict[str, Any]:
        world = self.sim.world
        run = world.run
        info: dict[str, Any] = {
            "track_id": world.track.track_id,
            "tick": world.tick,
            "episode_steps": self._episode_steps,
            "lap_time": run.lap_time(world.tick, config.CONTROL_DT),
            "checkpoint_index": run.next_cp,
            "checkpoint_count": len(world.track.checkpoints),
            "wall_hits": world.wall_hits,
            "wall_scrape_time": world.wall_scrape_time,
            "boosts_used": world.boosts_used,
            "boost_active": world.boost_active,
            "drift_time": world.drift_time,
            "peak_slip": world.peak_slip,
            "path_distance": world.path_distance,
            "final_reason": final_reason,
            "physics_version": PHYSICS_VERSION,
            "physics_fingerprint": physics_config_fingerprint(self.sim.cfg),
            "physics_config": physics_config_payload(self.sim.cfg),
            "action_adapter": self.action_adapter.config_payload(),
        }
        if raw_action is not None:
            info["action_name"] = self.action_adapter.name(raw_action)
        if self._last_breakdown is not None:
            info.update(reward_info(self.env_config.reward, self._last_breakdown))
        else:
            info.update(
                {
                    "reward_version": self.env_config.reward.version,
                    "reward_config": self.env_config.reward.payload(),
                    "reward_breakdown": None,
                }
            )
        return info


__all__ = ["ActionAdapterError", "EnvConfig", "GhostlineEnv"]
