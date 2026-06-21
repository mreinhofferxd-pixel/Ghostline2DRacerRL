"""Headless RL adapter for Ghostline.

The game and deterministic core do not depend on this package. It is the thin
outer wrapper that presents the existing ``Simulation``/``Action`` seam in a
Gymnasium-style shape for later training.
"""

from importlib import import_module

from .actions import (
    ACTION_NAMES,
    DRIVE_ACTION_NAMES,
    ContinuousActionAdapter,
    DiscreteActionAdapter,
    DriveDiscreteActionAdapter,
)
from .env import EnvConfig, GhostlineEnv
from .observations import OBSERVATION_FIELDS
from .rewards import REWARD_VERSION, RewardConfig

_ROLLOUT_EXPORTS = {
    "ROLLOUT_SCHEMA_VERSION",
    "RolloutArtifact",
    "RolloutRecorder",
    "RolloutSummary",
    "append_rollout_summary",
    "check_gymnasium_env",
    "load_rollout",
    "run_batch",
    "run_policy_episode",
    "run_rollout",
    "save_rollout",
    "validate_rollout",
}


def __getattr__(name: str):
    if name in _ROLLOUT_EXPORTS:
        module = import_module(f"{__name__}.rollout")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ACTION_NAMES",
    "DRIVE_ACTION_NAMES",
    "ContinuousActionAdapter",
    "DiscreteActionAdapter",
    "DriveDiscreteActionAdapter",
    "EnvConfig",
    "GhostlineEnv",
    "OBSERVATION_FIELDS",
    "REWARD_VERSION",
    "ROLLOUT_SCHEMA_VERSION",
    "RewardConfig",
    "RolloutArtifact",
    "RolloutRecorder",
    "RolloutSummary",
    "append_rollout_summary",
    "check_gymnasium_env",
    "load_rollout",
    "run_batch",
    "run_policy_episode",
    "run_rollout",
    "save_rollout",
    "validate_rollout",
]
