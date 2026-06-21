"""Action adapters for the Gymnasium-style RL boundary.

The sim still accepts only ``core.action.Action``. These adapters define how an
environment action value is translated into that seam.
"""

from __future__ import annotations

import random
from numbers import Integral

from ..core.action import Action


ACTION_NAMES: tuple[str, ...] = (
    "neutral",
    "throttle",
    "throttle_left",
    "throttle_right",
    "brake",
    "brake_left",
    "brake_right",
    "drift_left",
    "drift_right",
    "throttle_drift_left",
    "throttle_drift_right",
    "throttle_drift",
)


DISCRETE_ACTIONS: tuple[Action, ...] = (
    Action(),
    Action(throttle=1.0),
    Action(throttle=1.0, steer=-1.0),
    Action(throttle=1.0, steer=1.0),
    Action(brake=1.0),
    Action(brake=1.0, steer=-1.0),
    Action(brake=1.0, steer=1.0),
    Action(steer=-1.0, drift=True),
    Action(steer=1.0, drift=True),
    Action(throttle=1.0, steer=-1.0, drift=True),
    Action(throttle=1.0, steer=1.0, drift=True),
    Action(throttle=1.0, drift=True),
)

DRIVE_ACTION_NAMES: tuple[str, ...] = (
    "throttle",
    "throttle_left",
    "throttle_right",
    "throttle_drift_left",
    "throttle_drift_right",
)


DRIVE_DISCRETE_ACTIONS: tuple[Action, ...] = (
    Action(throttle=1.0),
    Action(throttle=1.0, steer=-1.0),
    Action(throttle=1.0, steer=1.0),
    Action(throttle=1.0, steer=-1.0, drift=True),
    Action(throttle=1.0, steer=1.0, drift=True),
)


class ActionAdapterError(ValueError):
    """An env action cannot be converted to the core ``Action`` contract."""


class DiscreteActionAdapter:
    """Human-button-like action map for early exploration/smoke tests."""

    kind = "discrete"
    n = len(DISCRETE_ACTIONS)

    def to_action(self, raw) -> Action:
        if isinstance(raw, Action):
            return raw.clamped()
        if hasattr(raw, "shape") and getattr(raw, "shape", None) == () and hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise ActionAdapterError(f"discrete action must be an integer, got {raw!r}")
        raw = int(raw)
        if raw < 0 or raw >= self.n:
            raise ActionAdapterError(f"discrete action {raw} outside [0, {self.n})")
        return DISCRETE_ACTIONS[raw]

    def sample(self, rng: random.Random) -> int:
        return rng.randrange(self.n)

    def name(self, raw) -> str:
        if isinstance(raw, Action):
            return "direct_action"
        if hasattr(raw, "shape") and getattr(raw, "shape", None) == () and hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            return "invalid"
        raw = int(raw)
        if raw < 0 or raw >= self.n:
            return "invalid"
        return ACTION_NAMES[raw]

    def config_payload(self) -> dict[str, object]:
        return {"kind": self.kind, "actions": list(ACTION_NAMES)}


class DriveDiscreteActionAdapter:
    """Small drive-only action set for first-lap PPO training."""

    kind = "drive_discrete"
    n = len(DRIVE_DISCRETE_ACTIONS)

    def to_action(self, raw) -> Action:
        if isinstance(raw, Action):
            return raw.clamped()
        if hasattr(raw, "shape") and getattr(raw, "shape", None) == () and hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise ActionAdapterError(f"drive discrete action must be an integer, got {raw!r}")
        raw = int(raw)
        if raw < 0 or raw >= self.n:
            raise ActionAdapterError(f"drive discrete action {raw} outside [0, {self.n})")
        return DRIVE_DISCRETE_ACTIONS[raw]

    def sample(self, rng: random.Random) -> int:
        return rng.randrange(self.n)

    def name(self, raw) -> str:
        if isinstance(raw, Action):
            return "direct_action"
        if hasattr(raw, "shape") and getattr(raw, "shape", None) == () and hasattr(raw, "item"):
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            return "invalid"
        raw = int(raw)
        if raw < 0 or raw >= self.n:
            return "invalid"
        return DRIVE_ACTION_NAMES[raw]

    def config_payload(self) -> dict[str, object]:
        return {"kind": self.kind, "actions": list(DRIVE_ACTION_NAMES)}


class ContinuousActionAdapter:
    """Continuous throttle/brake/steer with a thresholded drift button.

    Shape: ``[throttle, brake, steer, drift]`` where drift is considered pressed at
    or above ``drift_threshold``.
    """

    kind = "continuous"

    def __init__(self, drift_threshold: float = 0.5) -> None:
        self.drift_threshold = float(drift_threshold)

    def to_action(self, raw) -> Action:
        if isinstance(raw, Action):
            return raw.clamped()
        if isinstance(raw, (str, bytes)):
            raise ActionAdapterError(
                "continuous action must be [throttle, brake, steer, drift]"
            )
        try:
            length = len(raw)
        except TypeError as e:
            raise ActionAdapterError(
                "continuous action must be [throttle, brake, steer, drift]"
            ) from e
        if length != 4:
            raise ActionAdapterError(
                "continuous action must be [throttle, brake, steer, drift]"
            )
        throttle, brake, steer, drift = raw
        try:
            return Action(
                throttle=float(throttle),
                brake=float(brake),
                steer=float(steer),
                drift=float(drift) >= self.drift_threshold,
            ).clamped()
        except (TypeError, ValueError) as e:
            raise ActionAdapterError(f"continuous action contains non-numeric values: {raw!r}") from e

    def sample(self, rng: random.Random) -> tuple[float, float, float, float]:
        return (rng.random(), rng.random(), rng.uniform(-1.0, 1.0), rng.random())

    def name(self, raw) -> str:
        return "continuous"

    def config_payload(self) -> dict[str, object]:
        return {"kind": self.kind, "drift_threshold": self.drift_threshold}


def make_action_adapter(kind: str):
    if kind == "discrete":
        return DiscreteActionAdapter()
    if kind == "drive_discrete":
        return DriveDiscreteActionAdapter()
    if kind == "continuous":
        return ContinuousActionAdapter()
    raise ActionAdapterError(f"unknown action adapter {kind!r}")
