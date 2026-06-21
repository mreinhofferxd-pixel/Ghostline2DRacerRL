"""Tiny fallback spaces plus optional Gymnasium space construction."""

from __future__ import annotations

import random

from .actions import ContinuousActionAdapter


class SimpleDiscreteSpace:
    """Small subset of ``gymnasium.spaces.Discrete`` used by smoke tests."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._rng = random.Random()

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng.seed(seed)
        return [seed]

    def sample(self) -> int:
        return self._rng.randrange(self.n)


class SimpleBoxSpace:
    """Small subset of ``gymnasium.spaces.Box`` used when Gymnasium is absent."""

    def __init__(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        *,
        sampler_low: tuple[float, ...] | None = None,
        sampler_high: tuple[float, ...] | None = None,
    ) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self._sampler_low = sampler_low
        self._sampler_high = sampler_high
        self._rng = random.Random()

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng.seed(seed)
        return [seed]

    def sample(self) -> tuple[float, ...]:
        size = self.shape[0]
        lows = self._sampler_low or tuple(self.low for _ in range(size))
        highs = self._sampler_high or tuple(self.high for _ in range(size))
        return tuple(self._rng.uniform(lo, hi) for lo, hi in zip(lows, highs))


def make_action_space(adapter):
    try:
        from gymnasium import spaces
        import numpy as np
    except ModuleNotFoundError:
        if hasattr(adapter, "n"):
            return SimpleDiscreteSpace(adapter.n)
        if isinstance(adapter, ContinuousActionAdapter):
            return SimpleBoxSpace(
                -1.0,
                1.0,
                (4,),
                sampler_low=(0.0, 0.0, -1.0, 0.0),
                sampler_high=(1.0, 1.0, 1.0, 1.0),
            )
        raise TypeError(f"unsupported action adapter {adapter!r}")

    if hasattr(adapter, "n"):
        return spaces.Discrete(adapter.n)
    if isinstance(adapter, ContinuousActionAdapter):
        return spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
    raise TypeError(f"unsupported action adapter {adapter!r}")


def make_observation_space(size: int):
    try:
        from gymnasium import spaces
        import numpy as np
    except ModuleNotFoundError:
        return SimpleBoxSpace(-1.0, 1.0, (size,))
    return spaces.Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)
