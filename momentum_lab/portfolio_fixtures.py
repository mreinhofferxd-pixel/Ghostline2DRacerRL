"""Golden fixture generation for the Ghostline portfolio browser sim.

GL-19 keeps canonical sim ownership in this repo. The portfolio consumes these
small JSON files later, but the expected outcomes are generated here from the
Python sim and the locked RL replay artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import PHYSICS_VERSION
from .config import CONTROL_DT, DEFAULT_TRACK
from .core.action import Action
from .core.sim import Simulation
from .replay.recorder import ReplayData, play_replay
from .tracks import load_track_by_id

TRACK_ID = DEFAULT_TRACK
AI_FIXTURE_NAME = "ai_record_replay"
AI_EXPECTED_LAP_TIME = 3.965
AI_EXPECTED_LAP_TICKS = 239
AI_EXPECTED_WALL_HITS = 0
_LAP_TIME_TOLERANCE = 1e-3


class PortfolioFixtureError(RuntimeError):
    """A fixture source is missing or no longer matches the locked plan."""


@dataclass(frozen=True)
class FixtureExpected:
    lap_time: float
    lap_ticks: int
    wall_hits: int
    final_state_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lap_time": self.lap_time,
            "lap_ticks": self.lap_ticks,
            "wall_hits": self.wall_hits,
            "final_state_hash": self.final_state_hash,
        }


@dataclass(frozen=True)
class PortfolioFixture:
    name: str
    physics_version: str
    track_id: str
    actions: tuple[Action, ...]
    expected: FixtureExpected

    @property
    def filename(self) -> str:
        return f"{self.name}.json"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "physics_version": self.physics_version,
            "track_id": self.track_id,
            "actions": [_action_to_json(action) for action in self.actions],
            "expected": self.expected.to_dict(),
        }


def _action_to_json(action: Action) -> list[float | bool]:
    throttle, brake, steer, drift = action.clamped().as_tuple()
    return [throttle, brake, steer, drift]


def _expected_from_sim(sim: Simulation) -> FixtureExpected:
    world = sim.world
    return FixtureExpected(
        lap_time=round(world.run.lap_time(world.tick, CONTROL_DT), 3),
        lap_ticks=world.run.lap_ticks,
        wall_hits=world.wall_hits,
        final_state_hash=sim.state_hash(),
    )


def _run_scripted_fixture(name: str, actions: Iterable[Action]) -> PortfolioFixture:
    track = load_track_by_id(TRACK_ID)
    action_tuple = tuple(action.clamped() for action in actions)
    sim = Simulation()
    sim.reset(track=track)
    for action in action_tuple:
        sim.step(action)
    return PortfolioFixture(
        name=name,
        physics_version=PHYSICS_VERSION,
        track_id=TRACK_ID,
        actions=action_tuple,
        expected=_expected_from_sim(sim),
    )


def _round_axis(value: float) -> float:
    return round(value, 6)


def _checkpoint_finish_actions(max_steps: int = 4000) -> tuple[Action, ...]:
    """Generate a small deterministic clean lap with the authored racing line."""
    track = load_track_by_id(TRACK_ID)
    sim = Simulation()
    sim.reset(track=track)
    waypoint_index = 0
    actions: list[Action] = []
    for _ in range(max_steps):
        target = track.racing_line[waypoint_index % len(track.racing_line)]
        car = sim.world.car
        if math.hypot(target[0] - car.px, target[1] - car.py) <= 90.0:
            waypoint_index += 1
            target = track.racing_line[waypoint_index % len(track.racing_line)]

        desired = math.atan2(target[1] - car.py, target[0] - car.px)
        error = (desired - car.heading + math.pi) % (2.0 * math.pi) - math.pi
        steer = _round_axis(max(-1.0, min(1.0, 2.5 * error)))
        if abs(error) > 0.9:
            action = Action(brake=0.4, steer=steer)
        elif car.speed > 320.0:
            action = Action(steer=steer)
        else:
            action = Action(throttle=1.0, steer=steer)
        actions.append(action)
        sim.step(action)
        if sim.world.run.finished:
            break
    if not sim.world.run.valid:
        raise PortfolioFixtureError("checkpoint_finish did not complete a valid lap")
    return tuple(actions)


def scripted_fixtures() -> tuple[PortfolioFixture, ...]:
    """Fixtures that exercise browser-port mechanics without RL artifacts."""
    return (
        _run_scripted_fixture("straight_accel", [Action(throttle=1.0)] * 30),
        _run_scripted_fixture(
            "brake_turn",
            [Action(throttle=1.0)] * 24
            + [Action(brake=1.0, steer=-1.0)] * 24
            + [Action(throttle=1.0, steer=-1.0)] * 12,
        ),
        _run_scripted_fixture("wall_collision", [Action(throttle=1.0, steer=1.0)] * 45),
        _run_scripted_fixture("boost_pad", [Action(throttle=1.0)] * 50),
        _run_scripted_fixture("checkpoint_finish", _checkpoint_finish_actions()),
    )


def ai_record_replay_fixture(replay: ReplayData) -> PortfolioFixture:
    """Build the locked AI replay fixture from the canonical rollout replay."""
    if replay.track_id != TRACK_ID:
        raise PortfolioFixtureError(
            f"{AI_FIXTURE_NAME}: track_id {replay.track_id!r} != {TRACK_ID!r}"
        )
    if replay.physics_version != PHYSICS_VERSION:
        raise PortfolioFixtureError(
            f"{AI_FIXTURE_NAME}: physics_version {replay.physics_version!r} != {PHYSICS_VERSION!r}"
        )
    if not replay.valid:
        raise PortfolioFixtureError(f"{AI_FIXTURE_NAME}: replay is not a valid lap")
    if abs(replay.lap_time - AI_EXPECTED_LAP_TIME) > _LAP_TIME_TOLERANCE:
        raise PortfolioFixtureError(
            f"{AI_FIXTURE_NAME}: lap_time {replay.lap_time:.3f}s != {AI_EXPECTED_LAP_TIME:.3f}s"
        )
    if len(replay.actions) != AI_EXPECTED_LAP_TICKS:
        raise PortfolioFixtureError(
            f"{AI_FIXTURE_NAME}: action count {len(replay.actions)} != {AI_EXPECTED_LAP_TICKS}"
        )

    result = play_replay(replay, load_track_by_id(TRACK_ID))
    if not result.world.run.valid:
        raise PortfolioFixtureError(f"{AI_FIXTURE_NAME}: replay playback did not finish valid")
    if result.world.wall_hits != AI_EXPECTED_WALL_HITS:
        raise PortfolioFixtureError(
            f"{AI_FIXTURE_NAME}: wall_hits {result.world.wall_hits} != {AI_EXPECTED_WALL_HITS}"
        )

    # The public benchmark counts the full action stream as the lap tick count.
    # RunState's timer starts on tick 1, so its internal lap_ticks is one lower.
    expected = FixtureExpected(
        lap_time=AI_EXPECTED_LAP_TIME,
        lap_ticks=AI_EXPECTED_LAP_TICKS,
        wall_hits=AI_EXPECTED_WALL_HITS,
        final_state_hash=result.final_hash,
    )
    return PortfolioFixture(
        name=AI_FIXTURE_NAME,
        physics_version=PHYSICS_VERSION,
        track_id=TRACK_ID,
        actions=tuple(action.clamped() for action in replay.actions),
        expected=expected,
    )


def build_fixtures(ai_replay: ReplayData) -> tuple[PortfolioFixture, ...]:
    return scripted_fixtures() + (ai_record_replay_fixture(ai_replay),)


def write_fixtures(out_dir: str | Path, ai_replay: ReplayData) -> tuple[Path, ...]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fixture in build_fixtures(ai_replay):
        path = out / fixture.filename
        path.write_text(json.dumps(fixture.to_dict(), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return tuple(written)
