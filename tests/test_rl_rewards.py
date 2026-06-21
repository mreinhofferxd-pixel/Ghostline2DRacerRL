"""Reward shaping contracts for the RL wrapper."""

from __future__ import annotations

import pytest

from momentum_lab.core.car import Car
from momentum_lab.core.collision import Segment
from momentum_lab.core.sim import World
from momentum_lab.core.timing import RunState
from momentum_lab.core.track import Track
from momentum_lab.rl.rewards import RewardConfig, RewardState, compute_reward
from momentum_lab.rl.train import first_lap_reward, time_attack_reward


def _wall_world(x: float) -> World:
    return World(
        car=Car(px=x, py=100.0),
        track=Track(
            track_id="wall_reward_test",
            spawn=(200.0, 100.0),
            spawn_heading=0.0,
            walls=(Segment(100.0, 0.0, 100.0, 300.0),),
        ),
    )


def test_wall_proximity_penalty_increases_near_walls():
    cfg = RewardConfig(
        time_penalty=0.0,
        wall_proximity_penalty=-1.0,
        wall_proximity_threshold=50.0,
    )
    far = _wall_world(220.0)
    near = _wall_world(120.0)

    far_reward = compute_reward(RewardState.from_world(far), far, far, cfg)
    near_reward = compute_reward(RewardState.from_world(far), far, near, cfg)

    assert far_reward.wall_proximity == 0.0
    assert near_reward.wall_proximity < 0.0
    assert near_reward.total < far_reward.total
    # Center is 20 px from the wall, car radius is 14, so shell clearance is 6.
    danger = (50.0 - 6.0) / 50.0
    assert near_reward.wall_proximity == pytest.approx(-(danger * danger))


def test_first_lap_reward_v3_is_progress_dominant():
    """v3 must rank "drove far" above "crawled and parked".

    The continuous wall terms (proximity/heading) inverted the ranking in v2 by
    punishing the racing line; v3 disables them and makes a full stage of progress
    worth at least a checkpoint so total reward tracks distance covered.
    """
    cfg = first_lap_reward()

    assert cfg.version == "reward_first_lap_v3"
    # Progress is the star: a single full stage should outweigh the discrete bonus.
    assert cfg.progress_scale >= cfg.checkpoint_bonus
    # A discrete impact still hurts, but continuous wall taxes are tamed/off.
    assert cfg.wall_hit_penalty < 0.0
    assert -1.0 <= cfg.wall_scrape_penalty_per_second <= 0.0
    assert cfg.wall_proximity_penalty == 0.0
    # Drifting through corners must not be penalized.
    assert cfg.heading_alignment_scale == 0.0


def _time_and_wall_score(cfg: RewardConfig, *, steps: int, wall_hits: int) -> float:
    """The lap-time-sensitive part of the total reward.

    The two observed time-attack runs (5.15s vs 6.50s) had near-identical
    progress/target-speed/checkpoint/finish totals, so the ranking between them is
    decided by the time penalty (per step) and the wall-hit penalty. Isolating
    those two components is enough to reproduce — and check the fix for — the
    inversion described in ``time_attack_reward``'s docstring.
    """
    return steps * cfg.time_penalty + wall_hits * cfg.wall_hit_penalty


def test_time_attack_prefers_fast_line_over_clean_slow_line():
    """Regression for the v1 reward/lap-time misalignment, still guarded under v4.

    Observed runs: the 5.15s finisher took 310 control steps with 2 wall hits;
    a later fine-tune drove a clean 6.50s lap in 391 steps with 0 hits. Under
    v1's weights (-0.05/step time, -8/wall-hit) the clean-but-slow line scored
    higher on the time+wall components, so the reward fought the goal. The current
    time-attack reward must rank the faster line above it with finish-time and
    efficiency bonuses included.
    """
    fast = dict(steps=310, wall_hits=2)  # b7_6, 5.15s
    slow = dict(steps=391, wall_hits=0)  # b7_7, 6.50s (clean but slower)

    # The pre-fix weights (kept here only to demonstrate the inversion).
    v1 = RewardConfig(version="reward_time_attack_v1", time_penalty=-0.05, wall_hit_penalty=-8.0)
    current = time_attack_reward()

    # The bug: v1 ranked the slow clean lap above the fast one.
    assert _time_and_wall_score(v1, **slow) > _time_and_wall_score(v1, **fast)
    before = RewardState(target=None, next_cp=4, finished=False, wall_hits=0, wall_scrape_time=0.0)
    w_before = _finished_world(0, finished=False)
    fast_world = _finished_world(fast["steps"], path_distance=2500.0)
    fast_world.wall_hits = fast["wall_hits"]
    slow_world = _finished_world(slow["steps"], path_distance=2600.0)
    slow_world.wall_hits = slow["wall_hits"]

    assert compute_reward(before, w_before, fast_world, current).total > compute_reward(
        before,
        w_before,
        slow_world,
        current,
    ).total


def test_time_attack_reward_v6_is_pinned():
    cfg = time_attack_reward()
    assert cfg.version == "reward_time_attack_v6"
    assert time_attack_reward().payload() == cfg.payload()
    # The dominant lever stays a direct, finish-gated lap-time bonus.
    assert cfg.finish_time_bonus_scale > 0.0
    assert cfg.finish_time_reference_steps > 0.0
    # Finish-gated racing efficiency and a moderate wall-hit penalty (kept from v4).
    assert cfg.avg_speed_bonus_scale > 0.0
    assert cfg.avg_speed_reference > 0.0
    assert cfg.path_efficiency_bonus_scale > 0.0
    assert cfg.path_distance_reference > 0.0
    assert cfg.wall_hit_penalty == -6.0
    # The per-step drift cost from v5 is kept.
    assert cfg.drift_penalty_per_second < 0.0
    # v6 chases the last ticks to the human lap: a stronger inside-line gradient
    # (path scale up + reference pulled further in than v5) and more speed-carry
    # pressure than v5's 0.35 target_speed.
    assert cfg.path_efficiency_bonus_scale >= 1100.0
    assert cfg.path_distance_reference <= 2250.0
    assert cfg.target_speed_scale > 0.35
    # Still finishes: finish/progress structure kept.
    assert cfg.finish_bonus == 500.0
    assert cfg.progress_scale == 140.0


def _drift_world(drift_time: float) -> World:
    """A finished world carrying a given accumulated drift time."""
    world = _finished_world(280, path_distance=2300.0)
    world.drift_time = drift_time
    return world


def test_drift_penalty_charges_newly_accumulated_drift():
    """v5's drift cost is charged per second of new drift, like wall scrape.

    Telemetry (human vs b7_19 sector trace) showed the policy over-drifts into CP2,
    running wide and slow. A small per-step cost nudges it off gratuitous drift; it
    must scale with the drift time added this step and be off when disabled.
    """
    cfg = time_attack_reward()
    assert cfg.drift_penalty_per_second < 0.0
    before = RewardState(
        target=None, next_cp=4, finished=False, wall_hits=0, wall_scrape_time=0.0, drift_time=0.5
    )
    w_before = _drift_world(0.5)

    # Drifting another 0.2s this step costs 0.2 * the per-second rate.
    drifted = compute_reward(before, w_before, _drift_world(0.7), cfg)
    assert drifted.drift == pytest.approx(0.2 * cfg.drift_penalty_per_second)
    assert drifted.drift < 0.0

    # No new drift -> no drift cost; a drop in cumulative drift never pays out.
    steady = compute_reward(before, w_before, _drift_world(0.5), cfg)
    assert steady.drift == 0.0
    assert compute_reward(before, w_before, _drift_world(0.4), cfg).drift == 0.0

    # Disabled by default (v1-v4 behavior): no drift term unless configured.
    off = RewardConfig(drift_penalty_per_second=0.0)
    assert compute_reward(before, w_before, _drift_world(0.9), off).drift == 0.0


def _finished_world(
    lap_ticks: int,
    *,
    finished: bool = True,
    path_distance: float = 0.0,
) -> World:
    return World(
        car=Car(px=100.0, py=100.0),
        track=Track(track_id="finish_reward_test", spawn=(100.0, 100.0), spawn_heading=0.0),
        run=RunState(started=True, finished=finished, valid=finished, next_cp=4, lap_ticks=lap_ticks),
        path_distance=path_distance,
    )


def test_finish_time_bonus_rewards_faster_laps():
    """v3's finish-time bonus must pay more for a faster lap, and only on the close.

    The bonus is ``finish_time_bonus_scale * max(0, reference - lap_ticks)`` paid the
    step the lap closes, so the real objective (total lap time) gets a direct, strong
    gradient instead of relying only on the per-step time proxy.
    """
    cfg = time_attack_reward()
    before = RewardState(target=None, next_cp=4, finished=False, wall_hits=0, wall_scrape_time=0.0)
    w_before = _finished_world(0, finished=False)

    scale, ref = cfg.finish_time_bonus_scale, cfg.finish_time_reference_steps
    fast = compute_reward(before, w_before, _finished_world(310), cfg)
    slow = compute_reward(before, w_before, _finished_world(391), cfg)

    assert fast.finish_time == pytest.approx(scale * (ref - 310))
    assert slow.finish_time == pytest.approx(scale * (ref - 391))
    assert fast.finish_time > slow.finish_time
    assert fast.total > slow.total

    # A lap at/over the reference earns no bonus (clamped at 0), never a penalty.
    assert compute_reward(before, w_before, _finished_world(int(ref) + 50), cfg).finish_time == 0.0

    # The bonus is paid only on the closing step, not on ordinary mid-lap steps.
    assert compute_reward(before, w_before, _finished_world(310, finished=False), cfg).finish_time == 0.0
    already_done = RewardState(target=None, next_cp=4, finished=True, wall_hits=0, wall_scrape_time=0.0)
    assert compute_reward(already_done, w_before, _finished_world(310), cfg).finish_time == 0.0


def test_finish_efficiency_rewards_speed_and_shorter_path():
    cfg = time_attack_reward()
    before = RewardState(target=None, next_cp=4, finished=False, wall_hits=0, wall_scrape_time=0.0)
    w_before = _finished_world(0, finished=False)

    efficient = compute_reward(
        before,
        w_before,
        _finished_world(286, path_distance=2480.0),
        cfg,
    )
    long_line = compute_reward(
        before,
        w_before,
        _finished_world(286, path_distance=2600.0),
        cfg,
    )
    slow_line = compute_reward(
        before,
        w_before,
        _finished_world(320, path_distance=2480.0),
        cfg,
    )

    assert efficient.avg_speed > slow_line.avg_speed
    assert efficient.path_efficiency > long_line.path_efficiency
    assert efficient.total > long_line.total
    assert efficient.total > slow_line.total


def test_reward_version_pins_constants():
    """A reward version string must map to exactly one constant set.

    Earlier runs reused ``reward_first_lap_v1`` for two different weightings,
    which made saved totals impossible to compare across models.
    """
    cfg = first_lap_reward()
    payload = cfg.payload()
    # Re-deriving a config from the published version must reproduce the weights.
    again = first_lap_reward()
    assert again.payload() == payload
