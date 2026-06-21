"""B7.1 Gymnasium-style wrapper smoke tests."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from momentum_lab.config import CONTROL_HZ
from momentum_lab.core.action import Action
from momentum_lab.rl import EnvConfig, GhostlineEnv
from momentum_lab.rl.actions import DiscreteActionAdapter
from momentum_lab.rl.observations import OBSERVATION_FIELDS


def _scripted_lap_action(world) -> Action:
    """Small pure-pursuit policy used only to prove env termination."""
    line = world.track.racing_line
    if not line:
        return Action(throttle=1.0)
    # Store the waypoint on the function object so the loop can be tiny and
    # deterministic without a class.
    wp = getattr(_scripted_lap_action, "wp", 0)
    target = line[wp % len(line)]
    car = world.car
    if math.hypot(target[0] - car.px, target[1] - car.py) <= 90.0:
        wp += 1
        target = line[wp % len(line)]
    setattr(_scripted_lap_action, "wp", wp)

    desired = math.atan2(target[1] - car.py, target[0] - car.px)
    err = (desired - car.heading + math.pi) % (2.0 * math.pi) - math.pi
    steer = max(-1.0, min(1.0, 2.5 * err))
    if abs(err) > 0.9:
        return Action(brake=0.4, steer=steer)
    if car.speed > 320.0:
        return Action(steer=steer)
    return Action(throttle=1.0, steer=steer)


def test_random_policy_can_reset_and_step_headlessly():
    env = GhostlineEnv()
    env.action_space.seed(123)
    obs, info = env.reset(seed=0)
    assert len(obs) == len(OBSERVATION_FIELDS)
    assert info["track_id"] == "track_01_easy_loop"

    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.sample_action())
        assert len(obs) == len(OBSERVATION_FIELDS)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert info["physics_fingerprint"]
        assert info["reward_version"] == "reward_v1"


def test_same_seed_and_actions_are_deterministic():
    actions = [1, 2, 3, 9, 10, 1, 1, 4, 0, 3] * 30

    def run():
        env = GhostlineEnv()
        obs, info = env.reset(seed=42)
        trace = [(tuple(obs), 0.0, False, False, info["tick"])]
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            trace.append((tuple(obs), reward, terminated, truncated, info["tick"]))
            if terminated or truncated:
                break
        return trace, env.sim.state_hash()

    trace_a, hash_a = run()
    trace_b, hash_b = run()
    assert trace_a == trace_b
    assert hash_a == hash_b


def test_action_adapter_routes_to_core_action():
    env = GhostlineEnv()
    env.reset(seed=0)
    env.step(DiscreteActionAdapter.n - 1)
    assert isinstance(env.last_action, Action)
    assert env.last_action.throttle == 1.0
    assert env.last_action.drift is True


def test_direct_action_is_accepted_for_scripted_eval_policies():
    env = GhostlineEnv()
    env.reset(seed=0)
    env.step(Action(throttle=1.0, steer=0.25, drift=True))
    assert env.last_action == Action(throttle=1.0, steer=0.25, drift=True)
    assert env.last_action.drift is True


def test_scripted_lap_terminates_episode():
    env = GhostlineEnv(EnvConfig(max_episode_steps=4000))
    obs, info = env.reset(seed=0)
    setattr(_scripted_lap_action, "wp", 0)
    for _ in range(4000):
        action = _scripted_lap_action(env.sim.world)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert info["final_reason"] == "lap_complete"
    assert info["lap_time"] > 0.0
    assert info["checkpoint_index"] == info["checkpoint_count"]


def test_short_episode_limit_truncates():
    env = GhostlineEnv(EnvConfig(max_episode_steps=3))
    env.reset(seed=0)
    for _ in range(2):
        _obs, _reward, terminated, truncated, _info = env.step(0)
        assert not terminated
        assert not truncated
    _obs, _reward, terminated, truncated, info = env.step(0)
    assert terminated is False
    assert truncated is True
    assert info["final_reason"] == "time_limit"


def test_step_after_episode_end_requires_reset():
    env = GhostlineEnv(EnvConfig(max_episode_steps=1))
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_observation_contract_excludes_racing_line_hint():
    assert not any("racing_line" in field for field in OBSERVATION_FIELDS)
    env = GhostlineEnv()
    obs, _info = env.reset(seed=0)
    assert len(obs) == len(OBSERVATION_FIELDS)
    assert len([field for field in OBSERVATION_FIELDS if field.startswith("raycast_")]) == 16


def test_lookahead_observation_is_opt_in_and_widens_obs():
    from momentum_lab.rl.observations import (
        LOOKAHEAD_OBSERVATION_FIELDS,
        ObservationConfig,
    )

    base = ObservationConfig()
    look = ObservationConfig(include_lookahead=True)
    # Off by default so existing models keep their observation dimension.
    assert base.include_lookahead is False
    assert len(look.fields) == len(base.fields) + len(LOOKAHEAD_OBSERVATION_FIELDS)
    # Lookahead fields are inserted before the raycasts, contiguous.
    assert all(f in look.fields for f in LOOKAHEAD_OBSERVATION_FIELDS)
    assert look.fields[: len(base.fields) - 16] == base.fields[: len(base.fields) - 16]

    env = GhostlineEnv(EnvConfig(observation=look))
    obs, _info = env.reset(seed=0)
    assert len(obs) == len(look.fields)
    assert env.observation_space.shape == (len(look.fields),)


def test_lookahead_has_next2_reflects_remaining_gates():
    """``has_next2`` is 1 while a gate exists after the target, 0 at the finish."""
    from momentum_lab.core.car import Car
    from momentum_lab.core.checkpoints import Gate
    from momentum_lab.core.sim import World
    from momentum_lab.core.timing import RunState
    from momentum_lab.core.track import Track
    from momentum_lab.rl.observations import ObservationConfig, observe

    look = ObservationConfig(include_lookahead=True)
    idx = look.fields.index("has_next2")
    cp = Gate.from_endpoints(100.0, 0.0, 100.0, 200.0)
    fin = Gate.from_endpoints(300.0, 0.0, 300.0, 200.0)
    track = Track(
        track_id="lookahead_test",
        spawn=(50.0, 100.0),
        spawn_heading=0.0,
        checkpoints=(cp,),
        finish=fin,
    )
    # Targeting the checkpoint -> the finish is the next-next gate.
    targeting_cp = World(car=Car(px=50.0, py=100.0), track=track, run=RunState(next_cp=0))
    assert observe(targeting_cp, look)[idx] == 1.0
    # Targeting the finish -> nothing after it; neutral sentinels, flag off.
    targeting_finish = World(car=Car(px=50.0, py=100.0), track=track, run=RunState(next_cp=1))
    assert observe(targeting_finish, look)[idx] == 0.0


def test_wall_sensors_observation_is_opt_in_and_widens_obs():
    from momentum_lab.rl.observations import (
        WALL_SENSOR_OBSERVATION_FIELDS,
        ObservationConfig,
    )

    base = ObservationConfig()
    walls = ObservationConfig(include_wall_sensors=True)
    # Off by default so existing models keep their observation dimension.
    assert base.include_wall_sensors is False
    assert len(walls.fields) == len(base.fields) + len(WALL_SENSOR_OBSERVATION_FIELDS)
    # Wall-sensor fields are inserted before the raycasts, contiguous.
    assert all(f in walls.fields for f in WALL_SENSOR_OBSERVATION_FIELDS)
    assert walls.fields[: len(base.fields) - 16] == base.fields[: len(base.fields) - 16]

    # Stacks on top of lookahead (the real curriculum config uses both).
    both = ObservationConfig(include_lookahead=True, include_wall_sensors=True)
    look = ObservationConfig(include_lookahead=True)
    assert len(both.fields) == len(look.fields) + len(WALL_SENSOR_OBSERVATION_FIELDS)
    assert both.fields[-16:] == look.fields[-16:]  # raycasts stay last

    env = GhostlineEnv(EnvConfig(observation=both))
    obs, _info = env.reset(seed=0)
    assert len(obs) == len(both.fields)
    assert env.observation_space.shape == (len(both.fields),)


def test_wall_sensors_measure_gate_frame_distances():
    """The three sensors read wall distance in the *target gate's* frame, not the
    car heading -- forward along the gate normal, left/right along its tangent."""
    from momentum_lab.core.car import Car
    from momentum_lab.core.checkpoints import Gate
    from momentum_lab.core.collision import Segment
    from momentum_lab.core.sim import World
    from momentum_lab.core.timing import RunState
    from momentum_lab.core.track import Track
    from momentum_lab.rl.observations import ObservationConfig, observe

    cfg = ObservationConfig(include_wall_sensors=True)
    i_fwd = cfg.fields.index("gate_fwd_wall_dist")
    i_left = cfg.fields.index("gate_left_wall_dist")
    i_right = cfg.fields.index("gate_right_wall_dist")

    # Finish gate from (100,200)->(100,0): tangent (0,-1), forward normal = +x.
    # So left (normal +90deg) = +y, right = -y.
    fin = Gate.from_endpoints(100.0, 200.0, 100.0, 0.0)
    assert round(fin.nx, 6) == 1.0 and round(fin.ny, 6) == 0.0
    walls = (
        Segment(250.0, 0.0, 250.0, 200.0),  # ahead (+x) at distance 200 from x=50
        Segment(0.0, 300.0, 200.0, 300.0),  # left  (+y) at distance 200 from y=100
        Segment(0.0, 50.0, 200.0, 50.0),    # right (-y) at distance 50 from y=100
    )
    track = Track(
        track_id="wall_sensor_test",
        spawn=(50.0, 100.0),
        spawn_heading=0.0,
        walls=walls,
        finish=fin,
    )
    # Heading deliberately != gate frame: the sensors must ignore it.
    world = World(car=Car(px=50.0, py=100.0, heading=1.3), track=track, run=RunState(next_cp=0))
    obs = observe(world, cfg)
    md = cfg.raycast_max_dist
    assert obs[i_fwd] == pytest.approx(200.0 / md, abs=1e-6)
    assert obs[i_left] == pytest.approx(200.0 / md, abs=1e-6)
    assert obs[i_right] == pytest.approx(50.0 / md, abs=1e-6)
    # The inside (right) wall is perceived as closest regardless of car heading.
    assert obs[i_right] < obs[i_left]


def test_wall_sensors_round_trip_through_manifest():
    """A model trained with wall sensors re-evals at the same obs dim; an old
    manifest without the field defaults it off (backward compatible)."""
    from momentum_lab.rl.observations import ObservationConfig
    from momentum_lab.rl.train import _env_config_payload, env_config_from_manifest

    cfg = EnvConfig(observation=ObservationConfig(include_lookahead=True, include_wall_sensors=True))
    manifest = {"env_config": _env_config_payload(cfg)}
    restored = env_config_from_manifest(manifest)
    assert restored.observation.include_wall_sensors is True
    assert restored.observation.include_lookahead is True
    assert len(restored.observation.fields) == len(cfg.observation.fields)

    # Legacy manifest: observation dict predates the field -> defaults off.
    legacy = {"env_config": _env_config_payload(cfg)}
    del legacy["env_config"]["observation"]["include_wall_sensors"]
    assert env_config_from_manifest(legacy).observation.include_wall_sensors is False


def test_importing_rl_package_does_not_import_pygame():
    code = (
        "import sys; "
        "import momentum_lab.rl; "
        "raise SystemExit(1 if 'pygame' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
