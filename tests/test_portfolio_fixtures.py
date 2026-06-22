"""GL-19 portfolio fixture generation contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import momentum_lab.portfolio_fixtures as pf
from momentum_lab import PHYSICS_VERSION
from momentum_lab.core.action import Action
from momentum_lab.core.sim import Simulation
from momentum_lab.physics_identity import physics_config_fingerprint, physics_config_payload
from momentum_lab.replay.recorder import InitialState, ReplayData
from momentum_lab.tracks import load_track_by_id


def _fake_ai_replay() -> ReplayData:
    track = load_track_by_id(pf.TRACK_ID)
    sim = Simulation()
    sim.reset(track=track)
    return ReplayData(
        track_id=pf.TRACK_ID,
        physics_version=PHYSICS_VERSION,
        physics_config=physics_config_payload(sim.cfg),
        physics_fingerprint=physics_config_fingerprint(sim.cfg),
        seed=None,
        initial_state=InitialState.from_world(sim.world),
        lap_time=3.965050363241023,
        valid=True,
        actions=tuple(Action(throttle=1.0) for _ in range(239)),
        frames=(),
    )


def _stub_play_replay(monkeypatch, *, wall_hits: int = 0) -> None:
    result = SimpleNamespace(
        final_hash="f2fd2ed736b7b796",
        world=SimpleNamespace(run=SimpleNamespace(valid=True), wall_hits=wall_hits),
    )
    monkeypatch.setattr(pf, "play_replay", lambda replay, track: result)


def test_scripted_fixtures_are_deterministic():
    first = [fixture.to_dict() for fixture in pf.scripted_fixtures()]
    second = [fixture.to_dict() for fixture in pf.scripted_fixtures()]

    assert first == second
    assert [fixture["name"] for fixture in first] == [
        "straight_accel",
        "brake_turn",
        "wall_collision",
        "boost_pad",
        "checkpoint_finish",
    ]
    assert all(fixture["physics_version"] == PHYSICS_VERSION for fixture in first)
    assert all(fixture["track_id"] == pf.TRACK_ID for fixture in first)
    assert all(fixture["actions"] for fixture in first)
    by_name = {fixture["name"]: fixture for fixture in first}
    assert by_name["wall_collision"]["expected"]["wall_hits"] == 1
    assert by_name["checkpoint_finish"]["expected"]["lap_time"] > 0
    assert by_name["checkpoint_finish"]["expected"]["lap_ticks"] > 0


def test_ai_record_replay_fixture_locks_plan_outcome(monkeypatch):
    _stub_play_replay(monkeypatch)

    fixture = pf.ai_record_replay_fixture(_fake_ai_replay()).to_dict()

    assert fixture["name"] == "ai_record_replay"
    assert fixture["physics_version"] == PHYSICS_VERSION
    assert fixture["track_id"] == pf.TRACK_ID
    assert len(fixture["actions"]) == 239
    assert fixture["expected"] == {
        "lap_time": 3.965,
        "lap_ticks": 239,
        "wall_hits": 0,
        "final_state_hash": "f2fd2ed736b7b796",
    }


def test_ai_record_replay_fixture_rejects_drift_from_plan(monkeypatch):
    _stub_play_replay(monkeypatch)
    replay = _fake_ai_replay()

    with pytest.raises(pf.PortfolioFixtureError):
        pf.ai_record_replay_fixture(
            ReplayData(
                **{
                    **replay.__dict__,
                    "lap_time": 4.2,
                }
            )
        )


def test_write_fixtures_writes_required_files(tmp_path, monkeypatch):
    _stub_play_replay(monkeypatch)

    written = pf.write_fixtures(tmp_path, _fake_ai_replay())

    names = sorted(path.name for path in written)
    assert names == [
        "ai_record_replay.json",
        "boost_pad.json",
        "brake_turn.json",
        "checkpoint_finish.json",
        "straight_accel.json",
        "wall_collision.json",
    ]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in written]
    assert all(set(payload) == {"name", "physics_version", "track_id", "actions", "expected"} for payload in payloads)
    assert all(
        set(payload["expected"]) == {"lap_time", "lap_ticks", "wall_hits", "final_state_hash"}
        for payload in payloads
    )
