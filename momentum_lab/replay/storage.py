"""JSON storage helpers for replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .recorder import ReplayData


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def replays_dir() -> Path:
    return project_root() / "replays"


def _safe_track_id(track_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in track_id)


def last_replay_path() -> Path:
    return replays_dir() / "last_run.json"


def best_replay_path(track_id: str) -> Path:
    return replays_dir() / f"best_{_safe_track_id(track_id)}.json"


def save_replay(replay: ReplayData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(replay.to_dict(), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def load_replay(path: str | Path) -> ReplayData:
    path = Path(path)
    return ReplayData.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_best_replay(track_id: str) -> ReplayData | None:
    path = best_replay_path(track_id)
    if not path.exists():
        return None
    return load_replay(path)
