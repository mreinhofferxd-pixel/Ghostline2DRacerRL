"""Append-only JSONL metrics writer for completed runs."""

from __future__ import annotations

import json
from pathlib import Path

from .summary import RunSummary


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_dir() -> Path:
    return project_root() / "runs"


def runs_jsonl_path() -> Path:
    return runs_dir() / "runs.jsonl"


def append_run_summary(summary: RunSummary, path: str | Path | None = None) -> Path:
    path = Path(path) if path is not None else runs_jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
    return path
