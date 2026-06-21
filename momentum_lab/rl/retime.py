"""Retroactively re-time saved runs to sub-tick lap times (B9 migration).

Reporting/metrics layer only. The sub-tick finish-line interpolation B9 added is
applied live to *new* runs (``RunState.lap_time``); this brings the already-saved
portfolio up to parity without re-running or re-training anything.

For each legacy run it recovers the finish-line crossing fraction ``t`` from the
two cached control-tick frames straddling the close (the same positions the live
``RunState`` saw, so the recomputed ``t`` is bit-identical to what the sim would
have produced), then rewrites ``lap_time`` to ``integer_lap_time - (1 - t) * dt``
and stamps ``timing_v2``. ``lap_ticks`` (the canonical integer for the state
machine / replay validation), physics, frames, and fingerprints are untouched, so
replay validation is unaffected. The migration is idempotent: a record already
marked ``timing_v2`` is skipped.

Covers the three artifact shapes the reports read:
  * RolloutArtifact JSON  (``{"summary": ..., "replay": ...}``)  -- the leaderboard.
  * rollout summary JSONL rows  -- re-timed via their referenced artifact's frames.
  * bare ReplayData JSON under ``replays/``  -- the in-game best lap / ghost.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .. import TIMING_VERSION
from ..config import CONTROL_DT
from ..core.checkpoints import Gate
from ..tracks import TrackError, load_track_by_id
from .rollout import project_root, rl_runs_dir

LEGACY_TIMING_VERSION = "timing_v1"


@lru_cache(maxsize=None)
def _track_info(track_id: str) -> tuple[Gate, int] | None:
    """``(finish_gate, checkpoint_count)`` for ``track_id``; ``None`` if unloadable."""
    try:
        track = load_track_by_id(track_id)
    except TrackError:
        return None
    if track.finish is None:
        return None
    return track.finish, len(track.checkpoints)


def _frame_xyc(frame: Any) -> tuple[float, float, int]:
    if isinstance(frame, dict):
        return float(frame["x"]), float(frame["y"]), int(frame["cp"])
    return float(frame.x), float(frame.y), int(frame.cp)


def finish_fraction_from_frames(
    finish: Gate, frames: Iterable[Any], checkpoint_count: int
) -> float | None:
    """Fraction ``t`` in ``(0, 1]`` of the closing control step at which the lap's
    finish crossing happened, recovered from cached control-tick positions.

    The first forward crossing of the finish line *after every checkpoint is cleared*
    (``cp >= checkpoint_count``) is the lap close, matching ``RunState`` semantics.
    Returns ``None`` when no such crossing is present (e.g. an unfinished run). Frames
    may be dicts (stored JSON) or :class:`~momentum_lab.replay.recorder.Frame` objects.
    """
    prev: tuple[float, float] | None = None
    for frame in frames:
        x, y, cp = _frame_xyc(frame)
        if prev is not None and cp >= checkpoint_count:
            direction, t = finish.crossing_with_fraction(prev[0], prev[1], x, y)
            if direction > 0:
                return t
        prev = (x, y)
    return None


def subtick_lap_time(integer_lap_time: float, fraction: float) -> float:
    """Apply the sub-tick correction: drop the within-step remainder of the closing
    tick. ``fraction`` is the finish-crossing fraction (closer to 1 = later in the
    step = smaller correction)."""
    return integer_lap_time - (1.0 - fraction) * CONTROL_DT


def _replay_fraction(replay: dict[str, Any]) -> float | None:
    track_id = replay.get("track_id")
    if not isinstance(track_id, str):
        return None
    info = _track_info(track_id)
    if info is None:
        return None
    finish, checkpoint_count = info
    frames = replay.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        return None
    return finish_fraction_from_frames(finish, frames, checkpoint_count)


def _is_legacy(record: dict[str, Any]) -> bool:
    return record.get("timing_version") in (None, LEGACY_TIMING_VERSION)


def retime_artifact_dict(artifact: dict[str, Any]) -> bool:
    """Re-time a RolloutArtifact dict in place. Returns ``True`` if it was changed."""
    summary = artifact.get("summary")
    replay = artifact.get("replay")
    if not isinstance(summary, dict) or not isinstance(replay, dict):
        return False
    if not summary.get("valid") or not _is_legacy(summary):
        return False
    fraction = _replay_fraction(replay)
    if fraction is None:
        return False
    new_lap = subtick_lap_time(float(summary["lap_time"]), fraction)
    summary["lap_time"] = new_lap
    summary["timing_version"] = TIMING_VERSION
    # Keep the embedded replay block consistent with a freshly-recorded artifact.
    if "lap_time" in replay:
        replay["lap_time"] = new_lap
    replay["timing_version"] = TIMING_VERSION
    return True


def retime_replay_dict(replay: dict[str, Any]) -> bool:
    """Re-time a bare ReplayData dict in place. Returns ``True`` if it was changed."""
    if not isinstance(replay, dict) or "summary" in replay:
        return False  # an artifact wrapper, not a bare replay
    if not replay.get("valid") or not _is_legacy(replay):
        return False
    fraction = _replay_fraction(replay)
    if fraction is None:
        return False
    replay["lap_time"] = subtick_lap_time(float(replay["lap_time"]), fraction)
    replay["timing_version"] = TIMING_VERSION
    return True


def retime_summary_row(row: dict[str, Any], summary_path: Path) -> bool:
    """Re-time one rollout-summary JSONL row in place using its artifact's frames.

    The row stores only outcome metrics (no frames); the crossing fraction comes
    from the artifact it points at. Returns ``True`` if the row was changed.
    """
    if not isinstance(row, dict) or not row.get("valid") or not _is_legacy(row):
        return False
    artifact_path = _resolve_artifact_path(row.get("artifact_path"), summary_path)
    if artifact_path is None or not artifact_path.exists():
        return False
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    replay = artifact.get("replay") if isinstance(artifact, dict) else None
    if not isinstance(replay, dict):
        return False
    fraction = _replay_fraction(replay)
    if fraction is None:
        return False
    row["lap_time"] = subtick_lap_time(float(row["lap_time"]), fraction)
    row["timing_version"] = TIMING_VERSION
    return True


def _resolve_artifact_path(value: Any, summary_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_candidate = project_root() / candidate
    if root_candidate.exists():
        return root_candidate
    summary_candidate = summary_path.parent / candidate
    if summary_candidate.exists():
        return summary_candidate
    return root_candidate


@dataclass
class RetimeReport:
    artifacts_scanned: int = 0
    artifacts_changed: int = 0
    rows_scanned: int = 0
    rows_changed: int = 0
    replays_changed: int = 0
    files_written: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts_scanned": self.artifacts_scanned,
            "artifacts_changed": self.artifacts_changed,
            "rows_scanned": self.rows_scanned,
            "rows_changed": self.rows_changed,
            "replays_changed": self.replays_changed,
            "files_written": self.files_written,
        }


def _dump_full(data: Any) -> str:
    """Match the on-disk format the rollout/replay writers use."""
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def retime_tree(
    root: str | Path | None = None,
    *,
    replays_dir: str | Path | None = None,
    include_replays: bool = True,
    dry_run: bool = False,
) -> RetimeReport:
    """Re-time every legacy run under ``root`` (and ``replays/``) to sub-tick.

    Rewrites RolloutArtifact JSONs, rollout-summary JSONL rows, and bare ReplayData
    files in place (unless ``dry_run``). Idempotent and safe to re-run.
    """
    root_path = Path(root) if root is not None else rl_runs_dir()
    report = RetimeReport()

    for path in sorted(root_path.rglob("*.json")):
        artifact = _load_json(path)
        if not (
            isinstance(artifact, dict)
            and isinstance(artifact.get("summary"), dict)
            and isinstance(artifact.get("replay"), dict)
        ):
            continue
        report.artifacts_scanned += 1
        if retime_artifact_dict(artifact):
            report.artifacts_changed += 1
            report.files_written.append(str(path))
            if not dry_run:
                path.write_text(_dump_full(artifact), encoding="utf-8")

    for path in sorted(root_path.rglob("*.jsonl")):
        changed = False
        out_lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                out_lines.append(line)
                continue
            try:
                row = json.loads(stripped)
            except ValueError:
                out_lines.append(line)
                continue
            report.rows_scanned += 1
            if isinstance(row, dict) and retime_summary_row(row, path):
                report.rows_changed += 1
                changed = True
                out_lines.append(json.dumps(row, sort_keys=True))
            else:
                out_lines.append(line)
        if changed:
            report.files_written.append(str(path))
            if not dry_run:
                path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if include_replays:
        rdir = Path(replays_dir) if replays_dir is not None else project_root() / "replays"
        if rdir.exists():
            for path in sorted(rdir.glob("*.json")):
                replay = _load_json(path)
                if isinstance(replay, dict) and retime_replay_dict(replay):
                    report.replays_changed += 1
                    report.files_written.append(str(path))
                    if not dry_run:
                        path.write_text(_dump_full(replay), encoding="utf-8")

    return report


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-time saved Ghostline runs to sub-tick lap times (B9 migration)."
    )
    parser.add_argument("--root", type=Path, default=rl_runs_dir(), help="RL runs root to scan.")
    parser.add_argument("--replays-dir", type=Path, default=None, help="Override the replays/ dir.")
    parser.add_argument("--no-replays", action="store_true", help="Skip the replays/ files.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args(argv)

    report = retime_tree(
        args.root,
        replays_dir=args.replays_dir,
        include_replays=not args.no_replays,
        dry_run=args.dry_run,
    )
    payload = report.to_dict()
    payload["dry_run"] = args.dry_run
    payload["files_written"] = len(report.files_written)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
