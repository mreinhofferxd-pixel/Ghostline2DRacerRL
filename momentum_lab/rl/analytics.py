"""Rank and visualize saved RL rollout runs.

B7.4a is intentionally read-only over existing RL artifacts: it does not train,
replay, or mutate the sim. It uses rollout summaries for fast scanning and the
cached replay frames for failure triage and static SVG exports.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from ..config import CONTROL_HZ, WORLD_HEIGHT, WORLD_WIDTH
from ..core.checkpoints import Gate
from ..core.sim import Simulation
from ..core.track import Track
from ..replay import Frame
from ..tracks import TrackError, load_track_by_id
from .rewards import RewardConfig, RewardState, compute_reward
from .rollout import RolloutArtifact, load_rollout, project_root, rl_runs_dir


STALL_SPEED = 40.0
RECENT_WINDOW_FRAMES = CONTROL_HZ

_MISSING = "-"
_VALID_GROUP_KEYS = {
    "model",
    "policy",
    "seed",
    "track_id",
    "reward_version",
    "status",
}


@dataclass(frozen=True)
class AnalyzedRun:
    """One rollout or summary row after triage signals have been derived."""

    source_path: Path
    artifact_path: Path | None
    summary_path: Path | None
    line_number: int | None
    model: str
    policy: str
    episode: int | None
    track_id: str
    seed: int | None
    reward_version: str
    valid: bool
    terminated: bool
    truncated: bool
    final_reason: str | None
    episode_steps: int
    lap_time: float
    checkpoint_index: int
    checkpoint_count: int
    total_reward: float
    wall_hits: int
    wall_scrape_time: float
    boosts_used: int
    drift_time: float
    peak_slip: float
    final_x: float | None
    final_y: float | None
    final_speed: float | None
    end_avg_speed: float | None
    final_wall: bool
    recent_wall: bool
    stalled: bool
    target_distance: float | None
    stage_progress: float | None
    status: str
    # Stable content-hash id (see ``_run_id``). Same behavioral rollout -> same id,
    # regardless of file path or the (no-op) eval seed, so it both dedupes identical
    # rollouts and gives a human a fixed handle ("run 142485") across re-runs.
    run_id: str = ""
    # How many behaviorally-identical rollouts collapsed into this representative.
    duplicate_count: int = field(default=1, compare=False)
    reward_components: dict[str, float] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    artifact: RolloutArtifact | None = field(default=None, compare=False, repr=False)

    @property
    def top_positive_reward(self) -> tuple[str, float] | None:
        values = {name: value for name, value in self.reward_components.items() if value > 0.0}
        if not values:
            return None
        return max(values.items(), key=lambda item: item[1])

    @property
    def top_negative_reward(self) -> tuple[str, float] | None:
        values = {name: value for name, value in self.reward_components.items() if value < 0.0}
        if not values:
            return None
        return min(values.items(), key=lambda item: item[1])

    @property
    def sort_key(self) -> tuple[float, ...]:
        """Higher is better for ranking."""
        valid_rank = 1.0 if self.valid else 0.0
        lap_rank = -self.lap_time if self.valid else 0.0
        progress = self.stage_progress if self.stage_progress is not None else -999.0
        stall_rank = -1.0 if self.stalled else 0.0
        return (
            valid_rank,
            lap_rank,
            float(self.checkpoint_index),
            progress,
            self.total_reward,
            -float(self.wall_hits),
            stall_rank,
            float(self.boosts_used),
            self.drift_time,
            -float(self.episode_steps),
        )


@dataclass(frozen=True)
class RankedGroup:
    key: tuple[tuple[str, str], ...]
    runs: tuple[AnalyzedRun, ...]

    @property
    def label(self) -> str:
        if not self.key:
            return "all"
        return " / ".join(f"{name}={value}" for name, value in self.key)


@dataclass(frozen=True)
class VisualReport:
    index_path: Path
    overview_svg_path: Path
    run_svg_paths: tuple[Path, ...]


@dataclass(frozen=True)
class TraceSectorStats:
    label: str
    start_time: float
    end_time: float
    split_time: float
    path_distance: float
    avg_speed: float
    min_speed: float
    max_speed: float
    drift_time: float
    wall_frames: int
    wall_contacts: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


@dataclass(frozen=True)
class BoostTrace:
    index: int
    start_time: float
    end_time: float
    duration: float
    entry_speed: float
    exit_speed: float
    entry_xy: tuple[float, float]
    exit_xy: tuple[float, float]
    entry_cp: int
    exit_cp: int


@dataclass(frozen=True)
class TraceRunStats:
    run: AnalyzedRun
    path_distance: float
    avg_speed: float
    sectors: tuple[TraceSectorStats, ...]
    boosts: tuple[BoostTrace, ...]
    wall_contact_windows: tuple[tuple[float, float], ...]
    svg_path: Path | None = None


@dataclass(frozen=True)
class TraceComparison:
    left: TraceRunStats
    right: TraceRunStats
    overview_svg_path: Path | None = None


def scan_runs(root: str | Path | None = None) -> list[AnalyzedRun]:
    """Scan ``root`` for rollout summaries and artifacts."""
    root_path = Path(root) if root is not None else rl_runs_dir()
    records: list[AnalyzedRun] = []
    seen_artifacts: set[Path] = set()

    for summary_path in sorted(root_path.rglob("*.jsonl")):
        records.extend(_scan_summary_file(summary_path, seen_artifacts))

    for artifact_path in sorted(root_path.rglob("*.json")):
        resolved = _safe_resolve(artifact_path)
        if resolved in seen_artifacts:
            continue
        artifact = _try_load_rollout(artifact_path)
        if artifact is None:
            continue
        records.append(
            _analyze_artifact(
                artifact,
                artifact_path=artifact_path,
                summary_path=None,
                line_number=None,
                row={},
            )
        )
        seen_artifacts.add(resolved)

    return records


def rank_runs(
    records: Iterable[AnalyzedRun],
    *,
    top_n: int = 30,
    group_by: Iterable[str] = ("model", "policy"),
) -> list[RankedGroup]:
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    group_keys = tuple(group_by)
    unknown = [key for key in group_keys if key not in _VALID_GROUP_KEYS]
    if unknown:
        raise ValueError(f"unknown group key(s): {', '.join(unknown)}")

    groups: dict[tuple[tuple[str, str], ...], list[AnalyzedRun]] = {}
    for record in records:
        key = tuple((name, _group_value(record, name)) for name in group_keys)
        groups.setdefault(key, []).append(record)

    ranked: list[RankedGroup] = []
    for key, runs in groups.items():
        distinct = _dedupe_runs(runs)
        top = sorted(distinct, key=lambda run: run.sort_key, reverse=True)[:top_n]
        ranked.append(RankedGroup(key=key, runs=tuple(top)))
    return sorted(ranked, key=lambda group: group.label)


def parse_group_by(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "all", "none"}:
        return ()
    keys = tuple(part.strip() for part in normalized.split(",") if part.strip())
    unknown = [key for key in keys if key not in _VALID_GROUP_KEYS]
    if unknown:
        allowed = ", ".join(sorted(_VALID_GROUP_KEYS))
        raise ValueError(f"unknown group key(s): {', '.join(unknown)}; allowed: {allowed}")
    return keys


def format_markdown_report(
    groups: Iterable[RankedGroup],
    *,
    total_records: int,
    root: str | Path | None = None,
) -> str:
    root_text = _display_path(Path(root)) if root is not None else _display_path(rl_runs_dir())
    lines = [
        "# Ghostline RL Run Analytics",
        "",
        f"Scanned {total_records} run(s) under `{root_text}`.",
        "",
    ]
    any_rows = False
    for group in groups:
        lines.append(f"## {group.label}")
        if not group.runs:
            lines.extend(["", "_No runs._", ""])
            continue
        any_rows = True
        lines.append(
            "| # | run | n | status | seed | cp | lap | progress | reward | walls | "
            "top + | top - | end speed | final xy | boosts | drift | artifact |"
        )
        lines.append(
            "| -: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "--- | --- | ---: | --- | ---: | ---: | --- |"
        )
        for rank, run in enumerate(group.runs, start=1):
            artifact = _display_path(run.artifact_path) if run.artifact_path else _MISSING
            lines.append(
                "| "
                f"{rank} | {run.run_id} | {run.duplicate_count} | "
                f"{run.status} | {_fmt_optional_int(run.seed)} | "
                f"{run.checkpoint_index}/{run.checkpoint_count} | "
                f"{_fmt_lap(run)} | "
                f"{_fmt_float(run.stage_progress)} | "
                f"{run.total_reward:.2f} | {run.wall_hits} | "
                f"{_fmt_component(run.top_positive_reward)} | "
                f"{_fmt_component(run.top_negative_reward)} | "
                f"{_fmt_optional_float(run.end_avg_speed, digits=0)} | "
                f"{_fmt_xy(run.final_x, run.final_y)} | {run.boosts_used} | "
                f"{run.drift_time:.2f}s | `{artifact}` |"
            )
        if any(run.reward_components for run in group.runs):
            lines.append("")
            lines.append("Reward components:")
            lines.append(
                "| # | progress | target speed | heading | checkpoint | finish | "
                "finish time | avg speed | path eff | time | wall hit | wall scrape | wall prox | drift | sum |"
            )
            lines.append(
                "| -: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for rank, run in enumerate(group.runs, start=1):
                c = run.reward_components
                lines.append(
                    "| "
                    f"{rank} | "
                    f"{_fmt_component_value(c.get('progress'))} | "
                    f"{_fmt_component_value(c.get('target_speed'))} | "
                    f"{_fmt_component_value(c.get('heading_alignment'))} | "
                    f"{_fmt_component_value(c.get('checkpoint'))} | "
                    f"{_fmt_component_value(c.get('finish'))} | "
                    f"{_fmt_component_value(c.get('finish_time'))} | "
                    f"{_fmt_component_value(c.get('avg_speed'))} | "
                    f"{_fmt_component_value(c.get('path_efficiency'))} | "
                    f"{_fmt_component_value(c.get('time'))} | "
                    f"{_fmt_component_value(c.get('wall_hit'))} | "
                    f"{_fmt_component_value(c.get('wall_scrape'))} | "
                    f"{_fmt_component_value(c.get('wall_proximity'))} | "
                    f"{_fmt_component_value(c.get('drift'))} | "
                    f"{_fmt_component_value(_reward_component_sum(c))} |"
                )
        lines.append("")
    if not any_rows:
        lines.append("_No RL rollout runs were found._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(
    groups: Iterable[RankedGroup],
    path: str | Path,
    *,
    total_records: int,
    root: str | Path | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_markdown_report(groups, total_records=total_records, root=root),
        encoding="utf-8",
    )
    return output_path


def write_visual_report(
    groups: Iterable[RankedGroup],
    visual_dir: str | Path,
    *,
    limit: int = 30,
) -> VisualReport:
    if limit < 1:
        raise ValueError("visual limit must be >= 1")
    visual_path = Path(visual_dir)
    visual_path.mkdir(parents=True, exist_ok=True)
    # Per-run SVGs are regenerated every report; clear stale ones (old rank numbers
    # or the pre-run-id naming) so the dir only holds the current ranking.
    for stale in visual_path.glob("run_*.svg"):
        stale.unlink()
    runs = _dedupe_runs(_flatten_groups(groups))[:limit]
    visual_runs = [run for run in runs if run.artifact is not None and run.artifact.replay.frames]
    if not visual_runs:
        overview_path = visual_path / "top_runs.svg"
        overview_path.write_text(_empty_svg("No rollout artifacts with frames"), encoding="utf-8")
        index_path = visual_path / "index.html"
        index_path.write_text(_visual_index_html(overview_path, (), ()), encoding="utf-8")
        return VisualReport(index_path=index_path, overview_svg_path=overview_path, run_svg_paths=())

    track = load_track_by_id(visual_runs[0].track_id)
    overview_path = visual_path / "top_runs.svg"
    overview_path.write_text(_overview_svg(track, visual_runs), encoding="utf-8")

    run_paths: list[Path] = []
    for i, run in enumerate(visual_runs, start=1):
        # rank prefix orders the files; the run id is the stable handle a human can
        # quote ("run 142485") and stays the same as the portfolio grows.
        filename = f"run_{i:03d}_id{run.run_id}_{_safe_name(run.model)}_{_safe_name(run.policy)}.svg"
        path = visual_path / filename
        path.write_text(_run_svg(track, run, rank=i), encoding="utf-8")
        run_paths.append(path)

    index_path = visual_path / "index.html"
    index_path.write_text(
        _visual_index_html(overview_path, tuple(run_paths), tuple(visual_runs)),
        encoding="utf-8",
    )
    return VisualReport(
        index_path=index_path,
        overview_svg_path=overview_path,
        run_svg_paths=tuple(run_paths),
    )


def find_run_by_id(records: Iterable[AnalyzedRun], run_id: str) -> AnalyzedRun:
    """Return the deduped analyzed run with ``run_id``.

    The id is the stable 6-digit content id shown in the ranked report. Matching
    after dedupe keeps old deterministic-eval triplicates from producing ambiguous
    results.
    """
    normalized = str(run_id).strip()
    matches = [run for run in _dedupe_runs(records) if run.run_id == normalized]
    if not matches:
        raise ValueError(f"run id {normalized!r} was not found")
    if len(matches) > 1:
        raise ValueError(f"run id {normalized!r} matched multiple distinct runs")
    return matches[0]


def dedupe_runs(records: Iterable[AnalyzedRun]) -> list[AnalyzedRun]:
    """Collapse behaviorally-identical rollouts to one representative row each.

    Public wrapper over the internal dedupe used by ranking, so other read-only
    consumers (e.g. the interactive visualizer's selection layer) collapse the old
    deterministic-eval triplicates the same way without re-implementing the keying.
    """
    return _dedupe_runs(records)


def run_trace_stats(
    run: AnalyzedRun,
    *,
    visual_dir: str | Path | None = None,
) -> TraceRunStats:
    """Per-run path/sector/boost/wall trace stats for one visualizable rollout.

    Public wrapper over the internal trace stats used by ``compare_run_traces`` so
    the visualizer can reuse the exact same sector splits. Raises if ``run`` has no
    cached frames to derive a trajectory from.
    """
    visual_path = Path(visual_dir) if visual_dir is not None else None
    return _trace_run_stats(run, visual_dir=visual_path)


def compare_run_traces(
    left: AnalyzedRun,
    right: AnalyzedRun,
    *,
    visual_dir: str | Path | None = None,
) -> TraceComparison:
    """Compare two visualizable rollout traces without mutating artifacts."""
    visual_path = Path(visual_dir) if visual_dir is not None else None
    left_stats = _trace_run_stats(left, visual_dir=visual_path)
    right_stats = _trace_run_stats(right, visual_dir=visual_path)
    overview = None
    if visual_path is not None:
        candidate = visual_path / "top_runs.svg"
        if candidate.exists():
            overview = candidate
    return TraceComparison(left=left_stats, right=right_stats, overview_svg_path=overview)


def format_trace_comparison_report(
    comparison: TraceComparison,
    *,
    root: str | Path | None = None,
) -> str:
    left = comparison.left
    right = comparison.right
    lines = [
        "# Ghostline Trace Comparison",
        "",
        f"Left: {_run_label(left.run)}",
        f"Right: {_run_label(right.run)}",
        "",
    ]
    links = _trace_links(comparison)
    if links:
        lines.extend(["## Trajectory links", *links, ""])

    lines.extend(
        [
            "## Headline",
            "| metric | left | right | right - left |",
            "| --- | ---: | ---: | ---: |",
            _metric_row("lap", left.run.lap_time, right.run.lap_time, suffix="s"),
            _metric_row("steps", left.run.episode_steps, right.run.episode_steps, digits=0),
            _metric_row("path", left.path_distance, right.path_distance, suffix=" px", digits=1),
            _metric_row("avg speed", left.avg_speed, right.avg_speed, suffix=" px/s", digits=0),
            _metric_row("wall hits", left.run.wall_hits, right.run.wall_hits, digits=0),
            _metric_row("boosts", left.run.boosts_used, right.run.boosts_used, digits=0),
            _metric_row("drift", left.run.drift_time, right.run.drift_time, suffix="s"),
            "",
        ]
    )

    sector_pairs = list(zip(left.sectors, right.sectors))
    if sector_pairs:
        largest_time = max(sector_pairs, key=lambda pair: pair[1].split_time - pair[0].split_time)
        largest_path = max(sector_pairs, key=lambda pair: pair[1].path_distance - pair[0].path_distance)
        lines.extend(
            [
                "## Findings",
                (
                    f"- Largest time gap: {largest_time[0].label} "
                    f"({_fmt_signed(largest_time[1].split_time - largest_time[0].split_time)}s)."
                ),
                (
                    f"- Largest path gap: {largest_path[0].label} "
                    f"({_fmt_signed(largest_path[1].path_distance - largest_path[0].path_distance, digits=1)} px)."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Sectors",
            (
                "| sector | left split | right split | gap | left path | right path | "
                "path gap | left avg/min/max | right avg/min/max | drift gap | wall frames |"
            ),
            (
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: | ---: |"
            ),
        ]
    )
    for l_sector, r_sector in sector_pairs:
        lines.append(
            "| "
            f"{l_sector.label} | "
            f"{l_sector.split_time:.3f}s | {r_sector.split_time:.3f}s | "
            f"{_fmt_signed(r_sector.split_time - l_sector.split_time)}s | "
            f"{l_sector.path_distance:.1f} | {r_sector.path_distance:.1f} | "
            f"{_fmt_signed(r_sector.path_distance - l_sector.path_distance, digits=1)} | "
            f"{_speed_triplet(l_sector)} | {_speed_triplet(r_sector)} | "
            f"{_fmt_signed(r_sector.drift_time - l_sector.drift_time)}s | "
            f"{l_sector.wall_frames}/{r_sector.wall_frames} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Boost Windows",
            "| boost | left window | right window | left entry->exit | right entry->exit | exit gap |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for index in range(max(len(left.boosts), len(right.boosts))):
        l_boost = left.boosts[index] if index < len(left.boosts) else None
        r_boost = right.boosts[index] if index < len(right.boosts) else None
        lines.append(
            "| "
            f"{index + 1} | "
            f"{_fmt_boost_window(l_boost)} | {_fmt_boost_window(r_boost)} | "
            f"{_fmt_boost_speeds(l_boost)} | {_fmt_boost_speeds(r_boost)} | "
            f"{_fmt_boost_exit_gap(l_boost, r_boost)} |"
        )
    if not left.boosts and not right.boosts:
        lines.append("| - | - | - | - | - | - |")
    lines.append("")

    lines.extend(
        [
            "## Wall Contacts",
            f"- Left: {_fmt_windows(left.wall_contact_windows)}; summary hits={left.run.wall_hits}.",
            f"- Right: {_fmt_windows(right.wall_contact_windows)}; summary hits={right.run.wall_hits}.",
            "",
            "## Artifacts",
            f"- Left artifact: `{_display_path(left.run.artifact_path)}`",
            f"- Right artifact: `{_display_path(right.run.artifact_path)}`",
        ]
    )
    if root is not None:
        lines.append(f"- Root: `{_display_path(Path(root))}`")
    return "\n".join(lines).rstrip() + "\n"


def write_trace_comparison_report(
    comparison: TraceComparison,
    path: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_trace_comparison_report(comparison, root=root),
        encoding="utf-8",
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank and visualize saved Ghostline RL rollout runs."
    )
    parser.add_argument("--root", type=Path, default=rl_runs_dir())
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--group-by",
        default="model,policy",
        help=(
            "Comma-separated grouping keys. Use 'all' for one global table. "
            "Allowed: model, policy, seed, track_id, reward_version, status."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown report path.")
    parser.add_argument("--visual-dir", type=Path, default=None, help="Optional SVG/HTML output dir.")
    parser.add_argument("--visual-limit", type=int, default=30)
    parser.add_argument(
        "--compare-run",
        nargs=2,
        metavar=("LEFT_RUN_ID", "RIGHT_RUN_ID"),
        default=None,
        help="Write/print a read-only trace comparison for two stable run ids.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=None,
        help="Optional markdown path for --compare-run output.",
    )
    parser.add_argument(
        "--compare-visual-dir",
        type=Path,
        default=None,
        help="Directory containing existing per-run SVGs for comparison links.",
    )
    parser.add_argument("--no-print", action="store_true")
    args = parser.parse_args(argv)

    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.visual_limit < 1:
        parser.error("--visual-limit must be >= 1")
    try:
        group_by = parse_group_by(args.group_by)
    except ValueError as e:
        parser.error(str(e))

    records = scan_runs(args.root)
    if args.compare_run is not None:
        visual_dir = args.compare_visual_dir or args.visual_dir or (args.root / "analysis")
        try:
            left = find_run_by_id(records, args.compare_run[0])
            right = find_run_by_id(records, args.compare_run[1])
            comparison = compare_run_traces(left, right, visual_dir=visual_dir)
        except ValueError as e:
            parser.error(str(e))
        report = format_trace_comparison_report(comparison, root=args.root)
        output_path = args.compare_output or args.output
        if output_path is not None:
            write_trace_comparison_report(comparison, output_path, root=args.root)
        if not args.no_print:
            print(report, end="")
        return 0

    groups = rank_runs(records, top_n=args.top, group_by=group_by)
    report = format_markdown_report(groups, total_records=len(records), root=args.root)

    if args.output is not None:
        write_markdown_report(groups, args.output, total_records=len(records), root=args.root)
    if args.visual_dir is not None:
        visual = write_visual_report(groups, args.visual_dir, limit=args.visual_limit)
        report += (
            f"\nVisual report: `{_display_path(visual.index_path)}`\n"
            f"Overview SVG: `{_display_path(visual.overview_svg_path)}`\n"
        )
    if not args.no_print:
        print(report, end="")
    return 0


def _scan_summary_file(
    summary_path: Path,
    seen_artifacts: set[Path],
) -> list[AnalyzedRun]:
    records: list[AnalyzedRun] = []
    for line_number, line in enumerate(summary_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{summary_path}:{line_number}: invalid JSONL row ({e})") from e
        if not isinstance(row, dict):
            raise ValueError(f"{summary_path}:{line_number}: expected a JSON object")
        artifact_path = _resolve_artifact_path(row.get("artifact_path"), summary_path)
        artifact = _try_load_rollout(artifact_path) if artifact_path is not None else None
        if artifact is not None:
            seen_artifacts.add(_safe_resolve(artifact_path))
            records.append(
                _analyze_artifact(
                    artifact,
                    artifact_path=artifact_path,
                    summary_path=summary_path,
                    line_number=line_number,
                    row=row,
                )
            )
        else:
            records.append(_analyze_summary_row(row, summary_path, line_number, artifact_path))
    return records


def _analyze_artifact(
    artifact: RolloutArtifact,
    *,
    artifact_path: Path,
    summary_path: Path | None,
    line_number: int | None,
    row: dict[str, Any],
) -> AnalyzedRun:
    summary = artifact.summary
    frames = artifact.replay.frames
    final = frames[-1] if frames else None
    end_avg_speed = _average_speed(frames[-RECENT_WINDOW_FRAMES:])
    final_wall = bool(final.wall) if final is not None else False
    recent_wall = any(frame.wall for frame in frames[-RECENT_WINDOW_FRAMES:])
    stalled = bool(end_avg_speed is not None and end_avg_speed < STALL_SPEED and not summary.valid)
    target_distance, stage_progress = _target_metrics(summary.track_id, frames, summary.checkpoint_index)
    reward_components = _reward_component_totals(artifact)
    status = _status(
        valid=summary.valid,
        truncated=summary.truncated,
        final_reason=summary.final_reason,
        stalled=stalled,
        recent_wall=recent_wall,
        final_wall=final_wall,
        wall_hits=summary.wall_hits,
    )
    run = AnalyzedRun(
        source_path=summary_path or artifact_path,
        artifact_path=artifact_path,
        summary_path=summary_path,
        line_number=line_number,
        model=_infer_model(row, artifact_path=artifact_path, summary_path=summary_path),
        policy=_infer_policy(row, artifact),
        episode=_optional_int(row.get("episode")) or _episode_from_path(artifact_path),
        track_id=summary.track_id,
        seed=summary.seed,
        reward_version=summary.reward_version,
        valid=summary.valid,
        terminated=summary.terminated,
        truncated=summary.truncated,
        final_reason=summary.final_reason,
        episode_steps=summary.episode_steps,
        lap_time=summary.lap_time,
        checkpoint_index=summary.checkpoint_index,
        checkpoint_count=summary.checkpoint_count,
        total_reward=summary.total_reward,
        wall_hits=summary.wall_hits,
        wall_scrape_time=summary.wall_scrape_time,
        boosts_used=summary.boosts_used,
        drift_time=summary.drift_time,
        peak_slip=summary.peak_slip,
        final_x=None if final is None else final.x,
        final_y=None if final is None else final.y,
        final_speed=None if final is None else final.speed,
        end_avg_speed=end_avg_speed,
        final_wall=final_wall,
        recent_wall=recent_wall,
        stalled=stalled,
        target_distance=target_distance,
        stage_progress=stage_progress,
        status=status,
        reward_components=reward_components,
        artifact=artifact,
    )
    return _with_run_id(run)


def _analyze_summary_row(
    row: dict[str, Any],
    summary_path: Path,
    line_number: int,
    artifact_path: Path | None,
) -> AnalyzedRun:
    checkpoint_index = _int(row.get("checkpoint_index"), default=0)
    checkpoint_count = _int(row.get("checkpoint_count"), default=0)
    valid = _bool(row.get("valid"), default=False)
    truncated = _bool(row.get("truncated"), default=False)
    final_reason = _optional_str(row.get("final_reason"))
    wall_hits = _int(row.get("wall_hits"), default=0)
    status = _status(
        valid=valid,
        truncated=truncated,
        final_reason=final_reason,
        stalled=False,
        recent_wall=False,
        final_wall=False,
        wall_hits=wall_hits,
    )
    run = AnalyzedRun(
        source_path=summary_path,
        artifact_path=artifact_path,
        summary_path=summary_path,
        line_number=line_number,
        model=_infer_model(row, artifact_path=artifact_path, summary_path=summary_path),
        policy=_optional_str(row.get("policy")) or _MISSING,
        episode=_optional_int(row.get("episode")),
        track_id=_optional_str(row.get("track_id")) or _MISSING,
        seed=_optional_int(row.get("seed")),
        reward_version=_optional_str(row.get("reward_version")) or _MISSING,
        valid=valid,
        terminated=_bool(row.get("terminated"), default=False),
        truncated=truncated,
        final_reason=final_reason,
        episode_steps=_int(row.get("episode_steps"), default=0),
        lap_time=_float(row.get("lap_time"), default=0.0),
        checkpoint_index=checkpoint_index,
        checkpoint_count=checkpoint_count,
        total_reward=_float(row.get("total_reward"), default=0.0),
        wall_hits=wall_hits,
        wall_scrape_time=_float(row.get("wall_scrape_time"), default=0.0),
        boosts_used=_int(row.get("boosts_used"), default=0),
        drift_time=_float(row.get("drift_time"), default=0.0),
        peak_slip=_float(row.get("peak_slip"), default=0.0),
        final_x=None,
        final_y=None,
        final_speed=None,
        end_avg_speed=None,
        final_wall=False,
        recent_wall=False,
        stalled=False,
        target_distance=None,
        stage_progress=None,
        status=status,
        artifact=None,
    )
    return _with_run_id(run)


def _trace_run_stats(run: AnalyzedRun, *, visual_dir: Path | None) -> TraceRunStats:
    artifact = run.artifact
    if artifact is None or not artifact.replay.frames:
        raise ValueError(f"run {run.run_id} has no rollout artifact frames to compare")
    frames = artifact.replay.frames
    computed_path = _path_distance_from_points(
        [_initial_xy(artifact), *((frame.x, frame.y) for frame in frames)]
    )
    path_distance = artifact.summary.path_distance if artifact.summary.path_distance > 0.0 else computed_path
    avg_speed = path_distance / run.lap_time if run.lap_time > 0.0 else 0.0
    return TraceRunStats(
        run=run,
        path_distance=path_distance,
        avg_speed=avg_speed,
        sectors=_trace_sectors(run, frames, _initial_xy(artifact)),
        boosts=_boost_traces(frames),
        wall_contact_windows=_frame_windows(frames, lambda frame: frame.wall),
        svg_path=_find_run_svg(run, visual_dir),
    )


def _trace_sectors(
    run: AnalyzedRun,
    frames: tuple[Frame, ...],
    initial_xy: tuple[float, float],
) -> tuple[TraceSectorStats, ...]:
    checkpoint_count = max(1, run.checkpoint_count)
    end_indices: list[tuple[str, int, float]] = []
    for target_cp in range(1, checkpoint_count):
        end_idx = _first_cp_index(frames, target_cp)
        if target_cp == 1:
            label = "start->CP1"
        else:
            label = f"CP{target_cp - 1}->CP{target_cp}"
        end_indices.append((label, end_idx, frames[end_idx].t))

    final_label = "start->finish" if checkpoint_count == 1 else f"CP{checkpoint_count - 1}->finish"
    final_time = run.lap_time if run.valid and run.lap_time > 0.0 else frames[-1].t
    end_indices.append((final_label, len(frames) - 1, final_time))

    sectors: list[TraceSectorStats] = []
    prev_end_idx = -1
    prev_time = 0.0
    prev_xy = initial_xy
    for label, end_idx, end_time in end_indices:
        start_idx = min(prev_end_idx + 1, end_idx)
        sector_frames = frames[start_idx : end_idx + 1]
        if not sector_frames:
            sector_frames = (frames[end_idx],)
        points = [prev_xy, *((frame.x, frame.y) for frame in sector_frames)]
        speeds = [frame.speed for frame in sector_frames]
        drift_time = sum(1 for frame in sector_frames if frame.drift) / CONTROL_HZ
        wall_frames = sum(1 for frame in sector_frames if frame.wall)
        sectors.append(
            TraceSectorStats(
                label=label,
                start_time=prev_time,
                end_time=end_time,
                split_time=max(0.0, end_time - prev_time),
                path_distance=_path_distance_from_points(points),
                avg_speed=sum(speeds) / len(speeds),
                min_speed=min(speeds),
                max_speed=max(speeds),
                drift_time=drift_time,
                wall_frames=wall_frames,
                wall_contacts=len(_frame_windows(sector_frames, lambda frame: frame.wall)),
                start_xy=prev_xy,
                end_xy=(frames[end_idx].x, frames[end_idx].y),
            )
        )
        prev_end_idx = end_idx
        prev_time = end_time
        prev_xy = (frames[end_idx].x, frames[end_idx].y)
    return tuple(sectors)


def _first_cp_index(frames: tuple[Frame, ...], target_cp: int) -> int:
    for i, frame in enumerate(frames):
        if frame.cp >= target_cp:
            return i
    return len(frames) - 1


def _boost_traces(frames: tuple[Frame, ...]) -> tuple[BoostTrace, ...]:
    traces: list[BoostTrace] = []
    start_idx: int | None = None
    for i, frame in enumerate(frames):
        if frame.boost and start_idx is None:
            start_idx = i
        elif not frame.boost and start_idx is not None:
            traces.append(_boost_trace(len(traces) + 1, frames, start_idx, i))
            start_idx = None
    if start_idx is not None:
        traces.append(_boost_trace(len(traces) + 1, frames, start_idx, len(frames) - 1))
    return tuple(traces)


def _boost_trace(index: int, frames: tuple[Frame, ...], start_idx: int, end_idx: int) -> BoostTrace:
    start = frames[start_idx]
    end = frames[end_idx]
    return BoostTrace(
        index=index,
        start_time=start.t,
        end_time=end.t,
        duration=max(0.0, end.t - start.t),
        entry_speed=start.speed,
        exit_speed=end.speed,
        entry_xy=(start.x, start.y),
        exit_xy=(end.x, end.y),
        entry_cp=start.cp,
        exit_cp=end.cp,
    )


def _frame_windows(
    frames: tuple[Frame, ...],
    predicate,
) -> tuple[tuple[float, float], ...]:
    windows: list[tuple[float, float]] = []
    start: Frame | None = None
    last_true: Frame | None = None
    for frame in frames:
        if predicate(frame):
            if start is None:
                start = frame
            last_true = frame
        elif start is not None and last_true is not None:
            windows.append((start.t, last_true.t))
            start = None
            last_true = None
    if start is not None and last_true is not None:
        windows.append((start.t, last_true.t))
    return tuple(windows)


def _initial_xy(artifact: RolloutArtifact) -> tuple[float, float]:
    x, y, _vx, _vy, _heading = artifact.replay.initial_state.car
    return x, y


def _path_distance_from_points(points: Iterable[tuple[float, float]]) -> float:
    total = 0.0
    previous: tuple[float, float] | None = None
    for point in points:
        if previous is not None:
            total += _distance(previous[0], previous[1], point[0], point[1])
        previous = point
    return total


def _find_run_svg(run: AnalyzedRun, visual_dir: Path | None) -> Path | None:
    if visual_dir is None:
        return None
    matches = sorted(visual_dir.glob(f"run_*_id{run.run_id}_*.svg"))
    return matches[0] if matches else None


def _target_metrics(
    track_id: str,
    frames: tuple[Frame, ...],
    checkpoint_index: int,
) -> tuple[float | None, float | None]:
    if not frames:
        return None, None
    try:
        track = load_track_by_id(track_id)
    except TrackError:
        return None, None
    target = _target_gate(track, checkpoint_index)
    if target is None:
        return None, None

    final = frames[-1]
    stage_frame = next((frame for frame in frames if frame.cp == checkpoint_index), frames[0])
    start_distance = _distance(stage_frame.x, stage_frame.y, *target.center)
    final_distance = _distance(final.x, final.y, *target.center)
    if start_distance <= 1e-9:
        progress = 0.0
    else:
        progress = (start_distance - final_distance) / start_distance
    return final_distance, progress


def _target_gate(track: Track, checkpoint_index: int) -> Gate | None:
    if checkpoint_index < len(track.checkpoints):
        return track.checkpoints[checkpoint_index]
    return track.finish


def _reward_component_totals(artifact: RolloutArtifact) -> dict[str, float]:
    if not artifact.replay.actions:
        return {}
    try:
        track = load_track_by_id(artifact.replay.track_id)
    except TrackError:
        return {}

    cfg = _reward_config_from_payload(artifact.summary.reward_config)
    sim = Simulation()
    sim.reset(track=track, seed=artifact.replay.seed)
    artifact.replay.initial_state.apply_to(sim)

    totals: dict[str, float] = {}
    for action in artifact.replay.actions:
        before_world = sim.snapshot()
        before_reward = RewardState.from_world(before_world)
        after_world = sim.step(action)
        breakdown = compute_reward(before_reward, before_world, after_world, cfg).to_dict()
        for name, value in breakdown.items():
            if name == "total":
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
    return {name: totals[name] for name in sorted(totals)}


def _reward_config_from_payload(payload: dict[str, float | str]) -> RewardConfig:
    fields = RewardConfig.__dataclass_fields__
    kwargs: dict[str, Any] = {}
    for name in fields:
        if name in payload:
            kwargs[name] = payload[name]
    return RewardConfig(**kwargs)


def _status(
    *,
    valid: bool,
    truncated: bool,
    final_reason: str | None,
    stalled: bool,
    recent_wall: bool,
    final_wall: bool,
    wall_hits: int,
) -> str:
    if valid:
        return "lap_complete"
    if stalled and (recent_wall or final_wall):
        return "stalled_on_wall"
    if recent_wall or final_wall:
        return "wall_contact"
    if stalled:
        return "stalled"
    if wall_hits > 0:
        return "wall_hit_then_timeout" if truncated else "wall_hit"
    if truncated:
        return "time_limit"
    return final_reason or "unfinished"


def _average_speed(frames: Iterable[Frame]) -> float | None:
    values = [frame.speed for frame in frames]
    if not values:
        return None
    return sum(values) / len(values)


def _flatten_groups(groups: Iterable[RankedGroup]) -> list[AnalyzedRun]:
    runs: list[AnalyzedRun] = []
    for group in groups:
        runs.extend(group.runs)
    return sorted(runs, key=lambda run: run.sort_key, reverse=True)


def _run_signature(run: AnalyzedRun) -> str:
    """A behavioral fingerprint of a rollout, independent of file path or eval seed.

    For artifact runs it is the full cached trajectory (pose + state flags per
    frame); two byte-identical deterministic rollouts (e.g. the old per-episode
    triplicates) share it, while genuinely different lines differ. Summary-only rows
    (no cached frames) fall back to their outcome metrics. The eval seed is excluded
    on purpose: it has no effect on a deterministic rollout, so it must not split
    identical runs into distinct ids.
    """
    art = run.artifact
    if art is not None and art.replay.frames:
        body = ";".join(
            f"{f.x:.3f},{f.y:.3f},{f.angle:.4f},{f.speed:.3f},"
            f"{int(f.drift)}{int(f.wall)}{int(f.boost)},{f.cp}"
            for f in art.replay.frames
        )
        return f"{run.model}|{run.track_id}|{run.reward_version}|F|{body}"
    return "|".join(
        str(part)
        for part in (
            run.model,
            run.track_id,
            run.reward_version,
            "M",
            run.episode_steps,
            round(run.lap_time, 4),
            run.checkpoint_index,
            round(run.total_reward, 4),
            run.wall_hits,
            run.final_x,
            run.final_y,
            round(run.drift_time, 4),
            run.boosts_used,
        )
    )


def _run_id(run: AnalyzedRun) -> str:
    """Stable 6-digit id from the behavioral signature (e.g. ``"142485"``)."""
    digest = hashlib.sha1(_run_signature(run).encode("utf-8")).hexdigest()
    return f"{int(digest, 16) % 1_000_000:06d}"


def _with_run_id(run: AnalyzedRun) -> AnalyzedRun:
    return replace(run, run_id=_run_id(run))


def _dedupe_key(run: AnalyzedRun) -> tuple[str, ...]:
    """Return the leaderboard duplicate key."""
    rid = run.run_id or _run_id(run)
    if run.valid and run.lap_time > 0.0:
        return (
            "lap",
            f"{run.lap_time:.3f}",
            f"{run.total_reward:.2f}",
            str(run.wall_hits),
        )
    return ("behavior", rid)


def _dedupe_runs(runs: Iterable[AnalyzedRun]) -> list[AnalyzedRun]:
    """Collapse behaviorally-identical rollouts, keeping one representative.

    Keyed on the content ``run_id`` (not the file path), so the old per-episode
    triplicates — different files, identical trajectories — become a single row.
    The representative carries ``duplicate_count`` = how many collapsed.
    """
    by_key: dict[tuple[str, ...], AnalyzedRun] = {}
    order: list[tuple[str, ...]] = []
    for run in runs:
        key = _dedupe_key(run)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = run if run.duplicate_count else replace(run, duplicate_count=1)
            order.append(key)
        else:
            count = existing.duplicate_count + run.duplicate_count
            representative = existing if existing.sort_key >= run.sort_key else run
            by_key[key] = replace(representative, duplicate_count=count)
    return [by_key[key] for key in order]


def _group_value(run: AnalyzedRun, name: str) -> str:
    value = getattr(run, name)
    if value is None:
        return _MISSING
    if name == "seed":
        return _seed_name(value)
    return str(value)


def _infer_model(
    row: dict[str, Any],
    *,
    artifact_path: Path | None,
    summary_path: Path | None,
) -> str:
    explicit = _optional_str(row.get("model"))
    if explicit:
        return explicit
    if artifact_path is not None and artifact_path.parent.parent.name == "evals":
        return artifact_path.parent.name
    if summary_path is not None and summary_path.parent.name == "evals":
        return summary_path.stem
    if summary_path is not None and summary_path.parent.parent.name == "evals":
        return summary_path.parent.name
    return _MISSING


def _infer_policy(row: dict[str, Any], artifact: RolloutArtifact) -> str:
    explicit = _optional_str(row.get("policy"))
    if explicit:
        return explicit
    actions = artifact.summary.action_adapter.get("kind")
    if isinstance(actions, str) and actions:
        return actions
    return _MISSING


def _episode_from_path(path: Path) -> int | None:
    match = re.search(r"_ep(\d+)_", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _try_load_rollout(path: Path | None) -> RolloutArtifact | None:
    if path is None or not path.exists():
        return None
    try:
        return load_rollout(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _resolve_artifact_path(value: Any, summary_path: Path) -> Path | None:
    raw = _optional_str(value)
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    root_candidate = project_root() / path
    if root_candidate.exists():
        return root_candidate
    summary_candidate = summary_path.parent / path
    if summary_candidate.exists():
        return summary_candidate
    return root_candidate


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _display_path(path: Path | None) -> str:
    if path is None:
        return _MISSING
    try:
        return str(path.resolve().relative_to(project_root()))
    except (OSError, ValueError):
        return str(path)


def _seed_name(seed: int | None) -> str:
    return "seed_none" if seed is None else f"seed_{seed}"


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def _run_label(run: AnalyzedRun) -> str:
    return (
        f"`{run.run_id}` {run.model}/{run.policy} "
        f"({run.status}, lap={_fmt_lap(run)}, artifact=`{_display_path(run.artifact_path)}`)"
    )


def _trace_links(comparison: TraceComparison) -> list[str]:
    links: list[str] = []
    if comparison.overview_svg_path is not None:
        links.append(f"- Overview: `{_display_path(comparison.overview_svg_path)}`")
    if comparison.left.svg_path is not None:
        links.append(f"- Left SVG: `{_display_path(comparison.left.svg_path)}`")
    if comparison.right.svg_path is not None:
        links.append(f"- Right SVG: `{_display_path(comparison.right.svg_path)}`")
    return links


def _metric_row(
    label: str,
    left: float,
    right: float,
    *,
    suffix: str = "",
    digits: int = 3,
) -> str:
    return (
        f"| {label} | {_fmt_metric(left, suffix=suffix, digits=digits)} | "
        f"{_fmt_metric(right, suffix=suffix, digits=digits)} | "
        f"{_fmt_signed(right - left, suffix=suffix, digits=digits)} |"
    )


def _fmt_metric(value: float, *, suffix: str = "", digits: int = 3) -> str:
    return f"{value:.{digits}f}{suffix}"


def _fmt_signed(value: float, *, suffix: str = "", digits: int = 3) -> str:
    return f"{value:+.{digits}f}{suffix}"


def _speed_triplet(sector: TraceSectorStats) -> str:
    return f"{sector.avg_speed:.0f}/{sector.min_speed:.0f}/{sector.max_speed:.0f}"


def _fmt_boost_window(boost: BoostTrace | None) -> str:
    if boost is None:
        return _MISSING
    return f"{boost.start_time:.3f}-{boost.end_time:.3f}s (cp {boost.entry_cp}->{boost.exit_cp})"


def _fmt_boost_speeds(boost: BoostTrace | None) -> str:
    if boost is None:
        return _MISSING
    return f"{boost.entry_speed:.0f}->{boost.exit_speed:.0f}"


def _fmt_boost_exit_gap(left: BoostTrace | None, right: BoostTrace | None) -> str:
    if left is None or right is None:
        return _MISSING
    return f"{_fmt_signed(right.exit_speed - left.exit_speed, digits=0)} px/s"


def _fmt_windows(windows: tuple[tuple[float, float], ...]) -> str:
    if not windows:
        return "none"
    return ", ".join(f"{start:.3f}-{end:.3f}s" for start, end in windows)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _int(value: Any, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _fmt_optional_int(value: int | None) -> str:
    return _MISSING if value is None else str(value)


def _fmt_optional_float(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return _MISSING
    return f"{value:.{digits}f}"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return _MISSING
    return f"{value:.2f}"


def _fmt_lap(run: AnalyzedRun) -> str:
    """Lap time, shown only for valid laps (it is the time-trial ranking metric).

    Millisecond precision so sub-tick laps that share a closing control tick stay
    distinguishable (B9): at 2 decimals two such laps collapse to the same string.
    """
    if not run.valid:
        return _MISSING
    return f"{run.lap_time:.3f}s"


def _fmt_xy(x: float | None, y: float | None) -> str:
    if x is None or y is None:
        return _MISSING
    return f"{x:.0f},{y:.0f}"


def _fmt_component(component: tuple[str, float] | None) -> str:
    if component is None:
        return _MISSING
    name, value = component
    return f"{name} {value:+.2f}"


def _fmt_component_value(value: float | None) -> str:
    if value is None:
        return _MISSING
    return f"{value:+.2f}"


def _reward_component_sum(components: dict[str, float]) -> float | None:
    if not components:
        return None
    return sum(components.values())


def _empty_svg(message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WORLD_WIDTH:.0f} {WORLD_HEIGHT:.0f}">'
        '<rect width="100%" height="100%" fill="#181a1e"/>'
        f'<text x="32" y="48" fill="#f2f4f8" font-family="Arial" font-size="24">'
        f"{html.escape(message)}</text></svg>\n"
    )


def _overview_svg(track: Track, runs: list[AnalyzedRun]) -> str:
    parts = [_svg_header("Top RL runs"), _svg_track(track)]
    colors = (
        "#e4572e",
        "#17bebb",
        "#ffc914",
        "#7768ae",
        "#76b041",
        "#ff6b6b",
        "#4d96ff",
        "#d65db1",
        "#00a676",
        "#f78154",
    )
    for i, run in enumerate(runs, start=1):
        assert run.artifact is not None
        frames = run.artifact.replay.frames
        color = colors[(i - 1) % len(colors)]
        parts.append(
            _polyline(
                frames,
                stroke=color,
                stroke_width=4.0,
                opacity=0.55,
                max_points=500,
            )
        )
        final = frames[-1]
        parts.append(
            f'<circle cx="{final.x:.2f}" cy="{final.y:.2f}" r="6" fill="{color}" '
            'stroke="#111318" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{final.x + 8:.2f}" y="{final.y - 8:.2f}" fill="#f2f4f8" '
            f'font-family="Arial" font-size="14">{i}</text>'
        )
    parts.append(_svg_footer())
    return "\n".join(parts)


def _run_svg(track: Track, run: AnalyzedRun, *, rank: int) -> str:
    assert run.artifact is not None
    frames = run.artifact.replay.frames
    parts = [_svg_header(f"Run {rank}"), _svg_track(track)]
    parts.append(_polyline(frames, stroke="#f2f4f8", stroke_width=3.0, opacity=0.75))
    parts.extend(_state_segments(frames, lambda frame: frame.drift, "#4d96ff", 5.0))
    parts.extend(_state_segments(frames, lambda frame: frame.boost, "#32d583", 6.0))
    parts.extend(_state_segments(frames, lambda frame: frame.wall, "#ff453a", 7.0))
    first, final = frames[0], frames[-1]
    parts.append(f'<circle cx="{first.x:.2f}" cy="{first.y:.2f}" r="5" fill="#32d583"/>')
    parts.append(
        f'<circle cx="{final.x:.2f}" cy="{final.y:.2f}" r="8" fill="#ffcc00" '
        'stroke="#111318" stroke-width="2"/>'
    )
    dup = f" | x{run.duplicate_count}" if run.duplicate_count > 1 else ""
    lap = f" | lap={run.lap_time:.3f}s" if run.valid else ""
    label = (
        f"#{rank} run {run.run_id} | {run.model}/{run.policy} | {run.status}{lap} | "
        f"cp={run.checkpoint_index}/{run.checkpoint_count} | "
        f"reward={run.total_reward:.2f} | walls={run.wall_hits}{dup}"
    )
    parts.append(
        '<rect x="18" y="18" width="980" height="38" rx="4" fill="#111318" '
        'fill-opacity="0.78"/>'
    )
    parts.append(
        f'<text x="32" y="43" fill="#f2f4f8" font-family="Arial" font-size="17">'
        f"{html.escape(label)}</text>"
    )
    parts.append(_svg_footer())
    return "\n".join(parts)


def _svg_header(title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WORLD_WIDTH:.0f} {WORLD_HEIGHT:.0f}" '
        'width="1280" height="720" role="img" '
        f'aria-label="{html.escape(title)}">'
        '<rect width="100%" height="100%" fill="#181a1e"/>'
    )


def _svg_footer() -> str:
    return "</svg>\n"


def _svg_track(track: Track) -> str:
    parts: list[str] = []
    if track.surface_outer:
        parts.append(f'<polygon points="{_points(track.surface_outer)}" fill="#34383f"/>')
    if track.surface_inner:
        parts.append(f'<polygon points="{_points(track.surface_inner)}" fill="#1d3027"/>')
    for pad in track.boost_pads:
        left, top, right, bottom = pad.bounds
        parts.append(
            f'<rect x="{left:.2f}" y="{top:.2f}" width="{right - left:.2f}" '
            f'height="{bottom - top:.2f}" fill="#32d583" fill-opacity="0.22" '
            'stroke="#32d583" stroke-width="2"/>'
        )
    for segment in track.walls:
        parts.append(
            f'<line x1="{segment.x1:.2f}" y1="{segment.y1:.2f}" '
            f'x2="{segment.x2:.2f}" y2="{segment.y2:.2f}" '
            'stroke="#b5bdc9" stroke-width="3" stroke-linecap="round" '
            'stroke-opacity="0.65"/>'
        )
    for i, gate in enumerate(track.checkpoints, start=1):
        parts.append(_gate_line(gate, "#ffd166", label=str(i)))
    if track.finish is not None:
        parts.append(_gate_line(track.finish, "#f2f4f8", label="F"))
    return "\n".join(parts)


def _gate_line(gate: Gate, color: str, *, label: str) -> str:
    cx, cy = gate.center
    return (
        f'<line x1="{gate.x1:.2f}" y1="{gate.y1:.2f}" x2="{gate.x2:.2f}" y2="{gate.y2:.2f}" '
        f'stroke="{color}" stroke-width="4" stroke-dasharray="10 7" stroke-linecap="round"/>'
        f'<text x="{cx + 6:.2f}" y="{cy - 6:.2f}" fill="{color}" '
        f'font-family="Arial" font-size="18">{html.escape(label)}</text>'
    )


def _points(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _polyline(
    frames: Iterable[Frame],
    *,
    stroke: str,
    stroke_width: float,
    opacity: float,
    max_points: int | None = None,
) -> str:
    frame_list = list(frames)
    if max_points is not None and len(frame_list) > max_points:
        step = max(1, len(frame_list) // max_points)
        sampled = frame_list[::step]
        if sampled[-1] is not frame_list[-1]:
            sampled.append(frame_list[-1])
        frame_list = sampled
    if len(frame_list) < 2:
        return ""
    points = " ".join(f"{frame.x:.2f},{frame.y:.2f}" for frame in frame_list)
    return (
        f'<polyline points="{points}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" stroke-opacity="{opacity:.2f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _state_segments(
    frames: tuple[Frame, ...],
    predicate,
    color: str,
    width: float,
) -> list[str]:
    segments: list[list[Frame]] = []
    current: list[Frame] = []
    for i, frame in enumerate(frames):
        if predicate(frame):
            if not current and i > 0:
                current.append(frames[i - 1])
            current.append(frame)
        elif current:
            if len(current) >= 2:
                segments.append(current)
            current = []
    if len(current) >= 2:
        segments.append(current)
    return [
        _polyline(segment, stroke=color, stroke_width=width, opacity=0.9)
        for segment in segments
    ]


def _visual_index_html(
    overview_path: Path,
    run_paths: tuple[Path, ...],
    runs: tuple[AnalyzedRun, ...],
) -> str:
    rows: list[str] = []
    for i, run in enumerate(runs, start=1):
        svg_name = run_paths[i - 1].name if i - 1 < len(run_paths) else ""
        artifact = _display_path(run.artifact_path)
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(run.run_id)}</td>"
            f"<td>{run.duplicate_count}</td>"
            f"<td>{html.escape(run.model)}</td>"
            f"<td>{html.escape(run.policy)}</td>"
            f"<td>{html.escape(run.status)}</td>"
            f"<td>{_fmt_optional_int(run.seed)}</td>"
            f"<td>{run.checkpoint_index}/{run.checkpoint_count}</td>"
            f"<td>{_fmt_lap(run)}</td>"
            f"<td>{_fmt_float(run.stage_progress)}</td>"
            f"<td>{run.total_reward:.2f}</td>"
            f"<td>{run.wall_hits}</td>"
            f'<td><a href="{html.escape(svg_name)}">run svg</a></td>'
            f"<td><code>{html.escape(artifact)}</code></td>"
            "</tr>"
        )
    rows_html = "\n".join(rows) or '<tr><td colspan="14">No visualizable runs</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ghostline RL Run Analytics</title>
<style>
body {{ margin: 24px; font-family: Arial, sans-serif; background: #111318; color: #f2f4f8; }}
a {{ color: #7dd3fc; }}
img {{ max-width: 100%; border: 1px solid #38404a; background: #181a1e; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 14px; }}
th, td {{ border-bottom: 1px solid #303844; padding: 8px 10px; text-align: left; }}
th {{ color: #b5bdc9; }}
code {{ color: #d4d8e1; }}
</style>
</head>
<body>
<h1>Ghostline RL Run Analytics</h1>
<p>Overview of the top visualized rollout trajectories. Per-run SVGs highlight drift in blue, boost in green, and wall contact in red.</p>
<p><a href="{html.escape(overview_path.name)}">Open overview SVG</a></p>
<img src="{html.escape(overview_path.name)}" alt="Top rollout trajectory overview">
<table>
<thead>
<tr><th>#</th><th>Run</th><th>n</th><th>Model</th><th>Policy</th><th>Status</th><th>Seed</th><th>CP</th><th>Lap</th><th>Progress</th><th>Reward</th><th>Walls</th><th>Visual</th><th>Artifact</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
