"""Portfolio export: bundle the canonical Ghostline replay viewer, static
analytics, Track 1 geometry, and the two benchmark replays into a portable static
tree the personal website can host under ``/ghostline/...``.

GL-02 / GL-03 / GL-19. This reuses the already-generated outputs (the interactive replay
viewer from :mod:`momentum_lab.rl.visualizer` and the static analytics report from
:mod:`momentum_lab.rl.analytics`); it does not train, re-evaluate, or regenerate
them. It copies them, audits the HTML for portability (no ``file://`` or absolute
asset paths, and every referenced asset resolves), copies the canonical Track 1
JSON and the two benchmark replay artifacts, writes a small ``manifest.json``,
and emits golden browser-sim fixtures from the canonical Python sim.

Usage::

    python -m momentum_lab.portfolio_export --out <dir>

Output tree (relative to ``--out``)::

    replay_viewer/index.html
    analysis/index.html
    analysis/*.svg
    data/track_01_easy_loop.json
    data/ghostline_ai_3_965.json
    data/michi_dev_4_028.json
    data/manifest.json
    fixtures/straight_accel.json
    fixtures/brake_turn.json
    fixtures/wall_collision.json
    fixtures/boost_pad.json
    fixtures/checkpoint_finish.json
    fixtures/ai_record_replay.json

The two benchmark rows (Ghostline AI run 970622 and Michi/dev run 750982) are
fixed values matching the challenge plan; the export verifies the located replay
artifacts still match them so a future data change is caught rather than silently
exported. The AI fixture is built from that same run 970622 replay artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import PHYSICS_VERSION, TIMING_VERSION
from .config import DEFAULT_TRACK
from .portfolio_fixtures import (
    PortfolioFixtureError,
    write_fixtures as write_portfolio_fixtures,
)
from .rl.analytics import AnalyzedRun, find_run_by_id, scan_runs
from .rl.rollout import load_rollout, rl_runs_dir
from .tracks import tracks_dir

TRACK_ID = DEFAULT_TRACK  # "track_01_easy_loop"

# Tolerance when checking a located replay artifact against its locked benchmark
# row. Lap times are sub-tick (timing_v2) seconds; half a millisecond is plenty.
_LAP_TIME_TOLERANCE = 1e-3


class PortfolioExportError(RuntimeError):
    """A required source artifact is missing or fails the portability audit."""


@dataclass(frozen=True)
class Benchmark:
    """One fixed leaderboard benchmark row plus the data file it exports to.

    ``lap_time`` / ``lap_ticks`` / ``wall_hits`` are the locked values from the
    challenge plan. ``run_id`` is the stable content id used to find the canonical
    replay artifact in ``runs/rl``.
    """

    id: str
    label: str
    run_id: str
    lap_time: float
    lap_ticks: int
    wall_hits: int
    data_filename: str

    def manifest_row(self) -> dict[str, object]:
        # Exactly the shape the plan's manifest.json specifies (no extra keys).
        return {
            "id": self.id,
            "label": self.label,
            "run_id": self.run_id,
            "lap_time": self.lap_time,
            "lap_ticks": self.lap_ticks,
            "wall_hits": self.wall_hits,
        }


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        id="ghostline_ai",
        label="Ghostline AI",
        run_id="970622",
        lap_time=3.965,
        lap_ticks=239,
        wall_hits=0,
        data_filename="ghostline_ai_3_965.json",
    ),
    Benchmark(
        id="michi_dev",
        label="Michi/dev",
        run_id="750982",
        lap_time=4.028,
        lap_ticks=242,
        wall_hits=0,
        data_filename="michi_dev_4_028.json",
    ),
)


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    files: tuple[Path, ...]

    def tree(self) -> str:
        """A sorted, ``--out``-relative listing of every written file."""
        rels = sorted(str(p.relative_to(self.out_dir)).replace("\\", "/") for p in self.files)
        return "\n".join(rels)


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #
def _analysis_dir(root: Path) -> Path:
    return root / "analysis"


def _require(path: Path, *, what: str, fix: str) -> Path:
    if not path.exists():
        raise PortfolioExportError(f"missing {what}: {path}\n  {fix}")
    return path


def replay_viewer_source(root: Path) -> Path:
    return _require(
        _analysis_dir(root) / "replay_viewer" / "index.html",
        what="replay viewer",
        fix=(
            "Generate it first, e.g.:\n"
            "    python -m momentum_lab.rl.visualizer --story progression --reference 750982"
        ),
    )


def analysis_index_source(root: Path) -> Path:
    return _require(
        _analysis_dir(root) / "index.html",
        what="static analytics report",
        fix=(
            "Generate it first, e.g.:\n"
            "    python -m momentum_lab.rl.analytics --visual-dir runs/rl/analysis"
        ),
    )


def track_source() -> Path:
    return _require(
        tracks_dir() / f"{TRACK_ID}.json",
        what="Track 1 geometry",
        fix="Build it with: python tools/build_track_01.py",
    )


# --------------------------------------------------------------------------- #
# Copy steps
# --------------------------------------------------------------------------- #
def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def _copy_replay_viewer(root: Path, out: Path) -> list[Path]:
    return [_copy(replay_viewer_source(root), out / "replay_viewer" / "index.html")]


def _copy_analysis(root: Path, out: Path) -> list[Path]:
    src_index = analysis_index_source(root)
    written = [_copy(src_index, out / "analysis" / "index.html")]
    # The analytics report's own SVGs are ``top_runs.svg`` + ``run_*.svg``. Skip
    # leading-underscore files (e.g. a stray ``_overlay.svg`` from an ad-hoc
    # ``--compare-run``): they are internal scratch by the repo's convention and are
    # not referenced by the report, so the export stays reproducible and free of
    # leftover assets.
    for svg in sorted(src_index.parent.glob("*.svg")):
        if svg.name.startswith("_"):
            continue
        written.append(_copy(svg, out / "analysis" / svg.name))
    return written


def _copy_track(out: Path) -> list[Path]:
    return [_copy(track_source(), out / "data" / f"{TRACK_ID}.json")]


def _copy_benchmarks(records: list[AnalyzedRun], out: Path) -> list[Path]:
    written: list[Path] = []
    for bench in BENCHMARKS:
        run = _benchmark_run(records, bench)
        if run.artifact_path is None:
            raise PortfolioExportError(
                f"benchmark {bench.id!r} (run {bench.run_id}) has no replay artifact on disk"
            )
        written.append(_copy(run.artifact_path, out / "data" / bench.data_filename))
    return written


def _benchmark_run(records: list[AnalyzedRun], bench: Benchmark) -> AnalyzedRun:
    try:
        run = find_run_by_id(records, bench.run_id)
    except ValueError as e:
        raise PortfolioExportError(
            f"benchmark {bench.id!r} (run {bench.run_id}) not found in {rl_runs_dir()}: {e}"
        ) from e
    _verify_benchmark(bench, run)
    return run


def _verify_benchmark(bench: Benchmark, run: AnalyzedRun) -> None:
    """Guard that the located artifact still matches the locked benchmark row.

    ``lap_ticks`` is intentionally not checked: it is the plan's fixed sub-tick lap
    length, which differs from the artifact's recorded ``episode_steps`` (the run
    keeps recording for one settle tick past the finish). Lap time, wall hits, and
    validity are the identifying outcomes and must match.
    """
    problems: list[str] = []
    if not run.valid:
        problems.append("run is not a valid lap")
    if abs(run.lap_time - bench.lap_time) > _LAP_TIME_TOLERANCE:
        problems.append(f"lap_time {run.lap_time:.3f}s != expected {bench.lap_time:.3f}s")
    if run.wall_hits != bench.wall_hits:
        problems.append(f"wall_hits {run.wall_hits} != expected {bench.wall_hits}")
    if problems:
        raise PortfolioExportError(
            f"benchmark {bench.id!r} (run {bench.run_id}) no longer matches the plan: "
            + "; ".join(problems)
        )


def _write_manifest(out: Path) -> list[Path]:
    manifest = {
        "track_id": TRACK_ID,
        "physics_version": PHYSICS_VERSION,
        "timing_version": TIMING_VERSION,
        "benchmarks": [bench.manifest_row() for bench in BENCHMARKS],
    }
    path = out / "data" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [path]


def _write_fixtures(records: list[AnalyzedRun], out: Path) -> list[Path]:
    ai_benchmark = next(bench for bench in BENCHMARKS if bench.id == "ghostline_ai")
    run = _benchmark_run(records, ai_benchmark)
    if run.artifact_path is None:
        raise PortfolioExportError(
            f"benchmark {ai_benchmark.id!r} (run {ai_benchmark.run_id}) has no replay artifact on disk"
        )
    try:
        artifact = load_rollout(run.artifact_path)
        return list(write_portfolio_fixtures(out / "fixtures", artifact.replay))
    except (OSError, ValueError, PortfolioFixtureError) as e:
        raise PortfolioExportError(
            f"could not generate GL-19 fixtures from run {ai_benchmark.run_id}: {e}"
        ) from e


# --------------------------------------------------------------------------- #
# Portability audit (GL-03)
# --------------------------------------------------------------------------- #
_REF_RE = re.compile(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_ABS_LOCAL_RE = re.compile(r'^(?:[A-Za-z]:[\\/]|/|\\)')
_EXTERNAL_SCHEMES = ("http://", "https://", "data:", "mailto:", "//")


def _audit_portable(html_path: Path) -> None:
    """Fail loudly if an exported HTML file would 404 or escape its own dir.

    The generators already emit relative references, so this is a regression guard
    rather than a rewriter: it rejects ``file://`` and absolute local paths, and
    confirms every relative asset reference resolves inside the export.
    """
    text = html_path.read_text(encoding="utf-8")
    if "file://" in text:
        raise PortfolioExportError(f"{html_path} contains a file:// reference")
    base = html_path.parent
    for ref in _REF_RE.findall(text):
        lowered = ref.lower()
        if lowered.startswith("#") or lowered.startswith(_EXTERNAL_SCHEMES):
            continue
        if _ABS_LOCAL_RE.match(ref):
            raise PortfolioExportError(
                f"{html_path} references an absolute path {ref!r}; must be relative"
            )
        local = ref.split("?", 1)[0].split("#", 1)[0]
        if not local:
            continue
        if not (base / local).exists():
            raise PortfolioExportError(
                f"{html_path} references {ref!r}, which is not in the export"
            )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def export(out_dir: str | Path, *, root: str | Path | None = None, clean: bool = False) -> ExportResult:
    """Build the portable portfolio tree under ``out_dir`` and return what was written."""
    out = Path(out_dir)
    root_path = Path(root) if root is not None else rl_runs_dir()
    if clean and out.exists():
        shutil.rmtree(out)

    # Resolve the two benchmark artifacts up front so a bad/missing run fails before
    # we write anything.
    records = scan_runs(root_path)

    written: list[Path] = []
    written += _copy_replay_viewer(root_path, out)
    written += _copy_analysis(root_path, out)
    written += _copy_track(out)
    written += _copy_benchmarks(records, out)
    written += _write_manifest(out)
    written += _write_fixtures(records, out)

    _audit_portable(out / "replay_viewer" / "index.html")
    _audit_portable(out / "analysis" / "index.html")

    return ExportResult(out_dir=out, files=tuple(written))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the portable Ghostline portfolio asset tree (GL-02/GL-03/GL-19)."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the export tree.")
    parser.add_argument(
        "--root",
        type=Path,
        default=rl_runs_dir(),
        help="RL runs root to read viewer/analytics/artifacts from (default: runs/rl).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before exporting (for a clean tree).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = export(args.out, root=args.root, clean=args.clean)
    except PortfolioExportError as e:
        parser.error(str(e))

    if not args.quiet:
        print(
            json.dumps(
                {
                    "out": str(result.out_dir),
                    "files": len(result.files),
                    "track_id": TRACK_ID,
                    "physics_version": PHYSICS_VERSION,
                    "timing_version": TIMING_VERSION,
                    "benchmarks": [b.id for b in BENCHMARKS],
                },
                sort_keys=True,
            )
        )
        print(result.tree())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
