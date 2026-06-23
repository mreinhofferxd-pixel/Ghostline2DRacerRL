"""Portfolio export contracts (GL-02 / GL-03).

Hermetic: these build a fake ``runs/rl`` tree and stub the run-location layer so
the export runs without the gitignored RL artifacts. They cover the manifest shape
the challenge plan locks, the portability audit (no ``file://`` / absolute / dead
asset references), the benchmark-match guard, and an end-to-end export into a clean
directory. Nothing here trains, re-evaluates, or mutates sim state.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from momentum_lab import PHYSICS_VERSION, TIMING_VERSION
import momentum_lab.portfolio_export as px
from momentum_lab.portfolio_export import (
    BENCHMARKS,
    TRACK_ID,
    PortfolioExportError,
    _audit_portable,
    _verify_benchmark,
    export,
)

# A self-contained viewer page: no external src/href, like the real replay viewer.
SELF_CONTAINED = "<!doctype html><html><body><canvas></canvas></body></html>"


def _fake_root(root):
    """A minimal stand-in for ``runs/rl`` with the two reused HTML outputs."""
    analysis = root / "analysis"
    (analysis / "replay_viewer").mkdir(parents=True)
    (analysis / "replay_viewer" / "index.html").write_text(SELF_CONTAINED, encoding="utf-8")
    (analysis / "top_runs.svg").write_text("<svg/>", encoding="utf-8")
    (analysis / "run_001_idX.svg").write_text("<svg/>", encoding="utf-8")
    (analysis / "index.html").write_text(
        '<img src="top_runs.svg"><a href="run_001_idX.svg">run svg</a>',
        encoding="utf-8",
    )
    return root


def _fake_run(tmp_path, bench):
    artifact = tmp_path / f"artifact_{bench.run_id}.json"
    artifact.write_text(json.dumps({"kind": "ghostline_rl_rollout"}), encoding="utf-8")
    return SimpleNamespace(
        valid=True,
        lap_time=bench.lap_time,
        wall_hits=bench.wall_hits,
        artifact_path=artifact,
    )


def _stub_run_location(monkeypatch, tmp_path):
    runs = {b.run_id: _fake_run(tmp_path, b) for b in BENCHMARKS}
    monkeypatch.setattr(px, "scan_runs", lambda root: list(runs.values()))
    monkeypatch.setattr(px, "find_run_by_id", lambda records, rid: runs[rid])
    monkeypatch.setattr(px, "load_rollout", lambda path: SimpleNamespace(replay=object()))

    def fake_write_fixtures(out_dir, replay):
        path = out_dir / "straight_accel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": "straight_accel"}), encoding="utf-8")
        return (path,)

    monkeypatch.setattr(px, "write_portfolio_fixtures", fake_write_fixtures)
    return runs


def test_export_produces_full_tree(tmp_path, monkeypatch):
    root = _fake_root(tmp_path / "runs_rl")
    _stub_run_location(monkeypatch, tmp_path)
    out = tmp_path / "out"

    result = export(out, root=root)

    names = {str(p.relative_to(out)).replace("\\", "/") for p in result.files}
    assert "replay_viewer/index.html" in names
    assert "analysis/index.html" in names
    assert "analysis/top_runs.svg" in names
    assert "analysis/run_001_idX.svg" in names
    assert f"data/{TRACK_ID}.json" in names
    assert "data/manifest.json" in names
    assert "fixtures/straight_accel.json" in names
    for bench in BENCHMARKS:
        assert f"data/{bench.data_filename}" in names
    # Every listed file exists on disk.
    for path in result.files:
        assert path.exists()


def test_manifest_matches_plan_shape(tmp_path, monkeypatch):
    root = _fake_root(tmp_path / "runs_rl")
    _stub_run_location(monkeypatch, tmp_path)
    out = tmp_path / "out"

    export(out, root=root)

    manifest = json.loads((out / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["track_id"] == TRACK_ID
    assert manifest["physics_version"] == PHYSICS_VERSION
    assert manifest["timing_version"] == TIMING_VERSION

    expected = [
        ("ghostline_ai", "Ghostline AI", "881229", 3.948, 238, 0),
        ("michi_dev", "Michi/dev", "750982", 4.028, 242, 0),
    ]
    assert len(manifest["benchmarks"]) == len(expected)
    for row, (bid, label, run_id, lap_time, lap_ticks, walls) in zip(manifest["benchmarks"], expected):
        assert set(row) == {"id", "label", "run_id", "lap_time", "lap_ticks", "wall_hits"}
        assert row["id"] == bid
        assert row["label"] == label
        assert row["run_id"] == run_id
        assert row["lap_time"] == lap_time
        assert row["lap_ticks"] == lap_ticks
        assert row["wall_hits"] == walls


def test_audit_rejects_file_scheme(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<img src="file:///C:/assets/x.svg">', encoding="utf-8")
    with pytest.raises(PortfolioExportError):
        _audit_portable(page)


def test_audit_rejects_absolute_paths(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<a href="/assets/x.svg">x</a>', encoding="utf-8")
    with pytest.raises(PortfolioExportError):
        _audit_portable(page)
    page.write_text(r'<img src="C:\assets\x.svg">', encoding="utf-8")
    with pytest.raises(PortfolioExportError):
        _audit_portable(page)


def test_audit_rejects_dead_relative_reference(tmp_path):
    page = tmp_path / "index.html"
    page.write_text('<img src="missing.svg">', encoding="utf-8")
    with pytest.raises(PortfolioExportError):
        _audit_portable(page)


def test_audit_accepts_present_relative_and_external(tmp_path):
    (tmp_path / "a.svg").write_text("<svg/>", encoding="utf-8")
    page = tmp_path / "index.html"
    page.write_text(
        '<img src="a.svg"><a href="https://example.com/y">e</a><a href="#top">t</a>',
        encoding="utf-8",
    )
    _audit_portable(page)  # must not raise


def test_verify_benchmark_passes_on_match():
    bench = BENCHMARKS[0]
    run = SimpleNamespace(
        valid=True, lap_time=bench.lap_time, wall_hits=bench.wall_hits, artifact_path=None
    )
    _verify_benchmark(bench, run)  # must not raise


def test_verify_benchmark_raises_on_outcome_mismatch():
    bench = BENCHMARKS[0]
    slow = SimpleNamespace(
        valid=True, lap_time=bench.lap_time + 0.5, wall_hits=bench.wall_hits, artifact_path=None
    )
    with pytest.raises(PortfolioExportError):
        _verify_benchmark(bench, slow)
    walls = SimpleNamespace(
        valid=True, lap_time=bench.lap_time, wall_hits=bench.wall_hits + 1, artifact_path=None
    )
    with pytest.raises(PortfolioExportError):
        _verify_benchmark(bench, walls)
    invalid = SimpleNamespace(
        valid=False, lap_time=bench.lap_time, wall_hits=bench.wall_hits, artifact_path=None
    )
    with pytest.raises(PortfolioExportError):
        _verify_benchmark(bench, invalid)


def test_missing_replay_viewer_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(px, "scan_runs", lambda root: [])
    with pytest.raises(PortfolioExportError):
        export(tmp_path / "out", root=tmp_path / "empty_root")
