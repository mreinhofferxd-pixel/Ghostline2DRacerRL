"""B7.6 interactive RL replay and story viewer.

Read-only browser viewer built on top of the static B7.4a analytics
(:mod:`momentum_lab.rl.analytics`). It turns the *already saved* rollout
artifacts (canonical ``Action`` stream + derived frame cache) into an interactive
``index.html``: overlay/scrub multiple rollouts, read model decisions over time
(throttle/brake/steer/drift), and step through a curated "the model got better"
story.

Like the analytics module it does not train, re-evaluate, or mutate sim state, and
it never introduces a second trajectory format -- it consumes the canonical
artifacts that ``analytics.scan_runs`` already loads. The browser output is a single
self-contained file (the view-model JSON is embedded), so it opens straight off
``file://`` and is easy to embed later in a portfolio page.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import CONTROL_HZ, WORLD_HEIGHT, WORLD_WIDTH
from ..core.checkpoints import Gate
from ..core.track import Track
from ..tracks import load_track_by_id
from .analytics import AnalyzedRun, dedupe_runs, run_trace_stats, scan_runs
from .rollout import rl_runs_dir

VIEW_SCHEMA_VERSION = 1

# Same accent palette as the static overview SVG, so a run keeps a recognizable
# color between the markdown/SVG report and the interactive viewer.
PALETTE = (
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
# The reference run (usually the human best) gets a fixed, high-contrast color.
REFERENCE_COLOR = "#f2f4f8"


# --------------------------------------------------------------------------- #
# Story presets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StoryStep:
    run_id: str
    caption: str


@dataclass(frozen=True)
class StoryPreset:
    name: str
    title: str
    steps: tuple[StoryStep, ...]


def build_progression_story(records: Iterable[AnalyzedRun]) -> StoryPreset:
    """Build the default "model got better" story from whatever runs exist.

    Steps are chosen by *rule* (not a hand-edited id list) so the story keeps
    working as the portfolio grows: an early stall -> the first clean lap -> a
    mid time-attack lap -> the current best, plus a human reference if present.
    Any step with no candidate is skipped.
    """
    visual = [r for r in dedupe_runs(records) if _has_frames(r)]
    # The human run is the *reference* line, not part of the learning story, so it
    # never gets picked as the model's "current best" even when it is faster.
    human = next((r for r in visual if _is_human(r)), None)
    model_runs = [r for r in visual if not _is_human(r)]
    # Rank by lap time, then prefer the cleaner lap on ties -- the same ordering
    # keep-best uses to pick a checkpoint (``_policy_eval_is_better``), so a clean
    # lap is never shown as "Current best" below an equally fast one with wall hits.
    valid = sorted(
        (r for r in model_runs if r.valid and r.lap_time > 0.0),
        key=lambda r: (r.lap_time, r.wall_hits),
    )
    steps: list[StoryStep] = []
    used: set[str] = set()

    def add(run: AnalyzedRun | None, caption: str) -> None:
        if run is None or run.run_id in used:
            return
        used.add(run.run_id)
        steps.append(StoryStep(run_id=run.run_id, caption=caption))

    nonvalid = [r for r in model_runs if not r.valid]
    if nonvalid:
        early = min(
            nonvalid,
            key=lambda r: (
                r.checkpoint_index,
                r.stage_progress if r.stage_progress is not None else -1.0,
                -r.episode_steps,
            ),
        )
        add(early, "Early policy: stalls out before completing the lap.")

    if valid:
        first_lap = valid[-1]  # slowest valid lap = the first competent finish
        add(first_lap, f"First clean lap: {first_lap.lap_time:.3f}s.")
        if len(valid) >= 3:
            mid = valid[len(valid) // 2]
            add(mid, f"Time-attack fine-tuning: {mid.lap_time:.3f}s.")
        best = valid[0]
        add(best, f"Current best: {best.lap_time:.3f}s, {best.wall_hits} wall hits.")

    if human is not None:
        cap = (
            f"Human reference: {human.lap_time:.3f}s."
            if human.valid
            else "Human reference line."
        )
        add(human, cap)

    return StoryPreset(
        name="progression",
        title="How the policy learned Track 1",
        steps=tuple(steps),
    )


def build_progression_story(records: Iterable[AnalyzedRun]) -> StoryPreset:
    """Build a portfolio-sized "how the model learned" story from saved runs.

    The milestones are rule-based rather than hand-picked ids: early failure, first
    finish, reward/selection improvements, observation and gamma pivots, the human
    benchmark, the last 3.965s plateau, and the current best.
    """
    visual = [r for r in dedupe_runs(records) if _has_frames(r)]
    human = next((r for r in visual if _is_human(r)), None)
    model_runs = [r for r in visual if not _is_human(r)]
    valid = sorted(
        (r for r in model_runs if r.valid and r.lap_time > 0.0),
        key=lambda r: (r.lap_time, r.wall_hits, -r.total_reward),
    )
    best = valid[0] if valid else None
    steps: list[StoryStep] = []
    used: set[str] = set()

    def add(run: AnalyzedRun | None, caption: str) -> None:
        if run is None or run.run_id in used:
            return
        used.add(run.run_id)
        steps.append(StoryStep(run_id=run.run_id, caption=caption))

    nonvalid = [r for r in model_runs if not r.valid]
    if nonvalid:
        early = min(
            nonvalid,
            key=lambda r: (
                r.checkpoint_index,
                r.stage_progress if r.stage_progress is not None else -1.0,
                -r.episode_steps,
            ),
        )
        add(early, "Early policy: it understands motion, then stalls before the lap exists.")

    if valid:
        first_lap = valid[-1]
        add(first_lap, f"First finish: the agent completes Track 1 in {first_lap.lap_time:.3f}s.")
        add(
            _slowest_below(valid, 5.25, exclude=used | _best_id(best)),
            "Time attack begins: warm-starting turns a finisher into a fast lap.",
        )
        add(
            _slowest_below(valid, 4.85, exclude=used | _best_id(best)),
            "Direct finish-time reward: the objective starts matching the stopwatch.",
        )
        add(
            _slowest_below(valid, 4.60, exclude=used | _best_id(best)),
            "Keep-best plus racing efficiency: the policy stops relaxing off its sharpest line.",
        )
        add(
            _best_matching(
                valid,
                lambda r: "timeattack_v5" in r.model and r.lap_time <= 4.12,
                exclude=used | _best_id(best),
            ),
            "V5 plateau: a clean 4.1s lap exposes the remaining apex-speed problem.",
        )
        add(
            _best_matching(
                valid,
                lambda r: "lookahead" in r.model
                and "g997" not in r.model
                and "wallsensors" not in r.model,
                exclude=used | _best_id(best),
            ),
            "Lookahead reset: richer observations make fresh lap learning reliable again.",
        )
        add(
            _best_matching(
                valid,
                lambda r: "lookahead_g997" in r.model and "wallsensors" not in r.model,
                exclude=used | _best_id(best),
            ),
            "Gamma 0.997: longer credit assignment breaks the old 246-tick ceiling.",
        )
        add(
            _best_matching(
                valid,
                lambda r: "wallsensors_g997" in r.model and r.lap_time <= 3.97,
                exclude=used | _best_id(best),
            ),
            "Final stretch phase: wall sensors reach 3.965s and set up the last chase.",
        )

    if human is not None:
        caption = (
            _human_caption(human)
            if human.valid
            else "Human reference line."
        )
        if best is None or human.lap_time <= best.lap_time:
            if best is not None:
                add(best, _best_caption(best))
            add(human, caption)
        else:
            add(human, caption)
            add(best, _best_caption(best))
    elif best is not None:
        add(best, _best_caption(best))

    return StoryPreset(
        name="progression",
        title="Ghostline: from wall-stalls to a sub-4s lap",
        steps=tuple(steps),
    )


def _best_id(run: AnalyzedRun | None) -> set[str]:
    return set() if run is None else {run.run_id}


def _best_caption(run: AnalyzedRun) -> str:
    if "wallsensors" in run.model:
        return (
            f"Current best, final stretch: wall-sensor policy reaches "
            f"{run.lap_time:.3f}s in {run.episode_steps} ticks with "
            f"{run.wall_hits} wall hits."
        )
    return (
        f"Current best, final stretch: {run.lap_time:.3f}s in "
        f"{run.episode_steps} ticks with {run.wall_hits} wall hits."
    )


def _human_caption(run: AnalyzedRun) -> str:
    label = "DER GOAT" if run.model == "DER GOAT" else "Michi"
    return (
        f"Michi reference: {label} current best at {run.lap_time:.3f}s. "
        "The AI is chasing this line."
    )


def _slowest_below(
    runs: Iterable[AnalyzedRun],
    ceiling: float,
    *,
    exclude: set[str],
) -> AnalyzedRun | None:
    candidates = [
        run
        for run in runs
        if run.run_id not in exclude and 0.0 < run.lap_time <= ceiling
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda run: (run.lap_time, -run.wall_hits, run.total_reward))


def _best_matching(
    runs: Iterable[AnalyzedRun],
    predicate,
    *,
    exclude: set[str],
) -> AnalyzedRun | None:
    candidates = [run for run in runs if run.run_id not in exclude and predicate(run)]
    if not candidates:
        return None
    return min(candidates, key=lambda run: (run.lap_time, run.wall_hits, -run.total_reward))


# --------------------------------------------------------------------------- #
# Selection layer
# --------------------------------------------------------------------------- #
def select_runs(
    records: Iterable[AnalyzedRun],
    *,
    top_n: int | None = None,
    run_ids: Iterable[str] = (),
    artifact_paths: Iterable[str] = (),
    models: Iterable[str] = (),
    reward_versions: Iterable[str] = (),
    policies: Iterable[str] = (),
    statuses: Iterable[str] = (),
    include_summary_only: bool = False,
) -> list[AnalyzedRun]:
    """Select runs for the viewer from scanned analytics records.

    Runs are de-duplicated first (the old deterministic-eval triplicates collapse
    to one). Criteria AND across types and OR within a list. ``models`` matches by
    substring (so a version prefix like ``b7_1`` selects the whole family, which is
    also how a stochastic eval batch -- all rollouts of one model -- is selected).
    ``run_ids`` selects an explicit, order-preserving set. Unless
    ``include_summary_only`` is set, only runs with cached frames (visualizable)
    are returned.
    """
    run_ids = tuple(str(r) for r in run_ids)
    artifact_paths = tuple(str(p) for p in artifact_paths)
    models = tuple(models)
    reward_versions = tuple(reward_versions)
    policies = tuple(policies)
    statuses = tuple(statuses)

    pool = dedupe_runs(records)
    if not include_summary_only:
        pool = [r for r in pool if _has_frames(r)]

    def matches(run: AnalyzedRun) -> bool:
        if reward_versions and run.reward_version not in reward_versions:
            return False
        if policies and run.policy not in policies:
            return False
        if statuses and run.status not in statuses:
            return False
        if models and not any(m in run.model for m in models):
            return False
        if artifact_paths:
            path_text = str(run.artifact_path) if run.artifact_path else ""
            if not any(p in path_text for p in artifact_paths):
                return False
        return True

    filtered = [run for run in pool if matches(run)]

    if run_ids:
        by_id = {run.run_id: run for run in filtered}
        return [by_id[rid] for rid in run_ids if rid in by_id]
    if top_n is not None:
        ranked = sorted(filtered, key=lambda run: run.sort_key, reverse=True)
        return ranked[: max(0, top_n)]
    return filtered


# --------------------------------------------------------------------------- #
# View-model
# --------------------------------------------------------------------------- #
def build_view_model(
    runs: Iterable[AnalyzedRun],
    *,
    reference_id: str | None = None,
    story: StoryPreset | None = None,
    max_frames: int | None = None,
) -> dict:
    """Build the JSON-serializable view-model the browser viewer consumes."""
    runs = list(runs)
    if not runs:
        raise ValueError("build_view_model: no runs selected")

    track_id = next((r.track_id for r in runs if r.track_id), runs[0].track_id)
    track = load_track_by_id(track_id)

    run_views: list[dict] = []
    palette_i = 0
    for run in runs:
        is_reference = reference_id is not None and run.run_id == reference_id
        if is_reference:
            color = REFERENCE_COLOR
        else:
            color = PALETTE[palette_i % len(PALETTE)]
            palette_i += 1
        run_views.append(
            _run_view(run, color=color, is_reference=is_reference, max_frames=max_frames)
        )

    story_payload = None
    if story is not None:
        present = {rv["run_id"] for rv in run_views}
        steps = [
            {"run_id": step.run_id, "caption": step.caption}
            for step in story.steps
            if step.run_id in present
        ]
        story_payload = {"name": story.name, "title": story.title, "steps": steps}
        captions = {step.run_id: step.caption for step in story.steps}
        for rv in run_views:
            if not rv["caption"] and rv["run_id"] in captions:
                rv["caption"] = captions[rv["run_id"]]

    return {
        "schema_version": VIEW_SCHEMA_VERSION,
        "control_hz": CONTROL_HZ,
        "track": _track_view(track),
        "runs": run_views,
        "story": story_payload,
        "reference_id": reference_id,
    }


def _run_view(
    run: AnalyzedRun,
    *,
    color: str,
    is_reference: bool,
    max_frames: int | None,
) -> dict:
    frames = run.artifact.replay.frames if run.artifact is not None else ()
    actions = run.artifact.replay.actions if run.artifact is not None else ()
    idxs = _sample_indices(len(frames), max_frames)

    frame_cols = _frame_columns(frames, idxs)
    action_cols = _action_columns(actions, idxs)

    sectors: list[dict] = []
    path_distance = 0.0
    if frames:
        stats = run_trace_stats(run)
        path_distance = round(stats.path_distance, 1)
        sectors = [
            {
                "label": s.label,
                "split_time": round(s.split_time, 3),
                "path_distance": round(s.path_distance, 1),
                "avg_speed": round(s.avg_speed, 1),
                "min_speed": round(s.min_speed, 1),
                "max_speed": round(s.max_speed, 1),
                "drift_time": round(s.drift_time, 3),
                "wall_frames": s.wall_frames,
            }
            for s in stats.sectors
        ]

    return {
        "run_id": run.run_id,
        "model": run.model,
        "policy": run.policy,
        "status": run.status,
        "reward_version": run.reward_version,
        "seed": run.seed,
        "valid": run.valid,
        "lap_time": round(run.lap_time, 3),
        "wall_hits": run.wall_hits,
        "boosts_used": run.boosts_used,
        "drift_time": round(run.drift_time, 3),
        "total_reward": round(run.total_reward, 2),
        "path_distance": path_distance,
        "checkpoint_index": run.checkpoint_index,
        "checkpoint_count": run.checkpoint_count,
        "duplicate_count": run.duplicate_count,
        "color": color,
        "caption": "",
        "is_reference": is_reference,
        "frames": frame_cols,
        "actions": action_cols,
        "sectors": sectors,
    }


def _frame_columns(frames, idxs: list[int]) -> dict:
    return {
        "t": [round(frames[i].t, 3) for i in idxs],
        "x": [round(frames[i].x, 2) for i in idxs],
        "y": [round(frames[i].y, 2) for i in idxs],
        "angle": [round(frames[i].angle, 4) for i in idxs],
        "speed": [round(frames[i].speed, 2) for i in idxs],
        "cp": [frames[i].cp for i in idxs],
        "drift": [int(frames[i].drift) for i in idxs],
        "wall": [int(frames[i].wall) for i in idxs],
        "boost": [int(frames[i].boost) for i in idxs],
    }


def _action_columns(actions, idxs: list[int]) -> dict:
    n = len(actions)
    valid = [i for i in idxs if i < n]
    return {
        "throttle": [round(actions[i].throttle, 3) for i in valid],
        "brake": [round(actions[i].brake, 3) for i in valid],
        "steer": [round(actions[i].steer, 3) for i in valid],
        "drift": [int(actions[i].drift) for i in valid],
    }


def _sample_indices(n: int, max_frames: int | None) -> list[int]:
    """Index set for optional downsampling; always keeps the final frame."""
    if n <= 0:
        return []
    if max_frames is None or max_frames < 2 or n <= max_frames:
        return list(range(n))
    step = math.ceil(n / max_frames)
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def _track_view(track: Track) -> dict:
    return {
        "track_id": track.track_id,
        "world": [WORLD_WIDTH, WORLD_HEIGHT],
        "surface_outer": [[x, y] for x, y in track.surface_outer],
        "surface_inner": [[x, y] for x, y in track.surface_inner],
        "racing_line": [[x, y] for x, y in track.racing_line],
        "walls": [[w.x1, w.y1, w.x2, w.y2] for w in track.walls],
        "boost_pads": [list(pad.bounds) for pad in track.boost_pads],
        "checkpoints": [
            _gate_view(gate, str(i)) for i, gate in enumerate(track.checkpoints, start=1)
        ],
        "finish": _gate_view(track.finish, "F") if track.finish is not None else None,
    }


def _gate_view(gate: Gate, label: str) -> dict:
    cx, cy = gate.center
    return {
        "x1": gate.x1,
        "y1": gate.y1,
        "x2": gate.x2,
        "y2": gate.y2,
        "cx": cx,
        "cy": cy,
        "label": label,
    }


def _has_frames(run: AnalyzedRun) -> bool:
    return run.artifact is not None and bool(run.artifact.replay.frames)


def _is_human(run: AnalyzedRun) -> bool:
    return run.policy == "human_keyboard" or run.model == "Michi"


# --------------------------------------------------------------------------- #
# Viewer writer
# --------------------------------------------------------------------------- #
def viewer_dir(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else rl_runs_dir()
    return base / "analysis" / "replay_viewer"


def write_replay_viewer(view_model: dict, out_dir: str | Path | None = None) -> Path:
    """Write a single self-contained ``index.html`` and return its path."""
    out_path = Path(out_dir) if out_dir is not None else viewer_dir()
    out_path.mkdir(parents=True, exist_ok=True)
    index_path = out_path / "index.html"
    index_path.write_text(render_viewer_html(view_model), encoding="utf-8")
    return index_path


def render_viewer_html(view_model: dict) -> str:
    # Escape ``<`` inside the JSON so an embedded string can never close the
    # <script> element; "<" parses back to "<" in the browser.
    data_json = json.dumps(view_model, separators=(",", ":")).replace("<", "\\u003c")
    return _VIEWER_TEMPLATE.replace("__GHOSTLINE_DATA__", data_json)


# --------------------------------------------------------------------------- #
# On-demand re-evaluation (B7.6b)
# --------------------------------------------------------------------------- #
def sb3_available() -> bool:
    return importlib.util.find_spec("stable_baselines3") is not None


def reevaluate_models(
    model_paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
    eval_seed: int = 10_000,
) -> list[str]:
    """Materialize a standard rollout artifact for each saved model, then return
    the model stems so the caller can include them in the selection.

    For a model file that has no saved rollout artifact yet, this loads the model,
    runs one deterministic eval rollout (reusing ``train.evaluate_model``), and
    writes the artifact + a summary row under ``<root>/evals/<stem>`` so the next
    ``scan_runs`` picks it up exactly like any other eval. ``train`` rebuilds the
    matching observation/action/reward env config from the model's manifest (or, if
    there is none, from the saved policy's action space) so ``predict`` stays valid.

    Requires Stable-Baselines3; callers should gate on :func:`sb3_available`.
    """
    from .train import evaluate_model, infer_env_config_for_model

    base = Path(root) if root is not None else rl_runs_dir()
    stems: list[str] = []
    for raw in model_paths:
        model_path = Path(raw)
        if not model_path.exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        stem = model_path.stem
        evaluate_model(
            model_path,
            episodes=1,
            seed=eval_seed,
            env_config=infer_env_config_for_model(model_path),
            deterministic=True,
            output_dir=base / "evals" / stem,
            summary_path=base / "evals" / f"{stem}.jsonl",
        )
        stems.append(stem)
    return stems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _split_csv(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the interactive Ghostline RL replay/story viewer."
    )
    parser.add_argument("--root", type=Path, default=rl_runs_dir())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top", type=int, default=None, help="Select the best N runs.")
    parser.add_argument("--run-id", action="append", default=[], help="Stable run id (repeatable / CSV).")
    parser.add_argument("--model", action="append", default=[], help="Model name substring (repeatable / CSV).")
    parser.add_argument("--reward-version", action="append", default=[])
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--status", action="append", default=[])
    parser.add_argument("--story", default=None, help="Curated story preset (e.g. 'progression').")
    parser.add_argument("--reference", default=None, help="Run id to mark as the reference line.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional per-run frame downsample cap.")
    parser.add_argument(
        "--reevaluate",
        action="append",
        default=[],
        help="Saved model .zip to materialize a rollout for before scanning "
        "(repeatable / CSV). Requires stable-baselines3.",
    )
    parser.add_argument("--include-summary-only", action="store_true")
    parser.add_argument("--no-print", action="store_true")
    args = parser.parse_args(argv)

    run_ids = _split_csv(args.run_id)
    models = _split_csv(args.model)
    reward_versions = _split_csv(args.reward_version)
    policies = _split_csv(args.policy)
    statuses = _split_csv(args.status)
    reeval_paths = _split_csv(args.reevaluate)

    if args.story is not None and args.story != "progression":
        parser.error(f"unknown story preset {args.story!r}; available: progression")

    reeval_stems: list[str] = []
    if reeval_paths:
        if not sb3_available():
            parser.error(
                "--reevaluate requires stable-baselines3; install with "
                "`python -m pip install -e .[train]`."
            )
        try:
            reeval_stems = reevaluate_models(reeval_paths, root=args.root)
        except FileNotFoundError as e:
            parser.error(str(e))

    records = scan_runs(args.root)
    if not records:
        parser.error(f"no RL runs found under {args.root}")

    reeval_ids = _reeval_run_ids(records, reeval_stems)
    story = build_progression_story(records) if args.story == "progression" else None
    explicit = bool(
        args.top is not None
        or run_ids
        or models
        or reward_versions
        or policies
        or statuses
    )

    if explicit:
        selected = select_runs(
            records,
            top_n=args.top,
            run_ids=run_ids,
            models=models,
            reward_versions=reward_versions,
            policies=policies,
            statuses=statuses,
            include_summary_only=args.include_summary_only,
        )
    elif story is not None:
        selected = select_runs(
            records,
            run_ids=tuple(step.run_id for step in story.steps),
            include_summary_only=args.include_summary_only,
        )
    elif reeval_ids:
        # Only re-evaluation was requested: show just those materialized runs.
        selected = select_runs(records, run_ids=tuple(reeval_ids), include_summary_only=args.include_summary_only)
    else:
        selected = select_runs(records, top_n=12, include_summary_only=args.include_summary_only)

    selected = _ensure_included(
        records, selected, story=story, reference=args.reference, extra_ids=reeval_ids
    )
    if not selected:
        parser.error("selection matched no runs")

    view_model = build_view_model(
        selected,
        reference_id=args.reference,
        story=story,
        max_frames=args.max_frames,
    )
    index_path = write_replay_viewer(view_model, args.out_dir or viewer_dir(args.root))
    if not args.no_print:
        print(
            json.dumps(
                {
                    "viewer": str(index_path),
                    "runs": len(view_model["runs"]),
                    "story": None if story is None else story.name,
                    "reference_id": args.reference,
                    "reevaluated": reeval_stems,
                },
                sort_keys=True,
            )
        )
    return 0


def _ensure_included(
    records: Iterable[AnalyzedRun],
    selected: list[AnalyzedRun],
    *,
    story: StoryPreset | None,
    reference: str | None,
    extra_ids: Iterable[str] = (),
) -> list[AnalyzedRun]:
    """Union in story-step runs, the reference run, and any forced ids.

    ``extra_ids`` (e.g. freshly re-evaluated models) are always included even when
    the base selection would have left them out.
    """
    by_id = {run.run_id: run for run in dedupe_runs(records)}
    out: list[AnalyzedRun] = []
    seen: set[str] = set()

    def push(run: AnalyzedRun | None) -> None:
        if run is None or run.run_id in seen:
            return
        seen.add(run.run_id)
        out.append(run)

    if story is not None:
        for step in story.steps:
            push(by_id.get(step.run_id))
    for run in selected:
        push(run)
    for rid in extra_ids:
        push(by_id.get(rid))
    if reference is not None:
        push(by_id.get(reference))
    return out


def _reeval_run_ids(records: Iterable[AnalyzedRun], model_stems: Iterable[str]) -> list[str]:
    """Run ids of visualizable runs materialized by ``--reevaluate`` (by model stem)."""
    stems = set(model_stems)
    ids: list[str] = []
    for run in dedupe_runs(records):
        if run.model in stems and _has_frames(run) and run.run_id not in ids:
            ids.append(run.run_id)
    return ids


# --------------------------------------------------------------------------- #
# Browser viewer template (single self-contained file; vanilla JS + canvas)
# --------------------------------------------------------------------------- #
_VIEWER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ghostline RL Replay Viewer</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: #111318; color: #f2f4f8; }
h1 { font-size: 18px; margin: 0; }
a { color: #7dd3fc; }
header { display: flex; align-items: center; gap: 16px; padding: 12px 18px; border-bottom: 1px solid #2a313b; flex-wrap: wrap; }
#story-bar { display: none; align-items: center; gap: 10px; margin-left: auto; }
#story-bar.on { display: flex; }
#story-caption { font-size: 14px; color: #d4d8e1; max-width: 520px; }
button { background: #1d2530; color: #f2f4f8; border: 1px solid #38404a; border-radius: 5px; padding: 5px 10px; cursor: pointer; font-size: 13px; }
button:hover { border-color: #7dd3fc; }
button.active { background: #244; border-color: #17bebb; }
main { display: flex; gap: 14px; padding: 14px 18px; align-items: flex-start; flex-wrap: wrap; }
#stage { flex: 1 1 620px; min-width: 360px; }
canvas#track { width: 100%; height: auto; background: #181a1e; border: 1px solid #2a313b; border-radius: 6px; display: block; }
#controls { flex: 0 0 320px; display: flex; flex-direction: column; gap: 12px; }
.panel { background: #161a20; border: 1px solid #2a313b; border-radius: 6px; padding: 10px 12px; }
.panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #8a93a1; margin: 0 0 8px; }
.row { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; }
.row label { color: #b5bdc9; }
input[type=range] { flex: 1; }
.layers { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 13px; }
.layers label { display: flex; align-items: center; gap: 5px; color: #d4d8e1; }
#runlist { max-height: 230px; overflow: auto; }
.runrow { display: flex; align-items: center; gap: 7px; padding: 4px 2px; font-size: 12px; border-bottom: 1px solid #20262f; }
.runrow .sw { width: 12px; height: 12px; border-radius: 3px; flex: 0 0 auto; border: 1px solid #00000066; }
.runrow .name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: pointer; }
.runrow.focused .name { color: #fff; font-weight: bold; }
.runrow .meta { color: #8a93a1; flex: 0 0 auto; }
.badge { font-size: 10px; padding: 1px 5px; border-radius: 8px; background: #233; color: #9fe; }
.badge.ref { background: #443; color: #fe9; }
.hint { font-size: 12px; color: #6f7886; }
</style>
</head>
<body>
<script type="application/json" id="ghostline-data">__GHOSTLINE_DATA__</script>
<header>
  <h1>Ghostline RL Replay Viewer</h1>
  <span class="hint" id="subtitle"></span>
  <div id="story-bar">
    <button id="story-prev">&#9664; Prev</button>
    <span id="story-caption"></span>
    <button id="story-next">Next &#9654;</button>
  </div>
</header>
<main>
  <div id="stage">
    <canvas id="track" width="100" height="100"></canvas>
  </div>
  <div id="controls">
    <div class="panel">
      <h2>Playback</h2>
      <div class="row">
        <button id="play">&#9654; Play</button>
        <button id="restart">&#8634;</button>
        <label>Speed</label>
        <select id="speed">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
      <div class="row"><label>Time</label><input type="range" id="scrub" min="0" max="1000" value="0"></div>
      <div class="row"><span class="hint" id="clock">0.00s</span></div>
      <div class="row">
        <label>Align</label>
        <select id="align">
          <option value="race">Race time</option>
          <option value="finish">Finish-normalized</option>
          <option value="stage">Checkpoint stage</option>
        </select>
      </div>
      <div class="row"><label>Trail</label><input type="range" id="trail" min="0" max="100" value="40"></div>
    </div>
    <div class="panel">
      <h2>Layers</h2>
      <div class="layers" id="layers">
        <label><input type="checkbox" data-layer="drift" checked> Drift</label>
        <label><input type="checkbox" data-layer="boost" checked> Boost</label>
        <label><input type="checkbox" data-layer="wall" checked> Wall</label>
        <label><input type="checkbox" data-layer="checkpoints" checked> Checkpoints</label>
        <label><input type="checkbox" data-layer="finalPos" checked> Final positions</label>
        <label><input type="checkbox" data-layer="labels" checked> Labels</label>
        <label><input type="checkbox" data-layer="racingLine"> Racing line</label>
      </div>
    </div>
    <div class="panel">
      <h2>Runs <span class="hint">(click a name to focus)</span></h2>
      <div id="runlist"></div>
    </div>
  </div>
</main>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("ghostline-data").textContent);
const TRACK = DATA.track;
const RUNS = DATA.runs;
const HZ = DATA.control_hz || 60;
const byId = {};
RUNS.forEach(r => { byId[r.run_id] = r; });

const state = {
  axis: 0,
  playing: false,
  speed: 1,
  align: "race",
  trailFrames: Math.round(0.40 * Math.max(...RUNS.map(r => r.frames.t.length || 1))),
  layers: { drift: true, boost: true, wall: true, checkpoints: true, finalPos: true, labels: true, racingLine: false },
  focus: null,
  runUI: {},
};
RUNS.forEach((r, i) => {
  state.runUI[r.run_id] = { visible: true, opacity: r.is_reference ? 0.85 : 0.9 };
});
state.focus = (RUNS.find(r => !r.is_reference) || RUNS[0] || {}).run_id || null;
const referenceId = DATA.reference_id;

// ---- cp-entry indices for stage alignment ---------------------------------
function cpEntries(run) {
  if (run._cp) return run._cp;
  const cp = run.frames.cp, n = cp.length;
  const count = run.checkpoint_count || 1;
  const entries = new Array(count + 1).fill(n > 0 ? n - 1 : 0);
  entries[0] = 0;
  for (let k = 1; k <= count; k++) {
    let idx = n > 0 ? n - 1 : 0;
    for (let i = 0; i < n; i++) { if (cp[i] >= k) { idx = i; break; } }
    entries[k] = idx;
  }
  run._cp = entries;
  return entries;
}

// ---- axis range -----------------------------------------------------------
function axisMax() {
  if (state.align === "finish") return 1;
  if (state.align === "stage") return Math.max(1, ...RUNS.map(r => r.checkpoint_count || 1));
  return Math.max(0.001, ...RUNS.map(r => r.frames.t.length ? r.frames.t[r.frames.t.length - 1] : 0));
}
function fullAxisSeconds() {
  if (state.align === "race") return axisMax();   // real-time playback
  return 6;
}

// ---- per-run fractional frame index for the current axis value -------------
function frameAt(run, axis) {
  const n = run.frames.t.length;
  if (n === 0) return null;
  if (state.align === "finish") return clamp(axis, 0, 1) * (n - 1);
  if (state.align === "stage") {
    const e = cpEntries(run);
    const count = run.checkpoint_count || 1;
    const s = clamp(axis, 0, count);
    const k = Math.min(count - 1, Math.floor(s));
    const lo = e[k], hi = e[k + 1];
    return lo + (hi - lo) * (s - k);
  }
  const t = run.frames.t;
  if (axis <= t[0]) return 0;
  if (axis >= t[n - 1]) return n - 1;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (t[mid] <= axis) lo = mid; else hi = mid; }
  const span = t[hi] - t[lo];
  return span > 1e-9 ? lo + (axis - t[lo]) / span : lo;
}
function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function lerp(a, b, f) { return a + (b - a) * f; }

function pose(run, fi) {
  if (fi == null) return null;
  const f = run.frames, n = f.t.length;
  const lo = Math.floor(fi), hi = Math.min(lo + 1, n - 1), fr = fi - lo, d = Math.round(fi);
  return {
    idx: clamp(d, 0, n - 1),
    x: lerp(f.x[lo], f.x[hi], fr),
    y: lerp(f.y[lo], f.y[hi], fr),
    angle: f.angle[lo],
    speed: lerp(f.speed[lo], f.speed[hi], fr),
    t: lerp(f.t[lo], f.t[hi], fr),
    cp: f.cp[clamp(d, 0, n - 1)],
    drift: f.drift[clamp(d, 0, n - 1)],
    wall: f.wall[clamp(d, 0, n - 1)],
    boost: f.boost[clamp(d, 0, n - 1)],
  };
}

// ---- canvas setup ----------------------------------------------------------
const cv = document.getElementById("track");
cv.width = TRACK.world[0];
cv.height = TRACK.world[1];
const ctx = cv.getContext("2d");

function drawTrack() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#181a1e";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (TRACK.surface_outer.length) fillPoly(TRACK.surface_outer, "#34383f");
  if (TRACK.surface_inner.length) fillPoly(TRACK.surface_inner, "#1d3027");
  if (state.layers.racingLine && TRACK.racing_line.length) {
    strokePts(TRACK.racing_line, "#ffd16688", 2, true);
  }
  ctx.lineCap = "round";
  ctx.strokeStyle = "#b5bdc9a8";
  ctx.lineWidth = 3;
  TRACK.walls.forEach(w => { ctx.beginPath(); ctx.moveTo(w[0], w[1]); ctx.lineTo(w[2], w[3]); ctx.stroke(); });
  TRACK.boost_pads.forEach(p => {
    ctx.fillStyle = "#32d58338"; ctx.strokeStyle = "#32d583"; ctx.lineWidth = 2;
    ctx.fillRect(p[0], p[1], p[2] - p[0], p[3] - p[1]);
    ctx.strokeRect(p[0], p[1], p[2] - p[0], p[3] - p[1]);
  });
  if (state.layers.checkpoints) {
    TRACK.checkpoints.forEach(g => drawGate(g, "#ffd166"));
    if (TRACK.finish) drawGate(TRACK.finish, "#f2f4f8");
  }
}
function fillPoly(pts, color) {
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
}
function strokePts(pts, color, w, dashed) {
  ctx.save(); if (dashed) ctx.setLineDash([10, 8]);
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.stroke(); ctx.restore();
}
function drawGate(g, color) {
  ctx.save(); ctx.setLineDash([10, 7]); ctx.lineWidth = 4; ctx.strokeStyle = color;
  ctx.beginPath(); ctx.moveTo(g.x1, g.y1); ctx.lineTo(g.x2, g.y2); ctx.stroke(); ctx.restore();
  if (state.layers.labels) {
    ctx.fillStyle = color; ctx.font = "18px Arial"; ctx.fillText(g.label, g.cx + 6, g.cy - 6);
  }
}

function trailWindow(run, idx) {
  const start = Math.max(0, idx - state.trailFrames);
  return [start, idx];
}
function drawRunTrail(run, idx, opacity) {
  const f = run.frames;
  const [a, b] = trailWindow(run, idx);
  if (b <= a) return;
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = run.color; ctx.lineWidth = 3.2; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(f.x[a], f.y[a]);
  for (let i = a + 1; i <= b; i++) ctx.lineTo(f.x[i], f.y[i]);
  ctx.stroke();
  overlay(run, a, b, "drift", "#4d96ff", 5);
  overlay(run, a, b, "boost", "#32d583", 6);
  overlay(run, a, b, "wall", "#ff453a", 7);
  ctx.globalAlpha = 1;
}
function overlay(run, a, b, key, color, w) {
  if (!state.layers[key]) return;
  const f = run.frames, flag = f[key];
  ctx.strokeStyle = color; ctx.lineWidth = w;
  let i = a;
  while (i <= b) {
    if (!flag[i]) { i++; continue; }
    let j = i;
    ctx.beginPath(); ctx.moveTo(f.x[Math.max(a, i - 1)], f.y[Math.max(a, i - 1)]);
    while (j <= b && flag[j]) { ctx.lineTo(f.x[j], f.y[j]); j++; }
    ctx.stroke(); i = j;
  }
}
function drawCar(run, p, focused) {
  ctx.globalAlpha = 1;
  ctx.fillStyle = run.color;
  ctx.beginPath(); ctx.arc(p.x, p.y, focused ? 9 : 6, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#111318"; ctx.lineWidth = 2; ctx.stroke();
  ctx.strokeStyle = "#111318"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(p.x, p.y);
  ctx.lineTo(p.x + Math.cos(p.angle) * 16, p.y + Math.sin(p.angle) * 16); ctx.stroke();
}
function drawFinalPos(run) {
  const f = run.frames, n = f.t.length; if (!n) return;
  const x = f.x[n - 1], y = f.y[n - 1];
  ctx.strokeStyle = run.valid ? "#ffd166" : "#ff6b6b"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
  if (!run.valid) {
    ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
    ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke();
  }
}

// ---- main render -----------------------------------------------------------
function render() {
  drawTrack();
  const axis = state.axis;
  RUNS.forEach(run => {
    const ui = state.runUI[run.run_id];
    if (!ui.visible || !run.frames.t.length) return;
    const fi = frameAt(run, axis);
    const p = pose(run, fi);
    if (!p) return;
    drawRunTrail(run, p.idx, ui.opacity);
    if (state.layers.finalPos && p.idx >= run.frames.t.length - 1) drawFinalPos(run);
    drawCar(run, p, run.run_id === state.focus);
  });
  document.getElementById("clock").textContent = clockLabel(axis);
}
function clockLabel(axis) {
  if (state.align === "finish") return (axis * 100).toFixed(0) + "%";
  if (state.align === "stage") return "stage " + axis.toFixed(2);
  return axis.toFixed(2) + "s";
}

// ---- run list UI -----------------------------------------------------------
const runlist = document.getElementById("runlist");
function runSortTime(r) { return r.valid ? r.lap_time : Infinity; }
function buildRunList() {
  runlist.innerHTML = "";
  // Slowest/unfinished at the top, fastest lap at the bottom (descending by time).
  const order = RUNS.slice().sort((a, b) => runSortTime(b) - runSortTime(a));
  order.forEach(run => {
    const ui = state.runUI[run.run_id];
    const row = document.createElement("div");
    row.className = "runrow" + (run.run_id === state.focus ? " focused" : "");
    row.dataset.id = run.run_id;
    const lap = run.valid ? run.lap_time.toFixed(3) + "s" : run.status;
    row.innerHTML =
      '<input type="checkbox" ' + (ui.visible ? "checked" : "") + ' data-act="vis">' +
      '<span class="sw" style="background:' + run.color + '"></span>' +
      '<span class="name" data-act="focus" title="' + run.model + '">' +
        run.run_id + ' &#183; ' + shorten(run.model) +
        (run.is_reference ? ' <span class="badge ref">ref</span>' : '') +
      '</span>' +
      '<span class="meta">' + lap + '</span>';
    runlist.appendChild(row);
  });
}
function shorten(m) { return m.length > 26 ? m.slice(0, 24) + "…" : m; }
runlist.addEventListener("click", e => {
  const row = e.target.closest(".runrow"); if (!row) return;
  const id = row.dataset.id, act = e.target.dataset.act;
  if (act === "vis") { state.runUI[id].visible = e.target.checked; render(); }
  else if (act === "focus" || e.target.classList.contains("name")) {
    state.focus = id; state.axis = 0;  // switching replay restarts the clock
    buildRunList(); syncScrub(); render();
  }
});

// ---- transport -------------------------------------------------------------
const scrub = document.getElementById("scrub");
const playBtn = document.getElementById("play");
function setAxisFromScrub() { state.axis = (scrub.value / 1000) * axisMax(); render(); }
function syncScrub() { scrub.value = Math.round((state.axis / axisMax()) * 1000); }
scrub.addEventListener("input", () => { state.playing = false; playBtn.textContent = "▶ Play"; setAxisFromScrub(); });
playBtn.addEventListener("click", () => {
  state.playing = !state.playing;
  playBtn.textContent = state.playing ? "⏸ Pause" : "▶ Play";
  if (state.playing && state.axis >= axisMax() - 1e-6) state.axis = 0;
});
document.getElementById("restart").addEventListener("click", () => { state.axis = 0; syncScrub(); render(); });
document.getElementById("speed").addEventListener("change", e => state.speed = parseFloat(e.target.value));
document.getElementById("align").addEventListener("change", e => { state.align = e.target.value; state.axis = 0; syncScrub(); render(); });
document.getElementById("trail").addEventListener("input", e => {
  const maxLen = Math.max(...RUNS.map(r => r.frames.t.length || 1));
  state.trailFrames = Math.round((e.target.value / 100) * maxLen);
  render();
});
document.querySelectorAll("#layers input").forEach(cb => cb.addEventListener("change", e => {
  state.layers[e.target.dataset.layer] = e.target.checked;
  render();
}));

let last = performance.now();
function loop(now) {
  const dt = (now - last) / 1000; last = now;
  if (state.playing) {
    state.axis += dt * state.speed * (axisMax() / fullAxisSeconds());
    if (state.axis >= axisMax()) { state.axis = axisMax(); state.playing = false; playBtn.textContent = "▶ Play"; }
    syncScrub(); render();
  }
  requestAnimationFrame(loop);
}

// ---- story mode ------------------------------------------------------------
let storyIndex = 0;
function initStory() {
  if (!DATA.story || !DATA.story.steps.length) return;
  document.getElementById("story-bar").classList.add("on");
  document.getElementById("subtitle").textContent = DATA.story.title;
  document.getElementById("story-prev").addEventListener("click", () => gotoStory(storyIndex - 1));
  document.getElementById("story-next").addEventListener("click", () => gotoStory(storyIndex + 1));
  gotoStory(0);
}
function gotoStory(i) {
  const steps = DATA.story.steps;
  storyIndex = clamp(i, 0, steps.length - 1);
  const step = steps[storyIndex];
  document.getElementById("story-caption").textContent =
    "(" + (storyIndex + 1) + "/" + steps.length + ") " + step.caption;
  state.focus = step.run_id;
  RUNS.forEach(r => { state.runUI[r.run_id].visible = (r.run_id === step.run_id || r.is_reference); });
  state.axis = 0; state.playing = true; playBtn.textContent = "⏸ Pause";
  buildRunList(); syncScrub(); render();
}

// ---- boot ------------------------------------------------------------------
document.getElementById("subtitle").textContent =
  RUNS.length + " runs · track " + TRACK.track_id;
buildRunList();
render();
initStory();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
