# Momentum Lab - Ghostline

A 2D top-down time-trial racer built on a **deterministic fixed-timestep
simulation core**, designed to later become a reinforcement-learning environment.
Use [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) for the compact current
architecture/status snapshot and [BACKLOG.md](BACKLOG.md) for the active task
queue.

## Status

In place:

- Fixed-timestep sim (120 Hz physics / 60 Hz control) decoupled from render, with
  interpolation.
- The `Action` seam: keyboard and later an RL policy produce the same input.
- Emergent drift from a local-frame grip model, high-speed grip understeer, and
  handbrake-style drift-and-catch behavior.
- `reset(seed)` / `snapshot()` / `restore()` / `state_hash()` and determinism
  tests.
- JSON track loading, segment walls, swept collision, wall raycasts, ordered
  checkpoints, finish gate, and tick-derived lap timing.
- Boost pads with sustained acceleration, per-pad cooldowns, HUD/debug rendering,
  and headless M5 acceptance tests.
- Replay recording/playback utilities that store seed + initial state + the
  canonical control-rate `Action` stream, plus derived frames for future ghosts.
- Completed laps write `replays/last_run.json`, persistent best replay files, and
  run metrics in `runs/runs.jsonl`.
- Visual best ghost playback loads a compatible best replay, interpolates its
  cached frames, draws a translucent ghost/trail, and shows live delta.
- One current generated track: Easy Loop (`track_01_easy_loop`).
- Gymnasium-style RL wrapper, rollout artifacts, PPO training/eval, analytics,
  static SVG reports, and trace comparison.

Not yet: menus/audio/controller polish, multi-track RL, and the interactive RL
replay/story viewer.

## Setup

```bash
python -m pip install -e .          # installs pygame-ce
python -m pip install -e .[dev]     # + pytest
```

## Run

```bash
python -m momentum_lab.main
```

Controls: **WASD/arrows** drive, **Space** drift, **G** ghost, **R** restart,
**F11** fullscreen/windowed, **F1** debug, **Esc** quit. In debug view, the white
line is heading and the green line is velocity; drift is the angle between them.

## Test

```bash
python -m pytest -q
python -m momentum_lab.eval
```
