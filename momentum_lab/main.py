"""Pygame entry point: fixed-timestep loop, input, and rendering glue.

This is the *only* module that touches real-time pacing. The simulation advances
in fixed control steps; rendering interpolates between the last two states. Run:

    python -m momentum_lab.main
"""

from __future__ import annotations

import dataclasses
import os

import pygame

from . import PHYSICS_VERSION, config
from .core.action import NEUTRAL
from .core.sim import Simulation
from .input import poll_action
from .metrics import RunSummary, append_run_summary
from .physics_identity import physics_config_fingerprint
from .replay import (
    GhostPlayback,
    ReplayData,
    ReplayError,
    ReplayRecorder,
    best_replay_path,
    last_replay_path,
    load_best_replay,
    save_replay,
)
from .render.renderer import Renderer
from .tracks import TrackError, load_track_by_id

# Live drift-tuning step sizes (playtest aid; see _retune).
GRIP_STEP = 0.005
DRIFT_MIN_STEP = 10.0
FALLOFF_STEP = 0.05
BRAKE_STEP = 50.0


@dataclasses.dataclass(frozen=True)
class CompletedRunSave:
    replay: ReplayData
    saved_best: bool


def _best_replay_is_legacy_stale(replay: ReplayData) -> bool:
    return replay.physics_config is None or replay.physics_fingerprint is None


def _should_save_best_replay(candidate: ReplayData, existing: ReplayData | None) -> bool:
    if not candidate.valid or _best_replay_is_legacy_stale(candidate):
        return False
    if existing is None or not existing.valid:
        return True
    if existing.physics_version != candidate.physics_version:
        return True
    if _best_replay_is_legacy_stale(existing):
        return True
    if existing.physics_fingerprint != candidate.physics_fingerprint:
        return False
    return candidate.lap_time < existing.lap_time


def _load_persistent_best(
    track_id: str, cfg: config.CarPhysics
) -> tuple[ReplayData, GhostPlayback] | None:
    try:
        best = load_best_replay(track_id)
    except (OSError, ValueError) as e:
        print(f"[replay] ignoring saved best for {track_id}: {e}")
        return None
    if best is None or not best.valid:
        return None
    if best.track_id != track_id:
        print(f"[replay] saved best track is {best.track_id}; expected {track_id}")
        return None
    if best.physics_version != PHYSICS_VERSION:
        print(
            f"[replay] saved best is {best.physics_version}; current is {PHYSICS_VERSION}"
        )
        return None
    if _best_replay_is_legacy_stale(best):
        print("[replay] saved best is legacy/stale: missing physics fingerprint")
        return None
    current_fingerprint = physics_config_fingerprint(cfg)
    if best.physics_fingerprint != current_fingerprint:
        print("[replay] saved best has a different physics config fingerprint")
        return None
    try:
        ghost = GhostPlayback(best)
    except ReplayError as e:
        print(f"[ghost] ignoring saved best for {track_id}: {e}")
        return None
    return best, ghost


def _load_track_or_none(track_id: str):
    try:
        track = load_track_by_id(track_id)
    except TrackError as e:
        print(f"[track] {e}; starting with no walls")
        return None
    print(f"[track] loaded {track.track_id}")
    return track


def _set_caption(track_id: str) -> None:
    pygame.display.set_caption(f"Momentum Lab - Ghostline (M8 tracks) - {track_id}")


def _make_display(fullscreen: bool) -> pygame.Surface:
    """Create the window / fullscreen surface the renderer scales its canvas onto.

    Fullscreen is borderless at the desktop resolution (smoother alt-tab than
    exclusive fullscreen, no display-mode switch); windowed is the world-size debug
    window and is resizable. World coordinates and physics are unaffected either way —
    only the renderer's final scale-blit changes.
    """
    if fullscreen:
        w, h = pygame.display.get_desktop_sizes()[0]
        return pygame.display.set_mode((w, h), pygame.NOFRAME)
    return pygame.display.set_mode(
        (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), pygame.RESIZABLE
    )


def _save_completed_run(
    sim: Simulation,
    recorder: ReplayRecorder,
) -> CompletedRunSave | None:
    replay = recorder.to_replay(sim)
    summary = RunSummary.from_world(sim.world, replay.frames, physics_cfg=sim.cfg)
    try:
        existing_best = load_best_replay(replay.track_id)
    except (OSError, ValueError) as e:
        print(f"[replay] replacing unreadable best for {replay.track_id}: {e}")
        existing_best = None
    save_best = _should_save_best_replay(replay, existing_best)
    try:
        last_path = save_replay(replay, last_replay_path())
        metrics_path = append_run_summary(summary)
        if save_best:
            save_replay(replay, best_replay_path(replay.track_id))
        suffix = " + best" if save_best else ""
        print(f"[run] saved {last_path} and {metrics_path}{suffix}")
    except OSError as e:
        print(f"[run] could not save replay/metrics: {e}")
        return None
    return CompletedRunSave(replay=replay, saved_best=save_best)


def _retune(sim: Simulation, **changes: float) -> None:
    """Live-adjust the car physics during a playtest.

    Builds one new frozen ``CarPhysics`` and points both ``sim.cfg`` (what the
    physics actually reads) and ``config.CAR`` (what the renderer's drift readout
    reads) at it, so HUD and feel never disagree. This is a dev affordance in the
    already-pygame-bound main loop: it never touches ``core/`` or the physics
    step, so the determinism tests (fresh Simulation, default CAR) are unaffected.
    """
    new = dataclasses.replace(sim.cfg, **changes)
    sim.cfg = new
    config.CAR = new


def main() -> None:
    pygame.init()
    # Fullscreen by default (backlog B3); MOMENTUM_WINDOWED=1 forces the debug window.
    fullscreen = config.START_FULLSCREEN and os.environ.get("MOMENTUM_WINDOWED", "") not in ("1", "true", "True")
    screen = _make_display(fullscreen)
    clock = pygame.time.Clock()

    track_ids = config.MVP_TRACKS
    track_index = track_ids.index(config.DEFAULT_TRACK)
    track = _load_track_or_none(track_ids[track_index])
    _set_caption(track.track_id if track is not None else "<empty>")

    sim = Simulation()
    sim.reset(track=track, seed=0)
    recorder = ReplayRecorder.start(sim)
    renderer = Renderer(screen)
    defaults = config.CAR  # original frozen constants, for the tuning reset key

    previous = sim.snapshot()  # state one control step behind `sim.world`
    accumulator = 0.0
    show_debug = True
    running = True
    loaded_best = None if track is None else _load_persistent_best(track.track_id, sim.cfg)
    ghost = loaded_best[1] if loaded_best is not None else None
    ghost_enabled = ghost is not None
    best_lap: float | None = loaded_best[0].lap_time if loaded_best is not None else None
    lap_recorded = False  # so a finished lap updates `best_lap` exactly once

    while running:
        # Real seconds since last frame (render capped); clamp to avoid catch-up spirals.
        frame_time = min(clock.tick(config.RENDER_FPS_CAP) / 1000.0, config.MAX_FRAME_TIME)
        accumulator += frame_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if not fullscreen:  # let the user freely rescale the debug window
                    renderer.set_display(pygame.display.set_mode(event.size, pygame.RESIZABLE))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if fullscreen:  # ESC leaves fullscreen first; quits from a window
                        fullscreen = False
                        renderer.set_display(_make_display(fullscreen))
                        _set_caption(track.track_id if track is not None else "<empty>")
                    else:
                        running = False
                elif event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    renderer.set_display(_make_display(fullscreen))
                    _set_caption(track.track_id if track is not None else "<empty>")
                elif event.key == pygame.K_r:
                    sim.reset(track=track, seed=0)
                    recorder = ReplayRecorder.start(sim)
                    previous = sim.snapshot()
                    accumulator = 0.0
                    lap_recorded = False  # new run; best_lap is kept
                elif event.key == pygame.K_TAB:
                    track_index = (track_index + 1) % len(track_ids)
                    track = _load_track_or_none(track_ids[track_index])
                    sim.reset(track=track, seed=0)
                    recorder = ReplayRecorder.start(sim)
                    previous = sim.snapshot()
                    accumulator = 0.0
                    loaded_best = (
                        None if track is None else _load_persistent_best(track.track_id, sim.cfg)
                    )
                    ghost = loaded_best[1] if loaded_best is not None else None
                    ghost_enabled = ghost is not None
                    best_lap = loaded_best[0].lap_time if loaded_best is not None else None
                    lap_recorded = False
                    _set_caption(track.track_id if track is not None else "<empty>")
                elif event.key == pygame.K_F1:
                    show_debug = not show_debug
                elif event.key == pygame.K_g:
                    if ghost is None:
                        print("[ghost] no compatible best replay loaded")
                    else:
                        ghost_enabled = not ghost_enabled
                # --- live drift tuning (playtest aid; not a car-control input) ---
                elif event.key == pygame.K_1:
                    _retune(sim, grip_normal=round(max(0.80, sim.cfg.grip_normal - GRIP_STEP), 4))
                elif event.key == pygame.K_2:
                    _retune(sim, grip_normal=round(min(0.999, sim.cfg.grip_normal + GRIP_STEP), 4))
                elif event.key == pygame.K_3:
                    _retune(sim, grip_drift=round(max(0.80, sim.cfg.grip_drift - GRIP_STEP), 4))
                elif event.key == pygame.K_4:
                    _retune(sim, grip_drift=round(min(0.999, sim.cfg.grip_drift + GRIP_STEP), 4))
                elif event.key == pygame.K_5:
                    _retune(sim, drift_min_speed=max(0.0, sim.cfg.drift_min_speed - DRIFT_MIN_STEP))
                elif event.key == pygame.K_6:
                    _retune(sim, drift_min_speed=min(sim.cfg.max_speed, sim.cfg.drift_min_speed + DRIFT_MIN_STEP))
                elif event.key == pygame.K_7:
                    _retune(sim, grip_turn_falloff=round(max(0.10, sim.cfg.grip_turn_falloff - FALLOFF_STEP), 4))
                elif event.key == pygame.K_8:
                    _retune(sim, grip_turn_falloff=round(min(1.0, sim.cfg.grip_turn_falloff + FALLOFF_STEP), 4))
                elif event.key == pygame.K_MINUS:
                    _retune(sim, drift_brake_accel=max(0.0, sim.cfg.drift_brake_accel - BRAKE_STEP))
                elif event.key == pygame.K_EQUALS:
                    _retune(sim, drift_brake_accel=sim.cfg.drift_brake_accel + BRAKE_STEP)
                elif event.key == pygame.K_0:
                    _retune(
                        sim,
                        grip_normal=defaults.grip_normal,
                        grip_drift=defaults.grip_drift,
                        drift_min_speed=defaults.drift_min_speed,
                        grip_turn_falloff=defaults.grip_turn_falloff,
                        drift_brake_accel=defaults.drift_brake_accel,
                    )
                elif event.key == pygame.K_p:
                    print(
                        f"[tuning] grip_normal={sim.cfg.grip_normal}  "
                        f"grip_drift={sim.cfg.grip_drift}  "
                        f"drift_min_speed={sim.cfg.drift_min_speed}  "
                        f"grip_turn_falloff={sim.cfg.grip_turn_falloff}  "
                        f"drift_brake_accel={sim.cfg.drift_brake_accel}"
                    )

        action = poll_action(pygame.key.get_pressed())

        # Fixed-timestep control loop. Action is held constant across each step.
        while accumulator >= config.CONTROL_DT:
            previous = sim.snapshot()
            recorder.step(sim, action)
            accumulator -= config.CONTROL_DT

        run = sim.world.run
        lap_time = run.lap_time(sim.world.tick, config.CONTROL_DT)
        if run.finished and run.valid and not lap_recorded:
            saved = _save_completed_run(sim, recorder)
            if saved is not None and saved.saved_best:
                best_lap = lap_time
                try:
                    ghost = GhostPlayback(saved.replay)
                    ghost_enabled = True
                except ReplayError as e:
                    print(f"[ghost] saved best cannot be used as a ghost: {e}")
            lap_recorded = True

        alpha = accumulator / config.CONTROL_DT
        ghost_pose = None
        ghost_trail = ()
        ghost_delta = None
        if ghost is not None and ghost_enabled:
            ghost_pose = ghost.sample(lap_time)
            ghost_trail = ghost.trail(lap_time)
            if run.started and run.finished:
                ghost_delta = lap_time - ghost.lap_time
            elif run.started:
                ghost_delta = ghost.delta_to_position(
                    sim.world.car.px,
                    sim.world.car.py,
                    lap_time,
                    checkpoint=run.next_cp,
                )
        renderer.render(
            previous.car,
            sim.world.car,
            alpha,
            fps=clock.get_fps(),
            action=action if running else NEUTRAL,
            debug=show_debug,
            walls=sim.world.track.walls,
            surface_outer=sim.world.track.surface_outer,
            surface_inner=sim.world.track.surface_inner,
            racing_line=sim.world.track.racing_line,
            boost_pads=sim.world.track.boost_pads,
            boost_active=sim.world.boost_active,
            boosts_used=sim.world.boosts_used,
            wall_hits=sim.world.wall_hits,
            run=run,
            checkpoints=sim.world.track.checkpoints,
            finish=sim.world.track.finish,
            lap_time=lap_time,
            best_lap=best_lap,
            drift_time=sim.world.drift_time,
            peak_slip=sim.world.peak_slip,
            ghost_pose=ghost_pose,
            ghost_trail=ghost_trail,
            ghost_delta=ghost_delta,
            ghost_available=ghost is not None,
            ghost_enabled=ghost_enabled,
        )
        renderer.present()

    pygame.quit()


if __name__ == "__main__":
    main()
