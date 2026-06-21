"""Headless acceptance report: `python -m momentum_lab.eval`.

Prints current feel constants, scripted experiments, and a PASS/FAIL
verdict on each acceptance criterion: the exact surface a coding LLM reads,
tweaks a constant in config.CarPhysics, and re-runs to tune by numbers.

Output is deliberately ASCII-only so it survives any console encoding, pipe, or CI.
"""

from __future__ import annotations

from ..config import CAR, DEFAULT_TRACK
from ..tracks import TrackError, load_track_by_id
from .harness import boost_sprint, corner_comparison, over_rotation, time_trial


def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    cfg = CAR
    print("=== Momentum Lab - headless evaluation ===")
    print(
        f"constants: grip_normal={cfg.grip_normal}  grip_drift={cfg.grip_drift}  "
        f"drift_min_speed={cfg.drift_min_speed}  boost_accel={cfg.boost_accel}  "
        f"max_boost_speed={cfg.max_boost_speed}\n"
    )

    # --- Experiment 1: drift / grip / brake through a corner -----------------
    entry, results = corner_comparison(cfg)
    print(f"[1] Cornering - 90deg turn from a shared entry @ {entry:.0f} px/s")
    print(f"    {'strategy':<12} {'exit_spd':>9} {'fwd_spd':>9} {'time':>7} "
          f"{'dist':>8} {'peak_slip':>10} {'done':>6}")
    for r in results.values():
        print(f"    {r.name:<12} {r.exit_speed:>9.1f} {r.exit_fwd_speed:>9.1f} "
              f"{r.time:>6.2f}s {r.distance:>6.0f}px {r.peak_slip_deg:>8.1f}deg "
              f"{str(r.completed):>6}")
    # Holding drift the whole corner is just handbraking (it scrubs); the controlled
    # technique is drift_catch, so that is what must beat braking.
    technique = results["drift_catch"]
    drift_beats_brake = (
        technique.completed
        and results["brake"].completed
        and technique.exit_speed > results["brake"].exit_speed
    )
    print(f"    -> controlled drift (catch) faster than braking: "
          f"{_verdict(drift_beats_brake)}  (d_exit_speed = "
          f"{technique.exit_speed - results['brake'].exit_speed:+.1f} px/s)")
    # The understeer payoff: drift-and-catch takes a tighter, quicker line than a
    # grip turn (which pushes wide at speed). This is what gives drift a purpose.
    g, dc = results["grip"], results["drift_catch"]
    tighter = dc.distance < g.distance and dc.time < g.time
    print(f"    -> drift_catch tighter line than grip turn (understeer): "
          f"{_verdict(tighter)}  (d_dist = {dc.distance - g.distance:+.0f}px, "
          f"d_time = {dc.time - g.time:+.2f}s)\n")

    # --- Experiment 2: over-rotation loses speed ----------------------------
    entry2, tidy, over = over_rotation(cfg)
    print(f"[2] Over-rotation - equal-duration coasting drifts @ {entry2:.0f} px/s")
    print(f"    {'line':<13} {'exit_spd':>9} {'speed_lost':>11} {'peak_slip':>10}")
    for r in (tidy, over):
        print(f"    {r.name:<13} {r.exit_speed:>9.1f} {r.speed_lost:>11.1f} "
              f"{r.peak_slip_deg:>8.1f}deg")
    over_costs = (
        over.peak_slip_deg > tidy.peak_slip_deg
        and over.exit_speed < tidy.exit_speed
    )
    print(f"    -> over-rotating loses speed: {_verdict(over_costs)}  "
          f"(d_exit_speed = {over.exit_speed - tidy.exit_speed:+.1f} px/s)\n")

    # --- Experiment 3: boost pad sprint (M5) --------------------------------
    sprint = boost_sprint(cfg)
    plain, boosted = sprint["plain"], sprint["boosted"]
    boost_ok = (
        plain.completed
        and boosted.completed
        and boosted.boosts_used > 0
        and boosted.time < plain.time
    )
    print("[3] Boost pads - same straight line with and without one pad")
    print(f"    {'line':<8} {'time':>7} {'top_spd':>9} {'boosts':>7} {'done':>6}")
    for r in (plain, boosted):
        print(f"    {r.name:<8} {r.time:>6.2f}s {r.top_speed:>8.1f} "
              f"{r.boosts_used:>7} {str(r.completed):>6}")
    print(f"    -> boosted line beats no-pad line: {_verdict(boost_ok)}  "
          f"(d_time = {boosted.time - plain.time:+.2f}s, "
          f"d_top_speed = {boosted.top_speed - plain.top_speed:+.1f} px/s)\n")

    # --- Experiment 4: a scripted time-trial lap (M3) -----------------------
    lap_ok = True
    try:
        track = load_track_by_id(DEFAULT_TRACK)
    except TrackError as e:
        print(f"[4] Time trial - skipped ({e})\n")
        track = None
    if track is not None and track.finish is not None:
        lap = time_trial(track, cfg)
        print(f"[4] Time trial - scripted pure-pursuit lap of {track.track_id}")
        print(f"    completed={lap.completed}  valid={lap.valid}  "
              f"checkpoints={lap.cp_reached}/{len(track.checkpoints)}")
        print(f"    lap_time={lap.lap_time:6.3f}s  splits="
              f"{[round(s, 3) for s in lap.splits]}")
        print(f"    top_speed={lap.top_speed:5.0f} px/s  wall_hits={lap.wall_hits}  "
              f"boosts={lap.boosts_used}  steps={lap.steps}")
        print(f"    drift_time={lap.drift_time:4.1f}s  peak_slip={lap.peak_slip_deg:4.0f}deg")
        lap_ok = lap.completed and lap.valid
        print(f"    -> a valid lap reports a time: {_verdict(lap_ok)}\n")
    elif track is not None:
        print(f"[4] Time trial - {track.track_id} has no finish line; skipped\n")

    all_pass = drift_beats_brake and tighter and over_costs and boost_ok and lap_ok
    print(f"=== overall: {_verdict(all_pass)} ===")


if __name__ == "__main__":
    main()
