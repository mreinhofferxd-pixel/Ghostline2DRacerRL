"""Scripted physics experiments + metrics — the agent-tunable measurement layer.

Each experiment drives the deterministic sim through a controlled maneuver and
returns plain numbers, so a coding LLM can change a constant in ``CarPhysics``,
re-run, and trust the delta (the sim is bit-reproducible per build + cfg). This
operationalizes the drift acceptance criteria:

  * "drift is faster than braking through a corner"  -> ``corner_comparison``
  * "over-rotation loses speed"                       -> ``over_rotation``

Strategies branch from one *shared, bit-identical* warmed-up entry state via the
sim's own ``snapshot``/``restore``, so comparisons are apples-to-apples. Nothing
here imports pygame, reads wall-clock, or passes a ``dt`` — same rules as core/.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..config import CAR, CONTROL_DT, CarPhysics
from ..core.action import Action
from ..core.sim import Simulation, World
from ..core.track import BoostPad, Track

# A policy maps (maneuver step index, current world) -> Action, held one control
# step. Constant-action strategies ignore both args.
Policy = Callable[[int, World], Action]


# --- trace recording ---------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    """One control-step snapshot of the quantities a drift metric cares about."""

    tick: int
    t: float
    px: float
    py: float
    speed: float
    heading: float
    slip: float  # signed slip angle (rad): angle between heading and velocity
    fwd_speed: float  # speed projected onto heading (rad)


def _sample(world: World) -> Sample:
    c = world.car
    return Sample(
        tick=world.tick,
        t=world.sim_time,
        px=c.px,
        py=c.py,
        speed=c.speed,
        heading=c.heading,
        slip=c.slip_angle,
        fwd_speed=c.forward_speed,
    )


@dataclass
class Trace:
    """An ordered list of per-control-step samples, with a few reducers."""

    samples: list[Sample]

    @property
    def first(self) -> Sample:
        return self.samples[0]

    @property
    def last(self) -> Sample:
        return self.samples[-1]

    def min_speed(self) -> float:
        return min(s.speed for s in self.samples)

    def peak_abs_slip(self) -> float:
        return max(abs(s.slip) for s in self.samples)

    def path_length(self) -> float:
        """Total distance travelled along the trajectory (world units)."""
        return sum(
            math.hypot(b.px - a.px, b.py - a.py)
            for a, b in zip(self.samples, self.samples[1:])
        )


def drive(
    sim: Simulation,
    policy: Policy,
    *,
    max_steps: int,
    stop: Callable[[World], bool] | None = None,
) -> Trace:
    """Step ``sim`` under ``policy`` until ``stop`` or ``max_steps``, recording a Trace."""
    samples = [_sample(sim.world)]
    for i in range(max_steps):
        sim.step(policy(i, sim.world))
        samples.append(_sample(sim.world))
        if stop is not None and stop(sim.world):
            break
    return Trace(samples)


def _heading_delta(curr: float, start: float) -> float:
    """Unsigned rotation from ``start`` to ``curr`` for the +steer (increasing) turns
    used by every scenario here. Range [0, 2π)."""
    return (curr - start) % (2.0 * math.pi)


def warmed_up(
    cfg: CarPhysics,
    *,
    heading: float = 0.0,
    spawn: tuple[float, float] = (200.0, 360.0),
    warmup_steps: int = 240,
) -> Simulation:
    """Return a sim accelerated to steady-state top speed along ``heading``.

    Every cornering strategy branches from this single state via snapshot/restore,
    guaranteeing a bit-identical entry condition (4 s of full throttle is well past
    the drag/clamp steady state, so entry speed ≈ ``cfg.max_speed``).
    """
    sim = Simulation(cfg)
    sim.reset(spawn=spawn, heading=heading, seed=0)
    drive(sim, lambda i, w: Action(throttle=1.0), max_steps=warmup_steps)
    return sim


# --- experiment 1: drift vs brake vs grip through a corner -------------------
@dataclass(frozen=True)
class CornerResult:
    name: str
    completed: bool  # reached the target heading change before max_steps
    exit_speed: float  # |v| at the exit gate
    exit_fwd_speed: float  # speed in the new facing direction at the gate
    time: float  # seconds spent in the maneuver
    distance: float  # path length travelled to complete the turn (tighter = shorter)
    min_speed: float  # lowest |v| during the maneuver
    peak_slip_deg: float


# Constant-action cornering strategies (all steer hard right to take the corner).
_STRATEGIES: dict[str, Policy] = {
    "grip": lambda i, w: Action(throttle=1.0, steer=1.0),
    "brake": lambda i, w: Action(brake=1.0, steer=1.0),
    "drift": lambda i, w: Action(throttle=1.0, steer=1.0, drift=True),
}


def corner_comparison(
    cfg: CarPhysics = CAR,
    *,
    target_turn_deg: float = 90.0,
    max_steps: int = 900,
    catch_frac: float = 0.65,
) -> tuple[float, dict[str, CornerResult]]:
    """Drive grip/brake/drift/drift_catch through one corner from a shared entry.

    Returns ``(entry_speed, {name: CornerResult})``. ``drift_catch`` models the real
    technique — drift to rotate the nose for the first ``catch_frac`` of the turn,
    then release so the recovered grip hooks the velocity onto the new heading and
    fires out. With high-speed understeer on, a grip turn pushes wide (long path,
    long time); drift_catch takes the tight, quick line *and* exits with speed.
    """
    sim = warmed_up(cfg)
    base = sim.snapshot()
    entry_speed = base.car.speed
    target = math.radians(target_turn_deg)
    start_heading = base.car.heading

    def drift_catch(i: int, w: World) -> Action:
        progress = _heading_delta(w.car.heading, start_heading) / target
        return Action(throttle=1.0, steer=1.0, drift=progress < catch_frac)

    strategies: dict[str, Policy] = {**_STRATEGIES, "drift_catch": drift_catch}
    results: dict[str, CornerResult] = {}
    for name, policy in strategies.items():
        sim.restore(base)
        trace = drive(
            sim,
            policy,
            max_steps=max_steps,
            stop=lambda w: _heading_delta(w.car.heading, start_heading) >= target,
        )
        last = trace.last
        results[name] = CornerResult(
            name=name,
            completed=_heading_delta(last.heading, start_heading) >= target - 1e-9,
            exit_speed=last.speed,
            exit_fwd_speed=last.fwd_speed,
            time=last.t - base.sim_time,
            distance=trace.path_length(),
            min_speed=trace.min_speed(),
            peak_slip_deg=math.degrees(trace.peak_abs_slip()),
        )
    return entry_speed, results


# --- experiment 2: over-rotation loses speed ---------------------------------
@dataclass(frozen=True)
class DriftResult:
    name: str
    exit_speed: float
    speed_lost: float  # entry_speed - exit_speed
    peak_slip_deg: float


def over_rotation(
    cfg: CarPhysics = CAR,
    *,
    steps: int = 90,
    tidy_steer: float = 0.4,
    over_steer: float = 1.0,
) -> tuple[float, DriftResult, DriftResult]:
    """Two coasting drifts of equal duration from one entry state; only steer differs.

    Both coast (no throttle/brake) and hold drift for ``steps`` control steps, so
    drag exposure is identical — the only difference is how far the car is rotated.
    The harder-steered line reaches a larger slip angle, and the grip model bleeds
    lateral velocity faster the larger that angle is, so it must exit slower. This
    is "over-rotation loses speed" as a measurable inequality.
    """
    sim = warmed_up(cfg)
    base = sim.snapshot()
    entry_speed = base.car.speed

    def run(name: str, steer: float) -> DriftResult:
        sim.restore(base)
        trace = drive(
            sim,
            lambda i, w: Action(steer=steer, drift=True),
            max_steps=steps,
        )
        return DriftResult(
            name=name,
            exit_speed=trace.last.speed,
            speed_lost=entry_speed - trace.last.speed,
            peak_slip_deg=math.degrees(trace.peak_abs_slip()),
        )

    return entry_speed, run("tidy", tidy_steer), run("over_rotated", over_steer)


# --- experiment 3: boost pad sprint (M5) ------------------------------------
@dataclass(frozen=True)
class BoostSprintResult:
    name: str
    completed: bool
    time: float
    steps: int
    finish_x: float
    final_x: float
    top_speed: float
    boosts_used: int


def boost_sprint(
    cfg: CarPhysics = CAR,
    *,
    finish_x: float = 1050.0,
    max_steps: int = 600,
) -> dict[str, BoostSprintResult]:
    """Compare one straight full-throttle line with and without a boost pad."""

    def run(name: str, pads: tuple[BoostPad, ...]) -> BoostSprintResult:
        track = Track(
            track_id=f"boost_sprint_{name}",
            spawn=(100.0, 360.0),
            spawn_heading=0.0,
            boost_pads=pads,
        )
        sim = Simulation(cfg)
        sim.reset(track=track, seed=0)
        top_speed = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            sim.step(Action(throttle=1.0))
            top_speed = max(top_speed, sim.world.car.speed)
            if sim.world.car.px >= finish_x:
                break
        return BoostSprintResult(
            name=name,
            completed=sim.world.car.px >= finish_x,
            time=steps * CONTROL_DT,
            steps=steps,
            finish_x=finish_x,
            final_x=sim.world.car.px,
            top_speed=top_speed,
            boosts_used=sim.world.boosts_used,
        )

    pad = BoostPad(420.0, 320.0, 560.0, 400.0)
    return {"plain": run("plain", ()), "boosted": run("boosted", (pad,))}


# --- experiment 4: a scripted time-trial lap (M3) ---------------------------
# The pure-pursuit autopilot follows the track's authored ``racing_line`` (the
# centerline emitted by tools/build_track_01.py). Following it in order traces the
# corridor without cutting into the inner wall and crosses each gate in order for
# free. This fallback is only used for tracks that don't author a line.
EASY_LOOP_LINE: tuple[tuple[float, float], ...] = (
    (1100.0, 560.0),  # bottom-right
    (1100.0, 160.0),  # top-right
    (180.0, 160.0),  # top-left
    (180.0, 560.0),  # bottom-left
)


@dataclass(frozen=True)
class LapResult:
    completed: bool  # the lap closed (all checkpoints in order, then finish)
    valid: bool
    lap_time: float  # seconds (0 if it never closed)
    splits: tuple[float, ...]  # checkpoint split times from the timer start
    cp_reached: int  # how many checkpoints were cleared
    steps: int  # control steps the autopilot used
    wall_hits: int
    boosts_used: int
    top_speed: float
    drift_time: float  # seconds the autopilot spent drifting (0 for a pure grip line)
    peak_slip_deg: float  # most sideways the car got over the lap


def _pursue(target: tuple[float, float], world: World, *, kp: float, cruise: float) -> Action:
    """One pure-pursuit control step toward ``target``: steer proportional to the
    heading error, full throttle when roughly aligned and under ``cruise``, ease /
    light-brake when the target is off to the side (so corners aren't overshot)."""
    car = world.car
    desired = math.atan2(target[1] - car.py, target[0] - car.px)
    err = (desired - car.heading + math.pi) % (2.0 * math.pi) - math.pi
    steer = max(-1.0, min(1.0, kp * err))
    if abs(err) > 0.9:  # pointing well off the target -> scrub speed for the turn
        return Action(brake=0.4, steer=steer)
    if car.speed > cruise:  # at cruising speed -> coast, don't push wide
        return Action(steer=steer)
    return Action(throttle=1.0, steer=steer)


def time_trial(
    track: Track,
    cfg: CarPhysics = CAR,
    *,
    line: tuple[tuple[float, float], ...] | None = None,
    max_steps: int = 4000,
    kp: float = 2.5,
    cruise: float = 320.0,
    switch_dist: float = 90.0,
) -> LapResult:
    """Drive a scripted pure-pursuit lap of ``track`` and report the timed result.

    Deterministic (fixed sim + fixed policy), pygame-free, and tunable by numbers —
    the agent-facing way to ask "does a valid lap report a time, and how fast?"
    without a human at the wheel. ``line`` is the racing line (looped); when not
    given it uses the track's authored ``racing_line`` (falling back to a generic
    ring line for tracks that don't author one).
    """
    if line is None:
        line = track.racing_line or EASY_LOOP_LINE
    sim = Simulation(cfg)
    sim.reset(track=track, seed=0)
    run = sim.world.run

    wp = 0  # index into the (looped) racing line
    top_speed = 0.0
    steps = 0
    for steps in range(1, max_steps + 1):
        target = line[wp % len(line)]
        sim.step(_pursue(target, sim.world, kp=kp, cruise=cruise))
        car = sim.world.car
        top_speed = max(top_speed, car.speed)
        if math.hypot(target[0] - car.px, target[1] - car.py) <= switch_dist:
            wp += 1  # reached this waypoint; aim at the next
        if run.finished:
            break

    return LapResult(
        completed=run.finished,
        valid=run.valid,
        lap_time=run.lap_time(sim.world.tick, CONTROL_DT),
        splits=tuple(run.split_times(CONTROL_DT)),
        cp_reached=run.next_cp,
        steps=steps,
        wall_hits=sim.world.wall_hits,
        boosts_used=sim.world.boosts_used,
        top_speed=top_speed,
        drift_time=sim.world.drift_time,
        peak_slip_deg=math.degrees(sim.world.peak_slip),
    )
