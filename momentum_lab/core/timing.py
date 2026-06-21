"""Lap timing + checkpoint progress: the run/session layer.

This is deliberately *not* in the physics ``Car``: lap/timer
fields from the car so the physics state stays a clean snapshot-able vector).
``RunState`` is tick-driven — never ``time.time()`` — so it is bit-reproducible and
round-trips through ``snapshot``/``restore`` exactly like the wall stats.

The lap rules:

  * **Timer start = exactly one rule:** standing start behind the line; the timer
    starts on the **first non-zero throttle**. Nothing is counted before that.
  * **Ordered gates:** only the *next* expected checkpoint can advance progress, so
    an out-of-order crossing is simply ignored (rejected). Reverse crossings (the
    wrong way through a gate) never count.
  * **Finish closes the lap** only after every checkpoint is passed in order.
  * A closed lap is **valid** (built the proper way); an ``R`` restart throws the
    run away rather than producing a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoints import Gate


@dataclass
class RunState:
    """Progress + lap timer for the current run. Lives in ``World``; tick-driven."""

    started: bool = False
    start_tick: int = 0
    next_cp: int = 0  # index of the next checkpoint that can advance progress
    finished: bool = False
    valid: bool = False
    lap_ticks: int = 0  # control ticks from start to finish, set when the lap closes
    cp_ticks: tuple[int, ...] = field(default_factory=tuple)  # split tick at each pass
    # Sub-tick finish offset: the fraction (0, 1] of the closing control step at which
    # the car actually crossed the finish line. The true crossing time is
    # ``lap_ticks - (1 - finish_fraction)`` ticks, so two laps that close on the same
    # tick but cross at different points report different ``lap_time``s. Reporting only;
    # ``lap_ticks`` stays the canonical integer the state machine and reward read.
    finish_fraction: float = 0.0

    def copy(self) -> "RunState":
        return RunState(
            started=self.started,
            start_tick=self.start_tick,
            next_cp=self.next_cp,
            finished=self.finished,
            valid=self.valid,
            lap_ticks=self.lap_ticks,
            cp_ticks=self.cp_ticks,  # tuple is immutable: safe to share
            finish_fraction=self.finish_fraction,
        )

    def update(
        self,
        *,
        throttle_on: bool,
        prev: tuple[float, float],
        curr: tuple[float, float],
        checkpoints: tuple[Gate, ...],
        finish: Gate | None,
        tick: int,
    ) -> None:
        """Fold one control step into the run. ``prev``/``curr`` are the car position
        before/after the step (the gate-crossing move segment); ``tick`` is the post-
        step control-tick count. Mutates in place; does nothing once the lap closes."""
        if not self.started:
            if throttle_on:  # the single timer-start rule
                self.started = True
                self.start_tick = tick
            return  # gates don't count until the clock is running
        if self.finished:
            return

        x0, y0 = prev
        x1, y1 = curr
        if self.next_cp < len(checkpoints):
            # Only the next gate in sequence counts; others are out-of-order -> ignored.
            if checkpoints[self.next_cp].crossing(x0, y0, x1, y1) > 0:
                self.cp_ticks = self.cp_ticks + (tick,)
                self.next_cp += 1
        elif finish is not None:
            # All checkpoints cleared in order: the finish line closes a valid lap.
            # Keep the fraction of the closing step at which the line was actually
            # crossed for sub-tick lap-time reporting; the boolean close is unchanged.
            direction, fraction = finish.crossing_with_fraction(x0, y0, x1, y1)
            if direction > 0:
                self.finished = True
                self.valid = True
                self.lap_ticks = tick - self.start_tick
                self.finish_fraction = fraction

    def lap_time(self, tick: int, control_dt: float) -> float:
        """Lap time in seconds. ``0`` before the timer starts; the live running time
        while the lap is open; once it closes, the **sub-tick** crossing time —
        ``lap_ticks`` minus the within-step remainder ``(1 - finish_fraction)`` — so
        two laps that close on the same control tick but cross the finish line at
        different points report different times. ``lap_ticks`` stays the canonical
        integer for the tick-driven state machine, replay validation, and reward."""
        if not self.started:
            return 0.0
        if self.finished:
            return (self.lap_ticks - (1.0 - self.finish_fraction)) * control_dt
        return (tick - self.start_tick) * control_dt

    def split_times(self, control_dt: float) -> list[float]:
        """Checkpoint split times (seconds from the timer start)."""
        return [(t - self.start_tick) * control_dt for t in self.cp_ticks]
