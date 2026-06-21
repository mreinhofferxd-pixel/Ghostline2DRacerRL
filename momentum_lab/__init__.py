"""Momentum Lab — 2D top-down time-trial racer (game layer: Ghostline)."""

__version__ = "0.1.0"
# v2: high-speed grip understeer bypassed by drift, so drift is the
# tool for tight corners at speed instead of being dominated by grip turns.
# v3: drift is a handbrake: cuts engine drive + scrubs forward speed,
# so you cannot hold max_speed while holding the drift button.
# v4: boost pads add sustained overspeed acceleration up to max_boost_speed
# with deterministic active/cooldown timers in World.
PHYSICS_VERSION = "physics_v4"

# Lap-time reporting precision, independent of PHYSICS_VERSION (physics is untouched).
# v1: lap_time rounded up to the whole closing control tick (multiples of 1/60 s).
# v2: sub-tick finish-line interpolation, so lap_time has ~ms precision (B9). Stamped
# into completed-run metrics / RL rollout summaries so old integer-tick records stay
# distinguishable from new sub-tick ones.
TIMING_VERSION = "timing_v2"
