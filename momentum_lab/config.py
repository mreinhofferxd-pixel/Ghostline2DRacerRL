"""Central configuration: clock tiers, world/render setup, and car physics.

All physics constants are tuned to ``PHYSICS_DT``. Because the simulation runs at
a fixed timestep, these values are part of the *physics version* (see
``momentum_lab.PHYSICS_VERSION``) — changing them invalidates recorded lap times
and ghosts.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Clock tiers --------------------------------------------------------------
PHYSICS_HZ = 120  # fine fixed step: integration + collision
CONTROL_HZ = 60  # action sample rate == replay rate == future env.step() rate

PHYSICS_DT = 1.0 / PHYSICS_HZ
CONTROL_DT = 1.0 / CONTROL_HZ

assert PHYSICS_HZ % CONTROL_HZ == 0, "PHYSICS_HZ must be an integer multiple of CONTROL_HZ"
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ  # physics integrations per control step

# Clamp the accumulator so a stall (debugger pause, window drag) can't trigger a
# spiral of death where we try to simulate seconds of catch-up in one frame.
MAX_FRAME_TIME = 0.25

# --- World + render -----------------------------------------------------------
WORLD_WIDTH = 1280.0
WORLD_HEIGHT = 720.0

WINDOW_WIDTH = 1280  # also the virtual render canvas; world coords map 1:1 to it
WINDOW_HEIGHT = 720
# Launch fullscreen (borderless at the desktop resolution). F11 toggles at runtime;
# set MOMENTUM_WINDOWED=1 in the environment to force the debug window instead.
START_FULLSCREEN = True
RENDER_FPS_CAP = 240  # cap presentation so we don't spin a core at 100%

BG_COLOR = (24, 26, 30)
GRID_COLOR = (36, 39, 45)
GRID_SPACING = 64

# Default spawn (used only when no track is loaded).
DEFAULT_SPAWN = (200.0, 360.0)
DEFAULT_HEADING = 0.0

# --- Walls / raycasts (M2) ----------------------------------------------------
# Track set is being rebuilt: tracks 2 and 3 were removed (poor layout/checkpoint
# spacing). Easy Loop is the only playable track until the Track 1 quality pass.
MVP_TRACKS = ("track_01_easy_loop",)
DEFAULT_TRACK = MVP_TRACKS[0]
RAYCAST_COUNT = 16  # rays around the car for the F1 view + future RL observation
RAYCAST_MAX_DIST = 1500.0  # cap (world units); ~world diagonal


@dataclass(frozen=True)
class CarPhysics:
    """Tunable car constants. Provisional Milestone-1 values; expect to retune."""

    # Longitudinal (px/s, px/s^2)
    engine_accel: float = 900.0
    brake_accel: float = 1500.0
    reverse_accel: float = 500.0
    max_speed: float = 650.0
    max_boost_speed: float = 950.0
    drag: float = 0.7  # linear drag coefficient, per second
    rolling_resistance: float = 12.0  # constant decel while coasting

    # Steering (kinematic, speed-scaled, reverse-aware)
    turn_rate: float = 3.4  # rad/s at full grip and full steer
    turn_full_speed: float = 260.0  # speed at which steering reaches full authority

    # High-speed understeer: above turn_full_speed a grip turn loses
    # steering authority, fading to this fraction by max_speed, so fast grip turns
    # push wide. Holding drift bypasses the falloff (uses drift_turn_authority),
    # which is what makes drift the tool for tight corners at speed. 1.0 here = no
    # understeer (the pre-change behavior).
    grip_turn_falloff: float = 0.55
    drift_turn_authority: float = 1.0  # turn-authority multiplier while drifting

    # Grip / drift: fraction of LATERAL velocity retained per physics substep.
    # Lower = more bite. Tuned to PHYSICS_DT = 1/120 s.
    grip_normal: float = 0.92
    grip_drift: float = 0.98
    drift_min_speed: float = 280.0  # below this, holding drift does nothing

    # Handbrake: while drifting, the engine can't put down full power
    # (drift_throttle_factor) and a handbrake decel scrubs forward speed
    # (drift_brake_accel). You trade speed to rotate — you cannot hold max_speed on
    # the handbrake. Part of the physics version.
    drift_throttle_factor: float = 0.35  # engine accel available while drifting
    drift_brake_accel: float = 650.0  # handbrake decel on forward speed (px/s^2)

    # Walls / collision. Part of the physics version: removing the
    # into-wall velocity component and scaling the retained tangential speed.
    wall_speed_loss_factor: float = 0.45  # tangential speed RETAINED on an impact
    wall_scrape_friction: float = 0.80  # tangential speed RETAINED on a scrape
    wall_impact_speed: float = 200.0  # into-wall normal speed above which = impact

    # Boost pads. A pad starts a short sustained acceleration window
    # that may push the car above max_speed up to max_boost_speed. Once the boost
    # ends, overspeed decays naturally through drag instead of snapping down.
    boost_accel: float = 800.0
    boost_duration: float = 0.35
    boost_pad_cooldown: float = 1.0
    boost_velocity_slip_angle: float = 0.35  # rad; above this, boost along velocity

    # Body
    radius: float = 14.0  # collision circle (used from Milestone 2)
    length: float = 34.0  # render size
    width: float = 18.0


CAR = CarPhysics()
