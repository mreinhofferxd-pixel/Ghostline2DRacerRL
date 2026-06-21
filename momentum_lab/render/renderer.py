"""Pygame renderer.

Takes the previous and current physics states plus an interpolation factor and
draws a smoothly interpolated frame (render runs a step "in the past" — the
standard fixed-timestep technique). It reads sim state; it never writes it.
"""

from __future__ import annotations

import math

import pygame

from .. import config
from ..core import collision
from ..core.action import Action
from ..core.car import Car

CAR_COLOR = (90, 175, 255)
CAR_NOSE_COLOR = (235, 245, 255)
HEADING_COLOR = (235, 245, 255)
VELOCITY_COLOR = (90, 230, 170)
DRIFT_COLOR = (255, 170, 60)
WALL_COLOR = (206, 213, 226)  # kerb / barrier edge (reads against the asphalt fill)
# Track surface fill (render-only; from Track.surface_outer/inner). Turns the track
# from a wireframe into a readable asphalt ribbon with a distinct infield.
SURFACE_ASPHALT = (46, 49, 57)
SURFACE_INFIELD = (29, 33, 39)
SURFACE_EDGE = (66, 71, 82)
RACING_LINE_COLOR = (58, 64, 76)  # faint ideal-line guide (debug only)
FINISH_LIGHT = (236, 240, 246)  # start/finish checker
FINISH_DARK = (32, 34, 39)
RAY_COLOR = (66, 92, 80)
RAY_HIT_COLOR = (120, 200, 160)
HUD_COLOR = (210, 214, 220)
HUD_DIM = (130, 135, 142)
# Checkpoint / lap colors.
GATE_PASSED_COLOR = (70, 120, 90)  # already cleared
GATE_TARGET_COLOR = (255, 210, 70)  # the gate to clear next
GATE_UPCOMING_COLOR = (90, 110, 140)  # later checkpoints
FINISH_COLOR = (235, 245, 255)
BEST_COLOR = (255, 210, 70)
BOOST_COLOR = (90, 230, 210)
BOOST_FILL = (34, 78, 76)
GHOST_FILL = (90, 175, 255, 70)
GHOST_OUTLINE = (150, 220, 255, 165)
GHOST_TRAIL = (90, 175, 255, 70)
DELTA_AHEAD = (90, 230, 170)
DELTA_BEHIND = (255, 170, 60)
SKID_COLOR = (44, 44, 50)  # tire scrub on the track surface (just above the bg)
SKID_MAX_POINTS = 1600  # per side; the trail length cap (~ several seconds of drift)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _fmt_time(t: float) -> str:
    """Lap time with millisecond precision; minutes only once they're needed."""
    if t >= 60.0:
        return f"{int(t // 60)}:{t % 60:06.3f}"
    return f"{t:5.3f}s"


def _lerp_angle(a: float, b: float, t: float) -> float:
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * t


class Camera:
    """World -> screen transform. Identity for Phase 1; the seam for zoom later."""

    def __init__(self, scale: float = 1.0, offset: tuple[float, float] = (0.0, 0.0)):
        self.scale = scale
        self.ox, self.oy = offset

    def to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        return (int(wx * self.scale + self.ox), int(wy * self.scale + self.oy))


class Renderer:
    """Draws the world natively at the display resolution.

    A single ``scale`` (device pixels per world unit) drives the camera, the fonts, and
    every stroke width, so the whole scene is rendered crisply at the real resolution
    and its canvas blits onto the display 1:1 (centered, with black letterbox bars for
    non-16:9 displays) — no upscaling, hence no blur. World coordinates and physics are
    untouched; ``scale`` only affects rendering. ``set_display`` rebuilds the canvas,
    camera, and fonts whenever the window size or fullscreen state changes.
    """

    # Logical design resolution; world coords span this and ``scale`` maps it to device px.
    BASE_W = config.WINDOW_WIDTH
    BASE_H = config.WINDOW_HEIGHT

    def __init__(self, display: pygame.Surface) -> None:
        # Skid-mark trails (render-only; never read by the sim). Two polylines of
        # world-space rear-corner points; a ``None`` entry marks a gap (non-drift).
        self._skid_left: list[tuple[float, float] | None] = []
        self._skid_right: list[tuple[float, float] | None] = []
        self._last_pos: tuple[float, float] | None = None
        self.set_display(display)

    def set_display(self, display: pygame.Surface) -> None:
        """Point the renderer at a new display and rebuild the scaled canvas, camera,
        and fonts so the scene renders natively at its resolution. Call on F11/resize."""
        self.display = display
        dw, dh = display.get_size()
        self.scale = min(dw / self.BASE_W, dh / self.BASE_H)
        # Canvas = world scaled to device px; it fits inside the display, so present()
        # centers it 1:1 with letterbox/pillarbox bars (no upscaling, no blur).
        cw, ch = round(self.BASE_W * self.scale), round(self.BASE_H * self.scale)
        self.canvas = pygame.Surface((cw, ch)).convert()
        self.screen = self.canvas
        self.camera = Camera(scale=self.scale)
        self.font = pygame.font.SysFont("consolas", max(10, round(18 * self.scale)))
        self.font_big = pygame.font.SysFont("consolas", max(12, round(22 * self.scale)), bold=True)

    def _s(self, size: float) -> int:
        """Scale a stroke width / radius from world units to device pixels (min 1)."""
        return max(1, round(size * self.scale))

    def _px(self, offset: float) -> int:
        """Scale a HUD pixel offset to the active resolution."""
        return round(offset * self.scale)

    def render(
        self,
        prev: Car,
        curr: Car,
        alpha: float,
        *,
        fps: float,
        action: Action,
        debug: bool,
        walls=(),
        surface_outer=(),
        surface_inner=(),
        racing_line=(),
        boost_pads=(),
        boost_active: bool = False,
        boosts_used: int = 0,
        wall_hits: int = 0,
        run=None,
        checkpoints=(),
        finish=None,
        lap_time: float = 0.0,
        best_lap: float | None = None,
        drift_time: float = 0.0,
        peak_slip: float = 0.0,
        ghost_pose=None,
        ghost_trail=(),
        ghost_delta: float | None = None,
        ghost_available: bool = False,
        ghost_enabled: bool = False,
    ) -> None:
        self._draw_background()
        # Asphalt + infield fill goes down first so everything else sits on the track.
        self._draw_surface(surface_outer, surface_inner)
        if debug:
            self._draw_racing_line(racing_line)

        # Interpolate the visible car between the two most recent physics states.
        px = _lerp(prev.px, curr.px, alpha)
        py = _lerp(prev.py, curr.py, alpha)
        heading = _lerp_angle(prev.heading, curr.heading, alpha)
        drifting = self._is_drifting(curr, action)

        # Skid marks belong on the ground, under the track furniture and the car.
        self._update_skids(px, py, heading, drifting)
        self._draw_skids()

        self._draw_boost_pads(boost_pads)
        self._draw_walls(walls)
        next_cp = run.next_cp if run is not None else 0
        self._draw_gates(checkpoints, finish, next_cp, debug)

        if debug:
            self._draw_rays(px, py, heading, walls)

        if ghost_pose is not None:
            self._draw_ghost_trail(ghost_trail)
            self._draw_ghost_car(ghost_pose)
        self._draw_car(px, py, heading, drifting)
        if debug:
            self._draw_debug_vectors(px, py, heading, curr)

        self._draw_hud(
            curr, action, fps, debug, wall_hits,
            run=run, total_cp=len(checkpoints), lap_time=lap_time, best_lap=best_lap,
            drift_time=drift_time, peak_slip=peak_slip,
            boost_active=boost_active, boosts_used=boosts_used,
            ghost_delta=ghost_delta, ghost_available=ghost_available,
            ghost_enabled=ghost_enabled,
        )
        if run is not None and run.finished:
            self._draw_lap_banner(lap_time, best_lap)

    def present(self) -> None:
        """Blit the natively-rendered canvas onto the display and flip.

        The canvas is already at device resolution (rendered through ``scale``), so this
        is a 1:1 centered blit with black letterbox/pillarbox bars on non-16:9 displays —
        no scaling, hence no blur.
        """
        dw, dh = self.display.get_size()
        cw, ch = self.canvas.get_size()
        if (cw, ch) != (dw, dh):
            self.display.fill((0, 0, 0))  # letterbox / pillarbox bars
        self.display.blit(self.canvas, ((dw - cw) // 2, (dh - ch) // 2))
        pygame.display.flip()

    # --- pieces ---------------------------------------------------------------
    def _draw_background(self) -> None:
        self.screen.fill(config.BG_COLOR)
        lw = self._s(1)
        # Grid drawn in world space through the camera, so spacing scales with the view.
        for x in range(0, self.BASE_W + 1, config.GRID_SPACING):
            pygame.draw.line(self.screen, config.GRID_COLOR,
                             self.camera.to_screen(x, 0), self.camera.to_screen(x, self.BASE_H), lw)
        for y in range(0, self.BASE_H + 1, config.GRID_SPACING):
            pygame.draw.line(self.screen, config.GRID_COLOR,
                             self.camera.to_screen(0, y), self.camera.to_screen(self.BASE_W, y), lw)

    def _update_skids(self, px: float, py: float, heading: float, drifting: bool) -> None:
        """Lay down rear-tire scrub marks while drifting; break the trail otherwise.
        Clears on a teleport (an ``R`` restart snaps the car far in a single frame)."""
        if self._last_pos is not None and (
            math.hypot(px - self._last_pos[0], py - self._last_pos[1]) > 120.0
        ):
            self._skid_left.clear()
            self._skid_right.clear()
        self._last_pos = (px, py)

        if drifting:
            cfg = config.CAR
            c, s = math.cos(heading), math.sin(heading)
            hl, hw = cfg.length / 2.0, cfg.width / 2.0
            # The two rear corners (local x = -hl) are where the scrub shows.
            self._skid_left.append((px - hl * c + hw * s, py - hl * s - hw * c))
            self._skid_right.append((px - hl * c - hw * s, py - hl * s + hw * c))
        elif self._skid_left and self._skid_left[-1] is not None:
            self._skid_left.append(None)  # gap: don't connect across a non-drift run
            self._skid_right.append(None)

        for store in (self._skid_left, self._skid_right):
            if len(store) > SKID_MAX_POINTS:
                del store[: len(store) - SKID_MAX_POINTS]

    def _draw_skids(self) -> None:
        for store in (self._skid_left, self._skid_right):
            prev: tuple[int, int] | None = None
            for p in store:
                if p is None:
                    prev = None
                    continue
                sp = self.camera.to_screen(p[0], p[1])
                if prev is not None:
                    pygame.draw.line(self.screen, SKID_COLOR, prev, sp, self._s(3))
                prev = sp

    def _draw_surface(self, outer, inner) -> None:
        """Fill the drivable ribbon: asphalt inside the outer loop, infield over the
        inner loop. Render-only (``Track.surface_outer/inner``); physics uses walls."""
        if len(outer) >= 3:
            pts = [self.camera.to_screen(x, y) for x, y in outer]
            pygame.draw.polygon(self.screen, SURFACE_ASPHALT, pts)
            pygame.draw.polygon(self.screen, SURFACE_EDGE, pts, self._s(1))
        if len(inner) >= 3:
            pts = [self.camera.to_screen(x, y) for x, y in inner]
            pygame.draw.polygon(self.screen, SURFACE_INFIELD, pts)
            pygame.draw.polygon(self.screen, SURFACE_EDGE, pts, self._s(1))

    def _draw_racing_line(self, line) -> None:
        if len(line) < 2:
            return
        pts = [self.camera.to_screen(x, y) for x, y in line]
        pygame.draw.lines(self.screen, RACING_LINE_COLOR, True, pts, self._s(1))

    def _draw_walls(self, walls) -> None:
        for s in walls:
            a = self.camera.to_screen(s.x1, s.y1)
            b = self.camera.to_screen(s.x2, s.y2)
            pygame.draw.line(self.screen, WALL_COLOR, a, b, self._s(4))

    def _draw_boost_pads(self, boost_pads) -> None:
        for pad in boost_pads:
            left, top, right, bottom = pad.bounds
            x, y = self.camera.to_screen(left, top)
            x2, y2 = self.camera.to_screen(right, bottom)
            rect = pygame.Rect(x, y, x2 - x, y2 - y)
            pygame.draw.rect(self.screen, BOOST_FILL, rect)
            pygame.draw.rect(self.screen, BOOST_COLOR, rect, self._s(2))
            cx, cy = pad.center
            center = self.camera.to_screen(cx, cy)
            pygame.draw.line(
                self.screen,
                BOOST_COLOR,
                self.camera.to_screen(left + 14, cy),
                self.camera.to_screen(right - 14, cy),
                self._s(2),
            )
            pygame.draw.circle(self.screen, BOOST_COLOR, center, self._s(4))

    def _draw_gates(self, checkpoints, finish, next_cp: int, debug: bool) -> None:
        """Checkpoint + finish gates. The next gate to clear is always highlighted so
        the route is readable; order numbers and the forward-normal arrow are debug
        (F1) only."""
        for i, g in enumerate(checkpoints):
            if i < next_cp:
                color = GATE_PASSED_COLOR
            elif i == next_cp:
                color = GATE_TARGET_COLOR
            else:
                color = GATE_UPCOMING_COLOR
            self._draw_gate(g, color, label=str(i + 1) if debug else None, debug=debug)
        if finish is not None:
            all_done = next_cp >= len(checkpoints)
            # Always a checkered band so the start/finish reads at a glance; once every
            # checkpoint is cleared, overlay a bright line so "the lap can close now".
            self._draw_finish_checker(finish)
            if all_done:
                a = self.camera.to_screen(finish.x1, finish.y1)
                b = self.camera.to_screen(finish.x2, finish.y2)
                pygame.draw.line(self.screen, GATE_TARGET_COLOR, a, b, self._s(3))
            if debug:
                self._draw_gate(finish, FINISH_COLOR, label="F", debug=True)

    def _draw_gate(self, g, color, *, label, debug: bool) -> None:
        a = self.camera.to_screen(g.x1, g.y1)
        b = self.camera.to_screen(g.x2, g.y2)
        pygame.draw.line(self.screen, color, a, b, self._s(3))
        cx, cy = g.center
        if debug:  # forward-normal arrow shows the required crossing direction
            base = self.camera.to_screen(cx, cy)
            tip = self.camera.to_screen(cx + g.nx * 30, cy + g.ny * 30)
            pygame.draw.line(self.screen, color, base, tip, self._s(2))
            pygame.draw.circle(self.screen, color, tip, self._s(3))
        if label is not None:
            surf = self.font.render(label, True, color)
            lx, ly = self.camera.to_screen(cx - g.nx * 20, cy - g.ny * 20)
            self.screen.blit(surf, (lx - surf.get_width() // 2, ly - surf.get_height() // 2))

    def _draw_finish_checker(self, g) -> None:
        """Draw the finish gate as an alternating light/dark checker band."""
        n = 9
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            ax = g.x1 + (g.x2 - g.x1) * t0
            ay = g.y1 + (g.y2 - g.y1) * t0
            bx = g.x1 + (g.x2 - g.x1) * t1
            by = g.y1 + (g.y2 - g.y1) * t1
            color = FINISH_LIGHT if i % 2 == 0 else FINISH_DARK
            pygame.draw.line(
                self.screen,
                color,
                self.camera.to_screen(ax, ay),
                self.camera.to_screen(bx, by),
                self._s(9),
            )

    def _draw_lap_banner(self, lap_time: float, best_lap: float | None) -> None:
        new_best = best_lap is not None and lap_time <= best_lap + 1e-9
        w, h = self.screen.get_size()
        title = self.font_big.render("LAP COMPLETE", True, HUD_COLOR)
        sub_color = BEST_COLOR if new_best else HUD_COLOR
        sub_text = f"{_fmt_time(lap_time)}" + ("   NEW BEST!" if new_best else "")
        sub = self.font_big.render(sub_text, True, sub_color)
        self.screen.blit(title, ((w - title.get_width()) // 2, h // 2 - self._px(40)))
        self.screen.blit(sub, ((w - sub.get_width()) // 2, h // 2 - self._px(8)))
        hint = self.font.render("R to restart", True, HUD_DIM)
        self.screen.blit(hint, ((w - hint.get_width()) // 2, h // 2 + self._px(28)))

    def _draw_rays(self, px: float, py: float, heading: float, walls) -> None:
        if not walls:
            return
        center = self.camera.to_screen(px, py)
        fan = collision.ray_fan(
            px, py, heading, walls, config.RAYCAST_COUNT, config.RAYCAST_MAX_DIST
        )
        for ang, dist in fan:
            end = self.camera.to_screen(px + math.cos(ang) * dist, py + math.sin(ang) * dist)
            pygame.draw.line(self.screen, RAY_COLOR, center, end, self._s(1))
            pygame.draw.circle(self.screen, RAY_HIT_COLOR, end, self._s(2))

    def _draw_car(self, px: float, py: float, heading: float, drifting: bool) -> None:
        cfg = config.CAR
        c, s = math.cos(heading), math.sin(heading)
        hl, hw = cfg.length / 2.0, cfg.width / 2.0
        body = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
        pts = [self.camera.to_screen(px + dx * c - dy * s, py + dx * s + dy * c) for dx, dy in body]
        pygame.draw.polygon(self.screen, CAR_COLOR, pts)
        pygame.draw.polygon(self.screen, DRIFT_COLOR if drifting else CAR_NOSE_COLOR, pts, self._s(2))
        # A short nose marker so the facing direction is unmistakable.
        nose = self.camera.to_screen(px + (hl + 8) * c, py + (hl + 8) * s)
        center = self.camera.to_screen(px, py)
        pygame.draw.line(self.screen, CAR_NOSE_COLOR, center, nose, self._s(2))

    def _draw_ghost_trail(self, trail) -> None:
        points = [self.camera.to_screen(p.x, p.y) for p in trail]
        if len(points) < 2:
            return
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(overlay, GHOST_TRAIL, False, points, self._s(2))
        self.screen.blit(overlay, (0, 0))

    def _draw_ghost_car(self, ghost) -> None:
        cfg = config.CAR
        c, s = math.cos(ghost.angle), math.sin(ghost.angle)
        hl, hw = cfg.length / 2.0, cfg.width / 2.0
        body = [(hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)]
        pts = [
            self.camera.to_screen(ghost.x + dx * c - dy * s, ghost.y + dx * s + dy * c)
            for dx, dy in body
        ]
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        outline = GHOST_OUTLINE
        if ghost.boost:
            outline = (*BOOST_COLOR, 175)
        elif ghost.drift:
            outline = (*DRIFT_COLOR, 175)
        pygame.draw.polygon(overlay, GHOST_FILL, pts)
        pygame.draw.polygon(overlay, outline, pts, self._s(2))
        nose = self.camera.to_screen(ghost.x + (hl + 7) * c, ghost.y + (hl + 7) * s)
        center = self.camera.to_screen(ghost.x, ghost.y)
        pygame.draw.line(overlay, outline, center, nose, self._s(2))
        self.screen.blit(overlay, (0, 0))

    def _draw_debug_vectors(self, px: float, py: float, heading: float, car: Car) -> None:
        center = self.camera.to_screen(px, py)
        # Collision circle (used from Milestone 2).
        pygame.draw.circle(self.screen, HUD_DIM, center, self._s(config.CAR.radius), self._s(1))
        # Heading vector (white) and velocity vector (green) — drift is the gap.
        hx = self.camera.to_screen(px + math.cos(heading) * 60, py + math.sin(heading) * 60)
        pygame.draw.line(self.screen, HEADING_COLOR, center, hx, self._s(2))
        if car.speed > 1.0:
            scale = 60.0 / config.CAR.max_speed
            vx = self.camera.to_screen(px + car.vx * scale, py + car.vy * scale)
            pygame.draw.line(self.screen, VELOCITY_COLOR, center, vx, self._s(2))
            # Lateral (scrub) component along the car's right axis — exactly what the
            # grip model bleeds each substep; its length is the drift "sideways" speed.
            rx, ry = -math.sin(heading), math.cos(heading)
            v_lat = car.vx * rx + car.vy * ry
            lat = self.camera.to_screen(px + rx * v_lat * scale, py + ry * v_lat * scale)
            pygame.draw.line(self.screen, DRIFT_COLOR, center, lat, self._s(2))

    def _draw_hud(
        self, car, action, fps, debug, wall_hits, *, run, total_cp, lap_time, best_lap,
        drift_time=0.0, peak_slip=0.0, boost_active=False, boosts_used=0,
        ghost_delta=None, ghost_available=False, ghost_enabled=False,
    ) -> None:
        drifting = self._is_drifting(car, action)
        cleared = min(run.next_cp, total_cp) if run is not None else 0
        all_cp = total_cp > 0 and cleared >= total_cp
        finished = run is not None and run.finished
        time_color = BEST_COLOR if finished and (best_lap is None or lap_time <= best_lap + 1e-9) else HUD_COLOR
        best_text = f"Best  {_fmt_time(best_lap)}" if best_lap is not None else "Best  --"
        if ghost_delta is None:
            delta_text = "Delta --"
            delta_color = HUD_DIM
        else:
            delta_text = f"Delta {ghost_delta:+6.3f}s"
            delta_color = DELTA_AHEAD if ghost_delta < -0.001 else DELTA_BEHIND
            if abs(ghost_delta) <= 0.001:
                delta_color = HUD_COLOR
        ghost_text = "Ghost --"
        ghost_color = HUD_DIM
        if ghost_available:
            ghost_text = f"Ghost {'ON ' if ghost_enabled else 'off'}"
            ghost_color = GHOST_OUTLINE[:3] if ghost_enabled else HUD_DIM
        lines = [
            (self.font_big, f"Lap  {_fmt_time(lap_time)}", time_color),
            (self.font, best_text, BEST_COLOR if best_lap is not None else HUD_DIM),
            (self.font, delta_text, delta_color),
            (self.font, ghost_text, ghost_color),
            (self.font, f"CP    {cleared}/{total_cp}", (90, 230, 170) if all_cp else HUD_COLOR),
            (self.font, f"Speed {car.speed:5.0f} px/s", HUD_COLOR),
            (self.font, f"Drift {'ON ' if drifting else 'off'}", DRIFT_COLOR if drifting else HUD_DIM),
            (self.font, f"Boost {'ON ' if boost_active else 'off'}  {boosts_used}", BOOST_COLOR if boost_active else HUD_DIM),
            (self.font, f"Walls {wall_hits}", HUD_COLOR if wall_hits else HUD_DIM),
        ]
        x0 = self._px(14)
        y = self._px(14)
        for font, text, color in lines:
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += surf.get_height() + self._px(4)

        if debug:
            cfg = config.CAR  # kept in sync with the live sim.cfg by main.py tuning
            dbg = [
                (f"fps {fps:4.0f}", HUD_DIM),
                (f"pos ({car.px:6.1f}, {car.py:6.1f})", HUD_DIM),
                (f"vel ({car.vx:6.1f}, {car.vy:6.1f})", HUD_DIM),
                (f"slip {math.degrees(car.slip_angle):+5.1f} deg", HUD_DIM),
                (f"drift {drift_time:4.1f}s  peak {math.degrees(peak_slip):3.0f} deg", HUD_DIM),
                ("-- drift tuning --", HUD_DIM),
                (f"grip_normal  {cfg.grip_normal:.3f}  [1/2]", HUD_COLOR),
                (f"grip_drift   {cfg.grip_drift:.3f}  [3/4]", HUD_COLOR),
                (f"drift_min    {cfg.drift_min_speed:4.0f}  [5/6]", HUD_COLOR),
                (f"grip_falloff {cfg.grip_turn_falloff:.2f}  [7/8]", HUD_COLOR),
                (f"drift_brake  {cfg.drift_brake_accel:4.0f}  [-/=]", HUD_COLOR),
                ("0 reset   P print to console", HUD_DIM),
            ]
            for text, color in dbg:
                surf = self.font.render(text, True, color)
                self.screen.blit(surf, (x0, y))
                y += surf.get_height() + self._px(2)

        hint = "WASD drive  SPACE drift  TAB track  G ghost  R restart  F1 debug  F11 fullscreen  ESC quit"
        surf = self.font.render(hint, True, HUD_DIM)
        self.screen.blit(surf, (x0, self.screen.get_height() - surf.get_height() - self._px(12)))

    @staticmethod
    def _is_drifting(car: Car, action: Action) -> bool:
        return bool(action.drift) and car.speed >= config.CAR.drift_min_speed
