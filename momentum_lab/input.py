"""Keyboard -> Action. The only place that translates raw keys into the seam."""

from __future__ import annotations

import pygame

from .core.action import Action


def poll_action(keys) -> Action:
    """Build an Action from a pygame key-state mapping (digital -> 0/1 axes)."""
    throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
    brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0

    steer = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        steer -= 1.0
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        steer += 1.0

    drift = bool(keys[pygame.K_SPACE])
    return Action(throttle=throttle, brake=brake, steer=steer, drift=drift)
