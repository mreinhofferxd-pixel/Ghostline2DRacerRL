"""Deterministic identity for simulation-affecting car physics constants."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields

from .config import CarPhysics

_RENDER_ONLY_FIELDS = frozenset({"length", "width"})


def physics_config_payload(cfg: CarPhysics) -> dict[str, float]:
    """Return the canonical replay/metrics payload for ``cfg``."""
    payload: dict[str, float] = {}
    for field in fields(cfg):
        if field.name in _RENDER_ONLY_FIELDS:
            continue
        payload[field.name] = float(getattr(cfg, field.name))
    return payload


def physics_config_fingerprint(cfg: CarPhysics) -> str:
    """Return a full SHA-256 digest for the simulation-affecting physics config."""
    payload = physics_config_payload(cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
