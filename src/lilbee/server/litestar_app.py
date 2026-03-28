"""Backwards-compatible re-export — all logic lives in app.py and auth.py now."""

from lilbee.server.app import _lifespan, create_app

__all__ = ["_lifespan", "create_app"]
