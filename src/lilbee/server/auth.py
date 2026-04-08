"""Session token auth middleware with decorator-based read-only marking."""

from __future__ import annotations

import hmac
import json
import logging
import secrets
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from litestar.exceptions import NotAuthorizedException
from litestar.types import ASGIApp, Receive, Scope, Send

from lilbee.config import cfg

log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_session_token: str | None = None  # None = auth disabled (tests), "" = not yet initialized


def read_only(fn: F) -> F:
    """Mark a route handler as read-only (no auth required)."""
    fn._lilbee_read_only = True  # type: ignore[attr-defined]
    return fn


def _server_json_path() -> Path:
    return cfg.data_dir / "server.json"


def generate_session_token() -> str:
    """Generate a random session token and persist to server.json."""
    global _session_token
    _session_token = secrets.token_urlsafe(32)
    path = _server_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"token": _session_token}))
    if sys.platform != "win32":
        path.chmod(0o600)
    return _session_token


def cleanup_session_token() -> None:
    """Remove server.json on shutdown and clear the in-memory token."""
    global _session_token
    _session_token = None
    path = _server_json_path()
    path.unlink(missing_ok=True)


class AuthMiddleware:
    """Bearer token auth middleware for mutating endpoints."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        handler = scope.get("route_handler")
        if handler and getattr(handler.fn, "_lilbee_read_only", False):
            await self.app(scope, receive, send)
            return

        if _session_token is None:  # auth disabled (tests set token to None)
            await self.app(scope, receive, send)
            return
        if not _session_token:
            raise NotAuthorizedException("Server token not initialized")

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        if hmac.compare_digest(auth_header, f"Bearer {_session_token}"):
            await self.app(scope, receive, send)
            return
        raise NotAuthorizedException("Missing or invalid bearer token")
