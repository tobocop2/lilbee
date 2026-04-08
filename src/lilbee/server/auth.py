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


def read_only(fn: F) -> F:
    """Mark a route handler as read-only (no auth required)."""
    fn._lilbee_read_only = True  # type: ignore[attr-defined]
    return fn


def _server_json_path() -> Path:
    return cfg.data_dir / "server.json"


class SessionManager:
    """Manages the server session token lifecycle.

    Replaces the old module-level ``_session_token`` global so that auth
    state is explicit and injectable rather than hidden mutable state.
    """

    def __init__(self) -> None:
        self.token: str | None = None

    def generate(self) -> str:
        """Generate a random session token and persist to server.json."""
        self.token = secrets.token_urlsafe(32)
        path = _server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"token": self.token}))
        if sys.platform != "win32":
            path.chmod(0o600)
        return self.token

    def cleanup(self) -> None:
        """Remove server.json on shutdown and clear the in-memory token."""
        self.token = None
        path = _server_json_path()
        path.unlink(missing_ok=True)

    def validate(self, auth_header: str) -> bool:
        """Check whether *auth_header* carries a valid bearer token."""
        if self.token is None:
            return True  # auth disabled (tests)
        if not self.token:
            raise NotAuthorizedException("Server token not initialized")
        return hmac.compare_digest(auth_header, f"Bearer {self.token}")


# Singleton instance — used by AuthMiddleware and the app lifespan.
session_manager = SessionManager()


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

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        if session_manager.validate(auth_header):
            await self.app(scope, receive, send)
            return
        raise NotAuthorizedException("Missing or invalid bearer token")
