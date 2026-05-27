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

from lilbee.core.config import cfg

log = logging.getLogger(__name__)

_TOKEN_BYTES = 32

F = TypeVar("F", bound=Callable[..., Any])


# Set of route-handler functions that bypass auth. Populated at import time by
# the @read_only decorator; checked by AuthMiddleware via membership lookup.
# Module-level set is intentional: route handlers are defined once at import,
# the registry has no other lifecycle, and the alternative (mutating an
# attribute on the function object) lands every check on a # type: ignore
# because mypy cannot see the dynamic attribute on Callable.
_READ_ONLY_HANDLERS: set[Callable[..., Any]] = set()


def read_only(fn: F) -> F:
    """Mark a route handler as read-only (no auth required)."""
    _READ_ONLY_HANDLERS.add(fn)
    return fn


def is_read_only(fn: Callable[..., Any]) -> bool:
    """True iff *fn* was decorated with :func:`read_only`."""
    return fn in _READ_ONLY_HANDLERS


def server_json_path() -> Path:
    """Return the path to the server session file."""
    return cfg.data_dir / "server.json"


class SessionManager:
    """Manages the server session token lifecycle.
    Replaces the old module-level ``_session_token`` global so that auth
    state is explicit and injectable rather than hidden mutable state.
    """

    def __init__(self) -> None:
        self.token: str | None = None

    def load_or_generate(self) -> str:
        """Return the persisted token if shape-valid; generate a new one otherwise."""
        path = server_json_path()
        existing = self._read_persisted_token(path)
        if existing is not None:
            self.token = existing
            return existing
        self.token = secrets.token_urlsafe(_TOKEN_BYTES)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"token": self.token}))
        if sys.platform != "win32":
            path.chmod(0o600)  # pragma: no cover - POSIX-only; Windows has no 0600 mode bits
        return self.token

    @staticmethod
    def _read_persisted_token(path: Path) -> str | None:
        """Return a previously-persisted token if shape-valid, else None."""
        try:
            raw = path.read_text()
        except (FileNotFoundError, OSError):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        token = data.get("token")
        if not isinstance(token, str) or len(token) < _TOKEN_BYTES:
            return None
        return token

    def cleanup(self) -> None:
        """Remove server.json on shutdown and clear the in-memory token."""
        self.token = None
        path = server_json_path()
        path.unlink(missing_ok=True)

    def validate(self, auth_header: str) -> bool:
        """Check whether *auth_header* carries a valid bearer token."""
        if self.token is None:
            return True  # auth disabled (tests)
        if not self.token:
            raise NotAuthorizedException("Server token not initialized")
        return hmac.compare_digest(auth_header, f"Bearer {self.token}")


# Singleton instance: used by AuthMiddleware and the app lifespan.
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
        if handler and is_read_only(handler.fn):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        if session_manager.validate(auth_header):
            await self.app(scope, receive, send)
            return
        raise NotAuthorizedException("Missing or invalid bearer token")
