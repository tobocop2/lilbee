"""Session token auth middleware with decorator-based read-only marking."""

from __future__ import annotations

import hmac
import json
import logging
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from litestar.exceptions import NotAuthorizedException
from litestar.types import ASGIApp, Receive, Scope, Send

from lilbee.core.config import cfg
from lilbee.core.security import harden_private_file, write_private_text

log = logging.getLogger(__name__)

_TOKEN_BYTES = 32

# Character floor for a persisted token. secrets.token_urlsafe(n) base64url-encodes
# n bytes, so the entropy count is not a length: reusing _TOKEN_BYTES here accepted
# tokens roughly a quarter weaker than the ones this module mints.
_MIN_TOKEN_CHARS = len(secrets.token_urlsafe(_TOKEN_BYTES))

F = TypeVar("F", bound=Callable[..., Any])


# Route handlers AuthMiddleware skips, registered at import by the decorator
# below. A module-level set rather than an attribute on the function object,
# which mypy cannot see and would put a # type: ignore on every check.
_SELF_AUTHENTICATING_HANDLERS: set[Callable[..., Any]] = set()


def auth_checked_in_handler(fn: F) -> F:
    """Mark a route whose token check runs inside the handler, not in middleware.

    Not an exemption: the route must still reject an unauthenticated caller
    itself. Only ``/v1/*`` uses this, to answer a bad token with the OpenAI
    error envelope instead of Litestar's 401 shape.
    ``test_every_route_is_authenticated`` holds the line.

    Must sit *below* the route decorator so it receives the raw function, which
    is what ``AuthMiddleware`` looks up via ``handler.fn``; stacked the other
    way the lookup misses. Enforced below.
    """
    if hasattr(fn, "fn"):
        raise TypeError(
            "@auth_checked_in_handler must be applied below the route decorator, "
            "so it sees the function rather than the route handler."
        )
    _SELF_AUTHENTICATING_HANDLERS.add(fn)
    return fn


def authenticates_itself(fn: Callable[..., Any]) -> bool:
    """True iff *fn* was decorated with :func:`auth_checked_in_handler`."""
    return fn in _SELF_AUTHENTICATING_HANDLERS


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
        # False until load_or_generate() or disable() runs. validate() fails
        # closed while unset so an app served without its lifespan (or after
        # cleanup) never silently accepts unauthenticated mutating requests.
        self._initialized: bool = False

    def load_or_generate(self) -> str:
        """Return the persisted token if shape-valid; generate a new one otherwise."""
        path = server_json_path()
        existing = self._read_persisted_token(path)
        if existing is not None:
            # The token is reused indefinitely and the file can arrive
            # world-readable (backup, older release), so narrow on every load.
            harden_private_file(path)
            self.token = existing
            self._initialized = True
            return existing
        self.token = secrets.token_urlsafe(_TOKEN_BYTES)
        write_private_text(path, json.dumps({"token": self.token}))
        self._initialized = True
        return self.token

    def disable(self) -> None:
        """Explicitly turn auth off (test harness / embedded read-only use).

        Distinct from the uninitialized state: validate() accepts any request
        once disabled, but denies until either this or load_or_generate() runs.
        """
        self.token = None
        self._initialized = True

    @staticmethod
    def _read_persisted_token(path: Path) -> str | None:
        """Return a previously-persisted token if shape-valid, else None.

        Total by design: every way the file can be unusable returns None so the
        caller mints a fresh token. A corrupt server.json must never be the
        reason the server refuses to boot, since nothing would point the user at
        the file to delete.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        except UnicodeDecodeError:
            # Not an OSError: a truncated write or a file clobbered by another
            # tool leaves bytes that are not valid UTF-8.
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        token = data.get("token")
        if not isinstance(token, str) or len(token) < _MIN_TOKEN_CHARS:
            return None
        return token

    def cleanup(self) -> None:
        """Remove server.json on shutdown and reset to the uninitialized state."""
        self.token = None
        self._initialized = False
        path = server_json_path()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A still-open handle on Windows makes unlink raise; the token is
            # already invalidated above and the file is rewritten on next boot.
            log.debug("Could not remove %s at shutdown.", path, exc_info=True)

    def validate(self, auth_header: str) -> bool:
        """Check whether *auth_header* carries a valid bearer token.

        Fails closed until initialized: a request reaching auth before the
        lifespan ran (or after cleanup) is denied rather than allowed.
        """
        if not self._initialized:
            raise NotAuthorizedException("Server authentication is not initialized")
        if self.token is None:
            return True  # auth explicitly disabled via disable()
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
        if handler and authenticates_itself(handler.fn):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        if session_manager.validate(auth_header):
            await self.app(scope, receive, send)
            return
        raise NotAuthorizedException("Missing or invalid bearer token")
