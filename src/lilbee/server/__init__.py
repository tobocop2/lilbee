"""lilbee HTTP server — framework-agnostic handlers + Litestar adapter."""


def create_app():  # type: ignore[no-untyped-def]
    """Create the HTTP server app (lazy import to avoid loading litestar at startup)."""
    from lilbee.server.litestar_app import create_app as _create

    return _create()
