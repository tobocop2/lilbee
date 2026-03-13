"""Litestar application factory for the lilbee HTTP API."""

from litestar import Litestar

from lilbee.server.handlers import add_files


def create_app() -> Litestar:
    """Build and return the Litestar ASGI application."""
    return Litestar(route_handlers=[add_files])
