"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

from lilbee.cli.sync import shutdown_executor
from lilbee.services import reset_services


def run_tui(*, auto_sync: bool = False, initial_view: str | None = None) -> None:
    """Launch the full-screen Textual TUI.

    *initial_view* deep-links to a named view (e.g. ``"Catalog"``) after
    the default chat screen is mounted. Used by ``lilbee model browse``.
    """
    import os

    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=auto_sync, initial_view=initial_view)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            shutdown_executor()
            reset_services()
        except (KeyboardInterrupt, Exception):
            # Rapid Ctrl+C during shutdown — force exit immediately
            os._exit(1)
