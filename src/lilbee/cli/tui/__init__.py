"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

from lilbee.cli.sync import shutdown_executor
from lilbee.services import reset_services


def run_tui(*, auto_sync: bool = False) -> None:
    """Launch the full-screen Textual TUI."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=auto_sync)
    try:
        app.run()
    finally:
        shutdown_executor()
        reset_services()
