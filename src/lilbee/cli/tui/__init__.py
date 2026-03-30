"""Textual TUI for lilbee -- full-screen interactive knowledge base."""

from __future__ import annotations

from lilbee.cli.sync import shutdown_executor
from lilbee.services import reset_services


def run_tui(*, auto_sync: bool = False) -> None:
    """Launch the full-screen Textual TUI."""
    import os
    import signal

    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=auto_sync)
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
            os.kill(os.getpid(), signal.SIGKILL)
