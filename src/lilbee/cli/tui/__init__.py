"""Textual TUI for lilbee — full-screen interactive knowledge base."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def run_tui(*, auto_sync: bool = False) -> None:
    """Launch the full-screen Textual TUI."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=auto_sync)
    app.run()
