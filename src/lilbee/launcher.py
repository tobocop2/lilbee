"""Thin CLI entry point — shows splash animation while heavy deps load.

This module imports only ``splash`` (which uses only stdlib + subprocess),
launches the animation process, then performs the heavy
``from lilbee.cli import app`` import while the bee animates on stderr.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``lilbee`` console script."""
    args = sys.argv[1:]
    is_interactive = not args or args[0] in ("chat", "")

    if not is_interactive:
        from lilbee.cli import app

        app()
        return

    from lilbee.splash import start, stop

    handle = start()

    try:
        from lilbee.cli import app
    finally:
        # Stop the splash BEFORE the TUI takes over the terminal. Otherwise
        # the subprocess's final writes (erase + home) land on Textual's
        # alt-screen and leave the cursor visible, producing stray artifacts
        # during background tasks and a stuck glyph in the top-left corner
        # of every screen (BEE-73k, BEE-jmj). stop() is synchronous and
        # waits for the child to exit, so by the time app() runs no more
        # splash writes can reach the terminal.
        stop(handle)

    try:
        app()
    except KeyboardInterrupt:
        sys.stderr.write("\033[?25h")
        sys.stderr.flush()
        raise SystemExit(130) from None
