"""Thin CLI entry point — shows splash animation while heavy deps load.

This module imports only ``_splash`` (which uses only stdlib), forks the
animation process, then performs the heavy ``from lilbee.cli import app``
import while the bee animates on stderr.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import suppress

_READY_FILE = "lilbee-splash-ready"


def _cleanup_ready_file(path: str) -> None:
    """Remove the ready file if it exists."""
    with suppress(OSError):
        os.remove(path)


def main() -> None:
    """Entry point for the ``lilbee`` console script."""
    args = sys.argv[1:]
    is_interactive = not args or args[0] in ("chat", "")

    if not is_interactive:
        from lilbee.cli import app

        app()
        return

    from lilbee.splash import start, stop

    ready_file = os.path.join(tempfile.gettempdir(), _READY_FILE)
    pid = start(ready_file=ready_file)

    try:
        from lilbee.cli import app
    except BaseException:
        stop(pid)
        _cleanup_ready_file(ready_file)
        raise

    try:
        app()
    except KeyboardInterrupt:
        sys.stderr.write("\033[?25h")
        sys.stderr.flush()
        raise SystemExit(130) from None
    finally:
        stop(pid)
        _cleanup_ready_file(ready_file)
        sys.stderr.write("\033[2J\033[H\033[?25h")
        sys.stderr.flush()
