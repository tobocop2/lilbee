"""Thin CLI entry point — shows splash animation while heavy deps load.

This module imports only ``_splash`` (which uses only stdlib), forks the
animation process, then performs the heavy ``from lilbee.cli import app``
import while the bee animates on stderr.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``lilbee`` console script."""
    from lilbee._splash import start, stop

    pid = start()
    try:
        from lilbee.cli import app
    except BaseException:
        stop(pid)
        raise
    stop(pid)

    try:
        app()
    except KeyboardInterrupt:
        sys.stderr.write("\033[?25h")
        sys.stderr.flush()
        raise SystemExit(130) from None
