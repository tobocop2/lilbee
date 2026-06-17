"""Thin CLI entry point: shows splash animation while heavy deps load.

This module imports only ``splash`` (which uses only stdlib + subprocess),
launches the animation process, then performs the heavy
``from lilbee.cli import app`` import while the bee animates on stderr.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

# Shells fetch tab completions by running the console script with
# ``_<PROG>_COMPLETE=...`` set (and an empty argv); click answers those itself.
_COMPLETION_ENV_SUFFIX = "_COMPLETE"


def _shell_completion_active() -> bool:
    """True when this process was spawned by a shell's tab-completion."""
    return any(
        name.startswith("_") and name.endswith(_COMPLETION_ENV_SUFFIX) for name in os.environ
    )


def _force_utf8_stdio() -> None:
    """Make stdio UTF-8 so a no-locale (GUI-spawned) launch can't crash on non-ASCII output."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            # Swallow on an already-detached stream; non-TextIOWrapper streams
            # (StringIO redirects, capture shims) need no reconfiguration.
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def main() -> None:
    """Entry point for the ``lilbee`` console script."""
    _force_utf8_stdio()
    args = sys.argv[1:]
    is_interactive = (not args or args[0] in ("chat", "")) and not _shell_completion_active()

    if not is_interactive:
        from lilbee.cli import app

        app()
        return

    from lilbee.runtime.splash import start, stop

    handle = start()

    try:
        from lilbee.cli import app
    except BaseException:
        stop(handle)
        raise
    else:
        # Stop the splash BEFORE the TUI takes over the terminal so the
        # subprocess's final writes don't land on Textual's alt-screen.
        stop(handle)

    try:
        app()
    except KeyboardInterrupt:
        sys.stderr.write("\033[?25h")
        sys.stderr.flush()
        raise SystemExit(130) from None
