"""Helper that reports whether a pipe's write end is closed."""

from __future__ import annotations

import os
import threading

_EOF_TIMEOUT = 5.0


def pipe_reaches_eof(read_fd: int) -> bool:
    """Return True when reading `read_fd` hits end of file, False if it still blocks."""
    done = threading.Event()

    def drain() -> None:
        while os.read(read_fd, 16):
            pass
        done.set()

    threading.Thread(target=drain, daemon=True).start()
    return done.wait(_EOF_TIMEOUT)
