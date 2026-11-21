"""Signal setup for test child processes started under a nohup launch."""

from __future__ import annotations

import os
import signal
from collections.abc import Callable


def child_preexec(then: Callable[[], None] | None = None) -> Callable[[], None] | None:
    """Return a ``preexec_fn`` that gives the child the default SIGHUP and SIGINT, then runs *then*.

    A nohup launch leaves SIGHUP ignored, a background job leaves SIGINT ignored,
    and a child inherits both. Windows has no ``preexec_fn``, so the result there
    is *then* alone.
    """
    if os.name != "posix":
        return then

    def setup() -> None:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if then is not None:
            then()

    return setup
