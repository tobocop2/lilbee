"""Signal setup for test child processes started under a nohup launch, and Ctrl+C from a run."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable

from lilbee.runtime.cancellation import CancelSignal

# Seconds a press from a run's thread gets to reach the main thread's handler.
_PRESS_TAKEN_S = 5.0


def press_ctrl_c_in_a_run(cancel: CancelSignal | None = None) -> None:
    """Press Ctrl+C from inside a cancellable run, as a terminal does.

    The run's loop thread blocks SIGINT, so the press goes to the process and the
    main thread's handler takes it. With *cancel*, return once the press set it.
    """
    os.kill(os.getpid(), signal.SIGINT)
    if cancel is None:
        return
    deadline = time.monotonic() + _PRESS_TAKEN_S
    while not cancel.is_set():
        assert time.monotonic() < deadline, "the Ctrl+C did not reach the main thread"
        time.sleep(0.001)


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
