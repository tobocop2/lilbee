"""Run a short-lived subprocess with a bounded reap.

``subprocess.run``'s timeout path waits indefinitely for the killed child to be
reaped, so a process wedged in uninterruptible I/O (a GPU driver, a stuck
parser) hangs the caller forever. This kills the child's process group and
abandons it after a bounded wait if it will not die, so a wedged child costs a
bounded wait rather than a permanent hang.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
from typing import Any

from lilbee.providers.fleet.child_guard import spawn_bound_child

log = logging.getLogger(__name__)


def run_bounded(
    argv: list[str],
    *,
    timeout_s: float,
    kill_wait_s: float,
    env: dict[str, str] | None = None,
    merge_stderr: bool = False,
    label: str | None = None,
    bind_lifetime: bool = False,
) -> tuple[str, int]:
    """Run *argv*, returning ``(stdout, returncode)``.

    The child runs in its own session so its whole group can be killed. With
    *merge_stderr* stderr is folded into the returned stdout; otherwise it is
    discarded (the caller wants clean stdout, e.g. JSON). The group is SIGKILLed
    and awaited for at most *kill_wait_s* on timeout and on any other abort -- a
    Ctrl-C reaches this process, not the child's own session -- then the
    exception is re-raised, abandoning an unkillable child rather than waiting.

    *bind_lifetime* additionally binds the child to this process, for a child
    that holds a resource nothing can reclaim by record (the device probe holds
    a GPU context and writes no state file). It is off by default because the
    Windows binding leaks a job-object handle per spawn by design, which is
    right for a handful of engines and wrong for a probe sampled every second.
    """
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT if merge_stderr else subprocess.DEVNULL,
        "text": True,
        # A probe writes its pipe in its own encoding; a GPU device name that is
        # not locale-decodable must not raise out of a sampling call.
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
        "start_new_session": os.name == "posix",
    }
    if bind_lifetime:
        # Short-lived, so no death pipe: its watcher would outlive the child.
        proc = spawn_bound_child(argv, death_pipe=False, **popen_kwargs)
    else:
        proc = subprocess.Popen(  # noqa: S603 - argv is trusted: a resolved binary or a fixed literal
            argv, **popen_kwargs
        )
    try:
        stdout, _ = proc.communicate(timeout=timeout_s)
    except BaseException:
        _abandon_group(proc, kill_wait_s, label or argv[0])
        raise
    return stdout or "", proc.returncode


def _abandon_group(proc: subprocess.Popen[str], kill_wait_s: float, label: str) -> None:
    """SIGKILL the child's group; log and give up if it cannot be reaped in time."""
    if os.name == "posix":
        # start_new_session made the child its own group leader.
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows has no process groups to kill
        proc.kill()
    try:
        proc.communicate(timeout=kill_wait_s)
    except subprocess.TimeoutExpired:
        log.warning(
            "%s (pid %d) ignored SIGKILL and was abandoned; it is likely wedged in a driver.",
            label,
            proc.pid,
        )
