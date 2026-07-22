"""Dump every thread's stack when a test wedges, then kill the worker.

pytest-timeout is already configured (``timeout = 60``), but on Windows it runs
its ``thread`` method: a ``threading.Timer`` whose callback is Python. If a test
wedges inside a native call that holds the GIL, no Python bytecode runs, the
timer callback never fires, and the worker hangs silently until the job's
``timeout-minutes`` kills it with no traceback (seen on CI, Windows py3.13).

``faulthandler.dump_traceback_later`` is the stdlib answer to exactly that case:
its watchdog is a C thread that walks the interpreter's thread states and writes
their stacks WITHOUT acquiring the GIL, so it fires on a GIL-frozen worker where
pytest-timeout cannot. ``exit=True`` then ``_exit``\\s the wedged worker so the
run fails fast instead of burning the job budget.

Opt-in via ``LILBEE_TEST_HANG_DUMP_S`` (seconds); unset or ``0`` is a no-op, so
local runs are untouched. Set it above pytest-timeout's value in CI so this is
only ever the backstop for the frozen case, never a pre-empt of a merely slow
test. The dump is written to ``LILBEE_TEST_HANG_DUMP_DIR/hang-<workerid>.txt``
(default: the pytest rootdir) so a CI step can surface it after the killed step;
the frozen worker cannot print it itself.
"""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path

import pytest

_ENV_SECONDS = "LILBEE_TEST_HANG_DUMP_S"
_ENV_DIR = "LILBEE_TEST_HANG_DUMP_DIR"

# The open dump-file handle. faulthandler writes to its raw fd from a C thread,
# so it must outlive every test; keep a module reference so it is not closed.
_dump_handle = None
_timeout_s = 0.0


def _worker_id(config: pytest.Config) -> str:
    """xdist worker id (``gw0``), or ``master`` when running single-process."""
    return getattr(config, "workerinput", {}).get("workerid", "master")


def pytest_configure(config: pytest.Config) -> None:
    global _dump_handle, _timeout_s
    raw = os.environ.get(_ENV_SECONDS, "").strip()
    _timeout_s = float(raw) if raw else 0.0
    if _timeout_s <= 0:
        return
    dump_dir = Path(os.environ.get(_ENV_DIR) or config.rootpath)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # One file per worker: xdist workers share a rootdir but not a process.
    _dump_handle = (dump_dir / f"hang-{_worker_id(config)}.txt").open("w")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if _dump_handle is None:
        return
    # Record which test armed the timer, so a dump names its trigger even though
    # faulthandler itself only prints stacks. Rewinds each test (no accumulation).
    _dump_handle.seek(0)
    _dump_handle.truncate()
    _dump_handle.write(f"hang watchdog armed for {item.nodeid} ({_timeout_s}s)\n")
    _dump_handle.flush()
    # Re-arms (resets) the single C-thread timer for this test. dump_traceback_later
    # dumps every thread by default, which is what we want on a wedged worker.
    faulthandler.dump_traceback_later(_timeout_s, file=_dump_handle, exit=True)


def pytest_runtest_teardown(item: pytest.Item) -> None:
    if _dump_handle is None:
        return
    faulthandler.cancel_dump_traceback_later()
    # The test finished, so its armed dump is stale; clear it.
    _dump_handle.seek(0)
    _dump_handle.truncate()
    _dump_handle.flush()


def pytest_unconfigure(config: pytest.Config) -> None:
    global _dump_handle
    if _dump_handle is not None:
        faulthandler.cancel_dump_traceback_later()
        _dump_handle.close()
        _dump_handle = None
