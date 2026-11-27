"""Run a lilbee command through the real entry point with a worker left on the loop's pool.

The command starts a worker on its loop's default pool and does not wait for it, as a
cancel leaves a converter caught in a native call. The worker ends after X_WORK_S
seconds, or never with "stuck". With X_HOLD=1 the command then waits for a signal;
else it ends at once: sync with an empty result, analyze as itself.

Usage: python _stuck_worker_entry.py <command> [args...]. It imports nothing from the
test suite, so it runs outside pytest.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import lilbee.app.analyze as analyze
import lilbee.data.ingest as ingest
from lilbee.data.types import SyncResult


def mark(text: str) -> None:
    os.write(1, f"\n{text}\n".encode())


never = threading.Event()
# With X_POOL=kept the worker runs on this pool, which the process holds to its exit.
kept_pool = ThreadPoolExecutor(max_workers=1)


def worker() -> None:
    work = os.environ["X_WORK_S"]
    never.wait(None if work == "stuck" else float(work))
    mark("WORKER DONE")


async def leave_a_worker() -> None:
    if os.environ.get("X_POOL") == "kept":
        kept_pool.submit(worker)
    else:
        asyncio.get_running_loop().run_in_executor(None, worker)
    if os.environ["X_HOLD"] == "1":
        mark("HOLDING")
        await asyncio.sleep(3600)


async def fake_sync(**kwargs: Any) -> SyncResult:
    await leave_a_worker()
    mark("COMMAND DONE")
    return SyncResult()


real_analysis = analyze.run_analysis


async def wrapped_analysis(*args: Any, **kwargs: Any) -> Any:
    await leave_a_worker()
    report = await real_analysis(*args, **kwargs)
    mark("COMMAND DONE")
    return report


def main() -> None:
    atexit.register(mark, "ATEXIT")
    ingest.sync = fake_sync  # type: ignore[assignment]
    analyze.run_analysis = wrapped_analysis  # type: ignore[assignment]
    sys.argv = ["lilbee", *sys.argv[1:]]
    from lilbee.runtime.launcher import main as launch

    launch()


if __name__ == "__main__":
    main()
