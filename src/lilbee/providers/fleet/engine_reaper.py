"""The engine reaper: a process that stops a shared engine once no user holds it.

One runs per engine dir. Users can die in any order and by any means, so the
reaper applies the last-user-out rule after every death, crash included.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.runtime.engine_lock import (
    build_lock,
    keep_warm_requested,
    live_users_exist,
    try_hold_reaper_lock,
)

log = logging.getLogger(__name__)

_POLL_S = 1.0


def _watch_is_over(engine_dir: Path, reaper_lock: FileLock) -> bool:
    """Under the build lock, stop an engine nobody holds; True when nothing is left to watch.

    The reaper lock is released before the build lock, so a build that follows
    always finds the dir unwatched and starts a new reaper.
    """
    # Heavy (about 290 ms and 40 MB): loaded only once no user is left, not for the
    # reaper's whole life beside a running engine.
    from lilbee.providers.fleet.swap_manager import engine_record_exists, stop_engine

    with build_lock(engine_dir):
        recorded = engine_record_exists(engine_dir)
        if live_users_exist(engine_dir) or (recorded and keep_warm_requested(engine_dir)):
            return False
        if recorded:
            stop_engine(engine_dir)
            log.info("Engine stopped at %s (every user is gone)", engine_dir)
        reaper_lock.release()
        return True


def watch(engine_dir: Path, reaper_lock: FileLock, *, poll_s: float = _POLL_S) -> None:
    """Poll *engine_dir* until its engine is stopped or gone."""
    while True:
        if not live_users_exist(engine_dir):
            try:
                if _watch_is_over(engine_dir, reaper_lock):
                    return
            except FileLockTimeout:
                log.info("Engine build lock at %s still held; checking again.", engine_dir)
        time.sleep(poll_s)


def main() -> None:
    """Watch the engine dir named in argv, unless another reaper already does."""
    engine_dir = Path(sys.argv[1])
    reaper_lock = try_hold_reaper_lock(engine_dir)
    if reaper_lock is not None:
        watch(engine_dir, reaper_lock)


if __name__ == "__main__":  # pragma: no cover - process entry glue
    main()
