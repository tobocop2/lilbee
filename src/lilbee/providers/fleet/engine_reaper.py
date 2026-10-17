"""The engine reaper: a process that ends a shared engine's use once no user holds it.

One runs per engine dir. Users can die in any order and by any means, so the
reaper applies the last-user-out rule after every death, crash included: it
stops the engine, or unloads a kept-warm engine's weights after the idle time.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from lilbee.runtime.engine_lock import (
    build_lock,
    keep_warm_idle_s,
    keep_warm_requested,
    live_users_exist,
    try_hold_reaper_lock,
)

log = logging.getLogger(__name__)

_POLL_S = 1.0
# A starting user probes the reaper lock for an instant; a claim that waits this
# long outlasts any probe, while a live reaper holds the lock for its whole life.
_CLAIM_TIMEOUT_S = 2.0


@dataclass
class _IdleCount:
    """When the last user of a kept-warm engine left, and whether its weights were unloaded."""

    since: float | None = None
    unloaded: bool = False

    def reset(self) -> None:
        self.since = None
        self.unloaded = False


def _watch_is_over(
    engine_dir: Path, reaper_lock: FileLock, idle: _IdleCount, clock: Callable[[], float]
) -> bool:
    """Under the build lock, end the use of an engine nobody holds; True when nothing is left.

    The reaper lock is released before the build lock, so a build that follows
    always finds the dir unwatched and starts a new reaper.
    """
    # Heavy (about 290 ms and 40 MB): loaded only once no user is left, not for the
    # reaper's whole life beside a running engine.
    from lilbee.providers.fleet.swap_manager import engine_record_exists, stop_engine

    with build_lock(engine_dir):
        if live_users_exist(engine_dir):
            idle.reset()
            return False
        recorded = engine_record_exists(engine_dir)
        if recorded and keep_warm_requested(engine_dir):
            _unload_when_idle(engine_dir, idle, clock())
            return False
        if recorded:
            stop_engine(engine_dir)
            log.info("Engine stopped at %s (every user is gone)", engine_dir)
        reaper_lock.release()
        return True


def _unload_when_idle(engine_dir: Path, idle: _IdleCount, now: float) -> None:
    """Unload a kept-warm engine's weights once, after its idle time with no user."""
    # Loaded with the heavy swap_manager import above, once no user is left.
    from lilbee.core.config.defaults import ENGINE_IDLE_TTL_MINUTES_DEFAULT
    from lilbee.providers.fleet.swap_manager import unload_engine_models

    if idle.since is None:
        idle.since = now
    idle_s = keep_warm_idle_s(engine_dir, unset_minutes=ENGINE_IDLE_TTL_MINUTES_DEFAULT)
    if idle.unloaded or now - idle.since < idle_s:
        return
    if unload_engine_models(engine_dir):
        idle.unloaded = True
        log.info("Engine weights unloaded at %s (idle with no user)", engine_dir)


def watch(
    engine_dir: Path,
    reaper_lock: FileLock,
    *,
    poll_s: float = _POLL_S,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll *engine_dir* until its engine is stopped or gone."""
    idle = _IdleCount()
    while True:
        if live_users_exist(engine_dir):
            idle.reset()
        else:
            try:
                if _watch_is_over(engine_dir, reaper_lock, idle, clock):
                    return
            except FileLockTimeout:
                log.info("Engine build lock at %s still held; checking again.", engine_dir)
        time.sleep(poll_s)


def main() -> None:
    """Watch the engine dir named in argv, unless another reaper already does."""
    engine_dir = Path(sys.argv[1])
    reaper_lock = try_hold_reaper_lock(engine_dir, timeout=_CLAIM_TIMEOUT_S)
    if reaper_lock is not None:
        watch(engine_dir, reaper_lock)


if __name__ == "__main__":  # pragma: no cover - process entry glue
    main()
