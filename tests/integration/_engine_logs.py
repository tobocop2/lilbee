"""Engine log tails for a failed integration test.

A test that times out on a CI runner shows only the test's own output. The
llama-swap and llama-server logs say whether the server stalled or the run was
slow, so a failed report carries their tails.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.runtime.engine_lock import machine_engine_dir

TAIL_LINES = 200
# The newest files only: a run that fails many tests would otherwise print every
# log of every earlier test under each failure.
MAX_FILES = 8
SECTION_TITLE = "Engine logs"
# The names the fleet gives the swap process log and each engine log.
_LOG_GLOBS = ("**/logs/llama-swap-*.log", "**/logs/engine-*.log")
# The reaper's log matches the engine pattern and says nothing about a stalled
# engine, so it must not take one of the newest slots.
_REAPER_LOG = "engine-reaper.log"
# A stalled test prints its tails this share of its timeout in, before the timeout ends it.
STALL_SHARE = 0.75


def log_roots(basetemp: Path | None) -> list[Path]:
    """Every folder an integration test can start an engine under.

    The fleet writes to the machine engine slot, which LILBEE_ENGINE_DIR moves; a
    test that makes its own data folder writes under that folder or the pytest
    temp root.
    """
    roots = [machine_engine_dir(), cfg.data_dir]
    if data_env := os.environ.get("LILBEE_DATA"):
        roots.append(Path(data_env))
    if basetemp is not None:
        roots.append(basetemp)
    return roots


def emit(text: str | None) -> None:
    """Write *text* to the real stderr at once, past pytest's capture."""
    if text:
        sys.__stderr__.write(f"\n{SECTION_TITLE}\n{text}\n")
        sys.__stderr__.flush()


def find_logs(roots: Iterable[Path]) -> list[Path]:
    """The newest engine log files under *roots*, newest first."""
    found: dict[Path, float] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in _LOG_GLOBS:
            for path in root.glob(pattern):
                if path.is_file() and path.name != _REAPER_LOG:
                    found[path.resolve()] = path.stat().st_mtime
    newest = sorted(found, key=lambda path: found[path], reverse=True)
    return newest[:MAX_FILES]


def tail(path: Path, lines: int = TAIL_LINES) -> str:
    """The last *lines* lines of *path*, with undecodable bytes replaced."""
    with path.open("rb") as handle:
        last = deque(handle, maxlen=lines)
    return b"".join(last).decode("utf-8", errors="replace")


def failure_section(roots: Iterable[Path]) -> str | None:
    """The tail of every engine log under *roots*, or None when there is none."""
    parts = [f"--- {path} (last {TAIL_LINES} lines) ---\n{tail(path)}" for path in find_logs(roots)]
    return "\n".join(parts) if parts else None
