"""Hold the records lock of a library and say when a take of it waits."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

from filelock import FileLock

from lilbee.data.ingest import skip_marker

_HANG_GUARD_S = 60.0
"""A take that never gets its cancel fails after this guard; no test waits for it."""


def files_under(root: Path) -> dict[str, bytes]:
    """Every file under *root* with its bytes, the held lock file aside."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock"
    }


class Outcome:
    """What a call wrapped by :func:`recorded` raised, once it has ended."""

    def __init__(self) -> None:
        self.ended = threading.Event()
        self.raised: BaseException | None = None


def recorded(real: Callable[..., Any]) -> tuple[Callable[..., Any], Outcome]:
    """Wrap *real* so a test can wait for its end on another thread and read what it raised."""
    outcome = Outcome()

    def _call(*args: Any, **kwargs: Any) -> Any:
        try:
            return real(*args, **kwargs)
        except BaseException as exc:
            outcome.raised = exc
            raise
        finally:
            outcome.ended.set()

    return _call, outcome


@contextlib.contextmanager
def records_lock_held_elsewhere(
    data_root: Path, on_wait: Callable[[], None]
) -> Iterator[Callable[[], None]]:
    """Hold the records lock the way another process does; a take that fails calls *on_wait* once.

    *on_wait* runs on the thread that waits, after its first failed take, whether
    or not the caller passed a cancel to the lock. The block gets a call that
    frees the lock before the block ends.
    """
    waited: list[bool] = []

    class _ReportsItsWait(FileLock):  # type: ignore[valid-type, misc]
        def acquire(self, *args: Any, cancel_check: Any = None, **kwargs: Any) -> Any:
            def _check() -> bool:
                if not waited:
                    waited.append(True)
                    on_wait()
                return cancel_check is not None and cancel_check()

            return super().acquire(*args, cancel_check=_check, **kwargs)

    holder = FileLock(str(data_root / skip_marker.SKIP_MARKER_FILENAME) + ".lock")
    holder.acquire()
    try:
        with (
            mock.patch.object(skip_marker, "FileLock", _ReportsItsWait),
            mock.patch.object(skip_marker, "_RECORDS_LOCK_TIMEOUT_S", _HANG_GUARD_S),
        ):
            yield holder.release
    finally:
        holder.release()
    assert waited, "no take of the records lock waited"
