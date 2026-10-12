"""Hold the records lock of a library and say when a take of it waits."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

from filelock import FileLock

from lilbee.data.ingest import skip_marker

_WAIT_WITHOUT_A_CANCEL_S = 2.0
"""A take that gets no cancel fails after this wait, not after the product's."""


def files_under(root: Path) -> dict[str, bytes]:
    """Every file under *root* with its bytes, the held lock file aside."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".lock"
    }


@contextlib.contextmanager
def records_lock_held_elsewhere(data_root: Path, on_wait: Callable[[], None]) -> Iterator[None]:
    """Hold the records lock the way another process does; a take that fails calls *on_wait* once.

    *on_wait* runs on the thread that waits, after its first failed take, whether
    or not the caller passed a cancel to the lock.
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
            mock.patch.object(skip_marker, "_RECORDS_LOCK_TIMEOUT_S", _WAIT_WITHOUT_A_CANCEL_S),
        ):
            yield
    finally:
        holder.release()
    assert waited, "no take of the records lock waited"
