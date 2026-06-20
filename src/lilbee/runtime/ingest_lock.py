"""Per-process ingest lock registry.

A runtime concurrency primitive shared by the HTTP add-files handler, the
TUI ingest task, and any other surface that wants to serialize concurrent
ingest of the same source file. Lives at the runtime layer so callers in
core/server/cli/tui can all use it without dragging in HTTP-layer code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


class IngestLockRegistry:
    """Per-source ingest locks with a serialized check-and-acquire step.

    The registry lock serializes lock creation and the check-and-acquire
    so concurrent ``/api/add`` calls cannot TOCTOU between
    ``locked()`` and ``acquire()``. One instance is held by ``Services``
    and discarded by ``reset_services()``.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock: asyncio.Lock | None = None

    def _get_registry_lock(self) -> asyncio.Lock:
        if self._registry_lock is None:
            self._registry_lock = asyncio.Lock()
        return self._registry_lock

    def reset(self) -> None:
        """Test hook: clear per-source locks and the registry lock."""
        self._locks.clear()
        self._registry_lock = None

    async def try_acquire(self, name: str) -> asyncio.Lock | None:
        """Acquire the lock for ``name`` or return ``None`` if already held."""
        async with self._get_registry_lock():
            lock = self._locks.get(name)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[name] = lock
            if lock.locked():
                return None
            await lock.acquire()
            return lock

    @staticmethod
    def canonical_source_name(p_str: str) -> str:
        """Match the basename ``copy_files`` writes under ``cfg.documents_dir``."""
        return Path(p_str).name

    async def acquire(self, paths: list[str]) -> tuple[list[tuple[str, asyncio.Lock]], list[str]]:
        """Return ``(acquired, busy)`` partitioning of ``paths`` by lock state."""
        acquired: list[tuple[str, asyncio.Lock]] = []
        busy: list[str] = []
        seen: set[str] = set()
        for p_str in paths:
            name = self.canonical_source_name(p_str)
            if name in seen:
                continue
            seen.add(name)
            lock = await self.try_acquire(name)
            if lock is None:
                busy.append(name)
            else:
                acquired.append((name, lock))
        return acquired, busy

    def release(self, acquired: list[tuple[str, asyncio.Lock]]) -> None:
        """Release every lock in ``acquired`` and evict its registry entry.

        Runs synchronously (no ``await``), so it is atomic with respect to
        ``try_acquire`` on the event loop. ``try_acquire`` only ever acquires a
        free lock, so these per-source locks never accrue waiters; dropping the
        entry once released keeps the registry from growing one lock per distinct
        filename for the whole process lifetime. Safe to call multiple times.
        """
        while acquired:
            name, lock = acquired.pop()
            if lock.locked():
                lock.release()
            if self._locks.get(name) is lock:
                del self._locks[name]
