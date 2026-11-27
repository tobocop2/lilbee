"""A cancel of an add ends its stream with one ``done`` event, whenever it arrives.

Rule 7 of the design: after a cancel the add stream ends with one ``done`` event that carries
``cancelled``. Rule 8: a cancel answers 404 only for an id that does not run. The two tests
come from the review of the cancel route.
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

from lilbee.app.ingest import RegisterResult
from lilbee.app.services import get_services
from lilbee.data.types import SyncResult
from lilbee.runtime.ingest_lock import IngestLockRegistry
from lilbee.server.handlers import add_files_stream, cancel_add
from tests.server.conftest import parse_sse_events
from tests.server.test_handlers import (  # noqa: F401 (autouse fixtures)
    isolated_env,
    mock_svc,
    reset_ingest_locks,
)


async def test_a_cancel_during_a_registration_that_reaches_no_corpus_ends_with_done(tmp_path):
    src = tmp_path / "taken.txt"
    src.write_text("contents", encoding="utf-8")
    in_registration = threading.Event()
    release = threading.Event()

    def _registering(_paths, *, force):
        in_registration.set()
        release.wait(20)
        return RegisterResult(name_taken=["taken.txt"])

    async def _cancel_during_registration() -> bool:
        await asyncio.to_thread(in_registration.wait, 20)
        try:
            return cancel_add("r-1")
        finally:
            release.set()

    with mock.patch("lilbee.server.handlers.ingest.register_sources", _registering):
        canceller = asyncio.create_task(_cancel_during_registration())
        frames = [f async for f in add_files_stream([str(src)], add_id="r-1")]

    assert await canceller is True
    events = parse_sse_events("".join(frames).encode())
    done = [data for event, data in events if event == "done"]
    assert len(done) == 1, [event for event, _data in events]
    assert done[0]["cancelled"] is True


async def test_a_cancel_while_the_add_waits_for_its_locks_reaches_it(tmp_path):
    src = tmp_path / "wait.txt"
    src.write_text("contents", encoding="utf-8")
    registry_lock = get_services().ingest_lock_registry._get_registry_lock()
    await registry_lock.acquire()
    waiting = asyncio.Event()
    acquire = IngestLockRegistry.acquire

    async def _acquire(registry, names):
        waiting.set()
        return await acquire(registry, names)

    async def _sync(**kwargs) -> SyncResult:
        if kwargs["cancel"].is_set():
            raise asyncio.CancelledError
        return SyncResult()

    stream = add_files_stream([str(src)], add_id="w-1")
    with (
        mock.patch.object(IngestLockRegistry, "acquire", _acquire),
        mock.patch("lilbee.data.ingest.sync", new=_sync),
    ):
        first = asyncio.ensure_future(anext(stream))
        try:
            await waiting.wait()
            assert registry_lock.locked() and not first.done(), "the add never waited"
            found = cancel_add("w-1")
        finally:
            registry_lock.release()
        frames = [await first] + [f async for f in stream]

    events = parse_sse_events("".join(frames).encode())
    done = [data for event, data in events if event == "done"]
    assert found is True, "POST /api/add/cancel answers 404 for an add the server took"
    assert len(done) == 1 and done[0]["cancelled"] is True, done
