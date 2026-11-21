"""Closing an add stream, and what an upload hands to its sync and reports back."""

import asyncio
import threading
from unittest import mock

import pytest

from lilbee.data.types import SyncResult
from lilbee.server.handlers import add_uploads_stream
from lilbee.server.handlers.ingest import _SYNC_STARTER, _ingest_stream
from lilbee.server.models import AddSummary
from tests.server.conftest import parse_sse_events
from tests.server.test_handlers import (  # noqa: F401 (autouse fixtures)
    isolated_env,
    mock_svc,
    reset_ingest_locks,
)

_PROGRESS_EVENT = "event: file_start\ndata: {}\n\n"


async def _close_a_running_stream(run) -> None:
    """Open an ingest stream whose run is going, read its first event, and close it."""
    gen = _ingest_stream([("note.md", "note")], run, "Test stream")
    await gen.__anext__()
    await gen.aclose()


async def test_closing_the_stream_during_a_run_cancels_the_run_and_raises_nothing() -> None:
    cancelled = asyncio.Event()

    async def run(_locked, sse) -> AddSummary:
        sse.queue.put_nowait(_PROGRESS_EVENT)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finally:
            sse.queue.put_nowait(None)
        return AddSummary(copied=[], errors=[])

    await _close_a_running_stream(run)

    assert cancelled.is_set()


async def test_a_run_that_fails_while_the_stream_closes_does_not_escape_the_close() -> None:
    cleaned_up = asyncio.Event()

    async def run(_locked, sse) -> AddSummary:
        sse.queue.put_nowait(_PROGRESS_EVENT)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaned_up.set()
            raise RuntimeError("the cleanup failed") from None
        finally:
            sse.queue.put_nowait(None)
        return AddSummary(copied=[], errors=[])

    await _close_a_running_stream(run)

    assert cleaned_up.is_set()


class _SyncSpy:
    """A stand-in for the sync that records its arguments and ends as its script says."""

    def __init__(self, result: SyncResult | None = None) -> None:
        self.kwargs: dict[str, object] = {}
        self.result = result
        self.cancelled = asyncio.Event()

    async def __call__(self, **kwargs: object) -> SyncResult:
        self.kwargs = dict(kwargs)
        on_progress = kwargs["on_progress"]
        on_progress("file_start", {})  # type: ignore[operator]
        if self.result is None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        return self.result  # type: ignore[return-value]


async def test_the_upload_sync_gets_the_streams_cancel_signal_and_its_starter() -> None:
    spy = _SyncSpy()
    gen = add_uploads_stream([("note.md", b"# A note\n")])
    with mock.patch("lilbee.data.ingest.sync", new=spy):
        await gen.__anext__()
        cancel = spy.kwargs["cancel"]
        assert isinstance(cancel, threading.Event)
        assert not cancel.is_set()
        await gen.aclose()
        # The inner stream unwinds when the loop finalizes it, which cancels the sync.
        await asyncio.wait_for(spy.cancelled.wait(), 30)

    assert cancel.is_set()
    assert spy.kwargs["started_by"] == _SYNC_STARTER
    assert spy.kwargs["quiet"] is True


@pytest.mark.parametrize("name", ["note.md", "docs/note.md"])
async def test_the_upload_done_summary_carries_the_sync_counts(name: str) -> None:
    spy = _SyncSpy(SyncResult(added=[name], updated=["old.md"], unchanged=3, failed=["bad.md"]))

    with mock.patch("lilbee.data.ingest.sync", new=spy):
        text = "".join([frame async for frame in add_uploads_stream([(name, b"# A note\n")])])

    done = [data for event, data in parse_sse_events(text.encode()) if event == "done"]
    assert len(done) == 1
    assert done[0]["copied"] == [name]
    assert done[0]["sync"]["added"] == [name]
    assert done[0]["sync"]["updated"] == ["old.md"]
    assert done[0]["sync"]["unchanged"] == 3
    assert done[0]["sync"]["failed"] == ["bad.md"]
