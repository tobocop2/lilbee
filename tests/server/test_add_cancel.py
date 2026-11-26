"""POST /api/add/cancel, the add_id of the add routes, and what a cancel by the user undoes."""

from __future__ import annotations

import asyncio
import gc
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.ingest import RegisterResult
from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.data.types import SyncResult
from lilbee.runtime.add_registry import AddIdRunningError
from lilbee.server.handlers import SseStream, cancel_add
from lilbee.server.handlers.ingest import _ingest_stream, _run_add, _run_upload
from lilbee.server.models import AddSummary
from tests.server.conftest import parse_sse_events
from tests.server.test_handlers import (  # noqa: F401 (autouse fixtures)
    _auth_headers,
    isolated_env,
    mock_svc,
    reset_ingest_locks,
)


class _Stub:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def _app():
    from lilbee.server.app import create_app

    return create_app()


@contextmanager
def _held(add_id: str, stop) -> Iterator[None]:
    """Hold *add_id* in the registry of the services, with *stop*, for the block."""
    release = get_services().add_registry.reserve(add_id, stop)
    try:
        yield
    finally:
        release()


def _is_held(add_id: str) -> bool:
    return add_id in get_services().add_registry._stops


def _stop_as_user(sse: SseStream) -> None:
    """What a cancel by the user leaves on *sse*; the run's task is not cancelled here."""
    sse.stopped_by_user = True
    sse.cancel.set()


def _stopping_sync(sse: SseStream, *, error: BaseException | None = None):
    """A sync the user stops while it runs; it raises as the cancel of its task does."""

    async def _sync(**_kwargs) -> SyncResult:
        _stop_as_user(sse)
        if error is not None:
            raise error
        raise asyncio.CancelledError

    return _sync


class TestTheCancelRoute:
    async def test_a_cancel_stops_the_add_that_runs_under_the_id(self):
        run = _Stub()
        with _held("a-1", run.stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add/cancel", json={"add_id": "a-1"}, headers=_auth_headers()
                )
        assert resp.status_code == 200
        assert resp.json() == {"add_id": "a-1", "cancelling": True}
        assert run.stops == 1

    async def test_a_cancel_of_an_id_that_does_not_run_answers_404(self):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/cancel", json={"add_id": "gone"}, headers=_auth_headers()
            )
        assert resp.status_code == 404
        assert "gone" in resp.json()["detail"]

    @pytest.mark.parametrize("add_id", ["", "has space", "a/b", "x" * 65])
    async def test_a_cancel_with_a_bad_id_answers_400(self, add_id):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/cancel", json={"add_id": add_id}, headers=_auth_headers()
            )
        assert resp.status_code == 400


class TestTheAddIdOfTheAddRoutes:
    async def test_an_add_with_a_bad_id_answers_400(self, tmp_path):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(tmp_path / "a.txt")], "add_id": "bad id"},
                headers=_auth_headers(),
            )
        assert resp.status_code == 400

    async def test_an_upload_with_a_bad_id_answers_400(self):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files={"data": ("a.md", b"# a\n")},
                params={"add_id": "x" * 65},
                headers=_auth_headers(),
            )
        assert resp.status_code == 400
        assert not (cfg.documents_dir / "a.md").exists()

    async def test_an_add_with_an_id_that_runs_answers_409(self, tmp_path):
        with _held("busy", _Stub().stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add",
                    json={"paths": [str(tmp_path / "a.txt")], "add_id": "busy"},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 409
        assert "busy" in resp.json()["detail"]

    async def test_an_upload_with_an_id_that_runs_answers_409_and_writes_nothing(self):
        with _held("busy", _Stub().stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add/upload",
                    files={"data": ("a.md", b"# a\n")},
                    params={"add_id": "busy"},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 409
        assert not (cfg.documents_dir / "a.md").exists()

    async def test_the_schema_publishes_the_answers_of_the_cancel(self):
        async with AsyncTestClient(_app()) as client:
            paths = (await client.get("/schema/openapi.json", headers=_auth_headers())).json()[
                "paths"
            ]
        assert {"200", "400", "404"} <= set(paths["/api/add/cancel"]["post"]["responses"])
        assert "409" in paths["/api/add"]["post"]["responses"]
        assert "409" in paths["/api/add/upload"]["post"]["responses"]
        params = {p["name"] for p in paths["/api/add/upload"]["post"]["parameters"]}
        assert "add_id" in params


class TestTheStreamHoldsTheId:
    async def test_the_id_is_held_from_the_open_of_the_stream_until_the_run_ends(self):
        seen: list[bool] = []

        async def run(_locked, sse) -> AddSummary:
            seen.append(_is_held("s-1"))
            sse.queue.put_nowait(None)
            return AddSummary(copied=[], errors=[])

        stream = _ingest_stream([("a.md", "a")], run, "Test stream", "s-1")
        assert _is_held("s-1")
        frames = [f async for f in stream]

        assert seen == [True]
        assert not _is_held("s-1")
        assert [event for event, _data in parse_sse_events("".join(frames).encode())] == ["done"]

    async def test_a_stream_whose_id_another_add_holds_is_refused_before_it_opens(self):
        run = mock.AsyncMock()
        with _held("s-1", _Stub().stop), pytest.raises(AddIdRunningError, match="'s-1'"):
            _ingest_stream([("a.md", "a")], run, "Test stream", "s-1")
        run.assert_not_called()
        assert get_services().ingest_lock_registry._locks == {}

    async def test_a_stream_whose_sources_are_all_busy_frees_its_id(self):
        locks = get_services().ingest_lock_registry
        held = await locks.try_acquire("a.md")
        assert held is not None
        run = mock.AsyncMock()
        # The test keeps the stream, so its collection cannot be what frees the id.
        stream = _ingest_stream([("a.md", "a")], run, "Test stream", "s-6")
        try:
            frames = [f async for f in stream]
        finally:
            locks.release([("a.md", held)])

        assert [event for event, _data in parse_sse_events("".join(frames).encode())] == [
            "already_ingesting"
        ]
        assert not _is_held("s-6")
        run.assert_not_called()

    async def test_a_stream_that_is_never_read_frees_its_id(self):
        stream = _ingest_stream([("a.md", "a")], mock.AsyncMock(), "Test stream", "s-4")
        assert _is_held("s-4")
        del stream
        gc.collect()
        assert not _is_held("s-4")

    async def test_a_cancel_after_the_run_ended_answers_not_found(self):
        async def run(_locked, sse) -> AddSummary:
            sse.queue.put_nowait(None)
            return AddSummary(copied=[], errors=[])

        found: list[bool] = []
        async for frame in _ingest_stream([("a.md", "a")], run, "Test stream", "s-3"):
            if frame.startswith("event: done"):
                found.append(cancel_add("s-3"))

        assert found == [False]

    async def test_a_cancel_before_the_stream_starts_reaches_the_run(self):
        summary = AddSummary(copied=[], errors=[], cancelled=True, not_added=["a.md"])

        async def run(_locked, sse) -> AddSummary:
            assert sse.stopped_by_user and sse.cancel.is_set()
            sse.queue.put_nowait(None)
            return summary

        stream = _ingest_stream([("a.md", "a")], run, "Test stream", "s-5")
        assert cancel_add("s-5") is True
        frames = [f async for f in stream]

        events = parse_sse_events("".join(frames).encode())
        assert [data for event, data in events if event == "done"] == [summary.model_dump()]

    async def test_a_stop_by_the_user_cancels_the_run_and_still_ends_with_done(self):
        summary = AddSummary(copied=[], errors=[], cancelled=True, not_added=["a.md"])
        parked = asyncio.Event()

        async def _park() -> None:
            parked.set()
            await asyncio.Event().wait()

        async def run(_locked, sse) -> AddSummary:
            try:
                await sse.stoppable(_park())
            except asyncio.CancelledError:
                assert sse.stopped_by_user
            sse.queue.put_nowait(None)
            return summary

        async def _cancel_when_parked() -> bool:
            await parked.wait()
            return cancel_add("s-2")

        canceller = asyncio.create_task(_cancel_when_parked())
        frames = [f async for f in _ingest_stream([("a.md", "a")], run, "Test stream", "s-2")]

        assert await canceller is True
        events = parse_sse_events("".join(frames).encode())
        assert [data for event, data in events if event == "done"] == [summary.model_dump()]


async def _parked_work(sse: SseStream) -> asyncio.Task[None]:
    """A run that awaits work through *sse* which waits until it is cancelled."""
    started = asyncio.Event()

    async def work() -> None:
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(sse.stoppable(work()))
    await started.wait()
    return task


class TestAStopByTheUser:
    async def test_the_stop_cancels_the_work_the_run_awaits(self):
        sse = SseStream()
        run = await _parked_work(sse)

        sse.stop_by_user()

        with pytest.raises(asyncio.CancelledError):
            await run
        assert sse.stopped_by_user and sse.cancel.is_set()
        assert sse._work is None

    async def test_the_stop_cancels_only_the_work_not_the_run(self):
        sse = SseStream()
        started = asyncio.Event()

        async def work() -> None:
            started.set()
            await asyncio.Event().wait()

        async def run() -> str:
            try:
                await sse.stoppable(work())
            except asyncio.CancelledError:
                await asyncio.to_thread(lambda: None)
                return "rolled back"
            return "finished"

        task = asyncio.create_task(run())
        await started.wait()
        sse.stop_by_user()

        assert await task == "rolled back"
        assert not task.cancelled()

    async def test_the_stop_before_any_work_leaves_the_work_to_see_the_cancel(self):
        sse = SseStream()
        sse.stop_by_user()

        async def work() -> bool:
            return sse.cancel.is_set()

        assert await sse.stoppable(work()) is True
        assert sse.stopped_by_user

    async def test_the_stop_leaves_a_run_that_already_stops_alone(self):
        sse = SseStream()
        sse.cancel.set()
        run = await _parked_work(sse)

        sse.stop_by_user()

        assert not sse.stopped_by_user and not run.cancelling()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run


class TestACancelOfAnAddByPath:
    async def test_the_cancel_unregisters_a_root_with_no_file_indexed(self, tmp_path):
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        sse = SseStream()

        with mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)):
            summary = await _run_add([str(src)], False, None, None, sse)

        assert summary.cancelled is True
        assert summary.not_added == ["scan.txt"] and summary.copied == []
        assert summary.errors == []
        assert "scan.txt" not in cfg.linked_roots

    async def test_an_error_after_the_cancel_is_named_with_it(self, tmp_path):
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        sse = SseStream()
        sync = _stopping_sync(sse, error=RuntimeError("the embedder went away"))

        with mock.patch("lilbee.data.ingest.sync", new=sync):
            summary = await _run_add([str(src)], False, None, None, sse)

        assert summary.cancelled is True and summary.not_added == ["scan.txt"]
        assert summary.errors == ["the embedder went away"]

    async def test_a_cancel_before_the_registration_adds_nothing(self, tmp_path):
        src = tmp_path / "scan.txt"
        src.write_text("contents", encoding="utf-8")
        sse = SseStream()
        _stop_as_user(sse)
        sync = mock.AsyncMock()

        with mock.patch("lilbee.data.ingest.sync", new=sync):
            summary = await _run_add([str(src)], False, None, None, sse)

        assert summary.cancelled is True and summary.not_added == ["scan.txt"]
        assert "scan.txt" not in cfg.linked_roots
        sync.assert_not_called()

    async def test_a_registration_that_fails_after_the_cancel_ends_as_the_cancel(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        sse = SseStream()

        def _failing(_paths, *, force):
            _stop_as_user(sse)
            raise OSError("the config file is read-only")

        with mock.patch("lilbee.server.handlers.ingest.register_sources", _failing):
            summary = await _run_add([str(src)], False, None, None, sse)

        assert summary.cancelled is True
        assert summary.copied == [] and summary.not_added == ["src"]
        assert summary.errors == ["the config file is read-only"]

    async def test_a_registration_that_fails_after_a_dropped_stream_ends_as_a_cancel(
        self, tmp_path
    ):
        sse = SseStream()

        def _failing(_paths, *, force):
            sse.cancel.set()
            raise OSError("the config file is read-only")

        with (
            mock.patch("lilbee.server.handlers.ingest.register_sources", _failing),
            pytest.raises(asyncio.CancelledError),
        ):
            await _run_add([str(tmp_path)], False, None, None, sse)

    async def test_a_root_that_took_in_a_source_names_it_in_the_cancelled_summary(self, tmp_path):
        sse = SseStream()
        registration = RegisterResult(
            registered=["notes"], absorbed_into={"notes": ["work"]}, absorbed_roots={}
        )
        notes = tmp_path / "notes"
        notes.mkdir()

        def _forget(rollback):
            rollback.not_added = []

        with (
            mock.patch("lilbee.server.handlers.ingest.register_sources", return_value=registration),
            mock.patch("lilbee.app.ingest.AddRollback.forget_unfinished", _forget),
            mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)),
        ):
            summary = await _run_add([str(notes)], False, None, None, sse)

        assert summary.cancelled is True
        assert summary.copied == ["notes"] and summary.absorbed == ["work"]


class TestACancelOfAnUpload:
    async def test_the_cancel_deletes_a_file_the_upload_created(self):
        sse = SseStream()
        with mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)):
            summary = await _run_upload([("sub/new.md", b"# new\n")], sse)

        assert summary.cancelled is True
        assert summary.not_added == ["sub/new.md"] and summary.copied == []
        assert not (cfg.documents_dir / "sub" / "new.md").exists()

    async def test_the_cancel_keeps_a_file_the_upload_overwrote(self):
        (cfg.documents_dir / "old.md").write_bytes(b"# before\n")
        sse = SseStream()
        with mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)):
            summary = await _run_upload([("old.md", b"# after\n")], sse)

        assert summary.not_added == [] and summary.copied == ["old.md"]
        assert (cfg.documents_dir / "old.md").read_bytes() == b"# after\n"

    async def test_the_cancel_keeps_a_file_the_upload_moved(self):
        content = b"# same\n"
        (cfg.documents_dir / "was.md").write_bytes(content)
        get_services().store.get_sources.return_value = [
            {"filename": "was.md", "file_hash": hashlib.sha256(content).hexdigest()}
        ]
        sse = SseStream()
        with mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)):
            summary = await _run_upload([("now.md", content)], sse)

        assert summary.not_added == []
        assert (cfg.documents_dir / "now.md").read_bytes() == content

    async def test_the_cancel_keeps_a_dangling_symlink_the_upload_wrote_through(self, tmp_path):
        link = cfg.documents_dir / "link.md"
        link.symlink_to(tmp_path / "elsewhere.md")
        sse = SseStream()
        with mock.patch("lilbee.data.ingest.sync", new=_stopping_sync(sse)):
            summary = await _run_upload([("link.md", b"# through\n")], sse)

        assert summary.not_added == [] and summary.copied == ["link.md"]
        assert link.is_symlink()

    async def test_the_cancel_keeps_a_created_file_the_sync_indexed(self):
        sse = SseStream()

        async def _indexed_then_stopped(**_kwargs) -> SyncResult:
            get_services().store.get_sources.return_value = [
                {"filename": "done.md", "ingested_at": "2026-11-25T10:00:00", "file_hash": "h"}
            ]
            _stop_as_user(sse)
            raise asyncio.CancelledError

        with mock.patch("lilbee.data.ingest.sync", new=_indexed_then_stopped):
            summary = await _run_upload([("done.md", b"# done\n"), ("not.md", b"# not\n")], sse)

        assert summary.not_added == ["not.md"] and summary.copied == ["done.md"]
        assert (cfg.documents_dir / "done.md").is_file()
        assert not (cfg.documents_dir / "not.md").exists()

    async def test_an_error_after_the_cancel_is_named_with_it(self):
        sse = SseStream()
        sync = _stopping_sync(sse, error=RuntimeError("the embedder went away"))
        with mock.patch("lilbee.data.ingest.sync", new=sync):
            summary = await _run_upload([("new.md", b"# new\n")], sse)

        assert summary.cancelled is True and summary.errors == ["the embedder went away"]

    async def test_a_dropped_stream_keeps_the_file_the_upload_created(self):
        sse = SseStream()

        async def _dropped(**_kwargs) -> SyncResult:
            sse.cancel.set()
            raise asyncio.CancelledError

        with (
            mock.patch("lilbee.data.ingest.sync", new=_dropped),
            pytest.raises(asyncio.CancelledError),
        ):
            await _run_upload([("new.md", b"# new\n")], sse)

        assert (cfg.documents_dir / "new.md").is_file()
