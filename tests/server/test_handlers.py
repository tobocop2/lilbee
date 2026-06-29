"""Tests for the /api/add endpoint and SSE progress streaming."""

import asyncio
import contextlib
from pathlib import Path
from unittest import mock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.server import auth as _auth_mod
from lilbee.server.handlers import MAX_ADD_FILES
from tests.server.conftest import parse_sse_events as _parse_sse_events


def _auth_headers() -> dict[str, str]:
    """Return Authorization header using the current session token."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    """Redirect config paths to temp dir for every test."""
    snapshot = cfg.model_copy()
    docs = tmp_path / "documents"
    docs.mkdir()
    cfg.documents_dir = docs
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.concept_graph = False
    yield docs
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so handlers never touch real backends."""
    from tests.conftest import make_mock_services

    embedder = mock.MagicMock()
    embedder.embed.return_value = [0.1] * 768
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    embedder.validate_model.return_value = None
    services = make_mock_services(embedder=embedder)
    # /api/chat now goes through chat_dispatch, which validates the requested
    # model against the KnownModelCache. Pre-load both the registry and the
    # cache so resolve() finds cfg.chat_model.
    chat_manifest = mock.MagicMock()
    chat_manifest.ref = cfg.chat_model
    chat_manifest.task = "chat"
    services.registry.list_installed = mock.MagicMock(return_value=[chat_manifest])
    services.known_models.refs = mock.MagicMock(return_value={cfg.chat_model})
    services.known_models.resolve = mock.MagicMock(
        side_effect=lambda model: model if model == cfg.chat_model else None
    )
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture(autouse=True)
def reset_ingest_locks():
    """Clear per-source ingest locks so each test starts with a clean registry.

    Locks are bound to the event loop; leaking them across tests produces
    cryptic 'attached to a different loop' errors.
    """
    from lilbee.app.services import get_services

    get_services().ingest_lock_registry.reset()
    yield
    get_services().ingest_lock_registry.reset()


def _make_xberg_result(text: str = "Some extracted text. " * 20, num_chunks: int = 1):
    chunks = []
    for i in range(num_chunks):
        chunk_text = text[i * len(text) // num_chunks : (i + 1) * len(text) // num_chunks]
        chunk = mock.MagicMock()
        chunk.content = chunk_text
        chunk.metadata = mock.MagicMock(chunk_index=i, first_page=None, last_page=None)
        chunks.append(chunk)
    result = mock.MagicMock()
    result.chunks = chunks
    result.content = text
    result.pages = []
    return result


@mock.patch(
    "lilbee.data.xberg_extract.aextract_document",
    new_callable=mock.AsyncMock,
    return_value=_make_xberg_result(),
)
class TestAddEndpoint:
    async def test_add_single_file(self, mock_extract_file, isolated_env, tmp_path):
        """POST /api/add with a valid file streams SSE events and adds it."""
        from lilbee.server.app import create_app

        src = tmp_path / "input.txt"
        src.write_text("Hello world content for testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        event_types = [e[0] for e in events]
        assert "file_start" in event_types
        assert "file_done" in event_types
        assert "done" in event_types

    async def test_add_nonexistent_file_in_errors(self, mock_extract_file, isolated_env, tmp_path):
        """Nonexistent paths appear in the summary errors list."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": ["/no/such/file.txt"]}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "/no/such/file.txt" in summary["errors"]

    async def test_add_with_force_flag(self, mock_extract_file, isolated_env, tmp_path):
        """The force flag allows overwriting existing files."""
        from lilbee.server.app import create_app

        src = tmp_path / "dup.txt"
        src.write_text("Version 1")
        (isolated_env / "dup.txt").write_text("Existing")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)], "force": True}, headers=_auth_headers()
            )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "dup.txt" in summary["copied"]

    async def test_done_event_has_correct_fields(self, mock_extract_file, isolated_env, tmp_path):
        """The done event includes added, updated, removed, failed counts."""
        from lilbee.server.app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for done event testing.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        done_data = next(d for t, d in events if t == "done")
        assert "added" in done_data
        assert "updated" in done_data
        assert "removed" in done_data
        assert "failed" in done_data

    async def test_file_start_has_total_and_current(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """file_start event includes total_files and current_file."""
        from lilbee.server.app import create_app

        src = tmp_path / "progress.txt"
        src.write_text("Progress tracking test.")

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
            )

        events = _parse_sse_events(resp.content)
        file_start = next(d for t, d in events if t == "file_start")
        assert file_start["total_files"] >= 1
        assert file_start["current_file"] >= 1

    async def test_add_with_enable_ocr(self, mock_extract_file, isolated_env, tmp_path):
        """enable_ocr parameter is temporarily set on cfg during sync."""
        from lilbee.server.app import create_app

        src = tmp_path / "doc.txt"
        src.write_text("Content for enable_ocr test.")
        original_ocr = cfg.enable_ocr

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(src)], "enable_ocr": True},
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        # enable_ocr should be restored after the call
        assert cfg.enable_ocr == original_ocr

    async def test_add_emits_heartbeat_during_slow_sync(
        self, mock_extract_file, isolated_env, tmp_path
    ):
        """Root-cause fix for obsidian-lilbee-v8y.

        A long-running vision OCR pass can go >120s without emitting a
        progress event. The plugin's STREAM_IDLE_TIMEOUT_MS (120s) then
        aborts the stream. The server must emit 'heartbeat' SSE events at
        cfg.sse_heartbeat_interval whenever the producer queue is idle so
        the plugin's withIdleTimeout keeps resetting. This test drives a
        sync that sleeps longer than the heartbeat interval and asserts
        at least one heartbeat event was delivered to the HTTP client.
        """
        from lilbee.server.app import create_app

        src = tmp_path / "slow.txt"
        src.write_text("Slow sync content for heartbeat test.")
        cfg.sse_heartbeat_interval = 0.2

        async def slow_sync(force_rebuild=False, quiet=False, *, on_progress=None, cancel=None):
            from lilbee.data.ingest import SyncResult

            await asyncio.sleep(0.6)
            return SyncResult(added=["slow.txt"])

        with mock.patch("lilbee.data.ingest.sync", side_effect=slow_sync):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/add", json={"paths": [str(src)]}, headers=_auth_headers()
                )

        assert resp.status_code == 201
        events = _parse_sse_events(resp.content)
        heartbeats = [d for t, d in events if t == "heartbeat"]
        assert heartbeats, f"expected heartbeat events during slow sync, got {events}"
        assert "ts" in heartbeats[0]


class TestAddValidation:
    async def test_empty_paths_returns_400(self, isolated_env):
        """POST /api/add with empty paths list returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": []}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_missing_paths_returns_400(self, isolated_env):
        """POST /api/add without paths key returns 400."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"force": True}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_too_many_files_returns_400(self, isolated_env):
        """POST /api/add with >100 paths returns 400."""
        from lilbee.server.app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES + 1)]
        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/add", json={"paths": paths}, headers=_auth_headers())
        assert resp.status_code == 400

    async def test_exactly_max_files_accepted(self, isolated_env, tmp_path):
        """POST /api/add with exactly 100 paths is accepted (paths can be nonexistent)."""
        from lilbee.server.app import create_app

        paths = [f"/fake/file_{i}.txt" for i in range(MAX_ADD_FILES)]
        with mock.patch(
            "lilbee.data.xberg_extract.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post("/api/add", json={"paths": paths}, headers=_auth_headers())
        # Should be 200 (all files nonexistent, but request is valid)
        assert resp.status_code == 201


class TestSseStreamCallback:
    async def test_callback_enqueues_formatted_sse(self):
        """The SSE callback formats events correctly."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        sse.callback("file_start", {"file": "test.txt", "total_files": 1, "current_file": 1})
        item = sse.queue.get_nowait()
        assert item is not None
        assert item.startswith("event: file_start\n")
        assert '"file": "test.txt"' in item

    async def test_callback_from_thread_uses_threadsafe(self):
        """When called from a worker thread, uses call_soon_threadsafe."""
        import concurrent.futures

        from lilbee.server.handlers import SseStream

        sse = SseStream()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(pool, sse.callback, "embed", {"chunk": 1})
            await future

        # Give the event loop a tick to process the call_soon_threadsafe
        await asyncio.sleep(0)
        item = sse.queue.get_nowait()
        assert item is not None
        assert "embed" in item


class TestSseStreamDrain:
    async def test_drain_delivers_every_event_across_poll_boundaries(self):
        """Events put near the poll timeout boundary are never dropped."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        total = 12

        async def produce() -> None:
            for i in range(total):
                # Sleeps straddle drain's 0.1s poll period to hit the boundary.
                await asyncio.sleep(0.03 + (i % 4) * 0.025)
                sse.queue.put_nowait(f"event: tick\ndata: {i}\n\n")
            sse.queue.put_nowait(None)

        task = asyncio.create_task(produce())
        received = [item async for item in sse.drain(task, "drain test")]
        assert len([r for r in received if "tick" in r]) == total

    async def test_drain_flushes_events_behind_the_sentinel(self):
        """A producer that finishes before drain starts still gets its progress out."""
        from lilbee.server.handlers import SseStream

        sse = SseStream()
        # Sentinel already queued; the progress callback is still in the loop's
        # ready callbacks, as happens when the producer outruns the consumer.
        sse.queue.put_nowait(None)
        asyncio.get_running_loop().call_soon(sse.queue.put_nowait, "event: embed\ndata: {}\n\n")
        task = asyncio.create_task(asyncio.sleep(0))
        received = [item async for item in sse.drain(task, "drain flush test")]
        assert any("embed" in r for r in received)


class TestOptionsPassthrough:
    """Verify generation options are extracted from request body and passed through."""

    async def test_ask_passes_options(self, isolated_env):
        from lilbee.server.app import create_app
        from lilbee.server.handlers.sse import _resolve_generation_options

        with mock.patch(
            "lilbee.server.handlers.rag._resolve_generation_options",
            wraps=_resolve_generation_options,
        ) as spy:
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/ask",
                    json={"question": "test", "options": {"temperature": 0.3}},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 201
        assert "answer" in resp.json()
        # The body's options must actually reach the generation-options resolver,
        # not just produce a successful response.
        spy.assert_any_call({"temperature": 0.3})

    async def test_chat_passes_options(self, isolated_env):
        from lilbee.server.app import create_app
        from lilbee.server.handlers.sse import _resolve_generation_options

        with mock.patch(
            "lilbee.server.handlers.rag._resolve_generation_options",
            wraps=_resolve_generation_options,
        ) as spy:
            async with AsyncTestClient(create_app()) as client:
                resp = await client.post(
                    "/api/chat",
                    json={
                        "question": "test",
                        "history": [],
                        "options": {"seed": 42},
                    },
                    headers=_auth_headers(),
                )
        assert resp.status_code == 201
        assert "answer" in resp.json()
        spy.assert_any_call({"seed": 42})

    async def test_ask_without_options(self, isolated_env):
        """Request without options field still works."""
        from lilbee.server.app import create_app

        async with AsyncTestClient(create_app()) as client:
            resp = await client.post("/api/ask", json={"question": "test"}, headers=_auth_headers())
        assert resp.status_code == 201


class TestCreateApp:
    def test_app_has_add_route(self):
        """The Litestar app registers the /api/add route."""
        from lilbee.server.app import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/api/add" in paths


class TestAddIngestMutex:
    """Tests for the per-source ingest mutex gating ``/api/add``."""

    async def _collect(self, gen):
        """Drain an async generator of SSE strings into (event, data) pairs."""
        text = ""
        async for frame in gen:
            text += frame
        return _parse_sse_events(text.encode())

    async def test_try_acquire_rejects_while_held(self, isolated_env):
        """A second ``_try_acquire_source`` for a held name returns ``None``."""
        from lilbee.app.services import get_services

        first = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert first is not None
        try:
            second = await get_services().ingest_lock_registry.try_acquire("doc.txt")
            assert second is None
        finally:
            first.release()

    async def test_try_acquire_succeeds_after_release(self, isolated_env):
        """Once released, the same name can be acquired again."""
        from lilbee.app.services import get_services

        first = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert first is not None
        first.release()
        second = await get_services().ingest_lock_registry.try_acquire("doc.txt")
        assert second is not None
        second.release()

    async def test_try_acquire_distinct_names_run_parallel(self, isolated_env):
        """Locks for different source names are independent."""
        from lilbee.app.services import get_services

        a = await get_services().ingest_lock_registry.try_acquire("a.txt")
        b = await get_services().ingest_lock_registry.try_acquire("b.txt")
        assert a is not None
        assert b is not None
        a.release()
        b.release()

    async def test_canonical_name_matches_basename(self, isolated_env):
        """Canonical source names match what ``copy_files`` writes on disk."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        assert IngestLockRegistry.canonical_source_name("/some/path/doc.txt") == "doc.txt"
        assert IngestLockRegistry.canonical_source_name("doc.txt") == "doc.txt"

    async def test_release_evicts_entry_so_registry_does_not_grow(self, isolated_env):
        """A long-lived daemon must not keep one lock per filename forever."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        registry = IngestLockRegistry()
        for i in range(50):
            acquired, busy = await registry.acquire([f"doc{i}.txt"])
            assert not busy
            registry.release(acquired)
        # Every name was released, so no entry should linger.
        assert registry._locks == {}

    async def test_release_keeps_entry_for_still_held_name(self, isolated_env):
        """Releasing one batch must not evict a name another batch still holds."""
        from lilbee.runtime.ingest_lock import IngestLockRegistry

        registry = IngestLockRegistry()
        held, _ = await registry.acquire(["doc.txt"])
        held_lock = held[0][1]
        # A second acquire for the same name is rejected (busy), nothing to evict.
        also, busy = await registry.acquire(["doc.txt"])
        assert also == [] and busy == ["doc.txt"]
        # Releasing a DIFFERENT batch (other.txt) must not evict doc.txt, and must
        # not disturb doc.txt's lock identity.
        other, _ = await registry.acquire(["other.txt"])
        registry.release(other)
        assert "other.txt" not in registry._locks  # the released name is evicted
        assert registry._locks.get("doc.txt") is held_lock  # held entry untouched
        registry.release(held)
        assert registry._locks == {}

    async def test_acquire_add_locks_dedups_repeated_paths(self, isolated_env):
        """Duplicate paths in one request collapse to a single acquired lock."""
        from lilbee.app.services import get_services

        registry = get_services().ingest_lock_registry
        acquired, busy = await registry.acquire(["/x/doc.txt", "/y/doc.txt"])
        try:
            assert busy == []
            assert [name for name, _ in acquired] == ["doc.txt"]
        finally:
            registry.release(acquired)

    async def test_second_concurrent_add_emits_already_ingesting(self, isolated_env, tmp_path):
        """Second /api/add for a held source yields already_ingesting, no done."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "holdme.txt"
        src.write_text("payload")

        lock = await get_services().ingest_lock_registry.try_acquire("holdme.txt")
        assert lock is not None
        try:
            events = await self._collect(add_files_stream([str(src)]))
        finally:
            lock.release()

        event_types = [e[0] for e in events]
        assert event_types == ["already_ingesting"]
        assert events[0][1] == {"source": "holdme.txt"}

    async def test_partial_contention_partitions_paths(self, isolated_env, tmp_path):
        """Held paths emit already_ingesting; free paths still run to done."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        held = tmp_path / "held.txt"
        held.write_text("held payload")
        free = tmp_path / "free.txt"
        free.write_text("free payload")

        lock = await get_services().ingest_lock_registry.try_acquire("held.txt")
        assert lock is not None
        try:
            with mock.patch(
                "lilbee.data.xberg_extract.aextract_document",
                new_callable=mock.AsyncMock,
                return_value=_make_xberg_result(),
            ):
                events = await self._collect(add_files_stream([str(held), str(free)]))
        finally:
            lock.release()

        event_types = [e[0] for e in events]
        already = [d for t, d in events if t == "already_ingesting"]
        assert {"source": "held.txt"} in already
        assert "done" in event_types
        # The contended path must not be ingested under the holder's lock: only
        # the free path is copied (regression guard for the lock-bypass bug).
        summary = [d for t, d in events if t == "done" and "copied" in d][-1]
        assert "free.txt" in summary["copied"]
        assert "held.txt" not in summary["copied"]

    async def test_concurrent_different_sources_run_in_parallel(self, isolated_env, tmp_path):
        """Disjoint sources do not contend: both requests complete with done."""
        from lilbee.server.handlers import add_files_stream

        a = tmp_path / "a.txt"
        a.write_text("alpha")
        b = tmp_path / "b.txt"
        b.write_text("beta")

        async def _run(path: Path):
            text = ""
            with mock.patch(
                "lilbee.data.xberg_extract.aextract_document",
                new_callable=mock.AsyncMock,
                return_value=_make_xberg_result(),
            ):
                async for frame in add_files_stream([str(path)]):
                    text += frame
            return _parse_sse_events(text.encode())

        events_a, events_b = await asyncio.gather(_run(a), _run(b))
        assert "done" in [t for t, _ in events_a]
        assert "done" in [t for t, _ in events_b]
        assert "already_ingesting" not in [t for t, _ in events_a]
        assert "already_ingesting" not in [t for t, _ in events_b]

    async def test_mutex_released_on_exception(self, isolated_env, tmp_path):
        """If ingest raises, the source mutex is released so retries succeed."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "boom.txt"
        src.write_text("contents")

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("ingest exploded")

        with mock.patch("lilbee.data.ingest.sync", new=_boom):
            events = await self._collect(add_files_stream([str(src)]))

        assert any(t == "error" for t, _ in events)

        retry = await get_services().ingest_lock_registry.try_acquire("boom.txt")
        assert retry is not None
        retry.release()

    async def test_mutex_released_on_task_cancellation(self, isolated_env, tmp_path):
        """Task cancellation during ingest still releases the source mutex."""
        from lilbee.app.services import get_services
        from lilbee.server.handlers import add_files_stream

        src = tmp_path / "slow.txt"
        src.write_text("contents")

        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(30)

        gen = add_files_stream([str(src)])
        with mock.patch("lilbee.data.ingest.sync", new=_hang):
            outer = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.05)
            outer.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await outer
            await gen.aclose()

        retry = await get_services().ingest_lock_registry.try_acquire("slow.txt")
        assert retry is not None
        retry.release()


class TestAddIngestHardening:
    """Option-A hardening: ``sync()`` always passes ``needs_cleanup=True``."""

    @staticmethod
    def _cleanup_sources(store) -> list[str]:
        """Sources whose batched write carried a cleanup delete."""
        sources: list[str] = []
        for call in store.write_chunks_batch.call_args_list:
            for item in call.args[0]:
                if item.needs_cleanup:
                    sources.append(item.source)
        return sources

    async def test_new_file_triggers_cleanup(self, isolated_env, tmp_path, mock_svc):
        """New files still carry a cleanup delete in their batched write: closes
        the orphaned-chunks race when a prior ingest died before upsert_source."""
        from lilbee.data.ingest import sync

        src = isolated_env / "fresh.txt"
        src.write_text("Fresh content.")

        store = mock_svc.store
        # No prior sources: existing_sources lookup returns nothing.
        store.get_sources.return_value = []

        with mock.patch(
            "lilbee.data.xberg_extract.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            await sync(quiet=True)

        # Option-A hardening: cleanup is requested even for 'new' files.
        assert "fresh.txt" in self._cleanup_sources(store)

    async def test_retry_after_orphaned_chunks_cleans_up(self, isolated_env, tmp_path, mock_svc):
        """If a prior run left chunks without an ``upsert_source`` record,
        the retry path removes them in the same transaction as the re-add.
        """
        from lilbee.data.ingest import sync

        src = isolated_env / "orphan.txt"
        src.write_text("Recovered content.")

        store = mock_svc.store
        # No source row, yet chunks exist on disk (the crashed-previous-run
        # scenario). The batched write's cleanup delete is idempotent.
        store.get_sources.return_value = []

        with mock.patch(
            "lilbee.data.xberg_extract.aextract_document",
            new_callable=mock.AsyncMock,
            return_value=_make_xberg_result(),
        ):
            await sync(quiet=True)

        # The stale chunks are removed in the same batched write that re-adds them.
        assert "orphan.txt" in self._cleanup_sources(store)
        store.write_chunks_batch.assert_called()
