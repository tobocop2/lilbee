"""Tests for the LlamaCppProvider batching queue and chat lock."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    """Reset provider singleton between tests."""
    from lilbee.services import reset_services

    reset_services()
    yield
    reset_services()


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with a test .gguf file."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-model.gguf").write_bytes(b"fake-gguf")
    cfg.models_dir = models
    cfg.embedding_model = "test-model"
    cfg.chat_model = "test-model"
    return models


@pytest.fixture()
def mock_llama_cpp() -> mock.MagicMock:
    """Inject a mock llama_cpp module into sys.modules."""
    mod = mock.MagicMock()
    sys.modules["llama_cpp"] = mod
    yield mod
    sys.modules.pop("llama_cpp", None)


def _make_embed_response(vectors: list[list[float]]) -> dict[str, Any]:
    """Build a mock create_embedding response."""
    return {"data": [{"embedding": v} for v in vectors]}


class TestEmbedQueue:
    def test_single_embed_request(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """One embed call returns the correct vectors."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        instance = mock.MagicMock()
        instance.create_embedding.return_value = _make_embed_response([[0.1, 0.2], [0.3, 0.4]])
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        result = provider.embed(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        instance.create_embedding.assert_called_once_with(input=["hello", "world"])
        provider.shutdown()

    def test_concurrent_embeds_batched(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Multiple concurrent embed calls are collected into fewer dispatch rounds."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        batch_sizes: list[int] = []
        batch_lock = threading.Lock()

        original_dispatch = LlamaCppProvider._dispatch_batch

        def tracking_dispatch(self_inner: Any, batch: list) -> None:
            with batch_lock:
                batch_sizes.append(len(batch))
            original_dispatch(self_inner, batch)

        instance = mock.MagicMock()
        instance.create_embedding.side_effect = lambda *, input: _make_embed_response(
            [[float(i)] for i in range(len(input))]
        )
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        results: list[list[list[float]] | None] = [None] * 5
        barrier = threading.Barrier(5)

        def worker(idx: int) -> None:
            barrier.wait()
            results[idx] = provider.embed([f"text-{idx}"])

        with mock.patch.object(LlamaCppProvider, "_dispatch_batch", tracking_dispatch):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        # All callers got a result
        for r in results:
            assert r is not None
            assert len(r) == 1

        # Batching collected multiple requests per dispatch round
        assert len(batch_sizes) < 5
        provider.shutdown()

    def test_embed_error_propagates(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """If create_embedding raises, all futures in the batch get the exception."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        instance = mock.MagicMock()
        instance.create_embedding.side_effect = RuntimeError("GPU out of memory")
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        errors: list[Exception | None] = [None] * 3
        barrier = threading.Barrier(3)

        def worker(idx: int) -> None:
            barrier.wait()
            try:
                provider.embed([f"text-{idx}"])
            except RuntimeError as exc:
                errors[idx] = exc

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        for err in errors:
            assert err is not None
            assert "GPU out of memory" in str(err)
        provider.shutdown()

    def test_concurrent_requests_all_dispatched(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """All concurrent embed requests are dispatched and return results."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        texts_received: list[list[str]] = []

        def fake_create_embedding(*, input: list[str]) -> dict[str, Any]:
            texts_received.append(input)
            return _make_embed_response([[1.0]] * len(input))

        instance = mock.MagicMock()
        instance.create_embedding.side_effect = fake_create_embedding
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        barrier = threading.Barrier(3)

        def worker(text: str) -> list[list[float]]:
            barrier.wait()
            return provider.embed([text])

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All three requests were dispatched (each gets its own create_embedding
        # call since _dispatch_batch processes requests individually)
        assert len(texts_received) == 3
        provider.shutdown()

    def test_sequential_embeds_still_work(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Single-threaded sequential usage works fine."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        instance = mock.MagicMock()
        instance.create_embedding.return_value = _make_embed_response([[1.0, 2.0]])
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        r1 = provider.embed(["first"])
        r2 = provider.embed(["second"])

        assert r1 == [[1.0, 2.0]]
        assert r2 == [[1.0, 2.0]]
        assert instance.create_embedding.call_count == 2
        provider.shutdown()


class TestChatLock:
    def test_chat_returns_string(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """Basic chat returns a string through the lock."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        instance = mock.MagicMock()
        instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello"
        provider.shutdown()

    def test_chat_serialized(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """Concurrent chat calls are serialized (no overlapping execution)."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        active = threading.Event()
        overlap_detected = threading.Event()

        def fake_chat(*, messages: Any, stream: bool = False, **kw: Any) -> dict:
            if active.is_set():
                overlap_detected.set()
            active.set()
            time.sleep(0.02)
            active.clear()
            return {"choices": [{"message": {"content": "ok"}}]}

        instance = mock.MagicMock()
        instance.create_chat_completion.side_effect = fake_chat
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        barrier = threading.Barrier(3)
        results: list[str | None] = [None] * 3

        def worker(idx: int) -> None:
            barrier.wait()
            results[idx] = provider.chat([{"role": "user", "content": "hi"}])  # type: ignore[assignment]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not overlap_detected.is_set(), "Chat calls overlapped — lock not working"
        for r in results:
            assert r == "ok"
        provider.shutdown()

    def test_chat_stream_through_lock(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Streaming chat works and holds the lock until iteration completes."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}]},
        ]
        instance = mock.MagicMock()
        instance.create_chat_completion.return_value = iter(stream_chunks)
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        # Lock should be held during streaming
        assert not provider._chat_lock.acquire(blocking=False), "Lock should be held during stream"

        tokens = list(result)
        assert tokens == ["Hello", " world"]

        # Lock released after iteration
        assert provider._chat_lock.acquire(blocking=False), "Lock should be released after stream"
        provider._chat_lock.release()
        provider.shutdown()


class TestShutdown:
    def test_shutdown_stops_worker(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """Shutdown sentinel stops the background worker thread."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        mock_llama_cpp.Llama.return_value = mock.MagicMock()

        provider = LlamaCppProvider()
        assert provider._worker.is_alive()

        provider.shutdown()
        assert not provider._worker.is_alive()

    def test_shutdown_during_batch_collection(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Sentinel arriving while collecting a batch stops the worker cleanly."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        instance = mock.MagicMock()
        instance.create_embedding.return_value = _make_embed_response([[1.0]])
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()

        # Submit a real request followed immediately by the shutdown sentinel.
        # The worker will process the first request, then during batch window
        # collection it will encounter the None sentinel and exit.
        result = provider.embed(["hello"])
        assert result == [[1.0]]

        # Now put a request and sentinel close together so sentinel arrives
        # during the batch window of the second request.
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import _EmbedRequest

        fut: Future[list[list[float]]] = Future()
        provider._embed_queue.put(_EmbedRequest(texts=["world"], future=fut))
        provider._embed_queue.put(None)

        # The worker should process "world" then see sentinel and exit
        result2 = fut.result(timeout=5)
        assert result2 == [[1.0]]
        provider._worker.join(timeout=2)
        assert not provider._worker.is_alive()


class TestLockedStreamIteratorClose:
    def test_close_releases_lock(self):
        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()
        stream = _LockedStreamIterator(iter([]), lock)
        stream.close()
        assert lock.acquire(blocking=False)
        lock.release()


class TestLockedStreamIteratorExceptionRelease:
    def test_non_stop_iteration_exception_releases_lock(self):
        """When the underlying response raises a non-StopIteration exception,
        the lock is released and the exception propagates."""
        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        def exploding_iter():
            yield {"choices": [{"delta": {"content": "ok"}}]}
            raise ValueError("boom")

        lock = threading.Lock()
        lock.acquire()
        stream = _LockedStreamIterator(exploding_iter(), lock)
        # First call succeeds
        assert next(stream) == "ok"
        # Second call hits the ValueError — lock should be released
        with pytest.raises(ValueError, match="boom"):
            next(stream)
        assert lock.acquire(blocking=False)
        lock.release()
