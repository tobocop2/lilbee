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
        instance.create_embedding.side_effect = [
            _make_embed_response([[0.1, 0.2]]),
            _make_embed_response([[0.3, 0.4]]),
        ]
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        result = provider.embed(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert instance.create_embedding.call_count == 2
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


class TestVisionModel:
    def test_load_vision_llama_creates_handler(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """_load_vision_llama creates a Llama instance with a chat handler."""
        from lilbee.providers.llama_cpp_provider import _load_vision_llama

        # Create mmproj file
        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")

        mock_handler = mock.MagicMock()
        mock_handler_cls = mock.MagicMock(return_value=mock_handler)

        # The mock_llama_cpp fixture puts a MagicMock in sys.modules["llama_cpp"],
        # so submodule imports like llama_cpp.llama_chat_format need to work too.
        mock_chat_format = mock.MagicMock()
        mock_chat_format.Llava15ChatHandler = mock_handler_cls
        sys.modules["llama_cpp.llama_chat_format"] = mock_chat_format

        try:
            _load_vision_llama(models_dir / "test-model.gguf", mmproj_path)

            mock_handler_cls.assert_called_once_with(clip_model_path=str(mmproj_path))
            # Llama called with chat_handler
            call_kwargs = mock_llama_cpp.Llama.call_args[1]
            assert call_kwargs["chat_handler"] is mock_handler
        finally:
            sys.modules.pop("llama_cpp.llama_chat_format", None)

    def test_find_mmproj_raises_when_missing(self, models_dir: Path) -> None:
        """_find_mmproj_for_model raises ProviderError when no mmproj found."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _find_mmproj_for_model

        with pytest.raises(ProviderError, match="mmproj"):
            _find_mmproj_for_model(models_dir / "test-model.gguf")

    def test_find_mmproj_finds_by_name(self, models_dir: Path) -> None:
        """_find_mmproj_for_model finds mmproj files in the models directory."""
        from lilbee.providers.llama_cpp_provider import _find_mmproj_for_model

        mmproj = models_dir / "model-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake")
        result = _find_mmproj_for_model(models_dir / "test-model.gguf")
        assert result == mmproj

    def test_is_vision_model_matches_config(self, models_dir: Path) -> None:
        """_is_vision_model returns True for cfg.vision_model."""
        from lilbee.providers.llama_cpp_provider import _is_vision_model

        cfg.vision_model = "test-vision"
        assert _is_vision_model("test-vision") is True
        assert _is_vision_model("test-chat") is False
        cfg.vision_model = ""

    def test_get_vision_llm_caches(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_vision_llm caches the vision model instance."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        mmproj = models_dir / "test-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake-mmproj")
        cfg.vision_model = "test-model"

        mock_handler = mock.MagicMock()
        mock_chat_format = mock.MagicMock()
        mock_chat_format.Llava15ChatHandler = mock.MagicMock(return_value=mock_handler)
        sys.modules["llama_cpp.llama_chat_format"] = mock_chat_format

        instance = mock.MagicMock()
        instance.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_llama_cpp.Llama.return_value = instance

        try:
            provider = LlamaCppProvider()
            provider.chat([{"role": "user", "content": "hi"}], model="test-model")
            provider.chat([{"role": "user", "content": "hi"}], model="test-model")

            # Llama should only be called once (cached)
            assert mock_llama_cpp.Llama.call_count == 1
            provider.shutdown()
        finally:
            sys.modules.pop("llama_cpp.llama_chat_format", None)
            cfg.vision_model = ""


class TestLoadLlamaNCtx:
    def test_default_n_ctx(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """When num_ctx is None, _load_llama passes n_ctx=0 and n_batch from metadata."""
        from lilbee.providers.llama_cpp_provider import _load_llama

        cfg.num_ctx = None
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "2048",
        }
        _load_llama(models_dir / "test-model.gguf", embedding=True)

        # Called twice: once for metadata (vocab_only), once for model
        assert mock_llama_cpp.Llama.call_count == 2
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 0
        assert call_kwargs["n_batch"] == 2048

    def test_custom_n_ctx(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """When num_ctx is set, _load_llama uses it for n_ctx and n_batch."""
        from lilbee.providers.llama_cpp_provider import _load_llama

        cfg.num_ctx = 8192
        _load_llama(models_dir / "test-model.gguf", embedding=True)

        # No metadata read needed when n_ctx is explicit
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 8192
        assert call_kwargs["n_batch"] == 8192

    def test_embedding_flag_passed(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """_load_llama passes embedding flag correctly."""
        from lilbee.providers.llama_cpp_provider import _load_llama

        mock_llama_cpp.Llama.return_value.metadata = {}
        _load_llama(models_dir / "test-model.gguf", embedding=True)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is True

        mock_llama_cpp.Llama.reset_mock()
        _load_llama(models_dir / "test-model.gguf", embedding=False)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is False


class TestSuppressStderrThreadSafety:
    def test_concurrent_suppress_stderr_no_corruption(self) -> None:
        """B3: _suppress_stderr serializes fd 2 manipulation via _STDERR_LOCK."""
        from lilbee.providers.llama_cpp_provider import _suppress_stderr

        results: list[int] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker(value: int) -> None:
            barrier.wait()
            try:
                result = _suppress_stderr(lambda v: v * 2, value)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors during concurrent _suppress_stderr: {errors}"
        assert sorted(results) == [0, 2, 4, 6]

    def test_suppress_stderr_uses_lock(self) -> None:
        """B3: Verify _suppress_stderr acquires _STDERR_LOCK."""
        from lilbee.providers.llama_cpp_provider import _STDERR_LOCK, _suppress_stderr

        lock_was_held = []

        def check_lock():
            # If the lock is held (by us), acquire(blocking=False) returns False
            locked = not _STDERR_LOCK.acquire(blocking=False)
            if not locked:
                _STDERR_LOCK.release()
            lock_was_held.append(locked)
            return 42

        result = _suppress_stderr(check_lock)
        assert result == 42
        assert lock_was_held == [True]
