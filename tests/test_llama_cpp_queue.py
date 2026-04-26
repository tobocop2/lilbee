"""Tests for the LlamaCppProvider batching queue and chat lock."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
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
    cfg.embedding_model = TEST_EMBED_REF
    cfg.chat_model = TEST_LOCAL_REF
    cfg.subprocess_embed = False
    patcher = mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        side_effect=lambda m: models / f"{m.rsplit('/', 1)[-1]}",
    )
    patcher.start()
    yield models
    patcher.stop()


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


class TestRerankQueue:
    def test_rerank_returns_score_per_candidate(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """rerank() returns one float per candidate in input order."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.reranker_model = TEST_EMBED_REF
        instance = mock.MagicMock()
        scores_iter = iter([0.81, 0.42, 0.13])

        def fake_embed(*, input):
            return {"data": [{"embedding": [next(scores_iter)]}]}

        instance.create_embedding.side_effect = fake_embed
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        scores = provider.rerank("q", ["a", "b", "c"])
        assert scores == [0.81, 0.42, 0.13]
        provider.shutdown()

    def test_rerank_pair_format(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """Each (query, candidate) pair joins into one ``"query</s></s>candidate"`` input.

        Splitting the pair into two sequences makes llama.cpp's
        ``pooling_type=LLAMA_POOLING_TYPE_RANK`` fail with ``llama_decode
        returned -1``.
        """
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.reranker_model = TEST_EMBED_REF
        instance = mock.MagicMock()
        captured_inputs: list[str] = []

        def fake_embed(*, input):
            captured_inputs.append(input)
            return {"data": [{"embedding": [0.5]}]}

        instance.create_embedding.side_effect = fake_embed
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        provider.rerank("what is paris", ["Paris is a city", "The moon is cheese"])
        assert captured_inputs == [
            "what is paris</s></s>Paris is a city",
            "what is paris</s></s>The moon is cheese",
        ]
        provider.shutdown()

    def test_rerank_empty_candidates_returns_empty(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Empty candidate list short-circuits without touching the model."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.reranker_model = TEST_EMBED_REF
        mock_llama_cpp.Llama.return_value = mock.MagicMock()

        provider = LlamaCppProvider()
        assert provider.rerank("q", []) == []
        provider.shutdown()

    def test_rerank_raises_when_model_unset(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Calling rerank() with no reranker_model surfaces a ProviderError."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.reranker_model = ""
        mock_llama_cpp.Llama.return_value = mock.MagicMock()

        provider = LlamaCppProvider()
        with pytest.raises(ProviderError, match="No reranker model configured"):
            provider.rerank("q", ["a"])
        provider.shutdown()

    def test_rerank_and_embed_use_isolated_cache_keys(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """rerank and embed on the same model produce separate Llama loads."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.embedding_model = TEST_EMBED_REF
        cfg.reranker_model = TEST_EMBED_REF
        instance = mock.MagicMock()
        instance.create_embedding.return_value = {"data": [{"embedding": [0.5]}]}
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        provider.embed(["x"])
        provider.rerank("q", ["a"])
        calls = mock_llama_cpp.Llama.call_args_list
        modes = sorted(c.kwargs.get("embedding") for c in calls if "embedding" in c.kwargs)
        assert modes.count(True) == 2
        assert any("pooling_type" in c.kwargs for c in calls)
        provider.shutdown()


class TestShutdown:
    def test_shutdown_stops_worker(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """Shutdown sentinel stops the background worker thread."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        mock_llama_cpp.Llama.return_value = mock.MagicMock()

        provider = LlamaCppProvider()
        assert provider._embed_thread.is_alive()

        provider.shutdown()
        assert not provider._embed_thread.is_alive()

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
        provider._embed_thread.join(timeout=2)
        assert not provider._embed_thread.is_alive()


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
        """``mtmd_backend.load_vision_llama`` wires a chat handler into Llama."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")
        mock_handler = mock.MagicMock()

        with mock.patch(
            "lilbee.providers.mtmd_backend.build_vision_chat_handler",
            return_value=mock_handler,
        ) as build:
            load_vision_llama(models_dir / "test-model.gguf", mmproj_path)

        build.assert_called_once_with(models_dir / "test-model.gguf", mmproj_path)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["chat_handler"] is mock_handler

    def test_find_mmproj_raises_when_missing(self, models_dir: Path) -> None:
        """find_mmproj_for_model raises ProviderError when no mmproj found."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        with pytest.raises(ProviderError, match="mmproj"):
            find_mmproj_for_model(models_dir / "test-model.gguf")

    def test_find_mmproj_finds_by_name(self, models_dir: Path) -> None:
        """find_mmproj_for_model finds mmproj files in the models directory."""
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        mmproj = models_dir / "model-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake")
        result = find_mmproj_for_model(models_dir / "test-model.gguf")
        assert result == mmproj


class TestLoadLlamaNCtx:
    def test_default_n_ctx(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """When num_ctx is None, load_llama passes n_ctx=0 and n_batch from metadata."""
        from lilbee.providers.llama_cpp_provider import load_llama
        from lilbee.providers.model_cache import MODE_EMBED

        cfg.num_ctx = None
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "2048",
        }
        load_llama(models_dir / "test-model.gguf", mode=MODE_EMBED)

        # Called twice: once for metadata (vocab_only), once for model
        assert mock_llama_cpp.Llama.call_count == 2
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 0
        assert call_kwargs["n_batch"] == 2048

    def test_custom_n_ctx(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """When num_ctx is set, load_llama uses it for n_ctx and n_batch."""
        from lilbee.providers.llama_cpp_provider import load_llama
        from lilbee.providers.model_cache import MODE_EMBED

        cfg.num_ctx = 8192
        load_llama(models_dir / "test-model.gguf", mode=MODE_EMBED)

        # No metadata read needed when n_ctx is explicit
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 8192
        assert call_kwargs["n_batch"] == 8192

    def test_embedding_flag_passed(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama passes embedding flag correctly."""
        from lilbee.providers.llama_cpp_provider import load_llama
        from lilbee.providers.model_cache import MODE_CHAT, MODE_EMBED

        mock_llama_cpp.Llama.return_value.metadata = {}
        load_llama(models_dir / "test-model.gguf", mode=MODE_EMBED)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is True

        mock_llama_cpp.Llama.reset_mock()
        cfg.num_ctx = 8192
        load_llama(models_dir / "test-model.gguf", mode=MODE_CHAT)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is False

    def test_chat_default_caps_at_safe_value_when_num_ctx_unset(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """num_ctx=None on chat mode caps at 8192 even when the model trains at 128K."""
        from lilbee.providers.llama_cpp_provider import DEFAULT_NUM_CTX, load_llama
        from lilbee.providers.model_cache import MODE_CHAT

        cfg.num_ctx = None
        # Metadata read returns a huge training context (mimics modern chat GGUFs)
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "llama",
            "llama.context_length": "131072",
        }

        load_llama(models_dir / "test-model.gguf", mode=MODE_CHAT)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == DEFAULT_NUM_CTX  # 8192

    def test_chat_default_uses_training_ctx_when_smaller(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """num_ctx=None on chat mode uses the training context when smaller than the cap."""
        from lilbee.providers.llama_cpp_provider import load_llama
        from lilbee.providers.model_cache import MODE_CHAT

        cfg.num_ctx = None
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "llama",
            "llama.context_length": "4096",
        }

        load_llama(models_dir / "test-model.gguf", mode=MODE_CHAT)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096

    def test_embed_mode_still_uses_training_ctx_when_unset(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Embedding models still get n_ctx=0 (full training context); regression guard."""
        from lilbee.providers.llama_cpp_provider import load_llama
        from lilbee.providers.model_cache import MODE_EMBED

        cfg.num_ctx = None
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "2048",
        }

        load_llama(models_dir / "test-model.gguf", mode=MODE_EMBED)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 0


class TestProviderInvalidateLoadCache:
    def test_invalidate_load_cache_clears_native_cache(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """invalidate_load_cache() with no path drops every cached model."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider
        from lilbee.providers.model_cache import MODE_CHAT, MODE_EMBED

        cfg.num_ctx = 8192
        mock_llama_cpp.Llama.return_value.metadata = {}
        provider = LlamaCppProvider()
        try:
            provider._cache.load_model(models_dir / "test-model.gguf", mode=MODE_CHAT)
            provider._cache.load_model(models_dir / "test-model.gguf", mode=MODE_EMBED)
            assert provider._cache.get_stats()["loaded_models"] == 2

            provider.invalidate_load_cache()

            assert provider._cache.get_stats()["loaded_models"] == 0
        finally:
            provider.shutdown()

    def test_invalidate_load_cache_with_path_evicts_only_that_model(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """invalidate_load_cache(path) leaves entries for other paths intact."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider
        from lilbee.providers.model_cache import MODE_CHAT

        cfg.num_ctx = 8192
        mock_llama_cpp.Llama.return_value.metadata = {}
        provider = LlamaCppProvider()
        try:
            other = models_dir / "other.gguf"
            other.write_bytes(b"fake-gguf")
            provider._cache.load_model(models_dir / "test-model.gguf", mode=MODE_CHAT)
            provider._cache.load_model(other, mode=MODE_CHAT)

            provider.invalidate_load_cache(models_dir / "test-model.gguf")

            stats = provider._cache.get_stats()
            assert stats["loaded_models"] == 1
            assert stats["models"][0]["path"] == str(other)
        finally:
            provider.shutdown()

    def test_protocol_default_is_noop_when_subclass_does_not_override(self) -> None:
        """A backend that inherits LLMProvider WITHOUT overriding the method
        gets the Protocol's no-op default body, which returns None."""
        from lilbee.providers.base import LLMProvider

        class _BackendWithNoOverride(LLMProvider):  # type: ignore[misc]
            """Concrete subclass that doesn't define invalidate_load_cache."""

        backend = _BackendWithNoOverride()
        assert backend.invalidate_load_cache() is None
        assert backend.invalidate_load_cache(Path("/tmp/anything.gguf")) is None

    def test_routing_provider_forwards_only_to_native_side(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """RoutingProvider forwards invalidate_load_cache to the lazily-built
        native provider only; never wakes the SDK side, even after both
        sub-providers have been instantiated."""
        from lilbee.providers.routing_provider import RoutingProvider

        cfg.num_ctx = 8192
        mock_llama_cpp.Llama.return_value.metadata = {}
        routing = RoutingProvider()
        try:
            # No native side built yet -> invalidation is a no-op (does not eagerly construct).
            routing.invalidate_load_cache()
            assert routing._llama_cpp is None

            # Force the native side to instantiate via a method that touches it.
            from lilbee.providers.model_cache import MODE_CHAT

            native = routing._get_llama_cpp()
            native._cache.load_model(models_dir / "test-model.gguf", mode=MODE_CHAT)  # type: ignore[attr-defined]
            assert native._cache.get_stats()["loaded_models"] == 1  # type: ignore[attr-defined]

            assert routing._sdk_provider is None  # SDK side never woke
            routing.invalidate_load_cache()
            assert native._cache.get_stats()["loaded_models"] == 0  # type: ignore[attr-defined]
            # SDK side STILL didn't wake after the invalidation call.
            assert routing._sdk_provider is None
        finally:
            routing.shutdown()


class TestSuppressStderrThreadSafety:
    def test_concurrent_suppress_native_stderr_no_corruption(self) -> None:
        """B3: suppress_native_stderr serializes fd 2 manipulation via _STDERR_LOCK."""
        from lilbee.providers.llama_cpp_provider import suppress_native_stderr

        results: list[int] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def worker(value: int) -> None:
            barrier.wait()
            try:
                result = suppress_native_stderr(lambda v: v * 2, value)
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors during concurrent suppress_native_stderr: {errors}"
        assert sorted(results) == [0, 2, 4, 6]

    def test_suppress_native_stderr_uses_lock(self) -> None:
        """B3: Verify suppress_native_stderr acquires _STDERR_LOCK."""
        from lilbee.providers.llama_cpp_provider import _STDERR_LOCK, suppress_native_stderr

        lock_was_held = []

        def check_lock():
            # If the lock is held (by us), acquire(blocking=False) returns False
            locked = not _STDERR_LOCK.acquire(blocking=False)
            if not locked:
                _STDERR_LOCK.release()
            lock_was_held.append(locked)
            return 42

        result = suppress_native_stderr(check_lock)
        assert result == 42
        assert lock_was_held == [True]
