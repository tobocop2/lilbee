"""Tests for the LlamaCppProvider batching queue and chat lock."""

from __future__ import annotations

import builtins
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.llama_cpp.log_dispatch import import_llama_cpp


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    """Reset provider singleton between tests."""
    from lilbee.core.services import reset_services

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
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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

        assert not overlap_detected.is_set(), "Chat calls overlapped: lock not working"
        for r in results:
            assert r == "ok"
        provider.shutdown()

    def test_chat_stream_through_lock(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Streaming chat works and holds the lock until iteration completes."""
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

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
        from lilbee.providers.llama_cpp import LlamaCppProvider

        mock_llama_cpp.Llama.return_value = mock.MagicMock()

        provider = LlamaCppProvider()
        assert provider._embed_thread.is_alive()

        provider.shutdown()
        assert not provider._embed_thread.is_alive()

    def test_shutdown_during_batch_collection(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Sentinel arriving while collecting a batch stops the worker cleanly."""
        from lilbee.providers.llama_cpp import LlamaCppProvider

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

        from lilbee.providers.llama_cpp.batching import EmbedRequest

        fut: Future[list[list[float]]] = Future()
        provider._embed_queue.put(EmbedRequest(texts=["world"], future=fut))
        provider._embed_queue.put(None)

        # The worker should process "world" then see sentinel and exit
        result2 = fut.result(timeout=5)
        assert result2 == [[1.0]]
        provider._embed_thread.join(timeout=2)
        assert not provider._embed_thread.is_alive()


class TestLockedStreamIteratorClose:
    def test_close_releases_lock(self):
        from lilbee.providers.llama_cpp.provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()
        stream = _LockedStreamIterator(iter([]), lock)
        stream.close()
        assert lock.acquire(blocking=False)
        lock.release()

    def test_close_drain_cap_does_not_hang_on_runaway_model(self):
        """A runaway model (never-closing <think>) must not block close() forever."""
        from lilbee.providers.llama_cpp.provider import (
            _LOCKED_STREAM_DRAIN_CAP,
            _LockedStreamIterator,
        )

        consumed = [0]

        def runaway() -> object:
            while True:
                consumed[0] += 1
                yield {"choices": [{"delta": {"content": "x"}}]}

        lock = threading.Lock()
        lock.acquire()
        stream = _LockedStreamIterator(runaway(), lock)
        stream.close()
        # Lock is released even though the iterator never naturally ends.
        assert lock.acquire(blocking=False)
        lock.release()
        # Drain stopped near the configured cap; no infinite consumption.
        assert consumed[0] <= _LOCKED_STREAM_DRAIN_CAP + 2


class TestLockedStreamIteratorExceptionRelease:
    def test_non_stop_iteration_exception_releases_lock(self):
        """When the underlying response raises a non-StopIteration exception,
        the lock is released and the exception propagates."""
        from lilbee.providers.llama_cpp.provider import _LockedStreamIterator

        def exploding_iter():
            yield {"choices": [{"delta": {"content": "ok"}}]}
            raise ValueError("boom")

        lock = threading.Lock()
        lock.acquire()
        stream = _LockedStreamIterator(exploding_iter(), lock)
        # First call succeeds
        assert next(stream) == "ok"
        # Second call hits the ValueError: lock should be released
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
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        with pytest.raises(ProviderError, match="mmproj"):
            find_mmproj_for_model(models_dir / "test-model.gguf")

    def test_find_mmproj_finds_by_name(self, models_dir: Path) -> None:
        """find_mmproj_for_model finds mmproj files in the models directory."""
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        mmproj = models_dir / "model-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake")
        result = find_mmproj_for_model(models_dir / "test-model.gguf")
        assert result == mmproj

    def test_load_vision_llama_clamps_n_ctx_to_training_window(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """LILBEE_NUM_CTX > vision training_ctx clamps to avoid n_ctx_seq overflow."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")
        cfg.num_ctx = 8192
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                return_value={"context_length": "2048"},
            ),
        ):
            load_vision_llama(models_dir / "test-model.gguf", mmproj_path)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 2048

    def test_load_vision_llama_unset_num_ctx_uses_model_default(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """When LILBEE_NUM_CTX is unset, vision passes n_ctx=0 (model picks)."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")
        cfg.num_ctx = None
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                return_value={"context_length": "2048"},
            ),
        ):
            load_vision_llama(models_dir / "test-model.gguf", mmproj_path)
        assert mock_llama_cpp.Llama.call_args[1]["n_ctx"] == 0

    def test_load_vision_llama_handles_missing_metadata(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """If GGUF metadata read fails, vision honors LILBEE_NUM_CTX verbatim."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")
        cfg.num_ctx = 4096
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                side_effect=RuntimeError("metadata read broken"),
            ),
        ):
            load_vision_llama(models_dir / "test-model.gguf", mmproj_path)
        assert mock_llama_cpp.Llama.call_args[1]["n_ctx"] == 4096


class TestLoadLlamaNCtx:
    def test_default_n_ctx(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """When num_ctx is None, load_llama passes n_ctx=0 and n_batch from metadata."""
        from lilbee.providers.llama_cpp.provider import load_llama
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
        """When num_ctx is set, embedding load clamps to the model's training context."""
        from lilbee.providers.llama_cpp.provider import load_llama
        from lilbee.providers.model_cache import MODE_EMBED

        cfg.num_ctx = 8192
        # Embedding model trains at 8192; cfg.num_ctx fits, no clamp.
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "8192",
        }
        load_llama(models_dir / "test-model.gguf", mode=MODE_EMBED)
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 8192
        assert call_kwargs["n_batch"] == 8192

    def test_embedding_flag_passed(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama passes embedding flag correctly."""
        from lilbee.providers.llama_cpp.provider import load_llama
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

    def test_chat_default_stays_below_safe_ceiling_when_num_ctx_unset(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """num_ctx=None on chat mode never lets a 128K training window dictate KV size."""
        from lilbee.providers.llama_cpp.provider import load_llama
        from lilbee.providers.model_cache import MODE_CHAT

        cfg.num_ctx = None
        # Metadata read returns a huge training context (mimics modern chat GGUFs)
        mock_llama_cpp.Llama.return_value.metadata = {
            "general.architecture": "llama",
            "llama.context_length": "131072",
        }

        load_llama(models_dir / "test-model.gguf", mode=MODE_CHAT)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        # Dynamic picker stays at or below the configured ceiling.
        assert call_kwargs["n_ctx"] <= cfg.num_ctx_max
        assert call_kwargs["n_ctx"] < 131072
        assert call_kwargs["n_ctx"] % 256 == 0

    def test_chat_default_uses_training_ctx_when_smaller(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """num_ctx=None on chat mode uses the training context when smaller than the cap."""
        from lilbee.providers.llama_cpp.provider import load_llama
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
        from lilbee.providers.llama_cpp.provider import load_llama
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
        from lilbee.providers.llama_cpp import LlamaCppProvider
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
        from lilbee.providers.llama_cpp import LlamaCppProvider
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
        from lilbee.providers.llama_cpp.log_dispatch import suppress_native_stderr

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
        from lilbee.providers.llama_cpp.log_dispatch import _STDERR_LOCK, suppress_native_stderr

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

    def test_stderr_suppressed_context_manager_holds_lock(self) -> None:
        """The new ``stderr_suppressed`` context manager takes _STDERR_LOCK
        for the duration of the ``with`` block, not per call inside."""
        from lilbee.providers.llama_cpp.log_dispatch import _STDERR_LOCK, stderr_suppressed

        lock_observations = []

        with stderr_suppressed():
            for _ in range(3):
                locked = not _STDERR_LOCK.acquire(blocking=False)
                if not locked:
                    _STDERR_LOCK.release()
                lock_observations.append(locked)

        assert lock_observations == [True, True, True]
        # Lock is released after the with-block exits.
        assert _STDERR_LOCK.acquire(blocking=False)
        _STDERR_LOCK.release()

    def test_dispatch_batch_holds_lock_once_for_many_texts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: per-chunk wrapping was the visible UI freeze on long PDFs.

        ``_dispatch_batch`` must hoist ``stderr_suppressed`` above the for-loop so
        we acquire ``_STDERR_LOCK`` once per dispatch, not once per text.
        """
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp import log_dispatch
        from lilbee.providers.llama_cpp.batching import EmbedRequest
        from lilbee.providers.llama_cpp.provider import LlamaCppProvider

        # _thread.lock has read-only attributes, so swap the entire object for
        # a counting stand-in that records each acquire.
        counter = _CountingLock()
        monkeypatch.setattr(log_dispatch, "_STDERR_LOCK", counter)

        provider = LlamaCppProvider.__new__(LlamaCppProvider)
        provider._get_embed_llm = lambda: _CountingLlama()  # type: ignore[method-assign]

        future: Future[list[list[float]]] = Future()
        req = EmbedRequest(texts=["a", "b", "c", "d", "e"], future=future)
        provider._dispatch_batch([req])

        assert future.result() == [[1.0], [1.0], [1.0], [1.0], [1.0]]
        # One acquire for the entire 5-text batch, not five.
        assert counter.acquire_count == 1, (
            f"expected 1 _STDERR_LOCK acquire for a 5-text batch, got "
            f"{counter.acquire_count}; per-chunk wrapping reintroduces the UI-freeze regression"
        )


class _CountingLlama:
    """Stand-in for a llama_cpp.Llama that returns a constant embedding."""

    def create_embedding(self, *, input: list[str]) -> dict:
        return {"data": [{"embedding": [1.0]}]}


class _CountingLock:
    """Re-entrant-free lock stand-in that records acquire calls for assertions."""

    def __init__(self) -> None:
        self._real = threading.Lock()
        self.acquire_count = 0

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        self.acquire_count += 1
        return self._real.acquire(*args, **kwargs)

    def release(self) -> None:
        self._real.release()

    def __enter__(self) -> _CountingLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


class TestImportLlamaCpp:
    """``import_llama_cpp`` converts a missing-libvulkan OSError into a ProviderError. (bb-387n)"""

    def test_returns_module_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: hands back the imported module."""
        sentinel = mock.MagicMock(name="llama_cpp_module")
        monkeypatch.setitem(sys.modules, "llama_cpp", sentinel)

        assert import_llama_cpp() is sentinel

    def test_libvulkan_oserror_raises_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare Linux installs without libvulkan get install instructions, not a raw OSError."""
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llama_cpp":
                raise OSError("libvulkan.so.1: cannot open shared object file")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)

        with pytest.raises(ProviderError) as ei:
            import_llama_cpp()
        message = str(ei.value)
        assert "vulkan-icd-loader" in message
        assert "libvulkan1" in message

    def test_unrelated_oserror_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vulkan OSErrors are not swallowed."""
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llama_cpp":
                raise OSError("libsomethingelse.so: not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)

        with pytest.raises(OSError, match="libsomethingelse"):
            import_llama_cpp()


class TestAbortCallbackWiring:
    """Every Llama construction site wires the abort_callback so Ctrl+C can interrupt ggml."""

    @pytest.fixture(autouse=True)
    def _reset_abort_flag(self) -> None:
        from lilbee.providers.llama_cpp.abort_signal import clear_abort

        clear_abort()
        yield
        clear_abort()

    def test_construct_llama_passes_abort_callback(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """``_construct_llama`` injects ``abort_callback`` into every Llama() call."""
        from lilbee.providers.llama_cpp.abort_signal import abort_callback as expected_cb
        from lilbee.providers.llama_cpp.provider import load_llama
        from lilbee.providers.model_cache import MODE_CHAT

        cfg.num_ctx = 2048
        mock_llama_cpp.Llama.return_value.metadata = {}

        load_llama(models_dir / "test-model.gguf", mode=MODE_CHAT)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["abort_callback"] is expected_cb

    def test_read_gguf_metadata_passes_abort_callback(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """``read_gguf_metadata`` wires the abort flag into its vocab-only Llama load."""
        from lilbee.providers.llama_cpp.abort_signal import abort_callback as expected_cb
        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llama_cpp.Llama.return_value.metadata = {}
        read_gguf_metadata(models_dir / "test-model.gguf")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["abort_callback"] is expected_cb
        assert call_kwargs["vocab_only"] is True

    def test_load_vision_llama_passes_abort_callback(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """``load_vision_llama`` wires the abort flag into the vision Llama load."""
        from lilbee.providers.llama_cpp.abort_signal import abort_callback as expected_cb
        from lilbee.providers.mtmd_backend import load_vision_llama

        mmproj_path = models_dir / "test-mmproj-f16.gguf"
        mmproj_path.write_bytes(b"fake-mmproj")
        with mock.patch(
            "lilbee.providers.mtmd_backend.build_vision_chat_handler",
            return_value=mock.MagicMock(),
        ):
            load_vision_llama(models_dir / "test-model.gguf", mmproj_path)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["abort_callback"] is expected_cb

    def test_chat_iterator_releases_lock_when_stream_returns_early(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """A stream that polls ``abort_callback`` and returns early frees the chat lock."""
        from lilbee.providers.llama_cpp import LlamaCppProvider
        from lilbee.providers.llama_cpp.abort_signal import (
            abort_callback as cb,
        )
        from lilbee.providers.llama_cpp.abort_signal import (
            request_abort,
        )

        def streaming_response() -> Any:
            yield {"choices": [{"delta": {"content": "first"}}]}
            # Simulate ggml polling abort_callback every chunk; flip the flag
            # mid-stream and verify the next iteration honors it.
            request_abort()
            if cb():
                return
            yield {"choices": [{"delta": {"content": "should-not-emit"}}]}  # pragma: no cover

        instance = mock.MagicMock()
        instance.create_chat_completion.return_value = streaming_response()
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        try:
            result = provider.chat([{"role": "user", "content": "hi"}], stream=True)
            tokens = list(result)
            assert tokens == ["first"]
            # Lock must be released after a clean stream stop.
            assert provider._chat_lock.acquire(blocking=False)
            provider._chat_lock.release()
        finally:
            provider.shutdown()
