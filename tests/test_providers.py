"""Tests for the LLM provider abstraction layer (mocked: no live servers needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock

import httpx
import pytest

from lilbee.core.config import cfg

if TYPE_CHECKING:
    from lilbee.providers.routing_provider import RoutingProvider
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    """Reset provider singleton between tests."""
    from lilbee.app.services import reset_services

    reset_services()
    yield
    reset_services()


TEST_MODEL_REPO = "org/test-model-GGUF"
TEST_MODEL_FILE = "test-model.gguf"
TEST_MODEL_REF = f"{TEST_MODEL_REPO}/{TEST_MODEL_FILE}"


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with a registered test model."""
    from lilbee.modelhub.registry import ModelManifest, ModelRegistry

    models = tmp_path / "models"
    models.mkdir()
    registry = ModelRegistry(models)

    source = tmp_path / TEST_MODEL_FILE
    source.write_bytes(b"fake-gguf")
    manifest = ModelManifest(
        hf_repo=TEST_MODEL_REPO,
        gguf_filename=TEST_MODEL_FILE,
        size_bytes=9,
        task="chat",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    registry.install(TEST_MODEL_REPO, TEST_MODEL_FILE, source, manifest)
    return models


@pytest.fixture()
def mock_llama_cpp() -> mock.MagicMock:
    """Inject a mock llama_cpp module into sys.modules.

    ``internals.LlamaContext`` is a real class (not a Mock) so the
    ``_llama_n_seq_max`` context manager's monkey-patch of its
    ``__init__`` succeeds. The class is otherwise inert; tests that
    care about Llama kwargs assert against ``mod.Llama.call_args``.
    """
    mod = mock.MagicMock()

    class _StubLlamaContext:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    mod.internals.LlamaContext = _StubLlamaContext
    sys.modules["llama_cpp"] = mod
    yield mod
    sys.modules.pop("llama_cpp", None)


# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_message(self) -> None:
        from lilbee.providers.base import ProviderError

        err = ProviderError("something broke")
        assert str(err) == "something broke"
        assert err.provider == ""

    def test_with_provider(self) -> None:
        from lilbee.providers.base import ProviderError

        err = ProviderError("fail", provider="test")
        assert err.provider == "test"


class TestWorkerErrorMessage:
    """`_worker_error_message` must not double-period when the inner message ends with one."""

    def test_no_double_period_when_exc_ends_with_period(self) -> None:
        from lilbee.providers.llama_cpp import LlamaCppProvider
        from lilbee.providers.worker.transport_pipe import WorkerError

        exc = WorkerError(
            "ProviderError",
            "Model 'X' not found in registry. Install it via the catalog or 'lilbee model pull'.",
            "",
        )
        msg = LlamaCppProvider._worker_error_message("Chat", exc)
        assert ".." not in msg
        assert msg.endswith(". Please try again.")
        assert "'lilbee model pull'. Please try again." in msg

    def test_appends_period_when_exc_has_none(self) -> None:
        from lilbee.providers.llama_cpp import LlamaCppProvider
        from lilbee.providers.worker.transport_pipe import WorkerError

        msg = LlamaCppProvider._worker_error_message(
            "Embed", WorkerError("RuntimeError", "boom", "")
        )
        assert msg == "Embed worker reported an error: RuntimeError: boom. Please try again."


# ---------------------------------------------------------------------------
# LlamaCppProvider
# ---------------------------------------------------------------------------


class TestLlamaCppProvider:
    @pytest.fixture(autouse=True)
    def _shutdown_provider(self, models_dir: Path) -> None:
        """Ensure any LlamaCppProvider created in a test is shut down.
        Also patches resolve_model_path so the daemon embed thread
        doesn't block on registry lookups for test .gguf files.
        """
        cfg.models_dir = models_dir
        cfg.embedding_model = TEST_MODEL_REF
        cfg.chat_model = TEST_MODEL_REF
        self._providers: list = []
        self._resolve_patcher = mock.patch(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            side_effect=lambda m: models_dir / f"{m.rsplit('/', 1)[-1]}",
        )
        self._resolve_patcher.start()
        yield
        for p in self._providers:
            p.shutdown()
        self._resolve_patcher.stop()

    def _make_provider(self) -> object:
        from lilbee.providers.llama_cpp import LlamaCppProvider

        p = LlamaCppProvider()
        self._providers.append(p)
        return p

    def test_list_models(self, models_dir: Path) -> None:
        provider = self._make_provider()
        result = provider.list_models()
        assert result == [TEST_MODEL_REF]

    def test_list_models_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        cfg.models_dir = empty

        provider = self._make_provider()
        assert provider.list_models() == []

    def test_list_models_no_dir(self, tmp_path: Path) -> None:
        cfg.models_dir = tmp_path / "nonexistent"

        provider = self._make_provider()
        assert provider.list_models() == []

    def test_pull_model_raises(self) -> None:
        provider = self._make_provider()
        with pytest.raises(NotImplementedError, match="cannot pull"):
            provider.pull_model("some-model")

    def test_list_chat_models_empty(self) -> None:
        # Native llama-cpp has no frontier-provider catalog.
        provider = self._make_provider()
        assert provider.list_chat_models("openai") == []

    def test_show_model_returns_none(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._make_provider()
        # Override the class-level resolve mock to raise for this test
        with mock.patch(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            assert provider.show_model("some-model") is None

    def test_show_model_returns_metadata_for_resolved_path(self, models_dir: Path) -> None:
        """Success path: resolve + read_gguf_metadata returns a dict."""
        provider = self._make_provider()
        meta = {"architecture": "qwen3", "context_length": "8192"}
        with mock.patch(
            "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
            return_value=meta,
        ):
            assert provider.show_model("test-model") == meta

    def test_get_capabilities_rerank_short_circuits(self) -> None:
        """Rerank refs return only ``["rerank"]`` without touching the loader."""
        provider = self._make_provider()
        with mock.patch(
            "lilbee.providers.llama_cpp.provider._is_rerank_model",
            return_value=True,
        ):
            assert provider.get_capabilities("any/rerank-model") == ["rerank"]

    def test_get_capabilities_completion_only_when_no_mmproj(self, models_dir: Path) -> None:
        """No mmproj sidecar -> ``["completion"]`` only."""
        from lilbee.providers.base import ProviderError

        provider = self._make_provider()
        with (
            mock.patch(
                "lilbee.providers.llama_cpp.provider._is_rerank_model",
                return_value=False,
            ),
            mock.patch(
                "lilbee.providers.llama_cpp.provider.find_mmproj_for_model",
                side_effect=ProviderError("no mmproj"),
            ),
        ):
            assert provider.get_capabilities("test-model") == ["completion"]

    def test_get_capabilities_appends_vision_when_mmproj_present(self) -> None:
        """An mmproj sidecar adds ``"vision"`` to the capability list."""
        provider = self._make_provider()
        with (
            mock.patch(
                "lilbee.providers.llama_cpp.provider._is_rerank_model",
                return_value=False,
            ),
            mock.patch(
                "lilbee.providers.llama_cpp.provider.find_mmproj_for_model",
                return_value=Path("/fake/mmproj.gguf"),
            ),
        ):
            assert provider.get_capabilities("test-model") == ["completion", "vision"]

    def test_get_capabilities_returns_completion_when_resolve_fails(self) -> None:
        """resolve_model_path failure short-circuits to ``["completion"]``."""
        from lilbee.providers.base import ProviderError

        provider = self._make_provider()
        with (
            mock.patch(
                "lilbee.providers.llama_cpp.provider._is_rerank_model",
                return_value=False,
            ),
            mock.patch(
                "lilbee.providers.llama_cpp.provider.resolve_model_path",
                side_effect=ProviderError("not found"),
            ),
        ):
            assert provider.get_capabilities("missing") == ["completion"]

    def testread_gguf_metadata(self, models_dir: Path) -> None:
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llm = MagicMock()
        mock_llm.metadata = {
            "general.architecture": "qwen3",
            "general.name": "Qwen3 8B",
            "general.file_type": "15",
            "qwen3.context_length": "32768",
            "qwen3.embedding_length": "4096",
            "qwen3.block_count": "32",
            "qwen3.attention.head_count_kv": "8",
            "qwen3.attention.head_count": "32",
            "qwen3.attention.key_length": "128",
            "qwen3.attention.value_length": "128",
            "tokenizer.chat_template": "{% if messages %}...",
        }
        with patch("llama_cpp.Llama", return_value=mock_llm):
            result = read_gguf_metadata(models_dir / "test-model.gguf")
        assert result["architecture"] == "qwen3"
        assert result["context_length"] == "32768"
        assert result["embedding_length"] == "4096"
        assert result["chat_template"] == "{% if messages %}..."
        assert result["name"] == "Qwen3 8B"
        # KV-shape fields surfaced for the dynamic n_ctx picker.
        assert result["block_count"] == "32"
        assert result["head_count_kv"] == "8"
        assert result["head_count"] == "32"
        assert result["key_length"] == "128"
        assert result["value_length"] == "128"
        mock_llm.close.assert_called_once()

    def testread_gguf_metadata_empty(self, models_dir: Path) -> None:
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llm = MagicMock()
        mock_llm.metadata = {}
        with patch("llama_cpp.Llama", return_value=mock_llm):
            result = read_gguf_metadata(models_dir / "test-model.gguf")
        assert result is None

    def testload_llama_sets_n_batch_for_embedding(self, models_dir: Path) -> None:
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None
        with (
            patch("llama_cpp.Llama") as mock_llama_cls,
            patch(
                "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                return_value={"context_length": "2048"},
            ),
        ):
            load_llama(models_dir / "test-model.gguf", mode="embed")
            call_kwargs = mock_llama_cls.call_args[1]
            assert call_kwargs["n_batch"] == 2048
            assert call_kwargs["n_ubatch"] == 2048
            assert call_kwargs["embedding"] is True

    def testload_llama_no_n_batch_for_chat(self, models_dir: Path) -> None:
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        with patch("llama_cpp.Llama"):
            load_llama(models_dir / "test-model.gguf", mode="chat")
            import llama_cpp

            call_kwargs = llama_cpp.Llama.call_args[1]
            assert "n_batch" not in call_kwargs

    def testload_llama_rerank_sets_pooling_type_rank(self, models_dir: Path) -> None:
        """mode='rerank' wires ``LLAMA_POOLING_TYPE_RANK`` into the Llama kwargs."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        fake_llama = mock.MagicMock()
        fake_module = mock.MagicMock()
        fake_module.Llama = fake_llama
        fake_module.LLAMA_POOLING_TYPE_RANK = 4

        # _llama_n_seq_max patches LlamaContext.__init__; provide a real
        # class so the monkey-patch assignment succeeds.
        class _StubCtx:
            def __init__(self, *_a, **_kw) -> None:
                pass

        fake_module.internals.LlamaContext = _StubCtx
        with (
            patch(
                "lilbee.providers.llama_cpp.provider.import_llama_cpp",
                return_value=fake_module,
            ),
            patch.dict("sys.modules", {"llama_cpp": fake_module}),
            patch(
                "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                return_value={"context_length": "2048"},
            ),
        ):
            load_llama(models_dir / "test-model.gguf", mode="rerank")
            assert fake_llama.call_args[1]["pooling_type"] == 4

    def testload_llama_abort_callback_override_replaces_default(self, models_dir: Path) -> None:
        """abort_callback_override threads a worker-side callback into Llama kwargs."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        fake_llama = mock.MagicMock()
        fake_module = mock.MagicMock()
        fake_module.Llama = fake_llama
        sentinel = lambda _u=None: False  # noqa: E731 -- intentional one-liner sentinel
        with patch(
            "lilbee.providers.llama_cpp.provider.import_llama_cpp",
            return_value=fake_module,
        ):
            load_llama(
                models_dir / "test-model.gguf",
                mode="chat",
                abort_callback_override=sentinel,
            )
            assert fake_llama.call_args[1]["abort_callback"] is sentinel

    def testload_llama_wraps_context_failure_with_diagnostic(
        self, models_dir: Path, tmp_path: Path
    ) -> None:
        """Opaque ``Failed to create llama_context`` is rewrapped with diagnostic context."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        model = models_dir / "tiny.gguf"
        model.write_bytes(b"x" * 1024)

        with (
            patch("llama_cpp.Llama", side_effect=ValueError("Failed to create llama_context")),
            pytest.raises(ValueError) as exc_info,
        ):
            load_llama(model, mode="chat")

        msg = str(exc_info.value)
        assert "tiny.gguf" in msg
        assert "n_ctx=" in msg
        assert "Failed to create llama_context" in msg

    def testload_llama_does_not_wrap_unrelated_value_errors(self, models_dir: Path) -> None:
        """ValueErrors that aren't the two known load-failure messages pass through unchanged."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        with (
            patch("llama_cpp.Llama", side_effect=ValueError("totally unrelated")),
            pytest.raises(ValueError, match="totally unrelated") as exc_info,
        ):
            load_llama(models_dir / "test-model.gguf", mode="chat")
        # bare re-raise preserves the original chain; the exception isn't its own cause
        assert exc_info.value.__cause__ is None

    def testload_llama_chat_passes_flash_attn_by_default(self, models_dir: Path) -> None:
        """Chat mode enables flash attention to halve the KV cache padding waste."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "auto"
        with patch("llama_cpp.Llama") as mock_llama_cls:
            load_llama(models_dir / "test-model.gguf", mode="chat")
            assert mock_llama_cls.call_args[1].get("flash_attn") is True

    def testload_llama_chat_skips_flash_attn_when_disabled(self, models_dir: Path) -> None:
        """LILBEE_FLASH_ATTENTION=0 leaves the kwarg unset (llama-cpp default)."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "0"
        try:
            with patch("llama_cpp.Llama") as mock_llama_cls:
                load_llama(models_dir / "test-model.gguf", mode="chat")
                assert "flash_attn" not in mock_llama_cls.call_args[1]
        finally:
            cfg.flash_attention = "auto"

    def testload_llama_falls_back_when_flash_attn_unsupported(self, models_dir: Path) -> None:
        """Older llama-cpp-python builds reject flash_attn=True; we drop it and retry."""
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "auto"
        instance = MagicMock()
        call_log: list[dict[str, object]] = []

        def fake_llama(**kwargs: object) -> object:
            call_log.append(kwargs)
            if kwargs.get("flash_attn"):
                raise TypeError("Llama.__init__() got an unexpected keyword argument 'flash_attn'")
            return instance

        with patch("llama_cpp.Llama", side_effect=fake_llama):
            result = load_llama(models_dir / "test-model.gguf", mode="chat")
        assert result is instance
        assert len(call_log) == 2
        assert call_log[0].get("flash_attn") is True
        assert "flash_attn" not in call_log[1]

    def testload_llama_retries_with_halved_ctx_on_oom(self, models_dir: Path) -> None:
        """A llama_context load failure halves n_ctx and retries before wrapping the error."""
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "0"  # keep the call shape simple
        instance = MagicMock()
        ctx_seen: list[int] = []

        def fake_llama(**kwargs: object) -> object:
            ctx_seen.append(int(kwargs["n_ctx"]))
            if int(kwargs["n_ctx"]) > 1024:
                raise ValueError("Failed to create llama_context")
            return instance

        try:
            with patch("llama_cpp.Llama", side_effect=fake_llama):
                result = load_llama(models_dir / "test-model.gguf", mode="chat")
            assert result is instance
            assert ctx_seen[0] == 4096
            assert ctx_seen[-1] <= 1024
            assert len(ctx_seen) >= 2
        finally:
            cfg.flash_attention = "auto"

    def testload_llama_resolves_n_gpu_layers_modes(self, models_dir: Path) -> None:
        """LILBEE_N_GPU_LAYERS supports 'auto', 'cpu', explicit int, and falls back on garbage."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "0"
        try:
            cases = [
                ("cpu", 0),
                ("auto", -1),
                ("12", 12),
                ("not-an-int", -1),
            ]
            for raw, expected in cases:
                cfg.n_gpu_layers = raw
                with patch("llama_cpp.Llama") as mock_llama_cls:
                    load_llama(models_dir / "test-model.gguf", mode="chat")
                    assert mock_llama_cls.call_args[1]["n_gpu_layers"] == expected
        finally:
            cfg.n_gpu_layers = "auto"
            cfg.flash_attention = "auto"

    def testload_llama_threads_main_gpu_when_set(self, models_dir: Path) -> None:
        """``cfg.main_gpu`` reaches the Llama() constructor; default leaves it absent."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "0"
        try:
            cfg.main_gpu = None
            with patch("llama_cpp.Llama") as mock_llama_cls:
                load_llama(models_dir / "test-model.gguf", mode="chat")
                assert "main_gpu" not in mock_llama_cls.call_args[1]

            cfg.main_gpu = 1
            with patch("llama_cpp.Llama") as mock_llama_cls:
                load_llama(models_dir / "test-model.gguf", mode="chat")
                assert mock_llama_cls.call_args[1]["main_gpu"] == 1
        finally:
            cfg.main_gpu = None
            cfg.flash_attention = "auto"

    def testapply_kv_cache_type_skips_when_internal_module_missing(self) -> None:
        """Older llama-cpp-python without ``llama_cpp.llama_cpp`` skips the KV-quant kwargs."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import _apply_kv_cache_type

        kwargs: dict[str, object] = {}
        with (
            patch.object(cfg, "kv_cache_type", "q8_0"),
            patch("lilbee.providers.llama_cpp.provider._ggml_type_map", return_value=None),
        ):
            _apply_kv_cache_type(kwargs)
        assert "type_k" not in kwargs
        assert "type_v" not in kwargs

    def testload_llama_oom_retry_halves_embed_batch_sizes(self, models_dir: Path) -> None:
        """OOM retry on embed loads halves n_batch and n_ubatch alongside n_ctx.

        Embed loads use the model's training context (8192 here) regardless of
        ``cfg.num_ctx`` so a chat-tuned setting doesn't clamp the rerank pair
        size; the OOM retry path bisects from that starting value.
        """
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096  # ignored for embed; kept for parity with chat tests
        cfg.flash_attention = "0"
        instance = MagicMock()
        seen: list[dict[str, int]] = []

        def fake_llama(**kwargs: object) -> object:
            seen.append({k: int(kwargs[k]) for k in ("n_ctx", "n_batch", "n_ubatch")})
            # Succeeds at n_ctx <= 2048 (third halving from 8192).
            if int(kwargs["n_ctx"]) > 2048:
                raise ValueError("Failed to create llama_context")
            return instance

        try:
            with (
                patch("llama_cpp.Llama", side_effect=fake_llama),
                patch(
                    "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                    return_value={"context_length": "8192"},
                ),
            ):
                load_llama(models_dir / "test-model.gguf", mode="embed")
            # First attempt at the model's training ctx (8192); retry halves
            # n_ctx + n_batch + n_ubatch together until the load succeeds.
            assert seen[0]["n_ctx"] == 8192
            assert seen[0]["n_batch"] == 8192
            assert seen[0]["n_ubatch"] == 8192
            assert seen[-1]["n_ctx"] <= 2048
            assert seen[-1]["n_batch"] == seen[-1]["n_ctx"]
            assert seen[-1]["n_ubatch"] == seen[-1]["n_ctx"]
        finally:
            cfg.flash_attention = "auto"

    def testhalve_ctx_for_retry_returns_false_with_no_n_ctx(self) -> None:
        """``_halve_ctx_for_retry`` is a no-op when n_ctx is missing or zero."""
        from lilbee.providers.llama_cpp.provider import _halve_ctx_for_retry

        kwargs: dict[str, object] = {}
        assert _halve_ctx_for_retry(kwargs, ValueError("oom")) is False

    def testkv_cache_type_rejects_unknown_values_at_assignment(self) -> None:
        """Pydantic enforces the KvCacheType enum; unknown labels raise at assignment."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be"):
            cfg.kv_cache_type = "totally-not-a-real-type"  # type: ignore[assignment]

    def testload_llama_kv_cache_type_q8_0_passes_ggml_type_to_llama(self, models_dir: Path) -> None:
        """LILBEE_KV_CACHE_TYPE=q8_0 maps to llama-cpp-python's GGML_TYPE_Q8_0 constant."""
        from unittest.mock import patch

        import llama_cpp.llama_cpp as _llc

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        cfg.flash_attention = "0"
        cfg.kv_cache_type = "q8_0"
        try:
            with patch("llama_cpp.Llama") as mock_llama_cls:
                load_llama(models_dir / "test-model.gguf", mode="chat")
                kwargs = mock_llama_cls.call_args[1]
                assert kwargs["type_k"] == _llc.GGML_TYPE_Q8_0
                assert kwargs["type_v"] == _llc.GGML_TYPE_Q8_0
        finally:
            cfg.kv_cache_type = "f16"
            cfg.flash_attention = "auto"

    def testload_llama_unrelated_typeerror_propagates(self, models_dir: Path) -> None:
        """A TypeError that isn't about flash_attn passes through unchanged."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 4096
        with (
            patch("llama_cpp.Llama", side_effect=TypeError("totally unrelated")),
            pytest.raises(TypeError, match="totally unrelated"),
        ):
            load_llama(models_dir / "test-model.gguf", mode="chat")

    def testload_llama_oom_at_min_ctx_raises_diagnostic(self, models_dir: Path) -> None:
        """When n_ctx is already at the floor, OOM retry gives up and wraps the error."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 512  # Already at the floor; halving can't progress.
        cfg.flash_attention = "0"
        try:
            model = models_dir / "tiny.gguf"
            model.write_bytes(b"x" * 1024)
            with (
                patch("llama_cpp.Llama", side_effect=ValueError("Failed to create llama_context")),
                pytest.raises(ValueError, match=r"Failed to load tiny\.gguf"),
            ):
                load_llama(model, mode="chat")
        finally:
            cfg.flash_attention = "auto"

    def testload_llama_dynamic_ctx_falls_back_to_static_cap_on_psutil_failure(
        self, models_dir: Path
    ) -> None:
        """If memory accounting raises (psutil missing/broken), use min(training, default)."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None
        cfg.flash_attention = "0"
        try:
            with (
                patch("llama_cpp.Llama") as mock_llama_cls,
                patch(
                    "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                    return_value={"context_length": "4096"},
                ),
                patch(
                    "lilbee.providers.model_cache.get_available_memory",
                    side_effect=OSError("psutil broken"),
                ),
            ):
                load_llama(models_dir / "test-model.gguf", mode="chat")
                # Falls back to min(4096, DEFAULT_NUM_CTX=8192) -> 4096
                assert mock_llama_cls.call_args[1]["n_ctx"] == 4096
        finally:
            cfg.flash_attention = "auto"

    def testload_llama_dynamic_ctx_handles_bad_training_ctx_metadata(
        self, models_dir: Path
    ) -> None:
        """A non-numeric context_length in metadata falls back to DEFAULT_NUM_CTX."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import (
            DEFAULT_NUM_CTX,
            load_llama,
        )

        cfg.num_ctx = None
        cfg.flash_attention = "0"
        try:
            with (
                patch("llama_cpp.Llama") as mock_llama_cls,
                patch(
                    "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                    return_value={"context_length": "not-a-number"},
                ),
                patch(
                    "lilbee.providers.model_cache.get_available_memory",
                    return_value=64 * 1024**3,
                ),
            ):
                load_llama(models_dir / "test-model.gguf", mode="chat")
                # With DEFAULT_NUM_CTX as the training fallback and 64 GB
                # available memory, the picker is bounded by that fallback.
                assert mock_llama_cls.call_args[1]["n_ctx"] <= DEFAULT_NUM_CTX
        finally:
            cfg.flash_attention = "auto"

    def testload_llama_dynamic_ctx_picks_smaller_for_tight_memory(self, models_dir: Path) -> None:
        """When LILBEE_NUM_CTX is unset, n_ctx is sized to the host's free memory."""
        from unittest.mock import patch

        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None
        cfg.flash_attention = "0"
        meta = {
            "context_length": "131072",
            "block_count": "32",
            "head_count_kv": "8",
            "key_length": "128",
            "value_length": "128",
        }
        try:
            with (
                patch("llama_cpp.Llama") as mock_llama_cls,
                patch(
                    "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
                    return_value=meta,
                ),
                patch(
                    "lilbee.providers.llama_cpp.provider.cfg.gpu_memory_fraction",
                    0.75,
                ),
                patch(
                    "lilbee.providers.model_cache.get_available_memory",
                    return_value=(8 * 1024**3),
                ),
            ):
                load_llama(models_dir / "test-model.gguf", mode="chat")
                ctx_used = mock_llama_cls.call_args[1]["n_ctx"]
            assert 512 <= ctx_used <= 16384
            assert ctx_used % 256 == 0
            # Should be much smaller than the 131072 training window on a tight host.
            assert ctx_used < 131072
        finally:
            cfg.num_ctx = None
            cfg.flash_attention = "auto"

    def testload_llama_routes_llama_logs_through_python_logger(
        self, models_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """load_llama installs the llama_log callback; ggml WARN demotes to Python INFO."""
        import logging
        from unittest.mock import patch

        from lilbee.providers.llama_cpp import log_dispatch, provider

        snap = log_dispatch._dispatcher.snapshot()
        log_dispatch._dispatcher.reset()
        try:
            with patch("llama_cpp.Llama"), patch("llama_cpp.llama_log_set") as mock_set:
                provider.load_llama(models_dir / "test-model.gguf", mode="chat")
                assert log_dispatch._dispatcher.installed is True
                mock_set.assert_called_once()

            with caplog.at_level(logging.INFO, logger="lilbee.llama_cpp"):
                log_dispatch._dispatcher.dispatch(2, b"init: embeddings required\n", None)
            records = [r for r in caplog.records if r.name == "lilbee.llama_cpp"]
            assert records, "no lilbee.llama_cpp records captured"
            assert records[-1].levelno == logging.INFO
            assert "embeddings required" in records[-1].message
        finally:
            log_dispatch._dispatcher.restore(snap)

    def testllama_log_dispatch_promotes_errors(self) -> None:
        """ggml ERROR maps to Python ERROR so real failures surface at the default level."""
        import logging

        from lilbee.providers.llama_cpp import log_dispatch

        snap = log_dispatch._dispatcher.snapshot()
        log_dispatch._dispatcher.pending.clear()
        log_dispatch._dispatcher.pending_level = 1  # _GGML_LOG_LEVEL_INFO
        logger = logging.getLogger("lilbee.llama_cpp")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger.addHandler(handler)
        try:
            log_dispatch._dispatcher.dispatch(3, b"fatal: out of memory\n", None)
        finally:
            logger.removeHandler(handler)
            log_dispatch._dispatcher.restore(snap)
        assert any(r.levelno == logging.ERROR and "out of memory" in r.message for r in records)

    def testllama_log_dispatch_coalesces_continuation_chunks(self) -> None:
        """CONT chunks buffer until a newline; a new non-CONT line also flushes the buffer."""
        import logging

        from lilbee.providers.llama_cpp import log_dispatch

        snap = log_dispatch._dispatcher.snapshot()
        log_dispatch._dispatcher.pending.clear()
        logger = logging.getLogger("lilbee.llama_cpp")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger.addHandler(handler)
        prior_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            # WARN starts a record
            log_dispatch._dispatcher.dispatch(2, b"loading model:", None)
            # CONT extends it (no newline yet -> still buffered)
            log_dispatch._dispatcher.dispatch(5, b" qwen3-0.6b", None)
            assert records == []
            # CONT with newline -> flush
            log_dispatch._dispatcher.dispatch(5, b" Q4_K_M\n", None)
            assert records, "newline should have flushed buffer"
            assert "loading model: qwen3-0.6b Q4_K_M" in records[-1].message
            records.clear()

            # Buffered chunk without newline; a new non-CONT message must
            # flush the prior buffer before starting fresh.
            log_dispatch._dispatcher.dispatch(2, b"first line", None)
            log_dispatch._dispatcher.dispatch(2, b"second line\n", None)
            messages = [r.message for r in records]
            assert "first line" in messages
            assert "second line" in messages
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prior_level)
            log_dispatch._dispatcher.restore(snap)

    def testllama_log_demotes_known_advisory_errors_to_warning(self) -> None:
        """Tokenizer / KV-cache advisories emit at GGML ERROR but aren't load failures."""
        import logging

        from lilbee.providers.llama_cpp import log_dispatch as prov

        snap = prov._dispatcher.snapshot()
        prov._dispatcher.pending.clear()
        logger = logging.getLogger("lilbee.llama_cpp")
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logger.addHandler(handler)
        try:
            advisories = (
                b"load: special_eos_id is not in special_eog_ids\n",
                b"init: embeddings required but some input tokens were not marked as outputs\n",
                b"llama_context: n_ctx_seq (3072) > n_ctx_train (2048)\n",
            )
            for line in advisories:
                prov._dispatcher.dispatch(3, line, None)
            for r in records:
                assert r.levelno == logging.WARNING, f"advisory '{r.message}' kept at ERROR"
        finally:
            logger.removeHandler(handler)
            prov._dispatcher.restore(snap)

    def testresolve_model_path_direct(self, models_dir: Path, tmp_path: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.llama_cpp.provider import resolve_model_path

            cfg.models_dir = models_dir
            abs_model = tmp_path / "standalone.gguf"
            abs_model.write_bytes(b"standalone-model")
            path = resolve_model_path(str(abs_model))
            assert path == abs_model
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_via_registry(self, models_dir: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.llama_cpp.provider import resolve_model_path

            cfg.models_dir = models_dir
            path = resolve_model_path(TEST_MODEL_REF)
            assert path.exists()
        finally:
            self._resolve_patcher.start()

    def test_resolve_model_path_rejects_bare_name_tag(self, models_dir: Path) -> None:
        """Bare ``name:tag`` strings are not HuggingFace refs and the registry rejects them."""
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.base import ProviderError
            from lilbee.providers.llama_cpp.provider import resolve_model_path

            cfg.models_dir = models_dir
            with pytest.raises(ProviderError, match="not found"):
                resolve_model_path("test-model:latest")
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_not_found(self, models_dir: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.base import ProviderError
            from lilbee.providers.llama_cpp.provider import resolve_model_path

            cfg.models_dir = models_dir
            with pytest.raises(ProviderError, match="not found"):
                resolve_model_path("org/Missing-GGUF/missing.gguf")
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_direct_not_exists(self, models_dir: Path, tmp_path: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.base import ProviderError
            from lilbee.providers.llama_cpp.provider import resolve_model_path

            cfg.models_dir = models_dir
            # Use a real absolute path that doesn't exist (works on all platforms)
            fake_path = str(tmp_path / "nonexistent" / "model.gguf")
            with pytest.raises(ProviderError, match="Model file not found"):
                resolve_model_path(fake_path)
        finally:
            self._resolve_patcher.start()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_provider_is_routing(self) -> None:
        from lilbee.providers.factory import create_provider
        from lilbee.providers.routing_provider import RoutingProvider

        cfg.llm_provider = "auto"
        provider = create_provider(cfg)
        assert isinstance(provider, RoutingProvider)

    def test_explicit_llama_cpp(self) -> None:
        from lilbee.providers.factory import create_provider
        from lilbee.providers.llama_cpp import LlamaCppProvider

        cfg.llm_provider = "llama-cpp"
        provider = create_provider(cfg)
        assert isinstance(provider, LlamaCppProvider)

    def test_unknown_provider_raises(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "unknown"
        with pytest.raises(ProviderError, match="Unknown LLM provider"):
            create_provider(cfg)

    def test_services_singleton(self) -> None:
        from lilbee.app.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "llama-cpp"
        p1 = get_services().provider
        p2 = get_services().provider
        assert p1 is p2
        reset_services()

    def test_services_reset_clears_singleton(self) -> None:
        from lilbee.app.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "llama-cpp"
        p1 = get_services().provider
        reset_services()
        p2 = get_services().provider
        assert p1 is not p2
        reset_services()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigProvider:
    def test_default_llm_provider(self, tmp_path) -> None:
        env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("LILBEE_")}
        # Point LILBEE_DATA at a clean tmp dir so the test doesn't pick up a
        # user-local config.toml whose model slots predate strict catalog
        # validation (round 4).
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
        with (
            mock.patch.dict(__import__("os").environ, env, clear=True),
            mock.patch("lilbee.core.settings.get", return_value=None),
        ):
            from lilbee.core.config import Config

            c = Config()
            assert c.llm_provider == "auto"
            assert c.remote_base_url == "http://localhost:11434"
            assert c.llm_api_key == ""

    def test_provider_env_override(self) -> None:
        import os

        with mock.patch.dict(
            os.environ,
            {
                "LILBEE_LLM_PROVIDER": "remote",
                "LILBEE_REMOTE_BASE_URL": "http://myhost:11434",
                "LILBEE_LLM_API_KEY": "sk-key",
            },
        ):
            from lilbee.core.config import Config

            c = Config()
            assert c.llm_provider == "remote"
            assert c.remote_base_url == "http://myhost:11434"
            assert c.llm_api_key == "sk-key"

    def test_models_dir_uses_canonical_location(self, tmp_path: Path) -> None:
        """models_dir always uses the canonical shared location, not data_root."""
        import os

        from lilbee.core.system import canonical_models_dir

        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(tmp_path / "test-lilbee")}):
            from lilbee.core.config import Config

            c = Config()
            assert c.models_dir == canonical_models_dir()


# ---------------------------------------------------------------------------
# RoutingProvider
# ---------------------------------------------------------------------------


class TestRoutingProvider:
    @pytest.fixture(autouse=True)
    def _shutdown_provider(self):
        """Ensure all LlamaCppProvider background threads are stopped."""
        self._to_shutdown: list = []
        yield
        for p in self._to_shutdown:
            p.shutdown()

    def _make_provider(self) -> RoutingProvider:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        # Track the real llama-cpp provider for shutdown (tests replace it with mocks)
        if rp._llama_cpp is not None:
            self._to_shutdown.append(rp._llama_cpp)
        self._to_shutdown.append(rp)
        return rp

    def test_routes_chat_to_litellm_for_api_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "hello"
        rp._sdk_provider = mock_litellm

        result = rp.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert result == "hello"
        mock_litellm.chat.assert_called_once()
        # Default stream=False path resolves to the str overload in
        # RoutingProvider.chat; the call must reach the backend with stream=False.
        kwargs = mock_litellm.chat.call_args.kwargs
        assert kwargs["stream"] is False

    def test_routes_chat_to_litellm_for_ollama_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "hello"
        rp._sdk_provider = mock_litellm

        result = rp.chat([{"role": "user", "content": "hi"}], model="ollama/qwen3:8b")
        assert result == "hello"
        mock_litellm.chat.assert_called_once()

    def test_routes_chat_with_stream_true_resolves_iterator_overload(self) -> None:
        """stream=True hits the Literal[True] overload and forwards stream=True."""
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()

        def _stream_chunks() -> object:
            yield "hello"
            yield " world"

        mock_litellm.chat.return_value = _stream_chunks()
        rp._sdk_provider = mock_litellm

        result = rp.chat([{"role": "user", "content": "hi"}], stream=True, model="openai/gpt-4o")
        assert list(result) == ["hello", " world"]  # type: ignore[arg-type]
        kwargs = mock_litellm.chat.call_args.kwargs
        assert kwargs["stream"] is True

    def test_routes_vision_ocr_to_llama_cpp_for_native_ref(self) -> None:
        """Native GGUF vision refs reach the llama-cpp vision worker pool."""
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_llama.vision_ocr.return_value = "page text"
        mock_litellm = mock.MagicMock()
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_litellm

        result = rp.vision_ocr(
            b"\x89PNG", "noctrex/LightOnOCR-2-1B-GGUF/LightOnOCR-2-1B-Q4_K_M.gguf", "ocr"
        )
        assert result == "page text"
        mock_llama.vision_ocr.assert_called_once_with(
            b"\x89PNG",
            "noctrex/LightOnOCR-2-1B-GGUF/LightOnOCR-2-1B-Q4_K_M.gguf",
            "ocr",
            timeout=None,
        )
        mock_litellm.vision_ocr.assert_not_called()

    def test_routes_vision_ocr_to_litellm_for_ollama_ref(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_litellm = mock.MagicMock()
        mock_litellm.vision_ocr.return_value = "remote text"
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_litellm

        result = rp.vision_ocr(b"\x89PNG", "ollama/llava:7b", "ocr", timeout=30.0)
        assert result == "remote text"
        mock_litellm.vision_ocr.assert_called_once_with(
            b"\x89PNG", "ollama/llava:7b", "ocr", timeout=30.0
        )
        mock_llama.vision_ocr.assert_not_called()

    def test_routes_chat_to_llama_cpp_for_local_ref(self) -> None:
        """Local HF refs dispatch to llama-cpp regardless of registry contents.

        The routing is strict: a ``<org>/<repo>/<file>.gguf`` shape means
        native. If the registry doesn't have the model, llama-cpp raises
        its own 'not installed' error; routing never falls through to
        litellm.
        """
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "org/Local-GGUF/local-model.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()

    def test_routes_embed_to_litellm_for_ollama_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.embed.return_value = [[0.1, 0.2]]
        rp._sdk_provider = mock_litellm

        cfg.embedding_model = "ollama/nomic-embed-text:latest"
        result = rp.embed(["test"])
        assert result == [[0.1, 0.2]]
        mock_litellm.embed.assert_called_once()

    def test_routes_embed_to_llama_cpp_for_local_ref(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.3, 0.4]]
        rp._llama_cpp = mock_llama

        cfg.embedding_model = (
            "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        )
        result = rp.embed(["test"])
        assert result == [[0.3, 0.4]]

    def test_local_ref_never_falls_through_to_litellm(self) -> None:
        """Local HF refs stay on llama-cpp even when litellm is installed.

        Prefix is the single source of truth: anything that parses as a
        local HF ref dispatches to llama-cpp. Users who want Ollama say
        so with 'ollama/<name>'.
        """
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.9, 1.0]]
        rp._sdk_provider = mock_litellm
        rp._llama_cpp = mock_llama

        cfg.embedding_model = "org/Local-GGUF/embed.gguf"
        result = rp.embed(["test"])
        assert result == [[0.9, 1.0]]
        mock_llama.embed.assert_called_once()
        mock_litellm.embed.assert_not_called()

    def test_list_models_native_only_when_sdk_unavailable(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf"]
        mock_sdk.list_models.assert_not_called()

    def test_list_models_union_when_both_available(self) -> None:
        """Native and remote listings are merged when the SDK backend is installed."""
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_models.return_value = ["ollama/qwen3:8b"]
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf", "ollama/qwen3:8b"]

    def test_list_models_remote_error_returns_native_only(self) -> None:
        """A failing SDK backend doesn't mask the native registry listing."""
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_models.side_effect = RuntimeError("remote down")
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf"]

    def test_get_llama_cpp_caches_instance(self) -> None:
        """``_get_llama_cpp`` memoizes the LlamaCppProvider on first call."""
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        self._to_shutdown.append(rp)
        first = rp._get_llama_cpp()
        self._to_shutdown.append(first)
        second = rp._get_llama_cpp()
        assert first is second

    def test_get_sdk_provider_caches_instance(self) -> None:
        """``_get_sdk_provider`` memoizes the SdkLLMProvider on first call."""
        rp = self._make_provider()
        first = rp._get_sdk_provider()
        second = rp._get_sdk_provider()
        assert first is second

    def test_list_chat_models_empty_when_sdk_unavailable(self) -> None:
        # list_chat_models must skip the SDK backend entirely when the SDK
        # is not installed; native llama-cpp never has a frontier catalog.
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        assert rp.list_chat_models("openai") == []
        mock_sdk.list_chat_models.assert_not_called()

    def test_list_chat_models_delegates_through_sdk_provider(self) -> None:
        # Pins the suppression chain: list_chat_models must reach the SDK
        # provider (not the adapter directly), so SdkLLMProvider._ensure_initialized
        # can apply cfg.json_mode before any SDK import inside the backend.
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_chat_models.return_value = ["openai/gpt-4o", "openai/gpt-4o-mini"]
        rp._sdk_provider = mock_sdk

        result = rp.list_chat_models("openai")

        assert result == ["openai/gpt-4o", "openai/gpt-4o-mini"]
        mock_sdk.list_chat_models.assert_called_once_with("openai")

    def test_show_model_delegates_by_prefix(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.show_model.return_value = {"parameters": "temp 0.7"}
        rp._sdk_provider = mock_litellm

        result = rp.show_model("ollama/qwen3:8b")
        assert result == {"parameters": "temp 0.7"}
        mock_litellm.show_model.assert_called_once_with("ollama/qwen3:8b")

    def test_show_model_local_ref_uses_llama_cpp(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.show_model.return_value = None
        rp._llama_cpp = mock_llama

        ref = "org/Local-GGUF/local.gguf"
        result = rp.show_model(ref)
        assert result is None
        mock_llama.show_model.assert_called_once_with(ref)

    def test_pull_model_raises_when_sdk_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        with pytest.raises(ProviderError, match="no pull-capable backend"):
            rp.pull_model("bad-model")
        mock_sdk.pull_model.assert_not_called()

    def test_pull_model_delegates_to_sdk_when_available(self) -> None:
        """With the SDK backend available, pull_model forwards to the SDK provider."""
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        rp._sdk_provider = mock_sdk
        captured: dict[str, object] = {}

        def _on_progress(evt: dict[str, object]) -> None:
            captured["saw"] = evt

        rp.pull_model("ollama/llama3:8b", on_progress=_on_progress)

        mock_sdk.pull_model.assert_called_once_with("ollama/llama3:8b", on_progress=_on_progress)

    def test_chat_with_explicit_api_model_override(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "saw it"
        rp._sdk_provider = mock_litellm

        cfg.chat_model = "org/Local-GGUF/local.gguf"
        result = rp.chat(
            [{"role": "user", "content": "describe"}],
            model="openai/gpt-4o",
        )
        assert result == "saw it"
        mock_litellm.chat.assert_called_once()

    def test_get_capabilities_delegates_by_prefix(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.get_capabilities.return_value = ["completion", "vision"]
        rp._sdk_provider = mock_litellm

        caps = rp.get_capabilities("ollama/llava:7b")
        assert caps == ["completion", "vision"]
        mock_litellm.get_capabilities.assert_called_once_with("ollama/llava:7b")

    def test_invalidate_load_cache_forwards_to_native(self) -> None:
        """``invalidate_load_cache`` releases the native side; SDK has no cache."""
        rp = self._make_provider()
        mock_native = mock.MagicMock()
        rp._llama_cpp = mock_native

        rp.invalidate_load_cache()
        mock_native.invalidate_load_cache.assert_called_once_with(None)

    def test_warm_up_pool_forwards_to_native(self) -> None:
        """``warm_up_pool`` lazily constructs the native provider and warms it."""
        rp = self._make_provider()
        mock_native = mock.MagicMock()
        with mock.patch.object(rp, "_get_llama_cpp", return_value=mock_native):
            rp.warm_up_pool()
        mock_native.warm_up_pool.assert_called_once_with()


# ---------------------------------------------------------------------------
# litellm_available guard
# ---------------------------------------------------------------------------


class TestAllChatModelsFor:
    """Prefix-stripping and chat-mode filtering in litellm catalog reads."""

    def _stub_litellm(
        self, models_by_provider: dict[str, set[str]], model_cost: dict[str, dict[str, str]]
    ) -> mock.MagicMock:
        stub = mock.MagicMock()
        stub.models_by_provider = models_by_provider
        stub.model_cost = model_cost
        return stub

    def test_strips_provider_prefix_from_catalog_names(self) -> None:
        """Names like ``mistral/codestral-latest`` come back bare."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        litellm = self._stub_litellm(
            {"mistral": {"mistral/codestral-latest", "mistral/mistral-large"}},
            {
                "mistral/codestral-latest": {"mode": "chat"},
                "mistral/mistral-large": {"mode": "chat"},
            },
        )
        result = LitellmSdkBackend._all_chat_models_for("mistral", litellm)
        assert result == ["codestral-latest", "mistral-large"]

    def test_passes_through_bare_names_unchanged(self) -> None:
        """OpenAI/Anthropic/Gemini names already lack a prefix and round-trip."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        litellm = self._stub_litellm(
            {"openai": {"gpt-4o", "gpt-4o-mini"}},
            {"gpt-4o": {"mode": "chat"}, "gpt-4o-mini": {"mode": "chat"}},
        )
        result = LitellmSdkBackend._all_chat_models_for("openai", litellm)
        assert result == ["gpt-4o", "gpt-4o-mini"]

    def test_dedupes_bare_and_prefixed_duplicates(self) -> None:
        """``deepseek-chat`` and ``deepseek/deepseek-chat`` collapse to one entry."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        litellm = self._stub_litellm(
            {"deepseek": {"deepseek-chat", "deepseek/deepseek-chat", "deepseek-reasoner"}},
            {
                "deepseek-chat": {"mode": "chat"},
                "deepseek/deepseek-chat": {"mode": "chat"},
                "deepseek-reasoner": {"mode": "chat"},
            },
        )
        result = LitellmSdkBackend._all_chat_models_for("deepseek", litellm)
        assert result == ["deepseek-chat", "deepseek-reasoner"]

    def test_filters_out_non_chat_modes(self) -> None:
        """Embedding-mode entries never appear in the chat catalog."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        litellm = self._stub_litellm(
            {"mistral": {"mistral/mistral-large", "mistral/mistral-embed"}},
            {
                "mistral/mistral-large": {"mode": "chat"},
                "mistral/mistral-embed": {"mode": "embedding"},
            },
        )
        result = LitellmSdkBackend._all_chat_models_for("mistral", litellm)
        assert result == ["mistral-large"]

    def test_unknown_provider_returns_empty(self) -> None:
        """A provider missing from the catalog yields ``[]`` without raising."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        litellm = self._stub_litellm({}, {})
        assert LitellmSdkBackend._all_chat_models_for("nonexistent", litellm) == []


class TestLitellmResponseView:
    """The shape adapter that owns getattr-default extraction over litellm responses."""

    def test_message_content_handles_missing_message(self) -> None:
        """A choice without a .message attribute (or message=None) yields ''."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        choice = mock.MagicMock(spec=[])  # spec=[] -> no attrs at all
        response = mock.MagicMock(choices=[choice])
        view = _LitellmResponseView(response)
        assert view.message_content == ""

    def test_message_content_handles_no_choices(self) -> None:
        """A response with an empty choices list yields '' for both shapes."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        response = mock.MagicMock(choices=[])
        view = _LitellmResponseView(response)
        assert view.message_content == ""
        assert view.delta_content == ""
        assert view.finish_reason is None

    def test_delta_content_handles_missing_delta(self) -> None:
        """A streaming chunk whose first choice has no .delta attribute yields ''."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        choice = mock.MagicMock(spec=[])
        response = mock.MagicMock(choices=[choice])
        view = _LitellmResponseView(response)
        assert view.delta_content == ""

    def test_delta_content_with_nonempty_delta(self) -> None:
        """The happy stream-path: delta.content carries the chunk text."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        choice = mock.MagicMock()
        choice.delta = mock.MagicMock(content="hello world")
        response = mock.MagicMock(choices=[choice])
        view = _LitellmResponseView(response)
        assert view.delta_content == "hello world"


class TestLitellmAvailable:
    """Exercises the un-patched ``litellm_available`` import probe."""

    @pytest.mark.real_litellm_probe
    def test_returns_false_when_not_installed(self) -> None:
        from lilbee.providers.litellm_sdk import litellm_available

        litellm_available.cache_clear()
        with mock.patch.dict("sys.modules", {"litellm": None}):
            assert litellm_available() is False
        litellm_available.cache_clear()

    @pytest.mark.real_litellm_probe
    def test_returns_true_when_module_present(self) -> None:
        from lilbee.providers.litellm_sdk import litellm_available

        litellm_available.cache_clear()
        with mock.patch.dict("sys.modules", {"litellm": mock.MagicMock()}):
            assert litellm_available() is True
        litellm_available.cache_clear()


class TestRequireLitellm:
    @pytest.mark.real_litellm_probe
    def test_raises_provider_error_with_install_hint(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_sdk import _require_litellm

        with (
            mock.patch.dict("sys.modules", {"litellm": None}),
            pytest.raises(ProviderError, match="lilbee\\[litellm\\] extra"),
        ):
            _require_litellm()

    def test_factory_raises_when_litellm_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import create_provider
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        cfg.llm_provider = "remote"
        with (
            mock.patch.object(LitellmSdkBackend, "available", return_value=False),
            pytest.raises(ProviderError, match="SDK backend adapter is not installed"),
        ):
            create_provider(cfg)


class TestLiteLLMShowModelCapabilities:
    """Tests for SdkLLMProvider.show_model capabilities parsing via litellm backend."""

    def _make_provider(self) -> SdkLLMProvider:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")

    def test_show_model_returns_capabilities(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {
            "capabilities": ["completion", "vision"],
            "parameters": "temperature 0.7",
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("llava:7b")

        assert result is not None
        assert result["capabilities"] == ["completion", "vision"]
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temperature 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("qwen3:8b")

        assert result is not None
        assert "capabilities" not in result
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_only_capabilities_no_params(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("some-model")

        assert result is not None
        assert result["capabilities"] == ["completion"]

    def test_show_model_empty_returns_none(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("empty-model")

        assert result is None

    def test_show_model_http_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            result = provider.show_model("bad-model")

        assert result is None

    def test_get_capabilities_returns_list(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion", "vision", "tools"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("llava:7b")

        assert caps == ["completion", "vision", "tools"]

    def test_get_capabilities_returns_empty_on_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            caps = provider.get_capabilities("bad-model")

        assert caps == []

    def test_get_capabilities_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temp 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("qwen3:8b")

        assert caps == []


class TestShowModelNotFound:
    def test_returns_none_for_missing_model(self) -> None:
        from lilbee.providers.llama_cpp import LlamaCppProvider

        provider = LlamaCppProvider()
        assert provider.show_model("nonexistent-model-xyz") is None


class TestReadMmprojProjectorType:
    def test_reads_projector_type(self, tmp_path: Path) -> None:
        import struct

        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        # Build a minimal GGUF file with clip.projector_type = "ldp"
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)  # version
        buf += struct.pack("<Q", 0)  # tensor_count
        buf += struct.pack("<Q", 1)  # kv_count = 1
        key = b"clip.projector_type"
        buf += struct.pack("<Q", len(key))
        buf += key
        buf += struct.pack("<I", 8)  # value_type = string
        value = b"ldp"
        buf += struct.pack("<Q", len(value))
        buf += value

        gguf_file = tmp_path / "test_mmproj.gguf"
        gguf_file.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(gguf_file) == "ldp"

    def test_exception_returns_none(self) -> None:
        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        assert read_mmproj_projector_type(Path("/nonexistent/file.gguf")) is None

    def test_non_string_projector_type_returns_none(self, tmp_path: Path) -> None:
        """If clip.projector_type is present but not a string (someone wrote it
        as an int or bool), the reader returns None instead of decoding bytes."""
        import struct

        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        # Build a GGUF where clip.projector_type is value_type=4 (uint32), not string.
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)  # version
        buf += struct.pack("<Q", 0)  # tensor_count
        buf += struct.pack("<Q", 1)  # kv_count = 1
        key = b"clip.projector_type"
        buf += struct.pack("<Q", len(key)) + key
        buf += struct.pack("<I", 4)  # value_type = uint32
        buf += struct.pack("<I", 42)  # bogus integer payload

        gguf_file = tmp_path / "wrong_type_mmproj.gguf"
        gguf_file.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(gguf_file) is None

    def test_reads_projector_type_past_bool_kv(self, tmp_path: Path) -> None:
        """Parser must skip bool KV pairs (1 byte each) to reach projector_type.
        LightOn OCR2's mmproj has ``clip.has_vision_encoder`` (bool) preceding
        ``clip.projector_type``. A bool skip-size regression would over-advance
        the stream and parse_header of the next key would raise.
        """
        import struct

        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 0)
        buf += struct.pack("<Q", 2)  # 2 KV pairs
        bool_key = b"clip.has_vision_encoder"
        buf += struct.pack("<Q", len(bool_key)) + bool_key
        buf += struct.pack("<I", 7)  # bool
        buf += b"\x01"  # single byte
        proj_key = b"clip.projector_type"
        buf += struct.pack("<Q", len(proj_key)) + proj_key
        buf += struct.pack("<I", 8)  # string
        proj_val = b"lightonocr"  # the real regression case
        buf += struct.pack("<Q", len(proj_val)) + proj_val

        f = tmp_path / "mmproj.gguf"
        f.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(f) == "lightonocr"


class TestMtmdLoadVisionLlama:
    """mtmd backend replaces the old projector-type → handler lookup."""

    def test_vision_uses_training_ctx_from_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """Vision load reads ``context_length`` from the GGUF and uses it as n_ctx."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        cfg.num_ctx = 512  # chat-tuned; must NOT clamp vision

        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                return_value={"context_length": "8192"},
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 8192
        assert call_kwargs["n_gpu_layers"] == -1

    def test_vision_threads_main_gpu_when_set(self, mock_llama_cpp: mock.MagicMock) -> None:
        """``cfg.main_gpu`` reaches the vision Llama() constructor when set."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        cfg.num_ctx = None
        cfg.main_gpu = 1
        try:
            with (
                mock.patch(
                    "lilbee.providers.mtmd_backend.read_gguf_metadata",
                    return_value={"context_length": "8192"},
                ),
                mock.patch(
                    "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                    return_value=mock.MagicMock(),
                ),
            ):
                load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
            assert mock_llama_cpp.Llama.call_args[1]["main_gpu"] == 1
        finally:
            cfg.main_gpu = None

    def test_vision_falls_back_to_default_when_metadata_missing(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """If the GGUF has no context_length, vision falls back to the explicit default."""
        from lilbee.providers.mtmd_backend import _VISION_FALLBACK_N_CTX, load_vision_llama

        cfg.num_ctx = None

        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                return_value=None,
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == _VISION_FALLBACK_N_CTX

    def test_without_mmproj_calls_find(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.mtmd_backend import load_vision_llama

        cfg.num_ctx = None

        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.find_mmproj_for_model",
                return_value=Path("found_mmproj.gguf"),
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ) as mock_handler,
        ):
            load_vision_llama(Path("model.gguf"))
        assert mock_llama_cpp.Llama.called
        mock_handler.assert_called_once()

    def test_abort_callback_override_replaces_default(self, mock_llama_cpp: mock.MagicMock) -> None:
        """Passing ``abort_callback_override`` threads a worker-side callback into Llama kwargs."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        cfg.num_ctx = None
        sentinel = lambda _u=None: False  # noqa: E731
        with mock.patch(
            "lilbee.providers.mtmd_backend.build_vision_chat_handler",
            return_value=mock.MagicMock(),
        ):
            load_vision_llama(
                Path("model.gguf"),
                mmproj_path=Path("mmproj.gguf"),
                abort_callback_override=sentinel,
            )
        assert mock_llama_cpp.Llama.call_args[1]["abort_callback"] is sentinel

    def test_resolve_vision_n_ctx_uses_training_context(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Vision load uses the model's training context, not cfg.num_ctx."""
        from lilbee.providers.mtmd_backend import load_vision_llama

        cfg.num_ctx = 8192  # chat-tuned; must not influence vision
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                return_value={"context_length": "4096"},
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        assert mock_llama_cpp.Llama.call_args[1]["n_ctx"] == 4096

    def test_resolve_vision_n_ctx_falls_back_when_metadata_unreadable(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """A metadata read failure falls back to the explicit vision default."""
        from lilbee.providers.mtmd_backend import _VISION_FALLBACK_N_CTX, load_vision_llama

        cfg.num_ctx = 4096
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_gguf_metadata",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch(
                "lilbee.providers.mtmd_backend.build_vision_chat_handler",
                return_value=mock.MagicMock(),
            ),
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        assert mock_llama_cpp.Llama.call_args[1]["n_ctx"] == _VISION_FALLBACK_N_CTX


class TestImportLlamaCpp:
    """``import_llama_cpp`` converts a missing-libvulkan OSError into a ProviderError."""

    def test_returns_module_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: hands back the imported module."""
        from lilbee.providers.llama_cpp.log_dispatch import import_llama_cpp

        sentinel = mock.MagicMock(name="llama_cpp_module")
        monkeypatch.setitem(sys.modules, "llama_cpp", sentinel)
        assert import_llama_cpp() is sentinel

    def test_libvulkan_oserror_raises_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare Linux installs without libvulkan get install instructions, not a raw OSError."""
        import builtins

        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.log_dispatch import import_llama_cpp

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llama_cpp":
                raise OSError("libvulkan.so.1: cannot open shared object file")
            return real_import(name, *args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)

        with pytest.raises(ProviderError) as ei:
            import_llama_cpp()
        assert "libvulkan1" in str(ei.value)

    def test_unrelated_oserror_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vulkan OSErrors are not swallowed."""
        import builtins

        from lilbee.providers.llama_cpp.log_dispatch import import_llama_cpp

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "llama_cpp":
                raise OSError("libsomethingelse.so: not found")
            return real_import(name, *args, **kwargs)  # type: ignore[no-any-return]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
        with pytest.raises(OSError, match="libsomethingelse"):
            import_llama_cpp()

    def test_gpu_devices_sets_visibility_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cfg.gpu_devices`` propagates to every backend's visible-devices env var."""
        from lilbee.core.config import cfg
        from lilbee.providers.llama_cpp.log_dispatch import (
            _GPU_VISIBLE_ENV_VARS,
            import_llama_cpp,
        )

        for name in _GPU_VISIBLE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(cfg, "gpu_devices", "0")
        monkeypatch.setitem(sys.modules, "llama_cpp", mock.MagicMock())

        import_llama_cpp()
        for name in _GPU_VISIBLE_ENV_VARS:
            assert os.environ.get(name) == "0", name

    def test_gpu_devices_none_leaves_env_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ``gpu_devices=None`` the visibility env vars are not set."""
        from lilbee.core.config import cfg
        from lilbee.providers.llama_cpp.log_dispatch import (
            _GPU_VISIBLE_ENV_VARS,
            import_llama_cpp,
        )

        for name in _GPU_VISIBLE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(cfg, "gpu_devices", None)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock.MagicMock())

        import_llama_cpp()
        for name in _GPU_VISIBLE_ENV_VARS:
            assert name not in os.environ, name

    def test_user_env_var_wins_over_cfg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pre-set ``GGML_VK_VISIBLE_DEVICES`` is preserved even when cfg has a value."""
        from lilbee.core.config import cfg
        from lilbee.providers.llama_cpp.log_dispatch import import_llama_cpp

        monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
        monkeypatch.setattr(cfg, "gpu_devices", "0")
        monkeypatch.setitem(sys.modules, "llama_cpp", mock.MagicMock())

        import_llama_cpp()
        assert os.environ["GGML_VK_VISIBLE_DEVICES"] == "1"


class TestReadChatTemplate:
    def test_reads_chat_template(self, tmp_path: Path) -> None:
        import struct

        from lilbee.providers.mtmd_backend import read_chat_template

        template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 0)
        buf += struct.pack("<Q", 1)
        key = b"tokenizer.chat_template"
        buf += struct.pack("<Q", len(key)) + key
        buf += struct.pack("<I", 8)
        value = template.encode("utf-8")
        buf += struct.pack("<Q", len(value)) + value
        f = tmp_path / "model.gguf"
        f.write_bytes(bytes(buf))
        assert read_chat_template(f) == template

    def test_returns_none_on_missing_field(self, tmp_path: Path) -> None:
        """A GGUF without tokenizer.chat_template returns None."""
        import struct

        from lilbee.providers.mtmd_backend import read_chat_template

        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 0)
        buf += struct.pack("<Q", 0)
        f = tmp_path / "empty.gguf"
        f.write_bytes(bytes(buf))
        assert read_chat_template(f) is None

    def test_returns_none_on_read_error(self) -> None:
        from lilbee.providers.mtmd_backend import read_chat_template

        assert read_chat_template(Path("/nonexistent/model.gguf")) is None


class TestBuildVisionChatHandler:
    """Use a stub Llava15ChatHandler so the test stays hermetic."""

    def _patched(self, instances: list[object]):
        class _StubHandler:
            CHAT_FORMAT = "stub-default-format"
            DEFAULT_SYSTEM_MESSAGE = "stub-default"

            def __init__(self, clip_model_path: str, verbose: bool = True):
                self.clip_model_path = clip_model_path
                self.verbose = verbose
                instances.append(self)

        return mock.patch("llama_cpp.llama_chat_format.Llava15ChatHandler", _StubHandler)

    def test_installs_gguf_template(self, tmp_path: Path) -> None:
        from lilbee.providers.mtmd_backend import build_vision_chat_handler

        template = "<|im_start|>user\n<|image_pad|>{{ content.text }}<|im_end|>"
        instances: list[object] = []
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_chat_template",
                return_value=template,
            ),
            self._patched(instances),
        ):
            handler = build_vision_chat_handler(tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
        assert "<|image_pad|>" not in type(handler).CHAT_FORMAT
        assert "{{ content.image_url.url }}" in type(handler).CHAT_FORMAT
        assert type(handler).DEFAULT_SYSTEM_MESSAGE is None

    def test_falls_back_when_no_template(self, tmp_path: Path) -> None:
        from lilbee.providers.mtmd_backend import build_vision_chat_handler

        instances: list[object] = []
        with (
            mock.patch(
                "lilbee.providers.mtmd_backend.read_chat_template",
                return_value=None,
            ),
            self._patched(instances),
        ):
            handler = build_vision_chat_handler(tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
        assert type(handler).CHAT_FORMAT == "stub-default-format"
        assert type(handler).DEFAULT_SYSTEM_MESSAGE is None


class TestAdaptGgufTemplate:
    """``adapt_gguf_template_for_mtmd`` rewrites GGUF image-placeholder tokens."""

    def test_replaces_image_pad(self) -> None:
        from lilbee.providers.mtmd_backend import adapt_gguf_template_for_mtmd

        template = "prefix <|image_pad|> suffix"
        out = adapt_gguf_template_for_mtmd(template)
        assert "<|image_pad|>" not in out
        assert "{{ content.image_url.url }}" in out

    def test_replaces_image_tag(self) -> None:
        from lilbee.providers.mtmd_backend import adapt_gguf_template_for_mtmd

        assert adapt_gguf_template_for_mtmd("A <image> B") == "A {{ content.image_url.url }} B"

    def test_noop_when_no_token(self) -> None:
        from lilbee.providers.mtmd_backend import adapt_gguf_template_for_mtmd

        template = "plain {{ content.image_url.url }} template"
        assert adapt_gguf_template_for_mtmd(template) == template


# ---------------------------------------------------------------------------
# LLMOptions / filter_options
# ---------------------------------------------------------------------------


class TestLLMOptions:
    def test_to_dict_omits_none(self) -> None:
        from lilbee.providers.base import LLMOptions

        opts = LLMOptions(temperature=0.7, top_p=None)
        result = opts.to_dict()
        assert result == {"temperature": 0.7}
        assert "top_p" not in result

    def test_to_dict_all_set(self) -> None:
        from lilbee.providers.base import LLMOptions

        opts = LLMOptions(temperature=0.5, seed=42)
        result = opts.to_dict()
        assert result["temperature"] == 0.5
        assert result["seed"] == 42


class TestFilterOptions:
    def test_filters_valid_options(self) -> None:
        from lilbee.providers.base import filter_options

        result = filter_options({"temperature": 0.8, "seed": 42})
        assert result == {"temperature": 0.8, "seed": 42}

    def test_strips_none_values(self) -> None:
        from lilbee.providers.base import filter_options

        result = filter_options({"temperature": 0.5})
        assert "top_p" not in result


class TestReadGgufMetadata:
    def test_reads_all_fields(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata returns parsed fields."""
        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = {
            "general.architecture": "llama",
            "llama.context_length": 4096,
            "llama.embedding_length": 4096,
            "tokenizer.chat_template": "template",
            "general.file_type": "7",
            "general.name": "Test Model",
        }
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))

        assert result == {
            "architecture": "llama",
            "context_length": "4096",
            "embedding_length": "4096",
            "chat_template": "template",
            "file_type": "7",
            "name": "Test Model",
        }
        mock_llm.close.assert_called_once()

    def test_returns_none_for_empty_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata returns None when no fields found."""
        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = {}
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))
        assert result is None

    def test_handles_none_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata handles None metadata."""
        from lilbee.providers.llama_cpp.gguf_meta import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = None
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))
        assert result is None


class TestLoadLlama:
    def test_embedding_uses_training_ctx_from_metadata(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """Embed load reads context_length from GGUF metadata and uses it as n_ctx."""
        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None

        with mock.patch(
            "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
            return_value={"context_length": "2048"},
        ):
            load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 2048
        assert call_kwargs["n_batch"] == 2048
        assert call_kwargs["n_ubatch"] == 2048
        assert call_kwargs["embedding"] is True

    def test_embedding_no_metadata_defaults_to_2048(self, mock_llama_cpp: mock.MagicMock) -> None:
        """Embed load defaults to 2048 when the GGUF metadata read fails."""
        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None

        with mock.patch(
            "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
            return_value=None,
        ):
            load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 2048
        assert call_kwargs["n_batch"] == 2048

    def test_embedding_ignores_cfg_num_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        """Chat-tuned cfg.num_ctx does not propagate to embed/rerank loads.

        Before this guard, ``min(cfg.num_ctx, embed_train_ctx)`` clamped the
        rerank context to the chat ctx, which produced 'llama_decode
        returned 1' on every rerank pair when a low-RAM user set a small
        cfg.num_ctx. The embed/rerank context is the model's training ctx
        regardless of cfg.num_ctx.
        """
        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = 512  # chat-sized; must NOT clamp embed
        with mock.patch(
            "lilbee.providers.llama_cpp.provider.read_gguf_metadata",
            return_value={"context_length": "8192"},
        ):
            load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 8192
        assert call_kwargs["n_batch"] == 8192

    def test_chat_mode(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama for chat does not set n_batch."""
        from lilbee.providers.llama_cpp.provider import load_llama

        cfg.num_ctx = None

        load_llama(Path("/test.gguf"), mode="chat")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is False
        assert "n_batch" not in call_kwargs


class TestFindMmprojForModel:
    def test_catalog_lookup(self) -> None:
        """find_mmproj_for_model uses catalog lookup first."""
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        with mock.patch(
            "lilbee.catalog.find_mmproj_file",
            return_value=Path("/found.gguf"),
        ):
            result = find_mmproj_for_model(Path("/models/model.gguf"))

        assert result == Path("/found.gguf")

    def test_directory_fallback(self, tmp_path: Path) -> None:
        """find_mmproj_for_model falls back to directory scan."""
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        model_path = tmp_path / "model.gguf"
        model_path.touch()
        mmproj = tmp_path / "model-mmproj-fp16.gguf"
        mmproj.touch()

        with mock.patch(
            "lilbee.catalog.find_mmproj_file",
            return_value=None,
        ):
            result = find_mmproj_for_model(model_path)

        assert result == mmproj

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """find_mmproj_for_model raises ProviderError when no mmproj found."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        model_path = tmp_path / "model.gguf"
        model_path.touch()

        with (
            mock.patch(
                "lilbee.catalog.find_mmproj_file",
                return_value=None,
            ),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(model_path)

    def test_hf_cache_blob_walks_to_snapshots(self, tmp_path: Path) -> None:
        """HF cache resolves main GGUFs to blob paths; the mmproj lives next to the
        snapshot symlink, not in blobs/. find_mmproj_for_model must walk up to the
        sibling snapshots/ tree when the model path lives under blobs/.
        """
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        snap = model_root / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        (snap / "mmproj-f16.gguf").touch()

        with mock.patch("lilbee.catalog.find_mmproj_file", return_value=None):
            result = find_mmproj_for_model(blob_path)
        assert result == snap / "mmproj-f16.gguf"

    def test_hf_cache_blob_without_snapshots_falls_through(self, tmp_path: Path) -> None:
        """blobs/ dir with no sibling snapshots/ tree returns None from the HF
        helper, allowing the flat-dir fallback to take over."""
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        blobs.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        # No snapshots/ sibling and no mmproj in blobs/.

        from lilbee.providers.base import ProviderError

        with (
            mock.patch("lilbee.catalog.find_mmproj_file", return_value=None),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(blob_path)

    def test_hf_cache_snapshots_without_mmproj_falls_through(self, tmp_path: Path) -> None:
        """snapshots/ tree exists but no mmproj GGUF lives in any snapshot --
        the HF helper returns None and the flat-dir fallback takes over."""
        from lilbee.providers.llama_cpp.gguf_meta import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        snap = model_root / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        # snapshot dir is present but contains no *mmproj*.gguf files.

        from lilbee.providers.base import ProviderError

        with (
            mock.patch("lilbee.catalog.find_mmproj_file", return_value=None),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(blob_path)


class TestReadMmprojProjectorTypePartial:
    def test_returns_projector_type(self, tmp_path: Path) -> None:
        """read_mmproj_projector_type reads clip.projector_type from GGUF."""
        import struct

        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        # Build a minimal GGUF with one KV pair: clip.projector_type = "ldp"
        f = tmp_path / "test.gguf"
        with open(f, "wb") as fp:
            fp.write(b"GGUF")  # magic
            fp.write(struct.pack("<I", 3))  # version
            fp.write(struct.pack("<Q", 0))  # tensor count
            fp.write(struct.pack("<Q", 1))  # kv count
            key = b"clip.projector_type"
            fp.write(struct.pack("<Q", len(key)))
            fp.write(key)
            fp.write(struct.pack("<I", 8))  # type 8 = string
            value = b"ldp"
            fp.write(struct.pack("<Q", len(value)))
            fp.write(value)

        result = read_mmproj_projector_type(f)
        assert result == "ldp"

    def test_skips_non_matching_keys(self, tmp_path: Path) -> None:
        """read_mmproj_projector_type skips unrelated keys."""
        import struct

        from lilbee.providers.llama_cpp.gguf_meta import read_mmproj_projector_type

        f = tmp_path / "test.gguf"
        with open(f, "wb") as fp:
            fp.write(b"GGUF")
            fp.write(struct.pack("<I", 3))
            fp.write(struct.pack("<Q", 0))
            fp.write(struct.pack("<Q", 2))  # 2 kv pairs
            # First KV: other.key = "value" (string)
            key1 = b"other.key"
            fp.write(struct.pack("<Q", len(key1)))
            fp.write(key1)
            fp.write(struct.pack("<I", 8))  # string type
            val1 = b"value"
            fp.write(struct.pack("<Q", len(val1)))
            fp.write(val1)
            # Second KV: clip.projector_type = "resampler"
            key2 = b"clip.projector_type"
            fp.write(struct.pack("<Q", len(key2)))
            fp.write(key2)
            fp.write(struct.pack("<I", 8))
            val2 = b"resampler"
            fp.write(struct.pack("<Q", len(val2)))
            fp.write(val2)

        result = read_mmproj_projector_type(f)
        assert result == "resampler"


class TestIsOllama:
    def test_localhost_default_port(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://localhost:11434") is True

    def test_127_default_port(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://127.0.0.1:11434") is True

    def test_ollama_in_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("https://ollama.example.com") is True

    def test_openai_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("https://api.openai.com") is False

    def test_custom_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://myserver:8080") is False


class TestRouteModel:
    """Wire-format routing lives in ``LitellmSdkBackend._route_model``.

    These tests exercise the helper directly so we do not depend on the
    internal composition of ``SdkLLMProvider`` + backend.
    """

    def test_ollama_url_adds_prefix(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        # Bare ollama refs are written with the prefix already; the helper
        # double-prefixes only the rare case where the user typed a bare
        # name on an Ollama base URL.
        ref = parse_model_ref("ollama/qwen3:8b")
        assert _route_model(ref, "http://localhost:11434") == "ollama/qwen3:8b"

    def test_non_ollama_url_no_prefix(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        # API-prefixed refs route as-is on non-Ollama base URLs.
        ref = parse_model_ref("openai/gpt-4o")
        assert _route_model(ref, "https://api.openai.com") == "openai/gpt-4o"

    def test_already_prefixed_passes_through(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("openai/gpt-4o")
        assert _route_model(ref, "http://localhost:11434") == "openai/gpt-4o"

    def test_provider_prefixed_on_non_ollama(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("anthropic/claude-sonnet-4-6")
        assert _route_model(ref, "https://api.anthropic.com") == "anthropic/claude-sonnet-4-6"

    def test_local_ref_on_non_ollama_url_returns_bare_name(self) -> None:
        """Local HF refs route bare (no prefix) when the api_base isn't Ollama."""
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
        assert _route_model(ref, "https://example.com/v1") == ref.name


class TestInjectProviderKeys:
    def test_injects_keys_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-test-openai"
        cfg.anthropic_api_key = "sk-test-anthropic"
        cfg.gemini_api_key = ""

        inject_provider_keys()

        import os

        assert os.environ.get("OPENAI_API_KEY") == "sk-test-openai"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-anthropic"
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_does_not_override_existing_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
        cfg.openai_api_key = "sk-from-config"

        inject_provider_keys()

        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-existing"

    def test_every_provider_key_routes_to_its_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each PROVIDER_KEYS entry round-trips cfg field -> env var.

        Iterates the canonical PROVIDER_KEYS tuple so a newly added
        provider is exercised here without a hand-edited assertion.
        """
        import os

        from lilbee.providers.sdk_backend import PROVIDER_KEYS
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        marker = "sk-pk-{}"
        for _prov, cfg_field, env_var, _label in PROVIDER_KEYS:
            monkeypatch.delenv(env_var, raising=False)
            setattr(cfg, cfg_field, marker.format(env_var))

        inject_provider_keys()

        for _prov, _cfg_field, env_var, _label in PROVIDER_KEYS:
            assert os.environ.get(env_var) == marker.format(env_var)


class TestLiteLLMListModelsRouting:
    def test_ollama_url_uses_api_tags(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "api/tags" in mock_get.call_args[0][0]
        assert result == ["llama3:8b"]

    def test_non_ollama_url_uses_v1_models(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(
            LitellmSdkBackend(), base_url="https://api.openai.com", api_key="sk-test"
        )
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "v1/models" in mock_get.call_args[0][0]
        assert result == ["gpt-4o", "gpt-4o-mini"]

    def test_non_ollama_returns_empty_on_error(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="https://api.openai.com")

        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = provider.list_models()

        assert result == []

    def test_v1_models_sends_auth_header(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(
            LitellmSdkBackend(), base_url="https://api.openai.com", api_key="sk-secret"
        )
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            provider.list_models()

        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-secret"


class TestSdkLLMProviderVisionOcr:
    """``SdkLLMProvider.vision_ocr`` translates to a multipart chat call."""

    def _make_provider(self) -> SdkLLMProvider:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")

    def test_builds_multipart_message_and_routes_to_chat(self) -> None:
        provider = self._make_provider()
        with mock.patch.object(provider, "chat", return_value="page text") as mock_chat:
            result = provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "ocr please")

        assert result == "page text"
        mock_chat.assert_called_once()
        messages = mock_chat.call_args[0][0]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "ocr please"
        assert mock_chat.call_args[1]["model"] == "ollama/llava:7b"
        assert mock_chat.call_args[1]["stream"] is False

    def test_empty_prompt_uses_default_ocr_prompt(self) -> None:
        from lilbee.vision import OCR_PROMPT

        provider = self._make_provider()
        with mock.patch.object(provider, "chat", return_value="ok") as mock_chat:
            provider.vision_ocr(b"\x89PNG", "ollama/llava:7b")

        text_part = mock_chat.call_args[0][0][0]["content"][1]
        assert text_part["text"] == OCR_PROMPT

    def test_positive_timeout_returns_chat_result(self) -> None:
        """A non-expiring positive timeout returns the chat response unchanged."""
        provider = self._make_provider()
        with mock.patch.object(provider, "chat", return_value="ok") as mock_chat:
            result = provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "p", timeout=5.0)

        assert result == "ok"
        mock_chat.assert_called_once()

    def test_timeout_expiry_raises_timeout_error(self) -> None:
        import time

        provider = self._make_provider()

        def slow_chat(*args, **kwargs):
            time.sleep(5)
            return "too late"

        with (
            mock.patch.object(provider, "chat", side_effect=slow_chat),
            pytest.raises(TimeoutError),
        ):
            provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "p", timeout=0.01)

    def test_zero_timeout_returns_chat_result(self) -> None:
        """``timeout=0`` skips the thread pool and returns chat's result."""
        provider = self._make_provider()
        with mock.patch.object(provider, "chat", return_value="ok") as mock_chat:
            result = provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "p", timeout=0)

        assert result == "ok"
        mock_chat.assert_called_once()

    def test_non_string_response_raises_provider_error(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._make_provider()
        with (
            mock.patch.object(provider, "chat", return_value=iter(["streamed"])),
            pytest.raises(ProviderError, match="non-text response"),
        ):
            provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "p")


class TestNeedsApiBase:
    def test_ollama_prefixed_model_needs_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("ollama/qwen3:8b").needs_api_base is True

    def test_local_hf_model_needs_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf").needs_api_base is True

    def test_openai_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("openai/gpt-4o").needs_api_base is False

    def test_anthropic_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("anthropic/claude-sonnet-4-6").needs_api_base is False

    def test_gemini_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("gemini/gemini-pro").needs_api_base is False


class TestChatApiBaseRouting:
    """Verify that chat() omits api_base for non-Ollama provider-prefixed models."""

    def _make_fake_litellm(self) -> mock.MagicMock:
        fake = mock.MagicMock()
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = "hello"
        fake.completion.return_value = resp
        return fake

    def test_ollama_model_passes_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="ollama/qwen3:0.6b")

        call_kwargs = fake.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert call_kwargs["model"] == "ollama/qwen3:0.6b"

    def test_frontier_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    def test_anthropic_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="anthropic/claude-sonnet-4-6")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs

    def test_chat_calls_inject_provider_keys(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with (
            mock.patch.dict("sys.modules", {"litellm": fake}),
            mock.patch("lilbee.providers.sdk_llm_provider.inject_provider_keys") as mock_inject,
        ):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        mock_inject.assert_called_once()


class TestEmbedApiBaseRouting:
    """Verify that embed() omits api_base for non-Ollama provider-prefixed models."""

    def test_ollama_embed_passes_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        cfg.embedding_model = "ollama/nomic-embed-text:latest"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_prefixed_embed_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        cfg.embedding_model = "openai/text-embedding-3-small"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert "api_base" not in call_kwargs


class TestSdkRerank:
    """Coverage for SdkLLMProvider.rerank + LitellmSdkBackend.rerank."""

    def _make_sdk_provider(self):
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")

    def test_rerank_returns_scores_in_candidate_order(self) -> None:
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        fake_response = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = fake_response
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            scores = provider.rerank("q", ["a", "b", "c"])
        assert scores == [0.3, 0.7, 0.9]
        kwargs = fake_litellm.rerank.call_args.kwargs
        assert kwargs["query"] == "q"
        assert kwargs["documents"] == ["a", "b", "c"]
        assert "cohere" in kwargs["model"]

    def test_rerank_empty_candidates_short_circuits(self) -> None:
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        assert provider.rerank("q", []) == []

    def test_supports_rerank_matches_backend_available(self) -> None:
        provider = self._make_sdk_provider()
        with mock.patch(
            "lilbee.providers.litellm_sdk.litellm_available",
            return_value=True,
        ):
            assert provider.supports_rerank() is True
        with mock.patch(
            "lilbee.providers.litellm_sdk.litellm_available",
            return_value=False,
        ):
            assert provider.supports_rerank() is False

    def test_rerank_maps_backend_error_to_provider_error(self) -> None:
        from lilbee.providers.base import ProviderError

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.side_effect = RuntimeError("boom")
        with (
            mock.patch.dict(sys.modules, {"litellm": fake_litellm}),
            pytest.raises(ProviderError, match="Rerank failed"),
        ):
            provider.rerank("q", ["a"])

    def test_rerank_accepts_object_style_response(self) -> None:
        """Pydantic-style responses with attribute access also parse."""
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        item = mock.MagicMock(index=0, relevance_score=0.42)
        response = mock.MagicMock(results=[item], model="cohere/rerank-english-v3.0")
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = response
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            scores = provider.rerank("q", ["a"])
        assert scores == [0.42]

    def test_backend_empty_candidates_skip_sdk(self) -> None:
        """Backend-level empty-candidates guard avoids importing litellm."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.model_ref import parse_model_ref
        from lilbee.providers.sdk_backend import RerankRequest

        backend = LitellmSdkBackend()
        request = RerankRequest(
            ref=parse_model_ref("cohere/rerank-english-v3.0"),
            query="q",
            candidates=[],
        )
        result = backend.rerank(request)
        assert result.scores == []

    def test_backend_forwards_api_key(self) -> None:
        """``api_key`` on the request is forwarded to litellm.rerank kwargs."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.model_ref import parse_model_ref
        from lilbee.providers.sdk_backend import RerankRequest

        backend = LitellmSdkBackend()
        request = RerankRequest(
            ref=parse_model_ref("cohere/rerank-english-v3.0"),
            query="q",
            candidates=["a"],
            api_base="http://localhost:11434",
            api_key="sk-test",
        )
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend.rerank(request)
        assert fake_litellm.rerank.call_args.kwargs["api_key"] == "sk-test"

    def test_provider_wraps_non_provider_error(self) -> None:
        """A non-``ProviderError`` raised inside the backend is re-wrapped."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider as _SdkLLMProvider

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = _SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        with (
            mock.patch.object(provider._backend, "rerank", side_effect=RuntimeError("wire error")),
            pytest.raises(ProviderError, match="Rerank failed: wire error"),
        ):
            provider.rerank("q", ["a"])


class TestIsRerankModel:
    def test_empty_model_returns_false(self) -> None:
        from lilbee.providers.llama_cpp.provider import _is_rerank_model

        assert _is_rerank_model("") is False

    def test_matches_featured_rerank_entry(self) -> None:
        from lilbee.catalog import FEATURED_RERANK
        from lilbee.providers.llama_cpp.provider import _is_rerank_model

        assert FEATURED_RERANK, "catalog must have at least one rerank entry"
        assert _is_rerank_model(FEATURED_RERANK[0].hf_repo) is True

    def test_non_rerank_model_returns_false(self) -> None:
        from lilbee.providers.llama_cpp.provider import _is_rerank_model

        assert _is_rerank_model("org/Definitely-Not-Rerank-GGUF") is False

    def test_substring_of_catalog_name_does_not_match(self) -> None:
        from lilbee.providers.llama_cpp.provider import _is_rerank_model

        assert _is_rerank_model("base") is False
        assert _is_rerank_model("reranker") is False

    def test_full_hf_ref_matches(self) -> None:
        """A full ``hf_repo/filename`` rerank ref also resolves."""
        from lilbee.catalog import FEATURED_RERANK
        from lilbee.providers.llama_cpp.provider import _is_rerank_model

        assert FEATURED_RERANK, "catalog must have at least one rerank entry"
        entry = FEATURED_RERANK[0]
        assert _is_rerank_model(entry.hf_repo) is True


class TestExtractRerankScore:
    """``_extract_rerank_score`` operates on one ``data`` item from a batch response."""

    def test_flat_list_embedding_returns_first_element(self) -> None:
        """llama-cpp-python 0.3.x returns ``list[float]`` with length n_embd=1."""
        from lilbee.providers.llama_cpp.batching import _extract_rerank_score

        assert _extract_rerank_score({"embedding": [0.73]}) == 0.73

    def test_scalar_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.batching import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*float"):
            _extract_rerank_score({"embedding": 0.73})

    def test_nested_list_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.batching import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*list"):
            _extract_rerank_score({"embedding": [[0.42]]})

    def test_empty_embedding_list_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.batching import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*list: \[\]"):
            _extract_rerank_score({"embedding": []})

    def test_non_numeric_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.batching import _extract_rerank_score

        with pytest.raises(ProviderError, match="unexpected score shape"):
            _extract_rerank_score({"embedding": "not-a-number"})

    def test_size_mismatch_at_batch_level_raises(self) -> None:
        """``_rerank_one_call`` (the batch wrapper) catches data-length mismatches."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp.batching import _rerank_one_call

        class _StubLlama:
            def create_embedding(self, *, input):
                # Return one entry for two pairs (mismatch).
                return {"data": [{"embedding": [0.5]}]}

        with pytest.raises(ProviderError, match="returned 1 entries for 2 pairs"):
            _rerank_one_call(_StubLlama(), ["q</s></s>a", "q</s></s>b"])


class TestRoutingProviderRerank:
    """Routing-level rerank dispatch between native llama-cpp and hosted SDK."""

    def _make_provider(self):
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    def test_rerank_routes_hosted_ref_to_sdk(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = True
        mock_sdk.rerank.return_value = [0.9, 0.1]
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        scores = rp.rerank("q", ["a", "b"])
        assert scores == [0.9, 0.1]
        mock_sdk.rerank.assert_called_once_with("q", ["a", "b"])
        mock_llama.rerank.assert_not_called()

    def test_rerank_raises_when_hosted_ref_and_sdk_backend_missing(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = False
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        with pytest.raises(ProviderError, match="hosted rerank backend not available"):
            rp.rerank("q", ["a", "b"])
        mock_sdk.rerank.assert_not_called()

    def test_supports_rerank_disabled_model_always_true(self) -> None:
        rp = self._make_provider()
        cfg.reranker_model = ""
        assert rp.supports_rerank() is True

    def test_supports_rerank_native_delegates_to_llama(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_llama.supports_rerank.return_value = True
        rp._llama_cpp = mock_llama

        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF"
        assert rp.supports_rerank() is True
        mock_llama.supports_rerank.assert_called_once()

    def test_supports_rerank_hosted_delegates_to_sdk(self) -> None:
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        rp._sdk_provider = mock_sdk
        cfg.reranker_model = "cohere/rerank-english-v3.0"

        mock_sdk.supports_rerank.return_value = False
        assert rp.supports_rerank() is False
        mock_sdk.supports_rerank.return_value = True
        assert rp.supports_rerank() is True

    def test_rerank_routes_bare_gguf_to_llama_cpp(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_llama.rerank.return_value = [0.5, 0.5]
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF"
        scores = rp.rerank("q", ["a", "b"])
        assert scores == [0.5, 0.5]
        mock_llama.rerank.assert_called_once_with("q", ["a", "b"])
        mock_sdk.rerank.assert_not_called()

    def test_empty_reranker_model_routes_to_litellm(self) -> None:
        """An empty ``cfg.reranker_model`` is treated as non-native (hosted)."""
        from lilbee.providers.routing_provider import _is_native_rerank_ref

        assert _is_native_rerank_ref("") is False

    def test_native_gguf_ref_routes_native_even_when_not_featured(self) -> None:
        """Any 3+ slash, .gguf-ending ref is recognised as a native reranker.

        Users install community GGUF rerankers that are not in FEATURED_ALL.
        Without this fall-back, the route silently falls through to the SDK
        provider and surfaces a misleading 'hosted rerank backend not
        available' error to the user (bb-4zvk).
        """
        from lilbee.providers.routing_provider import _is_native_rerank_ref

        assert (
            _is_native_rerank_ref(
                "pyarn/bge-reranker-v2-gemma-Q4_K_M-GGUF/bge-reranker-v2-gemma-q4_k_m.gguf"
            )
            is True
        )

    def test_non_gguf_two_part_ref_is_not_native(self) -> None:
        """Two-part refs (e.g. ``cohere/rerank-v3``) still go to the SDK."""
        from lilbee.providers.routing_provider import _is_native_rerank_ref

        assert _is_native_rerank_ref("cohere/rerank-english-v3.0") is False

    def test_rerank_with_empty_model_raises_provider_error(self) -> None:
        """rerank() raises ProviderError when cfg.reranker_model is empty."""
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        cfg.reranker_model = ""
        with pytest.raises(ProviderError, match="No reranker configured"):
            rp.rerank("q", ["a", "b"])


class TestLlamaCppHasRankPooling:
    def test_has_rank_pooling_reports_import_status(self) -> None:
        from lilbee.providers.llama_cpp.provider import _llama_cpp_has_rank_pooling

        fake_mod = mock.MagicMock()
        fake_mod.LLAMA_POOLING_TYPE_RANK = 4
        with mock.patch.dict(sys.modules, {"llama_cpp": fake_mod}):
            assert _llama_cpp_has_rank_pooling() is True
        with mock.patch.dict("sys.modules", {"llama_cpp": None}):
            assert _llama_cpp_has_rank_pooling() is False

    def test_supports_rerank_requires_rank_pooling(self) -> None:
        from lilbee.providers.llama_cpp import LlamaCppProvider

        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        try:
            with mock.patch(
                "lilbee.providers.llama_cpp.provider._llama_cpp_has_rank_pooling",
                return_value=True,
            ):
                assert provider.supports_rerank() is True
            with mock.patch(
                "lilbee.providers.llama_cpp.provider._llama_cpp_has_rank_pooling",
                return_value=False,
            ):
                assert provider.supports_rerank() is False
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()


class TestLlamaCppPdfOcr:
    """``LlamaCppProvider.pdf_ocr`` aggregates streamed pages into a list."""

    @staticmethod
    def _stub_provider(stream_chunks):
        """Build a provider whose pool accessor yields *stream_chunks* per stream call.

        Returns ``(provider, captured)`` where ``captured["payload"]`` is
        the second positional arg passed to the stub accessor's
        ``stream`` method.
        """
        import asyncio

        from lilbee.providers.llama_cpp import LlamaCppProvider

        captured: dict = {}

        async def _aiter():
            for c in stream_chunks:
                yield c

        accessor = mock.MagicMock()

        def _stream(_kind, payload):
            captured["payload"] = payload
            return _aiter()

        accessor.stream = _stream
        runtime = mock.MagicMock()
        runtime.run_sync = lambda coro, *, timeout: asyncio.new_event_loop().run_until_complete(
            coro
        )

        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        provider._get_pool_accessor = lambda *_a, **_kw: accessor
        provider._pool_runtime = lambda: runtime
        return provider, captured

    def test_pdf_ocr_aggregates_streamed_pages_in_order(self) -> None:
        from lilbee.providers.worker.transport import PdfOcrRequest
        from lilbee.vision import PageText, PdfOcrChunk

        chunks = [
            PdfOcrChunk(1, 3, "alpha"),
            PdfOcrChunk(2, 3, "beta"),
            PdfOcrChunk(3, 3, "gamma"),
        ]
        provider, captured = self._stub_provider(chunks)
        input_path = Path("/fake.pdf")
        try:
            pages = provider.pdf_ocr(input_path, backend="vision", model="m")
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()
        assert pages == [PageText(1, "alpha"), PageText(2, "beta"), PageText(3, "gamma")]
        assert isinstance(captured["payload"], PdfOcrRequest)
        assert captured["payload"].backend == "vision"
        # PdfOcrRequest.path is str(Path), which renders with the host
        # separator. Compare against the same str() conversion so the
        # assertion is platform-independent.
        assert captured["payload"].path == str(input_path)
        assert captured["payload"].model == "m"

    def test_pdf_ocr_propagates_per_page_progress(self) -> None:
        from lilbee.runtime.progress import EventType, ExtractEvent
        from lilbee.vision import PdfOcrChunk

        chunks = [PdfOcrChunk(1, 2, "a"), PdfOcrChunk(2, 2, "b")]
        provider, _ = self._stub_provider(chunks)
        events: list = []
        try:
            provider.pdf_ocr(
                Path("/scan.pdf"),
                backend="vision",
                on_progress=lambda et, ev: events.append((et, ev)),
            )
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()
        # Two streamed pages -> two EXTRACT events with matching page+total.
        assert [e[0] for e in events] == [EventType.EXTRACT, EventType.EXTRACT]
        assert events[0][1] == ExtractEvent(file="scan.pdf", page=1, total_pages=2)
        assert events[1][1] == ExtractEvent(file="scan.pdf", page=2, total_pages=2)

    def test_pdf_ocr_wraps_worker_error_as_provider_error(self) -> None:
        import asyncio

        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp import LlamaCppProvider
        from lilbee.providers.worker.transport_pipe import WorkerError

        async def _aiter():
            raise WorkerError("RuntimeError", "boom", "")
            yield  # pragma: no cover

        accessor = mock.MagicMock()
        accessor.stream = lambda _kind, _payload: _aiter()
        runtime = mock.MagicMock()
        runtime.run_sync = lambda coro, *, timeout: asyncio.new_event_loop().run_until_complete(
            coro
        )
        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        provider._get_pool_accessor = lambda *_a, **_kw: accessor
        provider._pool_runtime = lambda: runtime
        try:
            with pytest.raises(ProviderError, match="PDF OCR worker"):
                provider.pdf_ocr(Path("/x.pdf"), backend="vision")
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()

    def test_pdf_ocr_rejects_unexpected_frame_type(self) -> None:
        """A non-``PdfOcrChunk`` frame surfaces as ProviderError, not a silent unpack."""
        from lilbee.providers.base import ProviderError

        # Wire-format guard: if the worker contract regresses to a bare
        # tuple or anything else, the provider must surface a typed error
        # instead of unpacking it via positional access (which would let
        # the bug land silently in production).
        chunks = [("not", "a", "PdfOcrChunk")]
        provider, _ = self._stub_provider(chunks)
        try:
            with pytest.raises(ProviderError, match="unexpected frame type"):
                provider.pdf_ocr(Path("/x.pdf"), backend="vision")
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()

    def test_pdf_ocr_wraps_timeout_as_provider_error(self) -> None:
        """A pool TimeoutError surfaces as a friendly ProviderError, not the raw timeout."""
        import asyncio

        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp import LlamaCppProvider

        async def _aiter():
            raise TimeoutError("worker stalled")
            yield  # pragma: no cover

        accessor = mock.MagicMock()
        accessor.stream = lambda _kind, _payload: _aiter()
        runtime = mock.MagicMock()
        runtime.run_sync = lambda coro, *, timeout: asyncio.new_event_loop().run_until_complete(
            coro
        )
        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        provider._get_pool_accessor = lambda *_a, **_kw: accessor
        provider._pool_runtime = lambda: runtime
        try:
            with pytest.raises(ProviderError, match="PDF OCR worker timed out"):
                provider.pdf_ocr(Path("/scan.pdf"), backend="vision")
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()


class TestPdfDrainBudget:
    """``LlamaCppProvider._pdf_drain_budget`` sizes the streamed-drain timeout."""

    @staticmethod
    def _provider() -> Any:
        from lilbee.providers.llama_cpp import LlamaCppProvider

        with mock.patch("threading.Thread.start"):
            return LlamaCppProvider()

    def test_returns_no_cap_when_per_page_is_none(self) -> None:
        from lilbee.providers.llama_cpp.provider import _VISION_NO_CAP_TIMEOUT_S

        provider = self._provider()
        try:
            assert provider._pdf_drain_budget(Path("/x.pdf"), None) == _VISION_NO_CAP_TIMEOUT_S
            assert provider._pdf_drain_budget(Path("/x.pdf"), 0) == _VISION_NO_CAP_TIMEOUT_S
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()

    def test_returns_pages_times_per_page_plus_load_grace(self, monkeypatch) -> None:
        """Total budget = page_count * per_page + ``cfg.vision_load_budget_s``."""
        monkeypatch.setattr(
            "lilbee.providers.llama_cpp.provider.pdf_page_count",
            lambda _path: 8,
        )
        cfg.vision_load_budget_s = 100.0
        provider = self._provider()
        try:
            assert provider._pdf_drain_budget(Path("/x.pdf"), 30.0) == 8 * 30.0 + 100.0
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()

    def test_falls_back_to_no_cap_when_page_probe_fails(self, monkeypatch) -> None:
        """A pdfium probe failure mustn't kill an otherwise-valid run."""
        from lilbee.providers.llama_cpp.provider import _VISION_NO_CAP_TIMEOUT_S

        def _raise(_path):
            raise RuntimeError("pdfium probe boom")

        monkeypatch.setattr("lilbee.providers.llama_cpp.provider.pdf_page_count", _raise)
        provider = self._provider()
        try:
            assert provider._pdf_drain_budget(Path("/x.pdf"), 30.0) == _VISION_NO_CAP_TIMEOUT_S
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()


class TestLlamaNSeqMaxContextManager:
    """``_llama_n_seq_max`` patches ``internals.LlamaContext.__init__``."""

    def test_patched_init_sets_n_seq_max_then_calls_original(self) -> None:
        """Constructing a LlamaContext inside the with-block forces ``params.n_seq_max``.

        Outside the with-block, the original ``__init__`` is restored
        unchanged so non-embed loads (chat, vision) keep their default
        single-sequence behaviour.
        """
        from lilbee.providers.llama_cpp.provider import _llama_n_seq_max

        captured: list[Any] = []

        class _StubLlamaContext:
            def __init__(self, *, model: Any, params: Any, verbose: bool) -> None:
                captured.append((model, params, verbose, params.n_seq_max))

        fake_internals = mock.MagicMock()
        fake_internals.LlamaContext = _StubLlamaContext
        fake_module = mock.MagicMock()
        fake_module.internals = fake_internals

        with mock.patch.dict(sys.modules, {"llama_cpp": fake_module}):
            original_init = _StubLlamaContext.__init__
            with _llama_n_seq_max(7):
                params = mock.MagicMock()
                params.n_seq_max = 1  # initial value the patched body must overwrite.
                _StubLlamaContext(model="m", params=params, verbose=False)
            # Original is restored after the with-block exits.
            assert _StubLlamaContext.__init__ is original_init

        assert captured == [("m", mock.ANY, False, 7)]


class TestRoutingProviderPdfOcr:
    """``RoutingProvider.pdf_ocr`` dispatches by ref prefix, like ``vision_ocr``."""

    def test_native_ref_routes_to_llama_cpp(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        mock_native = mock.MagicMock()
        mock_native.pdf_ocr.return_value = ["p1", "p2"]
        rp._llama_cpp = mock_native
        progress = mock.MagicMock()
        native_ref = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"

        result = rp.pdf_ocr(
            Path("/x.pdf"),
            backend="vision",
            model=native_ref,
            per_page_timeout_s=12.5,
            quiet=False,
            on_progress=progress,
        )

        assert result == ["p1", "p2"]
        mock_native.pdf_ocr.assert_called_once_with(
            Path("/x.pdf"),
            backend="vision",
            model=native_ref,
            per_page_timeout_s=12.5,
            quiet=False,
            on_progress=progress,
        )

    def test_hosted_ref_routes_to_sdk_which_raises(self) -> None:
        """A hosted ``cfg.vision_model`` reaches the SDK side, which raises."""
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        mock_sdk = mock.MagicMock()
        mock_sdk.pdf_ocr.side_effect = NotImplementedError("hosted PDF OCR not supported")
        rp._sdk_provider = mock_sdk
        cfg.vision_model = ""

        with pytest.raises(NotImplementedError):
            rp.pdf_ocr(Path("/x.pdf"), backend="vision", model="openai/gpt-4-vision")
        mock_sdk.pdf_ocr.assert_called_once()


class TestSdkLLMProviderPdfOcr:
    """``SdkLLMProvider.pdf_ocr`` cannot rasterise PDFs and must raise."""

    def test_raises_not_implemented_with_user_facing_message(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        with pytest.raises(NotImplementedError, match="LILBEE_VISION_MODEL"):
            provider.pdf_ocr(Path("/scan.pdf"), backend="vision")
