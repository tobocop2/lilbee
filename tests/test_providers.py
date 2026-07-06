"""Tests for the LLM provider abstraction layer (mocked: no live servers needed)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import httpx
import pytest

from lilbee.core.config import cfg
from tests._sys_modules import inject_modules

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


def write_test_gguf(path: Path, *, arch: str | None, fields: dict[str, object]) -> Path:
    """Write a real (tensor-less) GGUF file for header-read tests.

    ``read_gguf_metadata`` parses headers with the ``gguf`` library, so these
    tests exercise the actual parser instead of mocking a native binding. Each
    field value is written as a string or a uint32 by its Python type; the
    architecture (when given) becomes ``general.architecture``.
    """
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch=arch or "")
    for key, value in fields.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, int(value))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    return path


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

    def test_retired_provider_string_rejected_at_config_boundary(self) -> None:
        # The retired "llama-cpp" / "multi-gpu" values are no longer accepted:
        # they fail enum validation at assignment like any other unknown value.
        from pydantic import ValidationError

        for retired in ("llama-cpp", "multi-gpu"):
            with pytest.raises(ValidationError):
                cfg.llm_provider = retired

    def test_unknown_provider_rejected_at_config_boundary(self) -> None:
        # llm_provider is a validated LlmProvider StrEnum: an invalid value is
        # rejected at assignment, so create_provider never sees a bad string.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cfg.llm_provider = "unknown"

    def test_services_singleton(self) -> None:
        from lilbee.app.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "auto"
        p1 = get_services().provider
        p2 = get_services().provider
        assert p1 is p2
        reset_services()

    def test_services_reset_clears_singleton(self) -> None:
        from lilbee.app.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "auto"
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
            # Blank by default; the resolved fallback is http://localhost:11434.
            assert c.ollama_base_url == ""
            assert c.llm_api_key == ""

    def test_provider_env_override(self) -> None:
        import os

        with mock.patch.dict(
            os.environ,
            {
                "LILBEE_LLM_PROVIDER": "remote",
                "LILBEE_OLLAMA_BASE_URL": "http://myhost:11434",
                "LILBEE_LLM_API_KEY": "sk-key",
            },
        ):
            from lilbee.core.config import Config

            c = Config()
            assert c.llm_provider == "remote"
            assert c.ollama_base_url == "http://myhost:11434"
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
        """Ensure any provider built during a test is shut down."""
        self._to_shutdown: list = []
        yield
        for p in self._to_shutdown:
            p.shutdown()

    def _make_provider(self) -> RoutingProvider:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        # Track the real llama-cpp provider for shutdown (tests replace it with mocks)
        if rp._local is not None:
            self._to_shutdown.append(rp._local)
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

    def test_supports_tools_delegates_to_sdk_backend_for_remote_ref(self) -> None:
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_tools.return_value = True
        rp._sdk_provider = mock_sdk

        assert rp.supports_tools("openai/gpt-4o") is True
        mock_sdk.supports_tools.assert_called_once_with("openai/gpt-4o")

    def test_supports_tools_delegates_to_local_backend_for_native_ref(self) -> None:
        rp = self._make_provider()
        mock_local = mock.MagicMock()
        mock_local.supports_tools.return_value = False
        rp._local = mock_local

        ref = "org/repo/chat.gguf"
        assert rp.supports_tools(ref) is False
        mock_local.supports_tools.assert_called_once_with(ref)

    def test_warm_progress_is_none_without_a_local_engine(self) -> None:
        rp = self._make_provider()
        rp._local = None
        assert rp.warm_progress() is None

    def test_warm_progress_delegates_to_local_engine(self) -> None:
        from lilbee.providers.warm_progress import WarmPhase, WarmProgress

        rp = self._make_provider()
        mock_local = mock.MagicMock()
        snapshot = WarmProgress(phase=WarmPhase.READING_WEIGHTS, bytes_done=1, bytes_total=2)
        mock_local.warm_progress.return_value = snapshot
        rp._local = mock_local
        assert rp.warm_progress() is snapshot
        mock_local.warm_progress.assert_called_once_with()

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

    def test_remote_chat_does_not_touch_n_ctx_resolution(self) -> None:
        """Cloud chat refs bypass ``_resolve_chat_ctx`` entirely.

        n_ctx / KV-cache sizing is meaningless for SDK-backed providers;
        if a future refactor accidentally routes a remote ref through the
        llama-cpp ctx picker we want this test to fail.
        """
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "ok"
        rp._sdk_provider = mock_litellm

        with mock.patch("lilbee.providers.engine_params.resolve_chat_ctx") as resolve_ctx:
            rp.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        resolve_ctx.assert_not_called()

    def test_routes_vision_ocr_to_local_engine_for_native_ref(self) -> None:
        """Native GGUF vision refs reach the local engine's vision servers."""
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_llama.vision_ocr.return_value = "page text"
        mock_litellm = mock.MagicMock()
        rp._local = mock_llama
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
        rp._local = mock_llama
        rp._sdk_provider = mock_litellm

        result = rp.vision_ocr(b"\x89PNG", "ollama/llava:7b", "ocr", timeout=30.0)
        assert result == "remote text"
        mock_litellm.vision_ocr.assert_called_once_with(
            b"\x89PNG", "ollama/llava:7b", "ocr", timeout=30.0
        )
        mock_llama.vision_ocr.assert_not_called()

    def test_routes_chat_to_local_engine_for_local_ref(self) -> None:
        """Local HF refs dispatch to the local engine regardless of registry contents.

        The routing is strict: a ``<org>/<repo>/<file>.gguf`` shape means
        native. If the registry doesn't have the model, the local engine
        raises its own 'not installed' error; routing never falls through
        to litellm.
        """
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        rp._local = mock_llama

        cfg.chat_model = "org/Local-GGUF/local-model.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()

    def test_routes_chat_to_local_engine_for_gguf_under_provider_named_org(self) -> None:
        """openai is a real HF org; an installed openai/<repo>/<file>.gguf stays local."""
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        mock_litellm = mock.MagicMock()
        rp._local = mock_llama
        rp._sdk_provider = mock_litellm

        cfg.chat_model = "openai/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()
        mock_litellm.chat.assert_not_called()

    def test_routes_embed_to_litellm_for_ollama_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.embed.return_value = [[0.1, 0.2]]
        rp._sdk_provider = mock_litellm

        cfg.embedding_model = "ollama/nomic-embed-text:latest"
        result = rp.embed(["test"])
        assert result == [[0.1, 0.2]]
        mock_litellm.embed.assert_called_once()

    def test_routes_embed_to_local_engine_for_local_ref(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.3, 0.4]]
        rp._local = mock_llama

        cfg.embedding_model = (
            "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        )
        result = rp.embed(["test"])
        assert result == [[0.3, 0.4]]

    def test_local_ref_never_falls_through_to_litellm(self) -> None:
        """Local HF refs stay on the local engine even when litellm is installed.

        The native GGUF shape is the single source of truth: anything that
        parses as a local HF ref dispatches to the local engine. Users who
        want Ollama say so with 'ollama/<name>'.
        """
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.9, 1.0]]
        rp._sdk_provider = mock_litellm
        rp._local = mock_llama

        cfg.embedding_model = "org/Local-GGUF/embed.gguf"
        result = rp.embed(["test"])
        assert result == [[0.9, 1.0]]
        mock_llama.embed.assert_called_once()
        mock_litellm.embed.assert_not_called()

    def test_list_models_native_only_when_sdk_unavailable(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._local = mock_llama

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
        rp._local = mock_llama

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
        rp._local = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_models.side_effect = RuntimeError("remote down")
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf"]

    def test_get_local_caches_instance(self) -> None:
        """``_get_local`` memoizes the local engine (FleetProvider) on first call."""
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        self._to_shutdown.append(rp)
        first = rp._get_local()
        self._to_shutdown.append(first)
        second = rp._get_local()
        assert first is second

    def test_get_sdk_provider_caches_instance(self) -> None:
        """``_get_sdk_provider`` memoizes the SdkLLMProvider on first call."""
        rp = self._make_provider()
        first = rp._get_sdk_provider()
        second = rp._get_sdk_provider()
        assert first is second

    def test_get_local_single_init_under_concurrency(self, monkeypatch) -> None:
        """Concurrent first-callers build the FleetProvider exactly once.

        Without the double-checked _init_lock, simultaneous callers each spawn a
        FleetProvider (duplicate role servers) and all but one leak.
        """
        import threading
        import time

        from lilbee.providers.routing_provider import RoutingProvider

        construct_count = {"n": 0}
        count_lock = threading.Lock()

        class FakeFleet:
            def __init__(self) -> None:
                with count_lock:
                    construct_count["n"] += 1
                # Widen the check-then-set window: without it the cheap __init__
                # runs to completion before the GIL yields, so a racing caller
                # never observes None and the test passes even on unlocked code.
                time.sleep(0.05)

            def shutdown(self) -> None: ...

        monkeypatch.setattr("lilbee.providers.fleet.provider.FleetProvider", FakeFleet)

        rp = RoutingProvider()
        self._to_shutdown.append(rp)
        barrier = threading.Barrier(8)
        results: list = []

        def worker() -> None:
            barrier.wait()
            results.append(rp._get_local())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert construct_count["n"] == 1
        assert len({id(r) for r in results}) == 1

    def test_get_sdk_provider_single_init_under_concurrency(self, monkeypatch) -> None:
        """Concurrent first-callers construct exactly one SdkLLMProvider.

        Asserting on the returned identity alone is false-green: without the
        lock, racing callers each construct an instance and the last write wins,
        so every late reader still sees the same final attribute. Count actual
        constructions instead.
        """
        import threading
        import time

        from lilbee.providers.routing_provider import RoutingProvider

        construct_count = {"n": 0}
        count_lock = threading.Lock()

        class FakeSdk:
            def __init__(self, *args, **kwargs) -> None:
                with count_lock:
                    construct_count["n"] += 1
                # Widen the check-then-set window so a racing caller observes the
                # uninitialized slot; otherwise the test is false-green on
                # unlocked code (see test_get_local_single_init_under_concurrency).
                time.sleep(0.05)

            def shutdown(self) -> None: ...

        monkeypatch.setattr("lilbee.providers.routing_provider.SdkLLMProvider", FakeSdk)

        rp = RoutingProvider()
        self._to_shutdown.append(rp)
        barrier = threading.Barrier(8)
        results: list = []

        def worker() -> None:
            barrier.wait()
            results.append(rp._get_sdk_provider())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert construct_count["n"] == 1
        assert len({id(r) for r in results}) == 1

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

    def test_show_model_local_ref_uses_local_engine(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.show_model.return_value = None
        rp._local = mock_llama

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
        rp._local = mock_native

        rp.invalidate_load_cache()
        mock_native.invalidate_load_cache.assert_called_once_with(None)

    def test_warm_up_pool_forwards_to_native(self) -> None:
        """``warm_up_pool`` lazily constructs the native provider and warms it."""
        rp = self._make_provider()
        mock_native = mock.MagicMock()
        with mock.patch.object(rp, "_get_local", return_value=mock_native):
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

    def test_tool_calls_extracts_function_id_name_arguments(self) -> None:
        """Non-stream ``tool_calls`` maps each litellm call to an SdkToolCall."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        call = mock.MagicMock(id="c1")
        call.function = mock.MagicMock(name="ignored")  # 'name' is a MagicMock kwarg trap
        call.function.name = "get_weather"
        call.function.arguments = '{"city":"SF"}'
        choice = mock.MagicMock()
        choice.message = mock.MagicMock(tool_calls=[call])
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        calls = view.tool_calls
        assert len(calls) == 1
        assert (calls[0].id, calls[0].name, calls[0].arguments) == (
            "c1",
            "get_weather",
            '{"city":"SF"}',
        )

    def test_tool_calls_empty_when_message_missing(self) -> None:
        """A choice whose first message is None yields no tool calls."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        choice = mock.MagicMock()
        choice.message = None
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        assert view.tool_calls == ()

    def test_tool_calls_empty_when_no_choices(self) -> None:
        """No choices -> no tool calls (the ``choice is None`` guard)."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        view = _LitellmResponseView(mock.MagicMock(choices=[]))
        assert view.tool_calls == ()

    def test_extract_tool_call_handles_missing_function(self) -> None:
        """A tool call with function=None yields empty name/arguments, not a crash."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        call = mock.MagicMock(id="c2")
        call.function = None
        choice = mock.MagicMock()
        choice.message = mock.MagicMock(tool_calls=[call])
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        only = view.tool_calls[0]
        assert (only.id, only.name, only.arguments) == ("c2", "", "")

    def test_delta_tool_calls_maps_opener_and_continuation(self) -> None:
        """Streaming deltas: opener carries id+name; later frames carry args only."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        opener = mock.MagicMock(index=0, id="c1")
        opener.function = mock.MagicMock()
        opener.function.name = "get_weather"
        opener.function.arguments = ""  # empty string normalises to None
        choice = mock.MagicMock()
        choice.delta = mock.MagicMock(tool_calls=[opener])
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        delta = view.delta_tool_calls[0]
        assert delta.index == 0
        assert delta.id == "c1"
        assert delta.name == "get_weather"
        assert delta.arguments_delta is None  # "" -> None

    def test_delta_tool_calls_uses_fallback_index_when_absent(self) -> None:
        """A delta without an .index uses its position in the array as the index."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        frame = mock.MagicMock(index=None, id=None)
        frame.function = None  # function=None -> name/args stay None
        choice = mock.MagicMock()
        choice.delta = mock.MagicMock(tool_calls=[frame])
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        delta = view.delta_tool_calls[0]
        assert delta.index == 0  # fallback to enumerate position
        assert delta.id is None
        assert delta.name is None
        assert delta.arguments_delta is None

    def test_delta_tool_calls_empty_when_delta_missing(self) -> None:
        """A streaming chunk whose first choice has no .delta yields no deltas."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        choice = mock.MagicMock(spec=["delta"])
        choice.delta = None
        view = _LitellmResponseView(mock.MagicMock(choices=[choice]))
        assert view.delta_tool_calls == ()

    def test_delta_tool_calls_empty_when_no_choices(self) -> None:
        """No choices -> no streaming tool-call deltas (the ``choice is None`` guard)."""
        from lilbee.providers.litellm_sdk import _LitellmResponseView

        view = _LitellmResponseView(mock.MagicMock(choices=[]))
        assert view.delta_tool_calls == ()


class TestLitellmAvailable:
    """Exercises the un-patched ``litellm_available`` install probe.

    The helper uses ``importlib.util.find_spec`` so the check is cheap to
    call on the UI thread (Settings's ``_FEATURE_GATED_GROUPS`` hits it
    synchronously during ``compose``). Mocking ``find_spec`` is the right
    boundary for these tests; ``sys.modules`` doesn't matter for the
    spec-lookup path.
    """

    @pytest.mark.real_litellm_probe
    def test_returns_false_when_not_installed(self) -> None:
        from lilbee.providers.litellm_sdk import litellm_available

        litellm_available.cache_clear()
        with mock.patch("importlib.util.find_spec", return_value=None):
            assert litellm_available() is False
        litellm_available.cache_clear()

    @pytest.mark.real_litellm_probe
    def test_returns_true_when_module_present(self) -> None:
        from lilbee.providers.litellm_sdk import litellm_available

        litellm_available.cache_clear()
        with mock.patch(
            "importlib.util.find_spec",
            return_value=mock.MagicMock(name="litellm_spec"),
        ):
            assert litellm_available() is True
        litellm_available.cache_clear()

    @pytest.mark.real_litellm_probe
    def test_does_not_execute_module_init(self) -> None:
        """Regression guard: the probe must not import the package itself.

        ``litellm.__init__`` loads provider plugins and takes multi-second
        time on Windows. Settings's compose calls this synchronously, so
        executing the module would block the UI thread on every fresh
        process. Asserting that no ``litellm`` entry lands in
        ``sys.modules`` after the probe runs codifies the contract.
        """
        from lilbee.providers.litellm_sdk import litellm_available

        litellm_available.cache_clear()
        with mock.patch(
            "importlib.util.find_spec",
            return_value=mock.MagicMock(name="litellm_spec"),
        ):
            had_module = "litellm" in sys.modules
            litellm_available()
            assert ("litellm" in sys.modules) is had_module
        litellm_available.cache_clear()


class TestLitellmBackendSupportsTools:
    """The SDK backend is optimistic about tool support: hosted models that lack
    a tool template simply return an empty tool_calls array, handled downstream."""

    def test_supports_tools_is_true_for_any_ref(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        backend = LitellmSdkBackend()
        assert backend.supports_tools("openai/gpt-4o") is True
        assert backend.supports_tools("ollama/llama3") is True


class TestRequireLitellm:
    @pytest.mark.real_litellm_probe
    def test_raises_provider_error_with_install_hint(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_sdk import _require_litellm

        with (
            inject_modules({"litellm": None}),
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

        return SdkLLMProvider(LitellmSdkBackend())

    def test_show_model_returns_capabilities(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {
            "capabilities": ["completion", "vision"],
            "parameters": "temperature 0.7",
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("ollama/llava:7b")

        assert result is not None
        assert result["capabilities"] == ["completion", "vision"]
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temperature 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("ollama/qwen3:8b")

        assert result is not None
        assert "capabilities" not in result
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_only_capabilities_no_params(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("ollama/some-model")

        assert result is not None
        assert result["capabilities"] == ["completion"]

    def test_show_model_empty_returns_none(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("ollama/empty-model")

        assert result is None

    def test_show_model_http_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            result = provider.show_model("ollama/bad-model")

        assert result is None

    def test_get_capabilities_returns_list(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion", "vision", "tools"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("ollama/llava:7b")

        assert caps == ["completion", "vision", "tools"]

    def test_get_capabilities_returns_empty_on_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            caps = provider.get_capabilities("ollama/bad-model")

        assert caps == []

    def test_get_capabilities_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temp 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("ollama/qwen3:8b")

        assert caps == []


class TestLMStudioCapabilityGuards:
    """LM Studio has no Ollama-style /api/pull or /api/show endpoints."""

    _LM_STUDIO_BASE = "http://localhost:1234/v1"

    def _backend(self):  # type: ignore[no-untyped-def]
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        return LitellmSdkBackend()

    def test_pull_model_raises_user_facing_error(self) -> None:
        from lilbee.providers.base import ProviderError

        backend = self._backend()
        with pytest.raises(ProviderError) as exc:
            backend.pull_model("qwen2.5-7b-instruct", base_url=self._LM_STUDIO_BASE)
        # User-facing message names LM Studio and the fix; no internal vocabulary.
        assert "LM Studio" in str(exc.value)
        assert "dispatch" not in str(exc.value).lower()

    def test_pull_model_makes_no_http_call(self) -> None:
        from lilbee.providers.base import ProviderError

        backend = self._backend()
        with mock.patch("httpx.Client") as client, pytest.raises(ProviderError):
            backend.pull_model("qwen2.5-7b-instruct", base_url=self._LM_STUDIO_BASE)
        client.assert_not_called()

    def test_show_model_returns_none_without_http_call(self) -> None:
        backend = self._backend()
        with mock.patch("httpx.post") as post:
            result = backend.show_model(
                "lm_studio/qwen2.5-7b-instruct", base_url=self._LM_STUDIO_BASE
            )
        assert result is None
        post.assert_not_called()


class TestShowModelNotFound:
    def test_returns_none_for_missing_model(self) -> None:
        from lilbee.providers.fleet.provider import FleetProvider

        provider = FleetProvider()
        assert provider.show_model("nonexistent-model-xyz") is None


class TestReadMmprojProjectorType:
    def test_reads_projector_type(self, tmp_path: Path) -> None:
        import struct

        from lilbee.providers.gguf_meta import read_mmproj_projector_type

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
        from lilbee.providers.gguf_meta import read_mmproj_projector_type

        assert read_mmproj_projector_type(Path("/nonexistent/file.gguf")) is None

    def test_non_string_projector_type_returns_none(self, tmp_path: Path) -> None:
        """If clip.projector_type is present but not a string (someone wrote it
        as an int or bool), the reader returns None instead of decoding bytes."""
        import struct

        from lilbee.providers.gguf_meta import read_mmproj_projector_type

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

        from lilbee.providers.gguf_meta import read_mmproj_projector_type

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


class TestVulkanGpuSelect:
    """``autoselect_best_gpu_index`` probes libvulkan via ctypes and picks discrete."""

    def test_pick_best_prefers_discrete_over_integrated(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
            _pick_best_device,
        )

        devices = [
            VulkanDevice(
                index=0, device_type=VkDeviceType.INTEGRATED_GPU, device_name="iGPU", vendor_id=0
            ),
            VulkanDevice(
                index=1, device_type=VkDeviceType.DISCRETE_GPU, device_name="dGPU", vendor_id=0
            ),
        ]
        best = _pick_best_device(devices)
        assert best is not None
        assert best.index == 1

    def test_pick_best_returns_none_for_cpu_only(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
            _pick_best_device,
        )

        devices = [
            VulkanDevice(
                index=0, device_type=VkDeviceType.CPU, device_name="llvmpipe", vendor_id=0
            ),
        ]
        assert _pick_best_device(devices) is None

    def test_pick_best_returns_none_for_empty_list(self) -> None:
        from lilbee.providers.fleet.gpu_select import _pick_best_device

        assert _pick_best_device([]) is None

    def test_rank_for_unknown_device_type_returns_zero(self) -> None:
        """Drivers may report a deviceType outside the Vulkan 1.0 enum; we treat as CPU-rank."""
        from lilbee.providers.fleet.gpu_select import _rank_for

        assert _rank_for(999) == 0

    def test_autoselect_returns_discrete_index_for_dual_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
        )

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                VulkanDevice(
                    index=0,
                    device_type=VkDeviceType.INTEGRATED_GPU,
                    device_name="iGPU",
                    vendor_id=0,
                ),
                VulkanDevice(
                    index=1,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="dGPU",
                    vendor_id=0,
                ),
            ],
        )
        assert gpu_select.autoselect_best_gpu_index() == "1"

    def test_autoselect_returns_none_when_single_visible_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-device hosts keep the default ordering; auto-pin would be churn."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
        )

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                VulkanDevice(
                    index=0,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="RTX 4090",
                    vendor_id=0,
                ),
            ],
        )
        assert gpu_select.autoselect_best_gpu_index() is None

    def test_autoselect_returns_none_when_loader_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: None)
        assert gpu_select.autoselect_best_gpu_index() is None

    def test_autoselect_returns_none_when_only_cpu_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All-CPU adapter list shouldn't force a pin: software rendering is never right."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
        )

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                VulkanDevice(
                    index=0, device_type=VkDeviceType.CPU, device_name="cpu0", vendor_id=0
                ),
                VulkanDevice(
                    index=1, device_type=VkDeviceType.CPU, device_name="cpu1", vendor_id=0
                ),
            ],
        )
        assert gpu_select.autoselect_best_gpu_index() is None

    def test_autoselect_returns_none_when_all_devices_same_rank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two discrete GPUs of equal rank: no auto-pin (no decision to make)."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            VulkanDevice,
        )

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                VulkanDevice(
                    index=0,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="dgpu0",
                    vendor_id=0,
                ),
                VulkanDevice(
                    index=1,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="dgpu1",
                    vendor_id=0,
                ),
            ],
        )
        assert gpu_select.autoselect_best_gpu_index() is None

    def test_loader_returns_none_on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """macOS uses Metal; the Vulkan probe explicitly skips."""
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "darwin")
        assert gpu_select._load_vulkan_loader() is None

    def test_loader_returns_none_when_library_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "linux")

        def _raises(_name: str) -> object:
            raise OSError("cannot find vulkan loader")

        monkeypatch.setattr(gpu_select.ctypes, "CDLL", _raises)
        monkeypatch.setattr(gpu_select.ctypes.util, "find_library", lambda _name: None)
        assert gpu_select._load_vulkan_loader() is None

    def test_loader_falls_back_to_find_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "linux")
        attempts: list[str] = []

        def _cdll(name: str) -> object:
            attempts.append(name)
            if name == "/resolved/libvulkan.so":
                return "loaded"
            raise OSError("not on direct path")

        monkeypatch.setattr(gpu_select.ctypes, "CDLL", _cdll)
        monkeypatch.setattr(
            gpu_select.ctypes.util, "find_library", lambda _name: "/resolved/libvulkan.so"
        )
        assert gpu_select._load_vulkan_loader() == "loaded"
        assert "libvulkan.so.1" in attempts

    def test_loader_probes_dll_on_win32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Windows the probe tries ``vulkan-1.dll`` before giving up."""
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        attempts: list[str] = []

        def _cdll(name: str) -> object:
            attempts.append(name)
            raise OSError("not loadable")

        monkeypatch.setattr(gpu_select.ctypes, "CDLL", _cdll)
        monkeypatch.setattr(gpu_select.ctypes.util, "find_library", lambda _name: None)
        assert gpu_select._load_vulkan_loader() is None
        assert "vulkan-1.dll" in attempts

    def test_enumerate_returns_none_when_loader_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select, "_load_vulkan_loader", lambda: None)
        assert gpu_select._enumerate_vulkan_devices() is None

    def test_enumerate_catches_oserror_from_ctypes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select, "_load_vulkan_loader", lambda: object())

        def _raises(_lib: object) -> object:
            raise OSError("symbol not found")

        monkeypatch.setattr(gpu_select, "_list_devices_with_instance", _raises)
        assert gpu_select._enumerate_vulkan_devices() is None

    def test_resolve_vk_symbols_stamps_argtypes(self) -> None:
        """``_resolve_vk_symbols`` reads five named attributes off the loader."""
        from lilbee.providers.fleet.gpu_select import _resolve_vk_symbols

        fake_lib = mock.MagicMock()
        create, destroy, enum_phys, get_props, get_mem = _resolve_vk_symbols(fake_lib)
        assert create is fake_lib.vkCreateInstance
        assert destroy is fake_lib.vkDestroyInstance
        assert enum_phys is fake_lib.vkEnumeratePhysicalDevices
        assert get_props is fake_lib.vkGetPhysicalDeviceProperties
        assert get_mem is fake_lib.vkGetPhysicalDeviceMemoryProperties

    def test_list_devices_with_instance_returns_parsed_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: simulated VK calls populate the device list."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import (
            VkDeviceType,
            _list_devices_with_instance,
        )

        fake_props = [
            (VkDeviceType.INTEGRATED_GPU, b"iGPU"),
            (VkDeviceType.DISCRETE_GPU, b"NVIDIA RTX"),
        ]
        device_count = len(fake_props)

        def _create_instance(_info_ref: object, _alloc: object, instance_ref: object) -> int:
            instance_ref._obj.value = 0xDEADBEEF
            return 0

        def _enum_physical(_instance: object, count_ref: object, handles_arr: object) -> int:
            if handles_arr is None:
                count_ref._obj.value = device_count
            else:
                for i in range(device_count):
                    handles_arr[i] = 0xCAFE + i
            return 0

        def _get_properties(handle: object, props_ref: object) -> None:
            i = handle - 0xCAFE
            dtype, name = fake_props[i]
            props_ref._obj.deviceType = dtype
            props_ref._obj.deviceName = name

        def _get_memory(handle: object, mem_ref: object) -> None:
            # device 1 reports a 12 GB device-local heap + a host heap to ignore.
            i = handle - 0xCAFE
            mem_ref._obj.memoryHeapCount = 2
            mem_ref._obj.memoryHeaps[0].size = (i + 1) * 12_000_000_000
            mem_ref._obj.memoryHeaps[0].flags = 1  # VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
            mem_ref._obj.memoryHeaps[1].size = 64_000_000_000
            mem_ref._obj.memoryHeaps[1].flags = 0  # host-visible, not VRAM

        monkeypatch.setattr(
            gpu_select,
            "_resolve_vk_symbols",
            lambda _lib: (
                _create_instance,
                lambda *_: None,
                _enum_physical,
                _get_properties,
                _get_memory,
            ),
        )
        devices = _list_devices_with_instance(object())
        assert len(devices) == device_count
        assert devices[0].device_type == VkDeviceType.INTEGRATED_GPU
        assert devices[1].device_type == VkDeviceType.DISCRETE_GPU
        assert devices[1].device_name == "NVIDIA RTX"
        assert devices[1].vram_bytes == 24_000_000_000

    def test_list_devices_returns_empty_when_create_instance_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import _list_devices_with_instance

        monkeypatch.setattr(
            gpu_select,
            "_resolve_vk_symbols",
            lambda _lib: (
                lambda *_a: 1,  # VK_ERROR
                lambda *_a: None,
                lambda *_a: 0,
                lambda *_a: None,
                lambda *_a: None,
            ),
        )
        assert _list_devices_with_instance(object()) == []

    def test_list_devices_returns_empty_when_first_enum_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import _list_devices_with_instance

        def _create_instance(_info: object, _alloc: object, instance_ref: object) -> int:
            instance_ref._obj.value = 0x1
            return 0

        monkeypatch.setattr(
            gpu_select,
            "_resolve_vk_symbols",
            lambda _lib: (
                _create_instance,
                lambda *_a: None,
                lambda *_a: 1,  # enum fails
                lambda *_a: None,
                lambda *_a: None,
            ),
        )
        assert _list_devices_with_instance(object()) == []

    def test_list_devices_returns_empty_when_count_is_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import _list_devices_with_instance

        def _create_instance(_info: object, _alloc: object, instance_ref: object) -> int:
            instance_ref._obj.value = 0x1
            return 0

        def _enum_physical(_instance: object, count_ref: object, _handles: object) -> int:
            count_ref._obj.value = 0
            return 0

        monkeypatch.setattr(
            gpu_select,
            "_resolve_vk_symbols",
            lambda _lib: (
                _create_instance,
                lambda *_a: None,
                _enum_physical,
                lambda *_a: None,
                lambda *_a: None,
            ),
        )
        assert _list_devices_with_instance(object()) == []

    def test_list_devices_returns_empty_when_second_enum_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import _list_devices_with_instance

        def _create_instance(_info: object, _alloc: object, instance_ref: object) -> int:
            instance_ref._obj.value = 0x1
            return 0

        def _enum_physical(_instance: object, count_ref: object, handles: object) -> int:
            if handles is None:
                count_ref._obj.value = 1
                return 0
            return 1  # second call fails

        monkeypatch.setattr(
            gpu_select,
            "_resolve_vk_symbols",
            lambda _lib: (
                _create_instance,
                lambda *_a: None,
                _enum_physical,
                lambda *_a: None,
                lambda *_a: None,
            ),
        )
        assert _list_devices_with_instance(object()) == []

    def test_loader_find_library_returns_none_after_oserror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """find_library succeeds but CDLL on that path still raises -> overall None."""
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "linux")

        def _cdll(_name: str) -> object:
            raise OSError("loader unhappy")

        monkeypatch.setattr(gpu_select.ctypes, "CDLL", _cdll)
        monkeypatch.setattr(gpu_select.ctypes.util, "find_library", lambda _n: "/lib/libvulkan.so")
        assert gpu_select._load_vulkan_loader() is None

    def test_enumerate_gpu_vram_returns_index_vram_pairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import VkDeviceType, VulkanDevice

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                VulkanDevice(
                    index=0,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="a",
                    vendor_id=0,
                    vram_bytes=24_000_000_000,
                ),
                VulkanDevice(
                    index=1,
                    device_type=VkDeviceType.DISCRETE_GPU,
                    device_name="b",
                    vendor_id=0,
                    vram_bytes=8_000_000_000,
                ),
            ],
        )
        assert gpu_select.enumerate_gpu_vram() == [(0, 24_000_000_000), (1, 8_000_000_000)]

    def test_enumerate_gpu_vram_none_when_probe_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: None)
        assert gpu_select.enumerate_gpu_vram() is None

    def test_device_local_vram_sums_device_local_heaps_only(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            _device_local_vram,
            _VkPhysicalDeviceMemoryProperties,
        )

        mem = _VkPhysicalDeviceMemoryProperties()
        mem.memoryHeapCount = 2
        mem.memoryHeaps[0].size = 8_000_000_000
        mem.memoryHeaps[0].flags = 1  # VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        mem.memoryHeaps[1].size = 16_000_000_000
        mem.memoryHeaps[1].flags = 0  # host-visible
        assert _device_local_vram(mem) == 8_000_000_000


class TestClassifyManifestVendor:
    """``_classify_manifest_vendor`` maps an ICD manifest filename to a vendor."""

    def test_nvidia_manifest_is_classified(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _classify_manifest_vendor,
        )

        assert _classify_manifest_vendor("nv-vk64.json") is PCIVendorID.NVIDIA

    def test_amdvlk_manifest_is_classified(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _classify_manifest_vendor,
        )

        assert _classify_manifest_vendor("amdvlk64.json") is PCIVendorID.AMD

    def test_radeon_manifest_is_classified_as_amd(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _classify_manifest_vendor,
        )

        assert _classify_manifest_vendor("radeon_icd.x64.json") is PCIVendorID.AMD

    def test_intel_manifest_is_classified(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _classify_manifest_vendor,
        )

        assert _classify_manifest_vendor("intel_icd.x64.json") is PCIVendorID.INTEL
        assert _classify_manifest_vendor("igvk_icd.json") is PCIVendorID.INTEL

    def test_classifier_is_case_insensitive(self) -> None:
        """Windows file paths are case-insensitive; the loader's match is too."""
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _classify_manifest_vendor,
        )

        assert _classify_manifest_vendor("NV-VK64.JSON") is PCIVendorID.NVIDIA
        assert _classify_manifest_vendor("AMDVLK64.JSON") is PCIVendorID.AMD

    def test_unknown_manifest_returns_none(self) -> None:
        """A manifest filename we don't recognise has no glob we can disable, so skip it."""
        from lilbee.providers.fleet.gpu_select import _classify_manifest_vendor

        assert _classify_manifest_vendor("mesa_dzn.json") is None
        assert _classify_manifest_vendor("microsoft_dozen.json") is None


class TestSelectBestVendor:
    """``_select_best_vendor`` walks the hardcoded preference order."""

    def test_nvidia_wins_over_amd(self) -> None:
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _select_best_vendor,
        )

        assert _select_best_vendor({PCIVendorID.NVIDIA, PCIVendorID.AMD}) is PCIVendorID.NVIDIA

    def test_amd_wins_over_intel(self) -> None:
        """AMD discrete + Intel iGPU laptops: keep the discrete card."""
        from lilbee.providers.fleet.gpu_select import (
            PCIVendorID,
            _select_best_vendor,
        )

        assert _select_best_vendor({PCIVendorID.AMD, PCIVendorID.INTEL}) is PCIVendorID.AMD

    def test_returns_none_on_empty_set(self) -> None:
        from lilbee.providers.fleet.gpu_select import _select_best_vendor

        assert _select_best_vendor(set()) is None


class TestVulkanVendorsPresent:
    """``_vulkan_vendors_present`` walks manifest paths, never the Vulkan loader."""

    def test_collects_vendors_from_windows_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows-style backslash paths classify by filename correctly."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        manifests = [
            r"C:\Windows\System32\nv-vk64.json",
            r"C:\Windows\System32\DriverStore\FileRepository\amdvlk64.inf_amd64_abc\amdvlk64.json",
        ]
        monkeypatch.setattr(
            gpu_select,
            "iter_vulkan_manifest_paths",
            lambda: iter(manifests),
        )
        assert gpu_select._vulkan_vendors_present() == {
            PCIVendorID.NVIDIA,
            PCIVendorID.AMD,
        }

    def test_collects_vendors_from_linux_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """POSIX-style forward-slash paths classify by filename correctly.

        Mesa RADV ships ``radeon_icd.x86_64.json`` (matches ``radeon*``),
        NVIDIA proprietary ships ``nvidia_icd.json`` (matches ``nv*``).
        """
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        manifests = [
            "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/intel_icd.x86_64.json",
        ]
        monkeypatch.setattr(
            gpu_select,
            "iter_vulkan_manifest_paths",
            lambda: iter(manifests),
        )
        assert gpu_select._vulkan_vendors_present() == {
            PCIVendorID.NVIDIA,
            PCIVendorID.AMD,
            PCIVendorID.INTEL,
        }

    def test_linux_amdvlk_manifest_classifies_as_amd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux AMDVLK ships as ``amd_icd64.json`` (no leading ``amdvlk`` prefix)."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(
            gpu_select,
            "iter_vulkan_manifest_paths",
            lambda: iter(["/usr/share/vulkan/icd.d/amd_icd64.json"]),
        )
        assert gpu_select._vulkan_vendors_present() == {PCIVendorID.AMD}

    def test_drops_unknown_manifest_filenames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """We can only disable manifests we have a glob for; unknown ones are dropped."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        manifests = [
            r"C:\Windows\System32\nv-vk64.json",
            r"C:\Windows\System32\mesa_dzn.json",  # Microsoft compat pack, not a vendor we pin
            "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",  # llvmpipe software renderer
        ]
        monkeypatch.setattr(
            gpu_select,
            "iter_vulkan_manifest_paths",
            lambda: iter(manifests),
        )
        assert gpu_select._vulkan_vendors_present() == {PCIVendorID.NVIDIA}

    def test_returns_empty_set_when_no_manifests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(
            gpu_select,
            "iter_vulkan_manifest_paths",
            lambda: iter([]),
        )
        assert gpu_select._vulkan_vendors_present() == set()


class TestIterVulkanManifestPaths:
    """``iter_vulkan_manifest_paths`` dispatches to the platform-specific walker."""

    def test_darwin_yields_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """macOS uses Metal directly; the Vulkan loader is not on the path."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.setattr(vulkan_icd_discovery.sys, "platform", "darwin")
        assert list(vulkan_icd_discovery.iter_vulkan_manifest_paths()) == []

    def test_windows_dispatches_to_registry_walker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.setattr(vulkan_icd_discovery.sys, "platform", "win32")
        monkeypatch.setattr(
            vulkan_icd_discovery,
            "_iter_windows_vulkan_manifest_paths",
            lambda: iter([r"C:\Windows\System32\nv-vk64.json"]),
        )
        monkeypatch.setattr(
            vulkan_icd_discovery,
            "_iter_linux_vulkan_manifest_paths",
            lambda: iter(["/should-not-be-called"]),
        )
        assert list(vulkan_icd_discovery.iter_vulkan_manifest_paths()) == [
            r"C:\Windows\System32\nv-vk64.json"
        ]

    def test_linux_dispatches_to_xdg_walker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.setattr(vulkan_icd_discovery.sys, "platform", "linux")
        monkeypatch.setattr(
            vulkan_icd_discovery,
            "_iter_windows_vulkan_manifest_paths",
            lambda: iter(["should-not-be-called"]),
        )
        monkeypatch.setattr(
            vulkan_icd_discovery,
            "_iter_linux_vulkan_manifest_paths",
            lambda: iter(["/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"]),
        )
        assert list(vulkan_icd_discovery.iter_vulkan_manifest_paths()) == [
            "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
        ]


class TestIterWindowsVulkanManifestPaths:
    """``_iter_windows_vulkan_manifest_paths`` combines every Khronos-documented registry path."""

    def test_combines_khronos_and_pnp_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The iterator must surface manifests from BOTH the legacy key and the PnP class walk.

        A regression here would silently miss either AMD adapters (live in
        the PnP path) or Microsoft software ICDs (live under
        ``Khronos\\Vulkan\\Drivers``).
        """
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg_stub = mock.MagicMock(name="winreg")
        monkeypatch.setitem(sys.modules, "winreg", winreg_stub)
        monkeypatch.setattr(
            vulkan_icd_discovery,
            "_iter_khronos_software_manifests",
            lambda _wr: iter([r"C:\soft\nv-vk64.json"]),
        )

        pnp_calls: list[str] = []

        def _fake_pnp(_wr: object, guid: str) -> object:
            pnp_calls.append(guid)
            return iter([rf"C:\pnp\{guid[:8]}\amdvlk64.json"])

        monkeypatch.setattr(vulkan_icd_discovery, "_iter_pnp_class_manifests", _fake_pnp)

        result = list(vulkan_icd_discovery._iter_windows_vulkan_manifest_paths())
        assert r"C:\soft\nv-vk64.json" in result
        assert any("amdvlk64.json" in path for path in result)
        # Both PnP class GUIDs (display adapter + software component) must be walked.
        assert vulkan_icd_discovery._PNP_DISPLAY_ADAPTER_CLASS_GUID in pnp_calls
        assert vulkan_icd_discovery._PNP_SOFTWARE_COMPONENT_CLASS_GUID in pnp_calls


class TestIterLinuxVulkanManifestPaths:
    """``_iter_linux_vulkan_manifest_paths`` walks the XDG ICD directory hierarchy.

    Tests monkeypatch ``_linux_vulkan_icd_directories`` directly so they run
    on any OS; the XDG env-var parsing in ``_xdg_dirs`` is covered separately
    by :class:`TestXdgDirs` with strings that don't collide with Windows
    drive-letter colons.
    """

    def test_yields_json_files_from_real_directory(self, tmp_path: Path) -> None:
        """Globs ``*.json`` in each yielded directory, skips non-json names."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        icd_dir = tmp_path / "icd.d"
        icd_dir.mkdir()
        (icd_dir / "nvidia_icd.json").write_text("{}")
        (icd_dir / "radeon_icd.x86_64.json").write_text("{}")
        (icd_dir / "not_an_icd.txt").write_text("ignored")

        with mock.patch.object(
            vulkan_icd_discovery, "_linux_vulkan_icd_directories", lambda: iter([icd_dir])
        ):
            result = sorted(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths())

        filenames = [os.path.basename(p) for p in result]
        assert "nvidia_icd.json" in filenames
        assert "radeon_icd.x86_64.json" in filenames
        assert "not_an_icd.txt" not in filenames

    def test_duplicate_directories_are_deduped(self, tmp_path: Path) -> None:
        """A directory yielded twice must not produce duplicate manifest paths."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        icd_dir = tmp_path / "icd.d"
        icd_dir.mkdir()
        (icd_dir / "intel_icd.x86_64.json").write_text("{}")

        with mock.patch.object(
            vulkan_icd_discovery,
            "_linux_vulkan_icd_directories",
            lambda: iter([icd_dir, icd_dir]),
        ):
            result = list(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths())
        intel_hits = [p for p in result if p.endswith("intel_icd.x86_64.json")]
        assert len(intel_hits) == 1

    def test_missing_directories_are_silently_skipped(self, tmp_path: Path) -> None:
        """A nonexistent directory doesn't raise; the walk continues."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        with mock.patch.object(
            vulkan_icd_discovery,
            "_linux_vulkan_icd_directories",
            lambda: iter([tmp_path / "does_not_exist"]),
        ):
            assert list(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths()) == []

    def test_directory_with_dot_json_suffix_is_filtered_out(self, tmp_path: Path) -> None:
        """``foo.json`` directories match the glob but ``is_file`` rejects them."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        icd_dir = tmp_path / "icd.d"
        icd_dir.mkdir()
        (icd_dir / "nvidia_icd.json").write_text("{}")
        (icd_dir / "stray_dir.json").mkdir()

        with mock.patch.object(
            vulkan_icd_discovery, "_linux_vulkan_icd_directories", lambda: iter([icd_dir])
        ):
            result = list(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths())
        assert any(p.endswith("nvidia_icd.json") for p in result)
        assert not any(p.endswith("stray_dir.json") for p in result)

    def test_unreadable_directory_does_not_abort_walk(self, tmp_path: Path) -> None:
        """``Path.glob`` raising OSError logs and continues to the next directory."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        readable = tmp_path / "ok"
        readable.mkdir()
        (readable / "nvidia_icd.json").write_text("{}")
        unreadable = tmp_path / "bad"
        unreadable.mkdir()

        real_glob = Path.glob

        def _fake_glob(self: Path, pattern: str) -> object:
            if self == unreadable:
                raise OSError("simulated EACCES")
            return real_glob(self, pattern)

        with (
            mock.patch.object(Path, "glob", _fake_glob),
            mock.patch.object(
                vulkan_icd_discovery,
                "_linux_vulkan_icd_directories",
                lambda: iter([unreadable, readable]),
            ),
        ):
            result = list(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths())
        assert any(p.endswith("nvidia_icd.json") for p in result)

    def test_unexpandable_path_does_not_abort_walk(self, tmp_path: Path) -> None:
        """``Path.expanduser`` raising ``RuntimeError`` (HOME unset) skips and continues."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        good_dir = tmp_path / "ok"
        good_dir.mkdir()
        (good_dir / "nvidia_icd.json").write_text("{}")
        bad_path = Path("~/flatpak/vulkan/icd.d")

        real_expanduser = Path.expanduser

        def _fake_expanduser(self: Path) -> Path:
            if self == bad_path:
                raise RuntimeError("HOME not set")
            return real_expanduser(self)

        with (
            mock.patch.object(Path, "expanduser", _fake_expanduser),
            mock.patch.object(
                vulkan_icd_discovery,
                "_linux_vulkan_icd_directories",
                lambda: iter([bad_path, good_dir]),
            ),
        ):
            result = list(vulkan_icd_discovery._iter_linux_vulkan_manifest_paths())
        assert any(p.endswith("nvidia_icd.json") for p in result)


class TestLinuxVulkanIcdDirectories:
    """``_linux_vulkan_icd_directories`` aggregates XDG + fixed-etc + Flatpak paths.

    Comparisons go through ``PurePosixPath`` so the test runs portably on
    Windows CI without taking a dependency on ``os.sep``.
    """

    def test_yields_expected_canonical_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All XDG_* vars unset -> spec defaults plus fixed /etc and Flatpak trees."""
        from pathlib import PurePosixPath

        from lilbee.providers.fleet import vulkan_icd_discovery

        for var in ("XDG_CONFIG_HOME", "XDG_CONFIG_DIRS", "XDG_DATA_HOME", "XDG_DATA_DIRS"):
            monkeypatch.delenv(var, raising=False)
        out = [
            PurePosixPath(str(p).replace("\\", "/"))
            for p in vulkan_icd_discovery._linux_vulkan_icd_directories()
        ]
        flat = [str(p) for p in out]
        # XDG_CONFIG_HOME / XDG_DATA_HOME defaults expand ``~``.
        assert any(p.endswith(".config/vulkan/icd.d") for p in flat)
        assert any(p.endswith(".local/share/vulkan/icd.d") for p in flat)
        # XDG_CONFIG_DIRS default.
        assert any(p.endswith("etc/xdg/vulkan/icd.d") for p in flat)
        # SYSCONFDIR / EXTRASYSCONFDIR build-time constants.
        assert "/usr/local/etc/vulkan/icd.d" in flat
        assert "/etc/vulkan/icd.d" in flat
        # XDG_DATA_DIRS default: /usr/local/share + /usr/share.
        assert "/usr/local/share/vulkan/icd.d" in flat
        assert "/usr/share/vulkan/icd.d" in flat
        # Flatpak export trees (user + system).
        assert any("flatpak/exports/share/vulkan/icd.d" in p for p in flat)


class TestXdgDirs:
    """``_xdg_dirs`` parses colon-separated XDG path lists.

    Inputs use POSIX-style placeholders ("share", "etc") rather than
    real ``tmp_path`` directories so the parser test is portable to
    Windows (where a ``tmp_path`` would contain a drive-letter colon
    and confuse the split).
    """

    def test_uses_default_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.delenv("X_UNSET_TEST", raising=False)
        out = list(vulkan_icd_discovery._xdg_dirs("X_UNSET_TEST", "default", "sub"))
        assert out == [Path("default") / "sub"]

    def test_splits_colon_delimited_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.setenv("X_LIST_TEST", "share:opt")
        out = list(vulkan_icd_discovery._xdg_dirs("X_LIST_TEST", "default", "sub"))
        assert out == [Path("share") / "sub", Path("opt") / "sub"]

    def test_empty_components_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Vulkan-Loader#2331: ``"a::b"`` and trailing colons must not yield ``Path("")``."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        monkeypatch.setenv("X_EXTRA_COLON_TEST", "::share:")
        out = list(vulkan_icd_discovery._xdg_dirs("X_EXTRA_COLON_TEST", "default", "sub"))
        assert out == [Path("share") / "sub"]


class TestIterKhronosSoftwareManifests:
    """``_iter_khronos_software_manifests`` honours the Khronos REG_DWORD = 0 enabled flag."""

    def test_yields_only_enabled_entries(self) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0  # any sentinel; OpenKey is mocked
        winreg.OpenKey.return_value = mock.MagicMock(name="hkey")

        values_by_key: dict[str, list[tuple[str, int, int]]] = {
            r"SOFTWARE\Khronos\Vulkan\Drivers": [
                (r"C:\nv-vk64.json", 0, 4),  # enabled
                (r"C:\disabled.json", 1, 4),  # explicitly disabled
            ],
            r"SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers": [
                (r"C:\amdvlk32.json", 0, 4),
            ],
        }

        opened_keys: list[str] = []

        def _open(_root: int, sub_path: str) -> mock.MagicMock:
            opened_keys.append(sub_path)
            return mock.MagicMock(name=f"hkey:{sub_path}")

        winreg.OpenKey.side_effect = _open

        def _enum_value(key: mock.MagicMock, i: int) -> tuple[str, int, int]:
            sub_path = opened_keys[-1]
            entries = values_by_key.get(sub_path, [])
            if i >= len(entries):
                raise OSError("end of values")
            return entries[i]

        winreg.EnumValue.side_effect = _enum_value

        result = list(vulkan_icd_discovery._iter_khronos_software_manifests(winreg))
        assert r"C:\nv-vk64.json" in result
        assert r"C:\amdvlk32.json" in result
        assert r"C:\disabled.json" not in result

    def test_missing_key_is_skipped(self) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        winreg.OpenKey.side_effect = OSError("key not present")

        assert list(vulkan_icd_discovery._iter_khronos_software_manifests(winreg)) == []


class TestIterPnpClassManifests:
    """``_iter_pnp_class_manifests`` reads VulkanDriverName{,Wow} off PnP adapter keys."""

    def test_yields_strings_and_multi_sz_lists(self) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        class_root = mock.MagicMock(name="class_root")
        subkey0 = mock.MagicMock(name="0000")
        subkey1 = mock.MagicMock(name="0001")

        winreg.OpenKey.side_effect = [class_root, subkey0, subkey1]
        winreg.EnumKey.side_effect = ["0000", "0001", OSError("end")]

        def _query(key: mock.MagicMock, name: str) -> tuple[object, int]:
            if key is subkey0 and name == "VulkanDriverName":
                return (r"C:\nv-vk64.json", 1)
            if key is subkey1 and name == "VulkanDriverName":
                # REG_MULTI_SZ surfaces as list[str]
                return (
                    [r"C:\amdvlk64.json", "", r"C:\amdvlk32.json"],
                    7,
                )
            raise OSError("missing value")

        winreg.QueryValueEx.side_effect = _query

        result = list(
            vulkan_icd_discovery._iter_pnp_class_manifests(
                winreg, vulkan_icd_discovery._PNP_DISPLAY_ADAPTER_CLASS_GUID
            )
        )
        assert r"C:\nv-vk64.json" in result
        assert r"C:\amdvlk64.json" in result
        assert r"C:\amdvlk32.json" in result
        # Empty strings in REG_MULTI_SZ are filtered out.
        assert "" not in result

    def test_missing_root_returns_nothing(self) -> None:
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        winreg.OpenKey.side_effect = OSError("class GUID has no key")

        assert (
            list(
                vulkan_icd_discovery._iter_pnp_class_manifests(
                    winreg, vulkan_icd_discovery._PNP_DISPLAY_ADAPTER_CLASS_GUID
                )
            )
            == []
        )

    def test_unopenable_subkey_is_skipped(self) -> None:
        """A subkey that EnumKey returned but OpenKey fails on must not abort the walk."""
        from lilbee.providers.fleet import vulkan_icd_discovery

        winreg = mock.MagicMock(name="winreg")
        winreg.HKEY_LOCAL_MACHINE = 0
        class_root = mock.MagicMock(name="class_root")
        subkey1 = mock.MagicMock(name="0001")

        # First OpenKey opens the class root, second opens 0000 (fails),
        # third opens 0001 (succeeds).
        winreg.OpenKey.side_effect = [class_root, OSError("locked"), subkey1]
        winreg.EnumKey.side_effect = ["0000", "0001", OSError("end")]

        def _query(_k: object, name: str) -> tuple[str, int]:
            if name == "VulkanDriverName":
                return (r"C:\nv-vk64.json", 1)
            raise OSError("no Wow value")

        winreg.QueryValueEx.side_effect = _query

        result = list(
            vulkan_icd_discovery._iter_pnp_class_manifests(
                winreg, vulkan_icd_discovery._PNP_DISPLAY_ADAPTER_CLASS_GUID
            )
        )
        assert result == [r"C:\nv-vk64.json"]


class TestDisableConflictingVulkanIcds:
    """``disable_conflicting_vulkan_icds`` picks the preferred vendor on dual-vendor Windows.

    Detection is registry-only: no Vulkan call runs while the disable env var
    is still unset, so the buggy vendor's ICD never gets pre-loaded into
    the process before we ask the loader to skip it.
    """

    def test_pins_nvidia_when_amd_also_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NVIDIA + AMD installed on Windows -> disable AMD ICD globs, keep NVIDIA."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA, PCIVendorID.AMD},
        )
        result = gpu_select.disable_conflicting_vulkan_icds()
        assert result is not None
        assert "amdvlk*" in result
        assert "radeon*" in result
        # NVIDIA's own ICD must NOT be disabled -- it's the one we're keeping.
        assert "nv*" not in result

    def test_pins_amd_when_intel_also_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AMD-discrete + Intel-iGPU laptops keep AMD; the documented crash is AMD vs NVIDIA."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.AMD, PCIVendorID.INTEL},
        )
        result = gpu_select.disable_conflicting_vulkan_icds()
        assert result is not None
        assert "intel*" in result
        assert "igvk*" in result
        # AMD's own ICD must NOT be disabled -- it's the discrete card.
        assert "amdvlk*" not in result

    def test_pins_nvidia_when_all_three_vendors_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Triple-vendor host: NVIDIA wins per ``_PREFERRED_VENDOR_ORDER``."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA, PCIVendorID.AMD, PCIVendorID.INTEL},
        )
        result = gpu_select.disable_conflicting_vulkan_icds()
        assert result is not None
        assert "amdvlk*" in result
        assert "intel*" in result
        assert "nv*" not in result

    def test_returns_none_on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """macOS uses Metal directly; the Vulkan loader is not on the path."""
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "darwin")
        assert gpu_select.disable_conflicting_vulkan_icds() is None

    def test_active_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Linux gets the same dual-vendor pin (Steam overlay + AMDVLK/RADV are documented)."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "linux")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA, PCIVendorID.AMD},
        )
        result = gpu_select.disable_conflicting_vulkan_icds()
        assert result is not None
        # Same preference order applies: keep NVIDIA, disable AMD globs.
        assert "amdvlk*" in result
        assert "amd_icd*" in result  # Linux AMDVLK manifest filename
        assert "radeon*" in result  # Mesa RADV
        assert "nv*" not in result

    def test_returns_none_when_only_one_vendor_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-vendor systems aren't at risk; no pin needed."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA},
        )
        assert gpu_select.disable_conflicting_vulkan_icds() is None

    def test_returns_none_when_no_vendors_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No matching manifests in the registry -> no pin (e.g. no Vulkan installed)."""
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(gpu_select, "_vulkan_vendors_present", set)
        assert gpu_select.disable_conflicting_vulkan_icds() is None

    def test_user_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the user set any Vulkan ICD-selection env var, we don't override it."""
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA, PCIVendorID.AMD},
        )
        # Source the override-var list from the enum so the test moves in
        # lockstep with the production set (e.g., when a new loader env var
        # joins the spec).
        for env_var in gpu_select.VulkanIcdEnvVar:
            monkeypatch.setattr(gpu_select.os, "environ", {env_var.value: "user-set"})
            assert gpu_select.disable_conflicting_vulkan_icds() is None

    def test_cfg_gpu_devices_pin_defers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the user has set ``cfg.gpu_devices``, we leave vendor selection alone.

        The lilbee-level GPU pin is a deliberate user choice. Forcing a
        vendor disable on top would be surprising: the dual-vendor mitigation
        is for the case where lilbee picks the GPU; if the user has picked,
        defer to them. Without this defer an AMD-discrete + NVIDIA host
        where the user pinned device 0 (their AMD card) would silently
        get its AMD ICD disabled by the NVIDIA-first preference.
        """
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import gpu_select
        from lilbee.providers.fleet.gpu_select import PCIVendorID

        monkeypatch.setattr(gpu_select.sys, "platform", "win32")
        monkeypatch.setattr(gpu_select.os, "environ", {})
        monkeypatch.setattr(
            gpu_select,
            "_vulkan_vendors_present",
            lambda: {PCIVendorID.NVIDIA, PCIVendorID.AMD},
        )
        monkeypatch.setattr(cfg, "gpu_devices", "1")
        assert gpu_select.disable_conflicting_vulkan_icds() is None


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


class TestTrainCtxFromMeta:
    """``train_ctx_from_meta`` is the single guard against ``context_length=0``.

    Every ctx resolver in ``engine_params`` (chat, embed/rerank, vision)
    routes through it; these tests cover the helper in isolation so an
    edge case can be added here without a full GGUF fixture.
    """

    @staticmethod
    def _path() -> Path:
        return Path("/tmp/test-model.gguf")

    def test_returns_metadata_value_when_positive(self) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        meta = {"context_length": "8192"}
        assert train_ctx_from_meta(meta, fallback=2048, model_path=self._path()) == 8192

    def test_returns_fallback_when_meta_is_none(self) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        assert train_ctx_from_meta(None, fallback=2048, model_path=self._path()) == 2048

    def test_returns_fallback_when_context_length_missing(self) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        assert train_ctx_from_meta({}, fallback=4096, model_path=self._path()) == 4096

    def test_clamps_zero_to_fallback(self) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        meta = {"context_length": "0"}
        assert train_ctx_from_meta(meta, fallback=2048, model_path=self._path()) == 2048

    def test_clamps_negative_to_fallback(self) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        meta = {"context_length": "-1"}
        assert train_ctx_from_meta(meta, fallback=2048, model_path=self._path()) == 2048

    def test_clamps_unparseable_to_fallback(self, caplog) -> None:
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        meta = {"context_length": "garbage"}
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.gguf_meta"):
            assert train_ctx_from_meta(meta, fallback=2048, model_path=self._path()) == 2048
        assert any("unparseable" in rec.message for rec in caplog.records)

    def test_each_loader_uses_its_own_fallback(self) -> None:
        """Embed / chat / vision picks reflect their respective task budgets."""
        from lilbee.providers.gguf_meta import train_ctx_from_meta

        zero = {"context_length": "0"}
        path = self._path()
        assert train_ctx_from_meta(zero, fallback=2048, model_path=path) == 2048  # embed
        assert train_ctx_from_meta(zero, fallback=8192, model_path=path) == 8192  # chat
        assert train_ctx_from_meta(zero, fallback=4096, model_path=path) == 4096  # vision


class TestReadGgufMetadata:
    def test_reads_all_fields(self, tmp_path: Path) -> None:
        """read_gguf_metadata returns the parsed header fields."""
        from lilbee.providers.gguf_meta import read_gguf_metadata

        path = write_test_gguf(
            tmp_path / "model.gguf",
            arch="llama",
            fields={
                "llama.context_length": 4096,
                "llama.embedding_length": 4096,
                "tokenizer.chat_template": "template",
                "general.file_type": 7,
                "general.name": "Test Model",
            },
        )

        assert read_gguf_metadata(path) == {
            "architecture": "llama",
            "context_length": "4096",
            "embedding_length": "4096",
            "chat_template": "template",
            "file_type": "7",
            "name": "Test Model",
        }

    def test_exposes_declared_pooling_type(self, tmp_path: Path) -> None:
        """An embedder's <arch>.pooling_type is surfaced for the embed-pooling choice."""
        from lilbee.providers.gguf_meta import read_gguf_metadata

        path = write_test_gguf(
            tmp_path / "model.gguf",
            arch="qwen3",
            fields={"qwen3.pooling_type": 3},
        )

        assert read_gguf_metadata(path) == {"architecture": "qwen3", "pooling_type": "3"}

    def test_returns_none_for_empty_metadata(self, tmp_path: Path) -> None:
        """read_gguf_metadata returns None when the header carries no fields."""
        from lilbee.providers.gguf_meta import read_gguf_metadata

        reader = mock.MagicMock()
        reader.fields = {}
        with mock.patch("lilbee.providers.gguf_meta.GGUFReader", return_value=reader):
            assert read_gguf_metadata(tmp_path / "model.gguf") is None

    def test_caches_by_path_and_mtime(self, tmp_path: Path) -> None:
        """A second read of the same file reuses the cache and never re-parses.

        Planning reads each model's metadata several times per build; the cache
        turns those repeats (each a full GGUFReader parse) into one.
        """
        from lilbee.providers import gguf_meta

        path = write_test_gguf(
            tmp_path / "model.gguf", arch="llama", fields={"llama.context_length": 4096}
        )
        gguf_meta._METADATA_CACHE.clear()
        with mock.patch.object(gguf_meta, "GGUFReader", wraps=gguf_meta.GGUFReader) as spy:
            first = gguf_meta.read_gguf_metadata(path)
            second = gguf_meta.read_gguf_metadata(path)
        assert first == second == {"architecture": "llama", "context_length": "4096"}
        assert spy.call_count == 1  # parsed once; second call served from cache
        assert second is not first  # returns a copy so callers can't mutate the entry


class TestFindMmprojForModel:
    def test_catalog_lookup(self) -> None:
        """find_mmproj_for_model uses catalog lookup first."""
        from lilbee.providers.gguf_meta import find_mmproj_for_model

        with mock.patch(
            "lilbee.catalog.find_mmproj_file",
            return_value=Path("/found.gguf"),
        ):
            result = find_mmproj_for_model(Path("/models/model.gguf"))

        assert result == Path("/found.gguf")

    def test_directory_fallback(self, tmp_path: Path) -> None:
        """find_mmproj_for_model falls back to directory scan."""
        from lilbee.providers.gguf_meta import find_mmproj_for_model

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
        from lilbee.providers.gguf_meta import find_mmproj_for_model

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
        from lilbee.providers.gguf_meta import find_mmproj_for_model

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
        from lilbee.providers.gguf_meta import find_mmproj_for_model

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
        from lilbee.providers.gguf_meta import find_mmproj_for_model

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

        from lilbee.providers.gguf_meta import read_mmproj_projector_type

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

        from lilbee.providers.gguf_meta import read_mmproj_projector_type

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


class TestOpenAIModelsUrl:
    def test_appends_v1_models_when_absent(self) -> None:
        from lilbee.providers.local_servers import openai_models_url

        assert openai_models_url("http://localhost:11434") == "http://localhost:11434/v1/models"

    def test_does_not_double_v1_when_present(self) -> None:
        from lilbee.providers.local_servers import openai_models_url

        assert openai_models_url("http://localhost:1234/v1") == "http://localhost:1234/v1/models"

    def test_tolerates_trailing_slash(self) -> None:
        from lilbee.providers.local_servers import openai_models_url

        assert openai_models_url("http://localhost:1234/v1/") == "http://localhost:1234/v1/models"


class TestDetectLocalServer:
    def test_ollama_default_port(self) -> None:
        from lilbee.providers.local_servers import OLLAMA, detect_local_server

        assert detect_local_server("http://localhost:11434") is OLLAMA

    def test_ollama_127_default_port(self) -> None:
        from lilbee.providers.local_servers import OLLAMA, detect_local_server

        assert detect_local_server("http://127.0.0.1:11434") is OLLAMA

    def test_ollama_in_url(self) -> None:
        from lilbee.providers.local_servers import OLLAMA, detect_local_server

        assert detect_local_server("https://ollama.example.com") is OLLAMA

    def test_lm_studio_default_port(self) -> None:
        from lilbee.providers.local_servers import LM_STUDIO, detect_local_server

        assert detect_local_server("http://localhost:1234") is LM_STUDIO

    def test_lm_studio_127_with_v1(self) -> None:
        from lilbee.providers.local_servers import LM_STUDIO, detect_local_server

        assert detect_local_server("http://127.0.0.1:1234/v1") is LM_STUDIO

    def test_lm_studio_port_not_confused_with_ollama(self) -> None:
        """LM Studio's pattern must not match Ollama's longer port substring."""
        from lilbee.providers.local_servers import OLLAMA, detect_local_server

        assert detect_local_server("http://localhost:11434") is OLLAMA

    def test_openai_url(self) -> None:
        from lilbee.providers.local_servers import detect_local_server

        assert detect_local_server("https://api.openai.com") is None

    def test_custom_url(self) -> None:
        from lilbee.providers.local_servers import detect_local_server

        assert detect_local_server("http://myserver:8080") is None


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

    def test_lm_studio_ref_routes_to_lm_studio_prefix(self) -> None:
        """LM Studio refs carry litellm's ``lm_studio/`` provider prefix."""
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("lm_studio/qwen2.5-7b-instruct")
        assert _route_model(ref, "http://localhost:1234/v1") == "lm_studio/qwen2.5-7b-instruct"

    def test_local_ref_on_lm_studio_url_adds_prefix(self) -> None:
        """A bare local ref forced through an LM Studio base URL gets prefixed."""
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
        assert _route_model(ref, "http://localhost:1234/v1") == f"lm_studio/{ref.name}"

    def test_lm_studio_ref_routes_regardless_of_api_base(self) -> None:
        """An ``lm_studio/`` ref keeps its prefix even with no api_base set."""
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("lm_studio/some-model")
        assert _route_model(ref, None) == "lm_studio/some-model"


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
    """``list_models`` merges across configured local servers; each server's
    URL selects its listing endpoint in the backend. Pinning a single
    configured server keeps the per-endpoint routing assertions sharp."""

    def test_ollama_url_uses_api_tags(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.local_servers import OLLAMA
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with (
            mock.patch(
                "lilbee.providers.sdk_llm_provider.configured_local_servers",
                return_value=[(OLLAMA, "http://localhost:11434")],
            ),
            mock.patch("httpx.get", return_value=mock_resp) as mock_get,
        ):
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "api/tags" in mock_get.call_args[0][0]
        assert result == ["llama3:8b"]

    def test_non_ollama_url_uses_v1_models(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.local_servers import LM_STUDIO
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), api_key="sk-test")
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with (
            mock.patch(
                "lilbee.providers.sdk_llm_provider.configured_local_servers",
                return_value=[(LM_STUDIO, "https://api.openai.com")],
            ),
            mock.patch("httpx.get", return_value=mock_resp) as mock_get,
        ):
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "v1/models" in mock_get.call_args[0][0]
        assert result == ["gpt-4o", "gpt-4o-mini"]

    def test_non_ollama_returns_empty_on_error(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.local_servers import LM_STUDIO
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())

        with (
            mock.patch(
                "lilbee.providers.sdk_llm_provider.configured_local_servers",
                return_value=[(LM_STUDIO, "https://api.openai.com")],
            ),
            mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")),
        ):
            result = provider.list_models()

        assert result == []

    def test_v1_models_sends_auth_header(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.local_servers import LM_STUDIO
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), api_key="sk-secret")
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = mock.MagicMock()

        with (
            mock.patch(
                "lilbee.providers.sdk_llm_provider.configured_local_servers",
                return_value=[(LM_STUDIO, "https://api.openai.com")],
            ),
            mock.patch("httpx.get", return_value=mock_resp) as mock_get,
        ):
            provider.list_models()

        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-secret"


class TestSdkLLMProviderVisionOcr:
    """``SdkLLMProvider.vision_ocr`` translates to a multipart chat call."""

    def _make_provider(self) -> SdkLLMProvider:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend())

    def test_builds_multipart_message_and_routes_to_chat(self) -> None:
        from lilbee.providers.base import ChatResult, FinishReason

        provider = self._make_provider()
        chat_result = ChatResult(text="page text", tool_calls=(), finish_reason=FinishReason.STOP)
        with mock.patch.object(provider, "chat", return_value=chat_result) as mock_chat:
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
        from lilbee.providers.base import ChatResult, FinishReason
        from lilbee.vision import OCR_PROMPT

        provider = self._make_provider()
        chat_result = ChatResult(text="ok", tool_calls=(), finish_reason=FinishReason.STOP)
        with mock.patch.object(provider, "chat", return_value=chat_result) as mock_chat:
            provider.vision_ocr(b"\x89PNG", "ollama/llava:7b")

        text_part = mock_chat.call_args[0][0][0]["content"][1]
        assert text_part["text"] == OCR_PROMPT

    def test_positive_timeout_returns_chat_result(self) -> None:
        """A non-expiring positive timeout returns the chat response unchanged."""
        from lilbee.providers.base import ChatResult, FinishReason

        provider = self._make_provider()
        chat_result = ChatResult(text="ok", tool_calls=(), finish_reason=FinishReason.STOP)
        with mock.patch.object(provider, "chat", return_value=chat_result) as mock_chat:
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

    def test_timeout_frees_caller_without_waiting_for_hung_call(self) -> None:
        # On timeout the caller must be freed at the deadline, not
        # blocked by the pool's shutdown(wait=True) until the hung call returns.
        import threading
        import time

        provider = self._make_provider()
        release = threading.Event()

        def hung_chat(*_args, **_kwargs):
            release.wait(timeout=10)
            return "too late"

        start = time.monotonic()
        with (
            mock.patch.object(provider, "chat", side_effect=hung_chat),
            pytest.raises(TimeoutError),
        ):
            provider.vision_ocr(b"\x89PNG", "ollama/llava:7b", "p", timeout=0.1)
        elapsed = time.monotonic() - start
        release.set()  # let the orphaned worker finish
        assert elapsed < 2.0  # freed at the deadline, not blocked on the 10s call

    def test_zero_timeout_returns_chat_result(self) -> None:
        """``timeout=0`` skips the thread pool and returns chat's result."""
        from lilbee.providers.base import ChatResult, FinishReason

        provider = self._make_provider()
        chat_result = ChatResult(text="ok", tool_calls=(), finish_reason=FinishReason.STOP)
        with mock.patch.object(provider, "chat", return_value=chat_result) as mock_chat:
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

        cfg.ollama_base_url = "http://localhost:11434"
        provider = SdkLLMProvider(LitellmSdkBackend())
        fake = self._make_fake_litellm()

        with inject_modules({"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="ollama/qwen3:0.6b")

        call_kwargs = fake.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert call_kwargs["model"] == "ollama/qwen3:0.6b"

    def test_frontier_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        fake = self._make_fake_litellm()

        with inject_modules({"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    def test_anthropic_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        fake = self._make_fake_litellm()

        with inject_modules({"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="anthropic/claude-sonnet-4-6")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs

    def test_chat_calls_inject_provider_keys(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        fake = self._make_fake_litellm()

        with (
            inject_modules({"litellm": fake}),
            mock.patch("lilbee.providers.sdk_llm_provider.inject_provider_keys") as mock_inject,
        ):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        mock_inject.assert_called_once()


class TestEmbedApiBaseRouting:
    """Verify that embed() omits api_base for non-Ollama provider-prefixed models."""

    def test_ollama_embed_passes_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        cfg.ollama_base_url = "http://localhost:11434"
        provider = SdkLLMProvider(LitellmSdkBackend())
        cfg.embedding_model = "ollama/nomic-embed-text:latest"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        with inject_modules({"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_prefixed_embed_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend())
        cfg.embedding_model = "openai/text-embedding-3-small"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        with inject_modules({"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert "api_base" not in call_kwargs


class TestSdkRerank:
    """Coverage for SdkLLMProvider.rerank + LitellmSdkBackend.rerank."""

    def _make_sdk_provider(self):
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend())

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
        provider = _SdkLLMProvider(LitellmSdkBackend())
        with (
            mock.patch.object(provider._backend, "rerank", side_effect=RuntimeError("wire error")),
            pytest.raises(ProviderError, match="Rerank failed: wire error"),
        ):
            provider.rerank("q", ["a"])


class TestRoutingProviderRerank:
    """Routing-level rerank dispatch between the native fleet and hosted SDK."""

    def _make_provider(self):
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    def test_rerank_routes_hosted_ref_to_sdk(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = True
        mock_sdk.rerank.return_value = [0.9, 0.1]
        rp._local = mock_llama
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
        rp._local = mock_llama
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
        rp._local = mock_llama

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

    def test_rerank_routes_bare_gguf_to_local_engine(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_llama.rerank.return_value = [0.5, 0.5]
        rp._local = mock_llama
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

    def test_lm_studio_gguf_path_reranker_is_not_native(self) -> None:
        """A local-server-prefixed GGUF path keeps the prefix exemption, like chat refs."""
        from lilbee.providers.routing_provider import _is_native_rerank_ref

        assert _is_native_rerank_ref("lm_studio/TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf") is False

    def test_rerank_routes_lm_studio_gguf_path_to_sdk(self) -> None:
        """An ``lm_studio/`` reranker whose id looks like a GGUF path goes to LM Studio."""
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = True
        mock_sdk.rerank.return_value = [0.7, 0.3]
        rp._local = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "lm_studio/TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf"
        scores = rp.rerank("q", ["a", "b"])
        assert scores == [0.7, 0.3]
        mock_sdk.rerank.assert_called_once_with("q", ["a", "b"])
        mock_llama.rerank.assert_not_called()

    def test_rerank_with_empty_model_raises_provider_error(self) -> None:
        """rerank() raises ProviderError when cfg.reranker_model is empty."""
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        cfg.reranker_model = ""
        with pytest.raises(ProviderError, match="No reranker configured"):
            rp.rerank("q", ["a", "b"])


class TestRoutingProviderPdfOcr:
    """``RoutingProvider.pdf_ocr`` dispatches by ref prefix, like ``vision_ocr``."""

    def test_native_ref_routes_to_local_engine(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        mock_native = mock.MagicMock()
        mock_native.pdf_ocr.return_value = ["p1", "p2"]
        rp._local = mock_native
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


class TestChatWithToolsRouting:
    def test_base_default_raises(self) -> None:
        from lilbee.providers.base import LLMProvider, ProviderError

        class _Bare(LLMProvider): ...

        with pytest.raises(ProviderError, match="does not support tool calling"):
            _Bare().chat_with_tools([], tools=[])

    def test_routing_dispatches_to_picked_backend(self, monkeypatch) -> None:
        from lilbee.providers.base import ChatToolResult
        from lilbee.providers.routing_provider import RoutingProvider

        backend = mock.MagicMock()
        backend.chat_with_tools.return_value = ChatToolResult(content="", tool_calls=[])
        rp = RoutingProvider()
        monkeypatch.setattr(rp, "_pick_backend", lambda _ref: backend)
        cfg.chat_model = "org/repo/model.gguf"
        rp.chat_with_tools(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
        )
        backend.chat_with_tools.assert_called_once()
        assert backend.chat_with_tools.call_args.kwargs["tool_choice"] == "auto"


class TestRoutingLifecycleForwarding:
    def test_cancel_and_reload_forward_to_local_when_present(self) -> None:
        from lilbee.providers.roles import WorkerRole
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        local = mock.MagicMock()
        rp._local = local
        rp.cancel_inference()
        local.cancel_inference.assert_called_once_with()
        rp.reload_role(WorkerRole.EMBED)
        local.reload_role.assert_called_once_with(WorkerRole.EMBED, wait=False)
        rp.reload_placement(wait=True)
        local.reload_placement.assert_called_once_with(wait=True)

    def test_cancel_and_reload_are_noop_without_local(self) -> None:
        from lilbee.providers.roles import WorkerRole
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()  # _local is None
        rp.cancel_inference()  # must not raise
        rp.reload_role(WorkerRole.CHAT)  # must not raise
        rp.reload_placement()  # must not raise

    def test_drop_loaded_models_async_forwards_to_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        local = mock.MagicMock()
        rp._local = local
        rp.drop_loaded_models_async()
        local.drop_loaded_models_async.assert_called_once_with()

    def test_drop_loaded_models_async_noop_without_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        RoutingProvider().drop_loaded_models_async()  # _local is None: must not raise

    def test_max_concurrent_chats_defaults_to_one_without_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        assert RoutingProvider().max_concurrent_chats() == 1  # _local is None

    def test_max_concurrent_chats_forwards_to_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        rp._local = mock.MagicMock()
        rp._local.max_concurrent_chats.return_value = 3
        assert rp.max_concurrent_chats() == 3

    def test_served_chat_ctx_is_none_without_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        assert RoutingProvider().served_chat_ctx() is None  # _local is None

    def test_served_chat_ctx_forwards_to_local(self) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        rp._local = mock.MagicMock()
        rp._local.served_chat_ctx.return_value = 16384
        assert rp.served_chat_ctx() == 16384

    def test_role_ready_forwards_to_local(self) -> None:
        from lilbee.providers.roles import WorkerRole
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        local = mock.MagicMock()
        local.role_ready.return_value = False
        rp._local = local
        assert rp.role_ready(WorkerRole.CHAT) is False
        local.role_ready.assert_called_once_with(WorkerRole.CHAT)

    def test_role_ready_true_without_local(self) -> None:
        from lilbee.providers.roles import WorkerRole
        from lilbee.providers.routing_provider import RoutingProvider

        # No local engine yet: treat as reachable so callers don't show a stuck
        # warming state before the fleet is ever constructed.
        assert RoutingProvider().role_ready(WorkerRole.CHAT) is True


def test_gguf_scalar_str_array_field_returns_none() -> None:
    from types import SimpleNamespace

    from gguf import GGUFValueType

    from lilbee.catalog.header_probe import gguf_scalar_str

    # An ARRAY-typed scalar field is not renderable as a single value.
    field = SimpleNamespace(types=[GGUFValueType.ARRAY], data=[0], parts=[b"x"])
    assert gguf_scalar_str(field) is None
