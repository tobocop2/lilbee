"""Settings write boundary to model cache eviction wiring."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.core.config import cfg
from lilbee.providers.base import LLMProvider


class _RecordingProvider:
    """Stand-in provider that records invalidate_load_cache invocations."""

    def __init__(self) -> None:
        self.calls: list[Path | None] = []

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        self.calls.append(model_path)


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = "ollama/test-chat-model:v1"
    cfg.embedding_model = "ollama/test-embed-model:v1"
    cfg.wiki = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for name in type(snapshot).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _install_recording_provider() -> _RecordingProvider:
    """Replace the services container with one whose provider records eviction calls."""
    from lilbee.app.services import set_services

    provider = _RecordingProvider()
    services = mock.MagicMock()
    services.provider = provider
    set_services(services)
    return provider


def _restore_services() -> None:
    from lilbee.app.services import set_services

    set_services(None)


def test_load_affecting_key_evicts_cache():
    """num_ctx change triggers invalidate_load_cache on the active provider."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"num_ctx": 4096})
        assert provider.calls == [None]
    finally:
        _restore_services()


def test_non_reloadable_model_change_evicts_cache():
    """Switching embedding_model or reranker_model evicts the cache so the
    next call respawns under the new cfg. These workers do not honor a
    per-call ``request.model`` override."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"embedding_model": "nomic-ai/nomic-embed-text-v1.5-GGUF"})
        apply_settings_update({"reranker_model": "ggml-org/bge-reranker-v2-m3-Q8_0-GGUF"})
        assert len(provider.calls) == 2
        assert all(c is None for c in provider.calls)
    finally:
        _restore_services()


def test_per_call_reloadable_model_swap_skips_provider_eviction():
    """Swapping chat_model or vision_model to a different ref does NOT touch
    the provider load cache: the chat / vision workers reload in place via
    ``_ensure_loaded`` on the next request, saving the 1-3 s spawn cost.
    Both calls exercise a real ref-to-ref swap, not the disable path."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"chat_model": "Qwen/Qwen3-0.6B-GGUF"})
        apply_settings_update({"vision_model": "lightonai/LightOnOCR-2.1B-GGUF"})
        assert provider.calls == []
    finally:
        _restore_services()


def test_sampling_param_change_does_not_evict():
    """Temperature, top_p, etc. are read per-call; eviction would be wasted work."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"temperature": 0.7})
        apply_settings_update({"top_p": 0.9})
        apply_settings_update({"top_k_sampling": 40})
        apply_settings_update({"repeat_penalty": 1.2})
        apply_settings_update({"seed": 42})
        apply_settings_update({"max_tokens": 1024})
        apply_settings_update({"rag_system_prompt": "You are helpful"})
        assert provider.calls == []
    finally:
        _restore_services()


def test_unknown_key_does_not_evict():
    """An unrelated setting change is a no-op for the provider cache."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"theme": "dracula"})
        apply_settings_update({"wiki": True})
        assert provider.calls == []
    finally:
        _restore_services()


def test_protocol_default_is_safe_for_litellm_provider():
    """A backend that subclasses LLMProvider but doesn't override
    invalidate_load_cache gets the Protocol's no-op default body."""

    class _BackendWithNoOverride(LLMProvider):  # type: ignore[misc]
        """Concrete subclass without an explicit invalidate_load_cache override."""

    backend = _BackendWithNoOverride()
    assert backend.invalidate_load_cache() is None
    assert backend.invalidate_load_cache(Path("/tmp/whatever.gguf")) is None


@pytest.fixture()
def _patch_chat_setup():
    with (
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._needs_setup",
            return_value=False,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        yield


async def test_app_set_setting_evicts_via_boundary(_patch_chat_setup):
    """End-to-end: LilbeeApp.set_setting routes through the write boundary."""
    provider = _install_recording_provider()
    try:
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

            app.set_setting("num_ctx", 16384)
            await pilot.pause()
            assert provider.calls == [None]

            # Sampling param does not touch the provider.
            app.set_setting("temperature", 0.5)
            await pilot.pause()
            assert provider.calls == [None]
    finally:
        _restore_services()


async def test_provider_availability_signal_fires_for_api_keys(_patch_chat_setup):
    """Adding an API key republishes on provider_availability_changed_signal."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        received: list[tuple[str, object]] = []
        app.provider_availability_changed_signal.subscribe(app, received.append)

        app.settings_changed_signal.publish(("gemini_api_key", "sk-test"))
        await pilot.pause()
        assert received == [("gemini_api_key", "sk-test")]

        app.settings_changed_signal.publish(("temperature", 0.5))
        await pilot.pause()
        assert received == [("gemini_api_key", "sk-test")]


async def test_provider_availability_signal_fires_for_each_provider_key(_patch_chat_setup):
    """The whitelist covers every provider key declared on the Config."""
    from lilbee.core.config.keys import PROVIDER_API_KEYS

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        received: list[tuple[str, object]] = []
        app.provider_availability_changed_signal.subscribe(app, received.append)

        for key in sorted(PROVIDER_API_KEYS):
            app.settings_changed_signal.publish((key, "value"))
            await pilot.pause()

        assert {payload[0] for payload in received} == set(PROVIDER_API_KEYS)
