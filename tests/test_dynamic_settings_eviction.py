"""Settings UI to model cache eviction wiring."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.cli.tui.app import LilbeeApp, _on_settings_changed_evict_cache
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
    from lilbee.core.services import set_services

    provider = _RecordingProvider()
    services = mock.MagicMock()
    services.provider = provider
    set_services(services)
    return provider


def _restore_services() -> None:
    from lilbee.core.services import set_services

    set_services(None)


def test_load_affecting_key_evicts_cache():
    """num_ctx change triggers invalidate_load_cache on the active provider."""
    provider = _install_recording_provider()
    try:
        _on_settings_changed_evict_cache(("num_ctx", 4096))
        assert provider.calls == [None]
    finally:
        _restore_services()


def test_model_name_change_evicts_cache():
    """Switching chat_model via Settings UI evicts the cache so the old model is gone."""
    provider = _install_recording_provider()
    try:
        _on_settings_changed_evict_cache(("chat_model", "qwen3:8b"))
        _on_settings_changed_evict_cache(("embedding_model", "nomic-embed-text"))
        _on_settings_changed_evict_cache(("vision_model", "llava:7b"))
        _on_settings_changed_evict_cache(("reranker_model", "bge-reranker-v2"))
        assert len(provider.calls) == 4
        assert all(c is None for c in provider.calls)
    finally:
        _restore_services()


def test_sampling_param_change_does_not_evict():
    """Temperature, top_p, etc. are read per-call; eviction would be wasted work."""
    provider = _install_recording_provider()
    try:
        _on_settings_changed_evict_cache(("temperature", 0.7))
        _on_settings_changed_evict_cache(("top_p", 0.9))
        _on_settings_changed_evict_cache(("top_k_sampling", 40))
        _on_settings_changed_evict_cache(("repeat_penalty", 1.2))
        _on_settings_changed_evict_cache(("seed", 42))
        _on_settings_changed_evict_cache(("max_tokens", 1024))
        _on_settings_changed_evict_cache(("rag_system_prompt", "You are helpful"))
        assert provider.calls == []
    finally:
        _restore_services()


def test_unknown_key_does_not_evict():
    """An unrelated setting change is a no-op."""
    provider = _install_recording_provider()
    try:
        _on_settings_changed_evict_cache(("theme", "dracula"))
        _on_settings_changed_evict_cache(("wiki", True))
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


# ---------------------------------------------------------------------------
# End-to-end: boot LilbeeApp, fire signal, observe provider call.
# ---------------------------------------------------------------------------


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


async def test_app_subscription_evicts_when_signal_fires(_patch_chat_setup):
    """End-to-end: LilbeeApp.on_mount subscribes; publishing the signal triggers eviction."""
    provider = _install_recording_provider()
    try:
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

            app.settings_changed_signal.publish(("num_ctx", 16384))
            await pilot.pause()

            assert provider.calls == [None]

            # Sampling param does not propagate
            app.settings_changed_signal.publish(("temperature", 0.5))
            await pilot.pause()
            assert provider.calls == [None]
    finally:
        _restore_services()


async def test_provider_availability_signal_fires_for_api_keys(_patch_chat_setup):
    """Adding an API key republishes on provider_availability_changed_signal.

    Subscribers (catalog, model picker) listen to the higher-level signal
    instead of duplicating the PROVIDER_API_KEYS whitelist.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        received: list[tuple[str, object]] = []
        app.provider_availability_changed_signal.subscribe(app, received.append)

        app.settings_changed_signal.publish(("gemini_api_key", "sk-test"))
        await pilot.pause()
        assert received == [("gemini_api_key", "sk-test")]

        # Non-key changes do not propagate.
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
