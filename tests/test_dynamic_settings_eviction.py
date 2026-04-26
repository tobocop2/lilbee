"""Settings UI -> model cache eviction wiring.

Verifies that publishing settings_changed_signal for a load-affecting key
calls invalidate_load_cache() on the active provider, while ignoring
sampling-only keys. The provider Protocol's no-op default keeps litellm
backends untouched; only native llama-cpp evicts.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from lilbee.cli.tui.app import LilbeeApp, _on_settings_changed_evict_cache
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.config import cfg
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
    cfg.chat_model = "test-chat-model.gguf"
    cfg.embedding_model = "test-embed-model"
    cfg.subprocess_embed = False
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
    from lilbee.services import set_services

    provider = _RecordingProvider()
    services = mock.MagicMock()
    services.provider = provider
    set_services(services)
    return provider


def _restore_services() -> None:
    from lilbee.services import set_services

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
        _on_settings_changed_evict_cache(("system_prompt", "You are helpful"))
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
