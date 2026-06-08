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
    """Stand-in provider recording the lifecycle calls the boundary makes.

    ``calls`` keeps the legacy invalidate_load_cache trail; ``reloaded_roles``
    and ``dropped`` capture the new off-thread per-role reload and whole-fleet
    drop that a load-affecting change now routes to.
    """

    def __init__(self) -> None:
        self.calls: list[Path | None] = []
        self.reloaded_roles: list[object] = []
        self.dropped = 0

    def invalidate_load_cache(self, model_path: Path | None = None) -> None:
        self.calls.append(model_path)

    def reload_role(self, role: object) -> None:
        self.reloaded_roles.append(role)

    def drop_loaded_models_async(self) -> None:
        self.dropped += 1


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
    """Replace the services container with one whose provider records lifecycle calls."""
    from lilbee.app.services import set_services

    provider = _RecordingProvider()
    services = mock.MagicMock()
    services.provider = provider
    # Mirror the real Services.reload_role pass-through so the boundary's
    # per-role reload reaches the recording provider.
    services.reload_role = provider.reload_role
    set_services(services)
    return provider


def _restore_services() -> None:
    from lilbee.app.services import set_services

    set_services(None)


def test_role_agnostic_load_key_drops_fleet_off_thread():
    """num_ctx has no single owning role, so it drops the whole fleet off-thread."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"num_ctx": 4096})
        assert provider.dropped == 1
        assert provider.reloaded_roles == []
    finally:
        _restore_services()


def test_num_ctx_max_change_drops_fleet():
    """num_ctx_max feeds the chat ctx picker but is role-agnostic at this layer,
    so it routes to the off-thread whole-fleet drop, not a per-role reload."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"num_ctx_max": 131072})
        assert provider.dropped == 1
        assert provider.reloaded_roles == []
    finally:
        _restore_services()


def test_kv_cache_type_change_drops_fleet():
    """kv_cache_type is a load-time launch flag; a change drops the fleet so the
    next call respawns under the new value."""
    from lilbee.core.config.enums import KvCacheType

    provider = _install_recording_provider()
    try:
        apply_settings_update({"kv_cache_type": KvCacheType.Q8_0})
        assert provider.dropped == 1
        assert provider.reloaded_roles == []
    finally:
        _restore_services()


def test_embed_and_rerank_model_change_reloads_only_those_roles():
    """Switching embedding_model / reranker_model reloads just that role's server
    off-thread; the whole fleet is never dropped (other roles keep serving)."""
    from lilbee.providers.roles import WorkerRole

    provider = _install_recording_provider()
    try:
        apply_settings_update({"embedding_model": "nomic-ai/nomic-embed-text-v1.5-GGUF"})
        apply_settings_update({"reranker_model": "ggml-org/bge-reranker-v2-m3-Q8_0-GGUF"})
        assert provider.reloaded_roles == [WorkerRole.EMBED, WorkerRole.RERANK]
        assert provider.dropped == 0
    finally:
        _restore_services()


def test_embedding_model_change_derives_embedding_dim(monkeypatch):
    """Switching embedding_model also persists the model's output width (from its
    GGUF header), so a fresh index is built at the right dimension instead of the
    768 default -- the gap that broke a Qwen3-Embedding (4096) corpus build."""
    from lilbee.app import settings as settings_mod

    _install_recording_provider()
    try:
        monkeypatch.setattr(settings_mod, "_embedder_dim_from_gguf", lambda _ref: 4096)
        apply_settings_update({"embedding_model": "Qwen/Qwen3-Embedding-8B-GGUF/x.gguf"})
        assert cfg.embedding_dim == 4096
    finally:
        _restore_services()


def test_embedding_model_change_leaves_dim_when_unreadable(monkeypatch):
    """An unresolvable/headerless embedder leaves embedding_dim untouched (no crash)."""
    from lilbee.app import settings as settings_mod

    _install_recording_provider()
    cfg.embedding_dim = 768
    try:
        monkeypatch.setattr(settings_mod, "_embedder_dim_from_gguf", lambda _ref: None)
        apply_settings_update({"embedding_model": "some/unreadable.gguf"})
        assert cfg.embedding_dim == 768
    finally:
        _restore_services()


def test_embedder_dim_from_gguf_reads_embedding_length(monkeypatch):
    from pathlib import Path

    from lilbee.app.settings import _embedder_dim_from_gguf

    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _ref: Path("/x.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"architecture": "bert", "embedding_length": "384"},
    )
    assert _embedder_dim_from_gguf("ref") == 384


def test_embedder_dim_from_gguf_handles_unresolvable_and_missing(monkeypatch):
    from pathlib import Path

    from lilbee.app.settings import _embedder_dim_from_gguf

    def boom(_ref):
        raise OSError("not installed")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", boom)
    assert _embedder_dim_from_gguf("ref") is None

    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _ref: Path("/x.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "bert"}
    )
    assert _embedder_dim_from_gguf("ref") is None

    # Non-integer or non-positive embedding_length is junk -> leave dim untouched.
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata",
        lambda _p: {"embedding_length": "notanint"},
    )
    assert _embedder_dim_from_gguf("ref") is None
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"embedding_length": "0"}
    )
    assert _embedder_dim_from_gguf("ref") is None


def test_chat_and_vision_model_change_reloads_those_roles():
    """A chat_model / vision_model swap reloads that role's server so the next
    request uses the new model. The fleet serves the configured model per role
    and rejects per-call overrides, so the reload is required (not optional)."""
    from lilbee.providers.roles import WorkerRole

    provider = _install_recording_provider()
    try:
        apply_settings_update({"chat_model": "Qwen/Qwen3-0.6B-GGUF"})
        apply_settings_update({"vision_model": "lightonai/LightOnOCR-2.1B-GGUF"})
        assert provider.reloaded_roles == [WorkerRole.CHAT, WorkerRole.VISION]
        assert provider.dropped == 0
    finally:
        _restore_services()


def test_sampling_param_change_does_not_touch_fleet():
    """Temperature, top_p, etc. are read per-call; no reload or drop is needed."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"temperature": 0.7})
        apply_settings_update({"top_p": 0.9})
        apply_settings_update({"top_k_sampling": 40})
        apply_settings_update({"repeat_penalty": 1.2})
        apply_settings_update({"seed": 42})
        apply_settings_update({"max_tokens": 1024})
        apply_settings_update({"rag_system_prompt": "You are helpful"})
        assert provider.reloaded_roles == []
        assert provider.dropped == 0
    finally:
        _restore_services()


def test_unknown_key_does_not_touch_fleet():
    """An unrelated setting change is a no-op for the fleet."""
    provider = _install_recording_provider()
    try:
        apply_settings_update({"theme": "dracula"})
        apply_settings_update({"wiki": True})
        assert provider.reloaded_roles == []
        assert provider.dropped == 0
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


def test_protocol_defaults_for_drop_and_role_ready():
    """A backend without managed servers gets the Protocol's safe defaults:
    drop_loaded_models_async delegates to the no-op invalidate, and every role
    reports ready (the SDK side is always reachable)."""
    from lilbee.providers.roles import WorkerRole

    class _BackendWithNoOverride(LLMProvider):  # type: ignore[misc]
        """Concrete subclass relying entirely on Protocol defaults."""

    backend = _BackendWithNoOverride()
    assert backend.drop_loaded_models_async() is None
    assert backend.role_ready(WorkerRole.CHAT) is True
    assert backend.role_ready(WorkerRole.EMBED) is True


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
            assert provider.dropped == 1

            # Sampling param does not touch the fleet.
            app.set_setting("temperature", 0.5)
            await pilot.pause()
            assert provider.dropped == 1
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


def test_llm_provider_write_resets_services():
    """Writing llm_provider through the boundary tears down the services singleton."""
    with mock.patch("lilbee.app.services.reset_services") as mock_reset:
        apply_settings_update({"llm_provider": "remote"})
    mock_reset.assert_called_once()


def test_unrelated_write_does_not_reset_services():
    """A write that isn't in PROVIDER_SWITCHING_KEYS leaves services alone."""
    with mock.patch("lilbee.app.services.reset_services") as mock_reset:
        apply_settings_update({"top_k": 12})
    mock_reset.assert_not_called()
