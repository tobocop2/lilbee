"""Tests for per-role readiness: can each model role serve right now."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.app.setup_state import chat_ready, embedding_ready, is_fresh_install
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat


@pytest.fixture
def isolated_data_dir(tmp_path):
    """Point cfg at a per-test data directory and restore the full snapshot."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    try:
        yield tmp_path
    finally:
        for field_name in type(snapshot).model_fields:
            setattr(cfg, field_name, getattr(snapshot, field_name))


def test_a_missing_lancedb_dir_is_a_fresh_install(isolated_data_dir):
    """A fresh data dir means the wizard runs, whatever the models say."""
    assert not cfg.lancedb_dir.exists()
    assert is_fresh_install() is True


def test_an_existing_lancedb_dir_is_not_a_fresh_install(isolated_data_dir):
    """A lilbee that has already stored something must not re-run first-run setup."""
    cfg.lancedb_dir.mkdir(parents=True)
    assert is_fresh_install() is False


def test_empty_refs_are_not_ready_and_say_nothing(isolated_data_dir, caplog):
    """Unconfigured roles are a state, not an error: False with no log noise."""
    import logging

    cfg.chat_model = ""
    cfg.embedding_model = ""
    with caplog.at_level(logging.DEBUG):
        assert chat_ready() is False
        assert embedding_ready() is False
    assert not caplog.records


def test_roles_are_independent(isolated_data_dir):
    """A resolvable chat ref makes chat ready while the embedder stays unconfigured."""
    cfg.chat_model = "owner/chat-GGUF/chat.Q4_K_M.gguf"
    cfg.embedding_model = ""
    with mock.patch(
        "lilbee.providers.engine_params.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert chat_ready() is True
        assert embedding_ready() is False


def test_readiness_ignores_a_fresh_data_dir(isolated_data_dir):
    """A fresh lilbee whose chat model resolves can still chat.

    Folding the fresh-data-dir check into readiness would lock chat behind an
    ingest that cannot be run from anywhere but chat.
    """
    assert not cfg.lancedb_dir.exists()
    cfg.chat_model = "owner/chat-GGUF/chat.Q4_K_M.gguf"
    with mock.patch(
        "lilbee.providers.engine_params.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert chat_ready() is True


def test_embedding_ready_when_the_native_ref_resolves(isolated_data_dir):
    """A resolvable native embedding ref means search has an engine."""
    cfg.lancedb_dir.mkdir(parents=True)
    # Explicit native ref so the check is deterministic regardless of the
    # developer's loaded config.toml (which may hold remote refs).
    cfg.embedding_model = "owner/embed-GGUF/embed.Q8_0.gguf"
    with mock.patch(
        "lilbee.providers.engine_params.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert embedding_ready() is True


def test_not_ready_when_the_native_ref_is_missing(isolated_data_dir):
    """An unresolvable native ref leaves the role with no engine."""
    from lilbee.providers.base import ProviderError

    cfg.lancedb_dir.mkdir(parents=True)
    cfg.chat_model = "owner/chat-GGUF/chat.Q4_K_M.gguf"
    with mock.patch(
        "lilbee.providers.engine_params.resolve_model_path",
        side_effect=ProviderError("no such model", provider="llama-cpp"),
    ):
        assert chat_ready() is False


def test_readiness_skips_the_native_probe_for_usable_remote_models(isolated_data_dir):
    """ollama/ and API-prefixed models bypass the llama-cpp registry check.

    The native resolver only knows about GGUFs, so a usable remote ref
    must not be sent through resolve_model_path. Usability is decided by
    ``validate_persisted_model`` (litellm present, server live, key set).
    """
    from lilbee.modelhub.model_manager import ValidationResult

    cfg.lancedb_dir.mkdir(parents=True)
    cfg.chat_model = "ollama/qwen3:0.6b"
    with (
        mock.patch(
            "lilbee.modelhub.model_manager.validate_persisted_model",
            return_value=ValidationResult.OK,
        ),
        mock.patch("lilbee.providers.engine_params.resolve_model_path") as resolve,
    ):
        assert chat_ready() is True
        resolve.assert_not_called()


def test_not_ready_when_a_remote_ref_is_unusable(isolated_data_dir):
    """An unusable remote ref (litellm missing, server down, no key) is not ready.

    Regression: this used to be skipped on the assumption that remote refs
    always resolve at call time, leaving a user with an unservable
    ``ollama/`` model stuck in a broken app.
    """
    from lilbee.modelhub.model_manager import ValidationResult

    cfg.lancedb_dir.mkdir(parents=True)
    cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
    with mock.patch(
        "lilbee.modelhub.model_manager.validate_persisted_model",
        return_value=ValidationResult.UNKNOWN,
    ):
        assert embedding_ready() is False


def test_a_file_at_the_lancedb_path_is_a_fresh_install(isolated_data_dir):
    """A stray file at the lancedb path is not a real data directory."""
    cfg.lancedb_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.write_text("not a directory")
    assert cfg.lancedb_dir.exists()
    assert not cfg.lancedb_dir.is_dir()
    assert is_fresh_install() is True


@pytest.fixture
def mock_services():
    from lilbee.app.services import set_services

    svc = mock.MagicMock()
    svc.provider.list_models.return_value = []
    svc.searcher._embedder.embedding_available.return_value = True
    set_services(svc)
    try:
        yield svc
    finally:
        set_services(None)


async def test_chat_screen_cached_across_navigation(isolated_data_dir, mock_services):
    """Navigating away from Chat and back reuses the same instance.
    ChatScreen is installed via install_screen, so on_mount (and therefore
    the setup check) runs only on first mount, not on every revisit."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.chat import ChatScreen

    cfg.lancedb_dir.mkdir(parents=True)

    with (
        mock.patch(
            "lilbee.cli.tui.app.models_ready",
            return_value=True,
        ),
        mock.patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=True,
        ),
    ):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await await_chat(app, pilot)
            await pilot.pause()
            chat = app.screen
            assert isinstance(chat, ChatScreen)

            app.switch_view("Catalog")
            await pilot.pause()
            assert isinstance(app.screen, CatalogScreen)

            app.switch_view("Chat")
            await pilot.pause()
            assert app.screen is chat  # same instance, not a new one
