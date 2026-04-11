"""Tests for ChatScreen._needs_setup fresh-install detection.

Verifies that the setup wizard shows even when the configured models are
resolvable globally (Ollama, HuggingFace cache) as long as the current data
directory has not been initialized yet. Prevents a regression where users
with common models already cached on their machine would silently skip the
onboarding wizard.
"""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.config import cfg


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


def _make_screen() -> ChatScreen:
    """Construct a ChatScreen without mounting it in an App.

    ``_needs_setup`` only reads module-level config, so bypassing ``__init__``
    (which would otherwise require an App context) is safe and keeps these
    tests narrowly scoped to the setup-detection logic.
    """
    return ChatScreen.__new__(ChatScreen)


def test_needs_setup_true_when_lancedb_dir_missing(isolated_data_dir):
    """A fresh LILBEE_DATA (no lancedb dir) must always show the wizard,
    even when the configured models are globally resolvable. The check must
    also short-circuit model resolution because the answer is already known.
    """
    assert not cfg.lancedb_dir.exists()
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ) as resolve:
        assert _make_screen()._needs_setup() is True
        resolve.assert_not_called()


def test_needs_setup_false_when_initialized_and_models_resolve(isolated_data_dir):
    """After first sync (lancedb dir exists) and both models resolve, the
    user should skip the wizard and go straight to chat.
    """
    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is False


def test_needs_setup_true_when_initialized_but_model_missing(isolated_data_dir):
    """An initialized data directory but a missing chat or embedding model
    should still show the wizard so the user can pick a valid one.
    """
    cfg.lancedb_dir.mkdir(parents=True)
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        side_effect=FileNotFoundError("no such model"),
    ):
        assert _make_screen()._needs_setup() is True


def test_needs_setup_true_when_lancedb_path_is_a_file(isolated_data_dir):
    """A stray file at the lancedb path is not a real data directory,
    so the wizard must still show. This exercises the ``is_dir`` check.
    """
    cfg.lancedb_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.lancedb_dir.write_text("not a directory")
    assert cfg.lancedb_dir.exists()
    assert not cfg.lancedb_dir.is_dir()
    with mock.patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/some/resolved/path",
    ):
        assert _make_screen()._needs_setup() is True
