"""Messages that reach a REST or MCP caller name the REST route beside the CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.crawler.messages import CHROMIUM_MISSING_MESSAGE
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind, WorkerRole, require_role_ref
from lilbee.providers.fleet import first_use
from lilbee.providers.fleet.client import _upstream_death_message
from lilbee.providers.fleet.engine_install import engine_install_hint
from lilbee.providers.fleet.engine_store import EngineFolderError
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.mistral import _DAMAGED_MODEL
from lilbee.server.chat_dispatch.dispatch import ModelNotFoundError

PULL = "POST /api/models/pull"
REMOVE = "DELETE /api/models/{model}"
LIST = "GET /api/models/installed"
ENGINE_INSTALL = "POST /api/engines/install"
CRAWLER_SETUP = "POST /setup/crawler"


def test_an_empty_model_role_names_the_pull_route() -> None:
    with pytest.raises(ProviderError) as refused:
        engine_params.resolve_model_path("")
    assert "lilbee model pull" in str(refused.value)
    assert PULL in str(refused.value)


def test_a_model_that_is_not_installed_names_the_pull_route() -> None:
    registry = MagicMock()
    registry.resolve.side_effect = KeyError("absent")
    with pytest.raises(ProviderError) as refused:
        engine_params.resolve_model_path("org/repo/model.gguf", registry)
    assert "lilbee model pull org/repo/model.gguf" in str(refused.value)
    assert PULL in str(refused.value)


def test_an_unset_role_names_the_pull_route() -> None:
    with pytest.raises(ProviderError) as refused:
        require_role_ref("", WorkerRole.CHAT)
    assert "lilbee model pull" in str(refused.value)
    assert PULL in str(refused.value)


def test_a_missing_model_in_a_chat_request_names_the_list_and_pull_routes() -> None:
    text = str(ModelNotFoundError("org/repo/model.gguf"))
    assert "lilbee model list" in text
    assert LIST in text
    assert "lilbee model pull org/repo/model.gguf" in text
    assert PULL in text


def test_a_damaged_model_file_names_the_remove_and_pull_routes() -> None:
    text = _upstream_death_message(ProviderErrorKind.MODEL_FILE_BROKEN, 500, "o/r/m.gguf", None)
    assert "lilbee model rm o/r/m.gguf" in text
    assert REMOVE in text
    assert PULL in text


def test_a_damaged_model_file_without_a_name_names_the_routes() -> None:
    text = _upstream_death_message(ProviderErrorKind.MODEL_FILE_BROKEN, 500, None, None)
    assert LIST in text
    assert REMOVE in text
    assert PULL in text


def test_the_mistral_damaged_model_advice_names_the_routes() -> None:
    assert LIST in _DAMAGED_MODEL
    assert REMOVE in _DAMAGED_MODEL
    assert PULL in _DAMAGED_MODEL


def test_the_engine_install_hint_names_the_install_route(monkeypatch) -> None:
    monkeypatch.setattr(first_use, "pack_lock", lambda: (None, "cell"))
    hint = engine_install_hint(bundled=False)
    assert "lilbee engine install" in hint
    assert ENGINE_INSTALL in hint


def test_a_damaged_engine_folder_names_the_install_route(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(choice, "install_offered", lambda *_a: True)
    text = str(EngineFolderError(tmp_path / "tei", "tei"))
    assert "lilbee engine install tei" in text
    assert ENGINE_INSTALL in text


def test_the_engine_list_hint_names_the_list_route() -> None:
    assert "lilbee engine list" in choice.LIST_HINT
    assert "GET /api/engines" in choice.LIST_HINT


def test_the_missing_browser_message_names_the_crawler_setup_route() -> None:
    assert "lilbee setup crawler" in CHROMIUM_MISSING_MESSAGE
    assert CRAWLER_SETUP in CHROMIUM_MISSING_MESSAGE
