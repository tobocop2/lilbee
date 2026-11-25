"""One installed model, one role label, on every surface that reads it.

A pull stores ``task="chat"`` for a vision model such as SmolVLM, because the
pull cannot tell the two apart. The role check, the model list, the model
detail, the chat listing and the vision route all must read the same label.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import install_fake_model, make_mock_services
from lilbee.app.models import ManifestData, ModelEntry, installed_chat_model_refs
from lilbee.app.services import set_services
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelRegistry
from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment
from lilbee.providers.fleet.engines.base import EngineName

_ENGINES = "lilbee.providers.fleet.engines.choice.installed_engines"

_REPO = "ggml-org/SmolVLM-256M-Instruct-GGUF"
_FILE = "SmolVLM-256M-Instruct-Q8_0.gguf"


@pytest.fixture
def smolvlm_stored_as_chat() -> str:
    """SmolVLM as a pull records it: the chat label, with the name that says vision."""
    return install_fake_model(_REPO, _FILE, "chat")


def _manifest(ref: str):
    manifest = ModelRegistry(cfg.models_dir).get_manifest(ref)
    assert manifest is not None
    return manifest


@pytest.fixture
def unknown_task_model() -> str:
    """A model whose manifest names a task this lilbee does not know (a newer writer)."""
    return install_fake_model("other/Plain-Chat-GGUF", "plain-Q4_K_M.gguf", "bogus")


class TestUnknownTaskStaysListed:
    """An unknown task string loads as stored, so the user can still see and remove the model."""

    def test_registry_lists_it_with_the_stored_string(self, unknown_task_model: str) -> None:
        manifests = ModelRegistry(cfg.models_dir).list_installed()
        assert [(m.ref, m.task) for m in manifests] == [(unknown_task_model, "bogus")]

    def test_model_list_table_shows_it(self, unknown_task_model: str) -> None:
        from typer.testing import CliRunner

        from lilbee.cli import app

        with patch("lilbee.app.models._collect_backend_entries", return_value=[]):
            result = CliRunner().invoke(app, ["model", "list"])
        assert result.exit_code == 0, result.output
        assert unknown_task_model in result.output

    def test_model_list_json_shows_it(self, unknown_task_model: str) -> None:
        import json

        from typer.testing import CliRunner

        from lilbee.cli import app

        with patch("lilbee.app.models._collect_backend_entries", return_value=[]):
            result = CliRunner().invoke(app, ["--json", "model", "list"])
        assert result.exit_code == 0, result.output
        assert unknown_task_model in [m["name"] for m in json.loads(result.output)["models"]]


class TestManifestLabel:
    def test_registry_reports_vision(self, smolvlm_stored_as_chat: str) -> None:
        assert _manifest(smolvlm_stored_as_chat).task == ModelTask.VISION

    def test_model_list_reports_vision(self, smolvlm_stored_as_chat: str) -> None:
        entry = ModelEntry.from_native(smolvlm_stored_as_chat, _manifest(smolvlm_stored_as_chat))
        assert entry.task == ModelTask.VISION

    def test_model_detail_reports_vision(self, smolvlm_stored_as_chat: str) -> None:
        data = ManifestData.from_manifest(_manifest(smolvlm_stored_as_chat))
        assert data.task == ModelTask.VISION

    def test_chat_listing_leaves_it_out(self, smolvlm_stored_as_chat: str) -> None:
        set_services(make_mock_services(registry=ModelRegistry(cfg.models_dir)))
        try:
            with patch(_ENGINES, return_value={EngineName.LLAMA: frozenset({ModelFormat.GGUF})}):
                assert smolvlm_stored_as_chat not in installed_chat_model_refs()
        finally:
            set_services(None)

    def test_vision_role_accepts_it(self, smolvlm_stored_as_chat: str) -> None:
        got = validate_model_task_assignment(
            "vision_model", smolvlm_stored_as_chat, allow_bypass=False
        )
        assert got == smolvlm_stored_as_chat

    def test_chat_role_refuses_it_with_the_same_label(self, smolvlm_stored_as_chat: str) -> None:
        with pytest.raises(TaskMismatchError) as caught:
            validate_model_task_assignment("chat_model", smolvlm_stored_as_chat, allow_bypass=False)
        assert caught.value.entry_task == ModelTask.VISION

    def test_vision_route_check_accepts_it(
        self, smolvlm_stored_as_chat: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.server.handlers import models as handlers

        monkeypatch.setattr(handlers, "_require_model_available", lambda model: model)
        got = handlers._require_model_for_task(smolvlm_stored_as_chat, ModelTask.VISION)
        assert got == smolvlm_stored_as_chat
