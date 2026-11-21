"""Tests for `lilbee model use`, which sets a role's model through the HTTP routes' service."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.core.config.enums import OcrMode
from lilbee.modelhub.role_validator import TaskMismatchError
from tests._cli_text import plain_cli_text as _flat

runner = CliRunner()

_CHAT_REF = "tiny/Tiny-1B-GGUF/tiny-1b-Q4_K_M.gguf"
_RERANK_REF = "rerank/bge-reranker-test-GGUF/bge-reranker-Q4_K_M.gguf"
_VISION_REF = "vision/Test-VL-GGUF/test-vl-Q4_K_M.gguf"
_MISSING_REF = "org/Nonexistent-GGUF/none.gguf"

_SERVICES = "lilbee.server.handlers.models.get_services"
_VALIDATE = "lilbee.server.handlers.models.validate_model_task_assignment"


def _use(role: str, ref: str, *flags: str):
    return runner.invoke(app, [*flags, "model", "use", role, ref])


class TestModelUseSetsEachRole:
    @patch(_SERVICES)
    def test_chat(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        result = _use("chat", _CHAT_REF, "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert cfg.chat_model == _CHAT_REF
        assert _CHAT_REF in result.output

    @patch(_SERVICES)
    def test_rerank(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_RERANK_REF]
        result = _use("rerank", _RERANK_REF, "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert cfg.reranker_model == _RERANK_REF

    @patch(_SERVICES)
    def test_vision(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        result = _use("vision", _VISION_REF, "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert cfg.vision_model == _VISION_REF


class TestModelUseAppliesTheServiceChecks:
    @pytest.mark.parametrize(
        ("role", "field"),
        [("chat", "chat_model"), ("rerank", "reranker_model"), ("vision", "vision_model")],
    )
    @patch(_SERVICES)
    def test_refuses_a_model_that_is_not_installed(self, mock_svc, role, field, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        before = getattr(cfg, field)
        result = _use(role, _MISSING_REF, "--data-dir", str(tmp_path))
        assert result.exit_code == 1
        assert "not available" in result.output
        assert getattr(cfg, field) == before

    @patch(_SERVICES)
    def test_refuses_an_empty_chat_ref(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        before = cfg.chat_model
        result = _use("chat", "", "--data-dir", str(tmp_path))
        assert result.exit_code == 1
        assert "not available" in result.output
        assert cfg.chat_model == before

    @pytest.mark.parametrize(
        ("role", "ref", "entry_task", "command"),
        [
            ("chat", _CHAT_REF, ModelTask.EMBEDDING, "lilbee use-embedder <ref>"),
            ("rerank", _RERANK_REF, ModelTask.CHAT, "lilbee model use chat <ref>"),
            ("vision", _VISION_REF, ModelTask.RERANK, "lilbee model use rerank <ref>"),
        ],
    )
    @patch(_VALIDATE)
    @patch(_SERVICES)
    def test_task_mismatch_names_the_command_for_the_model_task(
        self, mock_svc, mock_validate, role, ref, entry_task, command, tmp_path
    ):
        mock_svc.return_value.provider.list_models.return_value = [ref]
        mock_validate.side_effect = TaskMismatchError(ref, entry_task, ModelTask(role))
        result = _use(role, ref, "--data-dir", str(tmp_path))
        assert result.exit_code == 1
        flat = _flat(result.output)
        assert f"is a {entry_task} model, not {role}" in flat
        assert f"Set it with '{command}' instead." in flat

    @patch(_SERVICES)
    def test_empty_ref_clears_the_vision_model(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        cfg.vision_model = _VISION_REF
        result = _use("vision", "", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert cfg.vision_model == ""
        assert "vision model is now off." in _flat(result.output)

    @patch(_SERVICES)
    def test_vision_with_ocr_off_prints_the_service_warning(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        (tmp_path / "config.toml").write_text(f'ocr = "{OcrMode.OFF}"\n', encoding="utf-8")
        result = _use("vision", _VISION_REF, "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert "ocr is off" in result.output

    def test_embed_points_at_use_embedder(self, tmp_path):
        result = _use("embed", "some/ref", "--data-dir", str(tmp_path))
        assert result.exit_code == 2
        assert "lilbee use-embedder <ref>" in _flat(result.output)

    @patch(_SERVICES)
    def test_json_output_reports_the_model(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        result = _use("chat", _CHAT_REF, "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "model": _CHAT_REF,
            "reindex_required": False,
            "warnings": [],
        }

    @patch(_SERVICES)
    def test_json_refusal_is_an_error_envelope(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = []
        result = _use("chat", _MISSING_REF, "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 1
        assert "not available" in json.loads(result.output)["error"]


@pytest.mark.parametrize(
    ("key", "command"),
    [
        ("chat_model", "lilbee model use chat <ref>"),
        ("reranker_model", "lilbee model use rerank <ref>"),
        ("vision_model", "lilbee model use vision <ref>"),
        ("embedding_model", "lilbee use-embedder <ref>"),
    ],
)
def test_settings_refusal_names_the_command_that_sets_the_role(key, command, tmp_path):
    refused = runner.invoke(
        app, ["settings", "set", key, "some/Model-GGUF/model.gguf", "--data-dir", str(tmp_path)]
    )
    assert refused.exit_code == 1
    assert _flat(refused.output) == (
        f"Error: '{key}' must be set through the dedicated model route "
        f"(on the command line, {command}), not the general settings update."
    )


def test_embed_refusal_and_settings_refusal_name_the_same_command(tmp_path):
    settings = runner.invoke(
        app, ["settings", "set", "embedding_model", "x/Y-GGUF/z.gguf", "--data-dir", str(tmp_path)]
    )
    use = _use("embed", "x/Y-GGUF/z.gguf", "--data-dir", str(tmp_path))
    assert "lilbee use-embedder <ref>" in _flat(settings.output)
    assert "lilbee use-embedder <ref>" in _flat(use.output)


def test_help_lists_every_role_the_command_accepts():
    help_text = runner.invoke(app, ["model", "use", "--help"])
    assert help_text.exit_code == 0
    flat = _flat(help_text.output)
    assert "Set the model for a role" in flat
    assert "embed is refused: use 'lilbee use-embedder <ref>'" in flat


class TestModelUseNamesTheMissingEngine:
    """A model that no installed engine loads is assigned, with the engine notice beside it."""

    _NOTICE = "No installed engine loads the gguf format"

    @pytest.fixture
    def no_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("lilbee.app.engine_choice.installed_engines", dict)

    @pytest.mark.parametrize(
        ("role", "ref", "field"),
        [
            ("chat", _CHAT_REF, "chat_model"),
            ("rerank", _RERANK_REF, "reranker_model"),
            ("vision", _VISION_REF, "vision_model"),
        ],
    )
    @patch(_SERVICES)
    def test_warns_and_still_assigns(self, mock_svc, no_engine, role, ref, field, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [ref]
        result = _use(role, ref, "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert getattr(cfg, field) == ref
        flat = _flat(result.output)
        assert f"{role} model is now {ref}." in flat
        assert f"For {role}: {self._NOTICE}" in flat
        assert "Run lilbee engine list to see the installed engines." in flat

    @patch(_SERVICES)
    def test_json_carries_the_notice_in_warnings(self, mock_svc, no_engine, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        result = _use("chat", _CHAT_REF, "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        (warning,) = json.loads(result.output)["warnings"]
        assert warning.startswith(f"For chat: {self._NOTICE}")

    @patch(_SERVICES)
    def test_no_warning_when_an_engine_loads_the_model(self, mock_svc, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_CHAT_REF]
        result = _use("chat", _CHAT_REF, "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["warnings"] == []

    @patch(_SERVICES)
    def test_clearing_a_role_does_not_warn(self, mock_svc, no_engine, tmp_path):
        mock_svc.return_value.provider.list_models.return_value = [_VISION_REF]
        result = _use("vision", "", "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["warnings"] == []

    @patch(_SERVICES)
    def test_a_remote_model_needs_no_local_engine(self, mock_svc, no_engine, tmp_path):
        remote = "ollama/llama3"
        mock_svc.return_value.provider.list_models.return_value = [remote]
        with patch(_VALIDATE, return_value=remote):
            result = _use("chat", remote, "--json", "--data-dir", str(tmp_path))
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["warnings"] == []
