"""The mlx engine runs only Qwen3-Embedding for embed and Qwen3-Reranker for rerank.

Every place that sets a role refuses another `#mlx` model with a plain message that names the
supported family: the config load, the config assignment, the settings update (CLI, TUI, MCP and
HTTP config writes), the model routes and `lilbee use-embedder`.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from conftest import clean_env
from lilbee.app import models as app_models
from lilbee.app import settings as appset
from lilbee.app.services import get_services
from lilbee.catalog.types import ModelTask
from lilbee.cli import app
from lilbee.cli.tui.app import LilbeeApp
from lilbee.core.config import Config, RefusedVariableError, cfg
from lilbee.core.config.model import _build_cfg, mlx_role_refusal
from lilbee.mcp_server import settings_set
from lilbee.modelhub.role_validator import validate_model_task_assignment
from lilbee.providers.fleet.engines.mlx import role_refusal
from lilbee.providers.roles import WorkerRole
from lilbee.server.handlers import models as handlers
from tests._cli_text import plain_cli_text as _flat

EMBED_OK = "mlx-community/Qwen3-Embedding-0.6B-8bit#mlx"
RERANK_OK = "mlx-community/Qwen3-Reranker-0.6B-4bit#mlx"
BERT = "mlx-community/all-MiniLM-L6-v2-4bit#mlx"
CHAT = "mlx-community/Qwen3-0.6B-4bit#mlx"
EMBED_WORDS = "Qwen3-Embedding"
RERANK_WORDS = "Qwen3-Reranker"

runner = CliRunner()


class TestRoleRefusal:
    @pytest.mark.parametrize(
        ("role", "ref", "family"),
        [
            (WorkerRole.EMBED, BERT, EMBED_WORDS),
            (WorkerRole.EMBED, CHAT, EMBED_WORDS),
            (WorkerRole.EMBED, RERANK_OK, EMBED_WORDS),
            (WorkerRole.RERANK, CHAT, RERANK_WORDS),
            (WorkerRole.RERANK, EMBED_OK, RERANK_WORDS),
            (WorkerRole.EMBED, "Qwen3-Embedding-fans/bge-m3-mlx#mlx", EMBED_WORDS),
            (WorkerRole.RERANK, "Qwen3-Reranker-fans/bge-m3-mlx#mlx", RERANK_WORDS),
        ],
    )
    def test_an_unsupported_model_is_refused_and_the_family_is_named(
        self, role: WorkerRole, ref: str, family: str
    ) -> None:
        refusal = role_refusal(role, ref)
        assert refusal is not None
        assert family in refusal
        assert ref.removesuffix("#mlx") in refusal
        assert "for example" in refusal

    @pytest.mark.parametrize(
        ("role", "ref"),
        [
            (WorkerRole.EMBED, EMBED_OK),
            (WorkerRole.EMBED, "someone/qwen3-embedding-4b-4bit#mlx"),
            (WorkerRole.RERANK, RERANK_OK),
        ],
    )
    def test_a_supported_model_is_accepted(self, role: WorkerRole, ref: str) -> None:
        assert role_refusal(role, ref) is None

    @pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.VISION])
    def test_the_roles_that_the_embed_server_does_not_run_take_any_mlx_model(
        self, role: WorkerRole
    ) -> None:
        assert role_refusal(role, BERT) is None

    @pytest.mark.parametrize(
        "ref",
        ["org/Some-Model-GGUF/m-Q4_K_M.gguf", "org/Some-Model", "openai/some-model#mlx", ""],
    )
    def test_a_ref_that_is_not_a_native_mlx_model_is_not_the_business_of_the_mlx_engine(
        self, ref: str
    ) -> None:
        assert role_refusal(WorkerRole.EMBED, ref) is None

    def test_a_blank_ref_is_not_refused_for_any_field(self) -> None:
        assert mlx_role_refusal("embedding_model", "") is None
        assert mlx_role_refusal("reranker_model", "") is None


class TestConfigLoad:
    def test_a_config_file_with_an_unsupported_embedder_warns_and_takes_the_default(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.toml").write_text(f'embedding_model = "{BERT}"\n', encoding="utf-8")
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            built, warnings = _build_cfg()
        assert built.embedding_model != BERT
        assert len(warnings) == 1
        assert EMBED_WORDS in warnings[0]
        assert warnings[0].endswith("embedding_model uses its default")

    def test_a_config_file_with_an_unsupported_reranker_warns_and_takes_the_default(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "config.toml").write_text(f'reranker_model = "{CHAT}"\n', encoding="utf-8")
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            built, warnings = _build_cfg()
        assert built.reranker_model != CHAT
        assert RERANK_WORDS in warnings[0]

    def test_a_config_file_with_a_supported_pair_loads(self, tmp_path: Path) -> None:
        (tmp_path / "config.toml").write_text(
            f'embedding_model = "{EMBED_OK}"\nreranker_model = "{RERANK_OK}"\n', encoding="utf-8"
        )
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            built, warnings = _build_cfg()
        assert (built.embedding_model, built.reranker_model, warnings) == (
            EMBED_OK,
            RERANK_OK,
            (),
        )

    def test_a_variable_with_an_unsupported_embedder_stops_the_load(self, tmp_path: Path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_EMBEDDING_MODEL": BERT}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(RefusedVariableError, match=EMBED_WORDS),
        ):
            Config()


class TestConfigAssignment:
    @pytest.mark.parametrize(
        ("field", "ref", "family"),
        [
            ("embedding_model", BERT, EMBED_WORDS),
            ("embedding_model", RERANK_OK, EMBED_WORDS),
            ("reranker_model", CHAT, RERANK_WORDS),
            ("reranker_model", EMBED_OK, RERANK_WORDS),
        ],
    )
    def test_an_unsupported_model_is_refused_on_assignment(
        self, field: str, ref: str, family: str
    ) -> None:
        before = getattr(cfg, field)
        with pytest.raises(ValidationError, match=family):
            setattr(cfg, field, ref)
        assert getattr(cfg, field) == before

    def test_the_chat_role_takes_any_mlx_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "chat_model", "")
        cfg.chat_model = CHAT
        assert cfg.chat_model == CHAT


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    return tmp_path


class TestSettingsUpdate:
    """The one write path of the CLI settings, the TUI, the MCP server and PATCH /api/config."""

    @pytest.mark.parametrize(
        ("key", "ref", "family"),
        [
            ("embedding_model", BERT, EMBED_WORDS),
            ("reranker_model", CHAT, RERANK_WORDS),
        ],
    )
    def test_an_unsupported_model_is_refused_in_plain_words(
        self, root: Path, key: str, ref: str, family: str
    ) -> None:
        before = getattr(cfg, key)
        with pytest.raises(ValueError, match=family) as refused:
            appset.apply_settings_update({key: ref})
        assert not isinstance(refused.value, ValidationError)
        assert getattr(cfg, key) == before
        assert not (root / "config.toml").exists()

    def test_the_mcp_tool_answers_the_refusal(self, root: Path) -> None:
        result = settings_set({"embedding_model": BERT})
        assert EMBED_WORDS in result["error"]

    def test_the_tui_toasts_the_refusal_and_keeps_the_role(self, root: Path) -> None:
        toasts: list[str] = []
        host = SimpleNamespace(
            _reject_if_downloading=lambda value: False,
            notify=lambda message, severity: toasts.append(message),
        )
        set_active_model: Any = LilbeeApp.set_active_model
        before = cfg.reranker_model
        assert set_active_model(host, "reranker_model", EMBED_OK) is None
        assert RERANK_WORDS in toasts[0]
        assert cfg.reranker_model == before


class TestModelRoutes:
    @pytest.fixture
    def listed(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        refs = [BERT, CHAT, EMBED_OK, RERANK_OK]
        services = MagicMock()
        services.provider.list_models.return_value = refs
        monkeypatch.setattr(handlers, "get_services", lambda: services)
        return refs

    @pytest.mark.parametrize(
        ("task", "ref", "family"),
        [
            (ModelTask.EMBEDDING, BERT, EMBED_WORDS),
            (ModelTask.RERANK, CHAT, RERANK_WORDS),
        ],
    )
    def test_the_http_route_refuses_an_unsupported_model(
        self, listed: list[str], task: ModelTask, ref: str, family: str
    ) -> None:
        with pytest.raises(ValueError, match=family):
            handlers._require_model_for_task(ref, task)

    async def test_set_embedding_model_refuses_before_it_writes(self, listed: list[str]) -> None:
        before = cfg.embedding_model
        with pytest.raises(ValueError, match=EMBED_WORDS):
            await handlers.set_embedding_model(BERT)
        assert cfg.embedding_model == before

    async def test_set_reranker_model_refuses_before_it_writes(self, listed: list[str]) -> None:
        before = cfg.reranker_model
        with pytest.raises(ValueError, match=RERANK_WORDS):
            await handlers.set_reranker_model(EMBED_OK)
        assert cfg.reranker_model == before

    @pytest.mark.parametrize(
        ("field", "ref", "family"),
        [("embedding_model", BERT, EMBED_WORDS), ("reranker_model", CHAT, RERANK_WORDS)],
    )
    def test_the_role_check_cannot_be_bypassed_by_the_test_switch(
        self, monkeypatch: pytest.MonkeyPatch, field: str, ref: str, family: str
    ) -> None:
        monkeypatch.setenv("LILBEE_SKIP_MODEL_TASK_VALIDATION", "1")
        with pytest.raises(ValueError, match=family):
            validate_model_task_assignment(
                field, ref, hf_client_for=lambda: get_services().hf_client
            )

    def test_the_cli_refuses_an_unsupported_reranker(
        self, listed: list[str], tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["model", "use", "rerank", CHAT, "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert RERANK_WORDS in _flat(result.output)


class TestUseEmbedder:
    def test_the_command_refuses_an_unsupported_embedder_before_it_downloads(
        self, tmp_path: Path
    ) -> None:
        with patch.object(app_models, "pull_model_data") as pull:
            result = runner.invoke(app, ["use-embedder", BERT, "--data-dir", str(tmp_path)])
        assert result.exit_code == 1
        assert EMBED_WORDS in _flat(result.output)
        pull.assert_not_called()

    def test_the_function_refuses_before_it_reads_the_installed_models(self) -> None:
        with (
            patch.object(app_models, "get_services") as services,
            pytest.raises(ValueError, match=EMBED_WORDS),
        ):
            app_models.adopt_embedder(BERT)
        services.assert_not_called()
