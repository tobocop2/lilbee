"""`lilbee agent-config` writes a config block and never warms the inference engine."""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.app import services as services_mod
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()

# The claude block advertises no models, so it never builds the services container.
_CLIENTS = ["opencode", "hermes", "litellm"]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch) -> Iterator[None]:
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.worker_pool_eager_start = True
    server_json_path().write_text(json.dumps({"token": "tok"}), encoding="utf-8")
    (cfg.data_dir / "server.port").write_text("8765", encoding="utf-8")
    yield
    services_mod.reset_services()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.mark.parametrize("client", _CLIENTS)
def test_agent_config_does_not_warm_the_engine(client: str) -> None:
    """The command builds the real services container and the fleet stays cold."""
    provider = MagicMock()
    with (
        patch("lilbee.providers.factory.create_provider", return_value=provider),
        patch("lilbee.cli.commands.agent_config.client_chat_ctx", return_value=None),
    ):
        result = runner.invoke(app, ["agent-config", client])

    assert result.exit_code == 0, result.stderr
    provider.warm_up_pool.assert_not_called()
