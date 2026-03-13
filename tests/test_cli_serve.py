from dataclasses import fields, replace
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.config import cfg

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    snapshot = replace(cfg)
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


class TestServeCommand:
    @mock.patch("uvicorn.run")
    @mock.patch("lilbee.server.create_app")
    def test_default_host_port(self, mock_create_app, mock_uvicorn_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_uvicorn_run.assert_called_once_with("fake_app", host="127.0.0.1", port=7433)

    @mock.patch("uvicorn.run")
    @mock.patch("lilbee.server.create_app")
    def test_custom_host_port(self, mock_create_app, mock_uvicorn_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8080"])
        assert result.exit_code == 0
        mock_uvicorn_run.assert_called_once_with("fake_app", host="0.0.0.0", port=8080)

    @mock.patch("uvicorn.run")
    @mock.patch("lilbee.server.create_app")
    def test_short_flags(self, mock_create_app, mock_uvicorn_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve", "-H", "0.0.0.0", "-p", "9000"])
        assert result.exit_code == 0
        mock_uvicorn_run.assert_called_once_with("fake_app", host="0.0.0.0", port=9000)
