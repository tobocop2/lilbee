import asyncio
import json
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.config import cfg
from lilbee.server.auth import server_json_path

runner = CliRunner()


def _close_coro(coro, *_args, **_kwargs):
    """Consume and close the coroutine so Python doesn't warn about it."""
    coro.close()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    # CI sets LILBEE_DATA at workflow level; clear it so the token command's
    # apply_overrides() does not clobber cfg.data_dir during invocation.
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestTokenCommand:
    def test_prints_token_when_server_running(self):
        path = server_json_path()
        path.write_text(json.dumps({"token": "test-secret-token"}))

        result = runner.invoke(app, ["token"])
        assert result.exit_code == 0
        assert "test-secret-token" in result.output

    def test_exits_1_when_no_server(self):
        result = runner.invoke(app, ["token"])
        assert result.exit_code == 1
        assert "No running server found" in result.output

    def test_json_mode_prints_token(self):
        path = server_json_path()
        path.write_text(json.dumps({"token": "json-token-val"}))

        result = runner.invoke(app, ["--json", "token"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["token"] == "json-token-val"

    def test_json_mode_exits_1_when_no_server(self):
        result = runner.invoke(app, ["--json", "token"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data

    def test_corrupted_server_json_exits_1(self):
        path = server_json_path()
        path.write_text("not valid json{{{")

        result = runner.invoke(app, ["token"])
        assert result.exit_code == 1
        assert "Could not read server.json" in result.output

    def test_corrupted_server_json_json_mode(self):
        path = server_json_path()
        path.write_text("not valid json{{{")

        result = runner.invoke(app, ["--json", "token"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data
        assert "Could not read server.json" in data["error"]

    def test_missing_token_key_returns_empty(self):
        path = server_json_path()
        path.write_text(json.dumps({"other": "field"}))

        result = runner.invoke(app, ["token"])
        assert result.exit_code == 0


class TestServeCommand:
    @mock.patch("lilbee.cli.commands.asyncio.run", side_effect=_close_coro)
    @mock.patch("lilbee.server.create_app")
    def test_default_host_port(self, mock_create_app, mock_asyncio_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @mock.patch("lilbee.cli.commands.asyncio.run", side_effect=_close_coro)
    @mock.patch("lilbee.server.create_app")
    def test_custom_host_port(self, mock_create_app, mock_asyncio_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8080"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @mock.patch("lilbee.cli.commands.asyncio.run", side_effect=_close_coro)
    @mock.patch("lilbee.server.create_app")
    def test_short_flags(self, mock_create_app, mock_asyncio_run):
        mock_create_app.return_value = "fake_app"
        result = runner.invoke(app, ["serve", "-H", "0.0.0.0", "-p", "9000"])
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()


class TestPortFile:
    def test_port_file_path(self):
        from lilbee.cli.commands import _port_file

        assert _port_file() == cfg.data_dir / "server.port"


class TestRunServer:
    def test_writes_and_cleans_port_file(self):
        from lilbee.cli.commands import _run_server

        sock = mock.MagicMock()
        sock.getsockname.return_value = ("127.0.0.1", 54321)

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = [mock.MagicMock(sockets=[sock])]
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        port_path = cfg.data_dir / "server.port"
        # Port file should be cleaned up after shutdown
        assert not port_path.exists()
        fake_server_obj.startup.assert_awaited_once()
        fake_server_obj.main_loop.assert_awaited_once()
        fake_server_obj.shutdown.assert_awaited_once()

    def test_writes_correct_port(self):
        from lilbee.cli.commands import _run_server

        sock = mock.MagicMock()
        sock.getsockname.return_value = ("127.0.0.1", 9999)

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = [mock.MagicMock(sockets=[sock])]
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        written_port = None

        async def capture_port() -> None:
            nonlocal written_port
            port_path = cfg.data_dir / "server.port"
            if port_path.exists():
                written_port = port_path.read_text()

        fake_server_obj.main_loop = capture_port
        fake_config = mock.MagicMock()

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        assert written_port == "9999"

    def test_cleans_port_file_on_error(self):
        from lilbee.cli.commands import _run_server

        sock = mock.MagicMock()
        sock.getsockname.return_value = ("127.0.0.1", 12345)

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = [mock.MagicMock(sockets=[sock])]
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock(side_effect=RuntimeError("boom"))
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        assert not (cfg.data_dir / "server.port").exists()

    def test_no_servers_skips_port_file(self):
        from lilbee.cli.commands import _run_server

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = []
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        assert not (cfg.data_dir / "server.port").exists()

    def test_loads_config_when_not_loaded(self):
        from lilbee.cli.commands import _run_server

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = []
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()
        fake_config.loaded = False

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        fake_config.load.assert_called_once()

    def test_creates_data_dir_for_port_file(self, tmp_path):
        from lilbee.cli.commands import _run_server

        cfg.data_dir = tmp_path / "nonexistent" / "data"

        sock = mock.MagicMock()
        sock.getsockname.return_value = ("127.0.0.1", 8080)

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = [mock.MagicMock(sockets=[sock])]
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        # Dir was created, port file cleaned up after shutdown
        assert cfg.data_dir.exists()
        assert not (cfg.data_dir / "server.port").exists()

    @mock.patch("atexit.register")
    def test_registers_atexit_cleanup(self, mock_atexit):
        """Port file cleanup is registered via atexit for SIGTERM resilience."""
        from lilbee.cli.commands import _run_server

        sock = mock.MagicMock()
        sock.getsockname.return_value = ("127.0.0.1", 11111)

        fake_server_obj = mock.MagicMock()
        fake_server_obj.servers = [mock.MagicMock(sockets=[sock])]
        fake_server_obj.startup = mock.AsyncMock()
        fake_server_obj.main_loop = mock.AsyncMock()
        fake_server_obj.shutdown = mock.AsyncMock()

        fake_config = mock.MagicMock()

        asyncio.run(_run_server(fake_server_obj, fake_config, "127.0.0.1"))

        mock_atexit.assert_called_once()
        cleanup_fn = mock_atexit.call_args[0][0]
        # Write a port file and verify the cleanup function removes it
        port_path = cfg.data_dir / "server.port"
        port_path.write_text("11111")
        cleanup_fn()
        assert not port_path.exists()
