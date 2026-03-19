import asyncio
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.config import cfg

runner = CliRunner()


def _close_coro(coro, *_args, **_kwargs):
    """Consume and close the coroutine so Python doesn't warn about it."""
    coro.close()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
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
