"""What serve, mcp and the external-model list hand on, and the texts users read on failure."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from lilbee.app.services import ServicesStoppingError
from lilbee.cli import app
from lilbee.cli.commands import servers
from lilbee.core.config import cfg
from lilbee.runtime.lock import SERVER_LOCK_TIMEOUT, LockingUnsupportedError
from lilbee.server import app as server_app
from lilbee.server.handlers import models as model_handlers

runner = CliRunner()

_SERVERS_LOGGER = "lilbee.cli.commands.servers"
_REFUSAL = (
    "Another lilbee server is already running for this data directory. "
    "Stop it or wait for it to exit, then retry."
)


class _Lock:
    """A held server lock."""

    def release(self) -> None:
        """Nothing is held."""


class _Config:
    """Records the arguments uvicorn.Config gets."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


class _Server:
    """Records the config uvicorn.Server gets."""

    def __init__(self, *args: object) -> None:
        self.args = args


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """Run serve with every outside effect replaced by a recorder, and give back the records."""
    for name in ("LILBEE_DATA", servers.SCOPE_ENV, "LILBEE_PARENT_PID"):
        monkeypatch.delenv(name, raising=False)
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    asyncio_logger = logging.getLogger("asyncio")
    saved_levels = (asyncio_logger.level, logging.getLogger().level)
    asyncio_logger.setLevel(logging.WARNING)

    rec = SimpleNamespace(
        overrides=[],
        locks=[],
        run_server=[],
        run_main=[],
        app=object(),
        lock=_Lock(),
        lock_error=None,
        lock_held_elsewhere=False,
        run_error=None,
    )

    def acquire(*args, **kwargs):
        rec.locks.append((args, kwargs))
        if rec.lock_error is not None:
            raise rec.lock_error
        return None if rec.lock_held_elsewhere else rec.lock

    def run_server(*args):
        rec.run_server.append(args)
        return "the-run"

    def run_main(coro):
        rec.run_main.append(coro)
        if rec.run_error is not None:
            raise rec.run_error
        return 0

    monkeypatch.setattr(servers, "_watch_the_parent", lambda: None)
    monkeypatch.setattr(servers, "apply_overrides", lambda **kwargs: rec.overrides.append(kwargs))
    monkeypatch.setattr(servers, "_report_upgrade_as_log_lines", lambda: None)
    monkeypatch.setattr(servers, "setup_server_logging", lambda: None)
    monkeypatch.setattr(servers, "install_health_access_filter", lambda: None)
    monkeypatch.setattr(servers, "stop_on_interrupt", lambda: None)
    monkeypatch.setattr(servers, "_stop_server", lambda *args: None)
    monkeypatch.setattr(servers, "acquire_server_lock", acquire)
    monkeypatch.setattr(servers, "_run_server", run_server)
    monkeypatch.setattr(servers, "run_main", run_main)
    monkeypatch.setattr("uvicorn.Config", _Config)
    monkeypatch.setattr("uvicorn.Server", _Server)
    monkeypatch.setattr("lilbee.server.create_app", lambda: rec.app)
    yield rec
    asyncio_logger.setLevel(saved_levels[0])
    logging.getLogger().setLevel(saved_levels[1])
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_serve_applies_the_data_dir_and_the_global_flag(boot, tmp_path):
    chosen = tmp_path / "chosen"

    result = runner.invoke(app, ["serve", "--data-dir", str(chosen), "--global"])

    assert result.exit_code == 0
    assert boot.overrides == [{"data_dir": chosen, "use_global": True}]


def test_serve_without_flags_applies_no_data_dir_and_no_global(boot):
    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0
    assert boot.overrides == [{"data_dir": None, "use_global": False}]


def test_serve_takes_the_data_dir_lock_with_the_lock_timeout(boot):
    runner.invoke(app, ["serve"])

    assert boot.locks == [((cfg.data_dir,), {"timeout": SERVER_LOCK_TIMEOUT})]


def test_serve_builds_uvicorn_from_the_app_the_host_and_the_port(boot):
    result = runner.invoke(app, ["serve", "--host", "127.0.0.9", "--port", "4321"])

    assert result.exit_code == 0
    assert len(boot.run_server) == 1
    server, config, host = boot.run_server[0]
    assert isinstance(config, _Config)
    assert config.args == (boot.app,)
    assert config.kwargs == {
        "host": "127.0.0.9",
        "port": 4321,
        "timeout_graceful_shutdown": servers._GRACEFUL_SHUTDOWN_S,
    }
    assert isinstance(server, _Server)
    assert server.args == (config,)
    assert host == "127.0.0.9"


def test_serve_runs_the_server_coroutine_under_the_main_loop(boot):
    runner.invoke(app, ["serve"])

    assert boot.run_main == ["the-run"]


def test_serve_lowers_the_asyncio_logger_to_errors(boot):
    runner.invoke(app, ["serve"])

    assert logging.getLogger("asyncio").level == logging.ERROR


def test_serve_says_another_server_holds_the_data_directory(boot, caplog):
    boot.lock_held_elsewhere = True
    with caplog.at_level(logging.ERROR, logger=_SERVERS_LOGGER):
        result = runner.invoke(app, ["serve"])

    assert result.exit_code == servers.LOCK_REFUSAL_EXIT_CODE
    assert _REFUSAL in _flat(result.output)
    assert [r.getMessage() for r in caplog.records if r.name == _SERVERS_LOGGER] == [_REFUSAL]
    assert boot.run_main == []


def test_serve_says_why_a_data_directory_cannot_lock(boot):
    boot.lock_error = LockingUnsupportedError("this mount has no lock daemon")

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == servers.LOCK_REFUSAL_EXIT_CODE
    assert "this mount has no lock daemon" in _flat(result.output)
    assert "already running" not in result.output
    assert boot.run_main == []


def test_a_failed_server_run_logs_the_traceback_and_raises(boot, caplog):
    boot.run_error = RuntimeError("bind failed")

    with caplog.at_level(logging.ERROR, logger=_SERVERS_LOGGER):
        result = runner.invoke(app, ["serve"])

    assert isinstance(result.exception, RuntimeError)
    records = [r for r in caplog.records if r.name == _SERVERS_LOGGER]
    assert [r.getMessage() for r in records] == ["The server run failed."]
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is RuntimeError


@pytest.fixture
def mcp_boot(monkeypatch):
    """Run mcp_cmd with the stdio server and the stop steps replaced by recorders."""
    rec = SimpleNamespace(overrides=[], stops=[], echoes=[], main_error=None)

    def main():
        if rec.main_error is not None:
            raise rec.main_error

    def stop_status(status):
        rec.stops.append(status)
        return 0

    monkeypatch.setattr(servers, "apply_overrides", lambda **kwargs: rec.overrides.append(kwargs))
    monkeypatch.setattr(servers, "_report_upgrade_as_log_lines", lambda: None)
    monkeypatch.setattr(servers, "stop_on_interrupt", lambda: None)
    monkeypatch.setattr(servers, "reset_services_on_exit", lambda: None)
    monkeypatch.setattr(servers, "exit_status_after_stop", stop_status)
    monkeypatch.setattr("lilbee.mcp_server.main", main)
    monkeypatch.setattr(
        typer, "echo", lambda message, **kwargs: rec.echoes.append((message, kwargs))
    )
    return rec


def test_mcp_applies_the_data_dir_and_the_global_flag(mcp_boot, tmp_path):
    chosen = tmp_path / "chosen"

    runner.invoke(app, ["mcp", "--data-dir", str(chosen), "--global"])

    assert mcp_boot.overrides == [{"data_dir": chosen, "use_global": True}]


def test_mcp_without_flags_applies_no_data_dir_and_no_global(mcp_boot):
    runner.invoke(app, ["mcp"])

    assert mcp_boot.overrides == [{"data_dir": None, "use_global": False}]


def test_mcp_asks_for_the_exit_status_of_a_clean_run(mcp_boot):
    result = runner.invoke(app, ["mcp"])

    assert result.exit_code == 0
    assert mcp_boot.stops == [0]


def test_a_failed_mcp_server_writes_one_line_to_stderr_and_exits_1(mcp_boot):
    mcp_boot.main_error = RuntimeError("no engine")

    result = runner.invoke(app, ["mcp"])

    assert result.exit_code == 1
    assert mcp_boot.echoes == [("lilbee: the MCP server failed: no engine", {"err": True})]


def test_a_failed_mcp_server_without_text_names_its_error_type(mcp_boot):
    mcp_boot.main_error = KeyError()

    runner.invoke(app, ["mcp"])

    assert mcp_boot.echoes == [("lilbee: the MCP server failed: KeyError", {"err": True})]


def test_a_failed_mcp_server_keeps_its_traceback_in_the_debug_log(mcp_boot, caplog):
    mcp_boot.main_error = RuntimeError("no engine")

    with caplog.at_level(logging.DEBUG, logger=_SERVERS_LOGGER):
        runner.invoke(app, ["mcp"])

    records = [r for r in caplog.records if r.name == _SERVERS_LOGGER]
    assert [r.getMessage() for r in records] == ["The MCP server failed."]
    assert records[0].levelno == logging.DEBUG
    assert records[0].exc_info is not None
    assert records[0].exc_info[0] is RuntimeError


def test_the_stopping_response_answers_the_request_it_was_given(monkeypatch):
    seen = []
    monkeypatch.setattr(
        server_app, "_json_error_response", lambda request, exc: seen.append((request, exc))
    )
    request = object()

    server_app._services_stopping_response(request, ServicesStoppingError())

    assert len(seen) == 1
    assert seen[0][0] is request
    assert seen[0][1].status_code == 503
    assert "lilbee is stopping" in seen[0][1].detail


@pytest.fixture
def external(monkeypatch):
    """An empty external-model cache, and a provider that lists what the test sets."""
    model_handlers._external_cache.clear()
    provider = SimpleNamespace(calls=0, models=["model-a"], error=None)

    def list_models():
        provider.calls += 1
        if provider.error is not None:
            raise provider.error
        return provider.models

    services = SimpleNamespace(provider=SimpleNamespace(list_models=list_models))
    monkeypatch.setattr(model_handlers, "get_services", lambda: services)
    yield provider
    model_handlers._external_cache.clear()


def test_another_api_key_asks_the_provider_again(external, monkeypatch):
    monkeypatch.setattr(cfg, "llm_api_key", "key-a")
    asyncio.run(model_handlers.list_external_models())
    monkeypatch.setattr(cfg, "llm_api_key", "key-b")

    asyncio.run(model_handlers.list_external_models())

    assert external.calls == 2


def test_no_api_key_is_not_the_same_cache_entry_as_any_key(external, monkeypatch):
    monkeypatch.setattr(cfg, "llm_api_key", "")
    asyncio.run(model_handlers.list_external_models())
    monkeypatch.setattr(cfg, "llm_api_key", "XXXX")

    asyncio.run(model_handlers.list_external_models())

    assert external.calls == 2


def test_a_failed_model_list_logs_the_provider_error_as_a_warning(external, caplog):
    external.error = RuntimeError("connection refused")

    with caplog.at_level(logging.WARNING, logger=model_handlers.__name__):
        asyncio.run(model_handlers.list_external_models())

    records = [r for r in caplog.records if r.name == model_handlers.__name__]
    assert [r.getMessage() for r in records] == [
        "Failed to list external models: connection refused"
    ]
    assert records[0].levelno == logging.WARNING


def test_a_failed_model_list_returns_the_provider_error_text(external):
    external.error = RuntimeError("connection refused")

    result = asyncio.run(model_handlers.list_external_models())

    assert result.models == []
    assert result.error == "connection refused"
