"""Tests for the serve log-file routing."""

from __future__ import annotations

import faulthandler
import logging
import logging.config
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from lilbee.cli.commands import serve_logging
from lilbee.cli.commands.serve_logging import (
    _BACKUP_COUNT,
    _MAX_BYTES,
    _HealthAccessFilter,
    enable_fault_log,
    install_excepthook,
    install_health_access_filter,
    setup_server_log_file,
    setup_server_logging,
)
from lilbee.core.config import cfg


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_handler_levels = [handler.level for handler in before_handlers]
    before_level = root.level
    yield
    for handler in root.handlers:
        if handler not in before_handlers:
            handler.close()
    root.handlers[:] = before_handlers
    for handler, level in zip(before_handlers, before_handler_levels, strict=True):
        handler.setLevel(level)
    root.setLevel(before_level)


def _server_log_handlers(log_path: Path) -> list[RotatingFileHandler]:
    return [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == log_path
    ]


def test_installs_rotating_handler(data_root: Path) -> None:
    log_path = setup_server_log_file()
    assert log_path == data_root / "logs" / "server.log"
    handlers = _server_log_handlers(log_path)
    assert len(handlers) == 1
    assert handlers[0].maxBytes == _MAX_BYTES
    assert handlers[0].backupCount == _BACKUP_COUNT
    assert handlers[0].level == logging.INFO


def test_idempotent(data_root: Path) -> None:
    first = setup_server_log_file()
    second = setup_server_log_file()
    assert first == second
    assert len(_server_log_handlers(first)) == 1


_LOAD_WARNING = "config.toml: top_k = 'many' is not a whole number; top_k uses its default"


def _warning_lines(log_path: Path) -> list[str]:
    for handler in _server_log_handlers(log_path):
        handler.flush()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [line.split(" WARNING ", 1)[1] for line in lines if " WARNING " in line]


def test_a_serve_start_writes_each_load_warning_to_the_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real serve command: its setup, then the app build."""
    from typer.testing import CliRunner

    from lilbee.cli import app
    from lilbee.cli.commands import servers

    monkeypatch.setattr(serve_logging, "load_warnings", (_LOAD_WARNING,))
    log_path = tmp_path / "logs" / "server.log"
    handlers_at_run: list[int] = []

    def no_server(coro: object, *_args: object, **_kwargs: object) -> None:
        coro.close()  # type: ignore[attr-defined]
        handlers_at_run.append(len(_server_log_handlers(log_path)))
        logging.getLogger("lilbee.test").warning("control line")

    monkeypatch.setattr(servers.asyncio, "run", no_server)
    result = CliRunner().invoke(app, ["serve", "-d", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert handlers_at_run == [1]
    assert _warning_lines(log_path) == [
        f"lilbee.cli.commands.serve_logging: {_LOAD_WARNING}",
        "lilbee.test: control line",
    ]


def test_a_handler_installed_again_does_not_repeat_the_load_warnings(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(serve_logging, "load_warnings", (_LOAD_WARNING,))
    setup_server_logging()
    log_path = data_root / "logs" / "server.log"
    for handler in _server_log_handlers(log_path):
        logging.getLogger().removeHandler(handler)
        handler.close()
    assert setup_server_log_file() == log_path
    assert len(_server_log_handlers(log_path)) == 1
    logging.getLogger("lilbee.test").warning("control line")
    assert _warning_lines(log_path) == [
        f"lilbee.cli.commands.serve_logging: {_LOAD_WARNING}",
        "lilbee.test: control line",
    ]
    assert _LOAD_WARNING not in capsys.readouterr().err


def test_info_reaches_file_when_root_was_warning(data_root: Path) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    log_path = setup_server_log_file()
    logging.getLogger("lilbee.test").info("hello file")
    for handler in _server_log_handlers(log_path):
        handler.flush()
    assert "hello file" in log_path.read_text()


def test_does_not_lower_a_more_verbose_root(data_root: Path) -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    setup_server_log_file()
    assert logging.getLogger().level == logging.DEBUG


def test_reinstalls_after_dictconfig_root_reset(data_root: Path) -> None:
    log_path = setup_server_log_file()
    logging.config.dictConfig(
        {"version": 1, "disable_existing_loggers": False, "root": {"handlers": [], "level": "INFO"}}
    )
    assert not _server_log_handlers(log_path)
    setup_server_log_file()
    assert len(_server_log_handlers(log_path)) == 1
    logging.getLogger("lilbee.test").info("after reset")
    for handler in _server_log_handlers(log_path):
        handler.flush()
    assert "after reset" in log_path.read_text()


def test_create_app_keeps_the_server_log_handler(data_root: Path) -> None:
    from lilbee.server import create_app

    log_path = setup_server_log_file()
    create_app()
    assert len(_server_log_handlers(log_path)) == 1


@pytest.fixture(autouse=True)
def _reset_fault_log() -> Iterator[None]:
    yield
    handle = serve_logging._fault_log._handle
    if handle is not None and not handle.closed:
        faulthandler.disable()
        handle.close()
        # Restore pytest's stderr fault dumping.
        faulthandler.enable()
    serve_logging._fault_log._handle = None


@pytest.fixture(autouse=True)
def _restore_excepthook() -> Iterator[None]:
    before = sys.excepthook
    yield
    sys.excepthook = before
    serve_logging._except_hook.installed = False


def test_fault_log_enabled(data_root: Path) -> None:
    fault_path = enable_fault_log()
    assert fault_path == data_root / "logs" / "server-fault.log"
    assert fault_path.exists()
    assert faulthandler.is_enabled()
    # pytest enables faulthandler itself, so also assert our handle is live.
    handle = serve_logging._fault_log._handle
    assert handle is not None
    assert not handle.closed
    assert Path(handle.name) == fault_path


def test_fault_log_idempotent(data_root: Path) -> None:
    first = enable_fault_log()
    handle = serve_logging._fault_log._handle
    second = enable_fault_log()
    assert first == second
    assert serve_logging._fault_log._handle is handle


def test_fault_log_reopens_closed_handle(data_root: Path) -> None:
    first = enable_fault_log()
    handle = serve_logging._fault_log._handle
    assert handle is not None
    faulthandler.disable()
    handle.close()
    second = enable_fault_log()
    assert second == first
    reopened = serve_logging._fault_log._handle
    assert reopened is not None
    assert not reopened.closed
    assert faulthandler.is_enabled()


def test_setup_server_logging_installs_everything(data_root: Path) -> None:
    setup_server_logging()
    assert _server_log_handlers(data_root / "logs" / "server.log")
    assert (data_root / "logs" / "server-fault.log").exists()
    assert faulthandler.is_enabled()


def test_excepthook_logs_and_chains(data_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    called: list[type[BaseException]] = []
    sys.excepthook = lambda tp, val, tb: called.append(tp)
    install_excepthook()
    err = RuntimeError("kaboom")
    with caplog.at_level(logging.CRITICAL):
        sys.excepthook(RuntimeError, err, None)
    assert "kaboom" in caplog.text
    assert called == [RuntimeError]


def test_excepthook_install_idempotent(data_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    called: list[type[BaseException]] = []
    sys.excepthook = lambda tp, val, tb: called.append(tp)
    install_excepthook()
    install_excepthook()
    with caplog.at_level(logging.CRITICAL):
        sys.excepthook(RuntimeError, RuntimeError("once"), None)
    assert caplog.text.count("unhandled exception") == 1
    assert called == [RuntimeError]


@pytest.fixture(autouse=True)
def _isolate_access_filters() -> Iterator[None]:
    access_logger = logging.getLogger("uvicorn.access")
    before = list(access_logger.filters)
    for filt in before:
        access_logger.removeFilter(filt)
    yield
    for filt in access_logger.filters[:]:
        access_logger.removeFilter(filt)
    for filt in before:
        access_logger.addFilter(filt)


def _access_record(path: str) -> logging.LogRecord:
    access_logger = logging.getLogger("uvicorn.access")
    return access_logger.makeRecord(
        access_logger.name,
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1", "GET", path, "1.1", 200),
        None,
    )


def test_health_access_filter_drops_health_record() -> None:
    install_health_access_filter()
    access_logger = logging.getLogger("uvicorn.access")
    assert access_logger.filter(_access_record("/api/health")) is False


def test_health_access_filter_keeps_other_records() -> None:
    install_health_access_filter()
    access_logger = logging.getLogger("uvicorn.access")
    assert access_logger.filter(_access_record("/api/status"))


def test_install_health_access_filter_idempotent() -> None:
    install_health_access_filter()
    install_health_access_filter()
    access_logger = logging.getLogger("uvicorn.access")
    ours = [f for f in access_logger.filters if isinstance(f, _HealthAccessFilter)]
    assert len(ours) == 1


def _console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler


def test_serve_console_shows_info_when_the_cli_level_is_warning(data_root: Path) -> None:
    console = _console_handler(logging.WARNING)
    file_only = logging.FileHandler(data_root / "other.log", encoding="utf-8")
    file_only.setLevel(logging.ERROR)
    logging.getLogger().addHandler(file_only)

    setup_server_logging()

    assert console.level == logging.INFO
    assert file_only.level == logging.ERROR


def test_serve_console_keeps_a_debug_cli_level(data_root: Path) -> None:
    console = _console_handler(logging.DEBUG)

    setup_server_logging()

    assert console.level == logging.DEBUG


def test_serve_console_prints_info_lines_from_lilbee_loggers(
    data_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(logging.StreamHandler(sys.stderr))
    root.handlers[-1].setLevel(logging.NOTSET)

    setup_server_logging()
    logging.getLogger("lilbee.server.app").info("startup line")

    assert "startup line" in capsys.readouterr().err


class _Exited(BaseException):
    """Stands in for the process end, so the deadline runs inside the test."""


def test_the_deadline_writes_the_held_threads_and_exits_with_status_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.cli.commands import serve_logging

    statuses: list[int] = []

    def fake_exit(status: int) -> None:
        statuses.append(status)
        raise _Exited

    monkeypatch.setattr(serve_logging.cfg, "data_root", tmp_path)
    monkeypatch.setattr(serve_logging.os, "_exit", fake_exit)
    started: list[object] = []
    monkeypatch.setattr(
        serve_logging.threading.Thread, "start", lambda thread: started.append(thread)
    )
    serve_logging.arm_exit_deadline(0)
    assert len(started) == 1
    with pytest.raises(_Exited):
        serve_logging._exit_at_the_deadline(0)
    assert statuses == [1]
    assert "Timeout (0 s)!" in (tmp_path / "logs" / "server-fault.log").read_text(encoding="utf-8")
