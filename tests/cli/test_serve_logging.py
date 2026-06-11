"""Tests for the serve log-file routing."""

from __future__ import annotations

import faulthandler
import logging
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from lilbee.cli.commands import serve_logging
from lilbee.cli.commands.serve_logging import (
    _BACKUP_COUNT,
    _MAX_BYTES,
    enable_fault_log,
    install_excepthook,
    setup_server_log_file,
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
    before_level = root.level
    yield
    for handler in root.handlers:
        if handler not in before_handlers:
            root.removeHandler(handler)
            handler.close()
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


def test_pins_notset_stream_handler_to_old_root_level(data_root: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    stream_handler = logging.StreamHandler()
    root.addHandler(stream_handler)
    assert stream_handler.level == logging.NOTSET

    log_path = setup_server_log_file()
    logging.getLogger("lilbee.test").info("file only")
    for handler in _server_log_handlers(log_path):
        handler.flush()
    assert "file only" in log_path.read_text()
    assert stream_handler.level == logging.WARNING


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


def test_excepthook_logs_and_chains(data_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    called: list[type[BaseException]] = []
    sys.excepthook = lambda tp, val, tb: called.append(tp)
    install_excepthook()
    err = RuntimeError("kaboom")
    with caplog.at_level(logging.CRITICAL):
        sys.excepthook(RuntimeError, err, None)
    assert "kaboom" in caplog.text
    assert called == [RuntimeError]
