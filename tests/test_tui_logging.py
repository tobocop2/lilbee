"""Tests for lilbee.cli.tui.log_routing: TUI rotating-file log routing."""

from __future__ import annotations

import contextlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from lilbee.cli.tui import (
    _redirect_native_stderr_to,
    _restore_native_stderr,
    _silence_stderr_log_handlers,
)
from lilbee.cli.tui.log_routing import setup_tui_log_file
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_root_logger():
    """Snapshot root logger handlers; restore and close any added during the test."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    added = [h for h in root.handlers if h not in before]
    for handler in added:
        root.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_root", tmp_path)


def test_setup_tui_log_file_creates_dir(tmp_path):
    assert not (tmp_path / "logs").exists()
    setup_tui_log_file()
    assert (tmp_path / "logs").is_dir()


def test_setup_tui_log_file_returns_path(tmp_path):
    log_path = setup_tui_log_file()
    assert isinstance(log_path, Path)
    assert log_path == tmp_path / "logs" / "tui.log"


def test_setup_tui_log_file_attaches_handler():
    root = logging.getLogger()
    before = {id(h) for h in root.handlers}
    setup_tui_log_file()
    new_handlers = [h for h in root.handlers if id(h) not in before]
    assert len(new_handlers) == 1
    assert isinstance(new_handlers[0], RotatingFileHandler)


def test_setup_tui_log_file_idempotent():
    root = logging.getLogger()
    setup_tui_log_file()
    count_after_first = len(root.handlers)
    setup_tui_log_file()
    setup_tui_log_file()
    assert len(root.handlers) == count_after_first


def test_log_records_reach_file(tmp_path):
    log_path = setup_tui_log_file()
    logger = logging.getLogger("lilbee")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        logger.warning("probe-12345")
    finally:
        logger.setLevel(previous_level)
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = log_path.read_text()
    assert "probe-12345" in content
    assert "WARNING" in content
    assert "lilbee" in content


def test_silencer_preserves_file_handler():
    root = logging.getLogger()
    snapshot = set(root.handlers)
    setup_tui_log_file()
    added_file_handlers = [
        h for h in root.handlers if h not in snapshot and isinstance(h, RotatingFileHandler)
    ]
    assert len(added_file_handlers) == 1
    _silence_stderr_log_handlers()
    assert all(h in root.handlers for h in added_file_handlers)


def test_native_stderr_redirect_captures_fd2_writes(tmp_path):
    """fd 2 writes (the path C deps like tesseract use) must land in the log
    file, not on the terminal where Textual is painting. The Python-level
    silencer doesn't catch native code; this redirect does.
    """
    import os

    log_path = setup_tui_log_file()
    saved = _redirect_native_stderr_to(log_path)
    assert saved is not None, "redirect should install successfully on a writable path"
    try:
        os.write(2, b"native-probe-XYZ\n")
        os.fsync(2)
    finally:
        _restore_native_stderr(saved)
    assert "native-probe-XYZ" in log_path.read_text()


def test_native_stderr_redirect_returns_none_on_unwritable_path(tmp_path):
    """Bad path means we leave fd 2 alone and report None, not crash the TUI."""
    bogus = tmp_path / "does" / "not" / "exist" / "tui.log"
    saved = _redirect_native_stderr_to(bogus)
    assert saved is None


def test_restore_native_stderr_handles_none():
    """Restoring with no saved fd is a documented no-op."""
    _restore_native_stderr(None)


def test_rotation_caps_file_size(tmp_path, monkeypatch):
    """A small maxBytes override forces rotation; tui.log.1 must appear."""
    monkeypatch.setattr("lilbee.cli.tui.log_routing._MAX_BYTES", 512)
    log_path = setup_tui_log_file()
    logger = logging.getLogger("lilbee.test_rotation")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        # Each record line is well under 512 bytes; 50 records guarantees rollover.
        for i in range(50):
            logger.warning("rotation-probe-%03d-%s", i, "x" * 80)
    finally:
        logger.setLevel(previous_level)
    for handler in logging.getLogger().handlers:
        handler.flush()
    backup = log_path.parent / "tui.log.1"
    assert backup.exists(), (
        f"expected rotated backup at {backup}; logs dir: {list(log_path.parent.iterdir())}"
    )
