"""Tests for lilbee.cli.tui.log_routing: TUI rotating-file log routing."""

from __future__ import annotations

import contextlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from lilbee.cli.tui import _silence_stderr_log_handlers
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
    setup_tui_log_file()
    file_handlers_before = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers_before) == 1
    _silence_stderr_log_handlers()
    file_handlers_after = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers_after == file_handlers_before


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
