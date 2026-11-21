"""Whether a WARNING record reaches the terminal."""

import logging
import sys
from pathlib import Path

import pytest

from lilbee.cli.log_routing import warnings_reach_terminal


@pytest.fixture
def root():
    logger = logging.getLogger()
    before = list(logger.handlers)
    level = logger.level
    yield logger
    logger.setLevel(level)
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)


def test_a_stderr_handler_at_warning_puts_warnings_on_the_terminal(root, capsys):
    root.addHandler(logging.StreamHandler(sys.stderr))
    assert warnings_reach_terminal() is True


def test_a_stderr_handler_above_warning_keeps_warnings_off_the_terminal(root, capsys):
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    root.addHandler(handler)
    assert warnings_reach_terminal() is False


def test_a_log_file_keeps_warnings_off_the_terminal(root, tmp_path: Path):
    root.addHandler(logging.FileHandler(tmp_path / "cli.log"))
    assert warnings_reach_terminal() is False


def test_a_root_level_above_warning_keeps_warnings_off_the_terminal(root, capsys):
    root.addHandler(logging.StreamHandler(sys.stderr))
    root.setLevel(logging.ERROR)
    assert warnings_reach_terminal() is False
