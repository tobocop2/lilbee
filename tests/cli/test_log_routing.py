"""Whether a WARNING record reaches the terminal."""

import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

from lilbee.cli.log_routing import (
    route_diagnostics_to_log_file,
    set_explicit_verbosity,
    warnings_reach_terminal,
)


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


def _capture_is_on() -> bool:
    return warnings.showwarning.__module__ == "logging"


@pytest.fixture
def routed_app(tmp_path: Path):
    """A one-command app whose command routes diagnostics, like ask and search."""
    app = typer.Typer()
    seen: dict[str, Any] = {}

    @app.command()
    def run(ctx: typer.Context, fail: bool = False) -> None:
        set_explicit_verbosity(False)
        stderr_handler = logging.StreamHandler(sys.stderr)
        logging.getLogger().addHandler(stderr_handler)
        seen["stderr_handler"] = stderr_handler
        seen["before"] = list(logging.getLogger().handlers)
        with mock.patch("lilbee.cli.log_routing.cfg") as m_cfg:
            m_cfg.data_root = tmp_path
            seen["path"] = route_diagnostics_to_log_file(ctx)
        seen["handlers_during"] = list(logging.getLogger().handlers)
        seen["capture_during"] = _capture_is_on()
        if fail:
            raise SystemExit(1)

    return app, seen


@pytest.fixture
def restored_logging():
    """Put the root handlers and the warnings hook back, whatever a test does."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    logging.captureWarnings(False)
    yield
    logging.captureWarnings(False)
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)


@pytest.mark.parametrize("fail", [False, True])
def test_a_command_that_routes_diagnostics_leaves_the_root_handlers_as_it_found_them(
    routed_app, restored_logging, fail: bool
):
    app, seen = routed_app
    result = CliRunner().invoke(app, ["--fail"] if fail else [])
    assert result.exit_code == (1 if fail else 0)
    assert seen["path"] is not None
    assert seen["stderr_handler"] not in seen["handlers_during"]
    assert any(isinstance(h, RotatingFileHandler) for h in seen["handlers_during"])
    assert logging.getLogger().handlers == seen["before"]


@pytest.mark.parametrize("fail", [False, True])
def test_a_command_that_routes_diagnostics_turns_warning_capture_back_off(
    routed_app, restored_logging, fail: bool
):
    app, seen = routed_app
    assert _capture_is_on() is False
    CliRunner().invoke(app, ["--fail"] if fail else [])
    assert seen["capture_during"] is True
    assert _capture_is_on() is False


def test_a_command_leaves_warning_capture_on_when_it_was_on_before(routed_app, restored_logging):
    app, _ = routed_app
    logging.captureWarnings(True)
    CliRunner().invoke(app, [])
    assert _capture_is_on() is True


def test_the_log_file_handler_is_closed_when_the_command_ends(routed_app, restored_logging):
    app, seen = routed_app
    CliRunner().invoke(app, [])
    (file_handler,) = [h for h in seen["handlers_during"] if isinstance(h, RotatingFileHandler)]
    assert file_handler.stream is None
