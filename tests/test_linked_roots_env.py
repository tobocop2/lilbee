"""LILBEE_LINKED_ROOTS sets nothing: the load says so once and leaves linked_roots alone."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from conftest import clean_env
from lilbee.core.config import Config, refuse_environment
from lilbee.core.config.load_warnings import forget_shown
from lilbee.core.config.model import _build_cfg

_WARNING = (
    "LILBEE_LINKED_ROOTS is not a setting and is ignored; "
    "register a folder with `lilbee add`, or edit linked_roots in config.toml"
)


@pytest.fixture(autouse=True)
def _fresh_marks():
    forget_shown()
    yield
    forget_shown()


def _built_with(tmp_path: Path, **env: str) -> tuple[Config, tuple[str, ...]]:
    """cfg built in a fresh data root and *env*, with its load warnings."""
    with mock.patch.dict(os.environ, {**clean_env(tmp_path), **env}, clear=True):
        return _build_cfg()


def test_the_variable_is_named_in_one_warning(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path, LILBEE_LINKED_ROOTS="docs=/srv/docs")

    assert warnings == (_WARNING,)


def test_a_json_value_is_named_in_one_warning(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path, LILBEE_LINKED_ROOTS='{"docs": "/srv/docs"}')

    assert warnings == (_WARNING,)


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_variable_says_nothing(tmp_path, value) -> None:
    _built, warnings = _built_with(tmp_path, LILBEE_LINKED_ROOTS=value)

    assert warnings == ()


def test_an_unset_variable_says_nothing(tmp_path) -> None:
    _built, warnings = _built_with(tmp_path)

    assert warnings == ()


@pytest.mark.parametrize("value", [None, "docs=/srv/docs", '{"docs": "/srv/docs"}'])
def test_linked_roots_is_the_same_with_or_without_the_variable(tmp_path, value) -> None:
    env = {} if value is None else {"LILBEE_LINKED_ROOTS": value}
    built, _warnings = _built_with(tmp_path, **env)

    assert built.linked_roots == {}


def test_the_variable_does_not_stop_a_command(tmp_path) -> None:
    env = {**clean_env(tmp_path), "LILBEE_LINKED_ROOTS": "docs=/srv/docs"}
    with mock.patch.dict(os.environ, env, clear=True):
        refuse_environment()


def test_the_warning_is_logged_once_per_process(tmp_path, caplog) -> None:
    with caplog.at_level("WARNING", logger="lilbee.core.config.load_warnings"):
        _built_with(tmp_path, LILBEE_LINKED_ROOTS="docs=/srv/docs")
        _built_with(tmp_path, LILBEE_LINKED_ROOTS="docs=/srv/docs")

    assert [r.getMessage() for r in caplog.records] == [_WARNING]
