"""Tests for platform-level env helpers."""

import os
from unittest import mock

import pytest

from lilbee.platform import (
    default_data_dir,
    env,
    env_float,
    env_int,
    env_int_optional,
    find_local_root,
    is_ignored_dir,
)


class TestHelpers:
    @pytest.mark.parametrize(
        "env_vars, key, default, expected",
        [
            ({}, "NONEXISTENT", "fallback", "fallback"),
            ({"LILBEE_NONEXISTENT": "override"}, "NONEXISTENT", "fallback", "override"),
        ],
        ids=["default", "override"],
    )
    def test_env(self, env_vars, key, default, expected):
        clear = not env_vars
        with mock.patch.dict(os.environ, env_vars, clear=clear):
            assert env(key, default) == expected

    @pytest.mark.parametrize(
        "env_vars, key, default, expected",
        [
            ({}, "NONEXISTENT", 42, 42),
            ({"LILBEE_NONEXISTENT": "99"}, "NONEXISTENT", 42, 99),
        ],
        ids=["default", "override"],
    )
    def test_env_int(self, env_vars, key, default, expected):
        clear = not env_vars
        with mock.patch.dict(os.environ, env_vars, clear=clear):
            assert env_int(key, default) == expected

    @pytest.mark.parametrize(
        "env_vars, key, default, expected",
        [
            ({}, "NONEXISTENT", None, None),
            ({}, "NONEXISTENT", 0.5, 0.5),
            ({"LILBEE_TEMP": "0.7"}, "TEMP", None, 0.7),
        ],
        ids=["default-none", "default-value", "override"],
    )
    def test_env_float(self, env_vars, key, default, expected):
        clear = not env_vars
        with mock.patch.dict(os.environ, env_vars, clear=clear):
            assert env_float(key, default) == expected

    @pytest.mark.parametrize(
        "env_vars, key, expected",
        [
            ({}, "NONEXISTENT", None),
            ({"LILBEE_SEED": "42"}, "SEED", 42),
        ],
        ids=["default-none", "override"],
    )
    def test_env_int_optional(self, env_vars, key, expected):
        clear = not env_vars
        with mock.patch.dict(os.environ, env_vars, clear=clear):
            assert env_int_optional(key) == expected

    def test_default_data_dir_darwin(self):
        with mock.patch("sys.platform", "darwin"):
            result = default_data_dir()
            assert "Application Support" in str(result)
            assert str(result).endswith("lilbee")

    def test_default_data_dir_linux(self):
        with (
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/tmp/xdg"}, clear=False),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_data_dir()
            assert result.parts[-1] == "lilbee"

    def test_default_data_dir_linux_fallback(self):
        filtered = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
        with (
            mock.patch.dict(os.environ, filtered, clear=True),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_data_dir()
            assert result.parts[-3:] == (".local", "share", "lilbee")

    def test_default_data_dir_windows(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}, clear=False),
            mock.patch("sys.platform", "win32"),
        ):
            result = default_data_dir()
            assert str(tmp_path) in str(result)

    def test_default_data_dir_windows_fallback(self):
        filtered = {k: v for k, v in os.environ.items() if k != "LOCALAPPDATA"}
        with (
            mock.patch.dict(os.environ, filtered, clear=True),
            mock.patch("sys.platform", "win32"),
        ):
            result = default_data_dir()
            assert "lilbee" in str(result)


class TestFindLocalRoot:
    def test_finds_in_cwd(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        assert find_local_root(tmp_path) == tmp_path / ".lilbee"

    def test_finds_in_parent(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert find_local_root(child) == tmp_path / ".lilbee"

    def test_returns_none_when_absent(self, tmp_path):
        assert find_local_root(tmp_path) is None

    def test_defaults_to_cwd(self, tmp_path):
        (tmp_path / ".lilbee").mkdir()
        with mock.patch("lilbee.platform.Path.cwd", return_value=tmp_path):
            assert find_local_root() == tmp_path / ".lilbee"


class TestIsIgnoredDir:
    _DEFAULTS = frozenset({"node_modules", "__pycache__", "venv"})

    @pytest.mark.parametrize("name", [".git", ".venv", ".cache"])
    def test_hidden_dirs(self, name):
        assert is_ignored_dir(name, self._DEFAULTS)

    @pytest.mark.parametrize("name", ["node_modules", "__pycache__", "venv"])
    def test_known_junk(self, name):
        assert is_ignored_dir(name, self._DEFAULTS)

    def test_egg_info(self):
        assert is_ignored_dir("mypackage.egg-info", self._DEFAULTS)

    @pytest.mark.parametrize("name", ["src", "docs", "tests"])
    def test_normal_dirs_not_ignored(self, name):
        assert not is_ignored_dir(name, self._DEFAULTS)

    def test_custom_ignore_dirs(self):
        custom = frozenset({"custom_output"})
        assert is_ignored_dir("custom_output", custom)
        assert not is_ignored_dir("src", custom)
