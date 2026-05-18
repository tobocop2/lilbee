"""Tests for platform-level helpers."""

from pathlib import Path
from unittest import mock

import pytest

from lilbee.core.system import (
    default_data_dir,
    find_local_root,
    is_ignored_dir,
)


class TestHelpers:
    def test_default_data_dir_ends_with_lilbee(self):
        result = default_data_dir()
        assert isinstance(result, Path)
        assert result.parts[-1] == "lilbee"

    def test_default_data_dir_delegates_to_platformdirs(self, tmp_path):
        with mock.patch(
            "lilbee.core.system.user_data_dir",
            return_value=str(tmp_path / "lilbee"),
        ) as m:
            result = default_data_dir()
            assert result == tmp_path / "lilbee"
            m.assert_called_once_with("lilbee", appauthor=False)


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
        with mock.patch("lilbee.core.system.Path.cwd", return_value=tmp_path):
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
