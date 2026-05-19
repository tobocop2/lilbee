"""Tests for platform-level helpers."""

import os
from unittest import mock

import pytest

from lilbee.core.system import (
    chat_ctx_target_for_total_bytes,
    default_data_dir,
    find_local_root,
    is_ignored_dir,
)


class TestHelpers:
    def test_default_data_dir_darwin(self):
        with mock.patch("sys.platform", "darwin"):
            result = default_data_dir()
            assert "Application Support" in str(result)
            assert str(result).endswith("lilbee")

    def test_default_data_dir_linux(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path / "xdg")}, clear=False),
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


def _gib(n: float) -> int:
    return int(n * 1024**3)


class TestChatCtxTargetForTotalBytes:
    def test_under_16gb_stays_at_8k_floor(self):
        assert chat_ctx_target_for_total_bytes(_gib(4)) == 8192
        assert chat_ctx_target_for_total_bytes(_gib(8)) == 8192
        assert chat_ctx_target_for_total_bytes(_gib(15.99)) == 8192

    def test_16_to_32gb_picks_12k(self):
        assert chat_ctx_target_for_total_bytes(_gib(16)) == 12288
        assert chat_ctx_target_for_total_bytes(_gib(24)) == 12288
        assert chat_ctx_target_for_total_bytes(_gib(31.99)) == 12288

    def test_32_to_64gb_picks_16k(self):
        assert chat_ctx_target_for_total_bytes(_gib(32)) == 16384
        assert chat_ctx_target_for_total_bytes(_gib(48)) == 16384
        assert chat_ctx_target_for_total_bytes(_gib(63.99)) == 16384

    def test_64gb_and_above_picks_24k(self):
        assert chat_ctx_target_for_total_bytes(_gib(64)) == 24576
        assert chat_ctx_target_for_total_bytes(_gib(128)) == 24576
        assert chat_ctx_target_for_total_bytes(_gib(512)) == 24576

    def test_zero_or_negative_returns_floor(self):
        assert chat_ctx_target_for_total_bytes(0) == 8192
        assert chat_ctx_target_for_total_bytes(-1) == 8192
