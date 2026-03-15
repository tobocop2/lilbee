"""Tests for platform-level env helpers."""

import os
from unittest import mock


class TestHelpers:
    def test_env_returns_default(self):
        from lilbee.platform import env

        with mock.patch.dict(os.environ, {}, clear=True):
            assert env("NONEXISTENT", "fallback") == "fallback"

    def test_env_returns_override(self):
        from lilbee.platform import env

        with mock.patch.dict(os.environ, {"LILBEE_NONEXISTENT": "override"}):
            assert env("NONEXISTENT", "fallback") == "override"

    def test_env_int_returns_default(self):
        from lilbee.platform import env_int

        with mock.patch.dict(os.environ, {}, clear=True):
            assert env_int("NONEXISTENT", 42) == 42

    def test_env_int_returns_override(self):
        from lilbee.platform import env_int

        with mock.patch.dict(os.environ, {"LILBEE_NONEXISTENT": "99"}):
            assert env_int("NONEXISTENT", 42) == 99

    def test_env_float_returns_default_none(self):
        from lilbee.platform import env_float

        with mock.patch.dict(os.environ, {}, clear=True):
            assert env_float("NONEXISTENT") is None

    def test_env_float_returns_default_value(self):
        from lilbee.platform import env_float

        with mock.patch.dict(os.environ, {}, clear=True):
            assert env_float("NONEXISTENT", 0.5) == 0.5

    def test_env_float_returns_override(self):
        from lilbee.platform import env_float

        with mock.patch.dict(os.environ, {"LILBEE_TEMP": "0.7"}):
            assert env_float("TEMP") == 0.7

    def test_env_int_optional_returns_none(self):
        from lilbee.platform import env_int_optional

        with mock.patch.dict(os.environ, {}, clear=True):
            assert env_int_optional("NONEXISTENT") is None

    def test_env_int_optional_returns_value(self):
        from lilbee.platform import env_int_optional

        with mock.patch.dict(os.environ, {"LILBEE_SEED": "42"}):
            assert env_int_optional("SEED") == 42

    def test_default_data_dir_darwin(self):
        from lilbee.platform import default_data_dir

        with mock.patch("sys.platform", "darwin"):
            result = default_data_dir()
            assert "Application Support" in str(result)
            assert str(result).endswith("lilbee")

    def test_default_data_dir_linux(self):
        from lilbee.platform import default_data_dir

        with (
            mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/tmp/xdg"}, clear=False),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_data_dir()
            assert result.parts[-1] == "lilbee"

    def test_default_data_dir_linux_fallback(self):
        from lilbee.platform import default_data_dir

        env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_data_dir()
            assert result.parts[-3:] == (".local", "share", "lilbee")

    def test_default_data_dir_windows(self, tmp_path):
        from lilbee.platform import default_data_dir

        with (
            mock.patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}, clear=False),
            mock.patch("sys.platform", "win32"),
        ):
            result = default_data_dir()
            assert str(tmp_path) in str(result)

    def test_default_data_dir_windows_fallback(self):
        from lilbee.platform import default_data_dir

        env = {k: v for k, v in os.environ.items() if k != "LOCALAPPDATA"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.platform", "win32"),
        ):
            result = default_data_dir()
            assert "lilbee" in str(result)


class TestFindLocalRoot:
    def test_finds_in_cwd(self, tmp_path):
        from lilbee.platform import find_local_root

        (tmp_path / ".lilbee").mkdir()
        assert find_local_root(tmp_path) == tmp_path / ".lilbee"

    def test_finds_in_parent(self, tmp_path):
        from lilbee.platform import find_local_root

        (tmp_path / ".lilbee").mkdir()
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        assert find_local_root(child) == tmp_path / ".lilbee"

    def test_returns_none_when_absent(self, tmp_path):
        from lilbee.platform import find_local_root

        assert find_local_root(tmp_path) is None

    def test_defaults_to_cwd(self, tmp_path):
        from lilbee.platform import find_local_root

        (tmp_path / ".lilbee").mkdir()
        with mock.patch("lilbee.platform.Path.cwd", return_value=tmp_path):
            assert find_local_root() == tmp_path / ".lilbee"


class TestIsIgnoredDir:
    _DEFAULTS = frozenset({"node_modules", "__pycache__", "venv"})

    def test_hidden_dirs(self):
        from lilbee.platform import is_ignored_dir

        assert is_ignored_dir(".git", self._DEFAULTS)
        assert is_ignored_dir(".venv", self._DEFAULTS)
        assert is_ignored_dir(".cache", self._DEFAULTS)

    def test_known_junk(self):
        from lilbee.platform import is_ignored_dir

        assert is_ignored_dir("node_modules", self._DEFAULTS)
        assert is_ignored_dir("__pycache__", self._DEFAULTS)
        assert is_ignored_dir("venv", self._DEFAULTS)

    def test_egg_info(self):
        from lilbee.platform import is_ignored_dir

        assert is_ignored_dir("mypackage.egg-info", self._DEFAULTS)

    def test_normal_dirs_not_ignored(self):
        from lilbee.platform import is_ignored_dir

        assert not is_ignored_dir("src", self._DEFAULTS)
        assert not is_ignored_dir("docs", self._DEFAULTS)
        assert not is_ignored_dir("tests", self._DEFAULTS)

    def test_custom_ignore_dirs(self):
        from lilbee.platform import is_ignored_dir

        custom = frozenset({"custom_output"})
        assert is_ignored_dir("custom_output", custom)
        assert not is_ignored_dir("src", custom)
