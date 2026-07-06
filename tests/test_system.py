"""Tests for platform-level helpers."""

import os
from pathlib import Path
from unittest import mock

import pytest

from lilbee.core.system import (
    _mount_fstype,
    chat_ctx_target_for_total_bytes,
    default_data_dir,
    find_local_root,
    is_ignored_dir,
    is_network_path,
    scaled_chat_ctx_target_default,
    stderr_suppressed,
)

_MOUNTS = (
    "proc /proc proc rw 0 0\n"
    "/dev/sda1 / ext4 rw 0 0\n"
    "/dev/sda1 /workspace ext4 rw 0 0\n"
    "server:/vol /workspace/models nfs4 rw 0 0\n"
    "mfs#src /mnt/mfs fuse.mfs rw 0 0\n"
)


class TestNetworkPath:
    def test_local_ext4_is_not_network(self):
        assert _mount_fstype("/workspace/index/foo.gguf", _MOUNTS) == "ext4"

    def test_longest_mount_wins_nfs(self):
        # /workspace/models is nfs4 even though /workspace and / are ext4.
        assert _mount_fstype("/workspace/models/chat-00001.gguf", _MOUNTS) == "nfs4"

    def test_mount_fstype_skips_malformed_lines(self):
        mounts = "garbage\n/dev/sda1 / ext4 rw 0 0\n"
        assert _mount_fstype("/x/y.gguf", mounts) == "ext4"

    def test_is_network_path_true_for_nfs(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **_k: _MOUNTS)
        monkeypatch.setattr(Path, "resolve", lambda self: Path("/workspace/models/m.gguf"))
        assert is_network_path(Path("/workspace/models/m.gguf")) is True

    def test_is_network_path_true_for_fuse_network(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **_k: _MOUNTS)
        monkeypatch.setattr(Path, "resolve", lambda self: Path("/mnt/mfs/m.gguf"))
        assert is_network_path(Path("/mnt/mfs/m.gguf")) is True

    def test_is_network_path_false_for_local(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **_k: _MOUNTS)
        monkeypatch.setattr(Path, "resolve", lambda self: Path("/workspace/index/m.gguf"))
        assert is_network_path(Path("/workspace/index/m.gguf")) is False

    def test_is_network_path_false_when_mounts_unreadable(self, monkeypatch):
        def _raise(self, **_k):
            raise OSError("no /proc/mounts")

        monkeypatch.setattr(Path, "read_text", _raise)
        assert is_network_path(Path("/anything")) is False

    def test_is_network_path_uses_raw_path_when_resolve_fails(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda self, **_k: _MOUNTS)

        def _raise(self):
            raise OSError("resolve failed")

        monkeypatch.setattr(Path, "resolve", _raise)
        assert is_network_path(Path("/workspace/models/m.gguf")) is True


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
        assert chat_ctx_target_for_total_bytes(16 * 1024**3 - 1) == 8192

    def test_16_to_32gb_picks_12k(self):
        assert chat_ctx_target_for_total_bytes(_gib(16)) == 12288
        assert chat_ctx_target_for_total_bytes(_gib(24)) == 12288
        assert chat_ctx_target_for_total_bytes(32 * 1024**3 - 1) == 12288

    def test_32_to_64gb_picks_16k(self):
        assert chat_ctx_target_for_total_bytes(_gib(32)) == 16384
        assert chat_ctx_target_for_total_bytes(_gib(48)) == 16384
        assert chat_ctx_target_for_total_bytes(64 * 1024**3 - 1) == 16384

    def test_64gb_and_above_picks_24k(self):
        assert chat_ctx_target_for_total_bytes(_gib(64)) == 24576
        assert chat_ctx_target_for_total_bytes(_gib(128)) == 24576
        assert chat_ctx_target_for_total_bytes(_gib(512)) == 24576

    def test_zero_or_negative_returns_floor(self):
        assert chat_ctx_target_for_total_bytes(0) == 8192
        assert chat_ctx_target_for_total_bytes(-1) == 8192
        assert chat_ctx_target_for_total_bytes(1) == 8192


class TestScaledChatCtxTargetDefault:
    def test_reads_total_memory_and_tiers(self):
        # 40 GiB host -> 16384 from the tier table.
        with mock.patch("lilbee.core.system._read_total_memory_bytes", return_value=40 * 1024**3):
            assert scaled_chat_ctx_target_default() == 16384

    def test_psutil_failure_falls_back_to_floor(self):
        with mock.patch("lilbee.core.system._read_total_memory_bytes", return_value=0):
            assert scaled_chat_ctx_target_default() == 8192

    def test_read_total_memory_bytes_returns_zero_on_psutil_failure(self):
        # Drive the real except-branch in _read_total_memory_bytes: psutil.virtual_memory()
        # raises -> wrapper returns 0 -> scaled default lands on the floor.
        with mock.patch("psutil.virtual_memory", side_effect=RuntimeError("boom")):
            from lilbee.core.system import _read_total_memory_bytes

            assert _read_total_memory_bytes() == 0
            assert scaled_chat_ctx_target_default() == 8192


class TestStderrSuppressed:
    def test_fd2_points_at_devnull_inside_then_restores(self):
        devnull_stat = os.stat(os.devnull)
        with stderr_suppressed():
            inside = os.fstat(2)
        # Inside the block fd 2 is the null device...
        assert (inside.st_dev, inside.st_ino) == (devnull_stat.st_dev, devnull_stat.st_ino)
        # ...and afterwards fd 2 is restored to a valid descriptor (no OSError).
        os.fstat(2)

    def test_restores_fd2_even_when_body_raises(self):
        with pytest.raises(ValueError, match="boom"), stderr_suppressed():
            raise ValueError("boom")
        os.fstat(2)  # restored despite the exception

    def test_win32_passthrough_is_noop(self, monkeypatch):
        """On win32 the context manager is a passthrough; fd 2 is left alone."""
        import sys as _sys

        monkeypatch.setattr(_sys, "platform", "win32")
        original_stat = os.fstat(2)
        executed = []
        with stderr_suppressed():
            executed.append(True)
            current_stat = os.fstat(2)
        assert executed == [True]
        # fd 2 was not redirected.
        assert (current_stat.st_dev, current_stat.st_ino) == (
            original_stat.st_dev,
            original_stat.st_ino,
        )
