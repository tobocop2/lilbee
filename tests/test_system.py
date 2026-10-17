"""Tests for platform-level helpers."""

import os
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from lilbee.core import system as system_mod
from lilbee.core.system import (
    DiskSpelling,
    _Folder,
    _mount_fstype,
    chat_ctx_target_for_total_bytes,
    default_data_dir,
    default_state_dir,
    disk_path,
    executable_search_path,
    find_executable,
    find_local_root,
    is_ignored_dir,
    is_network_path,
    scaled_chat_ctx_target_default,
    stderr_suppressed,
)

# The run patches the module's default_cache_dir, so the platform original
# these formula tests check is stashed in the suite module.
from tests._suite_data_root import platform_default_cache_dir as default_cache_dir

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

    @pytest.fixture
    def mounts_file(self, tmp_path, monkeypatch):
        """Point the module's /proc/mounts seam at a test-controlled file."""
        mounts = tmp_path / "mounts"
        mounts.write_text(_MOUNTS)
        monkeypatch.setattr(system_mod, "_PROC_MOUNTS", mounts)
        return mounts

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mount-path semantics")
    def test_is_network_path_true_for_nfs(self, mounts_file):
        assert is_network_path(Path("/workspace/models/m.gguf")) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mount-path semantics")
    def test_is_network_path_true_for_fuse_network(self, mounts_file):
        assert is_network_path(Path("/mnt/mfs/m.gguf")) is True

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mount-path semantics")
    def test_is_network_path_false_for_local(self, mounts_file):
        assert is_network_path(Path("/workspace/index/m.gguf")) is False

    def test_is_network_path_false_when_mounts_unreadable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(system_mod, "_PROC_MOUNTS", tmp_path / "missing")
        assert is_network_path(Path("/anything")) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mount-path semantics")
    def test_is_network_path_uses_raw_path_when_resolve_fails(self, mounts_file, monkeypatch):
        # Path.resolve has no injectable seam, so this one branch patches it.
        # Raise only for the path under test; an unconditional patch also hits
        # the config validator and blows up in fixture teardown.
        target = Path("/workspace/models/m.gguf")
        real_resolve = Path.resolve

        def _raise(self, strict=False):
            if self == target:
                raise OSError("resolve failed")
            return real_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _raise)
        assert is_network_path(target) is True


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

    def test_default_state_dir_is_not_a_cache_dir(self, tmp_path):
        """Live engine records must not sit where a cleaner may wipe them."""
        with (
            mock.patch.dict(
                os.environ,
                {"XDG_STATE_HOME": str(tmp_path / "state"), "XDG_CACHE_HOME": str(tmp_path / "c")},
                clear=False,
            ),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_state_dir()
            assert result == tmp_path / "state" / "lilbee"

    def test_default_state_dir_linux_fallback(self):
        filtered = {k: v for k, v in os.environ.items() if k != "XDG_STATE_HOME"}
        with (
            mock.patch.dict(os.environ, filtered, clear=True),
            mock.patch("sys.platform", "linux"),
        ):
            result = default_state_dir()
            assert result.parts[-3:] == (".local", "state", "lilbee")

    def test_default_state_dir_darwin_avoids_purgeable_caches(self):
        with mock.patch("sys.platform", "darwin"):
            result = default_state_dir()
            assert "Caches" not in str(result)
            assert "Application Support" in str(result)

    def test_default_cache_dir_darwin_uses_the_purgeable_caches(self):
        """The counterpart to the state directory sits where a cleaner may empty it."""
        with mock.patch("sys.platform", "darwin"):
            assert default_cache_dir() == Path.home() / "Library" / "Caches" / "lilbee"

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


_NO_HOME = "Could not determine home directory."
# helper, environment variable, tail under that variable, tail under the home directory
_WINDOWS_DIR_CASES = (
    (default_data_dir, "LOCALAPPDATA", ("lilbee",), ("AppData", "Local", "lilbee")),
    (default_state_dir, "LOCALAPPDATA", ("lilbee",), ("AppData", "Local", "lilbee")),
    (
        default_cache_dir,
        "LOCALAPPDATA",
        ("lilbee", "cache"),
        ("AppData", "Local", "lilbee", "cache"),
    ),
)
_LINUX_DIR_CASES = (
    (default_data_dir, "XDG_DATA_HOME", ("lilbee",), (".local", "share", "lilbee")),
    (default_state_dir, "XDG_STATE_HOME", ("lilbee",), (".local", "state", "lilbee")),
    (default_cache_dir, "XDG_CACHE_HOME", ("lilbee",), (".cache", "lilbee")),
)


def _without_a_home(monkeypatch):
    """Make ``Path.home()`` raise, as it does on a host that resolves no home directory."""

    def _raise(_cls):
        raise RuntimeError(_NO_HOME)

    monkeypatch.setattr(Path, "home", classmethod(_raise))


class TestPlatformDirsWithoutAHomeDirectory:
    """A host with no resolvable home still gets its directories from the environment."""

    @pytest.mark.parametrize(("helper", "var", "tail", "home_tail"), _WINDOWS_DIR_CASES)
    def test_windows_dirs_come_from_the_variable(
        self, helper, var, tail, home_tail, monkeypatch, tmp_path
    ):
        _without_a_home(monkeypatch)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv(var, str(tmp_path))

        assert helper() == tmp_path.joinpath(*tail)

    @pytest.mark.parametrize(("helper", "var", "tail", "home_tail"), _LINUX_DIR_CASES)
    def test_linux_dirs_come_from_the_variable(
        self, helper, var, tail, home_tail, monkeypatch, tmp_path
    ):
        _without_a_home(monkeypatch)
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv(var, str(tmp_path))

        assert helper() == tmp_path.joinpath(*tail)

    @pytest.mark.parametrize(
        ("helper", "var", "tail", "home_tail"), (*_WINDOWS_DIR_CASES, *_LINUX_DIR_CASES)
    )
    def test_an_unresolvable_home_still_raises_where_nothing_names_the_directory(
        self, helper, var, tail, home_tail, monkeypatch
    ):
        """Control: with the variable unset there is no answer but the home directory."""
        _without_a_home(monkeypatch)
        monkeypatch.setattr(sys, "platform", "win32" if var == "LOCALAPPDATA" else "linux")
        monkeypatch.delenv(var, raising=False)

        with pytest.raises(RuntimeError, match=_NO_HOME):
            helper()


class TestPlatformDirsKeepTheirPaths:
    """Reading the variable first must not move any path on a host that has a home."""

    @pytest.mark.parametrize(
        ("helper", "var", "tail", "home_tail"), (*_WINDOWS_DIR_CASES, *_LINUX_DIR_CASES)
    )
    def test_the_home_default_still_answers_when_the_variable_is_unset(
        self, helper, var, tail, home_tail, monkeypatch
    ):
        monkeypatch.setattr(sys, "platform", "win32" if var == "LOCALAPPDATA" else "linux")
        monkeypatch.delenv(var, raising=False)

        assert helper() == Path.home().joinpath(*home_tail)

    @pytest.mark.parametrize(
        ("helper", "var", "tail", "home_tail"), (*_WINDOWS_DIR_CASES, *_LINUX_DIR_CASES)
    )
    def test_the_variable_wins_over_the_home_default(
        self, helper, var, tail, home_tail, monkeypatch, tmp_path
    ):
        """A host with both a home directory and the variable set follows the variable."""
        monkeypatch.setattr(sys, "platform", "win32" if var == "LOCALAPPDATA" else "linux")
        monkeypatch.setenv(var, str(tmp_path))

        assert helper() == tmp_path.joinpath(*tail)
        assert helper() != Path.home().joinpath(*home_tail)


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

    def test_64_to_128gb_picks_24k(self):
        assert chat_ctx_target_for_total_bytes(_gib(64)) == 24576
        assert chat_ctx_target_for_total_bytes(_gib(96)) == 24576
        assert chat_ctx_target_for_total_bytes(128 * 1024**3 - 1) == 24576

    def test_128gb_and_above_picks_the_agent_floor(self):
        # A server-class host defaults to a window that holds an agent client's
        # first turn. core.system cannot import the app-layer constant, so this
        # test is what keeps the two values equal.
        from lilbee.app.agent_configs.window import AGENT_CHAT_CTX_FLOOR

        assert AGENT_CHAT_CTX_FLOOR == 65536
        assert chat_ctx_target_for_total_bytes(_gib(128)) == AGENT_CHAT_CTX_FLOOR
        assert chat_ctx_target_for_total_bytes(_gib(251)) == AGENT_CHAT_CTX_FLOOR
        assert chat_ctx_target_for_total_bytes(_gib(512)) == AGENT_CHAT_CTX_FLOOR

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
    @pytest.mark.skipif(sys.platform == "win32", reason="fd redirection is a win32 no-op")
    def test_fd2_points_at_devnull_inside_then_restores(self):
        devnull_stat = os.stat(os.devnull)
        with stderr_suppressed():
            inside = os.fstat(2)
        # Inside the block fd 2 is the null device...
        assert (inside.st_dev, inside.st_ino) == (devnull_stat.st_dev, devnull_stat.st_ino)
        # ...and afterwards fd 2 is restored to a valid descriptor (no OSError).
        os.fstat(2)

    @pytest.mark.skipif(sys.platform == "win32", reason="fd redirection is a win32 no-op")
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


def test_stderr_suppressed_can_nest():
    """A suppressed block can re-enter this, directly or via a native helper
    that wraps its own stderr. A plain Lock self-deadlocked there."""
    from lilbee.core.system import stderr_suppressed

    with stderr_suppressed(), stderr_suppressed():
        pass


class TestCgroupCappedMemory:
    """Host introspection answers for this process, not for the machine."""

    def test_the_ctx_target_tiers_on_the_container_cap(self, monkeypatch, tmp_path):
        # A 4 GiB container on a 64 GiB host. Tiered against the host it asks for a
        # 24576-token window nothing in the container can back.
        from lilbee.core import system as sys_mod

        (tmp_path / "memory.max").write_text(f"{4 * 1024**3}\n")
        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )
        assert sys_mod.scaled_chat_ctx_target_default() == 8192

    def test_an_uncapped_host_still_tiers_on_its_own_ram(self, monkeypatch, tmp_path):
        from lilbee.core import system as sys_mod

        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path / "absent")
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )
        assert sys_mod.scaled_chat_ctx_target_default() == 24576

    def test_a_v1_limit_and_usage_are_read_too(self, monkeypatch, tmp_path):
        from lilbee.core import system as sys_mod

        v1 = tmp_path / "memory"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text(f"{4 * 1024**3}\n")
        (v1 / "memory.usage_in_bytes").write_text(f"{1024**3}\n")
        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path)
        assert sys_mod.cgroup_memory_limit() == 4 * 1024**3
        assert sys_mod.cgroup_memory_used() == 1024**3

    def test_unlimited_is_reported_as_no_limit(self, monkeypatch, tmp_path):
        from lilbee.core import system as sys_mod

        (tmp_path / "memory.max").write_text("max\n")
        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path)
        assert sys_mod.cgroup_memory_limit() is None

    def test_an_unreadable_limit_is_reported_as_no_limit(self, monkeypatch, tmp_path):
        from lilbee.core import system as sys_mod

        (tmp_path / "memory.max").write_text("not-a-number\n")
        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path)
        assert sys_mod.cgroup_memory_limit() is None

    def test_absent_cgroup_files_report_nothing(self, monkeypatch, tmp_path):
        from lilbee.core import system as sys_mod

        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path / "absent")
        assert sys_mod.cgroup_memory_limit() is None
        assert sys_mod.cgroup_memory_used() is None

    def test_a_v1_sentinel_limit_does_not_shrink_the_ctx_target(self, monkeypatch, tmp_path):
        # cgroup v1 spells unlimited as a near-int64 sentinel rather than a word.
        from lilbee.core import system as sys_mod

        v1 = tmp_path / "memory"
        v1.mkdir()
        (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
        monkeypatch.setattr(sys_mod, "_CGROUP_ROOT", tmp_path)
        monkeypatch.setattr(
            "psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3),
        )
        assert sys_mod.scaled_chat_ctx_target_default() == 24576


class TestExecutableDiscovery:
    """PATH plus the per-user install dirs a GUI-spawned server never inherits."""

    def test_search_path_keeps_the_inherited_path_first(self, monkeypatch):
        monkeypatch.setenv("PATH", "/inherited/bin")
        entries = executable_search_path().split(os.pathsep)
        assert entries[0] == "/inherited/bin"

    def test_search_path_adds_the_well_known_unix_dirs(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("PATH", "/usr/bin")
        entries = executable_search_path().split(os.pathsep)
        # Compare through Path so the host OS's separator applies: the CI
        # matrix runs this darwin-branch test on Windows too.
        assert str(Path("/opt/homebrew/bin")) in entries
        assert str(Path("/usr/local/bin")) in entries
        assert str(Path("~/.local/bin").expanduser()) in entries

    def test_search_path_adds_the_well_known_windows_dirs(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("PATH", "C:/Windows")
        entries = executable_search_path().split(os.pathsep)
        assert str(Path("~/AppData/Roaming/npm").expanduser()) in entries
        assert "/opt/homebrew/bin" not in entries

    def test_search_path_drops_an_empty_path_var(self, monkeypatch):
        monkeypatch.setenv("PATH", "")
        assert "" not in executable_search_path().split(os.pathsep)

    def test_find_executable_returns_the_resolved_path(self, tmp_path, monkeypatch):
        # The .exe suffix keeps Windows honest: shutil.which only matches
        # names carrying a PATHEXT extension there, and POSIX matches the
        # literal name either way.
        binary = tmp_path / "somecli.exe"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert find_executable("somecli.exe") == str(binary)

    def test_find_executable_returns_none_when_absent(self, monkeypatch):
        monkeypatch.setenv("PATH", "")
        assert find_executable("definitely-not-installed-xyz") is None

    def test_find_executable_searches_the_extra_dirs(self, tmp_path, monkeypatch):
        """The whole point: a binary outside PATH is still found."""
        extra = tmp_path / "extra"
        extra.mkdir()
        binary = extra / "othercli.exe"
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(0o755)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(system_mod, "_UNIX_BIN_DIRS", (str(extra),))
        monkeypatch.setattr(sys, "platform", "darwin")
        assert find_executable("othercli.exe") == str(binary)


_LSTAT = os.lstat


def _lstat_ignoring_case(path: str) -> os.stat_result:
    """``os.lstat`` as a disk that ignores letter case answers it."""
    try:
        return _LSTAT(path)
    except OSError:
        folder, name = os.path.split(path)
        listed = next((n for n in os.listdir(folder) if n.casefold() == name.casefold()), name)
        return _LSTAT(os.path.join(folder, listed))


class TestDiskPath:
    """The spelling of a path comes from the listing of each folder, never from folding it."""

    @pytest.fixture
    def shelf(self, tmp_path) -> Path:
        """The real path of a folder that holds ``Notes/plan.md``."""
        real = Path(os.path.realpath(tmp_path)) / "shelf"
        (real / "Notes").mkdir(parents=True)
        (real / "Notes" / "plan.md").write_text("# Plan\n", encoding="utf-8")
        return real

    def test_a_path_spelled_as_listed_comes_back_unchanged(self, shelf):
        assert disk_path(shelf / "Notes" / "plan.md") == shelf / "Notes" / "plan.md"

    def test_the_spelling_agrees_with_what_this_disk_says(self, shelf):
        typed = shelf / "NOTES" / "plan.md"
        one_entry = typed.exists()

        spelled = disk_path(typed)

        assert spelled == (shelf / "Notes" / "plan.md" if one_entry else typed)

    def test_another_unicode_form_agrees_with_what_this_disk_says(self, shelf):
        listed = unicodedata.normalize("NFC", "Ärger")
        (shelf / listed).mkdir()
        typed = shelf / unicodedata.normalize("NFD", listed)
        one_entry = typed.exists()

        spelled = disk_path(typed)

        assert spelled == (shelf / listed if one_entry else typed)
        assert spelled.name == (listed if one_entry else typed.name)

    def test_on_a_disk_that_ignores_case_the_listed_name_wins(self, shelf):
        with mock.patch.object(os, "lstat", _lstat_ignoring_case):
            assert _Folder(str(shelf)).listed_name("NOTES") == "Notes"
            assert disk_path(shelf / "NOTES" / "PLAN.MD") == shelf / "Notes" / "plan.md"

    def test_a_name_that_is_not_ascii_takes_the_listed_spelling(self, shelf):
        (shelf / "Ärger Übung").mkdir()
        with mock.patch.object(os, "lstat", _lstat_ignoring_case):
            assert _Folder(str(shelf)).listed_name("ärger übung") == "Ärger Übung"

    def test_two_listed_names_that_differ_by_case_stay_two(self, shelf):
        with mock.patch.object(os, "listdir", return_value=["Notes", "NOTES"]):
            assert _Folder(str(shelf)).listed_name("NOTES") == "NOTES"
            assert _Folder(str(shelf)).listed_name("Notes") == "Notes"

    def test_a_link_beside_the_folder_is_not_taken_for_it(self, shelf):
        (shelf / "alias").symlink_to(shelf / "Notes", target_is_directory=True)
        with (
            mock.patch.object(os, "listdir", return_value=["alias", "Notes"]),
            mock.patch.object(os, "lstat", _lstat_ignoring_case),
        ):
            assert _Folder(str(shelf)).listed_name("NOTES") == "Notes"

    def test_a_link_gives_the_path_it_points_to(self, shelf, tmp_path):
        link = tmp_path / "link"
        link.symlink_to(shelf / "Notes", target_is_directory=True)
        assert disk_path(link / "plan.md") == shelf / "Notes" / "plan.md"

    def test_a_part_that_does_not_exist_keeps_its_spelling(self, shelf):
        assert disk_path(shelf / "Notes" / "Missing" / "New.md") == (
            shelf / "Notes" / "Missing" / "New.md"
        )

    def test_a_folder_that_cannot_be_listed_keeps_the_typed_name(self, shelf):
        with mock.patch.object(os, "listdir", side_effect=PermissionError(13, "denied")):
            assert _Folder(str(shelf)).listed_name("NOTES") == "NOTES"

    def test_the_anchor_alone_is_the_anchor(self, shelf):
        assert disk_path(Path(shelf.anchor)) == Path(shelf.anchor)

    def test_one_spelling_reads_a_folder_one_time_for_many_paths(self, shelf):
        (shelf / "Drafts").mkdir()
        read: list[str] = []
        listdir = os.listdir

        def _listing(folder: str) -> list[str]:
            read.append(folder)
            return listdir(folder)

        spelling = DiskSpelling()
        with mock.patch.object(os, "listdir", _listing):
            spelled = [spelling.of(shelf / name) for name in ("Notes", "Drafts", "Notes", "New")]

        assert spelled == [shelf / "Notes", shelf / "Drafts", shelf / "Notes", shelf / "New"]
        assert read.count(str(shelf)) == 1

    def test_one_spelling_resolves_a_path_one_time(self, shelf):
        resolved: list[Path] = []
        realpath = os.path.realpath

        def _realpath(path: Path) -> str:
            resolved.append(path)
            return realpath(path)

        spelling = DiskSpelling()
        with mock.patch.object(os.path, "realpath", _realpath):
            spelled = [spelling.of(shelf / "Notes"), spelling.of(shelf / "Notes")]

        assert spelled == [shelf / "Notes", shelf / "Notes"]
        assert resolved == [shelf / "Notes"]

    def test_one_spelling_asks_about_each_listed_entry_one_time(self, shelf):
        (shelf / "Drafts").mkdir()
        asked: list[str] = []

        def _lstat(path: str) -> os.stat_result:
            asked.append(path)
            return _lstat_ignoring_case(path)

        def _keeps_typed_case(path: Path) -> str:
            """Resolve as a POSIX disk does; Windows would spell the parts as listed."""
            return os.path.abspath(path)

        spelling = DiskSpelling()
        with (
            mock.patch.object(os.path, "realpath", _keeps_typed_case),
            mock.patch.object(os, "lstat", _lstat),
        ):
            spelled = [spelling.of(shelf / name) for name in ("NOTES", "DRAFTS")]

        assert spelled == [shelf / "Notes", shelf / "Drafts"]
        assert asked.count(str(shelf / "Notes")) == 1
        assert asked.count(str(shelf / "Drafts")) == 1

    def test_two_names_of_one_file_give_the_name_listed_first(self, shelf):
        os.link(shelf / "Notes" / "plan.md", shelf / "Notes" / "copy.md")
        with (
            mock.patch.object(os, "listdir", return_value=["plan.md", "copy.md"]),
            mock.patch.object(os, "lstat", _lstat_ignoring_case),
        ):
            assert _Folder(str(shelf / "Notes")).listed_name("COPY.MD") == "plan.md"
