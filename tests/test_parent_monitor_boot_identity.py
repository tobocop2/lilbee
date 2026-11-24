"""The per-boot identity each platform gives, compared by equality, not derived from the clock."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

from lilbee import parent_monitor as pm


class _Libc:
    def __init__(self, *, result: int, text: bytes) -> None:
        self._result = result
        self._text = text

    def sysctlbyname(self, key: bytes, buffer, _size, _new, _newlen) -> int:
        assert key == b"kern.bootsessionuuid"
        buffer.value = self._text
        return self._result


class TestBootIdentity:
    def test_linux_reads_the_boot_id(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        boot_id = tmp_path / "sys/kernel/random/boot_id"
        boot_id.parent.mkdir(parents=True)
        boot_id.write_text("boot-A\n", encoding="utf-8")
        monkeypatch.setattr(pm, "_PROC_ROOT", tmp_path)
        monkeypatch.setattr(sys, "platform", "linux")
        assert pm.boot_identity() == "boot-A"

    def test_macos_reads_the_boot_session_uuid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(ctypes, "CDLL", lambda _n: _Libc(result=0, text=b"UUID-1"))
        assert pm.boot_identity() == "UUID-1"

    @pytest.mark.parametrize(
        ("result", "text"), [(-1, b"UUID-1"), (0, b""), (0, b"\xff\xfe")], ids=["rc", "empty", "bytes"]
    )
    def test_macos_read_that_fails_gives_none(
        self, monkeypatch: pytest.MonkeyPatch, result: int, text: bytes
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(ctypes, "CDLL", lambda _n: _Libc(result=result, text=text))
        assert pm.boot_identity() is None

    def test_macos_without_libc_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _missing(_name: object) -> object:
            raise OSError("no libc")

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(ctypes, "CDLL", _missing)
        assert pm.boot_identity() is None

    def test_windows_reads_the_boot_counter_at_the_shared_data_offset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        addresses: list[int] = []

        def _read(address: int) -> int:
            addresses.append(address)
            return 41

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(pm, "_read_u32", _read)
        assert pm.boot_identity() == "41"
        assert addresses == [0x7FFE02C4]

    def test_windows_read_that_fails_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail(_address: int) -> int:
            raise OSError("access violation")

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(pm, "_read_u32", _fail)
        assert pm.boot_identity() is None

    def test_read_u32_reads_the_value_at_an_address(self) -> None:
        cell = ctypes.c_uint32(7)
        assert pm._read_u32(ctypes.addressof(cell)) == 7

    def test_another_platform_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "freebsd14")
        assert pm.boot_identity() is None
