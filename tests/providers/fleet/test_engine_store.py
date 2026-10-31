"""Tests for the side by side engine store: layout, record, lock, stage, listing."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_store import (
    RECORD_NAME,
    ChecksumError,
    EngineConflictError,
    EngineStore,
    LicenceRefusedError,
    PinLock,
    UnsafeWheelError,
    licence_permitted,
)


def make_wheel(path: Path, licence: str | None = "MIT") -> str:
    meta = "Metadata-Version: 2.4\nName: lilbee-engine-fx\nVersion: 1\n"
    if licence:
        meta += f"License-Expression: {licence}\n"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("lilbee_engine_fx/bin/server")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "#!/bin/sh\n")
        plain = zipfile.ZipInfo("lilbee_engine_fx/data.txt")
        plain.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(plain, "x")
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", meta)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(sha: str, version: str = "1", licences: tuple[str, ...] = ("MIT",)) -> PinLock:
    return PinLock("fx", version, "cpu", {}, licences, sha)


@pytest.fixture
def wheel(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "fx.whl"
    return path, make_wheel(path)


@pytest.fixture
def store(tmp_path: Path) -> EngineStore:
    return EngineStore(tmp_path / "engines")


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_install_names_directory_and_lists_it(store: EngineStore, wheel: tuple[Path, str]) -> None:
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    assert build == store.root / "fx" / "1-cpu"
    server = build / "lilbee_engine_fx" / "bin" / "server"
    assert server.read_text(encoding="utf-8") == "#!/bin/sh\n"
    [found] = store.installed()
    assert (found.engine, found.version, found.backend, found.path) == ("fx", "1", "cpu", build)
    assert found.sha256 == sha
    assert found.licences == ("MIT",)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bits")
def test_unpack_keeps_only_the_execute_bit_the_wheel_sets(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    assert os.access(build / "lilbee_engine_fx" / "bin" / "server", os.X_OK)
    assert not os.access(build / "lilbee_engine_fx" / "data.txt", os.X_OK)


def test_builds_of_one_engine_list_side_by_side(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.install_from_wheel(pin(sha, "1"), path)
    store.install_from_wheel(pin(sha, "2"), path)
    other = PinLock("fx", "1", "cu125", {}, ("MIT",), sha)
    store.install_from_wheel(other, path)
    assert [(e.version, e.backend) for e in store.installed()] == [
        ("1", "cpu"),
        ("1", "cu125"),
        ("2", "cpu"),
    ]


def test_listing_an_empty_or_missing_root_is_empty(store: EngineStore) -> None:
    assert store.installed() == []
    store.root.mkdir()
    assert store.installed() == []


def test_second_install_returns_the_build_without_unpacking(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    first = store.install_from_wheel(pin(sha), path)
    with mock.patch.object(engine_store, "_unpack") as unpack:
        second = store.install_from_wheel(pin(sha), path)
    assert second == first
    unpack.assert_not_called()


_CHILD = textwrap.dedent(
    """
    import os, sys, time, uuid
    from pathlib import Path
    from lilbee.providers.fleet import engine_store as es

    root, wheel, sha, marks, start = sys.argv[1:]
    marks = Path(marks)
    real_unpack, real_check = es._unpack, es._check_wheel

    def check(lock, wheel):
        (marks / f"tried-{uuid.uuid4().hex}").write_text("x")
        real_check(lock, wheel)

    def unpack(w, s):
        (marks / f"unpack-{uuid.uuid4().hex}").write_text("x")
        try:
            os.close(os.open(marks / "holder", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            real_unpack(w, s)
            return
        deadline = time.time() + 30
        while len(list(marks.glob("tried-*"))) < 2 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(1)
        real_unpack(w, s)

    es._unpack, es._check_wheel = unpack, check
    while time.time() < float(start):
        time.sleep(0.01)
    pin = es.PinLock("fx", "1", "cpu", {}, ("MIT",), sha)
    es.EngineStore(Path(root)).install_from_wheel(pin, Path(wheel))
    """
)


def test_installer_processes_unpack_the_build_once_and_leave_no_stage(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path
) -> None:
    path, sha = wheel
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    marks = tmp_path / "marks"
    marks.mkdir()
    start = str(time.time() + 2)
    args = [sys.executable, str(script), str(store.root), str(path), sha, str(marks), start]
    children = [subprocess.Popen(args) for _ in range(3)]
    assert [c.wait(timeout=120) for c in children] == [0, 0, 0]
    assert len(list(marks.glob("tried-*"))) == 3
    assert len(list(marks.glob("unpack-*"))) == 1
    assert names(store.root / "fx") == ["1-cpu"]
    assert len(store.installed()) == 1


def test_one_lock_file_per_engine_outside_the_engine_directory(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.install_from_wheel(pin(sha), path)
    assert (store.root / ".locks" / "fx.lock").is_file()
    assert names(store.root / "fx") == ["1-cpu"]


def test_lock_is_taken_around_the_unpack(store: EngineStore, wheel: tuple[Path, str]) -> None:
    path, sha = wheel
    held: list[bool] = []
    real = engine_store.file_lock

    def spy(lock_path: Path, timeout: float):  # type: ignore[no-untyped-def]
        held.append(True)
        return real(lock_path, timeout)

    with mock.patch.object(engine_store, "file_lock", spy):
        store.install_from_wheel(pin(sha), path)
    assert held == [True]


@pytest.mark.parametrize(
    ("licence", "good", "expect"),
    [
        ("MIT", True, "installed"),
        ("Apache-2.0", True, "installed"),
        ("BSD-3-Clause", True, "installed"),
        ("MIT OR BSD-2-Clause", True, "installed"),
        ("GPL-3.0-only", True, "refused"),
        ("AGPL-3.0-or-later", True, "refused"),
        ("MIT OR GPL-2.0-only", True, "refused"),
        (None, True, "refused"),
        ("MIT", False, "refused"),
        ("GPL-3.0-only", False, "refused"),
    ],
)
def test_gate_refuses_and_leaves_nothing(
    tmp_path: Path, licence: str | None, good: bool, expect: str
) -> None:
    path = tmp_path / "w.whl"
    sha = make_wheel(path, licence)
    store = EngineStore(tmp_path / "engines")
    lock = pin(sha if good else "0" * 64)
    if expect == "installed":
        store.install_from_wheel(lock, path)
        assert len(store.installed()) == 1
        return
    with pytest.raises((ChecksumError, LicenceRefusedError)):
        store.install_from_wheel(lock, path)
    assert not store.root.exists()


def test_checksum_error_names_engine_and_version(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    with pytest.raises(ChecksumError, match="engine fx 1 does not match"):
        store.install_from_wheel(pin("0" * 64), wheel[0])


def test_lock_licence_outside_the_list_is_refused_before_the_wheel_is_read(
    store: EngineStore, tmp_path: Path
) -> None:
    with pytest.raises(LicenceRefusedError, match=r"licence 'GPL-3\.0-only'"):
        store.install_from_wheel(pin("0" * 64, licences=("GPL-3.0-only",)), tmp_path / "absent.whl")


def test_lock_without_licences_is_refused(store: EngineStore, tmp_path: Path) -> None:
    with pytest.raises(LicenceRefusedError, match="lists no licence"):
        store.install_from_wheel(pin("0" * 64, licences=()), tmp_path / "absent.whl")


def test_wheel_without_metadata_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "bare.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/b", "x")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(LicenceRefusedError, match="declares the licence ''"):
        store.install_from_wheel(pin(sha), path)


@pytest.mark.parametrize(
    ("expression", "ok"),
    [
        ("MIT", True),
        (" MIT ", True),
        ("MIT AND Apache-2.0", True),
        ("", False),
        ("MIT OR GPL", False),
    ],
)
def test_licence_permitted(expression: str, ok: bool) -> None:
    assert licence_permitted(expression) is ok


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "", "a b"])
def test_pin_names_cannot_leave_the_engine_directory(bad: str) -> None:
    with pytest.raises(ValueError, match="cannot name an engine directory"):
        PinLock(bad, "1", "cpu", {}, ("MIT",), "a")
    with pytest.raises(ValueError, match="cannot name an engine directory"):
        PinLock("fx", bad, "cpu", {}, ("MIT",), "a")


# Crash steps. A raise runs the cleanup; a kill does not, so each step is tested both ways.


def test_failure_while_unpacking_leaves_nothing_listed(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    with (
        mock.patch.object(engine_store, "_unpack", side_effect=OSError("disk")),
        pytest.raises(OSError, match="disk"),
    ):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []
    assert names(store.root / "fx") == []


def test_failure_before_the_record_leaves_nothing_listed(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    with (
        mock.patch.object(EngineStore, "_write_record", side_effect=OSError("disk")),
        pytest.raises(OSError, match="disk"),
    ):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []
    assert names(store.root / "fx") == []


def test_failure_at_the_rename_leaves_nothing_listed(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    with (
        mock.patch.object(Path, "rename", side_effect=PermissionError("held")),
        pytest.raises(PermissionError, match="held"),
    ):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []
    assert names(store.root / "fx") == []


def test_killed_after_unpack_is_not_listed_and_a_retry_installs(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    stage = store.root / "fx" / ".stage-dead"
    (stage / "lilbee_engine_fx" / "bin").mkdir(parents=True)
    assert store.installed() == []
    store.install_from_wheel(pin(sha), path)
    assert [e.version for e in store.installed()] == ["1"]


def test_killed_after_the_record_is_not_listed(store: EngineStore, wheel: tuple[Path, str]) -> None:
    _, sha = wheel
    stage = store.root / "fx" / ".stage-dead"
    stage.mkdir(parents=True)
    EngineStore._write_record(stage, pin(sha))
    assert (stage / RECORD_NAME).is_file()
    assert store.installed() == []


def test_build_directory_without_a_record_is_not_listed_and_is_replaced(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    leftover = store.root / "fx" / "1-cpu"
    leftover.mkdir(parents=True)
    (leftover / "partial.bin").write_text("x", encoding="utf-8")
    assert store.installed() == []
    store.install_from_wheel(pin(sha), path)
    assert [e.version for e in store.installed()] == ["1"]
    assert not (leftover / "partial.bin").exists()


@pytest.mark.parametrize(
    "content",
    [
        '{"engine": "fx"',
        "[]",
        '{"engine": "fx"}',
        '{"engine":"fx","version":"1","backend":"cpu",'
        '"sha256":"a","licences":5,"installed_at":"t","binary":null}',
    ],
)
def test_unreadable_record_is_not_listed(store: EngineStore, content: str) -> None:
    build = store.root / "fx" / "1-cpu"
    build.mkdir(parents=True)
    (build / RECORD_NAME).write_text(content, encoding="utf-8")
    assert store.installed() == []


def test_record_that_disagrees_with_its_directory_is_not_listed(store: EngineStore) -> None:
    build = store.root / "fx" / "9-cpu"
    build.mkdir(parents=True)
    record = {
        "engine": "fx",
        "version": "1",
        "backend": "cpu",
        "sha256": "a",
        "licences": ["MIT"],
        "installed_at": "t",
        "binary": None,
    }
    (build / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    assert store.installed() == []


def test_hidden_directories_and_loose_files_are_not_listed(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.install_from_wheel(pin(sha), path)
    (store.root / "packs").mkdir()
    (store.root / "note.txt").write_text("x", encoding="utf-8")
    (store.root / "fx" / "stray.txt").write_text("x", encoding="utf-8")
    (store.root / "fx" / ".trash-1").mkdir()
    assert [e.version for e in store.installed()] == ["1"]


def evil_wheel(path: Path, entry: str) -> str:
    make_wheel(path)
    with zipfile.ZipFile(path, "a") as archive:
        info = zipfile.ZipInfo(entry)
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "x")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
@pytest.mark.parametrize("form", ["relative", "absolute"])
def test_wheel_entry_outside_the_stage_is_refused_and_touches_nothing(
    store: EngineStore, tmp_path: Path, form: str
) -> None:
    victim = store.root / "fx" / "victim"
    victim.parent.mkdir(parents=True)
    victim.write_text("keep", encoding="utf-8")
    victim.chmod(0o644)
    entry = "../victim" if form == "relative" else str(victim)
    path = tmp_path / "evil.whl"
    sha = evil_wheel(path, entry)
    with pytest.raises(UnsafeWheelError, match="points outside the engine directory"):
        store.install_from_wheel(pin(sha), path)
    assert stat.S_IMODE(victim.stat().st_mode) == 0o644
    assert names(store.root / "fx") == ["victim"]


@pytest.mark.parametrize("entry", ["a/../../x", "\\x", "C:/x", "a\\..\\..\\x"])
def test_other_escaping_entry_forms_are_refused(
    store: EngineStore, tmp_path: Path, entry: str
) -> None:
    path = tmp_path / "evil.whl"
    sha = evil_wheel(path, entry)
    with pytest.raises(UnsafeWheelError):
        store.install_from_wheel(pin(sha), path)
    assert not store.root.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bits")
def test_build_whose_binary_lost_its_execute_bit_is_not_installed_and_is_replaced(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    binary = build / "lilbee_engine_fx" / "bin" / "server"
    binary.chmod(0o644)
    assert store.installed() == []
    store.install_from_wheel(pin(sha), path)
    assert os.access(binary, os.X_OK)
    assert len(store.installed()) == 1


def test_build_whose_binary_is_gone_is_not_installed_and_is_replaced(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    binary = build / "lilbee_engine_fx" / "bin" / "server"
    assert store.installed()[0].binary == binary
    binary.unlink()
    assert store.installed() == []
    store.install_from_wheel(pin(sha), path)
    assert binary.is_file()
    assert len(store.installed()) == 1


def test_wheel_without_a_bin_tool_records_no_binary(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "lib.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("p/data.txt", "x")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store.install_from_wheel(pin(sha), path)
    assert store.installed()[0].binary is None


def test_second_wheel_with_another_sha_is_refused_and_keeps_the_build(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path
) -> None:
    path, sha = wheel
    store.install_from_wheel(pin(sha), path)
    other = tmp_path / "other.whl"
    other_sha = evil_wheel(other, "lilbee_engine_fx/extra.txt")
    with pytest.raises(EngineConflictError, match="different wheel than its pin"):
        store.install_from_wheel(pin(other_sha), other)
    assert store.installed()[0].sha256 == sha
    assert not (store.root / "fx" / "1-cpu" / "lilbee_engine_fx" / "extra.txt").exists()
    assert names(store.root / "fx") == ["1-cpu"]


def rewrite_record(build: Path, binary: str) -> None:
    record = json.loads((build / RECORD_NAME).read_text(encoding="utf-8"))
    record["binary"] = binary
    (build / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("form", ["relative", "absolute", "symlink"])
def test_recorded_binary_outside_the_build_is_not_installed(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path, form: str
) -> None:
    if form == "symlink" and sys.platform == "win32":
        pytest.skip("symlinks need privileges on Windows")
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    outside = tmp_path / "outside_file"
    outside.write_text("x", encoding="utf-8")
    outside.chmod(0o755)
    if form == "symlink":
        (build / "link").symlink_to(outside)
    recorded = {"relative": "../../../outside_file", "absolute": str(outside), "symlink": "link"}
    rewrite_record(build, recorded[form])
    assert store.installed() == []


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_wheel_whose_tool_is_a_symlink_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "link.whl"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("p/bin/tool")
        info.external_attr = (stat.S_IFLNK | 0o755) << 16
        archive.writestr(info, "/bin/sh")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(UnsafeWheelError, match="as a symbolic link"):
        store.install_from_wheel(pin(sha), path)
    assert not store.root.exists()
