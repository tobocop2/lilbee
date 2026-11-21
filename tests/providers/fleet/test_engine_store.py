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


# --- package route -----------------------------------------------------------------------


def extract_into_target(wheel_path: Path):
    """A stand-in for the installer: unpack the wheel into the folder named by --target."""
    seen: list[list[str]] = []

    def run(argv: list[str], requirements: str) -> None:
        seen.append(argv)
        with zipfile.ZipFile(wheel_path) as archive:
            archive.extractall(argv[argv.index("--target") + 1])
        for tool in Path(argv[argv.index("--target") + 1]).glob("*/bin/*"):
            tool.chmod(0o755)

    return run, seen


def test_package_route_installs_the_staged_tree(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    run, seen = extract_into_target(path)
    build = store.install_via_package(pin(sha), run=run)
    assert build == store.root / "fx" / "1-cpu"
    assert [b.binary for b in store.installed()] == [build / "lilbee_engine_fx" / "bin" / "server"]
    assert "--require-hashes" in seen[0]
    assert names(store.root / "fx") == ["1-cpu"]


def test_package_route_skips_the_installer_for_a_complete_build(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    store.install_from_wheel(pin(sha), path)
    run = mock.Mock()
    assert store.install_via_package(pin(sha), run=run) == store.root / "fx" / "1-cpu"
    run.assert_not_called()


def test_package_route_refuses_a_licence_before_it_runs_the_installer(
    store: EngineStore,
) -> None:
    run = mock.Mock()
    with pytest.raises(LicenceRefusedError, match=r"GPL-3\.0-only"):
        store.install_via_package(pin("0" * 64, licences=("GPL-3.0-only",)), run=run)
    run.assert_not_called()


def test_package_route_refuses_a_wheel_licence_the_pin_does_not_show(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "gpl.whl"
    sha = make_wheel(path, licence="GPL-3.0-only")
    run, _ = extract_into_target(path)
    with pytest.raises(LicenceRefusedError, match=r"GPL-3\.0-only"):
        store.install_via_package(pin(sha), run=run)
    assert names(store.root / "fx") == []


def test_package_route_refuses_a_wheel_without_a_licence(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "none.whl"
    sha = make_wheel(path, licence=None)
    run, _ = extract_into_target(path)
    with pytest.raises(LicenceRefusedError, match="declares the licence ''"):
        store.install_via_package(pin(sha), run=run)
    assert names(store.root / "fx") == []


def test_package_route_refuses_a_link_that_leaves_the_stage(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path
) -> None:
    path, sha = wheel
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    inner, _ = extract_into_target(path)

    def run(argv: list[str], requirements: str) -> None:
        inner(argv, requirements)
        (Path(argv[argv.index("--target") + 1]) / "escape").symlink_to(outside)

    with pytest.raises(UnsafeWheelError, match="escape"):
        store.install_via_package(pin(sha), run=run)
    assert names(store.root / "fx") == []
    assert outside.read_text(encoding="utf-8") == "secret"


def test_package_route_refuses_a_tool_that_is_a_link(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    inner, _ = extract_into_target(path)

    def run(argv: list[str], requirements: str) -> None:
        inner(argv, requirements)
        stage = Path(argv[argv.index("--target") + 1])
        tool = stage / "lilbee_engine_fx" / "bin" / "server"
        tool.unlink()
        tool.symlink_to(stage / "lilbee_engine_fx" / "data.txt")

    with pytest.raises(UnsafeWheelError, match="symbolic link"):
        store.install_via_package(pin(sha), run=run)
    assert names(store.root / "fx") == []


def test_package_route_leaves_nothing_when_the_installer_fails(store: EngineStore) -> None:
    def run(argv: list[str], requirements: str) -> None:
        raise engine_store.PackageInstallError("The engine installer failed: no index")

    with pytest.raises(engine_store.PackageInstallError, match="no index"):
        store.install_via_package(pin("0" * 64), run=run)
    assert names(store.root / "fx") == []


def test_installer_runner_passes_the_requirement_on_standard_input(tmp_path: Path) -> None:
    out = tmp_path / "stdin.txt"
    script = f"import sys, pathlib; pathlib.Path({str(out)!r}).write_text(sys.stdin.read())"
    engine_store._run_installer([sys.executable, "-c", script], "pkg==1 --hash=sha256:ab\n")
    assert out.read_text(encoding="utf-8") == "pkg==1 --hash=sha256:ab\n"


def test_installer_runner_reports_the_last_line_of_a_failure() -> None:
    script = "import sys; sys.stderr.write('warn\\nno matching index\\n'); sys.exit(2)"
    with pytest.raises(engine_store.PackageInstallError, match="no matching index"):
        engine_store._run_installer([sys.executable, "-c", script], "")


def test_installer_runner_reports_a_failure_without_output() -> None:
    with pytest.raises(engine_store.PackageInstallError, match="no output"):
        engine_store._run_installer([sys.executable, "-c", "import sys; sys.exit(1)"], "")


@pytest.mark.parametrize(("uv", "want"), [("/bin/uv", "uv"), (None, "pip")])
def test_installer_prefers_uv_then_pip(uv: str | None, want: str) -> None:
    with mock.patch.object(engine_store.shutil, "which", return_value=uv):
        assert engine_store.installer() == want


# --- removal, prune, sweep ---------------------------------------------------------------


def installed_store(store: EngineStore, tmp_path: Path, versions: tuple[str, ...] = ("1",)) -> str:
    path = tmp_path / "fx.whl"
    sha = make_wheel(path)
    for version in versions:
        store.install_from_wheel(pin(sha, version=version), path)
    return sha


def test_remove_refuses_a_build_in_use_and_keeps_it(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path)
    with pytest.raises(engine_store.EngineInUseError, match="lilbee engine stop"):
        store.remove("fx", "1", "cpu", in_use=lambda *_: True)
    assert (store.root / "fx" / "1-cpu" / RECORD_NAME).is_file()


def test_remove_deletes_an_idle_build_with_a_canary_warning(
    store: EngineStore, tmp_path: Path
) -> None:
    installed_store(store, tmp_path)
    seen: list[tuple[str, str, str]] = []
    result = store.remove("fx", "1", "cpu", in_use=lambda e, v, b: seen.append((e, v, b)) or False)
    assert seen == [("fx", "1", "cpu")]
    assert result.canary_warning is True
    assert names(store.root / "fx") == []
    assert store.installed() == []


def test_remove_leaves_a_locked_folder_in_trash_and_the_sweep_takes_it(
    store: EngineStore, tmp_path: Path
) -> None:
    installed_store(store, tmp_path)
    with mock.patch.object(engine_store.shutil, "rmtree"):
        store.remove("fx", "1", "cpu", in_use=lambda *_: False)
    left = names(store.root / "fx")
    assert len(left) == 1
    assert left[0].startswith(".trash-")
    assert store.installed() == []
    assert [p.name for p in store.sweep()] == left
    assert names(store.root / "fx") == []


def test_remove_refuses_the_bundled_engine(store: EngineStore) -> None:
    with pytest.raises(engine_store.BundledEngineError, match="inside the executable"):
        store.remove("llama.cpp", "bundled", "bundled", in_use=lambda *_: False)


def test_remove_reports_a_build_that_is_not_installed(store: EngineStore) -> None:
    with pytest.raises(engine_store.EngineNotInstalledError, match="fx 7"):
        store.remove("fx", "7", "cpu", in_use=lambda *_: False)


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a b"])
def test_remove_refuses_a_name_that_could_leave_the_root(
    store: EngineStore, tmp_path: Path, bad: str
) -> None:
    victim = tmp_path / "engines" / "x"
    victim.mkdir(parents=True)
    with pytest.raises(engine_store.StorePathError):
        store.remove(bad, "1", "cpu", in_use=lambda *_: False)
    with pytest.raises(engine_store.StorePathError):
        store.remove("fx", bad, "cpu", in_use=lambda *_: False)
    assert victim.is_dir()


def test_remove_refuses_a_build_folder_that_is_a_link_out_of_the_root(
    store: EngineStore, tmp_path: Path
) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "file.txt").write_text("data", encoding="utf-8")
    (store.root / "fx").mkdir(parents=True)
    (store.root / "fx" / "1-cpu").symlink_to(outside, target_is_directory=True)
    with pytest.raises(engine_store.StorePathError, match="not inside"):
        store.remove("fx", "1", "cpu", in_use=lambda *_: False)
    assert (outside / "file.txt").read_text(encoding="utf-8") == "data"


def test_prune_keeps_the_release_pin_and_the_pin_in_use(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1", "2", "3"))
    gone = store.prune(keep={("fx", "3", "cpu")}, in_use=lambda e, v, b: v == "2")
    assert [p.name for p in gone] == ["1-cpu"]
    assert [b.version for b in store.installed()] == ["2", "3"]


def test_prune_skips_a_build_it_cannot_delete(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1",))
    with mock.patch.object(EngineStore, "remove", side_effect=OSError("busy")):
        assert store.prune(keep={("fx", "9", "cpu")}, in_use=lambda *_: False) == []
    assert [b.version for b in store.installed()] == ["1"]


_HOLD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from lilbee.core.security import file_lock

    with file_lock(Path(sys.argv[1]), 30):
        print("held", flush=True)
        sys.stdin.readline()
    """
)


def hold_lock(tmp_path: Path, lock_path: Path) -> subprocess.Popen[str]:
    script = tmp_path / "hold.py"
    script.write_text(_HOLD, encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(script), str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "held"
    return child


def release(child: subprocess.Popen[str]) -> None:
    assert child.stdin is not None
    child.stdin.write("\n")
    child.stdin.flush()
    child.wait(timeout=30)


def test_sweep_removes_leftovers_that_no_install_holds_however_new(
    store: EngineStore, tmp_path: Path
) -> None:
    installed_store(store, tmp_path)
    for leftover in (".stage-aa", ".trash-bb"):
        (store.root / "fx" / leftover).mkdir()
        (store.root / "fx" / leftover / "f").write_text("x", encoding="utf-8")
    (store.root / ".download-cc").mkdir()
    (store.root / "packs").mkdir()
    (store.root / "packs" / "fx-1-cpu.whl").write_text("pack", encoding="utf-8")
    gone = store.sweep()
    assert sorted(p.name for p in gone) == [".download-cc", ".stage-aa", ".trash-bb"]
    assert names(store.root / "fx") == ["1-cpu"]
    assert (store.root / "packs" / "fx-1-cpu.whl").is_file()
    assert len(store.installed()) == 1


def test_sweep_keeps_the_stage_of_an_install_that_holds_the_engine_lock(
    store: EngineStore, tmp_path: Path
) -> None:
    (store.root / "fx" / ".stage-live").mkdir(parents=True)
    (store.root / "other" / ".stage-dead").mkdir(parents=True)
    child = hold_lock(tmp_path, store.root / ".locks" / "fx")
    try:
        gone = store.sweep()
        assert [p.name for p in gone] == [".stage-dead"]
        assert (store.root / "fx" / ".stage-live").is_dir()
    finally:
        release(child)
    assert [p.name for p in store.sweep()] == [".stage-live"]


def test_sweep_keeps_the_download_folder_of_a_live_install(
    store: EngineStore, tmp_path: Path
) -> None:
    (store.root / ".download-live").mkdir(parents=True)
    (store.root / ".download-live" / "part").write_text("x", encoding="utf-8")
    child = hold_lock(tmp_path, store.root / ".locks" / ".download-live")
    try:
        assert store.sweep() == []
        assert (store.root / ".download-live" / "part").is_file()
    finally:
        release(child)
    assert [p.name for p in store.sweep()] == [".download-live"]
    assert not (store.root / ".locks" / ".download-live.lock").exists()


def test_sweep_ignores_a_root_that_does_not_exist(store: EngineStore) -> None:
    assert store.sweep() == []


def test_sweep_does_not_follow_a_stage_link_out_of_the_root(
    store: EngineStore, tmp_path: Path
) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "file.txt").write_text("data", encoding="utf-8")
    (store.root / "fx").mkdir(parents=True)
    (store.root / "fx" / ".stage-link").symlink_to(outside, target_is_directory=True)
    assert store.sweep() == []
    assert (outside / "file.txt").read_text(encoding="utf-8") == "data"


def test_sweep_does_not_follow_a_download_link_out_of_the_root(
    store: EngineStore, tmp_path: Path
) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "file.txt").write_text("data", encoding="utf-8")
    store.root.mkdir()
    (store.root / ".download-link").symlink_to(outside, target_is_directory=True)
    assert store.sweep() == []
    assert (outside / "file.txt").read_text(encoding="utf-8") == "data"


def test_sweep_does_not_follow_an_engine_link_out_of_the_root(
    store: EngineStore, tmp_path: Path
) -> None:
    outside = tmp_path / "keep"
    (outside / ".stage-x").mkdir(parents=True)
    (outside / ".stage-x" / "file.txt").write_text("data", encoding="utf-8")
    (outside / ".trash-y").mkdir()
    store.root.mkdir()
    (store.root / "fx").symlink_to(outside, target_is_directory=True)
    assert store.sweep() == []
    assert (outside / ".stage-x" / "file.txt").read_text(encoding="utf-8") == "data"
    assert (outside / ".trash-y").is_dir()


def test_sweep_skips_one_bad_link_and_clears_the_other_leftovers(
    store: EngineStore, tmp_path: Path
) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "file.txt").write_text("data", encoding="utf-8")
    engine = store.root / "fx"
    (engine / ".stage-real").mkdir(parents=True)
    (engine / ".trash-real").mkdir()
    (engine / ".stage-link").symlink_to(outside, target_is_directory=True)
    (engine / ".trash-link").symlink_to(outside, target_is_directory=True)
    gone = store.sweep()
    assert sorted(p.name for p in gone) == [".stage-real", ".trash-real"]
    assert names(engine) == [".stage-link", ".trash-link"]
    assert (outside / "file.txt").read_text(encoding="utf-8") == "data"


def test_prune_refuses_an_empty_keep_set(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1", "2"))
    with pytest.raises(ValueError, match="at least one build to keep"):
        store.prune(keep=set(), in_use=lambda *_: False)
    assert [b.version for b in store.installed()] == ["1", "2"]


def test_ensure_holds_a_download_lock_while_it_fetches(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    during: list[list[Path]] = []

    def fetch(lock: PinLock, dest: Path) -> Path:
        dest.mkdir(parents=True)
        during.append(store.sweep())
        return path

    store.ensure(pin(sha), fetch)
    assert during == [[]]
    assert names(store.root) == [".locks", "fx"]
    assert sorted(p.name for p in (store.root / ".locks").iterdir()) == ["fx.lock"]


def test_sweep_engines_ignores_a_relative_models_dir(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "engines" / "fx" / ".stage-x").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert engine_store.sweep_engines(Path("models")) == []
    assert (tmp_path / "engines" / "fx" / ".stage-x").is_dir()


def test_sweep_engines_sweeps_the_folder_beside_the_models(tmp_path: Path) -> None:
    (tmp_path / "engines" / "fx" / ".stage-x").mkdir(parents=True)
    gone = engine_store.sweep_engines(tmp_path / "models")
    assert [p.name for p in gone] == [".stage-x"]


def test_sweep_engines_is_a_no_op_with_nothing_installed(tmp_path: Path) -> None:
    assert engine_store.sweep_engines(tmp_path / "models") == []
    assert names(tmp_path) == []


def test_launch_uses_engine_reads_the_user_locks_of_the_machine_engine(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    assert engine_store.launch_uses_engine("fx", "1", "cpu") is False
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.launch_uses_engine("fx", "1", "cpu") is True
    finally:
        hold.release_and_check_last()


def test_sweep_never_raises_when_the_root_cannot_be_read(store: EngineStore) -> None:
    store.root.mkdir()
    with mock.patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
        assert store.sweep() == []


def test_prune_also_clears_a_damaged_build(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1", "2"))
    (store.root / "fx" / "1-cpu" / RECORD_NAME).unlink()
    (store.root / "fx" / "stray-file").write_text("x", encoding="utf-8")
    (store.root / "packs").mkdir()
    (store.root / "packs" / "fx-1-cpu.whl").write_text("pack", encoding="utf-8")
    gone = store.prune(keep={("fx", "2", "cpu")}, in_use=lambda *_: False)
    assert [p.name for p in gone] == ["1-cpu"]
    assert names(store.root / "fx") == ["2-cpu", "stray-file"]
    assert (store.root / "packs" / "fx-1-cpu.whl").is_file()


def test_prune_with_no_root_removes_nothing(store: EngineStore) -> None:
    assert store.prune(keep={("fx", "1", "cpu")}, in_use=lambda *_: False) == []
