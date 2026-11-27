"""The engine store walk: one rule for every entry in a build slot.

The walk looks at each entry without following a link. An entry is absent, a build,
damage that remove and prune delete, or unreadable. Each cell builds its case in
tmp_path and checks the public store surface: installed() and damaged() (the engine
list), build_dirs() and prune(), remove(), sweep() and install_from_wheel().
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_store import (
    RECORD_NAME,
    ChecksumError,
    EngineNotInstalledError,
    EngineStore,
    PinLock,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and links")
needs_non_root = pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0, reason="root ignores modes"
)
ENGINE = "tei"


def idle(*_: object) -> bool:
    return False


@pytest.fixture
def store(tmp_path: Path) -> EngineStore:
    return EngineStore(tmp_path / "engines")


def build(store: EngineStore, version: str, binary: str | None = None) -> Path:
    """A complete build of the tei engine, written by hand: record and one data file."""
    folder = store.root / ENGINE / f"{version}-cpu"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "data.txt").write_text("x", encoding="utf-8")
    record = {
        "engine": ENGINE,
        "version": version,
        "backend": "cpu",
        "sha256": "0" * 64,
        "licences": ["MIT"],
        "installed_at": "2026-11-24T00:00:00+00:00",
        "binary": binary,
    }
    (folder / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")
    return folder


def make_wheel(path: Path) -> str:
    meta = "Metadata-Version: 2.4\nName: lilbee-engine-fx\nVersion: 1\nLicense-Expression: MIT\n"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("lilbee_engine_fx/bin/server")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "#!/bin/sh\n")
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", meta)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def names(directory: Path) -> list[str]:
    return sorted(os.listdir(directory))


def raised(call: object) -> BaseException | None:
    try:
        call()  # type: ignore[operator]
    except Exception as exc:
        return exc
    return None


def user_facing(exc: BaseException | None, folder: Path, fix: str) -> bool:
    """A store error (not a raw OSError) that names *folder* and holds *fix*."""
    return (
        exc is not None
        and type(exc) not in (PermissionError, FileNotFoundError, OSError)
        and str(folder) in str(exc)
        and fix in str(exc).lower()
    )


# An unreadable engines root.
@posix_only
@needs_non_root
def test_an_unreadable_root_names_the_root_and_the_fix(store: EngineStore) -> None:
    build(store, "1")
    store.root.chmod(0)
    try:
        listed = raised(store.installed)
        damage = raised(store.damaged)
        assert listed is None, f"installed() raised {listed!r}"
        assert damage is None, f"damaged() raised {damage!r}"
        assert any(
            str(store.root) in str(e) and "permission" in str(e).lower() for e in store.damaged()
        )
        gone = raised(lambda: store.remove(ENGINE, "1", "cpu", in_use=idle))
        assert user_facing(gone, store.root, "permission"), repr(gone)
    finally:
        store.root.chmod(0o755)


# The sweep skips a folder it cannot read, says so, and goes on.
@posix_only
@needs_non_root
def test_the_sweep_skips_a_denied_engine_folder_and_goes_on(
    store: EngineStore, caplog: pytest.LogCaptureFixture
) -> None:
    denied = store.root / "llama.cpp"
    (denied / ".trash-a").mkdir(parents=True)
    later = store.root / ENGINE / ".trash-b"
    later.mkdir(parents=True)
    denied.chmod(0)
    caplog.set_level(logging.DEBUG)
    try:
        swept = store.sweep()
        assert later in swept
        assert not later.exists()
        assert any(
            str(denied) in r.getMessage() and r.levelno >= logging.WARNING for r in caplog.records
        )
    finally:
        denied.chmod(0o755)


# Remove and prune leave no trash when a build folder is read-only.
@posix_only
@needs_non_root
def test_remove_clears_a_read_only_build(store: EngineStore) -> None:
    folder = build(store, "1")
    (folder / "sub").chmod(0o555)
    folder.chmod(0o555)
    store.remove(ENGINE, "1", "cpu", in_use=idle)
    assert names(store.root / ENGINE) == []


@posix_only
@needs_non_root
def test_prune_clears_a_read_only_build(store: EngineStore) -> None:
    folder = build(store, "1")
    build(store, "2")
    (folder / "sub").chmod(0o555)
    folder.chmod(0o555)
    gone = store.prune({(ENGINE, "2", "cpu")}, in_use=idle)
    assert [p.name for p in gone] == ["1-cpu"]
    assert names(store.root / ENGINE) == ["2-cpu"]


# Remove in a read-only engine folder names the folder and the fix.
@posix_only
@needs_non_root
def test_remove_in_a_read_only_engine_folder_names_it(store: EngineStore) -> None:
    build(store, "1")
    engine = store.root / ENGINE
    engine.chmod(0o555)
    try:
        exc = raised(lambda: store.remove(ENGINE, "1", "cpu", in_use=idle))
        assert user_facing(exc, engine, "writable"), repr(exc)
        assert [b.version for b in store.installed()] == ["1"]
    finally:
        engine.chmod(0o755)


# A build folder that goes during the walk is absent, not damaged.
class _Listing:
    def __init__(self, entries: list[object]) -> None:
        self._it = iter(entries)

    def __iter__(self) -> _Listing:
        return self

    def __next__(self) -> object:
        return next(self._it)

    def __enter__(self) -> _Listing:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def close(self) -> None:
        pass


def test_a_build_that_goes_after_the_listing_is_absent(
    store: EngineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = build(store, "1")
    engine = os.path.realpath(store.root / ENGINE)
    real_listdir, real_scandir = os.listdir, os.scandir

    def goes_after(path: object) -> bool:
        hit = isinstance(path, (str, os.PathLike)) and os.path.realpath(path) == engine
        return hit and folder.exists()

    def listdir(path: object = ".") -> list[str]:
        found = real_listdir(path)
        if goes_after(path):
            shutil.rmtree(folder)
        return found

    def scandir(path: object = ".") -> _Listing:
        with real_scandir(path) as it:
            found = list(it)
        if goes_after(path):
            shutil.rmtree(folder)
        return _Listing(found)

    monkeypatch.setattr(os, "listdir", listdir)
    monkeypatch.setattr(os, "scandir", scandir)
    assert [str(e.folder) for e in store.damaged()] == []  # the first walk meets the race
    assert not folder.exists()
    assert store.installed() == []


def test_remove_of_a_build_that_goes_says_not_installed(store: EngineStore) -> None:
    folder = build(store, "1")

    def goes(*_: object) -> bool:
        shutil.rmtree(folder)
        return False

    with pytest.raises(EngineNotInstalledError):
        store.remove(ENGINE, "1", "cpu", in_use=goes)


# The list and remove agree on a plain file with a build name.
def test_the_list_and_remove_agree_on_a_build_named_file(store: EngineStore) -> None:
    (store.root / ENGINE).mkdir(parents=True)
    stray = store.root / ENGINE / "1-cpu"
    stray.write_text("x", encoding="utf-8")
    offered = [e.folder for e in store.damaged()]
    if stray in offered:
        store.remove(ENGINE, "1", "cpu", in_use=idle)
        assert not os.path.lexists(stray)
        assert store.damaged() == []
    else:
        with pytest.raises(EngineNotInstalledError):
            store.remove(ENGINE, "1", "cpu", in_use=idle)


# A binary that is a symlink loop is damage that remove clears.
@posix_only
def test_remove_clears_a_build_whose_binary_is_a_loop(store: EngineStore) -> None:
    folder = build(store, "1", binary="server")
    (folder / "server").symlink_to("other")
    (folder / "other").symlink_to("server")
    assert store.installed() == []
    assert [e.folder for e in store.damaged()] == [folder]
    store.remove(ENGINE, "1", "cpu", in_use=idle)
    assert names(store.root / ENGINE) == []


# A loop or dangling build-folder link is pruned, and a reinstall succeeds.
def _bad_link(store: EngineStore, kind: str) -> Path:
    link = store.root / ENGINE / "1-cpu"
    link.parent.mkdir(parents=True, exist_ok=True)
    target = link if kind == "loop" else store.root / "nowhere"
    link.symlink_to(target, target_is_directory=True)
    return link


@posix_only
@pytest.mark.parametrize("kind", ["loop", "dangling"])
def test_prune_clears_a_bad_build_link(store: EngineStore, kind: str) -> None:
    build(store, "2")
    link = _bad_link(store, kind)
    gone = store.prune({(ENGINE, "2", "cpu")}, in_use=idle)
    assert link in gone
    assert not os.path.lexists(link)


@posix_only
@pytest.mark.parametrize("kind", ["loop", "dangling"])
def test_reinstall_over_a_bad_build_link(store: EngineStore, tmp_path: Path, kind: str) -> None:
    link = _bad_link(store, kind)
    wheel = tmp_path / "fx.whl"
    sha = make_wheel(wheel)
    lock = PinLock(ENGINE, "1", "cpu", {}, ("MIT",), sha, "server", ())
    assert store.install_from_wheel(lock, wheel) == link
    assert not link.is_symlink()
    assert [b.version for b in store.installed()] == ["1"]


# The rule keeps the target of a link out of the root.
@posix_only
def test_a_link_out_of_the_root_keeps_its_target(store: EngineStore, tmp_path: Path) -> None:
    outside = tmp_path / "keep"
    outside.mkdir()
    (outside / "file.txt").write_text("data", encoding="utf-8")
    (store.root / ENGINE).mkdir(parents=True)
    (store.root / ENGINE / "1-cpu").symlink_to(outside, target_is_directory=True)
    raised(lambda: store.remove(ENGINE, "1", "cpu", in_use=idle))
    raised(lambda: store.prune({(ENGINE, "9", "cpu")}, in_use=idle))
    assert (outside / "file.txt").read_text(encoding="utf-8") == "data"


# K ruling 3: a read-only engine folder keeps its mode; the sweep and the prune warn.
@posix_only
@needs_non_root
def test_the_sweep_keeps_the_mode_of_a_read_only_engine_folder(
    store: EngineStore, caplog: pytest.LogCaptureFixture
) -> None:
    engine = store.root / ENGINE
    (engine / ".trash-x" / "f").mkdir(parents=True)
    engine.chmod(0o555)
    try:
        with caplog.at_level(logging.WARNING):
            store.sweep()
        mode = stat.S_IMODE(os.lstat(engine).st_mode)
    finally:
        engine.chmod(0o755)
    assert mode == 0o555, oct(mode)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(engine) in text and "read-only" in text for text in warned), warned


@posix_only
@needs_non_root
def test_the_prune_warns_when_it_cannot_delete(
    store: EngineStore, caplog: pytest.LogCaptureFixture
) -> None:
    build(store, "1")
    build(store, "2")
    engine = store.root / ENGINE
    engine.chmod(0o555)
    try:
        with caplog.at_level(logging.WARNING):
            gone = store.prune({(ENGINE, "2", "cpu")}, in_use=idle)
    finally:
        engine.chmod(0o755)
    assert gone == []
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(engine) in text for text in warned), warned


def test_a_leftover_that_goes_mid_sweep_is_gone_and_the_sweep_goes_on(
    store: EngineStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = store.root / ENGINE
    (engine / ".stage-a").mkdir(parents=True)
    (engine / ".trash-b" / "f").mkdir(parents=True)
    listing = engine_store._entries

    def listed_then_gone(folder: Path, name: str) -> list[os.DirEntry[str]]:
        entries = listing(folder, name)
        if folder == engine:
            (engine / ".stage-a").rmdir()
        return entries

    monkeypatch.setattr(engine_store, "_entries", listed_then_gone)
    assert sorted(p.name for p in store.sweep()) == [".stage-a", ".trash-b"]
    assert names(engine) == []


def test_a_folder_that_stays_is_reported_false_and_is_not_listed_as_swept(
    store: EngineStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    engine = store.root / ENGINE
    stuck = engine / ".trash-stuck"
    stuck.mkdir(parents=True)
    (stuck / "keep.txt").write_text("x", encoding="utf-8")
    (engine / ".stage-ok").mkdir()
    real_unlink = os.unlink

    def refuse_keep(path: str | Path, *args: Any, **kwargs: Any) -> None:
        if str(path).endswith("keep.txt"):
            raise PermissionError(13, "Permission denied", str(path))
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse_keep)
    with caplog.at_level(logging.WARNING, logger=engine_store.log.name):
        assert engine_store.delete_tree(stuck) is False
        assert any(str(stuck) in r.getMessage() for r in caplog.records)
        swept = store.sweep()
    assert swept == [engine / ".stage-ok"]
    assert stuck.exists()


def test_a_cleanup_failure_never_hides_the_first_error(
    store: EngineStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    wheel = tmp_path / "fx.whl"
    make_wheel(wheel)
    lock = PinLock(ENGINE, "1", "cpu", {}, ("MIT",), "0" * 64, "server", ())

    def refuse(path: Path) -> bool:
        raise PermissionError(13, "denied", str(path))

    monkeypatch.setattr(engine_store, "delete_tree", refuse)
    with caplog.at_level(logging.WARNING), pytest.raises(ChecksumError):
        store.install_from_wheel(lock, wheel)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


@posix_only
def test_the_retry_adds_the_write_bit_only_inside_the_tree(tmp_path: Path) -> None:
    top = tmp_path / "trash"
    top.mkdir()
    tmp_path.chmod(0o555)
    try:
        engine_store._retry_writable(str(top), os.rmdir, str(top), None)
        outside = stat.S_IMODE(os.lstat(tmp_path).st_mode)
    finally:
        tmp_path.chmod(0o755)
    top.chmod(0o555)
    engine_store._retry_writable(str(top), os.rmdir, str(top / "gone"), None)
    assert outside == 0o555, oct(outside)
    assert stat.S_IMODE(os.lstat(top).st_mode) & stat.S_IWUSR


@posix_only
def test_the_retry_adds_the_write_bit_to_the_file_and_never_through_a_link(
    tmp_path: Path,
) -> None:
    top = tmp_path / "trash"
    top.mkdir()
    held = top / "held.txt"
    held.write_text("x", encoding="utf-8")
    held.chmod(0o444)
    outside = tmp_path / "outside.txt"
    outside.write_text("y", encoding="utf-8")
    outside.chmod(0o444)
    link = top / "link"
    link.symlink_to(outside)
    engine_store._retry_writable(str(top), os.unlink, str(held), None)
    engine_store._retry_writable(str(top), os.unlink, str(link), None)
    assert not held.exists()
    assert not link.exists()
    assert stat.S_IMODE(os.lstat(outside).st_mode) == 0o444


# K ruling 3: an engine folder that is a link inside the root is damage.
@posix_only
def test_an_engine_folder_link_is_damage_and_remove_unlinks_it(
    store: EngineStore, tmp_path: Path
) -> None:
    # The target has a dot name, so the walk skips it and only the link is under test.
    real = store.root / ".tei-real" / "1-cpu"
    real.mkdir(parents=True)
    (real / "data.txt").write_text("x", encoding="utf-8")
    (store.root / ENGINE).symlink_to(store.root / ".tei-real")
    assert [str(e.folder) for e in store.damaged()] == [str(store.root / ENGINE)]
    wheel = tmp_path / "fx.whl"
    lock = PinLock(ENGINE, "1", "cpu", {}, ("MIT",), make_wheel(wheel), "server", ())
    assert raised(lambda: store.install_from_wheel(lock, wheel)) is not None
    assert store.prune({("llama-cpp", "9", "cpu")}, in_use=idle) == []
    assert (store.root / ENGINE).is_symlink()
    store.remove(ENGINE, "1", "cpu", in_use=idle)
    assert not os.path.lexists(store.root / ENGINE)
    assert (real / "data.txt").read_text(encoding="utf-8") == "x"
