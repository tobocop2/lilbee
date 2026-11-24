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

import pytest

from lilbee.providers.fleet.engine_store import (
    RECORD_NAME,
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
            str(store.root) in str(e) and "permission" in str(e).lower()
            for e in store.damaged()
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
            str(denied) in r.getMessage() and r.levelno >= logging.WARNING
            for r in caplog.records
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
