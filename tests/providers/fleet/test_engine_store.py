"""Tests for the side by side engine store: layout, record, lock, stage, listing."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import textwrap
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import yaml

from lilbee.core.security import FileLockUnavailableError, file_lock
from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_store import (
    LOCK_DIR,
    RECORD_NAME,
    ChecksumError,
    EngineConflictError,
    EngineStore,
    LicenceRefusedError,
    PinLock,
    UnsafeWheelError,
    licence_permitted,
)
from lilbee.providers.fleet.engines.choice import PinError, pin_lock
from tests._pack_workflow import pack_workflow


def make_wheel(path: Path, licence: str | None = "MIT", body: str = "#!/bin/sh\n") -> str:
    meta = "Metadata-Version: 2.4\nName: lilbee-engine-fx\nVersion: 1\n"
    if licence:
        meta += f"License-Expression: {licence}\n"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("lilbee_engine_fx/bin/server")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, body)
        plain = zipfile.ZipInfo("lilbee_engine_fx/data.txt")
        plain.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(plain, "x")
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", meta)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(
    sha: str,
    version: str = "1",
    licences: tuple[str, ...] = ("MIT",),
    libraries: tuple[str, ...] = (),
) -> PinLock:
    return PinLock("fx", version, "cpu", {}, licences, sha, "server", libraries)


@pytest.fixture
def wheel(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "fx.whl"
    return path, make_wheel(path)


@pytest.fixture
def store(tmp_path: Path) -> EngineStore:
    return EngineStore(tmp_path / "engines")


def assert_no_build_or_copy(root: Path) -> None:
    """The engines folder holds no build, no private wheel copy and no lock file of one."""
    assert [p.name for p in root.glob("*") if p.name != LOCK_DIR] == []
    assert list((root / LOCK_DIR).glob(".download-*")) == []


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
    other = PinLock("fx", "1", "cu125", {}, ("MIT",), sha, "server")
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


_CHILDREN = 3

_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    from lilbee.core.security import FileLockUnavailableError
    from lilbee.providers.fleet import engine_store as es

    root, wheel, sha, marks, children = sys.argv[1:]
    marks, children, me = Path(marks), int(children), os.getpid()
    real_unpack, real_check = es._unpack, es._check_wheel
    real_lock, real_complete = es.file_lock, es.EngineStore.complete_build
    lock_path = Path(root) / es.LOCK_DIR / "fx"

    def mark(kind):
        (marks / f"{kind}-{me}").write_text("x")

    def others(kind):
        return {p.name.split("-")[1] for p in marks.glob(f"{kind}-*")} - {str(me)}

    # With the product lock present this is always true, because the first installer holds
    # it. It only tells a lock that does nothing apart: then the waiters pass the lock, and
    # the first installer waits for their "checked" marks instead of the "lockwait" marks.
    def lock_is_held_elsewhere():
        try:
            with real_lock(lock_path, 0):
                return False
        except FileLockUnavailableError:
            return True

    def check(lock, wheel):
        mark("tried")
        real_check(lock, wheel)

    def file_lock(path, timeout):
        mark("lockwait")
        return real_lock(path, timeout)

    def complete_build(self, lock):
        found = real_complete(self, lock)
        mark("checked")
        return found

    def unpack(w, s):
        mark("unpack")
        try:
            os.close(os.open(marks / "holder", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            real_unpack(w, s)
            return
        deadline = time.time() + 30
        while time.time() < deadline:
            if len(others("tried")) == children - 1:
                # "checked": read the store; with the lock held that cannot happen before
                # the first installer finishes. "lockwait": written just
                # before the lock call, so that installer is about to block, not yet blocked.
                free = others("checked")
                blocked = others("lockwait") - free
                settled = len(free | blocked) == children - 1
                if settled and (not blocked or lock_is_held_elsewhere()):
                    break
            time.sleep(0.01)
        else:
            raise TimeoutError("the other installers never reached the install")
        real_unpack(w, s)

    es._unpack, es._check_wheel, es.file_lock = unpack, check, file_lock
    es.EngineStore.complete_build = complete_build
    pin = es.PinLock("fx", "1", "cpu", {}, ("MIT",), sha, "server")
    es.EngineStore(Path(root)).install_from_wheel(pin, Path(wheel))
    """
)


def test_installer_processes_unpack_the_build_once_and_leave_no_stage(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path
) -> None:
    # The first installer waits inside its unpack until each other installer has read the
    # store or is about to block on the held lock, so the overlap does not depend on speed.
    path, sha = wheel
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    marks = tmp_path / "marks"
    marks.mkdir()
    args = [sys.executable, str(script), str(store.root), str(path), sha, str(marks)]
    procs = [subprocess.Popen([*args, str(_CHILDREN)]) for _ in range(_CHILDREN)]
    assert [c.wait(timeout=120) for c in procs] == [0] * _CHILDREN
    assert len(list(marks.glob("tried-*"))) == _CHILDREN
    assert len(list(marks.glob("unpack-*"))) == 1
    assert names(store.root / "fx") == ["1-cpu"]
    assert len(store.installed()) == 1


def test_one_lock_file_per_engine_outside_the_engine_directory(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    taken: list[Path] = []
    real = engine_store.file_lock

    def spy(lock_path: Path, timeout: float):  # type: ignore[no-untyped-def]
        taken.append(lock_path)
        return real(lock_path, timeout)

    with mock.patch.object(engine_store, "file_lock", spy):
        store.install_from_wheel(pin(sha), path)
    assert [p.parent for p in taken] == [store.root / ".locks"] * 2
    assert taken[0].name.startswith(".download-")
    assert taken[1] == store.root / ".locks" / "fx"
    assert names(store.root / "fx") == ["1-cpu"]


def test_lock_is_taken_around_the_unpack(store: EngineStore, wheel: tuple[Path, str]) -> None:
    path, sha = wheel
    held: list[bool] = []
    real = engine_store.file_lock

    def spy(lock_path: Path, timeout: float):  # type: ignore[no-untyped-def]
        held.append(lock_path.name == "fx")
        return real(lock_path, timeout)

    with mock.patch.object(engine_store, "file_lock", spy):
        store.install_from_wheel(pin(sha), path)
    assert held == [False, True]


def test_install_gives_up_when_another_process_holds_the_engine_lock(
    store: EngineStore, wheel: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = wheel
    monkeypatch.setattr(engine_store, "_LOCK_TIMEOUT_S", 0.2)
    refusals: list[FileLockUnavailableError] = []

    def attempt() -> None:
        with pytest.raises(FileLockUnavailableError) as caught:
            store.install_from_wheel(pin(sha), path)
        refusals.append(caught.value)

    worker = threading.Thread(target=attempt)
    with file_lock(store.root / LOCK_DIR / "fx", 30):
        worker.start()
        worker.join(timeout=10)
        gave_up = not worker.is_alive()
    worker.join(timeout=10)
    assert gave_up
    assert [str(r) for r in refusals] == [
        "another lilbee process held the lock on fx for 0.2 seconds"
    ]
    assert not (store.root / "fx").exists()


def test_install_takes_the_engine_lock_file_in_the_lock_folder(
    store: EngineStore, wheel: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The lock file is read while the lock is held: on Windows, filelock deletes it on release.
    held: list[tuple[Path, bool]] = []

    @contextmanager
    def watched_lock(path: Path, timeout_s: float) -> Iterator[None]:
        with file_lock(path, timeout_s):
            held.append((path, (store.root / LOCK_DIR / "fx.lock").is_file()))
            yield

    monkeypatch.setattr(engine_store, "file_lock", watched_lock)
    path, sha = wheel
    store.install_from_wheel(pin(sha), path)
    assert (store.root / LOCK_DIR / "fx", True) in held


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
    permitted = licence == "MIT"
    refusal = ChecksumError if permitted and not good else (ChecksumError, LicenceRefusedError)
    with pytest.raises(refusal) as caught:
        store.install_from_wheel(lock, path)
    if permitted and not good:
        assert f"Copy {path.name} again from the lilbee release" in str(caught.value)
    assert_no_build_or_copy(store.root)


def test_an_upper_case_pin_accepts_the_lower_case_digest_of_the_same_file() -> None:
    digest = hashlib.sha256(b"wheel").hexdigest()

    engine_store.require_pinned_digest("engine x", digest, digest.upper(), None)
    with pytest.raises(ChecksumError):
        engine_store.require_pinned_digest("engine x", digest, "0" * 64, None)


def test_checksum_error_for_a_copied_wheel_names_the_file_and_says_to_copy_it_again(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path = wheel[0]
    want = (
        f"The file {path} for engine fx 1 (cpu) does not match its pin, so lilbee did not "
        f"install it. Copy {path.name} again from the lilbee release that matches this "
        "version and run the install again."
    )
    with pytest.raises(ChecksumError) as caught:
        store.install_from_wheel(pin("0" * 64), path)
    assert str(caught.value) == want


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
        ("MIT AND PSF-2.0", True),
        ("PSF-2.0", True),
        ("GPL-3.0-only", False),
        ("PSF-2.0 AND GPL-3.0-only", False),
        ("PSF-2.0 OR GPL-3.0-only", False),
        ("", False),
        ("MIT OR GPL", False),
    ],
)
def test_licence_permitted(expression: str, ok: bool) -> None:
    assert licence_permitted(expression) is ok


_CUDA_EXPRESSION = "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND BSD-3-Clause"


_NVIDIA = frozenset({"NOTICE-NVIDIA-CUDA"})
_NONE: frozenset[str] = frozenset()
_EVERY_NOTICE = frozenset(rule.file for rule in engine_store.NOTICE_RULES.values())
_VC_TEXT = (
    "Redist.txt and the Distributable Code terms.\n"
    "These files are licensed under the Microsoft terms below, "
    "not under the other terms of this wheel.\n"
)
_OPENSSL_TEXT = (
    "This product includes software developed by the OpenSSL Project for use in the "
    "OpenSSL Toolkit (http://www.openssl.org/) and cryptographic software written by "
    "Eric Young (eay@cryptsoft.com).\n"
)


@pytest.mark.parametrize(
    ("expression", "notices", "ok"),
    [
        ("LicenseRef-NVIDIA-CUDA-EULA", _NVIDIA, True),
        (_CUDA_EXPRESSION, _NVIDIA, True),
        ("LicenseRef-NVIDIA-CUDA-EULA", _NONE, False),
        (_CUDA_EXPRESSION, _NONE, False),
        ("LicenseRef-NVIDIA-Other", _NVIDIA, False),
        ("LicenseRef-NVIDIA-CUDA-EULA-2", _NVIDIA, False),
        ("GPL-3.0-only AND LicenseRef-NVIDIA-CUDA-EULA", _NVIDIA, False),
        ("MIT", _NONE, True),
        ("LicenseRef-NVIDIA-CUDA-EULA", frozenset({"NOTICE-MICROSOFT-VC-RUNTIME"}), False),
    ],
)
def test_the_nvidia_cuda_term_passes_only_with_its_notice(
    expression: str, notices: frozenset[str], ok: bool
) -> None:
    assert licence_permitted(expression, notices) is ok


@pytest.mark.parametrize(
    "term",
    [
        "ISC",
        "Zlib",
        "0BSD",
        "MIT-0",
        "Unlicense",
        "CC0-1.0",
        "Unicode-3.0",
        "CNRI-Python",
        "CDLA-Permissive-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "NCSA",
        "BSL-1.0",
        "TCL",
        "bzip2-1.0.6",
        "X11",
        "blessing",
        "Python-2.0",
        "BSD-2-Clause",
    ],
)
def test_each_term_of_the_licence_plan_passes(term: str) -> None:
    assert licence_permitted(term) is True
    assert licence_permitted(f"MIT AND {term}") is True


@pytest.mark.parametrize(
    "term",
    [
        "GPL-2.0-only",
        "GPL-3.0-or-later",
        "AGPL-3.0-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "GPL-2.0-only WITH Classpath-exception-2.0",
        "GPL-3.0-only WITH LLVM-exception",
        "LicenseRef-Unknown",
        "(MIT AND ISC)",
    ],
)
def test_copyleft_and_unknown_terms_fail_even_with_every_notice(term: str) -> None:
    assert _EVERY_NOTICE
    assert licence_permitted(term, _EVERY_NOTICE) is False
    assert licence_permitted(f"MIT AND {term}", _EVERY_NOTICE) is False


def test_mpl_passes_only_with_its_source_pointer() -> None:
    assert licence_permitted("MPL-2.0") is False
    assert licence_permitted("MIT AND MPL-2.0", _NVIDIA) is False
    assert licence_permitted("MIT AND MPL-2.0", frozenset({"SOURCE-MPL-2.0"})) is True


def test_the_vc_runtime_passes_only_with_its_notice() -> None:
    term = "LicenseRef-Microsoft-VC-Runtime"
    assert licence_permitted(term) is False
    assert licence_permitted(term, _NVIDIA) is False
    assert licence_permitted(term, frozenset({"NOTICE-MICROSOFT-VC-RUNTIME"})) is True


def test_openssl_passes_only_with_its_notice() -> None:
    assert licence_permitted("OpenSSL") is False
    assert licence_permitted("OpenSSL", _NVIDIA) is False
    assert licence_permitted("MIT AND OpenSSL", frozenset({"NOTICE-OpenSSL"})) is True


@pytest.mark.parametrize(
    "term", ["LicenseRef-Intel-OpenVINO", "LicenseRef-Intel-OpenVINO-Distribution"]
)
def test_the_openvino_runtime_is_not_on_the_list(term: str) -> None:
    assert licence_permitted(term, _EVERY_NOTICE) is False


_CRATE_URL = "https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate"
_CHECKSUM = "04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d"
_OTHER_URL = "https://static.crates.io/crates/dtoa-short/dtoa-short-0.3.5.crate"
_OTHER_CHECKSUM = "cd1511a7b6a56299bd043a9c167a6d2bfb37bf84a6dfceaba651168adfb43c87"


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        (f"{_CRATE_URL} {_CHECKSUM}\n", True),
        (f"{_CRATE_URL}\t{_CHECKSUM}\n\n{_OTHER_URL}  {_OTHER_CHECKSUM}\n", True),
        # the value is checked against the committed list, not only its form
        (f"{_CRATE_URL} {'0' * 64}\n", False),
        (f"{_CRATE_URL} {_OTHER_CHECKSUM}\n", False),
        # every line must pass
        (f"{_CRATE_URL} {_CHECKSUM}\n{_OTHER_URL} {'0' * 64}\n", False),
        (f"{_CRATE_URL} {_CHECKSUM}\nMozilla Public License text\n", False),
        # a version the list does not hold
        (f"https://static.crates.io/crates/option-ext/option-ext-0.2.1.crate {_CHECKSUM}\n", False),
        # text before the URL
        (f"x{_CRATE_URL} {_CHECKSUM}\n", False),
        (f"https://evil.example/{_CRATE_URL} {_CHECKSUM}\n", False),
        (f"see {_CRATE_URL} {_CHECKSUM}\n", False),
        ("https://github.com/mozilla/option-ext\n", False),
        (f"{_CRATE_URL}\n", False),
        (f"{_CRATE_URL} {_CHECKSUM[:-1]}\n", False),
        (f"{_CRATE_URL} {'G' * 64}\n", False),
        (f"{_CRATE_URL} {_CHECKSUM.upper()}\n", False),
        (f"https://crates.io/crates/option-ext/option-ext-0.2.0.crate {_CHECKSUM}\n", False),
        (f"https://static.crates.io/crates/option-ext/other-0.2.0.crate {_CHECKSUM}\n", False),
        (f"https://static.crates.io/crates/option-ext/option-ext-0.2.0.zip {_CHECKSUM}\n", False),
        (f"{_CRATE_URL.replace('https', 'http')} {_CHECKSUM}\n", False),
        # text after the checksum
        (f"{_CRATE_URL} {_CHECKSUM} extra\n", False),
        (f"{_CRATE_URL} {_CHECKSUM}a\n", False),
        ("", False),
        ("\n\n", False),
    ],
)
def test_the_mpl_source_pointer_names_the_crate_download_and_its_checksum(
    text: str, ok: bool
) -> None:
    assert bool(engine_store._verified_notices([("SOURCE-MPL-2.0", text)])) is ok


_SDIST_URL = "https://files.pythonhosted.org/packages/source/t/tqdm/tqdm-4.70.1.tar.gz"
_SDIST_CHECKSUM = "cefd0eca11b2a37a3aee776544d4f4ae913f02688135b5556b8788dfa474afc4"


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        (f"{_SDIST_URL} {_SDIST_CHECKSUM}\n", True),
        # a Python tree and a Rust tree in one pack
        (f"{_SDIST_URL} {_SDIST_CHECKSUM}\n{_CRATE_URL}\t{_CHECKSUM}\n", True),
        (f"{_SDIST_URL} {'0' * 64}\n", False),
        (f"{_SDIST_URL} {_CHECKSUM}\n", False),
        (f"{_SDIST_URL.replace('4.70.1', '4.70.2')} {_SDIST_CHECKSUM}\n", False),
        (f"{_SDIST_URL.replace('/t/tqdm/', '/q/tqdm/')} {_SDIST_CHECKSUM}\n", False),
        (f"{_SDIST_URL.replace('/t/tqdm/tqdm-', '/t/tqdm/other-')} {_SDIST_CHECKSUM}\n", False),
        (f"{_SDIST_URL.replace('.tar.gz', '.zip')} {_SDIST_CHECKSUM}\n", False),
        (f"{_SDIST_URL.replace('https', 'http')} {_SDIST_CHECKSUM}\n", False),
        (
            f"{_SDIST_URL.replace('files.pythonhosted.org', 'evil.example')} {_SDIST_CHECKSUM}\n",
            False,
        ),
        (f"see {_SDIST_URL} {_SDIST_CHECKSUM}\n", False),
        (f"{_SDIST_URL} {_SDIST_CHECKSUM}\nMozilla Public License text\n", False),
        # text after the checksum
        (f"{_SDIST_URL} {_SDIST_CHECKSUM} extra\n", False),
        (f"{_SDIST_URL} {_SDIST_CHECKSUM}a\n", False),
        # no space between the URL and the checksum
        (f"{_SDIST_URL}{_SDIST_CHECKSUM}\n", False),
        # another character in place of a dot of the host
        (
            f"{_SDIST_URL.replace('files.pythonhosted', 'files-pythonhosted')} {_SDIST_CHECKSUM}\n",
            False,
        ),
        (
            f"{_SDIST_URL.replace('pythonhosted.org', 'pythonhosted-org')} {_SDIST_CHECKSUM}\n",
            False,
        ),
    ],
)
def test_the_mpl_source_pointer_names_a_pypi_source_archive_and_its_checksum(
    text: str, ok: bool
) -> None:
    assert bool(engine_store._verified_notices([("SOURCE-MPL-2.0", text)])) is ok


def test_a_pypi_package_name_with_a_dot_or_a_dash_is_a_listed_source() -> None:
    digest = "ab" * 32
    url = "https://files.pythonhosted.org/packages/source/z/zope.i-n/zope.i-n-1.2.tar.gz"
    listed = {("zope.i-n", "1.2"): digest}
    with mock.patch.object(engine_store, "MPL_PYPI_CHECKSUMS", listed):
        assert bool(engine_store._verified_notices([("SOURCE-MPL-2.0", f"{url} {digest}\n")]))


@pytest.mark.parametrize(
    ("name", "text", "ok"),
    [
        ("NOTICE-NVIDIA-CUDA", "notice", True),
        ("NOTICE-NVIDIA-CUDA", "  \n", False),
        ("NOTICE-MICROSOFT-VC-RUNTIME", _VC_TEXT, True),
        ("NOTICE-MICROSOFT-VC-RUNTIME", "Redist.txt only", False),
        ("NOTICE-OpenSSL", _OPENSSL_TEXT, True),
        ("NOTICE-OpenSSL", "software developed by the OpenSSL Project", False),
        ("NOTICE-OpenSSL", "cryptographic software written by Eric Young", False),
        ("NOTICE-Other", "notice", False),
    ],
)
def test_a_notice_file_must_hold_the_text_its_term_needs(name: str, text: str, ok: bool) -> None:
    assert bool(engine_store._verified_notices([(name, text)])) is ok


def _cuda_wheel(
    path: Path,
    *,
    notice: bool,
    licence: str = _CUDA_EXPRESSION,
    tool: str = "server",
    notice_path: str = "p-1.dist-info/licenses/NOTICE-NVIDIA-CUDA",
    notice_text: str | bytes = "notice",
    extras: tuple[tuple[str, str | bytes], ...] = (),
) -> str:
    with zipfile.ZipFile(path, "w") as archive:
        server = zipfile.ZipInfo(f"p/bin/{tool}")
        server.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(server, "#!/bin/sh\n")
        archive.writestr("p-1.dist-info/METADATA", f"Name: p\nLicense-Expression: {licence}\n")
        if notice:
            archive.writestr(notice_path, notice_text)
        for name, text in extras:
            archive.writestr(name, text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_cuda_wheel_with_its_notice_installs(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True)
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


def test_a_cuda_wheel_without_its_notice_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=False)
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        store.install_from_wheel(pin(sha), path)


def test_a_cuda_notice_outside_the_dist_info_folder_is_refused(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True, notice_path="p/licenses/NOTICE-NVIDIA-CUDA")
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        store.install_from_wheel(pin(sha), path)


def test_a_cuda_wheel_notice_with_a_stray_byte_still_installs(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True, notice_text=b"notice \xff")
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


_VC_EXPRESSION = "MIT AND Apache-2.0 WITH LLVM-exception AND LicenseRef-Microsoft-VC-Runtime"
_VC_NOTICE = "p-1.dist-info/licenses/NOTICE-MICROSOFT-VC-RUNTIME"
_MPL_NOTICE = "p-1.dist-info/licenses/SOURCE-MPL-2.0"


def test_a_vc_runtime_wheel_with_its_notice_installs(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "v.whl"
    sha = _cuda_wheel(
        path, notice=True, licence=_VC_EXPRESSION, notice_path=_VC_NOTICE, notice_text=_VC_TEXT
    )
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


def test_a_vc_runtime_wheel_with_the_nvidia_notice_is_refused(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "v.whl"
    sha = _cuda_wheel(path, notice=True, licence=_VC_EXPRESSION)
    with pytest.raises(LicenceRefusedError, match="LicenseRef-Microsoft-VC-Runtime"):
        store.install_from_wheel(pin(sha), path)


def test_a_vc_runtime_notice_without_the_scope_sentence_is_refused(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "v.whl"
    sha = _cuda_wheel(
        path,
        notice=True,
        licence=_VC_EXPRESSION,
        notice_path=_VC_NOTICE,
        notice_text="Redist.txt only",
    )
    with pytest.raises(LicenceRefusedError, match="LicenseRef-Microsoft-VC-Runtime"):
        store.install_from_wheel(pin(sha), path)


def test_an_mpl_wheel_needs_its_source_pointer(store: EngineStore, tmp_path: Path) -> None:
    pointer = f"{_CRATE_URL} {_CHECKSUM}\n"
    for name, notice, text in (
        ("none", False, ""),
        ("repository", True, "https://github.com/mozilla/option-ext\n"),
    ):
        refused = tmp_path / f"{name}.whl"
        sha = _cuda_wheel(
            refused,
            notice=notice,
            licence="MIT AND MPL-2.0",
            notice_path=_MPL_NOTICE,
            notice_text=text,
        )
        with pytest.raises(LicenceRefusedError, match=r"MPL-2\.0"):
            store.install_from_wheel(pin(sha), refused)
    accepted = tmp_path / "a.whl"
    sha = _cuda_wheel(
        accepted,
        notice=True,
        licence="MIT AND MPL-2.0",
        notice_path=_MPL_NOTICE,
        notice_text=pointer,
    )
    store.install_from_wheel(pin(sha), accepted)
    assert len(store.installed()) == 1


def test_an_openssl_wheel_needs_its_notice(store: EngineStore, tmp_path: Path) -> None:
    refused = tmp_path / "r.whl"
    sha = _cuda_wheel(refused, notice=False, licence="MIT AND OpenSSL")
    with pytest.raises(LicenceRefusedError, match="OpenSSL"):
        store.install_from_wheel(pin(sha), refused)
    accepted = tmp_path / "a.whl"
    sha = _cuda_wheel(
        accepted,
        notice=True,
        licence="MIT AND OpenSSL",
        notice_path="p-1.dist-info/licenses/NOTICE-OpenSSL",
        notice_text=_OPENSSL_TEXT,
    )
    store.install_from_wheel(pin(sha), accepted)
    assert len(store.installed()) == 1


def test_a_wheel_with_two_notices_passes_both_terms(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "b.whl"
    licence = f"{_CUDA_EXPRESSION} AND LicenseRef-Microsoft-VC-Runtime"
    _cuda_wheel(path, notice=True, licence=licence)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(_VC_NOTICE, _VC_TEXT)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


def test_a_wheel_with_an_unknown_licence_is_refused_even_with_the_notice(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True, licence="MIT AND LicenseRef-Mystery")
    with pytest.raises(LicenceRefusedError, match="LicenseRef-Mystery"):
        store.install_from_wheel(pin(sha), path)


def test_a_cuda_wheel_with_a_nested_notice_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(
        path, notice=True, notice_path="p/vendor/p-1.dist-info/licenses/NOTICE-NVIDIA-CUDA"
    )
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        store.install_from_wheel(pin(sha), path)


def test_a_cuda_wheel_with_an_empty_notice_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True, notice_text="")
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        store.install_from_wheel(pin(sha), path)


def test_the_mistral_cuda_pin_record_installs_a_cuda_wheel(
    store: EngineStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.catalog import runnable
    from lilbee.providers.fleet.engines import mistral

    monkeypatch.setattr(runnable, "session_host", lambda: object())
    monkeypatch.setattr(
        mistral.MistralAdapter,
        "pick_build",
        lambda self, host: mistral.BuildPick("cuda128-sm89", ""),
    )
    monkeypatch.setattr(mistral, "pack_platforms", lambda *args: {})
    monkeypatch.setattr(mistral, "pack_digest", lambda *args: None)
    record = mistral.MistralAdapter().pin_record()
    assert record["licenses"] == [_CUDA_EXPRESSION]
    path = tmp_path / "m.whl"
    sha = _cuda_wheel(path, notice=True, tool=str(record["tool"]))
    lock = pin_lock({**record, "sha256": sha})
    store.install_from_wheel(lock, path)
    assert len(store.installed()) == 1


def test_the_mistral_cpu_pin_record_states_only_mit(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.catalog import runnable
    from lilbee.providers.fleet.engines import mistral

    monkeypatch.setattr(runnable, "session_host", lambda: object())
    monkeypatch.setattr(
        mistral.MistralAdapter, "pick_build", lambda self, host: mistral.BuildPick("cpu", "")
    )
    monkeypatch.setattr(mistral, "pack_platforms", lambda *args: {})
    monkeypatch.setattr(mistral, "pack_digest", lambda *args: None)
    assert mistral.MistralAdapter().pin_record()["licenses"] == ["MIT"]


def test_a_pin_may_state_the_nvidia_term_without_a_notice() -> None:
    engine_store._check_licences(pin("0" * 64, licences=(_CUDA_EXPRESSION,)))


def test_a_pin_with_another_nvidia_term_is_refused() -> None:
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-Other"):
        engine_store._check_licences(pin("0" * 64, licences=("MIT AND LicenseRef-NVIDIA-Other",)))


def test_a_pin_may_state_a_vendor_term_and_mpl_without_a_notice() -> None:
    engine_store._check_licences(pin("0" * 64, licences=(_VC_EXPRESSION + " AND MPL-2.0",)))


def test_a_pin_with_a_copyleft_term_is_refused() -> None:
    with pytest.raises(LicenceRefusedError, match=r"LGPL-2\.1-only"):
        engine_store._check_licences(pin("0" * 64, licences=("MIT AND LGPL-2.1-only",)))


_GPL_TEXT = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n"
_LGPL_TEXT = "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999\n"
_DIST_LICENSES = "p-1.dist-info/licenses/"
_CUDA_NOTICE = Path(__file__).parent / "fixtures" / "NOTICE-NVIDIA-CUDA"
_WRAPPED_GPL = "GNU General\nPublic   License\n\n  Version 3, 29 June 2007\n"
_WRAPPED_LGPL = "gnu lesser\r\ngeneral public license\r\nversion 2.1\r\n"
_UTF16_GPL = codecs.BOM_UTF16_LE + _GPL_TEXT.encode("utf-16-le")
_MIT_TEXT = "MIT License\nPermission is hereby granted\n"
# NUL padding makes the bytes look like UTF-16 without a mark, but the title is UTF-8.
_NUL_PADDED_GPL = b"\x00\x01" * 1000 + _GPL_TEXT.encode()


@pytest.mark.parametrize(
    ("member", "text", "named"),
    [
        ("p/docs/gpl-3.0.txt", _GPL_TEXT, "gpl-3.0.txt"),
        ("p/docs/lgpl-2.1.txt", _LGPL_TEXT, "lgpl-2.1.txt"),
        ("p/LICENSES.txt", _GPL_TEXT, "LICENSES.txt"),
        ("p/THIRD-PARTY-LICENSES.txt", _LGPL_TEXT, "THIRD-PARTY-LICENSES.txt"),
        ("p/COPYING.GPLv2", "x", "COPYING.GPLv2"),
        ("p/LICENSE.GPLv3", "x", "LICENSE.GPLv3"),
        ("p/doc/gpl-notes.txt", "x", "gpl-notes.txt"),
        ("p/COPYRIGHT.txt", _GPL_TEXT, "COPYRIGHT.txt"),
        ("p/COPYING", _WRAPPED_GPL, "COPYING"),
        ("p/COPYING", _WRAPPED_LGPL, "COPYING"),
        ("p/COPYING", _GPL_TEXT.lower(), "COPYING"),
        ("p/COPYING", _UTF16_GPL, "COPYING"),
        ("p/COPYING", _UTF16_GPL + b"x", "COPYING"),
        ("p/COPYING", codecs.BOM_UTF16_BE + _GPL_TEXT.encode("utf-16-be"), "COPYING"),
        ("p/lib/libhistory.so.8", "ELF", "libhistory"),
        ("p/lib/libgmp.a", "ar archive", "libgmp"),
        ("p/lib/libreadline.a", "ar archive", "libreadline"),
        ("p/lib/libhistory.a", "ar archive", "libhistory"),
        (_DIST_LICENSES + "COPYING.LESSER", "any text", "COPYING.LESSER"),
        (_DIST_LICENSES + "LICENSE-AGPL-3.0", "any text", "LICENSE-AGPL-3.0"),
        (_DIST_LICENSES + "LICENSE.GPL", "any text", "LICENSE.GPL"),
        (_DIST_LICENSES + "COPYING", _GPL_TEXT, "COPYING"),
        (_DIST_LICENSES + "THIRD-PARTY/gmp/LICENSE", _LGPL_TEXT, "LICENSE"),
        (_DIST_LICENSES + "THIRD-PARTY/gmp/terms.txt", _LGPL_TEXT, "terms.txt"),
        ("p/COPYING", "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n", "p/COPYING"),
        ("p/LICENSE", "GNU LIBRARY GENERAL PUBLIC LICENSE\nVersion 2\n", "p/LICENSE"),
        ("p/NOTICE.txt", _GPL_TEXT, "p/NOTICE.txt"),
        ("p.libs/libquadmath-96973f99.so.0.0.0", "ELF", "libquadmath"),
        ("p/lib/libgmp.so.10", "ELF", "libgmp"),
        ("p/lib/libmpfr.6.dylib", "Mach-O", "libmpfr"),
        ("p/bin/readline.dll", "PE", "readline.dll"),
        ("p/lib/gmp-COPYING", _GPL_TEXT, "gmp-COPYING"),
        ("p/lib/readline-LICENSE", _GPL_TEXT, "readline-LICENSE"),
        ("p/lib/mpfr.COPYING.LESSER", "any text", "mpfr.COPYING.LESSER"),
        ("p/lib/libgmp-COPYING.txt", _LGPL_TEXT, "libgmp-COPYING.txt"),
        ("p/lib/history-gpl.txt", "any text", "history-gpl.txt"),
        ("p/COPYING", _GPL_TEXT.encode("utf-16-le"), "COPYING"),
        ("p/COPYING", _GPL_TEXT.encode("utf-16-be"), "COPYING"),
        ("p/COPYING", codecs.BOM_UTF32_LE + _GPL_TEXT.encode("utf-32-le"), "COPYING"),
        ("p/COPYING", codecs.BOM_UTF32_BE + _GPL_TEXT.encode("utf-32-be"), "COPYING"),
        ("p/COPYING", "GNU GENERAL\xa0PUBLIC LICENSE\nVersion 3".encode("latin-1"), "COPYING"),
        ("p/COPYING", "GNU GENERAL\xa0PUBLIC LICENSE\nVersion 3".encode(), "COPYING"),
        ("p/gpl/README", _GPL_TEXT, "README"),
        ("p/docs/GPL-3.0/terms.txt", _GPL_TEXT, "terms.txt"),
        pytest.param(
            "p/notice-model.bin", _NUL_PADDED_GPL, "notice-model.bin", id="nul-padded-utf8-title"
        ),
    ],
)
def test_a_wheel_that_carries_a_copyleft_file_is_refused_whatever_it_declares(
    store: EngineStore, tmp_path: Path, member: str, text: str | bytes, named: str
) -> None:
    path = tmp_path / "g.whl"
    sha = _cuda_wheel(path, notice=False, licence="MIT", extras=((member, text),))
    with pytest.raises(LicenceRefusedError, match=rf"GPL.*{re.escape(named)}"):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []


@pytest.mark.parametrize(
    ("member", "text"),
    [
        (_DIST_LICENSES + "LICENSE", "MIT License\nPermission is hereby granted\n"),
        (
            _DIST_LICENSES + "THIRD-PARTY/x/COPYING",
            "BSD 3-Clause, not the GNU general public license",
        ),
        ("p/lib/libquadmathx.so", "ELF"),
        ("p/lib/libgmpish.so", "ELF"),
        ("p/doc/copying.txt.gz", "binary"),
        ("p/doc/readline.txt", "a text file"),
        ("p/doc/history.txt", "a text file"),
        ("p/lib/libgmp.abc", "not an archive"),
        (_DIST_LICENSES + "LICENSE-bgpl.txt", "MIT License\nPermission is hereby granted\n"),
        (_DIST_LICENSES + "NOTICE-V3", "version 3 of the GNU General Public License applies"),
        (
            _DIST_LICENSES + "NOTICE-OTHER",
            "It names the GNU General Public License (GPL) and the GNU Lesser General "
            "Public License (LGPL), with source on request.",
        ),
        ("p/doc/terms.txt", _LGPL_TEXT),
        ("p/LICENSE-gplus", _MIT_TEXT),
        ("p/gplus.LICENSE", _MIT_TEXT),
        ("p/gplus.py", "print('hello')"),
        ("p/gplearn/gplearn.py", "print('hello')"),
        ("p/COPYRIGHT-agplot", _MIT_TEXT),
        ("p/gplus-1.0/LICENSE", _MIT_TEXT),
        ("p/gplus-1.0/terms.txt", _LGPL_TEXT),
        ("p/gpl/LICENSE", _MIT_TEXT),
        ("p/gpl/data.bin", "no title in it"),
        ("p/COPYING", _MIT_TEXT.encode("utf-16-le")),
        ("p/COPYING", "no title\xa0here".encode("latin-1")),
    ],
)
def test_a_wheel_with_permissive_licence_files_still_installs(
    store: EngineStore, tmp_path: Path, member: str, text: str | bytes
) -> None:
    path = tmp_path / "ok.whl"
    sha = _cuda_wheel(path, notice=False, licence="MIT", extras=((member, text),))
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", [""]),
        (b"a", ["a"]),
        (b"ab", ["ab"]),
        (b"a\x00b\x00c\x00", ["a\x00b\x00c\x00", "abc"]),
        (b"\x00a\x00b\x00c", ["\x00a\x00b\x00c", "abc"]),
        (b"a\x00\x00b\x00c", ["a\x00\x00b\x00c"]),
        (b"a\x00b\x00c", ["a\x00b\x00c", "ab\ufffd"]),
        (b"\x00a\x00b\x00", ["\x00a\x00b\x00", "ab\ufffd"]),
        (b"a\x00b\x01", ["a\x00b\x01"]),
        (b"a\x00\x00\x00", ["a\x00\x00\x00"]),
        (b"\x00\x00\x00a", ["\x00\x00\x00a"]),
        (b"\x00a\x01b", ["\x00a\x01b"]),
        (b"\xe9t\xe9", ["\xe9t\xe9"]),
    ],
)
def test_licence_texts_lists_every_decoding_of_the_bytes(data: bytes, expected: list[str]) -> None:
    assert engine_store._licence_texts(data) == expected


@pytest.mark.parametrize("encoding", ["utf-32-le", "utf-32-be", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("tail", [b"", b"x"])
def test_licence_texts_reads_a_wide_file_by_its_mark(encoding: str, tail: bytes) -> None:
    mark = {
        "utf-32-le": codecs.BOM_UTF32_LE,
        "utf-32-be": codecs.BOM_UTF32_BE,
        "utf-16-le": codecs.BOM_UTF16_LE,
        "utf-16-be": codecs.BOM_UTF16_BE,
    }[encoding]
    data = mark + "ab".encode(encoding) + tail
    expected = "ab" if not tail else "ab\ufffd"
    assert engine_store._licence_texts(data) == [data.decode("latin-1"), expected]


def test_the_real_nvidia_cuda_notice_names_the_gpl_and_still_installs(
    store: EngineStore, tmp_path: Path
) -> None:
    notice = _CUDA_NOTICE.read_bytes()
    assert b"GNU General Public License" in notice
    assert b"GNU Lesser General Public License" in notice
    path = tmp_path / "c.whl"
    sha = _cuda_wheel(path, notice=True, notice_text=notice)
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


def test_a_wheel_with_two_entries_of_one_notice_is_refused(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "d.whl"
    sha = _cuda_wheel(
        path,
        notice=True,
        licence="MIT AND OpenSSL",
        notice_path=_DIST_LICENSES + "NOTICE-OpenSSL",
        notice_text=_OPENSSL_TEXT,
    )
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr(_DIST_LICENSES + "NOTICE-OpenSSL", "")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(LicenceRefusedError, match=r"two entries.*NOTICE-OpenSSL"):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []


def test_a_wheel_with_two_entries_of_a_file_that_is_no_notice_still_installs(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "d.whl"
    _cuda_wheel(path, notice=False, licence="MIT", extras=((_DIST_LICENSES + "LICENSE", "MIT"),))
    with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
        archive.writestr(_DIST_LICENSES + "LICENSE", "MIT")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


@pytest.mark.parametrize("term", ["MPL-2.0", "OpenSSL", "LicenseRef-Microsoft-VC-Runtime"])
def test_a_pin_names_a_notice_term_and_the_archive_is_the_gate(
    store: EngineStore, tmp_path: Path, term: str
) -> None:
    path = tmp_path / "n.whl"
    sha = _cuda_wheel(path, notice=False, licence=f"MIT AND {term}")
    lock = pin(sha, licences=(f"MIT AND {term}",))
    engine_store._check_licences(lock)
    with pytest.raises(LicenceRefusedError, match=re.escape(term)):
        store.install_from_wheel(lock, path)
    with pytest.raises(LicenceRefusedError, match=re.escape(term)):
        store.ensure(lock, lambda _lock, _scratch: path)
    assert store.installed() == []


def test_the_windows_pack_workflow_writes_the_notice_the_check_reads() -> None:
    workflow = pack_workflow().read_text(encoding="utf-8")
    name = engine_store.NOTICE_RULES["LicenseRef-Microsoft-VC-Runtime"].file
    tools = Path(__file__).parents[3] / "tools" / "wheel-build"
    # The llama.cpp job writes the notice and checks it. For mistral.rs one script does each.
    llama = yaml.safe_load(workflow)["jobs"]["cell"]["steps"]
    texts = {
        "llama.cpp job": ("\n".join(step.get("run", "") for step in llama), 2),
        "pack_mistralrs.sh": ((tools / "pack_mistralrs.sh").read_text(encoding="utf-8"), 1),
        "check_mistralrs_wheel.sh": (
            (tools / "check_mistralrs_wheel.sh").read_text(encoding="utf-8"),
            1,
        ),
    }
    for place, (text, uses) in texts.items():
        assert text.count(name) == uses, place
        assert text.lower().count(name.lower()) == uses, place
    assert workflow.lower().count(name.lower()) == workflow.count(name)


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "", "a b"])
def test_pin_names_cannot_leave_the_engine_directory(bad: str) -> None:
    with pytest.raises(ValueError, match="cannot name an engine directory"):
        PinLock(bad, "1", "cpu", {}, ("MIT",), "a", "server")
    with pytest.raises(ValueError, match="cannot name an engine directory"):
        PinLock("fx", bad, "cpu", {}, ("MIT",), "a", "server")


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
    assert_no_build_or_copy(store.root)


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


def test_wheel_without_a_bin_tool_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "lib.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("p/data.txt", "x")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(UnsafeWheelError, match="has no server in its bin directory"):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []


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
def test_a_binary_that_is_a_symlink_loop_is_a_damaged_build(
    store: EngineStore, tmp_path: Path
) -> None:
    sha = installed_store(store, tmp_path, ("1", "2"))
    binary = store.root / "fx" / "1-cpu" / "lilbee_engine_fx" / "bin" / "server"
    binary.unlink()
    binary.symlink_to(binary.with_name("other"))
    binary.with_name("other").symlink_to(binary)
    assert [b.version for b in store.installed()] == ["2"]
    assert store.complete_build(pin(sha)) is None
    assert [p.name for *_, p in store.build_dirs()] == ["1-cpu", "2-cpu"]
    gone = store.prune(keep={("fx", "2", "cpu")}, in_use=lambda *_: False)
    assert [p.name for p in gone] == ["1-cpu"]
    assert names(store.root / "fx") == ["2-cpu"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_binary_that_links_to_a_file_inside_the_build_is_usable(
    store: EngineStore, tmp_path: Path
) -> None:
    installed_store(store, tmp_path)
    binary = store.root / "fx" / "1-cpu" / "lilbee_engine_fx" / "bin" / "server"
    binary.rename(binary.with_name("real"))
    binary.symlink_to("real")
    assert [b.version for b in store.installed()] == ["1"]


def test_wheel_whose_tool_is_a_symlink_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "link.whl"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("p/bin/server")
        info.external_attr = (stat.S_IFLNK | 0o755) << 16
        archive.writestr(info, "/bin/sh")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(UnsafeWheelError, match="as a symbolic link"):
        store.install_from_wheel(pin(sha), path)
    assert_no_build_or_copy(store.root)


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


def _damage_build(store: EngineStore, tmp_path: Path) -> Path:
    installed_store(store, tmp_path)
    build = store.root / "fx" / "1-cpu"
    (build / RECORD_NAME).unlink()
    (build / "survivor.txt").write_text("held", encoding="utf-8")
    return build


@pytest.mark.parametrize("route", ["wheel", "fetch"])
def test_install_refuses_to_replace_a_damaged_build_while_a_launch_holds_it(
    route: str, store: EngineStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _damage_build(store, tmp_path)
    path = tmp_path / "fx.whl"
    lock = pin(hashlib.sha256(path.read_bytes()).hexdigest())
    seen: list[tuple[Path, str, str, str]] = []
    monkeypatch.setattr(
        engine_store, "launch_uses_engine", lambda *build_name: seen.append(build_name) or True
    )
    install = {
        "wheel": lambda: store.install_from_wheel(lock, path),
        "fetch": lambda: store.ensure(lock, lambda _pin, _scratch: path),
    }[route]
    with pytest.raises(engine_store.EngineInUseError, match="lilbee engine stop"):
        install()
    assert seen == [(store.root, "fx", "1", "cpu")]
    assert (build / "survivor.txt").is_file()


def test_install_reads_no_hold_when_the_build_is_absent(
    store: EngineStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fx.whl"
    sha = make_wheel(path)
    monkeypatch.setattr(engine_store, "launch_uses_engine", lambda *_: True)
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


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
    lock_files: list[list[str]] = []

    def fetch(lock: PinLock, dest: Path) -> Path:
        dest.mkdir(parents=True)
        during.append(store.sweep())
        lock_files.append(sorted(p.name for p in (store.root / ".locks").iterdir()))
        return path

    store.ensure(pin(sha), fetch)
    assert during == [[]]
    assert names(store.root) == [".locks", "fx"]
    assert len(lock_files) == 1
    assert [n.startswith(".download-") and n.endswith(".lock") for n in lock_files[0]] == [True]
    assert not [p for p in (store.root / ".locks").iterdir() if p.name.startswith(".download-")]


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


def record_launch(slot: Path, binary: Path | str, router: Path | str = "/r/llama-swap") -> None:
    """Write the state file of a live engine whose one launch runs *binary* behind env."""
    argv = ["/usr/bin/env", "OV_DEVICE=GPU.0", str(binary), "-m", "m.gguf"]
    launch = {"role": "chat", "argv": argv, "model": "m", "server_binary": str(binary)}
    write_state(slot, [launch], router=str(router))


def write_state(
    slot: Path, launches: list[dict], name: str = "chat.1", router: str = "/r/llama-swap"
) -> None:
    slot.mkdir(parents=True, exist_ok=True)
    state = {"pid": 1, "launches": launches, "swap_binary": router}
    (slot / f"llama-swap.state.{name}.json").write_text(json.dumps(state), encoding="utf-8")


def test_launch_uses_engine_counts_the_router_a_live_engine_runs(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    root = tmp_path / "engines"
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    record_launch(
        tmp_path / "slot",
        root / "other" / "1-cpu" / "bin" / "srv",
        router=root / "llama.cpp" / "9-cpu" / "bin" / "llama-swap",
    )
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.launch_uses_engine(root, "llama.cpp", "9", "cpu") is True
        assert engine_store.launch_uses_engine(root, "other", "1", "cpu") is True
        assert engine_store.launch_uses_engine(root, "llama.cpp", "8", "cpu") is False
    finally:
        hold.release_and_check_last()


def test_recorded_launch_binaries_reads_the_binary_not_the_first_argv_word(
    tmp_path: Path,
) -> None:
    from lilbee.providers.fleet.swap_manager import recorded_launch_binaries

    record_launch(tmp_path, "/b/srv")
    assert recorded_launch_binaries(tmp_path) == [Path("/r/llama-swap"), Path("/b/srv")]


@pytest.mark.parametrize(
    ("launches", "router"),
    [
        ([], "/r/llama-swap"),
        ([{"role": "nonsense"}], "/r/llama-swap"),
        ([{"role": "chat", "argv": ["/c/srv"], "model": "m"}], "/r/llama-swap"),
        ([{"role": "chat", "model": "m", "argv": [], "server_binary": "/c/srv"}], ""),
    ],
    ids=["no-launch", "foreign-shape", "launch-without-binary", "record-without-router"],
)
def test_recorded_launch_binaries_is_none_when_a_binary_cannot_be_named(
    tmp_path: Path, launches: list[dict], router: str
) -> None:
    from lilbee.providers.fleet.swap_manager import recorded_launch_binaries

    record_launch(tmp_path, "/b/srv")
    write_state(tmp_path, launches, name="chat.2", router=router)
    assert recorded_launch_binaries(tmp_path) is None


def test_recorded_launch_binaries_is_none_with_an_unreadable_file_or_no_file(
    tmp_path: Path,
) -> None:
    from lilbee.providers.fleet.swap_manager import recorded_launch_binaries

    assert recorded_launch_binaries(tmp_path) is None
    (tmp_path / "llama-swap.state.chat.3.json").write_text("{", encoding="utf-8")
    assert recorded_launch_binaries(tmp_path) is None


def test_launch_uses_engine_counts_a_live_user_whose_state_file_is_unreadable(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    (tmp_path / "slot").mkdir()
    (tmp_path / "slot" / "llama-swap.state.chat.1.json").write_text("{", encoding="utf-8")
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.launch_uses_engine(tmp_path / "engines", "fx", "1", "cpu") is True
    finally:
        hold.release_and_check_last()


def test_launch_uses_engine_reads_a_binary_path_as_written(tmp_path: Path, monkeypatch) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    root = tmp_path / "engines"
    build = root / "fx" / "1-cpu"
    (build / "bin").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (build / "bin" / "link").symlink_to(tmp_path / "outside")
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    hold = hold_user_lock(tmp_path / "slot")
    try:
        record_launch(tmp_path / "slot", build / "bin" / "link" / "srv")
        assert engine_store.launch_uses_engine(root, "fx", "1", "cpu") is True
        record_launch(tmp_path / "slot", build / ".." / ".." / "outside" / "srv")
        assert engine_store.launch_uses_engine(root, "fx", "1", "cpu") is False
    finally:
        hold.release_and_check_last()


def test_launch_uses_engine_is_false_with_no_user_of_the_machine_engine(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    record_launch(tmp_path / "slot", tmp_path / "engines" / "fx" / "1-cpu" / "bin" / "srv")
    assert engine_store.launch_uses_engine(tmp_path / "engines", "fx", "1", "cpu") is False


def test_launch_uses_engine_counts_a_user_that_recorded_no_launch_yet(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.launch_uses_engine(tmp_path / "engines", "fx", "1", "cpu") is True
    finally:
        hold.release_and_check_last()


def test_launch_uses_engine_names_the_build_a_launch_runs(tmp_path: Path, monkeypatch) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    root = tmp_path / "engines"
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    record_launch(tmp_path / "slot", root / "fx" / "2-cpu" / "bin" / "srv")
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.launch_uses_engine(root, "fx", "2", "cpu") is True
        assert engine_store.launch_uses_engine(root, "fx", "1", "cpu") is False
        assert engine_store.launch_uses_engine(root, "fx", "2", "cuda") is False
        assert engine_store.launch_uses_engine(root, "other", "2", "cpu") is False
    finally:
        hold.release_and_check_last()


def test_remove_keeps_the_running_build_and_deletes_another_of_the_engine(
    store: EngineStore, tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    installed_store(store, tmp_path, ("1", "2"))
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    record_launch(tmp_path / "slot", store.root / "fx" / "2-cpu" / "bin" / "srv")
    in_use = partial(engine_store.launch_uses_engine, store.root)
    hold = hold_user_lock(tmp_path / "slot")
    try:
        store.remove("fx", "1", "cpu", in_use=in_use)
        with pytest.raises(engine_store.EngineInUseError, match="is in use"):
            store.remove("fx", "2", "cpu", in_use=in_use)
    finally:
        hold.release_and_check_last()
    assert names(store.root / "fx") == ["2-cpu"]


def test_prune_engines_keeps_the_running_build_and_removes_the_stale_one(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    store = EngineStore(engine_store.engines_root(tmp_path / "models"))
    installed_store(store, tmp_path, ("1", "2", "3"))
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    record_launch(tmp_path / "slot", store.root / "fx" / "2-cpu" / "bin" / "srv")
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.choice.release_pins", lambda _b: {("fx", "3", "cpu")}
    )
    hold = hold_user_lock(tmp_path / "slot")
    try:
        gone = engine_store.prune_engines(tmp_path / "models")
    finally:
        hold.release_and_check_last()
    assert [p.name for p in gone] == ["1-cpu"]
    assert names(store.root / "fx") == ["2-cpu", "3-cpu"]


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


def helper_wheel(path: Path, members: dict[str, int]) -> str:
    """A wheel holding each name of *members* under ``p/bin/`` with that file mode."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, kind in members.items():
            info = zipfile.ZipInfo(f"p/bin/{name}")
            info.external_attr = (kind | 0o755) << 16
            archive.writestr(info, "#!/bin/sh\n")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_names_the_server_when_a_helper_sorts_first(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "h.whl"
    sha = helper_wheel(path, {"gguf-parser": stat.S_IFREG, "server": stat.S_IFREG})
    store.install_from_wheel(pin(sha), path)
    [found] = store.installed()
    assert found.binary == store.root / "fx" / "1-cpu" / "p" / "bin" / "server"


def test_a_wheel_without_the_server_is_refused_beside_a_helper(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "h.whl"
    sha = helper_wheel(path, {"gguf-parser": stat.S_IFREG, "server.txt": stat.S_IFREG})
    with pytest.raises(UnsafeWheelError, match="has no server in its bin directory"):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []


def test_a_linked_server_is_refused_beside_a_regular_helper(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "h.whl"
    sha = helper_wheel(path, {"gguf-parser": stat.S_IFREG, "server": stat.S_IFLNK})
    with pytest.raises(UnsafeWheelError, match="as a symbolic link"):
        store.install_from_wheel(pin(sha), path)
    assert_no_build_or_copy(store.root)


def test_a_pin_with_no_tool_is_refused_when_read() -> None:
    record = {
        "engine": "fx",
        "version": "1",
        "backend": "cpu",
        "platforms": {},
        "licenses": ["MIT"],
        "sha256": "a" * 64,
    }
    with pytest.raises(PinError, match="tool"):
        pin_lock(record)


def test_a_pin_lock_cannot_be_built_without_a_tool() -> None:
    with pytest.raises(TypeError, match="tool"):
        PinLock("fx", "1", "cpu", {}, ("MIT",), "a" * 64)  # type: ignore[call-arg]


def _ggml_wheel(path: Path, *, mode: int = 0o755, files: tuple[str, ...] = ()) -> str:
    """A wheel whose server has *mode*; *files* sit in bin, or in the folder they name."""
    with zipfile.ZipFile(path, "w") as archive:
        server = zipfile.ZipInfo("p/bin/server")
        server.external_attr = (stat.S_IFREG | mode) << 16
        archive.writestr(server, "#!/bin/sh\n")
        for name in files:
            archive.writestr(f"p/{name}" if "/" in name else f"p/bin/{name}", "x")
        archive.writestr("p-1.dist-info/METADATA", "Name: p\nLicense-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_wheel_with_a_runnable_server_and_its_pinned_library_installs(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "g.whl"
    sha = _ggml_wheel(path, files=("libcore.so.0.9.5", "libextra.so"))
    store.install_from_wheel(pin(sha, libraries=("core",)), path)
    assert [found.binary for found in store.installed()] == [
        store.root / "fx" / "1-cpu" / "p" / "bin" / "server"
    ]


def test_a_server_without_the_execute_bit_is_refused(store: EngineStore, tmp_path: Path) -> None:
    path = tmp_path / "g.whl"
    sha = _ggml_wheel(path, mode=0o644, files=("libggml-base.so.0",))
    with pytest.raises(UnsafeWheelError, match="without the execute bit"):
        store.install_from_wheel(pin(sha), path)
    assert store.installed() == []


def test_a_wheel_without_a_pinned_library_is_refused_by_name(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "g.whl"
    sha = _ggml_wheel(path, files=("libextra.so", "libcore-extra.so"))
    with pytest.raises(UnsafeWheelError, match="no library file for 'core' in its p folder"):
        store.install_from_wheel(pin(sha, libraries=("core",)), path)
    assert store.installed() == []


def test_a_library_outside_bin_counts_for_the_pinned_library(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "g.whl"
    sha = _ggml_wheel(path, files=("lib/libcore.so.1",))
    store.install_from_wheel(pin(sha, libraries=("core",)), path)
    assert len(store.installed()) == 1


def test_a_wheel_with_no_pinned_libraries_installs_without_any(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "g.whl"
    sha = _ggml_wheel(path)
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


# --- the tool is a regular file directly under bin/ ---------------------------------------


def make_pack_wheel(path: Path) -> str:
    """A wheel with bin/lib/libx.so.1 beside bin/server, as a cuda pack lays it out."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("lilbee_engine_fx/bin/lib/libx.so.1", "lib")
        info = zipfile.ZipInfo("lilbee_engine_fx/bin/server")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "#!/bin/sh\n")
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_zip_route_records_bin_server_beside_a_bin_lib_folder(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "pack.whl"
    sha = make_pack_wheel(path)
    build = store.install_from_wheel(pin(sha), path)
    assert [b.binary for b in store.installed()] == [build / "lilbee_engine_fx" / "bin" / "server"]


class _PinnedAdapter:
    """A stand-in adapter that is not bundled and pins one build."""

    bundled = False

    def __init__(self, version: str, backend: object) -> None:
        self._pin = {"engine": "fx", "version": version, "backend": backend}

    def pin_record(self) -> dict[str, object]:
        return self._pin


def _pruned_models(tmp_path: Path, monkeypatch, adapter: object, versions: tuple[str, ...]) -> Path:
    """A models folder with an engines folder beside it holding fx builds of *versions*."""
    from lilbee.providers.fleet.engines import choice
    from lilbee.providers.fleet.engines.base import EngineName

    monkeypatch.setitem(choice.ADAPTERS, EngineName.MISTRAL, adapter)
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path / "data"))
    for var in ("XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        monkeypatch.setenv(var, str(tmp_path / "xdg"))
    installed_store(EngineStore(tmp_path / "engines"), tmp_path, versions)
    return tmp_path / "models"


def _versions(tmp_path: Path) -> list[str]:
    return [b.version for b in EngineStore(tmp_path / "engines").installed()]


def test_prune_engines_removes_a_build_no_release_pins(tmp_path: Path, monkeypatch) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1", "3"))
    gone = engine_store.prune_engines(models)
    assert [p.name for p in gone] == ["1-cpu"]
    assert _versions(tmp_path) == ["3"]


def test_prune_engines_keeps_a_build_a_live_process_uses(tmp_path: Path, monkeypatch) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1", "3"))
    hold = hold_user_lock(tmp_path / "slot")
    try:
        assert engine_store.prune_engines(models) == []
    finally:
        hold.release_and_check_last()
    assert _versions(tmp_path) == ["1", "3"]


def test_prune_engines_keeps_every_build_of_an_engine_whose_pin_names_no_backend(
    tmp_path: Path, monkeypatch
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", None), ("1", "3"))
    assert engine_store.prune_engines(models) == []
    assert _versions(tmp_path) == ["1", "3"]


def test_prune_engines_removes_nothing_when_no_adapter_needs_a_build(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.providers.fleet.engines import choice

    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    monkeypatch.setattr(choice, "ADAPTERS", {})
    assert engine_store.prune_engines(models) == []
    assert _versions(tmp_path) == ["1"]


def test_prune_engines_ignores_a_relative_models_dir(tmp_path: Path, monkeypatch) -> None:
    _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    monkeypatch.chdir(tmp_path)
    assert engine_store.prune_engines(Path("models")) == []
    assert _versions(tmp_path) == ["1"]


def test_prune_engines_reads_no_pin_with_nothing_installed(tmp_path: Path, monkeypatch) -> None:
    from lilbee.providers.fleet.engines import choice

    probe = mock.Mock(side_effect=AssertionError("pin read"))
    monkeypatch.setattr(choice, "ADAPTERS", {"x": mock.Mock(bundled=False, pin_record=probe)})
    assert engine_store.prune_engines(tmp_path / "models") == []
    probe.assert_not_called()


def test_prune_keeps_a_build_an_installer_holds_locked(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1",))
    with mock.patch.object(engine_store, "file_lock", side_effect=FileLockUnavailableError("held")):
        assert store.prune(keep={("fx", "9", "cpu")}, in_use=lambda *_: False) == []
    assert [b.version for b in store.installed()] == ["1"]


def test_prune_engines_sweeps_nothing_itself(tmp_path: Path, monkeypatch) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("1", "cpu"), ("1",))
    (tmp_path / "engines" / "fx" / ".stage-x").mkdir()
    assert engine_store.prune_engines(models) == []
    assert (tmp_path / "engines" / "fx" / ".stage-x").is_dir()


def test_every_shipped_pin_names_a_build_or_no_backend() -> None:
    from lilbee.providers.fleet.engines.choice import ADAPTERS

    pinned = [a.pin_record() for a in ADAPTERS.values() if not a.bundled]
    assert pinned
    for record in pinned:
        assert isinstance(record["engine"], str) and record["engine"]
        assert isinstance(record["version"], str) and record["version"]
        assert record["backend"] is None or isinstance(record["backend"], str)


def test_the_startup_hook_sweeps_a_stage_and_prunes_a_stale_build(
    tmp_path: Path, monkeypatch
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli.app import app
    from lilbee.core.config import cfg

    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1", "3"))
    (tmp_path / "engines" / "fx" / ".stage-x").mkdir()
    monkeypatch.setattr(cfg, "models_dir", models)
    CliRunner().invoke(app, ["version"])
    assert not (tmp_path / "engines" / "fx" / ".stage-x").exists()
    assert _versions(tmp_path) == ["3"]


def test_prune_engines_keeps_every_installed_backend_of_the_current_pin(
    tmp_path: Path, monkeypatch
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1", "3"))
    store = EngineStore(tmp_path / "engines")
    wheel_path = tmp_path / "fx.whl"
    sha = make_wheel(wheel_path)
    store.install_from_wheel(PinLock("fx", "3", "cuda", {}, ("MIT",), sha, "server"), wheel_path)
    gone = engine_store.prune_engines(models)
    assert [p.name for p in gone] == ["1-cpu"]
    assert sorted(p.name for p in (tmp_path / "engines" / "fx").iterdir()) == ["3-cpu", "3-cuda"]


_HYPHEN_BACKEND = "cuda128-sm89"


def _install_hyphen_backend(tmp_path: Path, version: str = "3") -> Path:
    """Install an fx build whose backend holds a hyphen; return the build directory."""
    path = tmp_path / "fx-hyphen.whl"
    sha = make_wheel(path)
    lock = PinLock("fx", version, _HYPHEN_BACKEND, {}, ("MIT",), sha, "server")
    return EngineStore(tmp_path / "engines").install_from_wheel(lock, path)


def test_prune_engines_keeps_the_current_pin_when_its_backend_holds_a_hyphen(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = _PinnedAdapter("3", _HYPHEN_BACKEND)
    models = _pruned_models(tmp_path, monkeypatch, adapter, ("1",))
    build = _install_hyphen_backend(tmp_path)
    assert [p.name for p in engine_store.prune_engines(models)] == ["1-cpu"]
    assert build.is_dir()
    assert [(b.version, b.backend) for b in EngineStore(tmp_path / "engines").installed()] == [
        ("3", _HYPHEN_BACKEND)
    ]


def test_a_cli_start_keeps_an_install_whose_backend_holds_a_hyphen(
    tmp_path: Path, monkeypatch
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli.app import app
    from lilbee.core.config import cfg

    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", _HYPHEN_BACKEND), ("1", "3"))
    build = _install_hyphen_backend(tmp_path)
    monkeypatch.setattr(cfg, "models_dir", models)
    stale = tmp_path / "engines" / "fx" / "1-cpu"
    CliRunner().invoke(app, ["version"])
    assert (build / RECORD_NAME).is_file()
    assert (tmp_path / "engines" / "fx" / "3-cpu").is_dir()
    assert not stale.exists()


def test_build_dirs_report_the_record_of_a_complete_build(
    store: EngineStore, tmp_path: Path
) -> None:
    build = _install_hyphen_backend(tmp_path)
    assert store.build_dirs() == [("fx", "3", _HYPHEN_BACKEND, build)]


def test_build_dirs_report_a_hyphenated_version_from_its_record(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "fx-rc.whl"
    lock = PinLock("fx", "1-rc1", "cpu", {}, ("MIT",), make_wheel(path), "server")
    build = store.install_from_wheel(lock, path)
    assert store.build_dirs() == [("fx", "1-rc1", "cpu", build)]


def test_build_dirs_list_a_damaged_build_once_for_each_hyphen(store: EngineStore) -> None:
    damaged = store.root / "fx" / f"3-{_HYPHEN_BACKEND}"
    damaged.mkdir(parents=True)
    for unreadable in ("-cpu", "3-", "3"):
        (store.root / "fx" / unreadable).mkdir()
    assert store.build_dirs() == [
        ("fx", "3", _HYPHEN_BACKEND, damaged),
        ("fx", "3-cuda128", "sm89", damaged),
    ]


def test_prune_keeps_a_damaged_build_when_a_launch_uses_any_reading(store: EngineStore) -> None:
    damaged = store.root / "fx" / f"1-{_HYPHEN_BACKEND}"
    damaged.mkdir(parents=True)
    gone = store.prune(
        keep={("fx", "9", "cpu")},
        in_use=lambda engine, version, backend: (version, backend) == ("1-cuda128", "sm89"),
    )
    assert gone == []
    assert damaged.is_dir()


@pytest.mark.parametrize(
    ("version", "backend"),
    [
        ("git-0123456789ab", "cpu"),
        ("1.0.0-rc1", "cpu"),
        ("1.0.0-rc1", _HYPHEN_BACKEND),
    ],
)
def test_prune_engines_keeps_a_pinned_build_with_no_record(
    tmp_path: Path, monkeypatch, version: str, backend: str
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter(version, backend), ("1",))
    pinned = tmp_path / "engines" / "fx" / f"{version}-{backend}"
    pinned.mkdir()
    stale = tmp_path / "engines" / "fx" / f"0.1-{_HYPHEN_BACKEND}"
    stale.mkdir()
    assert sorted(p.name for p in engine_store.prune_engines(models)) == sorted(
        ["1-cpu", stale.name]
    )
    assert pinned.is_dir()


@pytest.mark.parametrize(("pin_version", "stale"), [("3", "3-1-cpu"), ("1.0.0", "1.0.0-rc1-cpu")])
def test_prune_engines_keeps_a_stale_build_with_no_record_that_starts_with_the_pinned_version(
    tmp_path: Path, monkeypatch, pin_version: str, stale: str
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter(pin_version, "cpu"), ("1",))
    ambiguous = tmp_path / "engines" / "fx" / stale
    ambiguous.mkdir()
    assert [p.name for p in engine_store.prune_engines(models)] == ["1-cpu"]
    assert ambiguous.is_dir()


def test_prune_goes_on_after_a_build_it_must_keep(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1", "2"))
    gone = store.prune(keep={("fx", "9", "cpu")}, in_use=lambda e, v, b: v == "1")
    assert [p.name for p in gone] == ["2-cpu"]
    assert (store.root / "fx" / "1-cpu").is_dir()


def test_build_dirs_skip_hidden_stage_and_trash_folders(store: EngineStore) -> None:
    for hidden in (".stage-1", ".trash-1"):
        (store.root / "fx" / hidden).mkdir(parents=True)
    assert store.build_dirs() == []


def test_build_dirs_skip_a_file_beside_the_builds(store: EngineStore) -> None:
    (store.root / "fx").mkdir(parents=True)
    (store.root / "fx" / "0-note").write_text("x", encoding="utf-8")
    build = store.root / "fx" / "5-cpu"
    build.mkdir()
    assert store.build_dirs() == [("fx", "5", "cpu", build)]


def test_build_dirs_list_every_complete_build_of_an_engine(
    store: EngineStore, tmp_path: Path
) -> None:
    installed_store(store, tmp_path, ("1", "2"))
    assert [(v, b, p.name) for _, v, b, p in store.build_dirs()] == [
        ("1", "cpu", "1-cpu"),
        ("2", "cpu", "2-cpu"),
    ]


def test_build_dirs_read_a_one_letter_backend(store: EngineStore) -> None:
    build = store.root / "fx" / "3-x"
    build.mkdir(parents=True)
    assert store.build_dirs() == [("fx", "3", "x", build)]


def test_prune_does_not_wait_for_an_install_lock(store: EngineStore, tmp_path: Path) -> None:
    installed_store(store, tmp_path, ("1",))
    seen: list[float] = []
    real = engine_store.file_lock

    def spy(path, timeout):
        seen.append(timeout)
        return real(path, timeout)

    with mock.patch.object(engine_store, "file_lock", spy):
        store.prune(keep={("fx", "9", "cpu")}, in_use=lambda *_: False)
    assert seen == [0.0]


def test_prune_engines_survives_an_unreadable_engines_root(tmp_path: Path, monkeypatch) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    with mock.patch.object(EngineStore, "build_dirs", side_effect=PermissionError("denied")):
        assert engine_store.prune_engines(models) == []
    assert _versions(tmp_path) == ["1"]


def test_prune_leaves_a_folder_named_bundled_alone(store: EngineStore, tmp_path: Path) -> None:
    (store.root / "fx" / "1-bundled").mkdir(parents=True)
    assert store.prune({("fx", "9", "cpu")}, in_use=lambda *_: False) == []
    assert (store.root / "fx" / "1-bundled").is_dir()


def test_prune_engines_never_prunes_a_build_of_a_bundled_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    from lilbee.providers.fleet.engines import choice
    from lilbee.providers.fleet.engines.base import EngineName

    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    shipped = mock.Mock(bundled=True)
    shipped.name = "fx"
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MISTRAL, shipped)
    other = _PinnedAdapter("9", "cpu")
    other._pin["engine"] = "other"
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, other)
    (tmp_path / "engines" / "other" / "1-cpu").mkdir(parents=True)
    gone = engine_store.prune_engines(models)
    assert [p.parent.name for p in gone] == ["other"]
    assert _versions(tmp_path) == ["1"]


def test_prune_engines_removes_a_store_holding_only_an_old_pin(tmp_path: Path, monkeypatch) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    gone = engine_store.prune_engines(models)
    assert [p.name for p in gone] == ["1-cpu"]
    assert _versions(tmp_path) == []


def test_prune_engines_lets_a_probe_error_reach_the_caller(tmp_path: Path, monkeypatch) -> None:
    class Probing(_PinnedAdapter):
        def pin_record(self) -> dict[str, object]:
            raise OSError("host probe failed")

    models = _pruned_models(tmp_path, monkeypatch, Probing("3", "cpu"), ("1",))
    with pytest.raises(OSError, match="host probe failed"):
        engine_store.prune_engines(models)
    assert _versions(tmp_path) == ["1"]


def test_prune_engines_lets_an_error_other_than_oserror_reach_the_caller(
    tmp_path: Path, monkeypatch
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    with (
        mock.patch.object(EngineStore, "prune", side_effect=RuntimeError("bug")),
        pytest.raises(RuntimeError, match="bug"),
    ):
        engine_store.prune_engines(models)


def test_prune_engines_survives_an_oserror_while_pruning(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    models = _pruned_models(tmp_path, monkeypatch, _PinnedAdapter("3", "cpu"), ("1",))
    with (
        mock.patch.object(EngineStore, "prune", side_effect=PermissionError("denied")),
        caplog.at_level("WARNING"),
    ):
        assert engine_store.prune_engines(models) == []
    assert str(tmp_path / "engines") in caplog.text
    assert _versions(tmp_path) == ["1"]


def test_the_install_record_names_the_build_and_the_time_of_the_install_in_utc(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    build = store.install_from_wheel(pin(sha), path)
    record = json.loads((build / RECORD_NAME).read_text(encoding="utf-8"))
    assert set(record) == {
        "engine",
        "version",
        "backend",
        "sha256",
        "licences",
        "installed_at",
        "binary",
    }
    assert record["installed_at"].endswith("+00:00")
    assert [e.installed_at for e in store.installed()] == [record["installed_at"]]


def _on_first_open_rewrite(monkeypatch: pytest.MonkeyPatch, path: Path, other: bytes) -> list[Path]:
    """Overwrite *path* with *other* in place after the first archive opens; list every open."""
    opened: list[Path] = []

    class RewritingZipFile(zipfile.ZipFile):
        def __init__(self, file: Path, *args: Any, **kwargs: Any) -> None:
            super().__init__(file, *args, **kwargs)
            if not opened:
                path.write_bytes(other)
            opened.append(Path(file))

    monkeypatch.setattr(zipfile, "ZipFile", RewritingZipFile)
    return opened


def test_a_wheel_rewritten_after_the_check_installs_the_checked_bytes(
    store: EngineStore, wheel: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = wheel
    other = tmp_path / "other.whl"
    make_wheel(other, body="#!/bin/sh\necho swapped\n")
    opened = _on_first_open_rewrite(monkeypatch, path, other.read_bytes())
    build = store.install_from_wheel(pin(sha), path)
    assert path.read_bytes() == other.read_bytes()
    server = build / "lilbee_engine_fx" / "bin" / "server"
    assert server.read_text(encoding="utf-8") == "#!/bin/sh\n"
    assert len(opened) == 3
    assert path not in opened
    assert all(p.is_relative_to(store.root) for p in opened)


@pytest.mark.parametrize(
    ("licence", "good_digest", "failure"),
    [("MIT", False, ChecksumError), ("GPL-3.0-only", True, LicenceRefusedError)],
    ids=["wrong-digest", "licence"],
)
def test_the_private_copy_is_gone_after_a_failed_install(
    store: EngineStore,
    tmp_path: Path,
    licence: str,
    good_digest: bool,
    failure: type[Exception],
) -> None:
    path = tmp_path / "w.whl"
    sha = make_wheel(path, licence)
    with pytest.raises(failure):
        store.install_from_wheel(pin(sha if good_digest else "0" * 64), path)
    assert_no_build_or_copy(store.root)


class _ReadSpy:
    """Wrap a read handle and record the size asked by each read."""

    def __init__(self, handle: Any, reads: list[int]) -> None:
        self._handle = handle
        self._reads = reads

    def __enter__(self) -> _ReadSpy:
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()

    def read(self, size: int = -1) -> bytes:
        self._reads.append(size)
        data: bytes = self._handle.read(size)
        return data


def test_the_wheel_is_copied_one_mebibyte_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "big.whl"
    payload = b"x" * (5 * 1024 * 1024 // 2)
    source.write_bytes(payload)
    reads: list[int] = []
    real_open = Path.open

    def spy_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, mode, *args, **kwargs)
        return _ReadSpy(handle, reads) if self == source else handle

    monkeypatch.setattr(Path, "open", spy_open)
    target = tmp_path / "copy.whl"
    digest = engine_store.copy_with_digest(source, target)
    assert reads == [1024 * 1024] * 4
    assert target.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
