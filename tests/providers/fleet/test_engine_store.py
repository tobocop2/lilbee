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
from lilbee.providers.fleet.engines.choice import PinError, pin_lock


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
    pin = es.PinLock("fx", "1", "cpu", {}, ("MIT",), sha, "server")
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
    taken: list[Path] = []
    real = engine_store.file_lock

    def spy(lock_path: Path, timeout: float):  # type: ignore[no-untyped-def]
        taken.append(lock_path)
        return real(lock_path, timeout)

    with mock.patch.object(engine_store, "file_lock", spy):
        store.install_from_wheel(pin(sha), path)
    assert taken == [store.root / ".locks" / "fx"]
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


def test_checksum_error_names_engine_version_and_backend_and_the_next_step(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    want = (
        "The file for engine fx 1 (cpu) does not match its pin, so lilbee did not install it. "
        "Run the install again. If this happens again, report it."
    )
    with pytest.raises(ChecksumError) as caught:
        store.install_from_wheel(pin("0" * 64), wheel[0])
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
        ("", False),
        ("\n\n", False),
    ],
)
def test_the_mpl_source_pointer_names_the_crate_download_and_its_checksum(
    text: str, ok: bool
) -> None:
    assert bool(engine_store._verified_notices([("SOURCE-MPL-2.0", text)])) is ok


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


def _staged_cuda_tree(stage: Path, *, notice: bool) -> None:
    (stage / "p" / "bin").mkdir(parents=True)
    server = stage / "p" / "bin" / "server"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    (stage / "p-1.dist-info" / "licenses").mkdir(parents=True)
    (stage / "p-1.dist-info" / "METADATA").write_text(
        f"Name: p\nLicense-Expression: {_CUDA_EXPRESSION}\n", encoding="utf-8"
    )
    if notice:
        (stage / "p-1.dist-info" / "licenses" / "NOTICE-NVIDIA-CUDA").write_text(
            "notice", encoding="utf-8"
        )


def test_a_staged_cuda_tree_with_its_notice_passes_the_check(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=True)
    assert engine_store._check_staged_tree(pin("0" * 64), tmp_path) == "p/bin/server"


def test_a_staged_cuda_notice_with_a_stray_byte_still_passes(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=True)
    (tmp_path / "p-1.dist-info" / "licenses" / "NOTICE-NVIDIA-CUDA").write_bytes(b"notice \xff")
    assert engine_store._check_staged_tree(pin("0" * 64), tmp_path) == "p/bin/server"


def test_a_staged_tree_without_metadata_is_refused(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=True)
    (tmp_path / "p-1.dist-info" / "METADATA").unlink()
    with pytest.raises(LicenceRefusedError, match="declares the licence ''"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)


def test_a_staged_cuda_tree_with_an_empty_notice_is_refused(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=True)
    (tmp_path / "p-1.dist-info" / "licenses" / "NOTICE-NVIDIA-CUDA").write_text(
        "", encoding="utf-8"
    )
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)


def test_a_staged_mpl_tree_needs_its_source_pointer(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=False)
    metadata = tmp_path / "p-1.dist-info" / "METADATA"
    metadata.write_text("Name: p\nLicense-Expression: MIT AND MPL-2.0\n", encoding="utf-8")
    with pytest.raises(LicenceRefusedError, match=r"MPL-2\.0"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)
    pointer = tmp_path / "p-1.dist-info" / "licenses" / "SOURCE-MPL-2.0"
    pointer.write_text("https://github.com/mozilla/option-ext", encoding="utf-8")
    with pytest.raises(LicenceRefusedError, match=r"MPL-2\.0"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)
    pointer.write_text(f"{_CRATE_URL} {_CHECKSUM}\n", encoding="utf-8")
    assert engine_store._check_staged_tree(pin("0" * 64), tmp_path) == "p/bin/server"


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
    ],
)
def test_a_wheel_with_permissive_licence_files_still_installs(
    store: EngineStore, tmp_path: Path, member: str, text: str
) -> None:
    path = tmp_path / "ok.whl"
    sha = _cuda_wheel(path, notice=False, licence="MIT", extras=((member, text),))
    store.install_from_wheel(pin(sha), path)
    assert len(store.installed()) == 1


@pytest.mark.parametrize(
    ("member", "text"),
    [
        ("p-1.dist-info/licenses/COPYING.LESSER", "any text"),
        ("p-1.dist-info/licenses/COPYING", _GPL_TEXT),
        ("p.libs/libquadmath-96973f99.so.0.0.0", "ELF"),
        ("p/docs/gpl-3.0.txt", _GPL_TEXT),
        ("p/THIRD-PARTY-LICENSES.txt", _LGPL_TEXT),
        ("p/COPYING.GPLv2", "x"),
        ("p/COPYING", _WRAPPED_GPL),
        ("p/COPYING", _UTF16_GPL),
        ("p/lib/libhistory.so.8", "ELF"),
        ("p/lib/libgmp.a", "ar archive"),
    ],
)
def test_a_staged_tree_with_a_copyleft_file_is_refused_whatever_it_declares(
    tmp_path: Path, member: str, text: str | bytes
) -> None:
    _staged_cuda_tree(tmp_path, notice=False)
    (tmp_path / "p-1.dist-info" / "METADATA").write_text(
        "Name: p\nLicense-Expression: MIT\n", encoding="utf-8"
    )
    target = tmp_path / member
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    with pytest.raises(LicenceRefusedError, match="GPL"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)


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


def test_the_real_nvidia_cuda_notice_in_a_staged_tree_passes(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=True)
    (tmp_path / "p-1.dist-info" / "licenses" / "NOTICE-NVIDIA-CUDA").write_bytes(
        _CUDA_NOTICE.read_bytes()
    )
    assert engine_store._check_staged_tree(pin("0" * 64), tmp_path) == "p/bin/server"


def test_a_staged_tree_with_a_permissive_licence_file_passes(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=False)
    (tmp_path / "p-1.dist-info" / "METADATA").write_text(
        "Name: p\nLicense-Expression: MIT\n", encoding="utf-8"
    )
    (tmp_path / "p-1.dist-info" / "licenses" / "COPYING").write_text("MIT", encoding="utf-8")
    assert engine_store._check_staged_tree(pin("0" * 64), tmp_path) == "p/bin/server"


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
    workflow = (
        Path(__file__).parents[3] / ".github" / "workflows" / "llama-pack-cells.yml"
    ).read_text(encoding="utf-8")
    name = engine_store.NOTICE_RULES["LicenseRef-Microsoft-VC-Runtime"].file
    assert workflow.count(name) == 2
    assert workflow.lower().count(name.lower()) == 2


def test_a_staged_cuda_tree_without_its_notice_is_refused(tmp_path: Path) -> None:
    _staged_cuda_tree(tmp_path, notice=False)
    with pytest.raises(LicenceRefusedError, match="LicenseRef-NVIDIA-CUDA-EULA"):
        engine_store._check_staged_tree(pin("0" * 64), tmp_path)


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
    assert not store.root.exists()


def test_package_route_records_the_server_when_a_helper_sorts_first(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "h.whl"
    sha = helper_wheel(path, {"gguf-parser": stat.S_IFREG, "server": stat.S_IFREG})
    run, _ = extract_into_target(path)
    store.install_via_package(pin(sha), run=run)
    [found] = store.installed()
    assert found.binary == store.root / "fx" / "1-cpu" / "p" / "bin" / "server"


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


def test_package_route_records_bin_server_beside_a_bin_lib_folder(
    store: EngineStore, tmp_path: Path
) -> None:
    path = tmp_path / "pack.whl"
    sha = make_pack_wheel(path)
    run, _ = extract_into_target(path)
    build = store.install_via_package(pin(sha), run=run)
    assert [b.binary for b in store.installed()] == [build / "lilbee_engine_fx" / "bin" / "server"]


def test_package_route_does_not_take_a_folder_named_like_the_tool(
    store: EngineStore, wheel: tuple[Path, str]
) -> None:
    path, sha = wheel
    inner, _ = extract_into_target(path)

    def run(argv: list[str], requirements: str) -> None:
        inner(argv, requirements)
        tool = Path(argv[argv.index("--target") + 1]) / "lilbee_engine_fx" / "bin" / "server"
        tool.unlink()
        tool.mkdir()

    store.install_via_package(pin(sha), run=run)
    assert [b.binary for b in store.installed()] == [None]


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
    from lilbee.core.security import FileLockUnavailableError

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
