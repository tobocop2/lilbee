"""The scripts that build the mistral.rs pack read archives and reports and refuse anything else."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import shutil
import stat
import struct
import subprocess
import sys
import sysconfig
import tarfile
import time
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from tests._pack_workflow import pack_cells, pack_workflow
from tests._pe_fixture import pe_file

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "wheel-build"
QA = Path(__file__).resolve().parents[1] / "tools" / "qa"
WHEEL_SERVER = QA.parents[1] / "src" / "lilbee" / "providers" / "fleet" / "wheel_server.py"
PACKAGING = Path(__file__).resolve().parents[1] / "packaging" / "engine-wheel-mistral-rs"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repack = _load("repack_mistralrs")
rust_licences = _load("rust_licences")
rust_notices = _load("rust_notices")


def _tar(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_repack_places_the_tool(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"./mistralrs": b"bin"})
    repack.repack(archive, tmp_path / "bin")
    assert (tmp_path / "bin" / "mistralrs").read_bytes() == b"bin"
    # Windows has no execute bit: there, every file that exists passes.
    assert os.access(tmp_path / "bin" / "mistralrs", os.X_OK)


CUDA_LIBS = [
    "libcudart.so.12",
    "libnccl.so.2",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcurand.so.10",
]


def test_repack_places_the_five_cuda_libraries_beside_the_tool(tmp_path: Path) -> None:
    entries = {"mistralrs": b"bin", **{f"lib/{name}": name.encode() for name in CUDA_LIBS}}
    repack.repack(_tar(tmp_path / "a.tar.gz", entries), tmp_path / "bin")
    assert sorted(p.name for p in (tmp_path / "bin" / "lib").iterdir()) == sorted(CUDA_LIBS)
    assert (tmp_path / "bin" / "lib" / "libnccl.so.2").read_bytes() == b"libnccl.so.2"
    assert not os.access(tmp_path / "bin" / "lib" / "libnccl.so.2", os.X_OK)
    assert (tmp_path / "bin" / "mistralrs").read_bytes() == b"bin"


def test_repack_makes_no_lib_folder_for_a_cpu_archive(tmp_path: Path) -> None:
    repack.repack(_tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin"}), tmp_path / "bin")
    assert not (tmp_path / "bin" / "lib").exists()


@pytest.mark.parametrize(
    "name",
    ["lib/libother.so.1", "lib/sub/libcudart.so.12", "libcudart.so.12", "other/libnccl.so.2"],
)
def test_repack_refuses_a_library_that_is_not_one_of_the_five(tmp_path: Path, name: str) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin", name: b"rt"})
    with pytest.raises(SystemExit, match=r"is not the tool or a bundled library"):
        repack.repack(archive, tmp_path / "bin")
    assert not (tmp_path / "bin").exists()


def test_repack_refuses_a_linked_library(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tool = tarfile.TarInfo("mistralrs")
        tar.addfile(tool)
        info = tarfile.TarInfo("lib/libcudart.so.12")
        info.type = tarfile.SYMTYPE
        info.linkname = "libcudart.so.12.8"
        tar.addfile(info)
    with pytest.raises(SystemExit, match=r"libcudart\.so\.12 is a link"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_reads_a_zip_with_the_exe(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mistralrs.exe", b"exe")
    repack.repack(archive, tmp_path / "bin")
    assert (tmp_path / "bin" / "mistralrs.exe").read_bytes() == b"exe"


def test_repack_refuses_an_unknown_entry(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin", "extra.sh": b"x"})
    with pytest.raises(SystemExit, match=r"extra\.sh is not the tool"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_an_archive_without_the_tool(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {})
    with pytest.raises(SystemExit, match="expected one tool, found 0"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_an_archive_with_two_tools(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"a", "mistralrs.exe": b"b"})
    with pytest.raises(SystemExit, match="expected one tool, found 2"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_a_link(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.type = tarfile.SYMTYPE
        info.linkname = "other"
        tar.addfile(info)
    with pytest.raises(SystemExit, match="mistralrs is a link"):
        repack.repack(archive, tmp_path / "bin")


def _entry(licence_id: str, text: str, crate: str = "serde") -> dict:
    return {
        "id": licence_id,
        "name": f"{licence_id} licence",
        "text": text,
        "used_by": [
            {"crate": {"name": crate, "version": "1.0.0", "repository": "https://x.test/r"}}
        ],
    }


def test_rust_licences_write_a_file_for_each_text(tmp_path: Path) -> None:
    report = {
        "licenses": [
            _entry("MIT", "text one"),
            _entry("MIT", "text two", "rand"),
            _entry("ISC", "isc"),
        ]
    }
    assert rust_licences.write_licences(report, tmp_path) == ["ISC", "MIT"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ISC-001.txt",
        "MIT-001.txt",
        "MIT-002.txt",
    ]
    first = (tmp_path / "MIT-001.txt").read_text(encoding="utf-8")
    assert "serde 1.0.0 (https://x.test/r)" in first and first.rstrip().endswith("text one")


def test_rust_licences_refuse_an_empty_report(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="lists no licence text"):
        rust_licences.write_licences({"licenses": []}, tmp_path)


def test_rust_licences_refuse_an_empty_text(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="MIT text of serde is empty"):
        rust_licences.write_licences({"licenses": [_entry("MIT", "  ")]}, tmp_path)


_LOCK = """
[[package]]
name = "selectors"
version = "0.26.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc123"

[[package]]
name = "from-git"
version = "1.0.0"
source = "git+https://example.test/r#deadbeef"
"""


def _mpl_report(crate: str, version: str) -> dict:
    entry = _entry("MPL-2.0", "mpl text", crate)
    entry["used_by"][0]["crate"]["version"] = version
    return {"licenses": [entry, _entry("MIT", "mit")]}


def test_lock_checksums_keep_only_registry_packages() -> None:
    assert rust_notices.lock_checksums(_LOCK) == {("selectors", "0.26.0"): "abc123"}


def test_mpl_notice_names_the_exact_source_and_checksum() -> None:
    text = rust_notices.mpl_notice(
        _mpl_report("selectors", "0.26.0"), rust_notices.lock_checksums(_LOCK)
    )
    assert text is not None
    assert text.splitlines() == [
        "https://static.crates.io/crates/selectors/selectors-0.26.0.crate  abc123"
    ]
    assert "serde" not in text


def test_mpl_notice_is_absent_without_an_mpl_crate() -> None:
    assert rust_notices.mpl_notice({"licenses": [_entry("MIT", "mit")]}, {}) is None


def test_mpl_notice_refuses_a_crate_outside_the_registry() -> None:
    with pytest.raises(SystemExit, match=r"from-git 1\.0\.0 is not a crates\.io package"):
        rust_notices.mpl_notice(
            _mpl_report("from-git", "1.0.0"), rust_notices.lock_checksums(_LOCK)
        )


def test_vendored_files_take_notices_and_third_party_licences(tmp_path: Path) -> None:
    names = (
        "LICENSE",
        "NOTICE",
        "a/NOTICE.md",
        "a/third_party/x/LICENSE-BSD",
        "a/third_party/x/main.rs",
        "target/NOTICE",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("t", encoding="utf-8")
    found = [p.relative_to(tmp_path).as_posix() for p in rust_notices.vendored_files(tmp_path)]
    # The function sorts Path objects; Windows orders those without regard to case, so the
    # expectation sorts Path objects too.
    expected = ["NOTICE", "a/NOTICE.md", "a/third_party/x/LICENSE-BSD"]
    ordered = sorted(tmp_path / n for n in expected)
    assert found == [p.relative_to(tmp_path).as_posix() for p in ordered]


def test_write_notices_writes_each_file_into_the_tree(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "sub").mkdir(parents=True)
    (checkout / "sub" / "NOTICE").write_text("checkout notice", encoding="utf-8")
    crate = tmp_path / "registry" / "index-1" / "aws-lc-sys-1.0.0"
    crate.mkdir(parents=True)
    (crate / "NOTICE").write_text("crate notice", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    report = _mpl_report("selectors", "0.26.0")
    report["licenses"].append(_entry("OpenSSL", "ssl text", "aws-lc-sys"))
    names = rust_notices.write_notices(report, _LOCK, checkout, tmp_path / "registry", tree)
    assert names == [
        "SOURCE-MPL-2.0",
        "NOTICE-OpenSSL",
        "mistral.rs__sub__NOTICE",
        "aws-lc-sys-1.0.0__NOTICE",
    ]
    assert "Eric Young" in (tree / "NOTICE-OpenSSL").read_text(encoding="utf-8")
    vendored = tree / "THIRD-PARTY" / "vendored"
    assert (vendored / "aws-lc-sys-1.0.0__NOTICE").read_text(encoding="utf-8") == "crate notice"
    assert (vendored / "mistral.rs__sub__NOTICE").read_text(encoding="utf-8") == "checkout notice"


def _bash() -> str:
    """The bash on PATH, which runs the pack scripts on every OS.

    On Windows a bare ``bash`` starts the WSL launcher in System32 before PATH is searched.
    """
    found = shutil.which("bash")
    assert found is not None
    return found


def _env(tmp_path: Path, **env: str) -> dict[str, str]:
    """This environment with the running Python and the stubs of the test first on PATH."""
    path = os.pathsep.join(
        [str(tmp_path / "stubs"), str(Path(sys.executable).parent), os.environ["PATH"]]
    )
    return {**os.environ, "PATH": path, **env}


def _tree(tmp_path: Path) -> Path:
    """A copy of the files that the pack scripts read. Its chat check records its arguments."""
    root = tmp_path / "lilbee"
    if not root.exists():
        shutil.copytree(TOOLS, root / "tools" / "wheel-build")
        shutil.copytree(
            PACKAGING,
            root / "packaging" / PACKAGING.name,
            ignore=shutil.ignore_patterns("bin", "__pycache__", "*.egg-info"),
        )
        (root / "tools" / "qa").mkdir()
        shutil.copy(QA / "assert_engine_bundle.py", root / "tools" / "qa")
        # The script loads the wheel rules of the engine server from the source tree beside it.
        rule = root / "src" / "lilbee" / "providers" / "fleet"
        rule.mkdir(parents=True)
        shutil.copy(WHEEL_SERVER, rule)
        (root / "tools" / "qa" / "check_mistralrs_chat.py").write_text(
            "import pathlib, sys\n"
            "pathlib.Path(__file__).with_name('asked').write_text('\\n'.join(sys.argv[1:]))\n",
            encoding="utf-8",
        )
    return root


def _pack_stage(tmp_path: Path, stage: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run one stage of the pack script in the tree, under the shell options of the script."""
    return subprocess.run(
        [
            _bash(),
            "-c",
            f"set -euo pipefail; . tools/wheel-build/pack_mistralrs.sh; {stage}",
        ],
        cwd=_tree(tmp_path),
        env=_env(tmp_path, **env),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


_INFO = "lilbee_engine_mistral_rs-0.9.5.dist-info"
_LICENCES = f"{_INFO}/licenses"
_EXPRESSION = "MIT AND MPL-2.0 AND OpenSSL"
_HOST_PLATFORM = sysconfig.get_platform().replace("-", "_").replace(".", "_")
_HELD = {
    f"{_LICENCES}/LICENSE-mistral.rs": "mit",
    f"{_LICENCES}/SOURCE-MPL-2.0": "mpl",
    f"{_LICENCES}/NOTICE-OpenSSL": "ssl",
    f"{_LICENCES}/NOTICE-MICROSOFT-VC-RUNTIME": "vc",
    f"{_LICENCES}/THIRD-PARTY/mistral.rs/MIT-001.txt": "mit",
    f"{_LICENCES}/THIRD-PARTY/vendored/mistral.rs__NOTICE": "notice",
}


def _wheel(
    tmp_path: Path,
    drop: str | None = None,
    tool: bytes = b"#!/bin/sh\necho mistralrs 0.9.5\n",
    tool_mode: int = 0o755,
    tag: str = f"py3-none-{_HOST_PLATFORM}",
    name_tag: str = f"1.cpu-py3-none-{_HOST_PLATFORM}",
    expression: str = _EXPRESSION,
    newline: str = "\n",
    more: dict[str, bytes] | None = None,
    description: str = "",
) -> Path:
    """An installable mistral.rs wheel that a check passes, unless an argument breaks it."""
    files: dict[str, bytes] = {
        "lilbee_engine_mistral_rs/__init__.py": b"",
        "lilbee_engine_mistral_rs/bin/mistralrs": tool,
        f"{_INFO}/METADATA": (
            "Metadata-Version: 2.4\nName: lilbee-engine-mistral-rs\nVersion: 0.9.5\n"
            f"License-Expression: {expression}\n" + (f"\n{description}\n" if description else "")
        )
        .replace("\n", newline)
        .encode(),
        f"{_INFO}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            f"Tag: {tag}\nBuild: 1.cpu\n"
        )
        .replace("\n", newline)
        .encode(),
        **{name: text.encode() for name, text in _HELD.items()},
        **(more or {}),
    }
    if drop is not None:
        files = {name: data for name, data in files.items() if not name.startswith(drop)}
    files[f"{_INFO}/RECORD"] = "".join(f"{name},,\n" for name in files).encode()
    wheel = tmp_path / f"lilbee_engine_mistral_rs-0.9.5-{name_tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            mode = tool_mode if "/bin/mistralrs" in name else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, data)
    return wheel


def _check(
    tmp_path: Path,
    wheel: Path,
    stage: str = "main",
    crates: Path | None = None,
    expression: str = _EXPRESSION,
    **env: str,
) -> subprocess.CompletedProcess[str]:
    """Run the wheel check, or one stage of it, the way the pack script does."""
    if crates is None:
        crates = tmp_path / "crate-licences.txt"
        crates.write_text("MIT\nMPL-2.0\nOpenSSL\n", encoding="utf-8", newline="\n")
    script = _tree(tmp_path) / "tools" / "wheel-build" / "check_mistralrs_wheel.sh"
    args = [wheel.as_posix()]
    if stage in ("main", "check_licences"):
        args += [crates.as_posix(), expression]
    settings = {
        "RUNNER_OS": "Linux",
        "RUNNER_TEMP": str(tmp_path),
        "BACKEND": "cpu",
        "PLATFORM": _HOST_PLATFORM,
        "MISTRALRS_CHAT_MODEL": "owner/model",
        **env,
    }
    return subprocess.run(
        [
            _bash(),
            "-c",
            f'set -euo pipefail; script="$1"; shift; . "$script"; {stage} "$@"',
            "stage",  # $0 is not the script, so sourcing it does not run its main
            script.as_posix(),
            *args,
        ],
        cwd=tmp_path,
        env=_env(tmp_path, **settings),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize("runner", ["Linux", "macOS", "Windows"])
def test_the_check_passes_a_wheel_that_holds_every_licence(tmp_path: Path, runner: str) -> None:
    done = _check(tmp_path, _wheel(tmp_path), "check_licences", RUNNER_OS=runner)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("stage", ["check_tag", "check_licences"])
def test_the_check_reads_the_crlf_files_of_a_windows_build(tmp_path: Path, stage: str) -> None:
    done = _check(tmp_path, _wheel(tmp_path, newline="\r\n"), stage, RUNNER_OS="Windows")
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("missing", "named"),
    [
        ("LICENSE-mistral.rs", "the mistral.rs licence"),
        ("SOURCE-MPL-2.0", "SOURCE-MPL-2.0"),
        ("NOTICE-OpenSSL", "NOTICE-OpenSSL"),
        ("THIRD-PARTY/mistral.rs/", "the crate licences"),
        ("THIRD-PARTY/vendored/", "the vendored notices"),
    ],
)
def test_the_check_fails_a_wheel_that_lacks_a_licence(
    tmp_path: Path, missing: str, named: str
) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, drop=f"{_LICENCES}/{missing}"), "check_licences")
    assert bad.returncode == 1
    assert f"wheel lacks {named}" in bad.stderr


@pytest.mark.parametrize("kind", ["missing", "folder", "empty"])
def test_the_check_fails_without_the_list_of_crate_licences(tmp_path: Path, kind: str) -> None:
    crates = tmp_path / "list"
    if kind == "folder":
        crates.mkdir()
    elif kind == "empty":
        crates.touch()
    wheel = _wheel(tmp_path, drop=f"{_LICENCES}/NOTICE-OpenSSL")
    bad = _check(tmp_path, wheel, "check_licences", crates=crates)
    assert bad.returncode == 1
    assert "no crate licences in" in bad.stderr


def test_the_whole_check_stops_at_a_wheel_that_lacks_a_licence(tmp_path: Path) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, drop=f"{_LICENCES}/NOTICE-OpenSSL"))
    assert bad.returncode == 1
    assert "wheel lacks NOTICE-OpenSSL" in bad.stderr
    assert "mistralrs:" not in bad.stdout


def test_the_check_fails_a_wheel_with_another_licence_expression(tmp_path: Path) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, expression="MIT"), "check_licences")
    assert bad.returncode == 1
    assert f"wheel lacks the licence expression {_EXPRESSION}" in bad.stderr


def test_the_check_ignores_a_licence_expression_in_the_description(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, expression="MIT", description=f"License-Expression: {_EXPRESSION}")
    bad = _check(tmp_path, wheel, "check_licences")
    assert bad.returncode == 1
    assert f"wheel lacks the licence expression {_EXPRESSION}" in bad.stderr


@pytest.mark.parametrize(("runner", "status"), [("Linux", 0), ("macOS", 0), ("Windows", 1)])
def test_only_a_windows_wheel_needs_the_runtime_notice(
    tmp_path: Path, runner: str, status: int
) -> None:
    wheel = _wheel(tmp_path, drop=f"{_LICENCES}/NOTICE-MICROSOFT")
    done = _check(tmp_path, wheel, "check_licences", RUNNER_OS=runner)
    assert done.returncode == status
    assert ("wheel lacks the Visual C++ runtime notice" in done.stderr) == bool(status)


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ({"name_tag": f"1.cuda-py3-none-{_HOST_PLATFORM}"}, "wrong tag on"),
        ({"name_tag": "1.cpu-py3-none-any"}, "wrong tag on"),
        ({"name_tag": f"2.cpu-py3-none-{_HOST_PLATFORM}"}, "wrong tag on"),
        ({"tag": "py3-none-any"}, "lacks the tag py3-none-"),
    ],
)
def test_the_whole_check_fails_a_wheel_with_a_wrong_tag(
    tmp_path: Path, broken: dict[str, str], message: str
) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, **broken))
    assert bad.returncode == 1
    assert message in bad.stderr


_AIR_15 = b"\0air64-apple-macosx15.0.0\0"


def _library(
    major: int,
    minor: int = 0,
    patch: int = 0,
    *,
    platform: int = 0x8001,
    system: int = 0x81,
    body: bytes = _AIR_15,
) -> bytes:
    """A Metal library whose header names that system. The default names macOS."""
    version = (platform, 2, 8, 0, system, major, minor, patch, 24 + len(body))
    return struct.pack("<4sHHHBBHBBQ", b"MTLB", *version) + body


def _metal_wheel(
    tmp_path: Path, platform: str, *parts: bytes, more: dict[str, bytes] | None = None
) -> Path:
    """A Metal wheel with that platform tag. Its tool holds these parts."""
    return _wheel(
        tmp_path,
        tool=b"\xcf\xfa\xed\xfe" + b"".join(parts),
        name_tag=f"1.metal-py3-none-{platform}",
        tag=f"py3-none-{platform}",
        more=more,
    )


def _metal_floor(
    tmp_path: Path, platform: str, *parts: bytes, more: dict[str, bytes] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the floor check on a Metal wheel with these parts in its tool."""
    wheel = _metal_wheel(tmp_path, platform, *parts, more=more)
    return _check(tmp_path, wheel, "check_metal_floor", BACKEND="metal", PLATFORM=platform)


@pytest.mark.parametrize(
    ("platform", "parts"),
    [
        ("macosx_15_0_arm64", [_library(15)]),
        ("macosx_15_0_arm64", [_library(14, 6, 1), _library(15)]),
        ("macosx_26_0_arm64", [_library(15, 6)]),
        ("macosx_26_1_arm64", [_library(26, 1)]),
        ("macosx_15_10_arm64", [_library(15, 9, 9)]),
        ("macosx_15_0_arm64", [b"\0MTLBuffer\0MTLResidencySetDescriptor\0", _library(15)]),
    ],
)
def test_the_check_passes_a_metal_wheel_whose_tag_covers_its_libraries(
    tmp_path: Path, platform: str, parts: list[bytes]
) -> None:
    done = _metal_floor(tmp_path, platform, *parts)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    ("platform", "parts", "needs", "names"),
    [
        ("macosx_11_0_arm64", [_library(15)], "15.0.0", "11.0.0"),
        ("macosx_15_0_arm64", [_library(26)], "26.0.0", "15.0.0"),
        ("macosx_15_0_arm64", [_library(15, 10)], "15.10.0", "15.0.0"),
        ("macosx_15_9_arm64", [_library(15, 10)], "15.10.0", "15.9.0"),
        ("macosx_15_0_arm64", [_library(26, 1), _library(15)], "26.1.0", "15.0.0"),
        ("macosx_9_0_arm64", [_library(15)], "15.0.0", "9.0.0"),
        ("macosx_15_0_arm64", [_library(15, 0, 1)], "15.0.1", "15.0.0"),
    ],
)
def test_the_whole_check_fails_a_metal_wheel_whose_libraries_need_a_newer_macos(
    tmp_path: Path, platform: str, parts: list[bytes], needs: str, names: str
) -> None:
    wheel = _metal_wheel(tmp_path, platform, *parts)
    bad = _check(tmp_path, wheel, BACKEND="metal", PLATFORM=platform)
    assert bad.returncode == 1
    assert (
        f"the Metal libraries of {wheel.name} need macOS {needs}, "
        f"but its tag names macOS {names}\n" in bad.stderr
    )


@pytest.mark.parametrize(
    ("newer", "needs"),
    [
        (_library(26, body=b""), "26.0.0"),
        (_library(26, body=b"\0air64-apple-macosx26\0"), "26.0.0"),
        (_library(26, body=b"\0air64-apple-macos26.0\0"), "26.0.0"),
        (_library(15, 0, 1, body=b"\0air64-apple-macosx15.0.1\0"), "15.0.1"),
    ],
    ids=["no-text", "text-without-minor", "text-of-macos", "patch"],
)
def test_the_check_reads_the_header_of_a_metal_library_not_its_text(
    tmp_path: Path, newer: bytes, needs: str
) -> None:
    bad = _metal_floor(tmp_path, "macosx_15_0_arm64", newer, _library(15))
    assert bad.returncode == 1
    assert f"need macOS {needs}, but its tag names macOS 15.0.0" in bad.stderr


@pytest.mark.parametrize(
    ("header", "needs"),
    [
        # The 0.9.5 wheel (sha256 08c30b53), macOS 14.6 itself, the Metal compiler of macOS 26.
        ("01800200080000810f000000", "15.0.0"),
        ("01800200070000810e000600", "14.6.0"),
        ("01800200090000811a000000", "26.0.0"),
    ],
)
def test_the_check_reads_the_headers_that_the_metal_compiler_writes(
    tmp_path: Path, header: str, needs: str
) -> None:
    library = b"MTLB" + bytes.fromhex(header) + struct.pack("<Q", 24)
    bad = _metal_floor(tmp_path, "macosx_11_0_arm64", library)
    assert bad.returncode == 1
    assert f"need macOS {needs}, but its tag names macOS 11.0.0" in bad.stderr


def test_the_check_reads_the_metal_libraries_of_the_whole_wheel(tmp_path: Path) -> None:
    more = {"lilbee_engine_mistral_rs/share/kernels.bin": b"kernels" + _library(26)}
    bad = _metal_floor(tmp_path, "macosx_15_0_arm64", _library(15), more=more)
    assert bad.returncode == 1
    assert "need macOS 26.0.0, but its tag names macOS 15.0.0" in bad.stderr


@pytest.mark.parametrize(
    ("platform", "system"),
    [
        (0x0001, 0x82),
        (0x8001, 0x86),
        (0x8001, 0x00),
        (0x8001, 0x01),
        (0x0001, 0x81),
        (0x8002, 0x81),
    ],
    ids=["ios", "catalyst", "no-system", "macos-32-bit", "other-container", "new-container"],
)
def test_the_check_fails_a_metal_library_that_names_no_macos(
    tmp_path: Path, platform: int, system: int
) -> None:
    other = _library(11, platform=platform, system=system)
    bad = _metal_floor(tmp_path, "macosx_15_0_arm64", other, _library(15))
    assert bad.returncode == 1
    assert (
        "the Metal library at byte 4 of lilbee_engine_mistral_rs/bin/mistralrs names no macOS "
        f"version: its container version is {platform:#06x} "
        f"and its operating system is {system:#04x}"
    ) in bad.stderr


def test_the_check_fails_a_metal_library_whose_header_ends_the_file(tmp_path: Path) -> None:
    whole = _library(15)
    bad = _metal_floor(tmp_path, "macosx_15_0_arm64", whole, _library(15)[:12])
    assert bad.returncode == 1
    assert f"the Metal library at byte {4 + len(whole)} of " in bad.stderr
    assert "has no complete header" in bad.stderr


@pytest.mark.parametrize(
    "parts",
    [[], [b"cannot build for air64-apple-macosx15.0.0"], [b"\0MTLBuffer\0"]],
    ids=["nothing", "a-target-in-a-message", "a-class-name"],
)
def test_the_whole_check_fails_a_metal_wheel_without_a_metal_library(
    tmp_path: Path, parts: list[bytes]
) -> None:
    wheel = _metal_wheel(tmp_path, "macosx_15_0_arm64", *parts)
    bad = _check(tmp_path, wheel, BACKEND="metal", PLATFORM="macosx_15_0_arm64")
    assert bad.returncode == 1
    assert f"{wheel.name} holds no Metal library" in bad.stderr


@pytest.mark.parametrize("platform", ["win_amd64", "macosx_15_0_x86_64"])
def test_the_whole_check_fails_a_metal_wheel_whose_platform_names_no_macos(
    tmp_path: Path, platform: str
) -> None:
    wheel = _metal_wheel(tmp_path, platform, _library(15))
    bad = _check(tmp_path, wheel, BACKEND="metal", PLATFORM=platform)
    assert bad.returncode == 1
    assert f"the platform {platform} names no macOS version for arm64" in bad.stderr


def test_the_check_reads_the_metal_floor_of_a_metal_wheel_only(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, more={f"{_LICENCES}/notes.txt": b"air64-apple-macosx99.0.0"})
    cpu = _check(tmp_path, wheel, "check_metal_floor")
    assert cpu.returncode == 0, cpu.stderr
    metal = _check(
        tmp_path, wheel, "check_metal_floor", BACKEND="metal", PLATFORM="macosx_15_0_arm64"
    )
    assert metal.returncode == 1
    assert "holds no Metal library" in metal.stderr


_POSIX_TOOL = pytest.mark.skipif(
    sys.platform == "win32", reason="the tool of the test wheel is a POSIX shell script"
)


@_POSIX_TOOL
def test_the_whole_check_starts_the_tool_and_asks_a_cpu_wheel_one_question(tmp_path: Path) -> None:
    done = _check(tmp_path, _wheel(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "mistralrs 0.9.5" in done.stdout
    asked = (tmp_path / "lilbee" / "tools" / "qa" / "asked").read_text(encoding="utf-8")
    exe, model = asked.splitlines()
    assert exe.endswith("lilbee_engine_mistral_rs/bin/mistralrs")
    assert model == "owner/model"


@_POSIX_TOOL
def test_the_check_asks_a_metal_wheel_no_question(tmp_path: Path) -> None:
    done = _check(tmp_path, _wheel(tmp_path), "start_tool", BACKEND="metal")
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "lilbee" / "tools" / "qa" / "asked").exists()


@_POSIX_TOOL
def test_the_check_fails_a_wheel_whose_tool_fails_to_start(tmp_path: Path) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, tool=b"#!/bin/sh\nexit 3\n"), "start_tool")
    assert bad.returncode == 1
    assert "the installed tool does not start" in bad.stderr


@_POSIX_TOOL
def test_the_check_fails_a_wheel_whose_tool_is_not_executable(tmp_path: Path) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, tool_mode=0o644), "start_tool")
    assert bad.returncode == 1
    assert "the installed tool is not executable" in bad.stderr


@_POSIX_TOOL
def test_the_check_fails_a_wheel_without_the_tool(tmp_path: Path) -> None:
    bad = _check(tmp_path, _wheel(tmp_path, drop="lilbee_engine_mistral_rs/bin/"), "start_tool")
    assert bad.returncode == 1
    assert "the installed wheel lacks mistralrs" in bad.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="the test renames a library of an ELF file")
def test_the_check_fails_a_wheel_whose_tool_needs_a_library_that_does_not_resolve(
    tmp_path: Path,
) -> None:
    true = shutil.which("true")
    assert true is not None
    good = Path(true).read_bytes()
    assert b"libc.so.6\0" in good
    tool = good.replace(b"libc.so.6\0", b"libq.so.6\0")
    bad = _check(tmp_path, _wheel(tmp_path, tool=tool), "start_tool", BACKEND="metal")
    assert bad.returncode == 1
    assert "libq.so.6" in bad.stderr
    assert "the installed tool does not start" in bad.stderr


@_POSIX_TOOL
@pytest.mark.parametrize(
    ("library", "status"),
    [
        ("/usr/lib/libSystem.B.dylib", 0),
        ("/System/Library/Frameworks/Metal.framework/Metal", 0),
        ("/opt/homebrew/lib/libssl.dylib", 1),
        ("@rpath/libmissing.dylib", 1),
    ],
)
def test_the_check_fails_a_macos_tool_that_loads_a_library_outside_the_system(
    tmp_path: Path, library: str, status: int
) -> None:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    otool = stubs / "otool"
    otool.write_text(
        f'#!/bin/sh\necho "$2:"\necho "\t{library} (compatibility version 1.0.0)"\n',
        encoding="utf-8",
    )
    otool.chmod(0o755)
    done = _check(tmp_path, _wheel(tmp_path), "start_tool", RUNNER_OS="macOS", BACKEND="metal")
    assert done.returncode == status, done.stderr
    assert ("loads a library outside the system" in done.stderr) == bool(status)


def test_the_check_fails_a_macos_tool_whose_otool_fails(tmp_path: Path) -> None:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    otool = stubs / "otool"
    otool.write_text('#!/bin/sh\necho "$2:"\nexit 1\n', encoding="utf-8")
    otool.chmod(0o755)
    done = _check(tmp_path, _wheel(tmp_path), "start_tool", RUNNER_OS="macOS", BACKEND="metal")
    assert done.returncode == 1
    assert "otool failed on" in done.stderr
    assert "loads a library outside the system" not in done.stderr


_RUNTIME = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _windows_check(tmp_path: Path, bundled: dict[str, bytes]) -> subprocess.CompletedProcess[str]:
    """Run the start stage on a Windows wheel of *bundled* files and a tool that starts."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    # The check runs the interpreter that a Windows venv holds.
    stub = stubs / "python"
    stub.write_text(
        "#!/bin/sh\n"
        f'"{sys.executable}" "$@" || exit\n'
        'if [ "$1 $2" = "-m venv" ]; then\n'
        '  mkdir "$3/Scripts"\n'
        '  printf \'#!/bin/sh\\nexec "%s" "$@"\\n\' "$3/bin/python" > "$3/Scripts/python.exe"\n'
        '  chmod +x "$3/Scripts/python.exe"\n'
        "fi\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    folder = "lilbee_engine_mistral_rs/bin"
    more = {f"{folder}/{name}": data for name, data in bundled.items()}
    more[f"{folder}/mistralrs.exe"] = b"#!/bin/sh\necho mistralrs 0.9.5\n"
    return _check(tmp_path, _wheel(tmp_path, more=more), "start_tool", RUNNER_OS="Windows")


@_POSIX_TOOL
@pytest.mark.parametrize("missing", [None, *_RUNTIME])
def test_the_check_fails_a_windows_wheel_whose_dll_imports_a_runtime_dll_it_lacks(
    tmp_path: Path, missing: str | None
) -> None:
    bundled = {name: b"dll" for name in _RUNTIME if name != missing}
    bundled["engine.dll"] = pe_file(("kernel32.dll", *_RUNTIME))
    done = _windows_check(tmp_path, bundled)
    assert done.returncode == (0 if missing is None else 1), done.stderr
    if missing is not None:
        assert f"engine.dll imports {missing}" in done.stderr


@_POSIX_TOOL
def test_the_check_asks_a_windows_wheel_for_no_runtime_dll_that_nothing_imports(
    tmp_path: Path,
) -> None:
    done = _windows_check(tmp_path, {"engine.dll": pe_file(("kernel32.dll",))})
    assert done.returncode == 0, done.stderr


@_POSIX_TOOL
def test_the_check_fails_a_windows_wheel_whose_delay_loaded_dll_is_not_bundled(
    tmp_path: Path,
) -> None:
    done = _windows_check(tmp_path, {"engine.dll": pe_file((), ("vcomp140.dll",))})
    assert done.returncode == 1
    assert "engine.dll imports vcomp140.dll" in done.stderr


@pytest.mark.parametrize("name", ["LICENSE-mistral.rs", "NOTICE-OpenSSL", "SOURCE-MPL-2.0"])
def test_the_pack_refuses_an_empty_licence_file(tmp_path: Path, name: str) -> None:
    root = _tree(tmp_path)
    tree = root / "packaging" / PACKAGING.name
    (root / "crate-licences.txt").write_text("MIT\n", encoding="utf-8", newline="\n")
    for held in ("LICENSE-mistral.rs", "NOTICE-OpenSSL", "SOURCE-MPL-2.0"):
        (tree / held).write_text("text", encoding="utf-8")
    ok = _pack_stage(tmp_path, "name_licences")
    assert ok.returncode == 0, ok.stderr
    (tree / name).write_text("", encoding="utf-8")
    bad = _pack_stage(tmp_path, "name_licences")
    assert bad.returncode == 1
    assert f"empty licence file packaging/{PACKAGING.name}/{name}" in bad.stderr


def _wheel_listing(tmp_path: Path) -> set[str]:
    """Build the wheel as the pack script does and list the files under dist-info/licenses."""
    tree = _tree(tmp_path) / "packaging" / PACKAGING.name
    report = _mpl_report("selectors", "0.26.0")
    report["licenses"].append(_entry("OpenSSL", "ssl text", "aws-lc-sys"))
    checkout = tmp_path / "checkout"
    (checkout / "sub").mkdir(parents=True)
    (checkout / "sub" / "NOTICE").write_text("checkout notice", encoding="utf-8")
    names = rust_notices.write_notices(report, _LOCK, checkout, tmp_path / "registry", tree)
    assert {"SOURCE-MPL-2.0", "NOTICE-OpenSSL"} <= set(names)
    (tree / "LICENSE-mistral.rs").write_text("mistral.rs licence", encoding="utf-8")
    crates = tree / "THIRD-PARTY" / "mistral.rs"
    crates.mkdir(parents=True)
    (crates / "MIT.txt").write_text("mit", encoding="utf-8")
    stamped = _pack_stage(
        tmp_path, "stamp_project", MISTRALRS_TAG="v0.9.5", LICENSE_EXPRESSION=_EXPRESSION
    )
    assert stamped.returncode == 0, stamped.stderr
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return {
            name.split(".dist-info/licenses/")[1]
            for name in archive.namelist()
            if ".dist-info/licenses/" in name
        }


def test_the_built_wheel_holds_every_licence_file_the_pack_writes(tmp_path: Path) -> None:
    held = _wheel_listing(tmp_path)
    assert {
        "SOURCE-MPL-2.0",
        "NOTICE-OpenSSL",
        "LICENSE-mistral.rs",
        "THIRD-PARTY/mistral.rs/MIT.txt",
        "THIRD-PARTY/vendored/mistral.rs__sub__NOTICE",
    } <= held


def _wrap_archive(tmp_path: Path, run: str, archive: Path, checkout_time: int) -> str:
    """Wrap *archive* into a copy of the wheel project, the way the job does, and hash the wheel."""
    tree = tmp_path / run / "project"
    shutil.copytree(
        PACKAGING, tree, ignore=shutil.ignore_patterns("bin", "__pycache__", "*.egg-info")
    )
    packed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(TOOLS / "repack_mistralrs.py"),
            str(archive),
            str(tree / "lilbee_engine_mistral_rs" / "bin"),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    for path in (tree, *tree.rglob("*")):
        os.utime(path, (checkout_time, checkout_time))
    env = dict(line.split("=", 1) for line in packed.stdout.splitlines())
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / run / "dist")],
        cwd=tree,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
    )
    (wheel,) = (tmp_path / run / "dist").glob("*.whl")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_wrapping_one_archive_twice_gives_one_wheel(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.size = 3
        info.mtime = 1_700_000_000
        tar.addfile(info, io.BytesIO(b"bin"))
    first = _wrap_archive(tmp_path, "first", archive, 1_750_000_000)
    second = _wrap_archive(tmp_path, "second", archive, 1_760_000_000)
    assert first == second


@pytest.mark.parametrize(
    ("archive_name", "member_time"),
    [("a.tar.gz", 1_700_000_000), ("a.zip", 1_700_000_000)],
)
def test_repack_reports_the_time_of_the_tool_in_the_archive(
    tmp_path: Path, archive_name: str, member_time: int
) -> None:
    archive = tmp_path / archive_name
    if archive_name.endswith(".zip"):
        stamp = time.gmtime(member_time)
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(zipfile.ZipInfo("mistralrs.exe", stamp[:6]), b"exe")
    else:
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("./mistralrs")
            info.size = 3
            info.mtime = member_time
            tar.addfile(info, io.BytesIO(b"bin"))
    assert repack.repack(archive, tmp_path / "bin") == member_time


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the mode of the tool is a POSIX mode",
)
def test_the_repack_stage_exports_the_archive_time_to_the_later_stages(tmp_path: Path) -> None:
    """The wheel build and the retag run in this shell and read the exported time."""
    upstream = _tree(tmp_path) / "upstream"
    upstream.mkdir()
    with tarfile.open(upstream / "a.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.size = 3
        info.mtime = 1_700_000_000
        tar.addfile(info, io.BytesIO(b"bin"))
    done = _pack_stage(
        tmp_path, "repack; bash -c 'echo read=${SOURCE_DATE_EPOCH}'", ASSET="a.tar.gz"
    )
    assert done.returncode == 0, done.stderr
    assert "read=1700000000" in done.stdout.splitlines()


_CUDA_EXPRESSION = f"{_EXPRESSION} AND LicenseRef-NVIDIA-CUDA-EULA"
_CUDA_NOTICES = {
    f"{_LICENCES}/NOTICE-NVIDIA-CUDA": b"eula",
    f"{_LICENCES}/LICENSE-NCCL": b"bsd",
}
_CUDA_SM86 = f"1.cuda_sm86-py3-none-{_HOST_PLATFORM}"


def _cuda_licence_check(
    tmp_path: Path, drop: str | None = None, expression: str = _CUDA_EXPRESSION
) -> subprocess.CompletedProcess[str]:
    wheel = _wheel(
        tmp_path, drop=drop, name_tag=_CUDA_SM86, expression=expression, more=_CUDA_NOTICES
    )
    return _check(
        tmp_path, wheel, "check_licences", expression=expression, BACKEND="cuda-sm86"
    )


def test_the_check_passes_a_cuda_wheel_that_holds_the_nvidia_licences(tmp_path: Path) -> None:
    done = _cuda_licence_check(tmp_path)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("name", ["NOTICE-NVIDIA-CUDA", "LICENSE-NCCL"])
def test_the_check_fails_a_cuda_wheel_that_lacks_an_nvidia_licence(
    tmp_path: Path, name: str
) -> None:
    done = _cuda_licence_check(tmp_path, drop=f"{_LICENCES}/{name}")
    assert done.returncode == 1
    assert f"wheel lacks {name}" in done.stderr


def test_the_check_fails_a_cuda_wheel_whose_expression_lacks_the_cuda_eula(
    tmp_path: Path,
) -> None:
    done = _cuda_licence_check(tmp_path, expression=_EXPRESSION)
    assert done.returncode == 1
    assert "lacks LicenseRef-NVIDIA-CUDA-EULA" in done.stderr


def test_the_check_names_a_cuda_cell_by_its_wheel_local_tag(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, name_tag=_CUDA_SM86)
    done = _check(tmp_path, wheel, "check_tag", BACKEND="cuda-sm86")
    assert done.returncode == 0, done.stderr


def _libs(tmp_path: Path, names: list[str]) -> Path:
    folder = tmp_path / "libs"
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"lib")
    return folder


def test_the_check_passes_a_cuda_wheel_with_the_five_libraries(tmp_path: Path) -> None:
    folder = _libs(tmp_path, CUDA_LIBS)
    done = _check(tmp_path, folder, "check_cuda_libs", BACKEND="cuda-sm86")
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("missing", CUDA_LIBS)
def test_the_check_fails_a_cuda_wheel_that_lacks_a_library(tmp_path: Path, missing: str) -> None:
    folder = _libs(tmp_path, [name for name in CUDA_LIBS if name != missing])
    done = _check(tmp_path, folder, "check_cuda_libs", BACKEND="cuda-sm86")
    assert done.returncode == 1
    assert f"lacks bin/lib/{missing}" in done.stderr


def test_the_whole_check_of_a_cuda_wheel_stops_at_the_libraries_not_at_the_start(
    tmp_path: Path,
) -> None:
    """A build runner has no driver, so the tool cannot start; the check reads the libraries."""
    wheel = _wheel(
        tmp_path,
        name_tag=_CUDA_SM86,
        expression=_CUDA_EXPRESSION,
        more={
            **_CUDA_NOTICES,
            **{f"lilbee_engine_mistral_rs/bin/lib/{name}": b"lib" for name in CUDA_LIBS},
        },
        tool=b"#!/bin/sh\nexit 99\n",
    )
    done = _check(tmp_path, wheel, expression=_CUDA_EXPRESSION, BACKEND="cuda-sm86")
    assert done.returncode == 0, done.stderr


def test_the_pack_asks_a_cuda_cell_for_its_cuda_inputs(tmp_path: Path) -> None:
    base = {
        "MISTRALRS_REPO": "r/r",
        "MISTRALRS_TAG": "v0.9.5",
        "ASSET": "a.tar.gz",
        "BACKEND": "cuda-sm86",
        "PLATFORM": "manylinux_2_35_x86_64",
        "RUST_TARGET": "x86_64-unknown-linux-gnu",
        "CARGO_ABOUT_VERSION": "0.9.3",
        "ABOUT_TARGET": "x86_64-unknown-linux-musl",
        "MISTRALRS_CHAT_MODEL": "m",
        "OUT": str(tmp_path / "out"),
        "RUST_FEATURES": "cuda,flash-attn,nccl",
        "CUDA_VERSION": "12.8.1",
        "NCCL_TAG": "v2.26.2-1",
    }
    ok = _pack_stage(tmp_path, "require_inputs", **base)
    assert ok.returncode == 0, ok.stderr
    for name in ("CUDA_VERSION", "NCCL_TAG", "RUST_FEATURES"):
        done = _pack_stage(tmp_path, "require_inputs", **{**base, name: ""})
        assert done.returncode == 2
        assert f"{name} is not set for the CUDA cell cuda-sm86" in done.stderr


def test_the_pack_names_the_cuda_eula_in_the_licence_expression_of_a_cuda_cell(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    tree = root / "packaging" / PACKAGING.name
    (root / "crate-licences.txt").write_text("MIT\n", encoding="utf-8", newline="\n")
    (tree / "LICENSE-mistral.rs").write_text("text", encoding="utf-8")
    cuda = _pack_stage(tmp_path, "name_licences", BACKEND="cuda-sm86")
    cpu = _pack_stage(tmp_path, "name_licences", BACKEND="cpu")
    assert "LicenseRef-NVIDIA-CUDA-EULA" in cuda.stdout
    assert "LicenseRef-NVIDIA-CUDA-EULA" not in cpu.stdout


def test_the_pack_has_a_cuda_stage() -> None:
    job = yaml.safe_load(pack_workflow().read_text(encoding="utf-8"))["jobs"]["mistralrs"]
    assert job["steps"]
    pack = (TOOLS / "pack_mistralrs.sh").read_text(encoding="utf-8")
    check = (TOOLS / "check_mistralrs_wheel.sh").read_text(encoding="utf-8")
    for word in ("add_cuda_notices", "NCCL_TAG", "LicenseRef-NVIDIA-CUDA-EULA"):
        assert word in pack
    assert "check_cuda_libs" in check


def test_the_pack_matrix_holds_the_cpu_cells_and_the_six_cuda_cells() -> None:
    cells = pack_cells("mistralrs")
    assert {cell["cell"] for cell in cells} == {
        "mistralrs-linux-x86_64-cpu",
        "mistralrs-windows-x86_64-cpu",
        "mistralrs-macos-arm64-metal",
        *(f"mistralrs-linux-x86_64-cuda-sm{cc}" for cc in (80, 86, 89, 90, 100, 120)),
    }
    for cell in cells:
        if "cuda" in cell["cell"]:
            # The features of the cuda128 archives in the linux-cuda matrix of the v0.9.5
            # release.yml: features: "cuda flash-attn nccl".
            assert cell["rust_features"] == "cuda,flash-attn,nccl"
            assert cell["asset"].startswith("mistralrs-cuda128-sm")


def test_the_repack_takes_exactly_the_five_cuda_libraries() -> None:
    assert repack.CUDA_LIBS == frozenset(CUDA_LIBS)


# The Attachment A list of the CUDA EULA, cut to the lines the four EULA libraries need.
_NOTICE_EULA = (
    "Attachment A\nThe following CUDA Toolkit files may be distributed with applications "
    "developed by you.\nLinux\nlibcudart.so, libcublas.so, libcublasLt.so, libcurand.so\n"
    "Attachment B\n"
)
_EULA_LIBS = "libcublas.so, libcublasLt.so, libcudart.so, libcurand.so."
_SHA256SUM = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sha256sum") is None,
    reason="the notice step runs sha256sum on the Linux runner of a CUDA cell",
)


def _offline_notice_writer(root: Path) -> None:
    """Make the notice writer in the tree read a fixed EULA instead of the NVIDIA page."""
    writer = TOOLS / "write_cuda_notice.py"
    (root / "tools" / "wheel-build" / "write_cuda_notice.py").write_text(
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('writer', {str(writer)!r})\n"
        "writer = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(writer)\n"
        f"writer.eula_text = lambda version: ('https://eula.test', {_NOTICE_EULA!r})\n"
        "writer.main(sys.argv)\n",
        encoding="utf-8",
    )


def _notice_stage(tmp_path: Path, backend: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the notice step on a tree whose bin/lib holds the five libraries; curl is a stub."""
    root = _tree(tmp_path)
    _offline_notice_writer(root)
    lib = root / "packaging" / PACKAGING.name / "lilbee_engine_mistral_rs" / "bin" / "lib"
    lib.mkdir(parents=True)
    for name in CUDA_LIBS:
        (lib / name).write_bytes(name.encode())
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    curl = stubs / "curl"
    curl.write_text(
        f'#!/bin/sh\necho "$@" > {str(tmp_path / "curl-args")!r}\necho nccl-bsd-text\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    done = _pack_stage(
        tmp_path, "add_cuda_notices", BACKEND=backend, CUDA_VERSION="12.8.1", NCCL_TAG="v2.25.1-1"
    )
    return done, root / "packaging" / PACKAGING.name


@_SHA256SUM
def test_the_notice_step_names_the_four_eula_libraries_and_fetches_the_nccl_licence(
    tmp_path: Path,
) -> None:
    done, tree = _notice_stage(tmp_path, "cuda-sm86")
    assert done.returncode == 0, done.stderr
    notice = (tree / "NOTICE-NVIDIA-CUDA").read_text(encoding="utf-8")
    assert f"carries NVIDIA CUDA 12.8.1 shared libraries without change: {_EULA_LIBS}" in notice
    assert (tree / "LICENSE-NCCL").read_text(encoding="utf-8") == "nccl-bsd-text\n"
    assert (tmp_path / "curl-args").read_text(encoding="utf-8").split() == [
        "-fsSL",
        "--retry",
        "3",
        "https://raw.githubusercontent.com/NVIDIA/nccl/v2.25.1-1/LICENSE.txt",
    ]


@_SHA256SUM
def test_the_notice_step_of_a_cpu_cell_writes_no_nvidia_notice(tmp_path: Path) -> None:
    done, tree = _notice_stage(tmp_path, "cpu")
    assert done.returncode == 0, done.stderr
    assert not (tree / "NOTICE-NVIDIA-CUDA").exists()
    assert not (tree / "LICENSE-NCCL").exists()
    assert not (tmp_path / "curl-args").exists()


@_SHA256SUM
@pytest.mark.parametrize("mode", [["-b"], []], ids=["binary", "text"])
def test_the_notice_writer_reads_a_sha256sum_list_of_either_mode(
    tmp_path: Path, mode: list[str]
) -> None:
    """sha256sum writes '<digest> *<name>' with -b and '<digest>  <name>' without it."""
    four = [name for name in CUDA_LIBS if name != "libnccl.so.2"]
    lib = _libs(tmp_path, four)
    sums = tmp_path / "cuda-libs.sha256"
    sums.write_text(
        subprocess.run(
            ["sha256sum", *mode, *four], cwd=lib, capture_output=True, text=True, check=True
        ).stdout,
        encoding="utf-8",
    )
    root = _tree(tmp_path)
    _offline_notice_writer(root)
    writer = root / "tools" / "wheel-build" / "write_cuda_notice.py"
    notice = tmp_path / "NOTICE"
    done = subprocess.run(
        [sys.executable, str(writer), str(notice), "12.8.1", str(sums)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert _EULA_LIBS in notice.read_text(encoding="utf-8")
