"""The wheel tools of the TEI pack: the project stamp and the wheel check."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

from tests._pack_workflow import pack_workflow
from tests._workflow_steps import commands, steps

_TOOLS = Path(__file__).resolve().parents[1] / "tools/wheel-build"
_PROJECT = Path(__file__).resolve().parents[1] / "packaging/engine-wheel-tei/pyproject.toml"
_DIST = "lilbee_engine_tei-1.9.4.dist-info"
_CPU_NAME = "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
_MAC_NAME = "lilbee_engine_tei-1.9.4-1.metal-py3-none-macosx_14_0_arm64.whl"
_EXECUTABLE = 0o100755 << 16
_PLAIN = 0o100644 << 16
_CUTLASS = "THIRD-PARTY/tei-notices/git-candle-1/candle-flash-attn/cutlass/LICENSE.txt"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stamp_tool = _load("stamp_engine_wheel")
check_tool = _load("check_tei_wheel")


def _wheel(
    tmp_path: Path,
    name: str = _CPU_NAME,
    expression: str = "Apache-2.0 AND MIT",
    skip: tuple[str, ...] = (),
    extra: tuple[str, ...] = (),
    router_mode: int = _EXECUTABLE,
    router: bytes = b"\x7fELF",
) -> Path:
    entries = {
        "lilbee_engine_tei/bin/text-embeddings-router": (router, router_mode),
        f"{_DIST}/METADATA": (f"License-Expression: {expression}\n".encode(), _PLAIN),
        f"{_DIST}/licenses/LICENSE-tei": (b"Apache licence", _PLAIN),
        f"{_DIST}/licenses/LICENSE-openssl": (b"OpenSSL licence", _PLAIN),
        f"{_DIST}/licenses/THIRD-PARTY/tei/LICENSES.txt": (b"crate licences", _PLAIN),
    }
    for name_in_zip in extra:
        entries[f"{_DIST}/licenses/{name_in_zip}"] = (b"NVIDIA EULA text", _PLAIN)
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for entry, (data, mode) in entries.items():
            if entry.rsplit("/", 1)[-1] in skip:
                continue
            info = zipfile.ZipInfo(entry)
            info.external_attr = mode
            archive.writestr(info, data)
    return path


def _problems(wheel: Path, backend: str, expression: str) -> list[str]:
    return check_tool.problems(wheel, backend, expression)


def test_stamp_sets_version_expression_and_files_before_requires_python() -> None:
    text = stamp_tool.stamp(_PROJECT.read_text(encoding="utf-8"), "1.9.4", "Apache-2.0 AND MIT")
    lines = text.splitlines()
    assert 'version = "1.9.4"' in lines
    assert 'license = "Apache-2.0 AND MIT"' in lines
    assert 'license-files = ["LICENSE-*", "NOTICE-*", "SOURCE-*", "THIRD-PARTY/**"]' in lines
    assert lines.index('license = "Apache-2.0 AND MIT"') < lines.index('requires-python = ">=3.11"')


def test_stamp_refuses_a_project_without_a_version() -> None:
    with pytest.raises(SystemExit, match="no version line"):
        stamp_tool.stamp('[project]\nrequires-python = ">=3.11"\n', "1.9.4", "MIT")


def test_stamp_refuses_a_project_without_requires_python() -> None:
    with pytest.raises(SystemExit, match="no requires-python line"):
        stamp_tool.stamp('[project]\nversion = "0.0.0"\n', "1.9.4", "MIT")


def test_stamp_main_rewrites_the_file(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(_PROJECT.read_text(encoding="utf-8"), encoding="utf-8")
    stamp_tool.main(["stamp", str(project), "1.9.4", "MIT"])
    assert 'license = "MIT"' in project.read_text(encoding="utf-8")


def test_stamp_main_needs_three_arguments() -> None:
    with pytest.raises(SystemExit, match="Usage"):
        stamp_tool.main(["stamp"])


def test_a_complete_cpu_wheel_passes(tmp_path: Path) -> None:
    assert _problems(_wheel(tmp_path), "cpu", "Apache-2.0 AND MIT") == []


def test_a_complete_cuda_wheel_passes(tmp_path: Path) -> None:
    expression = "Apache-2.0 AND MIT AND LicenseRef-NVIDIA-CUDA-EULA"
    wheel = _wheel(tmp_path, expression=expression, extra=("NOTICE-NVIDIA-CUDA", _CUTLASS))
    assert _problems(wheel, "cuda-sm89", expression) == []


def test_a_cuda_wheel_without_the_cutlass_licence_is_named(tmp_path: Path) -> None:
    expression = "Apache-2.0 AND LicenseRef-NVIDIA-CUDA-EULA"
    wheel = _wheel(tmp_path, expression=expression, extra=("NOTICE-NVIDIA-CUDA",))
    assert _problems(wheel, "cuda-sm89", expression) == [
        "no cutlass licence file under licenses/THIRD-PARTY/tei-notices/"
    ]


def test_a_cuda_wheel_with_an_empty_cutlass_licence_is_named(tmp_path: Path) -> None:
    expression = "Apache-2.0 AND LicenseRef-NVIDIA-CUDA-EULA"
    wheel = _wheel(tmp_path, expression=expression, extra=("NOTICE-NVIDIA-CUDA", _CUTLASS))
    rewritten = tmp_path / "empty" / wheel.name
    rewritten.parent.mkdir()
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(rewritten, "w") as target:
        for info in source.infolist():
            data = b"" if info.filename.endswith("cutlass/LICENSE.txt") else source.read(info)
            target.writestr(info, data)
    assert _problems(rewritten, "cuda-sm89", expression) == [
        "no cutlass licence file under licenses/THIRD-PARTY/tei-notices/"
    ]


def test_a_router_without_the_execute_bit_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, router_mode=_PLAIN)
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "lilbee_engine_tei/bin/text-embeddings-router is not an executable file"
    ]


def test_a_missing_router_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, skip=("text-embeddings-router",))
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "no lilbee_engine_tei/bin/text-embeddings-router"
    ]


def test_missing_licence_files_are_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, skip=("LICENSE-tei", "LICENSE-openssl", "LICENSES.txt"))
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "no non-empty licenses/LICENSE-tei",
        "no non-empty licenses/LICENSE-openssl",
        "no non-empty file under licenses/THIRD-PARTY/tei/",
    ]


def test_a_macos_wheel_needs_no_openssl_licence(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, name=_MAC_NAME, skip=("LICENSE-openssl",))
    assert _problems(wheel, "metal", "Apache-2.0 AND MIT") == []


def test_a_cuda_wheel_without_the_notice_is_named(tmp_path: Path) -> None:
    expression = "Apache-2.0 AND LicenseRef-NVIDIA-CUDA-EULA"
    wheel = _wheel(tmp_path, expression=expression, extra=(_CUTLASS,))
    assert _problems(wheel, "cuda-sm80", expression) == ["no non-empty licenses/NOTICE-NVIDIA-CUDA"]


def test_a_cpu_wheel_with_the_notice_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, extra=("NOTICE-NVIDIA-CUDA",))
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "licenses/NOTICE-NVIDIA-CUDA in a build with no NVIDIA files"
    ]


def test_an_expression_that_differs_from_the_build_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, expression="Apache-2.0")
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "License-Expression is 'Apache-2.0', the build wrote 'Apache-2.0 AND MIT'"
    ]


def test_an_expression_without_apache_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, expression="MIT")
    assert _problems(wheel, "cpu", "MIT") == ["License-Expression lacks Apache-2.0"]


def test_a_cuda_backend_without_the_nvidia_term_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, extra=("NOTICE-NVIDIA-CUDA", _CUTLASS))
    assert _problems(wheel, "cuda-sm90", "Apache-2.0 AND MIT") == [
        "License-Expression and backend cuda-sm90 disagree on LicenseRef-NVIDIA-CUDA-EULA"
    ]


def test_a_wheel_without_metadata_is_named(tmp_path: Path) -> None:
    path = tmp_path / _CPU_NAME
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("lilbee_engine_tei/__init__.py", "")
    assert _problems(path, "cpu", "MIT") == [f"{_CPU_NAME}: expected one METADATA file, found 0"]


def test_main_prints_ok_for_a_good_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expression = tmp_path / "expression.txt"
    expression.write_text("Apache-2.0 AND MIT\n", encoding="utf-8")
    code = check_tool.main(["check", str(_wheel(tmp_path)), "cpu", str(expression)])
    assert (code, capsys.readouterr().out) == (0, "OK\n")


def test_main_prints_each_problem_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    expression = tmp_path / "expression.txt"
    expression.write_text("Apache-2.0 AND MIT\n", encoding="utf-8")
    wheel = _wheel(tmp_path, skip=("LICENSE-tei",))
    code = check_tool.main(["check", str(wheel), "cpu", str(expression)])
    assert (code, capsys.readouterr().out) == (1, "no non-empty licenses/LICENSE-tei\n")


def test_main_needs_three_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_tool.main(["check"]) == 2
    assert "Usage: check_tei_wheel.py" in capsys.readouterr().out


notice_tool = _load("write_cuda_notice")

_BUILD_LOG = """\
[cudarc 0.17.8] cargo:rustc-link-lib=static=cudart_static
[cudarc 0.17.8] cargo:rustc-link-lib=static=cublas_static
[cudarc 0.17.8] cargo:rustc-link-lib=static=cublasLt_static
[cudarc 0.17.8] cargo:rustc-link-lib=static=nvrtc_static
[cudarc 0.17.8] cargo:rustc-link-lib=static=culibos
[cudarc 0.17.8] cargo:rustc-link-lib=dylib=cuda
[openssl-sys 0.9.117] cargo:rustc-link-lib=dylib=ssl
[x 0.1.0] cargo::rustc-link-lib=stdc++
"""
_EULA_SECTION = (
    "Attachment A\nThe following CUDA Toolkit files may be distributed with applications "
    "developed by you.\nLinux\nlibcudart.so, libcudart_static.a, libcudadevrt.a\n"
    "Linux\nlibcublas.so, libcublasLt.so, libcublas_static.a, libcublasLt_static.a\n"
    "Linux\nlibnvrtc.so, libnvrtc_static.a, libculibos.a\nAttachment B\nlibcudnn.so.9 is not here\n"
)
_ALL_FOUR = (
    "libcublasLt_static.a, libcublas_static.a, libcudart_static.a, libculibos.a, libnvrtc_static.a"
)


def _fake_eula(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        notice_tool, "eula_text", lambda version: ("https://eula.test", _EULA_SECTION)
    )


def test_linked_files_are_the_nvidia_libraries_of_the_log() -> None:
    assert ", ".join(notice_tool.linked_nvidia_files(_BUILD_LOG)) == _ALL_FOUR


def test_linked_files_name_a_shared_library_by_its_so_file() -> None:
    log = "cargo:rustc-link-lib=dylib=cudnn\ncargo:rustc-link-lib=cufft"
    assert notice_tool.linked_nvidia_files(log) == ["libcudnn.so", "libcufft.so"]


def test_attachment_a_files_stop_at_attachment_b() -> None:
    allowed = notice_tool.attachment_a_files(_EULA_SECTION)
    assert {"libcudart_static.a", "libnvrtc.so", "libcublasLt_static.a"} <= allowed
    assert "libcudnn.so.9" not in allowed


def test_attachment_a_files_need_the_section() -> None:
    with pytest.raises(SystemExit, match="no Attachment A file list"):
        notice_tool.attachment_a_files("an unrelated page")


def test_a_library_missing_from_attachment_a_is_refused() -> None:
    allowed = notice_tool.attachment_a_files(_EULA_SECTION)
    files = notice_tool.linked_nvidia_files(_BUILD_LOG + "cargo:rustc-link-lib=dylib=cudnn\n")
    assert notice_tool.disallowed(files, allowed) == ["libcudnn.so"]


def test_main_writes_the_notice_for_the_linked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "build.log"
    log.write_text(_BUILD_LOG, encoding="utf-8")
    _fake_eula(monkeypatch)
    out = tmp_path / "NOTICE-NVIDIA-CUDA"
    notice_tool.main(["notice", str(out), "12.9.1", str(log)])
    text = out.read_text(encoding="utf-8")
    assert _ALL_FOUR in text
    assert "keeps the GPU code of this build's architecture only" in text
    assert "NVIDIA CUDA Toolkit EULA, version 12.9.1, from https://eula.test" in text


def test_main_refuses_a_library_that_attachment_a_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "build.log"
    log.write_text(_BUILD_LOG + "cargo:rustc-link-lib=dylib=cudnn\n", encoding="utf-8")
    _fake_eula(monkeypatch)
    with pytest.raises(SystemExit, match=r"does not list: libcudnn\.so"):
        notice_tool.main(["notice", str(tmp_path / "n"), "12.9.1", str(log)])
    assert not (tmp_path / "n").exists()


def test_main_refuses_a_log_with_no_nvidia_library(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text("cargo:rustc-link-lib=dylib=ssl\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="shows no NVIDIA library linked"):
        notice_tool.main(["notice", str(tmp_path / "n"), "12.9.1", str(log)])


_SHA_LIST = (
    f"{'a' * 64} *libcudart.so.12\n{'b' * 64} *libcublas.so.12\n{'c' * 64} *libcublasLt.so.12\n"
)


def test_shipped_files_are_the_nvidia_libraries_of_the_sha_list_by_their_so_names() -> None:
    other = f"{'d' * 64} *libgomp.so.1\n{'e' * 64}  libllama.so\nnot a sha line\n"
    assert notice_tool.shipped_nvidia_files(_SHA_LIST + other) == [
        "libcublas.so",
        "libcublasLt.so",
        "libcudart.so",
    ]
    assert notice_tool.shipped_nvidia_files(_BUILD_LOG) == []


def test_main_writes_the_notice_for_the_libraries_a_wheel_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sums = tmp_path / "cuda-libs.sha256"
    sums.write_text(_SHA_LIST, encoding="utf-8")
    _fake_eula(monkeypatch)
    out = tmp_path / "NOTICE-NVIDIA-CUDA"
    notice_tool.main(["notice", str(out), "12.4.1", str(sums)])
    text = out.read_text(encoding="utf-8")
    assert "carries NVIDIA CUDA 12.4.1 shared libraries without change" in text
    assert "libcublas.so, libcublasLt.so, libcudart.so." in text
    assert "into the router" not in text
    assert "NVIDIA CUDA Toolkit EULA, version 12.4.1, from https://eula.test" in text
    assert "Attachment A" in text


def test_main_refuses_a_carried_library_that_attachment_a_lacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sums = tmp_path / "cuda-libs.sha256"
    sums.write_text(_SHA_LIST + f"{'f' * 64} *libcudnn.so.9\n", encoding="utf-8")
    _fake_eula(monkeypatch)
    with pytest.raises(SystemExit, match=r"does not list: libcudnn\.so"):
        notice_tool.main(["notice", str(tmp_path / "n"), "12.4.1", str(sums)])
    assert not (tmp_path / "n").exists()


def test_main_needs_three_arguments_for_the_notice() -> None:
    with pytest.raises(SystemExit, match=r"Usage: write_cuda_notice\.py"):
        notice_tool.main(["notice"])


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _scan(
    tmp_path: Path,
    cargo_files: dict[str, str],
    source_files: dict[str, str],
    git_files: dict[str, str] | None = None,
) -> Path:
    registry = tmp_path / "cargo/registry/src/index-1"
    _tree(registry, cargo_files)
    _tree(tmp_path / "cargo/git/checkouts/candle-0a1b2c/6381023", git_files or {})
    source = tmp_path / "source"
    _tree(source, source_files)
    out = tmp_path / "out"
    subprocess.run(
        ["bash", str(_TOOLS / "vendored_notices.sh"), str(source), str(out)],
        check=True,
        capture_output=True,
        env={**os.environ, "CARGO_HOME": str(tmp_path / "cargo")},
    )
    return out


def _copied(out: Path) -> list[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def test_vendored_scan_copies_notices_and_third_party_licences(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {
            "kernels-0.1.0/NOTICE": "Apache notice",
            "kernels-0.1.0/third_party/cutlass/LICENSE.txt": "BSD text",
            "kernels-0.1.0/third_party/cutlass/main.cu": "code",
            "kernels-0.1.0/src/lib.rs": "code",
            "plain-1.0.0/LICENSE-MIT": "declared licence only",
        },
        {"NOTICE.md": "tei notice", "target/debug/NOTICE": "build output"},
    )
    assert _copied(out) == [
        "kernels-0.1.0/NOTICE",
        "kernels-0.1.0/third_party/cutlass/LICENSE.txt",
        "text-embeddings-inference/NOTICE.md",
    ]


def test_vendored_scan_copies_the_licence_of_a_git_submodule(tmp_path: Path) -> None:
    out = _scan(
        tmp_path,
        {"plain-1.0.0/LICENSE-MIT": "declared licence only"},
        {},
        {
            "candle-flash-attn/cutlass/LICENSE.txt": "BSD-3-Clause NVIDIA",
            "candle-flash-attn/cutlass/include/cutlass.h": "code",
            "candle-flash-attn/src/lib.rs": "code",
            "NOTICE": "candle notice",
            ".git/LICENSE": "repository metadata",
        },
    )
    prefix = "git-candle-0a1b2c-6381023"
    assert _copied(out) == [
        f"{prefix}/NOTICE",
        f"{prefix}/candle-flash-attn/cutlass/LICENSE.txt",
    ]
    cutlass = out / prefix / "candle-flash-attn/cutlass/LICENSE.txt"
    assert cutlass.read_text(encoding="utf-8") == "BSD-3-Clause NVIDIA"


def test_vendored_scan_runs_without_a_git_checkout(tmp_path: Path) -> None:
    registry = tmp_path / "cargo/registry/src/index-1"
    _tree(registry, {"kernels-0.1.0/NOTICE": "Apache notice"})
    (tmp_path / "source").mkdir()
    out = tmp_path / "out"
    result = subprocess.run(
        ["bash", str(_TOOLS / "vendored_notices.sh"), str(tmp_path / "source"), str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "CARGO_HOME": str(tmp_path / "cargo")},
    )
    assert result.returncode == 0, result.stderr
    assert _copied(out) == ["kernels-0.1.0/NOTICE"]


_CARGO_STUB = """#!/bin/sh
# Stands in for cargo-about: writes the file named by -o, relative to the working directory.
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    --format) json=1 ;;
  esac
  shift
done
if [ -n "${json:-}" ]; then
  echo '{"licenses":[{"id":"MIT"},{"id":"Apache-2.0"}]}' > "$out"
else
  echo "licence text" > "$out"
fi
"""


def test_rust_licences_writes_to_relative_output_paths(tmp_path: Path) -> None:
    stub = tmp_path / "bin/cargo"
    stub.parent.mkdir()
    stub.write_text(_CARGO_STUB, encoding="utf-8")
    stub.chmod(0o755)
    (tmp_path / "source").mkdir()
    result = subprocess.run(
        [
            *("bash", str(_TOOLS / "rust_licences.sh"), "source", "candle", "triple"),
            *("out/third", "out/terms.txt"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{stub.parent}{os.pathsep}{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/third/LICENSES.txt").read_text(encoding="utf-8") == "licence text\n"
    assert (tmp_path / "out/terms.txt").read_text(encoding="utf-8").split() == ["Apache-2.0", "MIT"]
    assert (tmp_path / "out/terms.json").is_file()


def test_vendored_scan_stops_without_a_cargo_registry(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(_TOOLS / "vendored_notices.sh"), str(tmp_path), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "CARGO_HOME": str(tmp_path / "none")},
    )
    assert result.returncode == 1
    assert "no cargo registry" in result.stderr


mpl_tool = _load("mpl_notice")

_ABOUT = {
    "licenses": [
        {
            "id": "MIT",
            "text": "MIT text",
            "used_by": [{"crate": {"name": "serde", "version": "1.0.0"}}],
        },
        {
            "id": "MPL-2.0",
            "text": "Mozilla text\n",
            "used_by": [
                {"crate": {"name": "option-ext", "version": "0.2.0"}},
                {"crate": {"name": "webpki-roots", "version": "1.0.2"}},
            ],
        },
    ]
}
_LOCK = {
    "package": [
        {"name": "option-ext", "version": "0.2.0", "checksum": "a" * 64},
        {"name": "webpki-roots", "version": "1.0.2", "checksum": "b" * 64},
        {"name": "serde", "version": "1.0.0", "checksum": "c" * 64},
        {"name": "text-embeddings-router", "version": "1.9.4"},
    ]
}


def test_mpl_notice_names_each_crate_file_and_checksum() -> None:
    text = mpl_tool.notice(_ABOUT, _LOCK)
    assert text is not None
    assert (
        f"https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate  {'a' * 64}\n" in text
    )
    assert (
        f"https://static.crates.io/crates/webpki-roots/webpki-roots-1.0.2.crate  {'b' * 64}\n"
        in text
    )
    assert "serde" not in text
    assert [line.split()[0].rsplit("/", 1)[-1] for line in text.splitlines()] == [
        "option-ext-0.2.0.crate",
        "webpki-roots-1.0.2.crate",
    ]


def _pack_check_accepts(name: str, text: str) -> bool:
    """True when the install check of the engine packs verifies the notice file *name*."""
    from lilbee.providers.fleet import engine_store

    return name in engine_store._verified_notices([(name, text)])


def _real_mpl_notice() -> str:
    """The SOURCE-MPL-2.0 text for the locked crate that the install check lists."""
    from lilbee.providers.fleet.engine_crates import MPL_CRATE_CHECKSUMS

    about = {
        "licenses": [
            {
                "id": "MPL-2.0",
                "text": "Mozilla text\n",
                "used_by": [{"crate": {"name": "option-ext", "version": "0.2.0"}}],
            }
        ]
    }
    checksum = MPL_CRATE_CHECKSUMS[("option-ext", "0.2.0")]
    lock = {"package": [{"name": "option-ext", "version": "0.2.0", "checksum": checksum}]}
    text = mpl_tool.notice(about, lock)
    assert text is not None
    return text


def test_the_install_check_accepts_the_mpl_notice_that_the_build_writes() -> None:
    assert _pack_check_accepts("SOURCE-MPL-2.0", _real_mpl_notice())


def test_the_install_check_refuses_the_mpl_text_beside_the_crate_lines() -> None:
    text = _real_mpl_notice() + "\nMozilla Public License Version 2.0\n"
    assert not _pack_check_accepts("SOURCE-MPL-2.0", text)


def test_the_install_check_refuses_a_crate_line_with_a_wrong_checksum() -> None:
    url = _real_mpl_notice().split()[0]
    assert not _pack_check_accepts("SOURCE-MPL-2.0", f"{url}  {'0' * 64}\n")


def test_mpl_notice_is_absent_without_an_mpl_crate() -> None:
    about = {"licenses": [_ABOUT["licenses"][0]]}
    assert mpl_tool.notice(about, _LOCK) is None


def test_mpl_notice_stops_on_a_crate_without_a_checksum() -> None:
    lock = {"package": [{"name": "option-ext", "version": "0.2.0", "checksum": "aa11"}]}
    with pytest.raises(SystemExit, match=r"no checksum for webpki-roots 1\.0\.2"):
        mpl_tool.notice(_ABOUT, lock)


def test_mpl_main_writes_the_file(tmp_path: Path) -> None:
    about = tmp_path / "about.json"
    about.write_text(json.dumps(_ABOUT), encoding="utf-8")
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        f'[[package]]\nname = "option-ext"\nversion = "0.2.0"\nchecksum = "{"a" * 64}"\n\n'
        f'[[package]]\nname = "webpki-roots"\nversion = "1.0.2"\nchecksum = "{"b" * 64}"\n',
        encoding="utf-8",
    )
    out = tmp_path / "SOURCE-MPL-2.0"
    mpl_tool.main(["mpl", str(about), str(lock), str(out)])
    assert f"webpki-roots-1.0.2.crate  {'b' * 64}" in out.read_text(encoding="utf-8")


def test_mpl_main_writes_nothing_without_an_mpl_crate(tmp_path: Path) -> None:
    about = tmp_path / "about.json"
    about.write_text(json.dumps({"licenses": [_ABOUT["licenses"][0]]}), encoding="utf-8")
    lock = tmp_path / "Cargo.lock"
    lock.write_text("", encoding="utf-8")
    mpl_tool.main(["mpl", str(about), str(lock), str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_mpl_main_needs_three_arguments() -> None:
    with pytest.raises(SystemExit, match=r"Usage: mpl_notice\.py"):
        mpl_tool.main(["mpl"])


def test_a_router_with_onnx_runtime_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, router=b"\x7fELF onnxruntime session")
    assert _problems(wheel, "cpu", "Apache-2.0 AND MIT") == [
        "the router holds ONNX Runtime, whose notices this wheel does not carry"
    ]


def test_an_mpl_expression_without_its_notice_is_named(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, expression="Apache-2.0 AND MPL-2.0")
    assert _problems(wheel, "cpu", "Apache-2.0 AND MPL-2.0") == [
        "no non-empty licenses/SOURCE-MPL-2.0 for the MPL-2.0 crates"
    ]


def test_an_mpl_expression_with_its_notice_passes(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, expression="Apache-2.0 AND MPL-2.0", extra=("SOURCE-MPL-2.0",))
    assert _problems(wheel, "cpu", "Apache-2.0 AND MPL-2.0") == []


def test_the_metal_build_enables_the_candle_backend() -> None:
    lines = [
        line
        for line in pack_workflow().read_text(encoding="utf-8").splitlines()
        if "text-embeddings-router" in line and "--features" in line
    ]
    assert len(lines) == 1
    assert "--features candle,metal,http" in lines[0]


_PYTHON_STUB = """#!/bin/sh
# Runs the .py tools; records where the wheel build starts and its --outdir.
case "$1" in
  *.py) exec {python} "$@" ;;
esac
if [ "$1 $2" = "-m build" ]; then
  pwd > "{record}/build-cwd"
  echo "$SOURCE_DATE_EPOCH" > "{record}/build-epoch"
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--outdir" ]; then
      echo "$2" > "{record}/build-outdir"
      mkdir -p "$2"
    fi
    shift
  done
fi
if [ "$1 $2" = "-m wheel" ]; then
  echo "$SOURCE_DATE_EPOCH $*" > "{record}/retag"
fi
"""


_MPL_CARGO_STUB = _CARGO_STUB.replace(
    '{"licenses":[{"id":"MIT"},{"id":"Apache-2.0"}]}',
    '{"licenses":[{"id":"MIT"},{"id":"MPL-2.0","used_by":'
    '[{"crate":{"name":"option-ext","version":"0.2.0"}}]}]}',
)
_CHECKSUM = "04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d"


_COMMIT_TIME = 1_700_000_000


def _commit(source: Path) -> None:
    """Commit the files of *source* at a fixed time, as the checkout of a tag is."""
    identity = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_AUTHOR_DATE": f"{_COMMIT_TIME} +0000",
        "GIT_COMMITTER_DATE": f"{_COMMIT_TIME} +0000",
    }
    for args in (("init", "-q"), ("add", "."), ("commit", "-q", "-m", "source")):
        subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            env={**os.environ, **identity},
        )


def _run_wheel_script(tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """Run tei_wheel.sh against stub tools; return the result, the wheel project and the record."""
    tools = tmp_path / "tools/wheel-build"
    shutil.copytree(_TOOLS, tools)
    project = tmp_path / "packaging/engine-wheel-tei"
    project.mkdir(parents=True)
    shutil.copy(_PROJECT, project / "pyproject.toml")
    source = tmp_path / "tei"
    source.mkdir()
    (source / "LICENSE").write_text("Apache licence", encoding="utf-8")
    (source / "Cargo.lock").write_text(
        f'[[package]]\nname = "option-ext"\nversion = "0.2.0"\nchecksum = "{_CHECKSUM}"\n',
        encoding="utf-8",
    )
    _commit(source)
    router = tmp_path / "router"
    router.write_bytes(b"\x7fELF")
    crate = tmp_path / "cargo/registry/src/index-1/kernels-0.1.0"
    crate.mkdir(parents=True)
    (crate / "NOTICE").write_text("kernels notice", encoding="utf-8")
    record = tmp_path / "record"
    record.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text(_MPL_CARGO_STUB, encoding="utf-8")
    python = bin_dir / "python-stub"
    python.write_text(_PYTHON_STUB.format(python=sys.executable, record=record), encoding="utf-8")
    cargo.chmod(0o755)
    python.chmod(0o755)
    result = subprocess.run(
        [
            *("bash", str(tools / "tei_wheel.sh"), str(source), str(router), "candle", "triple"),
            *("cpu", "linux_x86_64", "out"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CARGO_HOME": str(tmp_path / "cargo"),
            "PYTHON": str(python),
            "WHEEL_VERSION": "1.9.4",
        },
    )
    return result, project, record


def test_the_wheel_script_builds_into_an_absolute_output_directory(tmp_path: Path) -> None:
    result, _, record = _run_wheel_script(tmp_path)
    assert result.returncode == 0, result.stderr
    outdir = Path((record / "build-outdir").read_text(encoding="utf-8").strip())
    assert outdir.is_absolute()
    assert outdir.resolve() == (tmp_path / "out/dist").resolve()


def test_the_wheel_build_runs_with_the_commit_time_of_the_source(tmp_path: Path) -> None:
    result, _, record = _run_wheel_script(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (record / "build-epoch").read_text(encoding="utf-8").strip() == str(_COMMIT_TIME)


def test_the_wheel_retag_runs_with_the_commit_time_of_the_source(tmp_path: Path) -> None:
    result, _, record = _run_wheel_script(tmp_path)
    assert result.returncode == 0, result.stderr
    retag = (record / "retag").read_text(encoding="utf-8").strip()
    wheels = tmp_path / "out/dist/*.whl"
    assert retag == f"{_COMMIT_TIME} -m wheel tags --remove --platform-tag linux_x86_64 {wheels}"


def test_the_wheel_script_writes_the_mpl_source_file_into_the_project(tmp_path: Path) -> None:
    result, project, _ = _run_wheel_script(tmp_path)
    assert result.returncode == 0, result.stderr
    lines = (project / "SOURCE-MPL-2.0").read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"https://static.crates.io/crates/option-ext/option-ext-0.2.0.crate  {_CHECKSUM}"
    ]


def test_the_wheel_script_copies_the_vendored_notices_into_the_project(tmp_path: Path) -> None:
    result, project, _ = _run_wheel_script(tmp_path)
    assert result.returncode == 0, result.stderr
    copied = project / "THIRD-PARTY/tei-notices/kernels-0.1.0/NOTICE"
    assert copied.read_text(encoding="utf-8") == "kernels notice"


_MANYLINUX_STUB = """#!/bin/sh
# Stands in for a tool of the manylinux image; the build call records its environment.
name="$(basename "$0")"
case "${name}" in
  nproc) echo 1 ;;
  rustc) echo "rustc 1.0.0" ;;
  git) shift "$(($# - 1))"; mkdir -p "$1" ;;
  auditwheel) echo "${SOURCE_DATE_EPOCH-unset}" > "{record}/auditwheel-epoch" ;;
  nvprune) while [ "$#" -gt 0 ]; do [ "$1" = "-o" ] && : > "$2"; shift; done ;;
  cargo)
    mkdir -p target/release
    : > target/release/text-embeddings-router
    chmod 755 target/release/text-embeddings-router
    echo "${NVCC_APPEND_FLAGS-unset}" > "{record}/nvcc-flags"
    echo "${RUSTFLAGS-unset}" > "{record}/rustflags"
    ;;
esac
exit 0
"""
_IMAGE_TOOLS = ("dnf", "curl", "sh", "nproc", "rustc", "git", "nvcc", "nvprune", "cargo")
_IMAGE_TOOLS += ("auditwheel", "chown")
_IMAGE_ROOTS = re.compile(r"/(?:etc|usr/local|tmp|opt)(?=/)")


def _build_env(tmp_path: Path, cuda_cc: str, recorded: str) -> str:
    """Run tei_manylinux.sh on stub tools; return a variable as the cargo build sees it."""
    script = (_TOOLS / "tei_manylinux.sh").read_text(encoding="utf-8")
    script = re.sub(_IMAGE_ROOTS, lambda m: f"{tmp_path}{m.group(0)}", script)
    work = tmp_path / "work"
    (work / "tools/wheel-build").mkdir(parents=True)
    for name in ("install_cargo_about.sh", "tei_wheel.sh", "write_cuda_notice.py"):
        (work / "tools/wheel-build" / name).write_text("", encoding="utf-8")
    (work / "tools/wheel-build/source_date_epoch.sh").write_text(
        f"echo SOURCE_DATE_EPOCH={_COMMIT_TIME}\n", encoding="utf-8"
    )
    record = tmp_path / "record"
    record.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in _IMAGE_TOOLS:
        stub = bin_dir / name
        stub.write_text(_MANYLINUX_STUB.replace("{record}", str(record)), encoding="utf-8")
        stub.chmod(0o755)
    py = tmp_path / "opt/python/cp312-cp312/bin"
    py.mkdir(parents=True)
    (py / "python").symlink_to(bin_dir / "dnf")
    cuda = tmp_path / "usr/local/cuda-12.9"
    (cuda / "lib64/stubs").mkdir(parents=True)
    (cuda / "lib64/stubs/libcuda.so").write_text("", encoding="utf-8")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "etc").mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("NVCC_APPEND_FLAGS", "RUSTFLAGS")}
    env |= {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "TEI_REPO": "repo",
        "TEI_TAG": "tag",
        "RUSTC_VERSION": "1.0.0",
        "WHEEL_VERSION": "1.9.4",
        "BACKEND": "cuda",
        "WORK": str(work),
        "OUT": str(tmp_path / "out"),
        "OWNER": "1:1",
        "CUDA_CC": cuda_cc,
        "CUDA_PACKAGE_SUFFIX": "12-9",
        "CUDA_VERSION": "12.9.1",
    }
    rooted = tmp_path / "tei_manylinux_rooted.sh"
    rooted.write_text(script, encoding="utf-8")
    result = subprocess.run(
        ["bash", str(rooted)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return (record / recorded).read_text(encoding="utf-8").strip()


def test_the_cuda_build_compiles_kernels_position_independent(tmp_path: Path) -> None:
    assert _build_env(tmp_path, "89", "nvcc-flags") == "-Xcompiler -fPIC"


@pytest.mark.parametrize("cuda_cc", ["", "89"])
def test_auditwheel_repairs_with_the_commit_time_of_the_source(
    tmp_path: Path, cuda_cc: str
) -> None:
    assert _build_env(tmp_path, cuda_cc, "auditwheel-epoch") == str(_COMMIT_TIME)


def test_the_cpu_build_leaves_the_nvcc_flags_alone(tmp_path: Path) -> None:
    assert _build_env(tmp_path, "", "nvcc-flags") == "unset"


# The upstream source tree sets target-cpu=native in .cargo/config.toml; the RUSTFLAGS variable
# replaces that setting, so a router built on a CPU with AVX-512 still runs on one without it.
@pytest.mark.parametrize("cuda_cc", ["", "89"])
def test_the_linux_build_targets_the_portable_x86_64_baseline(tmp_path: Path, cuda_cc: str) -> None:
    assert _build_env(tmp_path, cuda_cc, "rustflags") == "-C target-cpu=x86-64-v2"


def test_the_metal_build_targets_the_first_apple_silicon_cpu() -> None:
    text = pack_workflow().read_text(encoding="utf-8")
    tei = text[text.index("\n  tei:\n") :]
    build = tei[tei.index("- name: Build the router for Metal") :]
    build = build[: build.index("- name: Collect the wheel")]
    lines = [line.strip() for line in build.splitlines()]
    exports = [line for line in lines if line.startswith("export RUSTFLAGS")]
    assert exports == ['export RUSTFLAGS="-C target-cpu=apple-m1"']
    assert build.index("export RUSTFLAGS") < build.index("cargo build")


def test_the_metal_repair_runs_with_the_commit_time_of_the_source() -> None:
    (build,) = (
        step["run"]
        for step in steps(pack_workflow(), "tei")
        if step.get("name") == "Build the router for Metal"
    )
    run = commands(build)
    epoch = "export $(bash tools/wheel-build/source_date_epoch.sh ${RUNNER_TEMP}/tei)"
    assert run.count(epoch) == 1
    wrap = run.index("tools/wheel-build/tei_wheel.sh")
    assert run.index(epoch) < wrap < run.index("delocate-wheel")


_SMOKE_QEMU_STUB = """#!/bin/sh
# Stands in for qemu-x86_64-static: records the call, then runs the program after -cpu <model>.
echo "$*" >> "{record}/qemu-args"
shift 2
exec "$@"
"""
_SMOKE_ROUTER_STUB = """#!/bin/sh
echo "router $*" >> "{record}/router-calls"
echo "text-embeddings-router 1.9.4"
"""
_SMOKE_UNAME_STUB = "#!/bin/sh\necho Linux\n"
_SMOKE_LDD_STUB = "#!/bin/sh\necho 'libc.so.6 => /lib/libc.so.6'\n"


def _run_smoke(
    tmp_path: Path, backend: str, cpu: str | None, qemu: str = _SMOKE_QEMU_STUB
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Run smoke_tei.sh on a stub router; return the record folder and the result."""
    record = tmp_path / "record"
    record.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "qemu-x86_64-static": qemu,
        "router": _SMOKE_ROUTER_STUB,
        "uname": _SMOKE_UNAME_STUB,
        "ldd": _SMOKE_LDD_STUB,
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(body.replace("{record}", str(record)), encoding="utf-8")
        stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "EMULATE_CPU"}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    if cpu is not None:
        env["EMULATE_CPU"] = cpu
    result = subprocess.run(
        ["bash", str(_TOOLS / "smoke_tei.sh"), str(bin_dir / "router"), backend],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=env,
    )
    return record, result


@pytest.mark.parametrize("backend", ["cuda-sm89", "cuda-sm120"])
def test_the_smoke_runs_the_router_on_an_emulated_cpu_without_avx(
    tmp_path: Path, backend: str
) -> None:
    record, result = _run_smoke(tmp_path, backend, "Nehalem")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (record / "qemu-args").read_text(encoding="utf-8").startswith("-cpu Nehalem ")
    assert (record / "router-calls").read_text(encoding="utf-8").strip() == "router --version"


def test_the_smoke_runs_the_router_directly_without_an_emulated_cpu(tmp_path: Path) -> None:
    record, result = _run_smoke(tmp_path, "cuda-sm89", None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (record / "qemu-args").exists()
    assert (record / "router-calls").read_text(encoding="utf-8").strip() == "router --version"


_SMOKE_SERVING_ROUTER_STUB = """#!/bin/sh
echo "router $*" >> "{record}/router-calls"
case "$1" in --version) echo "text-embeddings-router 1.9.4"; exit 0 ;; esac
exec sleep 30
"""
_SMOKE_CURL_STUB = """#!/bin/sh
case "$*" in
  */embed*) echo "[[$(yes 0 | head -n 384 | paste -s -d, -)]]" ;;
esac
"""


def test_the_cpu_smoke_serves_a_model_on_the_emulated_cpu(tmp_path: Path) -> None:
    record = tmp_path / "record"
    record.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "qemu-x86_64-static": _SMOKE_QEMU_STUB,
        "router": _SMOKE_SERVING_ROUTER_STUB,
        "uname": _SMOKE_UNAME_STUB,
        "ldd": _SMOKE_LDD_STUB,
        "curl": _SMOKE_CURL_STUB,
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(body.replace("{record}", str(record)), encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    result = subprocess.run(
        ["bash", str(_TOOLS / "smoke_tei.sh"), str(bin_dir / "router"), "cpu"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "EMULATE_CPU": "Nehalem",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "embedding length 384" in result.stdout
    calls = (record / "router-calls").read_text(encoding="utf-8").splitlines()
    assert [call.split()[1] for call in calls] == ["--version", "--model-id"]
    emulated = (record / "qemu-args").read_text(encoding="utf-8").splitlines()
    assert len(emulated) == 2
    assert all(line.startswith("-cpu Nehalem ") for line in emulated)


def test_the_emulated_smoke_fails_when_the_router_dies_on_the_cpu(tmp_path: Path) -> None:
    _, result = _run_smoke(tmp_path, "cuda-sm89", "Nehalem", "#!/bin/sh\nkill -ILL $$\n")
    assert result.returncode != 0


def test_the_linux_pack_cells_run_the_smoke_on_an_emulated_cpu() -> None:
    text = pack_workflow().read_text(encoding="utf-8")
    tei = text[text.index("\n  tei:\n") :]
    install = tei.index("- name: Install the CPU emulator")
    smoke = tei.index("- name: Start the router on an emulated CPU without AVX")
    step = tei[smoke : tei.index("- uses: actions/upload-artifact", smoke)]
    emulator = tei[install:smoke]
    assert install < smoke
    assert "if: matrix.os == 'linux'" in emulator
    assert "apt-get install -y qemu-user-static" in emulator
    assert "if: matrix.os == 'linux'" in step
    assert "EMULATE_CPU: Nehalem" in step
    assert step.count("smoke_tei.sh") == 1
    assert "continue-on-error" not in step
