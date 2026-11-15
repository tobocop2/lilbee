"""The wheel tools of the TEI pack: the project stamp and the wheel check."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools/wheel-build"
_PROJECT = Path(__file__).resolve().parents[1] / "packaging/engine-wheel-tei/pyproject.toml"
_DIST = "lilbee_engine_tei-1.9.4.dist-info"
_CPU_NAME = "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
_MAC_NAME = "lilbee_engine_tei-1.9.4-1.metal-py3-none-macosx_14_0_arm64.whl"
_EXECUTABLE = 0o100755 << 16
_PLAIN = 0o100644 << 16


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
    wheel = _wheel(tmp_path, expression=expression, extra=("NOTICE-NVIDIA-CUDA",))
    assert _problems(wheel, "cuda-sm89", expression) == []


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
    assert _problems(_wheel(tmp_path, expression=expression), "cuda-sm80", expression) == [
        "no non-empty licenses/NOTICE-NVIDIA-CUDA"
    ]


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
    wheel = _wheel(tmp_path, extra=("NOTICE-NVIDIA-CUDA",))
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


def test_main_needs_three_arguments_for_the_notice() -> None:
    with pytest.raises(SystemExit, match=r"Usage: write_cuda_notice\.py"):
        notice_tool.main(["notice"])


def _tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _scan(tmp_path: Path, cargo_files: dict[str, str], source_files: dict[str, str]) -> Path:
    registry = tmp_path / "cargo/registry/src/index-1"
    _tree(registry, cargo_files)
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
    assert text.endswith("Mozilla text\n")


def test_each_mpl_line_meets_the_pack_check_rule() -> None:
    text = mpl_tool.notice(_ABOUT, _LOCK)
    assert text is not None
    # The rule that the pack check applies to SOURCE-MPL-2.0 text, from the licence plan.
    rule = re.compile(
        r"https://static\.crates\.io/crates/(?P<crate>[\w-]+)/(?P=crate)-\d[\w.+-]*\.crate"
        r"[ \t]+[0-9a-f]{64}\b"
    )
    assert [match["crate"] for match in rule.finditer(text)] == ["option-ext", "webpki-roots"]


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


_WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/llama-pack-cells.yml"


def test_the_metal_build_enables_the_candle_backend() -> None:
    lines = [
        line
        for line in _WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "text-embeddings-router" in line and "--features" in line
    ]
    assert len(lines) == 1
    assert "--features candle,metal,http" in lines[0]


def test_the_wheel_script_writes_the_mpl_source_file_into_the_project() -> None:
    script = (_TOOLS / "tei_wheel.sh").read_text(encoding="utf-8")
    assert (
        'mpl_notice.py" "${out}/terms.json" "${src}/Cargo.lock" "${project}/SOURCE-MPL-2.0"'
        in script
    )


def test_the_wheel_script_builds_into_an_absolute_output_directory() -> None:
    script = (_TOOLS / "tei_wheel.sh").read_text(encoding="utf-8")
    resolve = script.index('out="$(cd "${out}" && pwd)"')
    assert resolve < script.index("-m build --wheel")
    assert resolve < script.index('--outdir "${out}/dist"')


def test_the_cuda_build_compiles_kernels_position_independent() -> None:
    script = (_TOOLS / "tei_manylinux.sh").read_text(encoding="utf-8")
    export = script.index('export NVCC_APPEND_FLAGS="-Xcompiler -fPIC"')
    assert export < script.index("cargo build -vv")
