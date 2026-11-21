"""The Windows bundler ships the Visual C++ runtime and fails on an absent file."""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
import subprocess
import sys
import zipfile

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the helper is driven here with a POSIX fake of cygpath",
)

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/bundle_windows_runtime.sh"
)
_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def _executable(path: pathlib.Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


# Folder names inside VC/Redist/MSVC/<version>/x64. VC145 is the toolset number expected for
# Visual Studio 2026 (toolset 14.5x); it is not confirmed from a runner or a Microsoft page.
_TOOLSETS = ("VC143", "VC145")


def _redist_tree(root: pathlib.Path, toolset: str, marker: bytes) -> None:
    """One x64 redist folder holding the CRT and OpenMP DLLs, every file set to `marker`."""
    crt = root / f"Microsoft.{toolset}.CRT"
    omp = root / f"Microsoft.{toolset}.OpenMP"
    crt.mkdir(parents=True)
    omp.mkdir(parents=True)
    for name in _CRT:
        (crt / name).write_bytes(marker)
    (omp / "vcomp140.dll").write_bytes(marker)


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
    _redist_tree(tmp_path / "redist/x64", "VC143", b"x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "fake").mkdir()
    _executable(tmp_path / "fake/cygpath", 'printf "%s\\n" "$2"\n')
    return tmp_path


def _launch(tree: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env["PATH"] = f"{tree / 'fake'}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(_SCRIPT), str(tree / "bin")],
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        stdin=subprocess.DEVNULL,
    )


def _run(tree: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return _launch(tree, {**os.environ, "VCToolsRedistDir": str(tree / "redist")})


def _run_without_vc_env(
    tree: pathlib.Path, vswhere: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "VCToolsRedistDir"}
    env["VSWHERE"] = str(vswhere)
    return _launch(tree, env)


def _vs_install(
    tree: pathlib.Path, versions: tuple[str, ...], toolset: str = "VC143"
) -> pathlib.Path:
    """A Visual Studio install with one redist tree per version, and a fake vswhere."""
    install = tree / "vs"
    redist = install / "VC/Redist/MSVC"
    for version in versions:
        _redist_tree(redist / version / "x64", toolset, version.encode())
    _executable(
        tree / "fake/vswhere.exe",
        f'printf "%s\\n" "$*" > "{tree}/vswhere.args"\nprintf "%s\\n" "{install}"\n',
    )
    return install


def test_ships_the_crt_and_the_openmp_runtime(tree) -> None:
    result = _run(tree)
    assert result.returncode == 0, result.stderr
    shipped = {p.name for p in (tree / "bin").iterdir()}
    assert {*_CRT, "vcomp140.dll"} <= shipped


@pytest.mark.parametrize("toolset", _TOOLSETS)
def test_ships_the_runtime_whatever_the_toolset_number(tree, toolset: str) -> None:
    _vs_install(tree, ("14.44.35112",), toolset)
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/vcomp140.dll").read_bytes() == b"14.44.35112"


def test_asks_vswhere_for_the_latest_install_with_the_cpp_tools(tree) -> None:
    _vs_install(tree, ("14.44.35112",))
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    asked = (tree / "vswhere.args").read_text(encoding="utf-8")
    assert "-latest" in asked
    assert "-products *" in asked
    assert "-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in asked
    assert "-property installationPath" in asked


def test_picks_the_highest_version_not_the_last_in_text_order(tree) -> None:
    # Text order puts 14.9 after 14.10; version order puts 14.10 last.
    _vs_install(tree, ("14.9.30000", "14.10.40000"), "VC145")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/msvcp140.dll").read_bytes() == b"14.10.40000"


def test_skips_the_onecore_and_non_redist_folders_and_the_v_folder(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    redist = install / "VC/Redist/MSVC"
    decoys = (
        "14.44.35112/onecore/x64",
        "14.44.35112/debug_nonredist/x64",
        "14.45.1/onecore/x64",
        "14.45.1/debug_nonredist/x64",
        "v143/x64",
    )
    for decoy in decoys:
        _redist_tree(redist / decoy, "VC143", b"decoy")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "bin/msvcp140.dll").read_bytes() == b"14.44.35112"


def test_vswhere_naming_no_install_refuses(tree) -> None:
    _executable(tree / "fake/vswhere.exe", "exit 0\n")
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert "no Visual Studio with the C++ tools found" in result.stderr


def test_an_absent_vswhere_refuses(tree) -> None:
    result = _run_without_vc_env(tree, tree / "fake/missing-vswhere.exe")
    assert result.returncode != 0
    assert "no Visual Studio with the C++ tools found" in result.stderr


def test_an_install_with_no_redist_folder_says_so(tree) -> None:
    install = _vs_install(tree, ())
    (install / "VC/Redist/MSVC").mkdir(parents=True)
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert "no MSVC redistributable folder under" in result.stderr


@pytest.mark.parametrize("missing", ["CRT", "OpenMP"])
def test_an_install_with_no_runtime_folder_names_it(tree, missing: str) -> None:
    install = _vs_install(tree, ("14.44.35112",), "VC145")
    folder = install / f"VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC145.{missing}"
    for entry in folder.iterdir():
        entry.unlink()
    folder.rmdir()
    result = _run_without_vc_env(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert f"no Microsoft.VC*.{missing} folder under" in result.stderr


def test_a_missing_redist_file_fails(tree) -> None:
    (tree / "redist/x64/Microsoft.VC143.OpenMP/vcomp140.dll").unlink()
    result = _run(tree)
    assert result.returncode != 0
    assert "vcomp140.dll" in result.stderr


# The real terms hold curly quotes (U+2018, U+2019, U+201C, U+201D): Windows-1252 encodes
# each as one byte that is not valid UTF-8.
_LSQ, _RSQ, _LDQ, _RDQ = (chr(code) for code in (0x2018, 0x2019, 0x201C, 0x201D))
_TERMS = (
    "DISTRIBUTABLE CODE. The software contains code that you may distribute.",
    f"Distribution Requirements. You must add {_LDQ}significant primary functionality{_RDQ}.",
    f"Distribution Restrictions. You may not use Microsoft{_RSQ}s {_LSQ}trademarks{_RSQ}.",
)
_SENTENCE = "They are distributable code"
_VERIFY = _SCRIPT.parent / "vs_distributable_terms.py"


def _runner_python(tree: pathlib.Path) -> str:
    """A launcher for a Python whose stdout encodes as Windows-1252, like the Windows runner's."""
    shim = tree / "fake/cp1252_stdout.py"
    shim.write_text(
        "import runpy, sys\n"
        "sys.stdout.reconfigure(encoding='cp1252')\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n",
        encoding="utf-8",
    )
    launcher = tree / "fake/python-cp1252"
    _executable(launcher, f'shift\nexec "{sys.executable}" -I "{shim}" "$@"\n')
    return str(launcher)


def _licence_docx(path: pathlib.Path, paragraphs: tuple[str, ...]) -> str:
    """Write a minimal Word file holding *paragraphs*; return its sha256."""
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_with_notice(
    tree: pathlib.Path,
    vswhere: pathlib.Path,
    paragraphs: tuple[str, ...] = (
        "UNRELATED. Other terms.",
        *_TERMS,
        "DEVELOPING EXTENSIONS. Not part of the notice.",
    ),
    sha256: str | None = None,
) -> subprocess.CompletedProcess[str]:
    docx = tree / "licence.docx"
    digest = _licence_docx(docx, paragraphs)
    env = {k: v for k, v in os.environ.items() if k != "VCToolsRedistDir"}
    env["VSWHERE"] = str(vswhere)
    env["VS_LICENSE_URL"] = docx.as_uri()
    env["VS_LICENSE_SHA256"] = sha256 or digest
    env["PYTHON"] = _runner_python(tree)
    env["PATH"] = f"{tree / 'fake'}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(_SCRIPT), str(tree / "bin"), str(tree / "NOTICE-VC"), str(tree / "kept.docx")],
        env=env,
        capture_output=True,
        encoding="utf-8",
        check=False,
        stdin=subprocess.DEVNULL,
    )


def test_writes_the_redistribution_notice_from_the_install_terms(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    terms = "Distributable Code terms for the runtime files. " * 40
    (install / "VC").joinpath("REDIST.TXT").write_bytes(terms.replace(" ", "\r\n", 1).encode())
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    notice = (tree / "NOTICE-VC").read_bytes()
    assert notice.startswith(
        b"These files are licensed under the Microsoft terms below, "
        b"not under the other terms of this wheel."
    )
    assert b"Distributable Code terms for the runtime files." in notice
    assert b"\r" not in notice


def test_the_notice_names_every_runtime_file_and_holds_the_distributable_code_terms(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    notice = (tree / "NOTICE-VC").read_text(encoding="utf-8")
    for name in (*_CRT, "vcomp140.dll"):
        assert name in notice
    assert _SENTENCE in notice
    for paragraph in _TERMS:
        assert paragraph in notice
    assert "UNRELATED" not in notice
    assert "DEVELOPING EXTENSIONS" not in notice


def test_a_licence_with_the_wrong_digest_writes_no_notice(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(tree, tree / "fake/vswhere.exe", sha256="0" * 64)
    assert result.returncode != 0
    assert "the Visual Studio licence has digest" in result.stderr
    assert not (tree / "NOTICE-VC").exists()


def test_a_licence_without_the_distributable_code_section_writes_no_notice(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(
        tree, tree / "fake/vswhere.exe", paragraphs=("UNRELATED. Other terms.",)
    )
    assert result.returncode != 0
    assert "no DISTRIBUTABLE CODE section" in result.stderr
    assert not (tree / "NOTICE-VC").exists()


def test_the_notice_is_valid_utf8_when_the_runner_writes_stdout_as_windows_1252(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    notice = (tree / "NOTICE-VC").read_bytes().decode("utf-8")
    assert f"{_LDQ}significant primary functionality{_RDQ}" in notice
    assert f"Microsoft{_RSQ}s {_LSQ}trademarks{_RSQ}" in notice


def test_keeps_a_copy_of_the_verified_licence_for_the_wheel_check(tree) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    assert (tree / "kept.docx").read_bytes() == (tree / "licence.docx").read_bytes()


@pytest.mark.parametrize("phrase", ["Distribution Requirements", "Distribution Restrictions"])
def test_a_section_without_a_required_phrase_writes_no_notice(tree, phrase: str) -> None:
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    kept = tuple(line for line in _TERMS if not line.startswith(phrase))
    assert len(kept) == len(_TERMS) - 1
    result = _run_with_notice(tree, tree / "fake/vswhere.exe", paragraphs=kept)
    assert result.returncode != 0
    assert f"the DISTRIBUTABLE CODE section has no '{phrase}'" in result.stderr
    assert not (tree / "NOTICE-VC").exists()


def _wheel_with_notice(
    path: pathlib.Path, notice: bytes, name: str = "NOTICE-MICROSOFT-VC-RUNTIME"
) -> str:
    """A wheel that holds *notice* in its licence folder."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"lilbee_engine_llama_cpp-1.12032.1.dist-info/licenses/{name}", notice)
    return str(path)


def _verify(docx: pathlib.Path, wheel: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(_VERIFY), "--verify", str(docx), wheel],
        capture_output=True,
        encoding="utf-8",
        check=False,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture
def built(tree) -> pathlib.Path:
    """A notice that the bundler wrote, and the licence copy it kept."""
    install = _vs_install(tree, ("14.44.35112",))
    (install / "VC").joinpath("REDIST.TXT").write_bytes(b"the list")
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode == 0, result.stderr
    return tree


def test_the_notice_in_a_built_wheel_decodes_as_utf8_and_holds_the_terms_line_for_line(
    built,
) -> None:
    wheel = _wheel_with_notice(built / "ok.whl", (built / "NOTICE-VC").read_bytes())
    result = _verify(built / "kept.docx", wheel)
    assert result.returncode == 0, result.stderr


def test_a_notice_in_windows_1252_fails_the_wheel_check(built) -> None:
    text = (built / "NOTICE-VC").read_text(encoding="utf-8")
    wheel = _wheel_with_notice(built / "cp.whl", text.encode("cp1252"))
    result = _verify(built / "kept.docx", wheel)
    assert result.returncode == 1
    assert "the notice is not valid UTF-8" in result.stderr


def test_a_notice_that_changes_a_line_of_the_terms_fails_the_wheel_check(built) -> None:
    text = (built / "NOTICE-VC").read_text(encoding="utf-8")
    changed = text.replace(f"{_LDQ}significant", "significant")
    assert changed != text
    wheel = _wheel_with_notice(built / "line.whl", changed.encode("utf-8"))
    result = _verify(built / "kept.docx", wheel)
    assert result.returncode == 1
    assert "does not hold the licence terms line for line" in result.stderr


def test_a_notice_without_the_distributable_code_sentence_fails_the_wheel_check(built) -> None:
    text = (built / "NOTICE-VC").read_text(encoding="utf-8")
    changed = text.replace(_SENTENCE, "They are code")
    assert changed != text
    wheel = _wheel_with_notice(built / "sentence.whl", changed.encode("utf-8"))
    result = _verify(built / "kept.docx", wheel)
    assert result.returncode == 1
    assert f"the notice lacks the sentence '{_SENTENCE}'" in result.stderr


def test_a_wheel_without_the_notice_fails_the_wheel_check(built) -> None:
    wheel = _wheel_with_notice(built / "none.whl", b"x", name="NOTICE-OTHER")
    result = _verify(built / "kept.docx", wheel)
    assert result.returncode == 1
    assert "the wheel has 0 Visual C++ runtime notices, not one" in result.stderr


def test_the_terms_tool_refuses_a_call_with_the_wrong_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(_VERIFY), "licence.docx"],
        capture_output=True,
        encoding="utf-8",
        check=False,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_fails_when_the_install_has_no_redistribution_terms(tree) -> None:
    _vs_install(tree, ("14.44.35112",))
    result = _run_with_notice(tree, tree / "fake/vswhere.exe")
    assert result.returncode != 0
    assert "no REDIST.TXT" in result.stderr
    assert not (tree / "NOTICE-VC").exists()


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/actions/bundle-llama-server/action.yml",
        ".github/workflows/build-multigpu.yml",
        ".github/workflows/warm-engine-cache.yml",
    ],
)
def test_every_engine_cache_key_hashes_the_helper(workflow: str) -> None:
    text = (_SCRIPT.parents[2] / workflow).read_text(encoding="utf-8")
    assert "tools/wheel-build/bundle_windows_runtime.sh" in text
