"""Every text writer of tools/wheel-build writes "\\n" line endings, also on Windows.

A pack script runs these writers on a Windows runner. Python there turns "\\n" into "\\r\\n" in
a text file unless the writer passes newline="\\n". The tests simulate that translation and read
the written bytes, because a text-mode read turns "\\r\\n" back into "\\n" and hides the defect.
"""

from __future__ import annotations

import importlib.util
import io
import json
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools/wheel-build"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def windows_newlines(monkeypatch: pytest.MonkeyPatch) -> Callable[[], Any]:
    """Return a context manager in which a text file opened for writing with no newline argument gets "\\r\\n"."""
    real_open = io.open

    def windows_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "w" in mode and "b" not in mode and kwargs.get("newline") is None:
            kwargs["newline"] = "\r\n"
        return real_open(file, mode, *args, **kwargs)

    @contextmanager
    def active() -> Iterator[None]:
        with monkeypatch.context() as patch:
            patch.setattr(io, "open", windows_open)
            yield

    return active


def _assert_lf_only(*paths: Path) -> None:
    for path in paths:
        data = path.read_bytes()
        assert data, f"{path.name} is empty"
        assert b"\r" not in data, f"{path.name} holds a carriage return"


def test_the_simulation_turns_a_default_write_into_crlf(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    with windows_newlines():
        (tmp_path / "plain").write_text("a\nb\n", encoding="utf-8")
        (tmp_path / "lf").write_text("a\nb\n", encoding="utf-8", newline="\n")
    assert (tmp_path / "plain").read_bytes() == b"a\r\nb\r\n"
    assert (tmp_path / "lf").read_bytes() == b"a\nb\n"


def test_llama_cuda_notice(
    windows_newlines: Callable[[], Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load("llama_cuda_notice")
    page = "<p>Attachment A</p>\n" + "<p>" + "licence text " * 2000 + "</p>\n"

    class Reply:
        def read(self) -> bytes:
            return page.encode()

    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *a, **k: Reply())
    notice = tmp_path / "NOTICE-NVIDIA-CUDA"
    with windows_newlines():
        tool.main(notice)
    _assert_lf_only(notice)


def test_rust_licences(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("rust_licences")
    use = {"crate": {"name": "a", "version": "1.0", "repository": None}}
    report = {"licenses": [{"id": "MIT", "name": "MIT License", "text": "text\n", "used_by": [use]}]}
    with windows_newlines():
        tool.write_licences(report, tmp_path / "out")
    _assert_lf_only(*(tmp_path / "out").iterdir())


def test_rust_notices(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("rust_notices")
    uses = [{"crate": {"name": "a", "version": "1.0"}}]
    report = {
        "licenses": [
            {"id": "MPL-2.0", "used_by": uses},
            {"id": "OpenSSL", "used_by": uses},
        ]
    }
    lock = '[[package]]\nname = "a"\nversion = "1.0"\nchecksum = "abc"\n'
    tree = tmp_path / "tree"
    tree.mkdir()
    with windows_newlines():
        tool.write_notices(report, lock, tmp_path / "checkout", tmp_path / "registry", tree)
    _assert_lf_only(tree / "SOURCE-MPL-2.0", tree / "NOTICE-OpenSSL")


def test_embedded_ui_notices(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("embedded_ui_notices")
    bundle = tmp_path / "bundle.js"
    bundle.write_text("/**\n * @license React\n * Licensed under the MIT License\n */\n", encoding="utf-8")
    output = tmp_path / "NOTICE-UI"
    with windows_newlines():
        tool.main(["embedded_ui_notices", str(bundle), str(output)])
    _assert_lf_only(output)


def test_mpl_notice(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("mpl_notice")
    uses = [{"crate": {"name": "a", "version": "1.0"}}]
    about = tmp_path / "about.json"
    about.write_text(json.dumps({"licenses": [{"id": "MPL-2.0", "used_by": uses}]}), encoding="utf-8")
    lock = tmp_path / "Cargo.lock"
    lock.write_text('[[package]]\nname = "a"\nversion = "1.0"\nchecksum = "abc"\n', encoding="utf-8")
    output = tmp_path / "SOURCE-MPL-2.0"
    with windows_newlines():
        tool.main(["mpl_notice", str(about), str(lock), str(output)])
    _assert_lf_only(output)


def test_mpl_pypi_notice(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("mpl_pypi_notice")
    packages = tmp_path / "packages.txt"
    packages.write_text("foo-1.0\n", encoding="utf-8")
    sources = tmp_path / "sources.txt"
    sources.write_text("dist/foo-1.0.tar.gz abc123\n", encoding="utf-8")
    output = tmp_path / "SOURCE-MPL-2.0"
    with windows_newlines():
        tool.main(["mpl_pypi_notice", str(packages), str(sources), str(output)])
    _assert_lf_only(output)


def test_stamp_engine_wheel(windows_newlines: Callable[[], Any], tmp_path: Path) -> None:
    tool = _load("stamp_engine_wheel")
    project = tmp_path / "pyproject.toml"
    project.write_bytes(b'[project]\nversion = "0"\nrequires-python = ">=3.11"\n')
    with windows_newlines():
        tool.main(["stamp_engine_wheel", str(project), "1.2.3", "MIT"])
    _assert_lf_only(project)


def test_collect_python_licences(
    windows_newlines: Callable[[], Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load("collect_python_licences")
    monkeypatch.setattr(tool, "collect", lambda *a, **k: ({"MIT", "BSD-3-Clause"}, {"pkg-1.0"}, []))
    monkeypatch.setattr(tool, "interpreter_licences", lambda *a, **k: ({"PSF-2.0"}, []))
    terms = tmp_path / "terms.txt"
    mpl = tmp_path / "mpl.txt"
    with windows_newlines():
        tool.main(["--mpl", str(mpl), str(tmp_path), str(tmp_path / "out"), str(terms)])
    _assert_lf_only(terms, mpl)
    terms.write_bytes(b"MIT\n")
    with windows_newlines():
        tool.main(["--interpreter", str(tmp_path), str(tmp_path), str(tmp_path / "out"), str(terms)])
    _assert_lf_only(terms)
