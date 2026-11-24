"""The two Python programs of the llama.cpp pack: the repack and the CUDA notice."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "wheel-build"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repack = _load("repack_llama_archives")
notice = _load("llama_cuda_notice")


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    path = tmp_path / "project" / "module" / "bin"
    path.mkdir(parents=True)
    return path


def _zip(path: Path, *names: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, f"{path.name}:{name}")
    return path


def _tar(path: Path, *names: str, links: dict[str, str] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            data = f"{path.name}:{name}".encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return path


def _names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def test_a_server_archive_gives_the_server_and_its_libraries_only(
    tmp_path: Path, bin_dir: Path
) -> None:
    archive = _tar(
        tmp_path / "a.tar.gz",
        "top/llama-server",
        "top/llama-cli",
        "top/libggml.so.0.1",
        "top/README.so.md",
        links={"top/libggml.so": "libggml.so.0.1"},
    )
    repack.main(bin_dir, [archive])
    assert _names(bin_dir) == ["libggml.so.0.1", "llama-server"]
    assert (bin_dir / "llama-server").read_bytes() == b"a.tar.gz:top/llama-server"
    # The archive mode is kept. Windows has no exec bit: it reports any writable file as 0o666.
    mode = 0o666 if os.name == "nt" else 0o755
    assert (bin_dir / "llama-server").stat().st_mode & 0o777 == mode


def test_a_runtime_archive_adds_its_libraries(tmp_path: Path, bin_dir: Path) -> None:
    server = _zip(tmp_path / "server.zip", "llama-server.exe", "ggml.dll")
    runtime = _zip(tmp_path / "runtime.zip", "cudart64_12.dll", "readme.txt")
    repack.main(bin_dir, [server, runtime])
    assert _names(bin_dir) == ["cudart64_12.dll", "ggml.dll", "llama-server.exe"]


def test_a_dylib_gets_the_name_of_its_install_id(
    tmp_path: Path, bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = {"libggml.0.1.dylib": "@rpath/libggml.0.dylib\n", "libplain.dylib": ""}

    def otool(argv: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ["otool", "-D"]
        assert options["check"] is True
        return subprocess.CompletedProcess(argv, 0, f"{argv[2]}:\n{ids[Path(argv[2]).name]}", "")

    monkeypatch.setattr(subprocess, "run", otool)
    archive = _tar(tmp_path / "a.tar.gz", *(f"t/{name}" for name in (*ids, "llama-server")))
    repack.main(bin_dir, [archive])
    assert _names(bin_dir) == ["libggml.0.dylib", "libplain.dylib", "llama-server"]


def test_the_openmp_licence_goes_into_the_wheel_project(tmp_path: Path, bin_dir: Path) -> None:
    licence = "LICENSE-LLVM-OpenMP"
    archive = _zip(tmp_path / "a.zip", "llama-server.exe", "libomp.dll", licence)
    repack.main(bin_dir, [archive])
    assert _names(bin_dir) == ["libomp.dll", "llama-server.exe"]
    assert (bin_dir.parents[1] / licence).read_bytes() == f"a.zip:{licence}".encode()


def test_openmp_without_its_licence_is_refused(tmp_path: Path, bin_dir: Path) -> None:
    archive = _zip(tmp_path / "a.zip", "llama-server.exe", "libomp.dll")
    with pytest.raises(SystemExit, match=r"a\.zip: libomp\.dll without LICENSE-LLVM-OpenMP"):
        repack.main(bin_dir, [archive])


@pytest.mark.parametrize(
    ("names", "count"),
    [(("a/llama-server", "b/llama-server"), 2), (("a/libggml.so",), 0)],
)
def test_an_archive_without_one_server_folder_is_refused(
    tmp_path: Path, bin_dir: Path, names: tuple[str, ...], count: int
) -> None:
    archive = _tar(tmp_path / "a.tar.gz", *names)
    with pytest.raises(SystemExit, match=f"expected one folder to repack, found {count}"):
        repack.main(bin_dir, [archive])


def test_two_files_of_one_name_are_refused(tmp_path: Path, bin_dir: Path) -> None:
    server = _zip(tmp_path / "server.zip", "llama-server.exe", "ggml.dll")
    runtime = _zip(tmp_path / "runtime.zip", "ggml.dll")
    with pytest.raises(SystemExit, match=r"runtime\.zip: two files install as ggml\.dll"):
        repack.main(bin_dir, [server, runtime])
    assert (bin_dir / "ggml.dll").read_bytes() == b"server.zip:ggml.dll"


def test_an_archive_whose_server_is_a_link_is_refused(tmp_path: Path, bin_dir: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", "t/keep", links={"t/llama-server": "keep"})
    with pytest.raises(SystemExit, match=r"a\.tar\.gz: no llama-server in the tree"):
        repack.main(bin_dir, [archive])


def test_a_member_that_would_leave_the_folder_is_refused(
    tmp_path: Path, bin_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `data` filter of the extraction refuses a path outside the scratch folder."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    archive = _tar(tmp_path / "a.tar.gz", "t/llama-server", "../escape")
    with pytest.raises(tarfile.OutsideDestinationError):
        repack.main(bin_dir, [archive])
    assert not (scratch / "escape").exists()


_LINE = "<p>The terms of the agreement, line by line.</p>"


def _page(monkeypatch: pytest.MonkeyPatch, page: str) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def urlopen(url: str, timeout: int) -> io.BytesIO:
        calls.append((url, timeout))
        return io.BytesIO(page.encode("utf-8"))

    monkeypatch.setattr(notice.urllib.request, "urlopen", urlopen)
    return calls


def test_the_notice_is_the_header_and_the_text_of_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = (
        "<html><style>p {}</style><script>run()</script><h1>  Attachment A </h1>"
        "<ul><li>café<li>two<br>three <b>bold</b></ul>" + _LINE * 600
    )
    calls = _page(monkeypatch, page)
    target = tmp_path / "NOTICE"
    notice.main(target)
    text = target.read_text(encoding="utf-8")  # a text file: Windows writes CRLF line ends
    assert calls == [("https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html", 60)]
    assert text.startswith("This wheel carries NVIDIA CUDA 12.4 runtime files: the CUDA Runtime")
    head, body = text.split(f"from {calls[0][0]}\n\n")
    assert len(head.splitlines()) == 7
    lines = body.splitlines()
    assert lines[:4] == ["Attachment A", "café", "two", "three bold"]
    assert lines[4:] == ["The terms of the agreement, line by line."] * 600
    assert text.endswith("line.\n")


@pytest.mark.parametrize("page", ["<p>Attachment A</p>", _LINE * 600])
def test_a_page_that_is_short_or_lacks_attachment_a_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str
) -> None:
    _page(monkeypatch, page)
    with pytest.raises(SystemExit, match="the EULA text from https://docs"):
        notice.main(tmp_path / "NOTICE")
    assert not (tmp_path / "NOTICE").exists()
