"""The repack of an upstream OpenVINO llama.cpp archive into the engine wheel tree."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "wheel-build" / "repack_openvino.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("repack_openvino", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repack_openvino = _load()
LICENSING = repack_openvino.LICENSING_DIR


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"x", link: str | None = None) -> None:
    info = tarfile.TarInfo(f"top/{name}")
    if link is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = link
        tar.addfile(info)
        return
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _linux_archive(path: Path, *, with_licence: bool = True, second_server: bool = False) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, "llama-server")
        _add(tar, "llama-cli")
        _add(tar, "libggml.so.0.28.0", b"ggml")
        _add(tar, "libggml.so.0", link="libggml.so.0.28.0")
        _add(tar, "libggml.so", link="libggml.so.0")
        _add(tar, "libggml-cpu-x64.so", b"cpu")
        _add(tar, "cache.json", b"{}")
        _add(tar, "LICENSE", b"MIT")
        if with_licence:
            _add(tar, f"{LICENSING}/LICENSE", b"Apache")
            _add(tar, f"{LICENSING}/EULA.txt", b"eula")
        if second_server:
            _add(tar, "sub/llama-server")
    return path


def test_the_server_and_each_library_install_under_the_soname(tmp_path: Path) -> None:
    wheel = tmp_path / "wheel"
    repack_openvino.repack(_linux_archive(tmp_path / "a.tar.gz"), wheel)
    bin_dir = wheel / "lilbee_engine_openvino" / "bin"
    assert sorted(p.name for p in bin_dir.iterdir()) == [
        "cache.json",
        "libggml-cpu-x64.so",
        "libggml.so.0",
        "llama-server",
    ]
    assert (bin_dir / "libggml.so.0").read_bytes() == b"ggml"
    assert not any(p.is_symlink() for p in bin_dir.iterdir())


def test_the_licences_go_to_the_wheel_root(tmp_path: Path) -> None:
    wheel = tmp_path / "wheel"
    repack_openvino.repack(_linux_archive(tmp_path / "a.tar.gz"), wheel)
    assert (wheel / "LICENSE-llama.cpp").read_bytes() == b"MIT"
    assert (wheel / "LICENSE-OpenVINO").read_bytes() == b"Apache"
    assert (wheel / "THIRD-PARTY" / "OpenVINO" / "EULA.txt").read_bytes() == b"eula"


def test_an_archive_without_the_openvino_licence_is_refused(tmp_path: Path) -> None:
    archive = _linux_archive(tmp_path / "a.tar.gz", with_licence=False)
    with pytest.raises(repack_openvino.RepackError, match="no openvino-licensing/LICENSE"):
        repack_openvino.repack(archive, tmp_path / "wheel")


def test_an_archive_with_two_servers_is_refused(tmp_path: Path) -> None:
    archive = _linux_archive(tmp_path / "a.tar.gz", second_server=True)
    with pytest.raises(repack_openvino.RepackError, match="found 2"):
        repack_openvino.repack(archive, tmp_path / "wheel")


def _windows_archive(path: Path, *, openmp_licence: bool) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in (
            "llama-server.exe",
            "openvino.dll",
            "tbb12.dll",
            "tbb12_debug.dll",
            "libomp.dll",
        ):
            zf.writestr(name, "x")
        zf.writestr("LICENSE", "MIT")
        zf.writestr(f"{LICENSING}/LICENSE", "Apache")
        if openmp_licence:
            zf.writestr("LICENSE-LLVM-OpenMP", "omp")
    return path


def test_a_windows_zip_drops_the_debug_libraries_and_keeps_the_openmp_licence(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "wheel"
    repack_openvino.repack(_windows_archive(tmp_path / "a.zip", openmp_licence=True), wheel)
    names = sorted(p.name for p in (wheel / "lilbee_engine_openvino" / "bin").iterdir())
    assert names == ["libomp.dll", "llama-server.exe", "openvino.dll", "tbb12.dll"]
    assert (wheel / "LICENSE-LLVM-OpenMP").read_bytes() == b"omp"


def test_libomp_without_its_licence_is_refused(tmp_path: Path) -> None:
    archive = _windows_archive(tmp_path / "a.zip", openmp_licence=False)
    with pytest.raises(repack_openvino.RepackError, match=r"libomp\.dll without"):
        repack_openvino.repack(archive, tmp_path / "wheel")


def test_main_returns_the_status_of_the_repack(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _linux_archive(tmp_path / "a.tar.gz", with_licence=False)
    assert repack_openvino.main([str(archive), str(tmp_path / "wheel")]) == 1
    assert "no openvino-licensing/LICENSE" in capsys.readouterr().err
    good = _linux_archive(tmp_path / "b.tar.gz")
    assert repack_openvino.main([str(good), str(tmp_path / "wheel2")]) == 0
    assert repack_openvino.main([]) == 2
