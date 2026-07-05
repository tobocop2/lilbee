"""Tests for tools/vendor/cuda_runtime.py."""

from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path

import pytest
from tools.vendor import cuda_runtime

from tests._wheel_fixture import make_wheel

_WINDOWS_DLLS = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
_LINUX_SONAMES = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")


def _make_llama_wheel(tmp: Path, *, plat: str = "win_amd64") -> Path:
    return make_wheel(
        tmp,
        name="llama_cpp_python",
        version="0.3.23",
        python="cp313",
        abi="cp313",
        plat=plat,
        pkg_dir="llama_cpp",
        pkg_files={
            "__init__.py": b"# llama_cpp\n",
            "lib/llama.dll": b"\x00" * 64,
            "lib/ggml-cuda.dll": b"\x01" * 64,
        },
    )


def _make_windows_cuda_bin(tmp: Path) -> Path:
    cuda_bin = tmp / "cuda" / "bin"
    cuda_bin.mkdir(parents=True)
    for name in _WINDOWS_DLLS:
        (cuda_bin / name).write_bytes(f"{name} contents".encode())
    return cuda_bin


def _make_linux_cuda_lib64(tmp: Path) -> Path:
    """Toolkit layout: real versioned files plus SONAME symlinks."""
    lib64 = tmp / "cuda" / "lib64"
    lib64.mkdir(parents=True)
    for soname in _LINUX_SONAMES:
        real = lib64 / f"{soname}.5.82"
        real.write_bytes(f"{soname} contents".encode())
        (lib64 / soname).symlink_to(real)
    return lib64


class TestBundleCudaRuntimeWindows:
    def test_injects_runtime_dlls_into_lib(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path)
        cuda_bin = _make_windows_cuda_bin(tmp_path)

        cuda_runtime.bundle_cuda_runtime(whl, cuda_bin)

        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())
        for dll in _WINDOWS_DLLS:
            assert f"llama_cpp/lib/{dll}" in names

    def test_preserves_existing_wheel_contents(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path)
        cuda_bin = _make_windows_cuda_bin(tmp_path)

        cuda_runtime.bundle_cuda_runtime(whl, cuda_bin)

        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())
            llama = zf.read("llama_cpp/lib/llama.dll")
        assert "llama_cpp/__init__.py" in names
        assert "llama_cpp/lib/ggml-cuda.dll" in names
        assert llama == b"\x00" * 64

    def test_record_covers_added_dlls(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path)
        cuda_bin = _make_windows_cuda_bin(tmp_path)

        cuda_runtime.bundle_cuda_runtime(whl, cuda_bin)

        with zipfile.ZipFile(whl) as zf:
            record = zf.read("llama_cpp_python-0.3.23.dist-info/RECORD").decode()

        data = b"cudart64_12.dll contents"
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        expected = f"llama_cpp/lib/cudart64_12.dll,sha256={digest},{len(data)}"
        assert expected in record.splitlines()

    def test_missing_runtime_dll_fails(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path)
        cuda_bin = tmp_path / "cuda" / "bin"
        cuda_bin.mkdir(parents=True)
        (cuda_bin / "cudart64_12.dll").write_bytes(b"cudart")

        with pytest.raises(RuntimeError, match=r"cublas64_\*\.dll"):
            cuda_runtime.bundle_cuda_runtime(whl, cuda_bin)

    def test_wheel_without_lib_dir_fails(self, tmp_path: Path) -> None:
        whl = make_wheel(
            tmp_path,
            name="llama_cpp_python",
            version="0.3.23",
            python="cp313",
            abi="cp313",
            plat="win_amd64",
            pkg_dir="llama_cpp",
            pkg_files={"__init__.py": b"# llama_cpp\n"},
        )
        cuda_bin = _make_windows_cuda_bin(tmp_path)

        with pytest.raises(RuntimeError, match="llama_cpp/lib"):
            cuda_runtime.bundle_cuda_runtime(whl, cuda_bin)

    def test_missing_cuda_lib_dir_fails(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path)

        with pytest.raises(RuntimeError, match="CUDA library directory"):
            cuda_runtime.bundle_cuda_runtime(whl, tmp_path / "nope")


class TestBundleCudaRuntimeLinux:
    def test_injects_sonames_dereferencing_symlinks(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path, plat="manylinux_2_17_x86_64.manylinux2014_x86_64")
        lib64 = _make_linux_cuda_lib64(tmp_path)

        cuda_runtime.bundle_cuda_runtime(whl, lib64)

        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())
            cudart = zf.read("llama_cpp/lib/libcudart.so.12")
        for soname in _LINUX_SONAMES:
            assert f"llama_cpp/lib/{soname}" in names
        # the SONAME entry holds the real bytes, and the long-versioned
        # filename is not duplicated alongside it
        assert cudart == b"libcudart.so.12 contents"
        assert "llama_cpp/lib/libcudart.so.12.5.82" not in names

    def test_missing_soname_fails(self, tmp_path: Path) -> None:
        whl = _make_llama_wheel(tmp_path, plat="manylinux_2_17_x86_64.manylinux2014_x86_64")
        lib64 = tmp_path / "cuda" / "lib64"
        lib64.mkdir(parents=True)
        (lib64 / "libcudart.so.12").write_bytes(b"cudart")
        (lib64 / "libcublas.so.12").write_bytes(b"cublas")

        with pytest.raises(RuntimeError, match=r"libcublasLt\.so\.\*"):
            cuda_runtime.bundle_cuda_runtime(whl, lib64)


class TestRuntimeLibPatterns:
    def test_windows_wheel_gets_dll_patterns(self) -> None:
        name = "llama_cpp_python-0.3.23-cp313-cp313-win_amd64.whl"
        assert cuda_runtime.runtime_lib_patterns(name) == cuda_runtime.WINDOWS_LIB_PATTERNS

    def test_linux_wheel_gets_so_patterns(self) -> None:
        name = "llama_cpp_python-0.3.23-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
        assert cuda_runtime.runtime_lib_patterns(name) == cuda_runtime.LINUX_LIB_PATTERNS

    def test_macos_wheel_rejected(self) -> None:
        name = "llama_cpp_python-0.3.23-cp313-cp313-macosx_11_0_arm64.whl"
        with pytest.raises(RuntimeError, match="Windows and Linux"):
            cuda_runtime.runtime_lib_patterns(name)


class TestMain:
    def test_main_integration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        whl = _make_llama_wheel(tmp_path)
        cuda_bin = _make_windows_cuda_bin(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["cuda_runtime.py", str(whl), "--cuda-lib", str(cuda_bin)],
        )

        cuda_runtime.main()

        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())
        assert "llama_cpp/lib/cublasLt64_12.dll" in names

    def test_main_missing_wheel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            ["cuda_runtime.py", "/nonexistent/llama.whl", "--cuda-lib", "/nonexistent/bin"],
        )
        with pytest.raises(SystemExit):
            cuda_runtime.main()
