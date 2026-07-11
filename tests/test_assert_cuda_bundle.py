"""Tests for the CUDA-bundle gate that keeps a runtime-less engine wheel off a release."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "qa" / "assert_cuda_bundle.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assert_cuda_bundle", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acb = _load()

_LINUX_RUNTIME = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")
_WINDOWS_RUNTIME = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")


def _wheel(tmp_path: Path, name: str, bin_files: tuple[str, ...]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for entry in bin_files:
            zf.writestr(f"lilbee_engine/bin/{entry}", b"x")
    return path


def test_linux_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    # The shipped-broken shape: ggml-cuda present, the runtime it links absent.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 3
    assert any("libcudart.so" in p for p in problems)
    assert any("libcublasLt.so" in p for p in problems)


def test_linux_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so", *_LINUX_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_windows_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll"),
    )
    assert len(acb.check(wheel)) == 3


def test_windows_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_cuda_detected_from_payload_not_filename(tmp_path: Path) -> None:
    # A freshly built wheel has no -1.cu125- build tag; that is added at attach time.
    # Detection must still fire, or the gate silently passes every build artifact.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll"),
    )
    assert len(acb.check(wheel)) == 3


def test_non_cuda_wheel_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.check(wheel) == []


def test_non_cuda_wheel_carrying_cuda_runtime_is_rejected(tmp_path: Path) -> None:
    # Guards the other direction: the copy step must stay scoped to CUDA backends.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "cudart64_12.dll"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "non-CUDA wheel ships CUDA runtime" in problems[0]


def test_empty_bin_dir_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lilbee_engine-1.0-py3-none-win_amd64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("lilbee_engine/__init__.py", b"")
    assert acb.check(path) == [f"{path.name}: no lilbee_engine/bin/ payload at all"]


def test_macos_cuda_payload_asserts_nothing(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-macosx_11_0_arm64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.check(wheel) == []


def test_main_returns_nonzero_and_names_the_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.main([str(wheel)]) == 1
    assert "libcudart.so" in capsys.readouterr().err


def test_main_returns_zero_on_a_good_wheel(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so", *_LINUX_RUNTIME),
    )
    assert acb.main([str(wheel)]) == 0


def test_main_rejects_a_missing_file(tmp_path: Path) -> None:
    assert acb.main([str(tmp_path / "nope.whl")]) == 1


def test_main_with_no_arguments_is_a_usage_error() -> None:
    assert acb.main([]) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
