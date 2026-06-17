"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import cuda_runtime
from lilbee.providers.fleet.cuda_runtime import (
    cuda_runtime_env,
    preflight_cuda_runtime,
)


def _force_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")


def test_runtime_env_empty_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/wheel/lib")])
    assert cuda_runtime_env() == {}


def test_runtime_env_empty_when_no_wheels(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert cuda_runtime_env() == {}


def test_runtime_env_sets_wheel_dirs_without_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_linux(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(
        cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/a/lib"), Path("/b/lib")]
    )
    assert cuda_runtime_env() == {"LD_LIBRARY_PATH": "/a/lib:/b/lib"}


def test_runtime_env_prepends_wheel_dirs_before_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_linux(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/a/lib")])
    assert cuda_runtime_env() == {"LD_LIBRARY_PATH": "/a/lib:/usr/lib"}


def test_wheel_lib_dir_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(cuda_runtime.importlib.util, "find_spec", lambda _name: spec)
    assert cuda_runtime._wheel_lib_dir("nvidia.cuda_runtime") == lib


def test_wheel_lib_dir_none_when_spec_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.importlib.util, "find_spec", lambda _name: None)
    assert cuda_runtime._wheel_lib_dir("nvidia.cuda_runtime") is None


def test_wheel_lib_dir_none_when_lib_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(cuda_runtime.importlib.util, "find_spec", lambda _name: spec)
    assert cuda_runtime._wheel_lib_dir("nvidia.cuda_runtime") is None


def test_wheel_lib_dir_handles_uninstalled_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> None:
        raise ModuleNotFoundError("No module named 'nvidia'")

    monkeypatch.setattr(cuda_runtime.importlib.util, "find_spec", _raise)
    assert cuda_runtime._wheel_lib_dir("nvidia.cuda_runtime") is None


def test_preflight_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("ldd must not run off Linux")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _boom)
    preflight_cuda_runtime(Path("/bin/llama-server"))  # no raise


def _have_ldd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.shutil, "which", lambda _name: "/usr/bin/ldd")


def test_preflight_passes_when_libs_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _have_ldd(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    clean = "\tlibcudart.so.12 => /usr/lib/libcudart.so.12 (0x00007f00)\n"
    monkeypatch.setattr(
        cuda_runtime.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=clean, stderr=""),
    )
    preflight_cuda_runtime(Path("/bin/llama-server"))  # no raise


def test_preflight_silent_when_ldd_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    monkeypatch.setattr(cuda_runtime.shutil, "which", lambda _name: None)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("ldd must not run when absent from PATH")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _boom)
    preflight_cuda_runtime(Path("/bin/llama-server"))  # no raise


def test_preflight_raises_actionable_error_when_libs_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_linux(monkeypatch)
    _have_ldd(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    missing = (
        "\tlibcudart.so.12 => not found\n"
        "\tlibcublas.so.12 => not found\n"
        "\tlibnvrtc.so.12 => not found\n"
    )
    monkeypatch.setattr(
        cuda_runtime.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=missing, stderr=""),
    )
    with pytest.raises(ProviderError) as exc:
        preflight_cuda_runtime(Path("/bin/llama-server"))
    message = str(exc.value)
    assert "libcudart.so.12" in message
    assert "nvidia-cuda-runtime-cu12" in message
    assert "nvidia-cublas-cu12" in message
    assert "nvidia-cuda-nvrtc-cu12" in message
    assert "LD_LIBRARY_PATH" in message


def test_preflight_silent_when_ldd_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _have_ldd(monkeypatch)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])

    def _raise(*_a: object, **_k: object) -> None:
        raise OSError("ldd failed on a static binary")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _raise)
    preflight_cuda_runtime(Path("/bin/llama-server"))  # no raise
