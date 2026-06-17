"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.fleet import cuda_runtime
from lilbee.providers.fleet.cuda_runtime import cuda_runtime_env


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


def test_cuda_wheel_lib_dirs_collects_only_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    found = {
        "nvidia.cuda_runtime": Path("/r/lib"),
        "nvidia.cublas": None,
        "nvidia.cuda_nvrtc": Path("/n/lib"),
    }
    monkeypatch.setattr(cuda_runtime, "_wheel_lib_dir", lambda name: found[name])
    assert cuda_runtime._cuda_wheel_lib_dirs() == [Path("/r/lib"), Path("/n/lib")]
