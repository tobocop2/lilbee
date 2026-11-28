"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import cuda_runtime, engine_diagnostics
from lilbee.providers.fleet.cuda_runtime import (
    assert_cuda_devices_usable,
    cuda_runtime_env,
)
from lilbee.providers.fleet.devices import FleetDevice


def _force_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")


def _have_ldd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: "/usr/bin/ldd")


def _ldd_returns(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    _have_ldd(monkeypatch)
    monkeypatch.setattr(
        engine_diagnostics.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=stdout, stderr=""),
    )


_LINKS_CUDA = "\tlibcudart.so.12 => /usr/lib/libcudart.so.12 (0x00007f00)\n"
_NO_CUDA = "\tlibstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f00)\n"


def test_links_cuda_runtime_true_when_soname_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is True


def test_links_cuda_runtime_true_when_soname_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    _ldd_returns(monkeypatch, "\tlibcudart.so.12 => not found\n")
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is True


def test_links_cuda_runtime_false_for_non_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is False


def test_links_cuda_runtime_false_when_ldd_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: None)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is False


def test_cuda_runtime_env_keeps_unrelated_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    binary = _engine_dir(tmp_path)
    wheel = Path("/wheel/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    result = cuda_runtime.cuda_runtime_env(binary)["LD_LIBRARY_PATH"]
    assert result == cuda_runtime.os.pathsep.join([str(tmp_path), str(wheel), "/usr/local/lib"])


def test_assert_devices_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not probe off Linux")

    monkeypatch.setattr(engine_diagnostics.subprocess, "run", _boom)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [], "")  # no raise


def test_assert_devices_passes_when_a_device_enumerated(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    device = FleetDevice("CUDA", 0, "gpu", 1, 1)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [device], "")  # no raise


def test_assert_devices_passes_for_non_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [], "")  # no raise


def test_assert_devices_passes_when_no_nvidia_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [], "")  # no raise


def test_assert_devices_raises_with_engine_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CUDA build plus an NVIDIA GPU, but the probe sees nothing. Must hard-fail,
    # surface the engine's real error, and list causes rather than assert one.
    _force_linux(monkeypatch)
    _have_ldd(monkeypatch)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    cuda_err = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected"
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    with pytest.raises(ProviderError) as exc:
        assert_cuda_devices_usable(Path("/bin/llama-server"), [], cuda_err + "\n")
    message = str(exc.value)
    assert "failed to initialize CUDA" in message  # the engine's own diagnostic, surfaced
    assert "nvidia-smi" in message
    assert "this computer has an NVIDIA GPU" in message
    assert "MIG" in message  # the one host shape that reliably produces this symptom
    assert "nvidia-cuda-runtime" in message  # the wheels to match, major-agnostic
    assert "CUDA_VISIBLE_DEVICES" in message  # causes listed, not one asserted


def test_runtime_env_empty_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/wheel/lib")])
    assert cuda_runtime_env(Path("/b/llama-server")) == {}


def test_runtime_env_is_only_the_binary_folder_when_no_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    binary = _engine_dir(tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert cuda_runtime_env(binary) == {"LD_LIBRARY_PATH": str(tmp_path)}


def test_runtime_env_sets_wheel_dirs_without_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    binary = _engine_dir(tmp_path)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    dirs = [Path("/a/lib"), Path("/b/lib")]
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: dirs)
    # os.pathsep / str(Path) keep this host-agnostic (':' vs ';', / vs \).
    assert cuda_runtime_env(binary) == {
        "LD_LIBRARY_PATH": os.pathsep.join([str(tmp_path), *(str(d) for d in dirs)])
    }


def test_runtime_env_puts_wheel_dirs_before_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    binary = _engine_dir(tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    lib = Path("/a/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [lib])
    assert cuda_runtime_env(binary) == {
        "LD_LIBRARY_PATH": os.pathsep.join([str(tmp_path), str(lib), "/usr/lib"])
    }


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


def _engine_dir(tmp_path: Path, *names: str) -> Path:
    """An engine binary in *tmp_path* with the named files beside it."""
    for name in names:
        (tmp_path / name).write_bytes(b"")
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"")
    return binary


def _no_ldd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_diagnostics.shutil, "which", lambda _name: None)


def test_runtime_cuda_module_beside_engine_marks_a_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    binary = _engine_dir(tmp_path, "libggml-cuda.so")
    assert cuda_runtime._is_cuda_build(binary, {}) is True


def test_versioned_runtime_cuda_module_marks_a_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    binary = _engine_dir(tmp_path, "libggml-cuda.so.0.15.1")
    assert cuda_runtime._is_cuda_build(binary, {}) is True


def test_names_that_only_resemble_the_cuda_module_do_not_mark_a_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    binary = _engine_dir(
        tmp_path, "libggml-cuda.txt", "libggml-cuda.so.bak", "libggml-vulkan.so", "libggml-cpu.so"
    )
    assert cuda_runtime._is_cuda_build(binary, {}) is False


def test_cuda_module_in_parent_directory_is_not_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    (tmp_path / "libggml-cuda.so").write_bytes(b"")
    (tmp_path / "bin").mkdir()
    binary = _engine_dir(tmp_path / "bin")
    assert cuda_runtime._is_cuda_build(binary, {}) is False


def test_cuda_module_marks_a_cuda_build_without_ldd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_ldd(monkeypatch)
    binary = _engine_dir(tmp_path, "libggml-cuda.so")
    assert cuda_runtime._is_cuda_build(binary, {}) is True


def test_unreadable_engine_directory_falls_back_to_ldd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    binary = tmp_path / "missing" / "llama-server"
    assert cuda_runtime._is_cuda_build(binary, {}) is True


def test_assert_raises_for_runtime_cuda_module_with_no_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    binary = _engine_dir(tmp_path, "libggml-cuda.so")
    cuda_err = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected"
    with pytest.raises(ProviderError, match="failed to initialize CUDA"):
        assert_cuda_devices_usable(binary, [], cuda_err + "\n")


def test_cuda_runtime_env_adds_wheel_dirs_for_runtime_cuda_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    wheel = Path("/wheel/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
    binary = _engine_dir(tmp_path, "libggml-cuda.so")
    assert cuda_runtime_env(binary) == {
        "LD_LIBRARY_PATH": os.pathsep.join([str(tmp_path), str(wheel)])
    }


def test_cuda_runtime_env_skips_wheel_dirs_when_runtime_ships_beside_a_cuda_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/wheel/lib")])
    binary = _engine_dir(tmp_path, "libggml-cuda.so", "libcudart.so.12")
    assert cuda_runtime_env(binary) == {"LD_LIBRARY_PATH": str(tmp_path)}


def _record_ldd(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, str]]]:
    """Replace ``ldd_output`` with a recorder that reports a non-CUDA link list."""
    calls: list[tuple[Path, dict[str, str]]] = []

    def fake(binary: Path, env: dict[str, str]) -> str:
        calls.append((binary, env))
        return _NO_CUDA

    monkeypatch.setattr(cuda_runtime, "ldd_output", fake)
    return calls


def test_is_cuda_build_asks_ldd_about_the_binary_under_the_given_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_ldd(monkeypatch)
    binary = _engine_dir(tmp_path)
    env = {"LD_LIBRARY_PATH": "/engine/lib"}
    assert cuda_runtime._is_cuda_build(binary, env) is False
    assert calls == [(binary, env)]


def test_cuda_runtime_env_asks_ldd_under_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux(monkeypatch)
    calls = _record_ldd(monkeypatch)
    monkeypatch.setenv("LILBEE_TEST_MARKER", "present")
    binary = _engine_dir(tmp_path)
    cuda_runtime_env(binary)
    assert len(calls) == 1
    asked_binary, asked_env = calls[0]
    assert asked_binary == binary
    assert asked_env == dict(os.environ)
    assert asked_env["LILBEE_TEST_MARKER"] == "present"


def test_assert_asks_ldd_under_the_environment_with_the_engine_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_linux(monkeypatch)
    calls = _record_ldd(monkeypatch)
    monkeypatch.setenv("LILBEE_TEST_MARKER", "present")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    binary = _engine_dir(tmp_path)
    assert_cuda_devices_usable(binary, [], "")
    # The first question comes from cuda_runtime_env; the last one is the check itself.
    asked_binary, asked_env = calls[-1]
    assert asked_binary == binary
    assert asked_env["LD_LIBRARY_PATH"] == str(tmp_path)
    assert asked_env["LILBEE_TEST_MARKER"] == "present"
