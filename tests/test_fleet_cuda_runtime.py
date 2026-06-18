"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import cuda_runtime
from lilbee.providers.fleet.cuda_runtime import (
    apply_cuda_runtime_env,
    assert_cuda_devices_usable,
    cuda_runtime_env,
    preflight_cuda_runtime,
)
from lilbee.providers.fleet.devices import FleetDevice


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


def _ldd_returns(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    _have_ldd(monkeypatch)
    monkeypatch.setattr(
        cuda_runtime.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout=stdout, stderr=""),
    )


_LINKS_CUDA = "\tlibcudart.so.12 => /usr/lib/libcudart.so.12 (0x00007f00)\n"
_NO_CUDA = "\tlibstdc++.so.6 => /usr/lib/libstdc++.so.6 (0x00007f00)\n"


def test_links_cuda_runtime_true_when_soname_present(monkeypatch: pytest.MonkeyPatch) -> None:
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is True


def test_links_cuda_runtime_true_when_soname_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CUDA build whose runtime is missing still names the soname -- it is a CUDA build.
    _ldd_returns(monkeypatch, "\tlibcudart.so.12 => not found\n")
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is True


def test_links_cuda_runtime_false_for_non_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _ldd_returns(monkeypatch, _NO_CUDA)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is False


def test_links_cuda_runtime_false_when_ldd_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.shutil, "which", lambda _name: None)
    assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is False


def test_apply_cuda_runtime_env_updates_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [Path("/wheel/lib")])
    apply_cuda_runtime_env()
    assert cuda_runtime.os.environ["LD_LIBRARY_PATH"] == "/wheel/lib"


def test_apply_cuda_runtime_env_noop_when_no_wheels(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    apply_cuda_runtime_env()
    assert "LD_LIBRARY_PATH" not in cuda_runtime.os.environ


def _cuda_gpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Linux box: a CUDA-linked binary and an NVIDIA GPU the driver can see."""
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])


def test_assert_devices_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not probe off Linux")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _boom)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_passes_when_a_device_enumerated(monkeypatch: pytest.MonkeyPatch) -> None:
    _cuda_gpu_host(monkeypatch)
    device = FleetDevice("CUDA", 0, "gpu", 1, 1)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [device])  # no raise


def test_assert_devices_passes_for_non_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Vulkan/CPU build on an NVIDIA box legitimately falls back; never hard-fails.
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_passes_when_no_nvidia_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # A CUDA build on a CPU-only host should fall back to CPU, not error.
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_raises_when_cuda_build_sees_no_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cuda_gpu_host(monkeypatch)
    with pytest.raises(ProviderError) as exc:
        assert_cuda_devices_usable(Path("/bin/llama-server"), [])
    message = str(exc.value)
    assert "no CUDA-capable device" in message
    assert "nvidia-smi" in message
    assert "12.4" in message  # names the matching runtime to pin (cu124 build)


def test_assert_devices_surfaces_engine_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    # The error reports what the engine actually said + lists causes, rather than
    # asserting a single guessed cause.
    _force_linux(monkeypatch)
    _have_ldd(monkeypatch)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    cuda_err = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected"

    def _run(cmd: list[str], *_a: object, **_k: object) -> SimpleNamespace:
        if "--list-devices" in cmd:
            return SimpleNamespace(stdout="", stderr=cuda_err + "\n")
        return SimpleNamespace(stdout=_LINKS_CUDA, stderr="")  # ldd resolves the soname

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _run)
    with pytest.raises(ProviderError) as exc:
        assert_cuda_devices_usable(Path("/bin/llama-server"), [])
    message = str(exc.value)
    assert "failed to initialize CUDA" in message  # the engine's own diagnostic, surfaced
    assert "Likely causes" in message
    assert "CUDA_VISIBLE_DEVICES" in message  # causes listed, not one asserted
