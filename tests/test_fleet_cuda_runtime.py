"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import cuda_runtime
from lilbee.providers.fleet.cuda_runtime import (
    apply_cuda_runtime_env,
    assert_cuda_devices_usable,
    cuda_runtime_env,
)
from lilbee.providers.fleet.devices import FleetDevice


def _force_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")


def _have_ldd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.shutil, "which", lambda _name: "/usr/bin/ldd")


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
    wheel = Path("/wheel/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
    apply_cuda_runtime_env()
    # str(Path) keeps this host-agnostic (/ vs \ between Linux and Windows).
    assert cuda_runtime.os.environ["LD_LIBRARY_PATH"] == str(wheel)


def test_apply_cuda_runtime_env_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    # bb-ziks.15: plan_all_launches re-applies on every reload pass; the wheel
    # dirs must not accumulate duplicate copies in LD_LIBRARY_PATH.
    _force_linux(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    wheels = [Path("/wheel/a"), Path("/wheel/b")]
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: wheels)
    apply_cuda_runtime_env()
    apply_cuda_runtime_env()
    apply_cuda_runtime_env()
    expected = cuda_runtime.os.pathsep.join(str(w) for w in wheels)
    assert cuda_runtime.os.environ["LD_LIBRARY_PATH"] == expected


def test_cuda_runtime_env_keeps_unrelated_existing_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    wheel = Path("/wheel/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [wheel])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    result = cuda_runtime.cuda_runtime_env()["LD_LIBRARY_PATH"]
    assert result == cuda_runtime.os.pathsep.join([str(wheel), "/usr/local/lib"])


def test_apply_cuda_runtime_env_noop_when_no_wheels(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    apply_cuda_runtime_env()
    assert "LD_LIBRARY_PATH" not in cuda_runtime.os.environ


def test_assert_devices_noop_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_runtime.sys, "platform", "darwin")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("must not probe off Linux")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _boom)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_passes_when_a_device_enumerated(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    device = FleetDevice("CUDA", 0, "gpu", 1, 1)
    assert_cuda_devices_usable(Path("/bin/llama-server"), [device])  # no raise


def test_assert_devices_passes_for_non_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _NO_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_passes_when_no_nvidia_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_linux(monkeypatch)
    _ldd_returns(monkeypatch, _LINKS_CUDA)
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [])
    assert_cuda_devices_usable(Path("/bin/llama-server"), [])  # no raise


def test_assert_devices_raises_with_engine_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bb-3xnx failure: a CUDA build + an NVIDIA GPU, but the probe sees nothing.
    # Must hard-fail, surface the engine's real error, and list causes (not assert one).
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
    assert "nvidia-smi" in message
    assert "12.4" in message
    assert "CUDA_VISIBLE_DEVICES" in message  # causes listed, not one asserted


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
    dirs = [Path("/a/lib"), Path("/b/lib")]
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: dirs)
    # os.pathsep / str(Path) keep this host-agnostic (':' vs ';', / vs \).
    assert cuda_runtime_env() == {"LD_LIBRARY_PATH": os.pathsep.join(str(d) for d in dirs)}


def test_runtime_env_prepends_wheel_dirs_before_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_linux(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    lib = Path("/a/lib")
    monkeypatch.setattr(cuda_runtime, "_cuda_wheel_lib_dirs", lambda: [lib])
    assert cuda_runtime_env() == {"LD_LIBRARY_PATH": os.pathsep.join([str(lib), "/usr/lib"])}


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


def test_ldd_output_none_when_subprocess_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _have_ldd(monkeypatch)

    def _raise(*_a: object, **_k: object) -> None:
        raise OSError("not an ELF binary")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _raise)
    assert cuda_runtime._ldd_output(Path("/bin/llama-server"), {}) is None


def test_device_probe_diagnostic_returns_tail_when_no_error_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cuda_runtime.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout="CUDA0: NVIDIA L40 (45 GiB)\n", stderr=""),
    )
    out = cuda_runtime._device_probe_diagnostic(Path("/bin/llama-server"), {})
    assert "NVIDIA L40" in out  # no error marker -> falls through to the output tail


def test_device_probe_diagnostic_when_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cuda_runtime.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout="", stderr=""),
    )
    out = cuda_runtime._device_probe_diagnostic(Path("/bin/llama-server"), {})
    assert out == "(the engine's device probe printed nothing)"


def test_device_probe_diagnostic_when_probe_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> None:
        raise OSError("binary not found")

    monkeypatch.setattr(cuda_runtime.subprocess, "run", _raise)
    out = cuda_runtime._device_probe_diagnostic(Path("/bin/llama-server"), {})
    assert out == "(the engine's device probe could not be run)"
