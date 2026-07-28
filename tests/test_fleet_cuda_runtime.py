"""Tests for CUDA-runtime wiring that lets the engine start on driver-only images."""

from __future__ import annotations

import os
import sys
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
    # plan_all_launches re-applies on every reload pass; the wheel
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
    assert "MIG" in message  # the one host shape that reliably produces this symptom
    assert "nvidia-cuda-runtime" in message  # the wheels to match, major-agnostic
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


def test_device_probe_diagnostic_returns_tail_when_no_error_line() -> None:
    out = cuda_runtime._device_probe_diagnostic("CUDA0: NVIDIA L40 (45 GiB)\n")
    assert "NVIDIA L40" in out  # no error marker -> falls through to the output tail


def test_device_probe_diagnostic_picks_the_cuda_error_line() -> None:
    output = "loading backends\nggml_cuda_init: CUDA error: unknown error\ntrailing noise\n"
    assert (
        cuda_runtime._device_probe_diagnostic(output) == "ggml_cuda_init: CUDA error: unknown error"
    )


def test_device_probe_diagnostic_when_no_output() -> None:
    assert (
        cuda_runtime._device_probe_diagnostic("") == "(the engine's device probe printed nothing)"
    )


def test_rocm_build_enumerating_nothing_on_an_amd_host_fails_loud(monkeypatch, tmp_path) -> None:
    """ROCm's silent-failure class had no counterpart to the CUDA guard.

    A version mismatch, an unsupported gfx target or no access to /dev/kfd all
    end with the runtime loaded and zero devices, which used to fall quietly to
    CPU on a machine bought for its GPU.
    """
    from lilbee.providers.base import ProviderError
    from lilbee.providers.fleet import cuda_runtime

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: False)
    monkeypatch.setattr(cuda_runtime, "_links_hip_runtime", lambda *_a: True)
    monkeypatch.setattr(cuda_runtime, "_amd_gpu_present", lambda: True)
    monkeypatch.setattr(cuda_runtime, "_amd_discrete_gpu_proven", lambda: True)

    with pytest.raises(ProviderError) as err:
        cuda_runtime.assert_gpu_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")
    assert "/dev/kfd" in str(err.value)


def test_rocm_build_is_silent_when_the_host_has_no_amd_gpu(monkeypatch, tmp_path) -> None:
    """A ROCm build on a machine without an AMD card is just a CPU host."""
    from lilbee.providers.fleet import cuda_runtime

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: False)
    monkeypatch.setattr(cuda_runtime, "_links_hip_runtime", lambda *_a: True)
    monkeypatch.setattr(cuda_runtime, "_amd_gpu_present", lambda: False)

    cuda_runtime.assert_gpu_devices_usable(tmp_path / "llama-server", [], "")


class TestAmdPresenceReadsTheKernelNotRocmTooling:
    """The failure this detects is ROCm being installed wrong, so a check built
    on amd-smi or rocm-smi would report "no GPU" for the very case it exists to
    catch. Driven through a fake sysfs rather than by reading the source, so it
    tests what the function does and not how it is written."""

    def _fake_sysfs(self, monkeypatch, tmp_path, *, kfd: bool, vendors: list[str]) -> None:
        from lilbee.providers.fleet import cuda_runtime

        drm = tmp_path / "sys/class/drm"
        for i, vendor in enumerate(vendors):
            device = drm / f"card{i}" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text(f"{vendor}\n")
        (tmp_path / "dev").mkdir(exist_ok=True)
        if kfd:
            (tmp_path / "dev/kfd").write_text("")

        real_path = cuda_runtime.Path

        class _RootedPath(type(real_path())):  # type: ignore[misc]
            def __new__(cls, *args):
                first = str(args[0]) if args else ""
                if first.startswith(("/dev/kfd", "/sys/class/drm")):
                    return real_path(str(tmp_path) + first)
                return real_path(*args)

        monkeypatch.setattr(cuda_runtime, "Path", _RootedPath)

    def test_an_amd_card_is_found_with_no_rocm_installed(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import cuda_runtime

        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x1002"])

        assert cuda_runtime._amd_gpu_present() is True

    def test_a_non_amd_card_is_not_an_amd_gpu(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import cuda_runtime

        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x10de"])

        assert cuda_runtime._amd_gpu_present() is False

    def test_no_kfd_means_the_kernel_driver_is_not_there(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import cuda_runtime

        self._fake_sysfs(monkeypatch, tmp_path, kfd=False, vendors=["0x1002"])

        assert cuda_runtime._amd_gpu_present() is False

    def test_it_never_shells_out(self, monkeypatch, tmp_path) -> None:
        from lilbee.providers.fleet import cuda_runtime

        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x1002"])

        def _boom(*_a: object, **_k: object) -> None:
            raise AssertionError("_amd_gpu_present shelled out; ROCm tooling may be broken")

        monkeypatch.setattr(cuda_runtime.subprocess, "run", _boom)

        assert cuda_runtime._amd_gpu_present() is True


class TestAnApuLaptopIsNotRefusedTheEngine:
    """Every check that finds an AMD GPU also finds an APU.

    amdgpu exposes /dev/kfd for integrated parts and the iGPU carries vendor
    0x1002, but AMD's population of GPUs ROCm does not support is large, and an
    unsupported gfx target is the normal case for an APU. Refusing to start
    there is worse than the slow CPU fallback the guard exists to catch: it
    breaks a laptop that worked.
    """

    def _rocm_host_with_no_devices(self, monkeypatch) -> None:
        from lilbee.providers.fleet import cuda_runtime

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: False)
        monkeypatch.setattr(cuda_runtime, "_links_hip_runtime", lambda *_a: True)
        monkeypatch.setattr(cuda_runtime, "_amd_gpu_present", lambda: True)

    def test_an_apu_only_host_warns_and_starts(self, monkeypatch, tmp_path, caplog) -> None:
        from lilbee.providers.fleet import cuda_runtime

        self._rocm_host_with_no_devices(monkeypatch)
        monkeypatch.setattr(cuda_runtime, "_amd_discrete_gpu_proven", lambda: False)

        with caplog.at_level("WARNING", logger=cuda_runtime.__name__):
            cuda_runtime.assert_gpu_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")

        assert "rocminfo" in caplog.text

    def test_an_unreachable_loader_is_not_read_as_proof_of_a_card(
        self, monkeypatch, tmp_path
    ) -> None:
        """No Vulkan loader means no evidence either way, which must not fail loud."""
        from lilbee.providers.fleet import cuda_runtime, gpu_select

        self._rocm_host_with_no_devices(monkeypatch)
        monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: None)

        cuda_runtime.assert_gpu_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")

    def test_a_discrete_amd_card_is_proven_from_the_loader(self, monkeypatch) -> None:
        from lilbee.providers.fleet import cuda_runtime, gpu_select

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(
                    0, gpu_select.VkDeviceType.DISCRETE_GPU, "RX 7900 XTX", 0x1002, 0
                )
            ],
        )

        assert cuda_runtime._amd_discrete_gpu_proven() is True

    def test_an_integrated_amd_adapter_is_not_a_discrete_card(self, monkeypatch) -> None:
        from lilbee.providers.fleet import cuda_runtime, gpu_select

        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(
                    0, gpu_select.VkDeviceType.INTEGRATED_GPU, "Radeon Graphics", 0x1002, 0
                )
            ],
        )

        assert cuda_runtime._amd_discrete_gpu_proven() is False


def test_hip_link_check_is_false_when_ldd_cannot_run(monkeypatch, tmp_path) -> None:
    """No linkage evidence is not evidence of linkage: a binary whose libraries
    cannot be listed must not be accused of linking ROCm."""
    from lilbee.providers.fleet import cuda_runtime

    monkeypatch.setattr(cuda_runtime, "_ldd_output", lambda *_a: None)

    assert cuda_runtime._links_hip_runtime(tmp_path / "llama-server", {}) is False


def test_an_unreadable_sysfs_vendor_entry_is_skipped(monkeypatch, tmp_path) -> None:
    """A card whose vendor file cannot be read is passed over, not fatal: sysfs
    entries come and go as devices are bound and unbound."""
    from lilbee.providers.fleet import cuda_runtime

    drm = tmp_path / "sys/class/drm"
    unreadable = drm / "card0" / "device"
    unreadable.mkdir(parents=True)
    (unreadable / "vendor").mkdir()  # a directory where a file is expected -> OSError
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev/kfd").write_text("")

    real_path = cuda_runtime.Path

    class _RootedPath(type(real_path())):  # type: ignore[misc]
        def __new__(cls, *args):
            first = str(args[0]) if args else ""
            if first.startswith(("/dev/kfd", "/sys/class/drm")):
                return real_path(str(tmp_path) + first)
            return real_path(*args)

    monkeypatch.setattr(cuda_runtime, "Path", _RootedPath)

    assert cuda_runtime._amd_gpu_present() is False


def test_hip_link_check_reads_the_sonames_the_binary_lists(monkeypatch, tmp_path) -> None:
    """A ROCm build is identified by its linkage, resolved or not, so a stub
    engine on a host with no ROCm still gets the AMD treatment."""
    from lilbee.providers.fleet import cuda_runtime

    monkeypatch.setattr(cuda_runtime, "_ldd_output", lambda *_a: "libamdhip64.so.6 => not found\n")
    assert cuda_runtime._links_hip_runtime(tmp_path / "llama-server", {}) is True

    monkeypatch.setattr(cuda_runtime, "_ldd_output", lambda *_a: "libc.so.6 => /lib/libc.so.6\n")
    assert cuda_runtime._links_hip_runtime(tmp_path / "llama-server", {}) is False
