"""Tests for the ROCm-build guards against AMD hosts the engine cannot serve."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import engine_diagnostics, gpu_select, rocm_runtime
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.rocm_runtime import assert_rocm_devices_usable


def _root_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *prefixes: str) -> None:
    """Reroute rocm_runtime's absolute /dev and /sys lookups under *tmp_path*."""
    real_path = rocm_runtime.Path

    class _RootedPath(type(real_path())):  # type: ignore[misc]
        def __new__(cls, *args):
            first = str(args[0]) if args else ""
            if first.startswith(prefixes):
                return real_path(str(tmp_path) + first)
            return real_path(*args)

    monkeypatch.setattr(rocm_runtime, "Path", _RootedPath)


def _bundled_engine(tmp_path, shipped: tuple[str, ...]):
    """A bundled engine dir shaped like bundle_rocm_runtime.sh output: the binary
    with rocBLAS lazy Tensile masters for *shipped* beside it."""
    library = tmp_path / "rocblas" / "library"
    library.mkdir(parents=True)
    for gfx in shipped:
        (library / f"TensileLibrary_lazy_{gfx}.dat").write_text("")
    return tmp_path / "llama-server"


def _rocm_device(name: str = "AMD Instinct MI50/MI60") -> FleetDevice:
    return FleetDevice("ROCm", 0, name, 32 * 1024**3, 32 * 1024**3)


def test_rocm_build_enumerating_nothing_on_an_amd_host_fails_loud(monkeypatch, tmp_path) -> None:
    """ROCm's silent-failure class had no counterpart to the CUDA guard.

    A version mismatch, an unsupported gfx target or no access to /dev/kfd all
    end with the runtime loaded and zero devices, which used to fall quietly to
    CPU on a machine bought for its GPU.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(rocm_runtime, "_links_hip_runtime", lambda *_a: True)
    monkeypatch.setattr(rocm_runtime, "_amd_gpu_present", lambda: True)
    monkeypatch.setattr(rocm_runtime, "_amd_discrete_gpu_proven", lambda: True)

    with pytest.raises(ProviderError) as err:
        assert_rocm_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")
    assert "/dev/kfd" in str(err.value)


def test_rocm_build_is_silent_when_the_host_has_no_amd_gpu(monkeypatch, tmp_path) -> None:
    """A ROCm build on a machine without an AMD card is just a CPU host."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(rocm_runtime, "_links_hip_runtime", lambda *_a: True)
    monkeypatch.setattr(rocm_runtime, "_amd_gpu_present", lambda: False)

    assert_rocm_devices_usable(tmp_path / "llama-server", [], "")


def test_the_guard_is_a_noop_off_linux(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("the guard inspected the binary off Linux")

    monkeypatch.setattr(rocm_runtime, "_links_hip_runtime", _boom)
    assert_rocm_devices_usable(tmp_path / "llama-server", [], "")


class TestAmdPresenceReadsTheKernelNotRocmTooling:
    """The failure this detects is ROCm being installed wrong, so a check built
    on amd-smi or rocm-smi would report "no GPU" for the very case it exists to
    catch. Driven through a fake sysfs rather than by reading the source, so it
    tests what the function does and not how it is written."""

    def _fake_sysfs(self, monkeypatch, tmp_path, *, kfd: bool, vendors: list[str]) -> None:
        drm = tmp_path / "sys/class/drm"
        for i, vendor in enumerate(vendors):
            device = drm / f"card{i}" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text(f"{vendor}\n")
        (tmp_path / "dev").mkdir(exist_ok=True)
        if kfd:
            (tmp_path / "dev/kfd").write_text("")
        _root_paths(monkeypatch, tmp_path, "/dev/kfd", "/sys/class/drm")

    def test_an_amd_card_is_found_with_no_rocm_installed(self, monkeypatch, tmp_path) -> None:
        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x1002"])

        assert rocm_runtime._amd_gpu_present() is True

    def test_a_non_amd_card_is_not_an_amd_gpu(self, monkeypatch, tmp_path) -> None:
        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x10de"])

        assert rocm_runtime._amd_gpu_present() is False

    def test_no_kfd_means_the_kernel_driver_is_not_there(self, monkeypatch, tmp_path) -> None:
        self._fake_sysfs(monkeypatch, tmp_path, kfd=False, vendors=["0x1002"])

        assert rocm_runtime._amd_gpu_present() is False

    def test_it_never_shells_out(self, monkeypatch, tmp_path) -> None:
        self._fake_sysfs(monkeypatch, tmp_path, kfd=True, vendors=["0x1002"])

        def _boom(*_a: object, **_k: object) -> None:
            raise AssertionError("_amd_gpu_present shelled out; ROCm tooling may be broken")

        monkeypatch.setattr(engine_diagnostics.subprocess, "run", _boom)

        assert rocm_runtime._amd_gpu_present() is True

    def test_an_unreadable_sysfs_vendor_entry_is_skipped(self, monkeypatch, tmp_path) -> None:
        """A card whose vendor file cannot be read is passed over, not fatal: sysfs
        entries come and go as devices are bound and unbound."""
        drm = tmp_path / "sys/class/drm"
        unreadable = drm / "card0" / "device"
        unreadable.mkdir(parents=True)
        (unreadable / "vendor").mkdir()  # a directory where a file is expected -> OSError
        (tmp_path / "dev").mkdir()
        (tmp_path / "dev/kfd").write_text("")
        _root_paths(monkeypatch, tmp_path, "/dev/kfd", "/sys/class/drm")

        assert rocm_runtime._amd_gpu_present() is False


class TestAnApuLaptopIsNotRefusedTheEngine:
    """Every check that finds an AMD GPU also finds an APU.

    amdgpu exposes /dev/kfd for integrated parts and the iGPU carries vendor
    0x1002, but AMD's population of GPUs ROCm does not support is large, and an
    unsupported gfx target is the normal case for an APU. Refusing to start
    there is worse than the slow CPU fallback the guard exists to catch: it
    breaks a laptop that worked.
    """

    def _rocm_host_with_no_devices(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(rocm_runtime, "_links_hip_runtime", lambda *_a: True)
        monkeypatch.setattr(rocm_runtime, "_amd_gpu_present", lambda: True)

    def test_an_apu_only_host_warns_and_starts(self, monkeypatch, tmp_path, caplog) -> None:
        self._rocm_host_with_no_devices(monkeypatch)
        monkeypatch.setattr(rocm_runtime, "_amd_discrete_gpu_proven", lambda: False)

        with caplog.at_level("WARNING", logger=rocm_runtime.__name__):
            assert_rocm_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")

        assert "rocminfo" in caplog.text

    def test_the_apu_warning_names_the_shipped_targets_when_known(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """With a bundled engine the shipped set is a readable fact, so the
        warning states it instead of guessing at an unsupported gfx target."""
        self._rocm_host_with_no_devices(monkeypatch)
        monkeypatch.setattr(rocm_runtime, "_amd_discrete_gpu_proven", lambda: False)
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx1103"})
        binary = _bundled_engine(tmp_path, ("gfx1030", "gfx1100"))

        with caplog.at_level("WARNING", logger=rocm_runtime.__name__):
            assert_rocm_devices_usable(binary, [], "rocBLAS error\n")

        assert "gfx1030" in caplog.text
        assert "gfx1103" in caplog.text

    def test_an_unreachable_loader_is_not_read_as_proof_of_a_card(
        self, monkeypatch, tmp_path
    ) -> None:
        """No Vulkan loader means no evidence either way, which must not fail loud."""
        self._rocm_host_with_no_devices(monkeypatch)
        monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: None)

        assert_rocm_devices_usable(tmp_path / "llama-server", [], "rocBLAS error\n")

    def test_a_discrete_amd_card_is_proven_from_the_loader(self, monkeypatch) -> None:
        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(
                    0, gpu_select.VkDeviceType.DISCRETE_GPU, "RX 7900 XTX", 0x1002, 0
                )
            ],
        )

        assert rocm_runtime._amd_discrete_gpu_proven() is True

    def test_an_integrated_amd_adapter_is_not_a_discrete_card(self, monkeypatch) -> None:
        monkeypatch.setattr(
            gpu_select,
            "_enumerate_vulkan_devices",
            lambda: [
                gpu_select.VulkanDevice(
                    0, gpu_select.VkDeviceType.INTEGRATED_GPU, "Radeon Graphics", 0x1002, 0
                )
            ],
        )

        assert rocm_runtime._amd_discrete_gpu_proven() is False


def test_hip_link_check_reads_the_sonames_the_binary_lists(monkeypatch, tmp_path) -> None:
    """A ROCm build is identified by its linkage, resolved or not, so a stub
    engine on a host with no ROCm still gets the AMD treatment."""
    monkeypatch.setattr(
        engine_diagnostics, "ldd_output", lambda *_a: "libamdhip64.so.6 => not found\n"
    )
    assert rocm_runtime._links_hip_runtime(tmp_path / "llama-server", {}) is True

    monkeypatch.setattr(engine_diagnostics, "ldd_output", lambda *_a: "libc.so.6 => /lib/x\n")
    assert rocm_runtime._links_hip_runtime(tmp_path / "llama-server", {}) is False


class TestEnumeratedCardWithoutShippedKernelsIsRefused:
    """A card the engine enumerates but the bundle has no rocBLAS kernels for does
    not fall back to CPU: rocBLAS aborts the engine at the first batched GEMM. The
    supported set is read from the shipped Tensile masters, not a constant, so it
    cannot drift from what is actually in the wheel."""

    def _linux(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)

    def test_an_mi50_against_a_gfx906_less_bundle_fails_loud(self, monkeypatch, tmp_path) -> None:
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx908", "gfx90a", "gfx1030"))
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx906"})

        with pytest.raises(ProviderError) as err:
            assert_rocm_devices_usable(binary, [_rocm_device()], "")
        message = str(err.value)
        assert "gfx906" in message
        assert "gfx1030" in message
        assert "HSA_OVERRIDE_GFX_VERSION" in message

    def test_a_covered_card_passes(self, monkeypatch, tmp_path) -> None:
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx1030"})

        assert_rocm_devices_usable(binary, [_rocm_device("Radeon RX 6800")], "")

    def test_a_mixed_host_warns_about_the_uncovered_card_only(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """One supported card beside an unsupported iGPU must not stop the engine."""
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1100",))
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx1100", "gfx1036"})

        with caplog.at_level("WARNING", logger=rocm_runtime.__name__):
            assert_rocm_devices_usable(binary, [_rocm_device()], "")
        assert "gfx1036" in caplog.text

    def test_an_engine_without_a_bundle_makes_no_claim(self, monkeypatch, tmp_path) -> None:
        """A system-ROCm engine has no shipped kernel list to check against."""
        self._linux(monkeypatch)
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx906"})

        assert_rocm_devices_usable(tmp_path / "llama-server", [_rocm_device()], "")

    def test_an_unreadable_kfd_topology_makes_no_claim(self, monkeypatch, tmp_path) -> None:
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: set())

        assert_rocm_devices_usable(binary, [_rocm_device()], "")

    def test_a_covering_hsa_override_is_respected(self, monkeypatch, tmp_path) -> None:
        """gfx1031 with HSA_OVERRIDE_GFX_VERSION=10.3.0 runs the gfx1030 kernels."""
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx1031"})

        assert_rocm_devices_usable(binary, [_rocm_device()], "")

    def test_an_override_naming_an_unshipped_target_warns_not_raises(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """The user overrode explicitly; respect it, but say what will happen."""
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.0.6")
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx906"})

        with caplog.at_level("WARNING", logger=rocm_runtime.__name__):
            assert_rocm_devices_usable(binary, [_rocm_device()], "")
        assert "gfx906" in caplog.text

    def test_a_malformed_override_does_not_disable_the_guard(
        self, monkeypatch, tmp_path
    ) -> None:
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "not-a-version")
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx906"})

        with pytest.raises(ProviderError):
            assert_rocm_devices_usable(binary, [_rocm_device()], "")

    def test_a_non_amd_device_list_is_left_alone(self, monkeypatch, tmp_path) -> None:
        self._linux(monkeypatch)
        binary = _bundled_engine(tmp_path, ("gfx1030",))
        monkeypatch.setattr(rocm_runtime, "_host_amd_gfx_targets", lambda: {"gfx906"})
        vulkan = FleetDevice("Vulkan", 0, "RX 6800", 16 * 1024**3, 16 * 1024**3)

        assert_rocm_devices_usable(binary, [vulkan], "")


class TestGfxTargetsAreReadFromTheDriverAndTheBundle:
    def test_gfx_names_format_minor_and_step_as_hex(self) -> None:
        assert rocm_runtime._gfx_name(90006) == "gfx906"
        assert rocm_runtime._gfx_name(90010) == "gfx90a"
        assert rocm_runtime._gfx_name(90012) == "gfx90c"
        assert rocm_runtime._gfx_name(100300) == "gfx1030"
        assert rocm_runtime._gfx_name(110001) == "gfx1101"

    def test_bundled_targets_come_from_the_lazy_tensile_masters(self, tmp_path) -> None:
        binary = _bundled_engine(tmp_path, ("gfx906", "gfx1030"))
        (tmp_path / "rocblas/library/TensileLibrary_gfx942.dat").write_text("")

        assert rocm_runtime._bundled_rocblas_gfx_targets(binary) == {"gfx906", "gfx1030"}

    def test_no_bundle_directory_is_none_not_empty(self, tmp_path) -> None:
        """None is "no claim"; an empty set would read as "supports nothing"."""
        assert rocm_runtime._bundled_rocblas_gfx_targets(tmp_path / "llama-server") is None

    def test_host_targets_come_from_kfd_topology_skipping_cpu_nodes(
        self, monkeypatch, tmp_path
    ) -> None:
        nodes = tmp_path / "sys/class/kfd/kfd/topology/nodes"
        (nodes / "0").mkdir(parents=True)
        (nodes / "0/properties").write_text("cpu_cores_count 16\ngfx_target_version 0\n")
        (nodes / "1").mkdir()
        (nodes / "1/properties").write_text("simd_count 240\ngfx_target_version 90006\n")
        _root_paths(monkeypatch, tmp_path, "/sys/class/kfd")

        assert rocm_runtime._host_amd_gfx_targets() == {"gfx906"}

    def test_a_missing_kfd_topology_is_an_empty_set(self, monkeypatch, tmp_path) -> None:
        _root_paths(monkeypatch, tmp_path, "/sys/class/kfd")

        assert rocm_runtime._host_amd_gfx_targets() == set()

    def test_an_unreadable_node_is_skipped_not_fatal(self, monkeypatch, tmp_path) -> None:
        """Topology entries come and go as devices bind and unbind."""
        nodes = tmp_path / "sys/class/kfd/kfd/topology/nodes"
        (nodes / "0/properties").mkdir(parents=True)  # a directory where a file is expected
        (nodes / "1").mkdir()
        (nodes / "1/properties").write_text("gfx_target_version 90006\n")
        _root_paths(monkeypatch, tmp_path, "/sys/class/kfd")

        assert rocm_runtime._host_amd_gfx_targets() == {"gfx906"}
