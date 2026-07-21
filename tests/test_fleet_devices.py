"""Tests for binary-native GPU enumeration and per-backend device pinning."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import devices as dev_mod
from lilbee.providers.fleet.devices import (
    FleetDevice,
    probe_devices,
    visible_env,
)

_CUDA_LISTING = """\
Available devices:
  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)
  CUDA1: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)
"""
_MIB = 1024 * 1024


def _fake_run(stdout: str):
    def _run(*_a: object, **_k: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    return _run


_TOPO_NVLINK = """\
\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV4\t0-31\t0
GPU1\tNV4\t X \t0-31\t0
"""
_TOPO_PCIE = """\
\tGPU0\tGPU1\tCPU Affinity
GPU0\t X \tPHB\t0-31
GPU1\tPHB\t X \t0-31
"""


# Captured from a live 3xH100 SXM pod: nvidia-smi underlines the header via SGR
# escapes even when stdout is not a tty, and adds NIC columns plus a legend.
_TOPO_REAL_H100 = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV18\tNV18\tSYS\t0,2,4,6,8,10\t0\t\tN/A\n"
    "GPU1\tNV18\t X \tNV18\tNODE\t1,3,5,7,9,11\t1\t\tN/A\n"
    "GPU2\tNV18\tNV18\t X \tNODE\t1,3,5,7,9,11\t1\t\tN/A\n"
    "NIC0\tSYS\tNODE\tNODE\t X \t\t\t\t\n"
    "\nLegend:\n\n  X    = Self\n  SYS  = Connection traversing PCIe\n"
)


class TestNvlinkTopology:
    def test_real_h100_output_parses_despite_ansi_escapes(self) -> None:
        # The SGR-wrapped header must still yield the GPU columns; without the
        # strip, no pairs parse and an NVLinked host is mis-flagged as PCIe-only.
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_REAL_H100)
        assert gpu_rows == {0, 1, 2}
        assert pairs == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}

    def test_real_h100_host_has_nvlink(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_REAL_H100))
        assert dev_mod.host_lacks_nvlink() is False

    def test_parse_finds_nvlink_pair(self) -> None:
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_NVLINK)
        assert gpu_rows == {0, 1}
        assert pairs == {frozenset({0, 1})}

    def test_parse_pcie_only_has_no_pairs(self) -> None:
        gpu_rows, pairs = dev_mod._parse_topo_matrix(_TOPO_PCIE)
        assert gpu_rows == {0, 1}
        assert pairs == set()

    def test_lacks_nvlink_true_for_pcie_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_PCIE))
        assert dev_mod.host_lacks_nvlink() is True

    def test_lacks_nvlink_false_when_linked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(_TOPO_NVLINK))
        assert dev_mod.host_lacks_nvlink() is False

    def test_unparseable_topo_makes_no_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Garbage output parses zero GPU rows: stay silent rather than mis-warn.
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run("some unrelated output\n"))
        assert dev_mod.host_lacks_nvlink() is False

    def test_single_gpu_host_is_not_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        single = "\tGPU0\tCPU Affinity\nGPU0\t X \t0-31\n"
        monkeypatch.setattr(dev_mod.subprocess, "run", _fake_run(single))
        assert dev_mod.host_lacks_nvlink() is False

    def test_probe_failure_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a: object, **_k: object) -> object:
            raise OSError("no nvidia-smi")

        monkeypatch.setattr(dev_mod.subprocess, "run", _boom)
        assert dev_mod.host_lacks_nvlink() is False


def _fake_listing(monkeypatch: pytest.MonkeyPatch, output: str, returncode: int = 0) -> None:
    monkeypatch.setattr(
        dev_mod, "_run_list_devices", lambda _binary, _timeout: (output, returncode)
    )


def test_probe_parses_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_listing(monkeypatch, _CUDA_LISTING)
    probe = probe_devices(Path("/bin/llama-server"))
    assert probe.devices == [
        FleetDevice("CUDA", 0, "NVIDIA GeForce RTX 3090", 24268 * _MIB, 23500 * _MIB),
        FleetDevice("CUDA", 1, "NVIDIA GeForce RTX 4090", 24564 * _MIB, 24000 * _MIB),
    ]
    assert probe.output == _CUDA_LISTING


def test_probe_drops_cpu_and_keeps_gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (
        "  CUDA0: NVIDIA (24268 MiB, 23000 MiB free)\n"
        "  CPU0: some cpu (64000 MiB)\n"
        "  Vulkan0: NVIDIA (24268 MiB, 23000 MiB free)\n"
    )
    _fake_listing(monkeypatch, listing)
    devices = probe_devices(Path("/bin/llama-server")).devices
    # CUDA outranks Vulkan; CPU dropped entirely.
    assert [d.backend for d in devices] == ["CUDA"]


def test_probe_defaults_free_to_total_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_listing(monkeypatch, "  Vulkan0: AMD (16000 MiB)\n")
    (device,) = probe_devices(Path("/bin/llama-server")).devices
    assert device.free_bytes == device.total_bytes == 16000 * _MIB


def test_probe_returns_a_single_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if a listing somehow names two GPU backends, pin exactly one (no mixed
    # index spaces). CUDA and ROCm tie on rank; the tie breaks deterministically.
    listing = (
        "  CUDA0: NVIDIA (24268 MiB, 23000 MiB free)\n  ROCm0: AMD (24268 MiB, 23000 MiB free)\n"
    )
    _fake_listing(monkeypatch, listing)
    backends = {d.backend for d in probe_devices(Path("/bin/llama-server")).devices}
    assert len(backends) == 1


def test_probe_returns_empty_when_no_gpu_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only a CPU device listed -> no GPU backend to pin -> empty.
    _fake_listing(monkeypatch, "  CPU0: host cpu (64000 MiB)\n")
    assert probe_devices(Path("/bin/llama-server")).devices == []


def test_probe_returns_empty_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> str:
        raise OSError("no such binary")

    monkeypatch.setattr(dev_mod, "_run_list_devices", _boom)
    probe = probe_devices(Path("/bin/llama-server"))
    assert probe.devices == []
    assert probe.output == ""


@pytest.mark.skipif(sys.platform == "win32", reason="shell-script fake binary")
def test_probe_runs_the_real_binary(tmp_path: Path) -> None:
    # End to end through Popen: a real script's stdout is parsed into devices.
    script = tmp_path / "llama-server"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{_CUDA_LISTING}EOF\n")
    script.chmod(0o755)
    probe = probe_devices(script)
    assert [d.index for d in probe.devices] == [0, 1]


def test_run_list_devices_returns_the_child_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class _HealthyProc:
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, None]:
            return (_CUDA_LISTING, None)

    monkeypatch.setattr(dev_mod.subprocess, "Popen", lambda *_a, **_k: _HealthyProc())
    assert dev_mod._run_list_devices(Path("/bin/llama-server"), 1.0) == (_CUDA_LISTING, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_probe_timeout_raises_and_kills_the_child(tmp_path: Path) -> None:
    """A probe wedged past the timeout raises a named error instead of hanging.

    The bug class this pins: on a host with a wedged GPU driver, --list-devices
    never returns, and treating that as 'no devices' (or waiting forever) turned
    the whole serve into a silent never-ready fleet (bb-0yf0).
    """
    script = tmp_path / "llama-server"
    script.write_text("#!/bin/sh\nsleep 30\n")
    script.chmod(0o755)
    with pytest.raises(ProviderError, match="did not respond"):
        probe_devices(script, timeout_s=0.2)


def test_probe_abandons_an_unreapable_child(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A child that survives SIGKILL (uninterruptible driver I/O) is abandoned
    after a bounded reap instead of blocking the caller forever.

    The POSIX group-kill path is forced (repo pattern: simulate the platform)
    so the same lines are exercised on every CI host, Windows included.
    """

    class _WedgedProc:
        pid = 12345

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 0)

    monkeypatch.setattr(dev_mod.subprocess, "Popen", lambda *_a, **_k: _WedgedProc())
    monkeypatch.setattr(dev_mod.os, "name", "posix")
    monkeypatch.setattr(dev_mod.os, "killpg", lambda *_a: None, raising=False)
    monkeypatch.setattr(dev_mod.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(dev_mod, "_PROBE_KILL_WAIT_S", 0.01)
    with caplog.at_level("WARNING"), pytest.raises(ProviderError, match="did not respond"):
        probe_devices(Path("/bin/llama-server"), timeout_s=0.01)
    assert "abandoned" in caplog.text


def test_probe_env_sets_pci_bus_order() -> None:
    assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_visible_env_cuda_pins_by_index_and_pins_order() -> None:
    env = visible_env((FleetDevice("CUDA", 2, "", 0, 0), FleetDevice("CUDA", 3, "", 0, 0)))
    assert env == {"CUDA_VISIBLE_DEVICES": "2,3", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_visible_env_rocm_emits_single_var_on_clean_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ROCR and HIP filter sequentially, so the child pins with one var only;
    # with neither parent var set it defaults to HIP at the absolute index.
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
    assert env == {"HIP_VISIBLE_DEVICES": "1"}


def test_visible_env_does_not_pin_vulkan_by_raw_index() -> None:
    """GGML_VK_VISIBLE_DEVICES indexes the raw loader enumeration, not this list.

    These indices come from the engine's filtered device list, so re-emitting
    them into that variable changes index space wherever ggml drops or merges a
    device, and setting it also disables ggml's type filter, support check and
    same-UUID dedup. Vulkan is pinned with --device instead.
    """
    assert visible_env((FleetDevice("Vulkan", 0, "", 0, 0),)) == {}


def test_visible_env_metal_and_empty_pin_nothing() -> None:
    assert visible_env(()) == {}
    assert visible_env((FleetDevice("Metal", 0, "", 0, 0),)) == {}


class TestPresetVisibleDeviceComposition:
    """A pod-preset visible-devices var makes probe indices RELATIVE; the child
    env must map them back through the parent list to the same physical devices."""

    def test_cuda_integer_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_cuda_integer_parent_subset_picks_the_right_physical_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        env = visible_env((FleetDevice("CUDA", 1, "", 0, 0),))
        assert env["CUDA_VISIBLE_DEVICES"] == "3"  # relative 1 -> physical 3

    def test_cuda_uuid_parent_list_composes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa, GPU-bbb")
        env = visible_env((FleetDevice("CUDA", 1, "", 0, 0),))
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"

    def test_cuda_index_past_parent_list_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An index outside the parent restriction is an invariant violation; pinning
        a bare absolute index into a (possibly UUID) parent list would select the
        wrong GPU, so composition fails loudly instead (bb-7jg1.13)."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        with pytest.raises(ValueError, match="outside the parent visible-devices list"):
            visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))

    def test_cuda_clean_env_emits_relative_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0), FleetDevice("CUDA", 1, "", 0, 0)))
        assert env == {"CUDA_VISIBLE_DEVICES": "0,1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    def test_preset_cuda_device_order_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
        env = visible_env((FleetDevice("CUDA", 0, "", 0, 0),))
        assert env["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"  # same order the probe used

    def test_probe_env_defaults_order_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
        assert dev_mod._probe_env()["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    def test_rocm_rocr_only_parent_emits_rocr_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Parent restricted via ROCR only: emit ROCR composed against it, and do
        # NOT also emit HIP (which would re-index within the ROCR survivors).
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
        assert env == {"ROCR_VISIBLE_DEVICES": "5"}  # relative 1 -> physical 5

    def test_rocm_hip_only_parent_emits_hip_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The finding's scenario: HIP-only parent. Emitting an absolute ROCR here
        # would select the wrong physical card, so only HIP is composed/emitted.
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
        env = visible_env((FleetDevice("HIP", 0, "", 0, 0),))
        assert env == {"HIP_VISIBLE_DEVICES": "1"}  # relative 0 -> physical 1, no ROCR

    def test_rocm_both_parents_set_emits_hip_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When both are set, the inherited ROCR stays in force and only HIP is
        # re-pinned (within the ROCR survivors), so the cap is never doubled.
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "6,7")
        env = visible_env((FleetDevice("ROCm", 1, "", 0, 0),))
        assert env == {"HIP_VISIBLE_DEVICES": "7"}  # relative 1 -> physical 7

    def test_vulkan_leaves_a_parent_restriction_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe already enumerated under the parent's restriction.

        Composing into the variable would re-filter an already-filtered list;
        the --device names are relative to what the engine reports under that
        same restriction, so the parent's value is inherited as-is.
        """
        monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1,2")
        assert visible_env((FleetDevice("Vulkan", 1, "", 0, 0),)) == {}


def test_software_rasterizer_is_not_planned_as_a_gpu() -> None:
    """Mesa's CPU rasterizer enumerates through Vulkan exactly like a GPU.

    It reports system RAM as VRAM, so a host with a real integrated GPU beside
    lavapipe would be planned as a two-GPU machine and tensor-split across a
    real adapter and a software renderer -- far slower than either the iGPU
    alone or plain CPU inference.
    """
    from lilbee.providers.fleet.devices import FleetDevice, _select_backend

    igpu = FleetDevice("Vulkan", 0, "Intel(R) Iris(R) Xe Graphics", 8 * 10**9, 7 * 10**9)
    llvmpipe = FleetDevice("Vulkan", 1, "llvmpipe (LLVM 17.0.6, 256 bits)", 15 * 10**9, 14 * 10**9)

    assert _select_backend([igpu, llvmpipe]) == [igpu]


def test_a_software_rasterizer_alone_is_no_gpu_at_all() -> None:
    """A GPU-less host with mesa Vulkan must plan as CPU-only, not as a big GPU."""
    from lilbee.providers.fleet.devices import FleetDevice, _select_backend

    lavapipe = FleetDevice("Vulkan", 0, "llvmpipe (LLVM 17.0.6, 256 bits)", 15 * 10**9, 14 * 10**9)

    assert _select_backend([lavapipe]) == []


def test_a_paravirtual_adapter_is_dropped_even_though_its_name_looks_real(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a VM ggml falls back to the first non-CPU adapter, which is VIRTUAL_GPU.

    Nothing in "Virtio-GPU Venus (Intel ...)" marks it as not a GPU, so the
    software-renderer name list walks straight past it; the loader's device
    type is the only thing that separates it from a real card.
    """
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _select_backend

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {
            "Virtio-GPU Venus (Intel(R) Iris(R) Xe Graphics)": gpu_select.VkDeviceType.VIRTUAL_GPU
        },
    )
    venus = FleetDevice(
        "Vulkan", 0, "Virtio-GPU Venus (Intel(R) Iris(R) Xe Graphics)", 15 * 10**9, 15 * 10**9
    )

    assert _select_backend([venus]) == []


def test_an_adapter_the_loader_cannot_type_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """No loader, no opinion: dropping devices on missing evidence would blind working hosts."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _select_backend

    monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", dict)
    card = FleetDevice("Vulkan", 0, "AMD Radeon RX 7900 XTX", 24 * 10**9, 24 * 10**9)

    assert _select_backend([card]) == [card]


def test_unified_memory_is_read_by_name_not_by_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    """--list-devices numbers survivors; the loader numbers everything it enumerated.

    A host whose loader lists a software rasterizer ahead of its integrated GPU
    has the iGPU at loader index 1 and at Vulkan0 in the engine's output, so an
    index comparison reads it as dedicated and hands placement the host's own
    RAM as GPU headroom.
    """
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {
            "llvmpipe (LLVM 17.0.6, 256 bits)": gpu_select.VkDeviceType.CPU,
            "Intel(R) Iris(R) Xe Graphics": gpu_select.VkDeviceType.INTEGRATED_GPU,
        },
    )

    parsed = _parse_devices("  Vulkan0: Intel(R) Iris(R) Xe Graphics (15690 MiB, 15690 MiB free)")

    assert [d.unified for d in parsed] == [True]


def test_an_amd_apu_on_the_rocm_path_is_not_sized_as_dedicated_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCm text output has no device type; hipMemGetInfo on an APU returns system RAM.

    The APU also ships a Vulkan driver, so the loader can answer what the ROCm
    listing cannot: a machine reporting adapters but no discrete one has no
    discrete card for ROCm to be enumerating.
    """
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {"AMD Radeon Graphics (RADV RENOIR)": gpu_select.VkDeviceType.INTEGRATED_GPU},
    )

    parsed = _parse_devices("  ROCm0: AMD Radeon Graphics (16000 MiB, 15000 MiB free)")

    assert [d.unified for d in parsed] == [True]


def test_a_discrete_nvidia_host_is_untouched_by_the_apu_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule may only ever add the shared-RAM budget to hosts that have no card."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {
            "NVIDIA GeForce RTX 3090": gpu_select.VkDeviceType.DISCRETE_GPU,
            "Intel(R) UHD Graphics 770": gpu_select.VkDeviceType.INTEGRATED_GPU,
        },
    )

    parsed = _parse_devices("  CUDA0: NVIDIA GeForce RTX 3090 (24268 MiB, 23500 MiB free)")

    assert [d.unified for d in parsed] == [False]


def test_no_vulkan_loader_leaves_cuda_devices_dedicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A headless CUDA container has no Vulkan ICD; silence must not read as integrated."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", dict)

    parsed = _parse_devices("  CUDA0: NVIDIA H100 80GB HBM3 (81559 MiB, 81000 MiB free)")

    assert [d.unified for d in parsed] == [False]


class TestAmdPinNeverSetsBothVars:
    """ROCr filters first and HIP re-indexes within the survivors.

    Writing the same index string to both selects the wrong cards or none:
    gpu_devices=1 on a two-GPU box exposes physical GPU 1 as index 0 through
    ROCr, and HIP then asks for index 1 of a one-device list.
    """

    @pytest.fixture(autouse=True)
    def _restore_env(self) -> Iterator[None]:
        # The pin writes os.environ in place, which monkeypatch does not track;
        # without this the pin leaks into every later test that reads the env.
        from lilbee.providers.fleet import gpu_env

        snapshot = {name: os.environ.get(name) for name in gpu_env._GPU_VISIBLE_ENV_VARS}
        try:
            yield
        finally:
            for name, value in snapshot.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _pin(self, monkeypatch: pytest.MonkeyPatch, value: str) -> dict[str, str]:
        from lilbee.core.config import cfg
        from lilbee.providers.fleet import gpu_env

        monkeypatch.setattr(cfg, "gpu_devices", value)
        assert gpu_env._apply_gpu_devices_pin() is True
        return {
            name: os.environ[name] for name in gpu_env._GPU_VISIBLE_ENV_VARS if name in os.environ
        }

    def test_a_clean_environment_gets_hip_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.fleet import gpu_env

        for name in gpu_env._GPU_VISIBLE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        applied = self._pin(monkeypatch, "1")

        assert "ROCR_VISIBLE_DEVICES" not in applied
        assert applied["HIP_VISIBLE_DEVICES"] == "1"

    def test_an_environment_already_masked_by_rocr_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_env

        for name in gpu_env._GPU_VISIBLE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")

        applied = self._pin(monkeypatch, "1")

        assert "HIP_VISIBLE_DEVICES" not in applied
        assert applied["ROCR_VISIBLE_DEVICES"] == "2,3"

    def test_the_pin_still_reaches_the_other_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_env

        for name in gpu_env._GPU_VISIBLE_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        applied = self._pin(monkeypatch, "1")

        assert applied["CUDA_VISIBLE_DEVICES"] == "1"
        assert applied["GGML_VK_VISIBLE_DEVICES"] == "1"


def test_a_dual_vendor_host_plans_onto_the_bigger_card_not_the_later_name() -> None:
    """A build loading both backends made the rank tie real; "ROCm" > "CUDA" decided it.

    A 4090 beside an RX 6600 planned onto the AMD card and the NVIDIA card
    idled, with nothing logged to say why.
    """
    from lilbee.providers.fleet.devices import _select_backend

    rtx4090 = FleetDevice("CUDA", 0, "NVIDIA GeForce RTX 4090", 24 * 10**9, 24 * 10**9)
    rx6600 = FleetDevice("ROCm", 0, "AMD Radeon RX 6600", 8 * 10**9, 8 * 10**9)

    assert _select_backend([rtx4090, rx6600]) == [rtx4090]


def test_the_dropped_backend_is_named_in_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """Silently planning half a machine is what made this hard to see."""
    from lilbee.providers.fleet.devices import _select_backend

    rtx4090 = FleetDevice("CUDA", 0, "NVIDIA GeForce RTX 4090", 24 * 10**9, 24 * 10**9)
    rx6600 = FleetDevice("ROCm", 0, "AMD Radeon RX 6600", 8 * 10**9, 8 * 10**9)

    with caplog.at_level("INFO", logger="lilbee.providers.fleet.devices"):
        _select_backend([rtx4090, rx6600])

    assert "ROCm" in caplog.text
    assert "CUDA" in caplog.text


def test_equal_memory_still_resolves_to_one_backend_deterministically() -> None:
    """Mixing index spaces is the hazard; a tie must still leave exactly one backend."""
    from lilbee.providers.fleet.devices import _select_backend

    cuda = FleetDevice("CUDA", 0, "NVIDIA", 16 * 10**9, 16 * 10**9)
    rocm = FleetDevice("ROCm", 0, "AMD", 16 * 10**9, 16 * 10**9)

    first = _select_backend([cuda, rocm])
    second = _select_backend([rocm, cuda])

    assert len({d.backend for d in first}) == 1
    assert first == second


def test_a_multi_card_backend_beats_one_bigger_card() -> None:
    """The fleet is planned across every device of the chosen backend, so sum, not max."""
    from lilbee.providers.fleet.devices import _select_backend

    # Each AMD card is smaller than the NVIDIA one, so only their sum wins.
    two_amd = [
        FleetDevice("ROCm", 0, "AMD Radeon RX 7600 XT", 16 * 10**9, 16 * 10**9),
        FleetDevice("ROCm", 1, "AMD Radeon RX 7600 XT", 16 * 10**9, 16 * 10**9),
    ]
    one_nvidia = FleetDevice("CUDA", 0, "NVIDIA GeForce RTX 4090", 24 * 10**9, 24 * 10**9)

    assert _select_backend([one_nvidia, *two_amd]) == two_amd


def test_a_listing_without_a_free_figure_asks_the_loader_before_assuming_all_of_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ggml omits the free column when the driver has no VK_EXT_memory_budget.

    Reading the omission as an empty card is how a desktop holding gigabytes of
    compositor and browser VRAM was planned as fully free.
    """
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select, "vulkan_free_bytes_by_name", lambda: {"AMD Radeon RX 7900 XTX": 21 * 1024**3}
    )

    (device,) = _parse_devices("  Vulkan0: AMD Radeon RX 7900 XTX (24576 MiB)")

    assert device.total_bytes == 24576 * _MIB
    assert device.free_bytes == 21 * 1024**3


def test_a_listing_free_figure_still_wins_over_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's own number describes the process that will allocate."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(gpu_select, "vulkan_free_bytes_by_name", lambda: {"AMD": 1})

    (device,) = _parse_devices("  Vulkan0: AMD (24576 MiB, 20000 MiB free)")

    assert device.free_bytes == 20000 * _MIB


def test_a_backend_the_loader_cannot_speak_for_keeps_the_heap_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Vulkan loader knows nothing about a CUDA listing's devices."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(gpu_select, "vulkan_free_bytes_by_name", lambda: {"NVIDIA H100": 1})

    (device,) = _parse_devices("  CUDA0: NVIDIA H100 (81559 MiB)")

    assert device.free_bytes == device.total_bytes


def test_a_rasterizer_only_loader_does_not_make_a_real_card_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mesa reports llvmpipe on any host with it installed, vendor ICD or not.

    That is ordinary on headless CUDA boxes and in containers. Concluding "no
    discrete GPU" from a list holding only rasterizers marked a real 24 GB card
    as sharing the host's memory and shrank every budget for it.
    """
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {"llvmpipe (LLVM 17.0.6, 256 bits)": gpu_select.VkDeviceType.CPU},
    )

    (device,) = _parse_devices("  CUDA0: NVIDIA RTX A5000 (24112 MiB, 23899 MiB free)")

    assert device.unified is False


def test_an_integrated_adapter_beside_a_rasterizer_still_reads_as_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Tiger Lake case: llvmpipe sits next to a real iGPU, which is the signal."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    monkeypatch.setattr(
        gpu_select,
        "vulkan_device_types_by_name",
        lambda: {
            "llvmpipe (LLVM 17.0.6, 256 bits)": gpu_select.VkDeviceType.CPU,
            "Intel(R) Iris(R) Xe Graphics": gpu_select.VkDeviceType.INTEGRATED_GPU,
        },
    )

    (device,) = _parse_devices("  ROCm0: AMD Radeon Graphics (16000 MiB)")

    assert device.unified is True


class TestAnEngineThatCannotAnswerIsNotTakenAsAuthoritative:
    """The probe merges stderr into stdout, so "it printed something" is not
    evidence that it understood the question.

    A build predating --list-devices prints usage text and exits non-zero. Read
    as an authoritative empty device list, that plans a GPU box as CPU-only.
    """

    _USAGE = (
        "error: invalid argument: --list-devices\n"
        "usage: llama-server [options]\n\n"
        "general:\n  -h, --help    show this help message and exit\n"
    )

    def test_usage_text_on_a_nonzero_exit_is_not_a_device_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_listing(monkeypatch, self._USAGE, returncode=1)

        probe = probe_devices(Path("/bin/llama-server"))

        assert probe.devices == []
        assert probe.spoke_protocol is False

    def test_a_clean_run_listing_nothing_is_a_device_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The header prints before the loop, so it survives a host with no GPU."""
        _fake_listing(monkeypatch, "Available devices:\n")

        probe = probe_devices(Path("/bin/llama-server"))

        assert probe.devices == []
        assert probe.spoke_protocol is True

    def test_a_zero_exit_without_the_header_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Output-format drift: text arrives, no device line parses, and treating
        that as "no GPUs" plans a GPU box onto the CPU in silence."""
        _fake_listing(monkeypatch, "ggml_vulkan: no devices found\n")

        probe = probe_devices(Path("/bin/llama-server"))

        assert probe.spoke_protocol is False

    def test_a_crash_after_the_header_is_not_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine that dies partway through enumerating prints the header first.

        Believing the truncated list plans against whatever it managed to name,
        or against nothing at all, on a host that has GPUs.
        """
        _fake_listing(monkeypatch, "Available devices:\n", returncode=-11)

        assert probe_devices(Path("/bin/llama-server")).spoke_protocol is False

    def test_a_probe_that_could_not_run_at_all_claims_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> tuple[str, int]:
            raise OSError("no such binary")

        monkeypatch.setattr(dev_mod, "_run_list_devices", _boom)

        assert probe_devices(Path("/bin/llama-server")).spoke_protocol is False


def test_the_loader_is_asked_once_per_parse_not_once_per_device_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free memory is read fresh rather than cached, so the per-line cost is
    back unless the parse samples it once. An N-device listing must not mean N
    loader inits to answer one question."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    calls: list[int] = []

    def _counting() -> dict[str, int]:
        calls.append(1)
        return {}

    monkeypatch.setattr(gpu_select, "vulkan_free_bytes_by_name", _counting)

    _parse_devices(
        "  Vulkan0: Card A (16000 MiB)\n"
        "  Vulkan1: Card B (16000 MiB)\n"
        "  Vulkan2: Card C (16000 MiB)\n"
    )

    assert len(calls) == 1, f"asked the Vulkan loader {len(calls)} times for one parse"


def test_a_listing_that_reports_free_never_touches_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's own figure wins, so the loader is not opened at all."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _parse_devices

    def _must_not_run() -> dict[str, int]:
        raise AssertionError("the loader was consulted despite a free figure being printed")

    monkeypatch.setattr(gpu_select, "vulkan_free_bytes_by_name", _must_not_run)

    (device,) = _parse_devices("  Vulkan0: Card A (16000 MiB, 9000 MiB free)")

    assert device.free_bytes == 9000 * _MIB


class TestRefusingEveryDeviceIsRecorded:
    """Dropping a device from lilbee's view does not stop the engine using it.

    ggml's own fallback takes the first non-CPU adapter, so a VM whose only
    adapter is paravirtual gets a CPU-shaped plan and a model offloaded onto the
    device lilbee just refused.
    """

    def test_refusing_the_only_listed_gpu_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from lilbee.providers.fleet import gpu_select

        monkeypatch.setattr(
            gpu_select,
            "vulkan_device_types_by_name",
            lambda: {"Virtio-GPU Venus": gpu_select.VkDeviceType.VIRTUAL_GPU},
        )
        _fake_listing(monkeypatch, "Available devices:\n  Vulkan0: Virtio-GPU Venus (15000 MiB)\n")

        probe = probe_devices(Path("/bin/llama-server"))

        assert probe.devices == []
        assert probe.refused_all is True

    def test_a_host_with_no_gpu_at_all_has_refused_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No device to keep the engine off, so nothing to say."""
        _fake_listing(monkeypatch, "Available devices:\n  CPU0: host cpu (64000 MiB)\n")

        probe = probe_devices(Path("/bin/llama-server"))

        assert probe.devices == []
        assert probe.refused_all is False

    def test_keeping_a_device_is_not_a_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_listing(monkeypatch, "Available devices:\n  CUDA0: NVIDIA (24268 MiB)\n")

        assert probe_devices(Path("/bin/llama-server")).refused_all is False


class TestSyclPinsByNameNotBySelector:
    """ONEAPI_DEVICE_SELECTOR is a selector over a backend runtime, not the index
    space --list-devices numbers.

    A device the engine calls SYCL1 need not be Level Zero ordinal 1: OpenCL
    devices interleave, discarded devices shift the numbering, and multi-tile
    cards appear as sub-devices. Composing a level_zero ordinal from a SYCL one
    could pin a different physical card than the probe enumerated.
    """

    def test_no_selector_is_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ONEAPI_DEVICE_SELECTOR", raising=False)

        assert visible_env((FleetDevice("SYCL", 1, "Intel Arc A770", 0, 0),)) == {}

    def test_a_parent_selector_is_left_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine enumerated behind it, so its names are already relative to it."""
        monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:2,3")

        assert visible_env((FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0),)) == {}

    def test_the_pin_is_the_name_the_engine_printed(self) -> None:
        from lilbee.providers.fleet.planning import _device_names

        devices = (
            FleetDevice("SYCL", 0, "Intel Arc A770", 0, 0),
            FleetDevice("SYCL", 2, "Intel Arc A770", 0, 0),
        )

        assert _device_names(devices) == ("SYCL0", "SYCL2")

    def test_cuda_still_pins_through_its_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA's mask and the probe's enumeration do share one space."""
        from lilbee.providers.fleet.planning import _device_names

        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        devices = (FleetDevice("CUDA", 1, "NVIDIA", 0, 0),)

        assert _device_names(devices) == ()
        assert visible_env(devices)["CUDA_VISIBLE_DEVICES"] == "1"
