"""The llama.cpp pack backend on AMD and Intel GPU hosts: rocm, vulkan, else cpu."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest, first_use, host_caps
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import _ROCM_GFX, LlamaCppAdapter, pack_backend
from lilbee.providers.fleet.host_caps import HostCaps, collect, gfx_target
from tests._published import render_table, wheel_name

AMD = 0x1002
INTEL = 0x8086
NVIDIA = 0x10DE
VERSION = "1.12032.1"
SHA = "cd" * 32
_ROOT = Path(__file__).resolve().parents[1]


def _host(
    system: str = "Linux",
    machine: str = "x86_64",
    *,
    vendors: tuple[int, ...] = (),
    gfx: tuple[str, ...] = (),
    loader: bool = False,
    numa: bool = False,
) -> HostCaps:
    return collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx2"]},
        nvidia=None,
        display_vendors=vendors,
        amd_gfx=gfx,
        vulkan_loader=loader,
        libnuma=numa,
    )


@pytest.mark.parametrize(
    ("host", "backend"),
    [
        pytest.param(_host(vendors=(AMD,), gfx=("gfx1100",), loader=True, numa=True), "rocm"),
        pytest.param(_host(vendors=(AMD,), gfx=("gfx1100",), loader=False, numa=True), "rocm"),
        pytest.param(_host(vendors=(AMD,), gfx=("gfx1100",), loader=True, numa=False), "vulkan"),
        pytest.param(_host(vendors=(AMD,), gfx=("gfx1010",), loader=True, numa=True), "vulkan"),
        pytest.param(_host(vendors=(AMD,), gfx=("gfx906",), loader=True, numa=True), "vulkan"),
        pytest.param(_host(vendors=(AMD,), loader=True, numa=True), "vulkan"),
        pytest.param(_host(vendors=(INTEL,), loader=True, numa=True), "vulkan"),
        pytest.param(_host(vendors=(AMD,), gfx=("gfx1100",), numa=False), "cpu"),
        pytest.param(_host(vendors=(INTEL,), loader=False), "cpu"),
        pytest.param(_host(vendors=(NVIDIA,), loader=True), "cpu"),
        pytest.param(_host(loader=True, numa=True), "cpu"),
        pytest.param(_host("Windows", "AMD64", vendors=(AMD,), loader=True), "vulkan"),
        pytest.param(_host("Windows", "AMD64", vendors=(INTEL,), loader=True), "vulkan"),
        pytest.param(
            _host("Windows", "AMD64", vendors=(AMD,), gfx=("gfx1100",), loader=True, numa=True),
            "vulkan",
        ),
        pytest.param(_host("Windows", "AMD64", vendors=(AMD,), loader=False), "cpu"),
        pytest.param(
            _host("Linux", "aarch64", vendors=(AMD,), gfx=("gfx1100",), loader=True, numa=True),
            "cpu",
        ),
        pytest.param(_host("Darwin", "arm64", vendors=(AMD,), loader=True), "metal"),
    ],
)
def test_the_backend_follows_the_gpu_and_the_host_libraries(host: HostCaps, backend: str) -> None:
    assert pack_backend(host) == backend


def test_a_usable_nvidia_card_wins_over_an_amd_gpu() -> None:
    facts = {"driver": "550", "cuda_driver": 12040, "gpus": [{"name": "c", "cc": [8, 6]}]}
    host = collect(
        system="Linux",
        machine="x86_64",
        cpu_info={},
        nvidia=facts,
        display_vendors=(AMD, NVIDIA),
        amd_gfx=("gfx1100",),
        vulkan_loader=True,
        libnuma=True,
    )
    assert pack_backend(host) == "cuda"


@pytest.mark.parametrize(
    ("system", "machine", "vendors", "gfx", "cell", "licences"),
    [
        ("Linux", "x86_64", (AMD,), ("gfx1100",), "linux-x86_64-rocm", "ROCM"),
        ("Linux", "x86_64", (INTEL,), (), "linux-x86_64-vulkan", ["MIT"]),
        ("Windows", "AMD64", (AMD,), (), "windows-x86_64-vulkan", ["MIT"]),
    ],
)
def test_the_pack_pin_names_the_cell_and_licences_of_the_gpu_pack(
    system: str,
    machine: str,
    vendors: tuple[int, ...],
    gfx: tuple[str, ...],
    cell: str,
    licences: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(system, machine, vendors=vendors, gfx=gfx, loader=True, numa=True)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    monkeypatch.setattr(
        engine_manifest,
        "MANIFESTS",
        render_table([(wheel_name(EngineName.LLAMA, VERSION, cell), SHA, 1)]),
    )
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == cell.rsplit("-", 1)[1]
    assert pin["sha256"] == SHA
    assert list(pin["platforms"]) == [cell]
    if licences == "ROCM":
        assert pin["licenses"] == [
            "MIT AND BSD-3-Clause AND NCSA AND Apache-2.0 WITH LLVM-exception"
        ]
    else:
        assert pin["licenses"] == licences


@pytest.mark.parametrize(
    ("version", "name"),
    [(110000, "gfx1100"), (90010, "gfx90a"), (100300, "gfx1030"), (90006, "gfx906")],
)
def test_the_kernel_gfx_version_becomes_the_gfx_name(version: int, name: str) -> None:
    assert gfx_target(version) == name


def test_the_gfx_targets_come_from_each_gpu_node_of_the_compute_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for node, version in (("0", 0), ("1", 110000), ("2", 90010)):
        folder = tmp_path / node
        folder.mkdir()
        (folder / "properties").write_text(
            f"cpu_cores_count 8\ngfx_target_version {version}\nsimd_count 0\n", encoding="utf-8"
        )
    monkeypatch.setattr(host_caps, "_SYSFS_KFD_NODES", tmp_path)
    assert host_caps._amd_gfx_targets() == ["gfx1100", "gfx90a"]


def test_a_host_with_no_compute_driver_reports_no_gfx_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_caps, "_SYSFS_KFD_NODES", tmp_path / "missing")
    assert host_caps._amd_gfx_targets() == []


def test_an_unreadable_compute_driver_folder_reports_no_gfx_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Denied:
        def glob(self, pattern: str) -> list[Path]:
            raise PermissionError(pattern)

    monkeypatch.setattr(host_caps, "_SYSFS_KFD_NODES", _Denied())
    assert host_caps._amd_gfx_targets() == []


def test_the_live_host_reads_the_vendors_the_gfx_targets_and_the_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_caps.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_caps.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host_caps, "host_facts", lambda: None)
    monkeypatch.setattr(host_caps, "_cpu_info", lambda: {})
    monkeypatch.setattr(host_caps, "_accel_devices", lambda: [])
    monkeypatch.setattr(host_caps, "installed_gpu_vendor_ids", lambda: frozenset({AMD}))
    monkeypatch.setattr(host_caps, "_amd_gfx_targets", lambda: ["gfx1100"])
    asked: list[str] = []

    def _find(name: str) -> str | None:
        asked.append(name)
        return f"lib{name}.so.1"

    monkeypatch.setattr(host_caps, "find_library", _find)
    host = host_caps.detect()
    assert host.display_vendors == {AMD}
    assert host.amd_gfx == ("gfx1100",)
    assert host.vulkan_loader and host.libnuma
    assert sorted(asked) == ["numa", "vulkan"]
    assert pack_backend(host) == "rocm"


def test_the_live_windows_host_asks_for_the_windows_loader_and_not_for_numa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_caps.platform, "system", lambda: "Windows")
    monkeypatch.setattr(host_caps.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(host_caps, "host_facts", lambda: None)
    monkeypatch.setattr(host_caps, "_cpu_info", lambda: {})
    monkeypatch.setattr(host_caps, "installed_gpu_vendor_ids", lambda: frozenset({INTEL}))
    asked: list[str] = []
    monkeypatch.setattr(host_caps, "find_library", lambda name: asked.append(name) or "x.dll")
    host = host_caps.detect()
    assert asked == ["vulkan-1"]
    assert host.amd_gfx == ()
    assert not host.libnuma
    assert pack_backend(host) == "vulkan"


def test_a_missing_library_is_reported_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_caps, "find_library", lambda _name: None)
    assert host_caps._has_library("vulkan") is False


@pytest.mark.parametrize(
    ("host", "advice"),
    [
        pytest.param(
            _host(vendors=(INTEL,)),
            "For the Vulkan build, install the Vulkan loader (libvulkan1 or vulkan-loader). ",
            id="linux-x86_64",
        ),
        pytest.param(
            _host("Windows", "AMD64", vendors=(INTEL,)),
            "For the Vulkan build, install or update the GPU driver, "
            "which provides the Vulkan loader. ",
            id="windows-x86_64",
        ),
        pytest.param(_host("Linux", "aarch64", vendors=(INTEL,)), "", id="linux-arm64"),
        pytest.param(_host("Windows", "ARM64", vendors=(INTEL,)), "", id="windows-arm64"),
        pytest.param(_host("Darwin", "x86_64", vendors=(INTEL,)), "", id="macos-x86_64"),
    ],
)
def test_the_warning_for_an_unserved_gpu_matches_the_host_platform(
    host: HostCaps,
    advice: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: frozenset({INTEL}))
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.messages == [
        "This computer has an Intel GPU, and no llama.cpp pack can use it here. "
        f"lilbee installs the CPU build. {advice}To use your own build, set "
        "LILBEE_LLAMA_SERVER_PATH to your llama-server."
    ]


def test_a_host_that_gets_the_vulkan_pack_is_not_warned(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = _host(vendors=(INTEL,), loader=True)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: frozenset({INTEL}))
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.records == []


def test_the_rocm_gfx_set_is_the_target_list_of_the_pack_build_less_gfx906() -> None:
    script = (_ROOT / "tools/wheel-build/cmake_args.sh").read_text(encoding="utf-8")
    wanted = re.search(r'^rocm_wanted_targets="([^"]+)"', script, re.MULTILINE)
    assert wanted is not None
    # The ROCm of the pack build has no GEMM kernels for gfx906, so the built wheel omits it.
    assert set(wanted[1].split()) - {"gfx906"} == _ROCM_GFX


@pytest.mark.parametrize(
    ("system", "machine", "gfx", "cell", "sha256", "size"),
    [
        (
            "Linux",
            "x86_64",
            ("gfx1100",),
            "linux-x86_64-rocm",
            "22b8e8e0df62382db4bdb624df6a569ee4ba147c4f2fce80a42b4e2ea24853e3",
            507195826,
        ),
        (
            "Linux",
            "x86_64",
            (),
            "linux-x86_64-vulkan",
            "4346a6bbbee15dbfa1e0e736f174fa86d568d23b52b69ab64959a5369e727662",
            56085481,
        ),
        (
            "Windows",
            "AMD64",
            (),
            "windows-x86_64-vulkan",
            "87e5a2e2735aedb7446273d0989fa9ec69843828fa57cc8e5ca12cee0e3e992d",
            64795105,
        ),
    ],
)
def test_the_shipped_table_pins_the_published_wheel_of_each_gpu_cell(
    system: str,
    machine: str,
    gfx: tuple[str, ...],
    cell: str,
    sha256: str,
    size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _host(system, machine, vendors=(AMD,), gfx=gfx, loader=True, numa=True)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    pin = LlamaCppAdapter().pack_pin()
    assert pin["sha256"] == sha256
    assert list(pin["platforms"]) == [cell]
    lock, lock_cell = first_use.pack_lock()
    assert lock_cell == cell
    # The downloader checks the byte count of the wheel against this size.
    assert lock.platforms[cell].size == size
    row = engine_manifest.MANIFESTS["llama.cpp"][VERSION][cell]
    assert row["url"].startswith(
        "https://github.com/tobocop2/lilbee/releases/download/engine-packs-20261116-llama-gpu/"
    )
    filename = row["url"].rsplit("/", 1)[1]
    build = "2" if system == "Windows" else "1"
    assert filename.startswith(
        f"lilbee_engine_llama_cpp-{VERSION}-{build}.{cell.rsplit('-', 1)[1]}-"
    )
    assert row["min_os"] == ("2.34" if system == "Linux" else None)
