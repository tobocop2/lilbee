"""The llama.cpp pack backend follows the host: metal, cuda with a NVIDIA driver, else cpu."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import Config, cfg
from lilbee.providers.fleet import engine_manifest, first_use
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import LlamaCppAdapter, pack_backend
from lilbee.providers.fleet.host_caps import HostCaps, collect
from tests._published import render_table, wheel_name

VERSION = "1.12032.1"
SHA = "ab" * 32

MAC = collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None, mac_ver="15.1")
LINUX = collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, nvidia=None)
WINDOWS = collect(system="Windows", machine="AMD64", cpu_info={"flags": ["avx2"]}, nvidia=None)
CELLS = {
    "mac": (MAC, "macos-arm64-metal"),
    "linux": (LINUX, "linux-x86_64-cpu"),
    "windows": (WINDOWS, "windows-x86_64-cpu"),
}


@pytest.fixture(autouse=True)
def _config(tmp_path: Path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _host(monkeypatch: pytest.MonkeyPatch, host: HostCaps, cell: str) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: "99.0")
    file = wheel_name(EngineName.LLAMA, VERSION, cell)
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table([(file, SHA, 1)]))


@pytest.mark.parametrize("name", CELLS)
def test_the_pack_pin_names_the_backend_of_the_host(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    host, cell = CELLS[name]
    _host(monkeypatch, host, cell)
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == cell.rsplit("-", 1)[1]
    assert pin["sha256"] == SHA
    assert list(pin["platforms"]) == [cell]


@pytest.mark.parametrize("name", CELLS)
def test_the_engine_pin_reads_the_row_of_the_host_backend(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    host, cell = CELLS[name]
    _host(monkeypatch, host, cell)
    assert list(LlamaCppAdapter().pin_record()["platforms"]) == [cell]


def test_the_first_use_lock_on_apple_silicon_is_the_metal_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host(monkeypatch, MAC, "macos-arm64-metal")
    lock, cell = first_use.pack_lock()
    assert cell == "macos-arm64-metal"
    assert lock.backend == "metal"


def test_an_unpublished_pack_on_apple_silicon_names_the_metal_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host(monkeypatch, MAC, "macos-arm64-metal")
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    with pytest.raises(first_use.PackNotPublishedError, match="macos-arm64-metal"):
        first_use.pack_lock()


def test_the_store_folder_and_the_install_message_on_apple_silicon_say_metal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _host(monkeypatch, MAC, "macos-arm64-metal")
    lock, _cell = first_use.pack_lock()
    root = engines_root(tmp_path / "models")
    store = EngineStore(root)
    assert store.build_dir(lock) == root / "llama.cpp" / f"{VERSION}-metal"
    view = engine_service.BuildView("llama.cpp", VERSION, lock.backend)
    assert view.label == f"llama.cpp {VERSION} (metal)"


def _nvidia(
    system: str, machine: str, *, cuda: int | None = 12040, cards: tuple[tuple[int, int], ...]
) -> HostCaps:
    nvml = {
        "driver": "550.135",
        "cuda_driver": cuda,
        "gpus": [{"name": f"card{i}", "cc": list(cc)} for i, cc in enumerate(cards)],
    }
    return collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvidia=nvml)


PASCAL = ((6, 1),)
NO_DRIVER_LIBRARY = None
UNREADABLE = {"driver": None, "cuda_driver": None, "gpus": None}


@pytest.mark.parametrize(
    ("system", "machine", "cell"),
    [("Linux", "x86_64", "linux-x86_64-cuda"), ("Windows", "AMD64", "windows-x86_64-cuda")],
)
def test_a_host_with_a_usable_nvidia_driver_installs_the_cuda_pack(
    system: str, machine: str, cell: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = _nvidia(system, machine, cards=PASCAL)
    assert pack_backend(host) == "cuda"
    _host(monkeypatch, host, cell)
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == "cuda"
    assert list(pin["platforms"]) == [cell]
    assert pin["licenses"] == ["MIT AND LicenseRef-NVIDIA-CUDA-EULA"]


@pytest.mark.parametrize(
    ("system", "machine", "cell"),
    [("Linux", "x86_64", "linux-x86_64-cpu"), ("Windows", "AMD64", "windows-x86_64-cpu")],
)
def test_a_host_with_no_nvidia_driver_installs_the_cpu_pack(
    system: str, machine: str, cell: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = collect(system=system, machine=machine, cpu_info={}, nvidia=NO_DRIVER_LIBRARY)
    assert pack_backend(host) == "cpu"
    _host(monkeypatch, host, cell)
    pin = LlamaCppAdapter().pack_pin()
    assert pin["backend"] == "cpu"
    assert pin["licenses"] == ["MIT"]


@pytest.mark.parametrize(
    "host",
    [
        pytest.param(
            collect(system="Linux", machine="x86_64", cpu_info={}, nvidia=UNREADABLE),
            id="driver-unreadable",
        ),
        pytest.param(_nvidia("Linux", "x86_64", cuda=12030, cards=PASCAL), id="driver-12.3"),
        pytest.param(_nvidia("Linux", "x86_64", cards=((3, 7),)), id="compute-3.7"),
        pytest.param(_nvidia("Linux", "x86_64", cards=()), id="no-card"),
        pytest.param(_nvidia("Linux", "aarch64", cards=PASCAL), id="linux-arm64"),
        pytest.param(_nvidia("Windows", "ARM64", cards=PASCAL), id="windows-arm64"),
    ],
)
def test_a_host_without_a_usable_cuda_driver_or_card_installs_the_cpu_pack(
    host: HostCaps,
) -> None:
    assert pack_backend(host) == "cpu"


def test_one_supported_card_among_old_ones_selects_the_cuda_pack() -> None:
    host = _nvidia("Linux", "x86_64", cards=((3, 7), (8, 6)))
    assert pack_backend(host) == "cuda"


def test_a_newer_driver_than_the_floor_selects_the_cuda_pack() -> None:
    assert pack_backend(_nvidia("Linux", "x86_64", cuda=13010, cards=((12, 0),))) == "cuda"


def test_apple_silicon_stays_on_metal_whatever_nvml_reports() -> None:
    assert pack_backend(_nvidia("Darwin", "arm64", cards=PASCAL)) == "metal"
