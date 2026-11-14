"""The llama.cpp pack backend follows the host: metal, cuda with a NVIDIA driver, else cpu."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import Config, cfg
from lilbee.providers.fleet import engine_manifest, first_use
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import (
    _CUDA_DRIVER_FLOOR,
    _CUDA_MIN_COMPUTE,
    LlamaCppAdapter,
    cuda_pack_gap,
    pack_backend,
)
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
    facts = {
        "driver": "550.135",
        "cuda_driver": cuda,
        "gpus": [{"name": f"card{i}", "cc": list(cc)} for i, cc in enumerate(cards)],
    }
    return collect(system=system, machine=machine, cpu_info={"flags": ["avx2"]}, nvidia=facts)


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


def test_apple_silicon_stays_on_metal_whatever_the_driver_reports() -> None:
    assert pack_backend(_nvidia("Darwin", "arm64", cards=PASCAL)) == "metal"


def test_a_card_older_than_the_pack_reaches_is_named_with_the_missing_build() -> None:
    gap = cuda_pack_gap(_nvidia("Linux", "x86_64", cards=((3, 7),)))
    assert "card0" in gap
    assert "compute capability 3.7" in gap
    assert "5.0" in gap
    assert "CPU" in gap


def test_a_driver_older_than_the_pack_needs_is_named_with_the_missing_build() -> None:
    gap = cuda_pack_gap(_nvidia("Linux", "x86_64", cuda=12030, cards=PASCAL))
    assert "CUDA 12.3" in gap
    assert "CUDA 12.4" in gap
    assert "card0" in gap


@pytest.mark.parametrize(
    "host",
    [
        pytest.param(_nvidia("Linux", "x86_64", cards=((12, 0),)), id="blackwell-served"),
        pytest.param(_nvidia("Linux", "x86_64", cards=((3, 7), (8, 6))), id="one-card-served"),
        pytest.param(_nvidia("Linux", "x86_64", cards=()), id="no-card"),
        pytest.param(_nvidia("Linux", "aarch64", cards=((3, 7),)), id="no-cuda-pack-for-host"),
        pytest.param(_nvidia("Darwin", "arm64", cards=((3, 7),)), id="metal"),
        pytest.param(
            collect(system="Linux", machine="x86_64", cpu_info={}, nvidia=UNREADABLE),
            id="driver-unreadable",
        ),
    ],
)
def test_a_host_the_cuda_pack_serves_or_never_had_has_no_gap(host: HostCaps) -> None:
    assert cuda_pack_gap(host) == ""


def test_the_first_use_warning_names_an_nvidia_card_the_pack_cannot_use(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = _nvidia("Linux", "x86_64", cards=((3, 7),))
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert "compute capability 3.7" in caplog.text


def test_the_first_use_warning_is_silent_for_a_card_the_pack_serves(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = _nvidia("Linux", "x86_64", cards=((12, 0),))
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.records == []


def test_the_mixed_host_with_an_old_nvidia_card_and_an_amd_gpu_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = _nvidia("Linux", "x86_64", cards=((3, 7),))
    monkeypatch.setattr(first_use, "session_host", lambda: host)
    monkeypatch.setattr(first_use, "installed_gpu_vendor_ids", lambda: frozenset({0x1002}))
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert len(caplog.records) == 1
    assert "compute capability 3.7" in caplog.text
    assert "AMD" not in caplog.text


_ROOT = Path(__file__).resolve().parents[1]
_PACK_CUDA_VERSION = re.compile(r"cuda-(\d+)\.(\d+)")


def _pack_build_cuda_versions() -> set[tuple[int, int]]:
    cells = yaml.safe_load(
        (_ROOT / ".github/workflows/llama-pack-cells.yml").read_text(encoding="utf-8")
    )
    matrix = cells["jobs"]["cell"]["strategy"]["matrix"]["include"]
    texts = [str(value) for cell in matrix if cell["backend"] == "cuda" for value in cell.values()]
    texts.append((_ROOT / "tools/wheel-build/llama_cuda_manylinux.sh").read_text(encoding="utf-8"))
    return {(int(m[1]), int(m[2])) for text in texts for m in _PACK_CUDA_VERSION.finditer(text)}


def test_the_driver_floor_is_the_cuda_version_the_pack_is_built_with() -> None:
    assert _pack_build_cuda_versions() == {_CUDA_DRIVER_FLOOR}


def test_the_compute_floor_follows_the_architectures_the_pack_build_targets() -> None:
    args = (_ROOT / "tools/wheel-build/cmake_args.sh").read_text(encoding="utf-8")
    assert "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all-major" in args
    assert _CUDA_MIN_COMPUTE == (5, 0)
