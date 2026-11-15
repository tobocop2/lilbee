"""The coverage table lists every cell lilbee ships, and no cell that nothing ships."""

from __future__ import annotations

import itertools
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from lilbee.providers.fleet.engine_digests import MANIFESTS
from lilbee.providers.fleet.engine_manifest import cell_key
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.engines.proof import HARDWARE_PROOF
from lilbee.providers.fleet.host_caps import Feature, HostArch, HostCaps, HostOS, Npu, NpuVendor
from tests.conformance.engine_coverage import (
    COVERAGE_ROWS,
    INFINITY,
    PRESET_CELL,
    Coverage,
    CoverageRow,
    Device,
)

ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = Path(__file__).parent / "fixtures" / "presets"
RECORD_DIR = PurePosixPath("tests/conformance/fixtures/compare")

Cell = tuple[str, str]
Manifests = Mapping[str, Mapping[str, Mapping[str, Any]]]


def _best_host(os: HostOS, arch: HostArch) -> HostCaps:
    """A host of *os* and *arch* with every device an engine could ask for."""
    return HostCaps(
        os=os,
        arch=arch,
        avx2=Feature(True),
        neon=Feature(True),
        gpus=(),
        nvidia_driver_cuda=None,
        nvidia_reason="",
        nvidia_known=True,
        macos_version=(15, 0),
        npus=(Npu(NpuVendor.INTEL, "Intel NPU"), Npu(NpuVendor.AMD, "AMD NPU", "XDNA2")),
        npu_known=True,
    )


def pack_cells(manifests: Manifests) -> set[Cell]:
    """The (engine, cell key) pairs that the pack table lists, over every version."""
    return {
        (engine, key)
        for engine, versions in manifests.items()
        for cells in versions.values()
        for key in cells
    }


def own_cells(
    adapters: Mapping[EngineName, EngineAdapter],
    manifests: Manifests,
    monkeypatch: pytest.MonkeyPatch,
) -> set[Cell]:
    """The cells of the engines that ship no pack, read from each adapter's own host gate."""
    found: set[Cell] = set()
    for name, adapter in adapters.items():
        if name.value in manifests:
            continue
        for os, arch in itertools.product(HostOS, HostArch):
            host = _best_host(os, arch)
            monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda host=host: host)
            key = cell_key(host, adapter.pin_record()["backend"])
            if adapter.host_gate(host).ok and key is not None:
                found.add((name.value, key))
    return found


def row_cells(rows: Iterable[CoverageRow]) -> set[Cell]:
    """The (engine, cell) pairs the rows name, preset rows apart."""
    return {(r.engine, r.cell) for r in rows if r.cell != PRESET_CELL}


def uncovered(shipped: set[Cell], rows: Iterable[CoverageRow]) -> list[Cell]:
    """The shipped cells that no row names."""
    return sorted(shipped - row_cells(rows))


def stale(shipped: set[Cell], rows: Iterable[CoverageRow]) -> list[Cell]:
    """The cells that rows name and nothing ships."""
    return sorted(row_cells(rows) - shipped)


def path_fault(record: str) -> str:
    """Why *record* is not a path under the compare folder; empty when it is."""
    path = PurePosixPath(record)
    if not record:
        return "no record"
    if path.is_absolute() or PureWindowsPath(record).anchor:
        return "record path is absolute"
    if ".." in path.parts:
        return "record path leaves its folder"
    return "" if path.is_relative_to(RECORD_DIR) else f"record is outside {RECORD_DIR}"


def content_fault(record: str, root: Path) -> str:
    """Why the tracked file *record* is empty or not JSON; empty when it parses."""
    text = (root / record).read_text(encoding="utf-8")
    if not text.strip():
        return "record is empty"
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return "record is not JSON"
    return ""


def record_fault(record: str, root: Path = ROOT) -> str:
    """Why *record* cannot back a COMPARED row; empty when it can."""
    if fault := path_fault(record):
        return fault
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", record],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        return "record is not tracked by git"
    return content_fault(record, root)


def row_faults(row: CoverageRow, root: Path = ROOT) -> list[str]:
    """What is missing from *row* for its status; empty when complete."""
    faults: list[str] = []
    if not row.reason:
        faults.append("no reason")
    if row.status is Coverage.DEFERRED and not row.host:
        faults.append("deferred with no host")
    if row.status is Coverage.EXCLUDED and not row.owner_decision:
        faults.append("excluded with no owner decision")
    if row.status is Coverage.COMPARED and (fault := record_fault(row.record, root)):
        faults.append(fault)
    return faults


def _row(**changes: Any) -> CoverageRow:
    base = CoverageRow("e", "c", Device.CPU, Coverage.DEFERRED, "why", host="h")
    return CoverageRow(**{**base.__dict__, **changes})


@pytest.fixture
def shipped(monkeypatch: pytest.MonkeyPatch) -> set[Cell]:
    return pack_cells(MANIFESTS) | own_cells(ADAPTERS, MANIFESTS, monkeypatch)


def test_registry_and_pack_table_are_not_empty() -> None:
    assert list(EngineName)
    assert MANIFESTS
    assert set(MANIFESTS) <= {e.value for e in EngineName}


def test_derived_cells_include_each_kind_of_source(shipped: set[Cell]) -> None:
    assert ("llama.cpp", "windows-x86_64-cuda") in shipped
    assert ("lemonade", "linux-x86_64-npu") in shipped
    assert ("mlx", "macos-arm64-metal") in shipped
    assert {engine for engine, _ in shipped} == {e.value for e in EngineName}


def test_every_shipped_cell_has_a_row(shipped: set[Cell]) -> None:
    assert uncovered(shipped, COVERAGE_ROWS) == []


def test_no_row_names_a_cell_nothing_ships(shipped: set[Cell]) -> None:
    assert stale(shipped, COVERAGE_ROWS) == []


def test_a_new_pack_cell_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    grown = {**MANIFESTS, "ik_llama.cpp": {"1": {"linux-x86_64-vulkan": {}}}}
    shipped = pack_cells(grown) | own_cells(ADAPTERS, grown, monkeypatch)
    assert uncovered(shipped, COVERAGE_ROWS) == [("ik_llama.cpp", "linux-x86_64-vulkan")]


def test_a_removed_row_is_named(shipped: set[Cell]) -> None:
    rows = [r for r in COVERAGE_ROWS if (r.engine, r.cell) != ("llama.cpp", "linux-x86_64-cuda")]
    assert uncovered(shipped, rows) == [("llama.cpp", "linux-x86_64-cuda")]


def test_a_fabricated_engine_is_named(shipped: set[Cell]) -> None:
    assert uncovered(shipped | {("fabricated-engine", "linux-x86_64-cpu")}, COVERAGE_ROWS) == [
        ("fabricated-engine", "linux-x86_64-cpu")
    ]


def test_a_row_for_an_unshipped_cell_is_named(shipped: set[Cell]) -> None:
    rows = [*COVERAGE_ROWS, _row(engine="llama.cpp", cell="linux-x86_64-vulkan")]
    assert stale(shipped, rows) == [("llama.cpp", "linux-x86_64-vulkan")]


def test_engine_rows_name_a_registry_engine_and_preset_rows_a_recorded_preset() -> None:
    engines = {e.value for e in EngineName}
    presets = {p.name.split("-")[0] for p in PRESET_DIR.glob("*.json")}
    assert INFINITY in presets
    for row in COVERAGE_ROWS:
        assert row.engine in (presets if row.cell == PRESET_CELL else engines), row


def test_each_engine_cell_and_device_appears_once() -> None:
    keys = [(r.engine, r.cell, r.device) for r in COVERAGE_ROWS]
    assert keys
    assert len(keys) == len(set(keys))


def test_every_row_is_complete_for_its_status() -> None:
    faults = [(r.engine, r.cell, r.device, f) for r in COVERAGE_ROWS for f in row_faults(r)]
    assert COVERAGE_ROWS
    assert faults == []


def test_npu_rows_are_excluded() -> None:
    npu = [r for r in COVERAGE_ROWS if r.device is Device.NPU]
    assert npu
    assert {r.status for r in npu} == {Coverage.EXCLUDED}


def proof_device(word: str) -> Device:
    """The coverage device that a proof.py device word names."""
    if word.endswith("NPU"):
        return Device.NPU
    if word.endswith("CPU"):
        return Device.CPU
    if "graphics" in word:
        return Device.INTEL_GPU
    raise AssertionError(f"proof.py names a device with no coverage device: {word}")


def cell_device(cell: str) -> Device:
    """The device a pack cell key runs on, from its backend part."""
    backend = cell.split("-", 2)[2]
    return next(d for d in Device if backend.startswith(d.value))


def proof_cells(rows: Iterable[CoverageRow]) -> list[tuple[str, Device]]:
    """The (engine, device) pairs of proof.py that no row names."""
    named = {(r.engine, r.device) for r in rows}
    return sorted(
        (engine, proof_device(word))
        for engine, devices in HARDWARE_PROOF.items()
        for word in devices
        if (engine, proof_device(word)) not in named
    )


def unknown_devices(rows: Iterable[CoverageRow]) -> list[tuple[str, str, Device]]:
    """The rows whose device neither proof.py nor the cell's own backend names."""
    unknown = []
    for r in rows:
        if r.cell == PRESET_CELL:
            continue
        known = (
            {proof_device(w) for w in HARDWARE_PROOF[r.engine]}
            if r.engine in HARDWARE_PROOF
            else {cell_device(r.cell)}
        )
        if r.device not in known:
            unknown.append((r.engine, r.cell, r.device))
    return unknown


def test_every_proof_device_has_a_row() -> None:
    assert HARDWARE_PROOF
    assert proof_cells(COVERAGE_ROWS) == []


def test_a_removed_device_row_is_named() -> None:
    rows = [r for r in COVERAGE_ROWS if (r.engine, r.device) != ("openvino", Device.INTEL_GPU)]
    assert proof_cells(rows) == [("openvino", Device.INTEL_GPU)]


def test_no_row_names_a_device_that_proof_and_the_cell_do_not_know() -> None:
    assert unknown_devices(COVERAGE_ROWS) == []


def test_a_row_with_an_unknown_device_is_named() -> None:
    rows = [*COVERAGE_ROWS, _row(engine="lemonade", cell="linux-x86_64-npu", device=Device.CPU)]
    assert unknown_devices(rows) == [("lemonade", "linux-x86_64-npu", Device.CPU)]
    cuda_on_cpu = _row(engine="tei", cell="linux-x86_64-cpu", device=Device.CUDA)
    assert unknown_devices([cuda_on_cpu]) == [("tei", "linux-x86_64-cpu", Device.CUDA)]


def test_cell_device_reads_the_backend_part() -> None:
    assert cell_device("linux-x86_64-cuda128-sm89") is Device.CUDA
    assert cell_device("macos-arm64-metal") is Device.METAL


def test_an_unknown_proof_device_word_is_refused() -> None:
    with pytest.raises(AssertionError, match="Quantum"):
        proof_device("Quantum unit")


def test_a_pack_ships_reason_names_a_cell_in_the_pack_table() -> None:
    claims = [(r.engine, r.cell) for r in COVERAGE_ROWS if "pack ships" in r.reason]
    assert claims
    assert [c for c in claims if c not in pack_cells(MANIFESTS)] == []


def test_a_deferred_row_without_a_host_is_a_fault() -> None:
    assert row_faults(_row(host="")) == ["deferred with no host"]


def test_an_excluded_row_without_an_owner_decision_is_a_fault() -> None:
    assert row_faults(_row(status=Coverage.EXCLUDED, host="")) == [
        "excluded with no owner decision"
    ]


def test_a_row_without_a_reason_is_a_fault() -> None:
    assert row_faults(_row(reason="")) == ["no reason"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with tracked and untracked records, good and bad."""
    folder = tmp_path / RECORD_DIR
    folder.mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    files = {
        folder / "good.json": '{"engine": "x"}',
        folder / "empty.json": "  \n",
        folder / "notjson.json": "not json",
        folder / "untracked.json": "{}",
        tmp_path / "outside" / "good.json": "{}",
    }
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    tracked = [str(p.relative_to(tmp_path)) for p in files if p.name != "untracked.json"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--", *tracked], cwd=tmp_path, check=True)
    return tmp_path


_FOLDER = str(RECORD_DIR)


@pytest.mark.parametrize(
    ("record", "fault"),
    [
        ("", "no record"),
        ("/etc/hosts", "record path is absolute"),
        ("C:\\Users\\x\\good.json", "record path is absolute"),
        (f"{_FOLDER}/../../../../../etc/hosts", "record path leaves its folder"),
        ("outside/good.json", f"record is outside {RECORD_DIR}"),
        (f"{_FOLDER}/untracked.json", "record is not tracked by git"),
        (f"{_FOLDER}/missing.json", "record is not tracked by git"),
        (f"{_FOLDER}/empty.json", "record is empty"),
        (f"{_FOLDER}/notjson.json", "record is not JSON"),
        (f"{_FOLDER}/good.json", ""),
    ],
)
def test_a_compared_row_needs_a_tracked_json_record(repo: Path, record: str, fault: str) -> None:
    row = _row(status=Coverage.COMPARED, host="", record=record)
    assert row_faults(row, repo) == ([fault] if fault else [])
