"""The coverage table lists every cell lilbee ships, and no cell that nothing ships."""

from __future__ import annotations

import itertools
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from lilbee.providers.fleet.engine_manifest import MANIFESTS, cell_key
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName, UserInstalled
from lilbee.providers.fleet.engines.choice import ADAPTERS, PinError, pin_lock
from lilbee.providers.fleet.host_caps import Feature, HostArch, HostCaps, HostOS, Npu, NpuVendor
from tests.conformance.compare_record import FAKE_DIR, Step, install_pin
from tests.conformance.engine_coverage import (
    COVERAGE_ROWS,
    INFINITY,
    LINUX_CPU,
    PRESET_CELL,
    Coverage,
    CoverageRow,
    Device,
    Route,
)

ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = Path(__file__).parent / "fixtures" / "presets"
RECORD_DIR = PurePosixPath("tests/conformance/fixtures/compare")
RECORDING_DIR = PurePosixPath("tests/conformance/fixtures/engines")
# The devices a cell whose backend part names no single device runs on.
MULTI_DEVICE_BACKENDS = {
    "openvino": {Device.CPU, Device.INTEL_IGPU, Device.INTEL_ARC, Device.NPU},
}

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


def path_fault(record: str, folder: PurePosixPath = RECORD_DIR) -> str:
    """Why *record* is not a path under *folder*; empty when it is."""
    path = PurePosixPath(record)
    if not record:
        return "no record"
    if path.is_absolute() or PureWindowsPath(record).anchor:
        return "record path is absolute"
    if ".." in path.parts:
        return "record path leaves its folder"
    if FAKE_DIR in path.parts:
        return "record is a fake"
    return "" if path.is_relative_to(folder) else f"record is outside {folder}"


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


def record_fault(record: str, root: Path = ROOT, folder: PurePosixPath = RECORD_DIR) -> str:
    """Why *record* cannot back a row with a record in *folder*; empty when it can."""
    if fault := path_fault(record, folder):
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
    if not row.device_label:
        faults.append("no device label")
    if row.status is Coverage.DEFERRED and not row.host:
        faults.append("deferred with no host")
    if row.status is Coverage.EXCLUDED and not row.owner_decision:
        faults.append("excluded with no owner decision")
    if row.status is Coverage.COMPARED:
        if not row.steps:
            faults.append("compared with no steps")
        if fault := record_fault(row.record, root):
            faults.append(fault)
    if row.status is Coverage.RECORDED and (fault := record_fault(row.record, root, RECORDING_DIR)):
        faults.append(fault)
    return faults


def _row(**changes: Any) -> CoverageRow:
    base = CoverageRow(
        "e", "c", Device.CPU, Coverage.DEFERRED, Route.PRESET, "why", host="h", device_label="CPU"
    )
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
    rows = [*COVERAGE_ROWS, _row(engine="llama.cpp", cell="linux-aarch64-vulkan")]
    assert stale(shipped, rows) == [("llama.cpp", "linux-aarch64-vulkan")]


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


def _backend(cell: str) -> str:
    return cell.split("-", 2)[2]


def cell_device(cell: str) -> Device:
    """The device a pack cell key runs on, from its backend part."""
    return next(d for d in Device if _backend(cell).startswith(d.value))


def cell_devices(cell: str) -> set[Device]:
    """Every device a cell runs on: the multi-device set of its backend, else its one device."""
    return MULTI_DEVICE_BACKENDS.get(_backend(cell)) or {cell_device(cell)}


def missing_devices(rows: Iterable[CoverageRow]) -> list[tuple[str, str, Device]]:
    """The devices of a multi-device cell that no row of its engine names."""
    named = {(r.engine, r.cell, r.device) for r in rows}
    cells = {
        (r.engine, r.cell)
        for r in rows
        if r.cell != PRESET_CELL and _backend(r.cell) in MULTI_DEVICE_BACKENDS
    }
    return sorted(
        (engine, cell, device)
        for engine, cell in cells
        for device in cell_devices(cell)
        if (engine, cell, device) not in named
    )


def unknown_devices(rows: Iterable[CoverageRow]) -> list[tuple[str, str, Device]]:
    """The rows whose device the cell's own backend does not name."""
    return [
        (r.engine, r.cell, r.device)
        for r in rows
        if r.cell != PRESET_CELL and r.device not in cell_devices(r.cell)
    ]


def test_every_device_of_a_multi_device_cell_has_a_row() -> None:
    assert missing_devices(COVERAGE_ROWS) == []


def test_a_removed_device_row_is_named() -> None:
    rows = [r for r in COVERAGE_ROWS if (r.engine, r.device) != ("openvino", Device.INTEL_ARC)]
    assert missing_devices(rows) == [("openvino", "linux-x86_64-openvino", Device.INTEL_ARC)]


def test_no_row_names_a_device_that_its_cell_does_not_know() -> None:
    assert unknown_devices(COVERAGE_ROWS) == []


def test_a_row_with_an_unknown_device_is_named() -> None:
    rows = [*COVERAGE_ROWS, _row(engine="lemonade", cell="linux-x86_64-npu", device=Device.CPU)]
    assert unknown_devices(rows) == [("lemonade", "linux-x86_64-npu", Device.CPU)]
    cuda_on_cpu = _row(engine="tei", cell="linux-x86_64-cpu", device=Device.CUDA)
    assert unknown_devices([cuda_on_cpu]) == [("tei", "linux-x86_64-cpu", Device.CUDA)]


def test_cell_device_reads_the_backend_part() -> None:
    assert cell_device("linux-x86_64-cuda128-sm89") is Device.CUDA
    assert cell_device("macos-arm64-metal") is Device.METAL
    assert cell_devices("linux-x86_64-openvino") == MULTI_DEVICE_BACKENDS["openvino"]


def test_the_intel_igpu_record_does_not_clear_arc() -> None:
    openvino = {r.device: r.status for r in COVERAGE_ROWS if r.engine == "openvino"}
    assert openvino[Device.INTEL_IGPU] is Coverage.RECORDED
    assert openvino[Device.INTEL_ARC] is Coverage.DEFERRED


def route_of(
    row: CoverageRow, manifests: Manifests, adapters: Mapping[EngineName, EngineAdapter]
) -> Route:
    """The route the code gives a row's engine: a preset, a pack, a user install or neither."""
    if row.cell == PRESET_CELL:
        return Route.PRESET
    if (row.engine, row.cell) in pack_cells(manifests):
        return Route.PACK
    adapter = adapters[EngineName(row.engine)]
    return Route.USER_INSTALLED if isinstance(adapter, UserInstalled) else Route.UNPUBLISHED


def route_faults(
    rows: Iterable[CoverageRow],
    manifests: Manifests = MANIFESTS,
    adapters: Mapping[EngineName, EngineAdapter] = ADAPTERS,
) -> list[tuple[str, str, Route]]:
    """The rows whose route differs from the route that the code gives them."""
    return [
        (r.engine, r.cell, r.route) for r in rows if r.route is not route_of(r, manifests, adapters)
    ]


def install_refusal(row: CoverageRow, monkeypatch: pytest.MonkeyPatch) -> str:
    """The error name of the install path's pin check for a row's engine; empty if it accepts."""
    os_word, arch_word, _ = row.cell.split("-", 2)
    host = _best_host(HostOS(os_word), HostArch(arch_word))
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: host)
    try:
        pin_lock(install_pin(row.engine))
    except PinError as exc:
        return type(exc).__name__
    return ""


def test_each_route_agrees_with_the_code() -> None:
    assert COVERAGE_ROWS
    assert route_faults(COVERAGE_ROWS) == []


def test_every_route_value_is_used() -> None:
    assert {r.route for r in COVERAGE_ROWS} == set(Route)


def test_every_pack_cell_has_a_pack_row() -> None:
    packed = {(r.engine, r.cell) for r in COVERAGE_ROWS if r.route is Route.PACK}
    assert packed == pack_cells(MANIFESTS)


def test_a_pack_route_on_a_cell_outside_the_pack_table_is_named() -> None:
    rows = [_row(engine="mlx", cell="macos-arm64-metal", route=Route.PACK)]
    assert route_faults(rows) == [("mlx", "macos-arm64-metal", Route.PACK)]


def test_a_pack_cell_with_another_route_is_named() -> None:
    rows = [_row(engine="tei", cell="linux-x86_64-cpu", route=Route.UNPUBLISHED)]
    assert route_faults(rows) == [("tei", "linux-x86_64-cpu", Route.UNPUBLISHED)]


def test_an_engine_that_lilbee_finds_is_not_an_unpublished_one() -> None:
    row = _row(engine="openvino", cell="linux-x86_64-openvino", route=Route.USER_INSTALLED)
    assert route_faults([row]) == [("openvino", "linux-x86_64-openvino", Route.USER_INSTALLED)]


def test_an_engine_that_has_no_pack_yet_is_not_a_user_installed_one() -> None:
    row = _row(engine="lemonade", cell="linux-x86_64-npu", route=Route.UNPUBLISHED)
    assert route_faults([row]) == [("lemonade", "linux-x86_64-npu", Route.UNPUBLISHED)]


def test_the_preset_route_is_only_for_preset_rows() -> None:
    assert {r.cell for r in COVERAGE_ROWS if r.route is Route.PRESET} == {PRESET_CELL}
    assert {r.route for r in COVERAGE_ROWS if r.cell == PRESET_CELL} == {Route.PRESET}


def test_the_install_path_refuses_each_row_that_has_no_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = [r for r in COVERAGE_ROWS if r.route in (Route.UNPUBLISHED, Route.USER_INSTALLED)]
    assert {r.engine for r in refused} == {"mlx", "openvino", "ovms", "lemonade"}
    assert {install_refusal(r, monkeypatch) for r in refused} == {"NoPublishedPackError"}


def test_the_install_path_accepts_each_pack_row_it_can_run_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted = [r for r in COVERAGE_ROWS if r.route is Route.PACK and r.cell == LINUX_CPU]
    assert {r.engine for r in accepted} == {"ik_llama.cpp", "llama.cpp", "mistral.rs", "tei"}
    assert {install_refusal(r, monkeypatch) for r in accepted} == {""}


def test_a_row_that_claims_a_pack_for_an_engine_with_none_is_refused_by_the_install_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row(engine="mlx", cell="macos-arm64-metal", route=Route.PACK)
    assert install_refusal(row, monkeypatch) == "NoPublishedPackError"


def test_a_compared_row_must_claim_steps() -> None:
    assert row_faults(_row(status=Coverage.COMPARED, host="", record="x")) == [
        "compared with no steps",
        f"record is outside {RECORD_DIR}",
    ]


def test_a_compared_row_may_not_name_a_fake_record() -> None:
    fake = f"{RECORD_DIR}/_fake/llama.cpp-linux-x86_64-cpu/1.0.0.json"
    row = _row(status=Coverage.COMPARED, host="", record=fake, steps=(Step.TOOL,))
    assert row_faults(row) == ["record is a fake"]


def test_a_deferred_row_without_a_host_is_a_fault() -> None:
    assert row_faults(_row(host="")) == ["deferred with no host"]


def test_an_excluded_row_without_an_owner_decision_is_a_fault() -> None:
    assert row_faults(_row(status=Coverage.EXCLUDED, host="")) == [
        "excluded with no owner decision"
    ]


def test_a_row_without_a_reason_is_a_fault() -> None:
    assert row_faults(_row(reason="")) == ["no reason"]


def test_a_row_without_a_device_label_is_a_fault() -> None:
    assert row_faults(_row(device_label="")) == ["no device label"]


def test_a_recorded_row_needs_a_recording_under_the_engines_folder() -> None:
    compare = "tests/conformance/fixtures/compare/llama.cpp-linux-x86_64-cpu/1.12032.1.json"
    assert row_faults(_row(status=Coverage.RECORDED, record="")) == ["no record"]
    assert row_faults(_row(status=Coverage.RECORDED, record=compare)) == [
        f"record is outside {RECORDING_DIR}"
    ]
    good = _row(status=Coverage.RECORDED, record=f"{RECORDING_DIR}/openvino-gpu.json")
    assert row_faults(good) == []


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
    row = _row(status=Coverage.COMPARED, host="", record=record, steps=(Step.TOOL,))
    assert row_faults(row, repo) == ([fault] if fault else [])


def _recordings_naming_a_cell() -> list[tuple[str, str, str]]:
    """The (file, engine, cell) of each recording whose version ends with the cell it ran on."""
    found = []
    for path in sorted((ROOT / RECORDING_DIR).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        engine, version = data.get("engine"), str(data.get("version", ""))
        if engine and version.split(" ")[-1] in {row.cell for row in COVERAGE_ROWS}:
            found.append((f"{RECORDING_DIR}/{path.name}", engine, version.split(" ")[-1]))
    return found


def test_the_recordings_that_name_a_cell_are_found() -> None:
    names = {file.rsplit("/", 1)[-1] for file, _, _ in _recordings_naming_a_cell()}
    assert {"llama-vulkan-linux.json", "mistralrs-cuda-sm89.json", "ovms-cpu.json"} <= names


@pytest.mark.parametrize(("file", "engine", "cell"), _recordings_naming_a_cell())
def test_a_recording_that_names_its_cell_is_the_record_of_that_row(
    file: str, engine: str, cell: str
) -> None:
    rows = [row for row in COVERAGE_ROWS if row.engine == engine and row.cell == cell]
    assert any(row.status is Coverage.RECORDED and row.record == file for row in rows), (
        f"{file} is in git for {engine} on {cell}, and no row of that cell reads recorded with it"
    )
