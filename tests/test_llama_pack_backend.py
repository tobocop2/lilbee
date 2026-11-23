"""The llama.cpp pack backend follows the host: metal, cuda with a NVIDIA driver, else cpu."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import Config, cfg
from lilbee.providers.fleet import engine_manifest, first_use
from lilbee.providers.fleet.engine_manifest import MANIFESTS
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import (
    _CUDA_DRIVER_FLOOR,
    _CUDA_MIN_COMPUTE,
    _UPSTREAM_TAG,
    _WHEEL_VERSION,
    LlamaCppAdapter,
    cuda_pack_gap,
    pack_backend,
)
from lilbee.providers.fleet.host_caps import HostCaps, collect
from tests._pack_workflow import (
    Cell,
    cells_below_the_record,
    pack_cells,
    pack_workflow,
)
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
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.messages == [cuda_pack_gap(host)]
    assert "compute capability 3.7" in caplog.messages[0]


def test_the_first_use_warning_is_silent_for_a_card_the_pack_serves(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = _nvidia("Linux", "x86_64", cards=((12, 0),))
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert caplog.records == []


def test_the_mixed_host_with_an_old_nvidia_card_and_an_amd_gpu_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = collect(
        system="Linux",
        machine="x86_64",
        cpu_info={"flags": ["avx2"]},
        nvidia={
            "driver": "550.135",
            "cuda_driver": 12040,
            "gpus": [{"name": "card0", "cc": [3, 7]}],
        },
        display_vendors=(0x1002, 0x10DE),
    )
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    with caplog.at_level("WARNING", logger=first_use.__name__):
        first_use.warn_if_gpu_unserved()
    assert len(caplog.records) == 1
    assert "compute capability 3.7" in caplog.text
    assert "AMD" not in caplog.text


_ROOT = Path(__file__).resolve().parents[1]
_PACK_LINUX_CUDA_BUILD = _ROOT / "tools/wheel-build/llama_cuda_manylinux.sh"
_PACK_CUDA_VERSION = re.compile(r"cuda(?:-|_dash=)(\d+)[.-](\d+)")
# At this tag, ggml-cuda's CMakeLists.txt defaults CUDA 12 without GGML_NATIVE to 50-virtual up.
_UPSTREAM_TAG_OF_THE_COMPUTE_FLOOR = "b12032"
_CUDA_ARCHITECTURE_OVERRIDES = ("CUDA_ARCHITECTURES", "CUDAARCHS", "GGML_NATIVE=ON")


def _pack_workflow_yaml() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(pack_workflow().read_text(encoding="utf-8")))


def _pack_matrix() -> list[dict[str, Any]]:
    return pack_cells("llama")


def _pack_build_cuda_versions() -> set[tuple[int, int]]:
    matrix = _pack_matrix()
    texts = [str(value) for cell in matrix if cell["backend"] == "cuda" for value in cell.values()]
    texts.append(_PACK_LINUX_CUDA_BUILD.read_text(encoding="utf-8"))
    return {(int(m[1]), int(m[2])) for text in texts for m in _PACK_CUDA_VERSION.finditer(text)}


def _uncommented(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_the_driver_floor_is_the_cuda_version_the_pack_is_built_with() -> None:
    assert _pack_build_cuda_versions() == {_CUDA_DRIVER_FLOOR}


def test_the_compute_floor_is_the_upstream_default_architecture_list_at_b12032() -> None:
    cells = _pack_workflow_yaml()
    assets = [cell["asset"] for cell in _pack_matrix() if "asset" in cell]
    linux_build = _uncommented(_PACK_LINUX_CUDA_BUILD.read_text(encoding="utf-8"))
    workflow = _uncommented(pack_workflow().read_text(encoding="utf-8"))
    assert cells["env"]["UPSTREAM_TAG"] == _UPSTREAM_TAG_OF_THE_COMPUTE_FLOOR
    assert assets
    assert all(f"llama-{_UPSTREAM_TAG_OF_THE_COMPUTE_FLOOR}-" in asset for asset in assets)
    assert "-DGGML_NATIVE=OFF" in linux_build
    assert [o for o in _CUDA_ARCHITECTURE_OVERRIDES if o in linux_build + workflow] == []
    assert _CUDA_MIN_COMPUTE == (5, 0)


def test_the_pinned_wheel_version_is_derived_from_the_upstream_tag() -> None:
    assert _UPSTREAM_TAG == "b12032"
    assert _WHEEL_VERSION == "1.12032.1"


def test_the_pinned_upstream_tag_is_the_tag_the_pack_workflow_builds() -> None:
    assert _pack_workflow_yaml()["env"]["UPSTREAM_TAG"] == _UPSTREAM_TAG


def test_the_pinned_wheel_version_is_the_version_the_pack_workflow_builds() -> None:
    assert _pack_workflow_yaml()["env"]["WHEEL_VERSION"] == _WHEEL_VERSION


def test_every_llama_pack_in_the_shipped_table_has_a_cell_in_the_pack_cells() -> None:
    shipped = set(MANIFESTS["llama.cpp"][_WHEEL_VERSION])
    built = {cell["cell"] for cell in _pack_matrix()}
    assert shipped
    assert shipped - built == set()


def test_every_shipped_llama_url_names_the_wheel_of_its_cell_apart_from_the_build_number() -> None:
    cells = {cell["cell"]: cell for cell in _pack_matrix()}
    shipped = MANIFESTS["llama.cpp"][_WHEEL_VERSION]
    assert shipped
    for key, row in shipped.items():
        cell = cells[key]
        built = re.sub(
            r"-\d+\.(\w+)-py3",
            rf"-{cell['build_number']}.\1-py3",
            row["url"].rsplit("/", 1)[1],
        )
        assert built == (
            f"lilbee_engine_llama_cpp-{_WHEEL_VERSION}-{cell['build_number']}.{cell['backend']}"
            f"-py3-none-{cell['platform']}.whl"
        ), key


def _llama_builds() -> list[Cell]:
    cells = _pack_matrix()
    assert cells
    return [(c["cell"], c["backend"], c["platform"], c["build_number"]) for c in cells]


def test_no_llama_cell_builds_a_number_below_a_published_wheel_of_its_name() -> None:
    """The publish step refuses an existing file name, and that keeps one name to one wheel."""
    assert cells_below_the_record("lilbee_engine_llama_cpp", _WHEEL_VERSION, _llama_builds()) == []


def test_the_windows_cells_write_and_check_the_runtime_notice_under_one_name() -> None:
    steps = _pack_workflow_yaml()["jobs"]["cell"]["steps"]
    noticed = [s for s in steps if _NOTICE_NAME in s.get("run", "")]
    assert len(noticed) == 2
    assert not _pack_workflow_yaml()["jobs"]["cell"].get("continue-on-error")
    for step in noticed:
        _assert_step_stops_on_failure(step)
    workflow = _uncommented("\n".join(step.get("run", "") for step in steps))
    notice = "NOTICE-MICROSOFT-VC-RUNTIME"
    bundle = [line for line in workflow.splitlines() if "bundle_windows_runtime.sh" in line]
    assert len(bundle) == 1
    assert f"packaging/engine-wheel/{notice} vs-licence.docx" in bundle[0]
    assert f'dist-info/licenses/{notice}$" wheel-listing.txt' in workflow


_NOTICE_NAME = "NOTICE-MICROSOFT-VC-RUNTIME"
_NON_FATAL_SHELL = re.compile(r"^\s*(set\s+\+\w*e|set\s+\+o\s+errexit|trap\b)", re.MULTILINE)


def _assert_step_stops_on_failure(step: dict[str, Any]) -> None:
    """Fail if the step can pass after a failed command."""
    assert not step.get("continue-on-error")
    assert not _NON_FATAL_SHELL.search(_uncommented(step["run"]))


def _enclosing_conditions(script: str, line_text: str) -> list[str]:
    """Return the `if` conditions that enclose the one script line holding `line_text`."""
    stack: list[str] = []
    hits: list[list[str]] = []
    for raw in script.splitlines():
        line = raw.strip()
        if line.startswith("if ") and line.endswith("; then"):
            stack.append(line[len("if ") : -len("; then")])
        elif line.startswith(("else", "elif ")):
            stack[-1] = f"not ({stack[-1]})"
        elif line == "fi":
            stack.pop()
        elif line_text in line:
            hits.append(list(stack))
    assert len(hits) == 1, line_text
    return hits[0]


def test_the_windows_cells_run_the_notice_check_under_the_windows_condition() -> None:
    steps = _pack_workflow_yaml()["jobs"]["cell"]["steps"]
    [start] = [s for s in steps if s.get("name") == "Start llama-server from the installed wheel"]
    windows_cells = [c for c in _pack_matrix() if c["cell"].startswith("windows-")]
    assert windows_cells
    assert all(c["runner"].startswith("windows-") for c in windows_cells)
    assert "if" not in start
    verify = "vs_distributable_terms.py --verify vs-licence.docx out/*.whl"
    assert _enclosing_conditions(start["run"], verify) == ['[ "${RUNNER_OS}" = Windows ]']


def test_the_notice_check_failure_fails_the_step_and_the_job() -> None:
    job = _pack_workflow_yaml()["jobs"]["cell"]
    step_name = "Start llama-server from the installed wheel"
    [start] = [s for s in job["steps"] if s.get("name") == step_name]
    verify = r"python -I tools/wheel-build/vs_distributable_terms\.py --verify vs-licence\.docx"
    verify += r"( --\S+)* out/\*\.whl"
    lines = [line.strip() for line in start["run"].splitlines()]
    at = [i for i, line in enumerate(lines) if "vs_distributable_terms.py --verify" in line]
    assert len(at) == 1
    before = lines[: at[0]]
    assert "continue-on-error" not in start
    assert "continue-on-error" not in job
    assert "shell" not in start
    assert any(re.match(r"set -[A-Za-z]*e", line) for line in before)
    assert any(re.match(r"set .*pipefail", line) for line in before)
    assert not [line for line in before if re.match(r"set\s+\+|trap\b", line)]
    assert not before[-1].endswith(("\\", "||", "&&", "|"))
    assert re.fullmatch(verify, lines[at[0]])


def test_the_pack_workflow_names_no_transient_branch() -> None:
    assert "fix/llama-pack-licences" not in pack_workflow().read_text(encoding="utf-8")
