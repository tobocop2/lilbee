"""An engine with no record for a device is experimental on every surface; `auto` skips it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.catalog.types import ModelFormat
from lilbee.cli import app as cli_app
from lilbee.cli.app import console as cli_console
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.cli.tui.screens.settings_engines import _build_label
from lilbee.providers.fleet.engine_manifest import cell_key
from lilbee.providers.fleet.engines import choice, proof
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests.test_engine_install import ENGINE, VERSION, _isolated, good, release, root  # noqa: F401

DEVICE = "Intel NPU"
CLAUSE = f", experimental: not run on {DEVICE}"
PLAIN_LINE = f"{ENGINE.value} {VERSION} (cpu)"
BUILD_LINE = f"{PLAIN_LINE}{CLAUSE}"
ABSENT_LINE = f"{ENGINE.value} not installed{CLAUSE}"
ROUTES = [routes.engine_builds_route, routes.engine_install_route]
MAC = collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None, mac_ver="15.2")
MAC_CELL = "macos-arm64-metal"
LINUX = collect(system="Linux", machine="x86_64", cpu_info={}, nvidia=None)


def _row(engine: str, cell: str, label: str, status: str) -> dict[str, Any]:
    return {"engine": engine, "cell": cell, "device_label": label, "status": status}


@pytest.fixture
def plant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Point the coverage table at a file of the given rows for one test."""

    def _plant(*rows: dict[str, Any]) -> None:
        planted = tmp_path / "coverage.json"
        planted.write_text(json.dumps({"rows": list(rows)}), encoding="utf-8")
        monkeypatch.setattr(proof, "COVERAGE_FILE", planted)
        proof.coverage.cache_clear()

    yield _plant
    proof.coverage.cache_clear()


@pytest.fixture
def wheel(good: Path) -> Path:  # noqa: F811  -- fixture imported from the install tests
    return good


@pytest.fixture
def unproven(plant: Any) -> None:
    plant(_row(ENGINE.value, _host_cell(), DEVICE, "deferred"))


def _cli(*argv: str) -> Any:
    return CliRunner().invoke(cli_app, ["engine", *argv])


def _cli_json(*argv: str) -> dict[str, Any]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return json.loads(result.output.strip().splitlines()[-1])


def test_the_clause_is_empty_for_no_device_and_names_every_other() -> None:
    assert proof.experimental_clause(()) == ""
    assert proof.experimental_clause(("Intel NPU",)) == ", experimental: not run on Intel NPU"
    assert (
        proof.experimental_clause(("Intel NPU", "AMD NPU"))
        == ", experimental: not run on Intel NPU and AMD NPU"
    )


def test_a_row_is_experimental_until_it_has_a_record(plant: Any) -> None:
    plant(
        _row("e", "c1", "CPU", "compared"),
        _row("e", "c1", "NVIDIA GPU", "recorded"),
        _row("e", "c1", "Intel NPU", "excluded"),
        _row("e", "c2", "Intel NPU", "deferred"),
        _row("e", "c2", "Apple GPU", "deferred"),
    )
    assert proof.experimental_devices("e", "c1") == ("Intel NPU",)
    assert proof.experimental_devices("e") == ("Intel NPU", "Apple GPU")
    assert proof.experimental_devices("e", "c3") == ("c3",)
    assert proof.experimental_devices("other", "c1") == ("c1",)
    assert proof.experimental_devices("other") == ()
    assert not proof.is_proven("e")
    assert proof.is_proven("other")


def test_the_shipped_table_marks_the_engines_without_a_record() -> None:
    assert proof.experimental_devices("openvino", "linux-x86_64-openvino") == (
        "Intel Arc graphics",
        "Intel NPU",
    )
    assert proof.experimental_devices("llama.cpp", "linux-x86_64-cpu") == ()
    assert proof.experimental_devices("llama.cpp", "linux-x86_64-cuda") == ()
    assert proof.experimental_devices("mlx", MAC_CELL) == ()
    assert proof.experimental_devices("llama.cpp", MAC_CELL) == ()
    assert proof.experimental_devices("mistral.rs", MAC_CELL) == ("Apple GPU",)
    assert proof.experimental_devices("tei", MAC_CELL) == ("Apple GPU",)


def test_the_note_says_what_experimental_means_and_where_to_report() -> None:
    assert "without the real device" in proof.EXPERIMENTAL_NOTE
    assert "https://github.com/tobocop2/lilbee/issues" in proof.EXPERIMENTAL_NOTE


def test_l1_a_missing_engine_and_an_installed_build_carry_the_same_sentence(
    unproven: None, wheel: Path
) -> None:
    listed = _cli("list").output
    assert ABSENT_LINE in listed
    assert proof.EXPERIMENTAL_NOTE in listed.replace("\n", " ")
    built = engine_service.install(ENGINE.value, wheel)
    assert built.label == BUILD_LINE
    assert BUILD_LINE in _cli("list").output
    folder_line = engine_service.FOLDER_LINE.format(folder=engine_service.engines_folder())
    assert run_engine_command("list").text == f"{BUILD_LINE}\n{folder_line}"


def test_l1_each_list_entry_stays_one_line_on_a_narrow_terminal(
    unproven: None, wheel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_console, "width", 40)
    assert ABSENT_LINE in _cli("list").output.splitlines()
    engine_service.install(ENGINE.value, wheel)
    assert BUILD_LINE in _cli("list").output.splitlines()


def test_l1_the_settings_row_carries_the_same_sentence(unproven: None) -> None:
    installed = engine_service.EngineState.INSTALLED
    row = engine_service.EngineRow(ENGINE.value, installed, VERSION, "cpu")
    assert _build_label(row) == BUILD_LINE


def test_l1_install_prints_the_same_sentence_on_the_cli_and_the_slash_command(
    unproven: None, wheel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli("install", ENGINE.value, "--wheel", str(wheel))
    assert f"Installed engine {BUILD_LINE}." in result.output
    monkeypatch.setattr(
        engine_service,
        "install",
        lambda name, wheel=None, **_: engine_service.InstalledBuild(ENGINE.value, VERSION, "cpu"),
    )
    assert BUILD_LINE in run_engine_command(f"install {ENGINE.value}").text


def test_l2_every_json_row_has_not_run_on(unproven: None, wheel: Path) -> None:
    engine_service.install(ENGINE.value, wheel)
    listed = _cli_json("list")
    rows = listed["builds"]
    assert rows == [
        {
            "engine": ENGINE.value,
            "version": VERSION,
            "backend": "cpu",
            "not_run_on": [DEVICE],
            "differs_from_pin": False,
        }
    ]
    assert listed["experimental"] == {ENGINE.value: [DEVICE]}
    with create_test_client(ROUTES, debug=False) as client:
        rest = client.get("/api/engines").json()
    assert (rest["builds"], rest["experimental"]) == (rows, {ENGINE.value: [DEVICE]})
    mcp = mcp_server.engine_list_tool()
    assert (mcp["builds"], mcp["experimental"]) == (rows, {ENGINE.value: [DEVICE]})


def _host_cell() -> str:
    return cell_key(runnable.session_host(), "cpu") or "linux-x86_64-cpu"


def test_l2_a_host_cell_missing_from_the_table_is_marked(plant: Any, wheel: Path) -> None:
    plant(_row(ENGINE.value, "some-other-cell", "CPU", "compared"))
    engine_service.install(ENGINE.value, wheel)
    cell = _host_cell()
    marked = f"{ENGINE.value} {VERSION} (cpu), experimental: not run on {cell}"
    assert marked in _cli("list").output
    listed = _cli_json("list")
    assert (listed["builds"][0]["not_run_on"], listed["experimental"]) == (
        [cell],
        {ENGINE.value: [cell]},
    )
    with create_test_client(ROUTES, debug=False) as client:
        assert client.get("/api/engines").json()["experimental"] == {ENGINE.value: [cell]}


def test_l2_a_proven_build_lists_an_empty_not_run_on(plant: Any, wheel: Path) -> None:
    plant(_row(ENGINE.value, _host_cell(), "CPU", "compared"))
    engine_service.install(ENGINE.value, wheel)
    listed = _cli_json("list")
    assert (listed["builds"][0]["not_run_on"], listed["experimental"]) == ([], {})
    assert engine_service.builds()[0].label == f"{ENGINE.value} {VERSION} (cpu)"
    assert proof.EXPERIMENTAL_NOTE not in _cli("list").output.replace("\n", " ")


def test_l2_cli_install_json_carries_not_run_on(unproven: None, wheel: Path) -> None:
    out = _cli_json("install", ENGINE.value, "--wheel", str(wheel))
    assert out["not_run_on"] == [DEVICE]
    assert out["installed"] == PLAIN_LINE


def _unrecorded_devices(engine: EngineName, backend: str) -> list[str]:
    """The devices of one shipped cell that have no record, read from the coverage file itself."""
    rows = json.loads(proof.COVERAGE_FILE.read_text(encoding="utf-8"))["rows"]
    cell = cell_key(LINUX, backend)
    return list(
        dict.fromkeys(
            row["device_label"]
            for row in rows
            if row["engine"] == engine.value
            and row["cell"] == cell
            and row["status"] not in proof.HAS_RECORD
        )
    )


@pytest.mark.parametrize(
    ("engine", "backend"),
    [
        (EngineName.LLAMA, "cpu"),
        (EngineName.LLAMA, "rocm"),
        (EngineName.LLAMA, "vulkan"),
        (EngineName.OPENVINO, "openvino"),
        (EngineName.OVMS, "openvino"),
        (EngineName.LEMONADE, "npu"),
    ],
)
def test_l2_cli_install_json_keeps_the_label_plain_on_the_shipped_table(
    engine: EngineName, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_run = _unrecorded_devices(engine, backend)
    built = engine_service.InstalledBuild(engine.value, VERSION, backend)
    monkeypatch.setattr(runnable, "session_host", lambda: LINUX)
    monkeypatch.setattr(engine_service, "install", lambda name, wheel=None, **_: built)
    out = _cli_json("install", engine.value)
    assert out["installed"] == f"{engine.value} {VERSION} ({backend})"
    assert out["not_run_on"] == not_run
    text = _cli("install", engine.value).output.replace("\n", " ")
    assert f"Installed engine {built.label}." in text
    assert (proof.experimental_clause(tuple(not_run)) in built.label) and bool(not_run) == (
        built.label != built.plain_label
    )


def test_l2_cli_remove_json_names_an_experimental_build_plain(unproven: None, wheel: Path) -> None:
    assert engine_service.install(ENGINE.value, wheel).label == BUILD_LINE
    out = _cli_json("remove", ENGINE.value, VERSION, "cpu", "--yes")
    assert out["removed"] == PLAIN_LINE


def test_l2_the_install_response_carries_not_run_on(
    unproven: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = engine_service.InstalledBuild(ENGINE.value, VERSION, "cpu")
    monkeypatch.setattr(engine_service, "install", lambda name, wheel=None: built)
    with create_test_client(ROUTES, debug=False) as client:
        rest = client.post("/api/engines/install", json={"engine": ENGINE.value}).json()
    assert rest["not_run_on"] == [DEVICE]
    assert mcp_server.engine_install_tool(ENGINE.value)["not_run_on"] == [DEVICE]


def _pick(installed: dict[EngineName, frozenset[ModelFormat]]) -> EngineName | None:
    return choice._default(ModelFormat.MLX, lambda: MAC, installed, None)  # type: ignore[arg-type]


def test_l3_auto_skips_an_engine_with_a_device_not_run(plant: Any) -> None:
    installed = {
        EngineName.MLX: frozenset({ModelFormat.MLX}),
        EngineName.LLAMA: frozenset({ModelFormat.MLX}),
    }
    plant(_row("mlx", MAC_CELL, "Apple GPU", "recorded"))
    assert _pick(installed) is EngineName.MLX
    plant(_row("mlx", MAC_CELL, "Apple GPU", "deferred"))
    assert _pick(installed) is EngineName.LLAMA


def test_l3_auto_still_picks_the_bundled_llama_cpp_while_it_is_experimental(plant: Any) -> None:
    plant(_row("llama.cpp", MAC_CELL, "Apple GPU", "deferred"))
    assert not proof.is_proven("llama.cpp")
    assert _pick({EngineName.LLAMA: frozenset({ModelFormat.MLX})}) is EngineName.LLAMA
