"""An engine that has not run on a device says so on every surface, and `auto` never picks it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog.types import ModelFormat
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.providers.fleet.engines import choice, proof
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.proof import HardwareProof
from lilbee.providers.fleet.host_caps import collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests.test_engine_install import ENGINE, VERSION, _isolated, good, release, root  # noqa: F401

DEVICE = "Intel NPU"
BUILD_LINE = f"{ENGINE.value} {VERSION} (cpu), not run on {DEVICE}"
ABSENT_LINE = f"{ENGINE.value} not installed, not run on {DEVICE}"
ROUTES = [routes.engine_builds_route, routes.engine_install_route]


@pytest.fixture
def wheel(good: Path) -> Path:  # noqa: F811  -- fixture imported from the install tests
    return good


@pytest.fixture
def unproven(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proof, "HARDWARE_PROOF", {ENGINE.value: {DEVICE: HardwareProof.NOT_RUN}})


def _cli(*argv: str) -> Any:
    return CliRunner().invoke(cli_app, ["engine", *argv])


def _cli_json(*argv: str) -> dict[str, Any]:
    result = CliRunner().invoke(cli_app, ["--json", "engine", *argv])
    return json.loads(result.output.strip().splitlines()[-1])


def test_the_clause_is_empty_for_no_device_and_names_every_other() -> None:
    assert proof.not_run_clause(()) == ""
    assert proof.not_run_clause(("Intel NPU",)) == ", not run on Intel NPU"
    assert proof.not_run_clause(("Intel NPU", "AMD NPU")) == ", not run on Intel NPU and AMD NPU"


def test_l1_a_missing_engine_and_an_installed_build_carry_the_same_sentence(
    unproven: None, wheel: Path
) -> None:
    assert ABSENT_LINE in _cli("list").output
    built = engine_service.install(ENGINE.value, wheel)
    assert built.label == BUILD_LINE
    assert BUILD_LINE in _cli("list").output
    assert run_engine_command("list").text == BUILD_LINE


def test_l1_install_prints_the_same_sentence_on_the_cli_and_the_slash_command(
    unproven: None, wheel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli("install", ENGINE.value, "--wheel", str(wheel))
    assert f"Installed engine {BUILD_LINE}." in result.output
    monkeypatch.setattr(
        engine_service,
        "install",
        lambda name, wheel=None, **_: engine_service.InstalledBuild(
            ENGINE.value, VERSION, "cpu", not_run_on=(DEVICE,)
        ),
    )
    assert BUILD_LINE in run_engine_command(f"install {ENGINE.value}").text


def test_l2_every_json_row_has_not_run_on(unproven: None, wheel: Path) -> None:
    engine_service.install(ENGINE.value, wheel)
    rows = _cli_json("list")["builds"]
    assert rows == [
        {"engine": ENGINE.value, "version": VERSION, "backend": "cpu", "not_run_on": [DEVICE]}
    ]
    with create_test_client(ROUTES, debug=False) as client:
        assert client.get("/api/engines").json()["builds"] == rows
    assert mcp_server.engine_list_tool()["builds"] == rows


def test_l2_a_proven_build_lists_an_empty_not_run_on(wheel: Path) -> None:
    engine_service.install(ENGINE.value, wheel)
    assert _cli_json("list")["builds"][0]["not_run_on"] == []
    assert engine_service.builds()[0].label == f"{ENGINE.value} {VERSION} (cpu)"


def test_l2_cli_install_json_carries_not_run_on(unproven: None, wheel: Path) -> None:
    out = _cli_json("install", ENGINE.value, "--wheel", str(wheel))
    assert out["not_run_on"] == [DEVICE]
    assert out["installed"] == BUILD_LINE


def test_l2_the_install_response_carries_not_run_on(
    unproven: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = engine_service.InstalledBuild(ENGINE.value, VERSION, "cpu", not_run_on=(DEVICE,))
    monkeypatch.setattr(engine_service, "install", lambda name, wheel=None: built)
    with create_test_client(ROUTES, debug=False) as client:
        rest = client.post("/api/engines/install", json={"engine": ENGINE.value}).json()
    assert rest["not_run_on"] == [DEVICE]
    assert mcp_server.engine_install_tool(ENGINE.value)["not_run_on"] == [DEVICE]


def test_l3_auto_skips_an_engine_with_a_device_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    mac = collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None, mac_ver="15.2")
    installed = {
        EngineName.MLX: frozenset({ModelFormat.MLX}),
        EngineName.LLAMA: frozenset({ModelFormat.MLX}),
    }

    def pick() -> EngineName | None:
        return choice._default(ModelFormat.MLX, lambda: mac, installed, None)  # type: ignore[arg-type]

    assert pick() is EngineName.MLX
    monkeypatch.setattr(proof, "HARDWARE_PROOF", {"mlx": {"Apple GPU": HardwareProof.NOT_RUN}})
    assert pick() is EngineName.LLAMA
