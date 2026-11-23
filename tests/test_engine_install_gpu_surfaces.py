"""A fake Vulkan or ROCm host installs its llama.cpp pack through the CLI, REST and MCP."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import lilbee.mcp_server as mcp_server
from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.cli import app as cli_app
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import binary as binary_mod
from lilbee.providers.fleet import engine_manifest, first_use
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.llama import _WHEEL_VERSION
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.server.routes import engines as routes
from tests._litestar_test_client import create_test_client
from tests._published import render_table, wheel_name
from tests.test_first_use_engine import make_pack

AMD = 0x1002
INTEL = 0x8086
ENGINE = EngineName.LLAMA.value
NUMA_ADVICE = "For the ROCm build, install the NUMA library (libnuma1 or numactl-libs). "
LOADER_ADVICE = "For the Vulkan build, install the Vulkan loader (libvulkan1 or vulkan-loader). "


def _linux(**facts: Any) -> HostCaps:
    return collect(system="Linux", machine="x86_64", cpu_info={"flags": ["avx2"]}, **facts)


ROCM_HOST = _linux(
    nvidia=None, display_vendors=(AMD,), amd_gfx=("gfx1100",), libnuma=True, vulkan_loader=True
)
VULKAN_HOST = _linux(nvidia=None, display_vendors=(INTEL,), vulkan_loader=True)
ROCM_HOST_NO_LIBRARIES = _linux(nvidia=None, display_vendors=(AMD,), amd_gfx=("gfx1100",))


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshot = cfg.model_copy()
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    monkeypatch.setenv("LILBEE_MCP_ENGINES_ENABLED", "true")
    cfg.data_root = tmp_path
    cfg.models_dir = tmp_path / "models"
    cfg.llama_server_path = ""
    cfg.allow_http_engines = True
    cfg.mcp_engines_enabled = True
    cfg.preferred_engine = EngineSetting.AUTO
    cfg.engine_choices = None
    monkeypatch.setattr(binary_mod, "_bundled_tool", lambda _tool: None)
    monkeypatch.setattr(binary_mod.shutil, "which", lambda _name: None)
    yield
    mcp_server.set_http_mounted(False)
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _on_host(
    host: HostCaps, cell: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Make *host* the computer, publish a pack for *cell*, and return the cells fetched."""
    wheel = tmp_path / "pack" / "llama.whl"
    wheel.parent.mkdir()
    sha = make_pack(wheel)
    table = render_table([(wheel_name(EngineName.LLAMA, _WHEEL_VERSION, cell), sha, 1)])
    monkeypatch.setattr(engine_manifest, "MANIFESTS", table)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: host)
    fetched: list[str] = []

    def fake(_lock: Any, fetched_cell: str, *, dest: Path, **_signals: Any) -> Path:
        fetched.append(fetched_cell)
        dest.mkdir(parents=True, exist_ok=True)
        copy = dest / wheel.name
        copy.write_bytes(wheel.read_bytes())
        return copy

    monkeypatch.setattr(first_use, "fetch_frozen", fake)
    return fetched


def _cli() -> str:
    result = CliRunner().invoke(cli_app, ["--json", "engine", "install", ENGINE])
    assert result.exit_code == 0, result.output
    built = json.loads(result.stdout.strip().splitlines()[-1])["installed"]
    return str(built).rsplit("(", 1)[1].rstrip(")")


def _rest() -> str:
    with create_test_client([routes.engine_install_route], debug=False) as client:
        res = client.post("/api/engines/install", json={"engine": ENGINE})
    assert res.status_code == 200, res.text
    return str(res.json()["backend"])


def _mcp() -> str:
    body = mcp_server.engine_install_tool(ENGINE)
    assert "error" not in body, body
    return str(body["backend"])


SURFACES = [
    pytest.param(_cli, id="cli"),
    pytest.param(_rest, id="rest"),
    pytest.param(_mcp, id="mcp"),
]


@pytest.mark.parametrize(
    ("host", "backend"),
    [
        pytest.param(ROCM_HOST, "rocm", id="rocm"),
        pytest.param(VULKAN_HOST, "vulkan", id="vulkan"),
    ],
)
@pytest.mark.parametrize("surface", SURFACES)
def test_a_gpu_host_installs_its_gpu_pack_on_every_surface(
    surface: Callable[[], str],
    host: HostCaps,
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetched = _on_host(host, f"linux-x86_64-{backend}", tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=first_use.__name__):
        assert surface() == backend
    assert fetched == [f"linux-x86_64-{backend}"]
    assert caplog.records == []


@pytest.mark.parametrize("surface", SURFACES)
def test_a_rocm_card_without_the_libraries_installs_the_cpu_pack_and_the_warning_names_both(
    surface: Callable[[], str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetched = _on_host(ROCM_HOST_NO_LIBRARIES, "linux-x86_64-cpu", tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger=first_use.__name__):
        assert surface() == "cpu"
    assert fetched == ["linux-x86_64-cpu"]
    assert caplog.messages == [
        "This computer has an AMD GPU, and no llama.cpp pack can use it here. "
        f"lilbee installs the CPU build. {NUMA_ADVICE}{LOADER_ADVICE}To use your own build, "
        "set LILBEE_LLAMA_SERVER_PATH to your llama-server."
    ]
