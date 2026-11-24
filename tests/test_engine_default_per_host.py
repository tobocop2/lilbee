"""The engine that auto picks on each hardware class of the host fixtures, every engine installed.

The table covers each of the four roles. The opt-in test also runs with no role.

The hosts come from tests/fixtures/host_caps: a probe file is a reading of a real machine, an accel
file is written from the PCI id tables. Every shipped adapter counts as installed, so the rule has
every engine to pick from and only its own order and the host gates decide.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines import proof
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    AUTO,
    choose_engine,
    host_answer,
    serving_engines,
)
from lilbee.providers.fleet.engines.proof import coverage
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.roles import WorkerRole

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"


def _recorded(name: str) -> HostCaps:
    """The host read on a real machine, with its macOS version when it has one."""
    probe = json.loads((FIXTURES / f"{name}.probe.json").read_text(encoding="utf-8"))
    cpu = {k: probe[k] for k in ("arch", "arch_string_raw", "brand_raw", "flags")}
    return collect(
        system=probe["system"],
        machine=probe["machine"],
        cpu_info=cpu,
        nvidia=probe.get("nvml"),
        mac_ver=probe["mac_ver"],
    )


def _documented(name: str) -> HostCaps:
    """The host written from the PCI id tables; its CPU and macOS version are not read."""
    data = json.loads((FIXTURES / f"{name}.accel.json").read_text(encoding="utf-8"))
    accel = data["accel"]
    return collect(
        system=data["system"],
        machine=data["machine"],
        cpu_info={},
        nvidia=None,
        accel_devices=None if accel is None else [tuple(pair) for pair in accel],
    )


def _load(fixture: Path) -> HostCaps:
    name = fixture.name.split(".")[0]
    return _recorded(name) if fixture.name.endswith(".probe.json") else _documented(name)


# One host for each fixture file, so a new fixture is a new host and needs a row in DEFAULTS.
HOSTS: dict[str, HostCaps] = {
    p.name.split(".")[0]: _load(p) for p in sorted(FIXTURES.glob("*.json"))
}

EVERY_ENGINE = {name: adapter.formats for name, adapter in ADAPTERS.items()}

# The product rule: auto picks llama.cpp, and mlx on Apple Silicon. Every other engine is opt-in.
# The set is written here, not read from the coverage table or from the code under test.
AUTO_PICKS = frozenset({EngineName.LLAMA, EngineName.MLX})
OPT_IN = frozenset(EngineName) - AUTO_PICKS
MLX_HOSTS = frozenset({"mac-m1pro"})

ROLES: tuple[WorkerRole | None, ...] = (*WorkerRole, None)


def _auto(host: HostCaps, role: WorkerRole | None, fmt: ModelFormat) -> EngineName | None:
    return choose_engine(
        fmt=fmt,
        host=lambda: host,
        installed=EVERY_ENGINE,
        preferred=AUTO,
        override=None,
        role=role,
    ).name


def _defaults(host: HostCaps) -> dict[str, dict[str, str]]:
    """The engine auto picks for each role, by format; a format with no pick is left out."""
    picks: dict[str, dict[str, str]] = {}
    for role, fmt in itertools.product(WorkerRole, ModelFormat):
        name = _auto(host, role, fmt)
        if name is not None:
            picks.setdefault(role.value, {})[fmt.value] = name.value
    return picks


def test_the_opt_in_engines_include_ik_llama() -> None:
    assert EngineName.IK_LLAMA in OPT_IN
    assert EngineName.LLAMA not in OPT_IN
    assert EngineName.MLX not in OPT_IN


def test_an_opt_in_engine_can_serve_on_some_hardware_class() -> None:
    """The guard below is not empty: an opt-in engine loads a format, serves a role, and runs."""
    servable = [
        (host_name, name)
        for host_name, host in HOSTS.items()
        for role, fmt in itertools.product(WorkerRole, ModelFormat)
        for name in serving_engines(fmt, role)
        if name in OPT_IN and host_answer(name, lambda host=host: host).ok
    ]
    assert ("gcp-epyc-nogpu", EngineName.IK_LLAMA) in servable


@pytest.mark.parametrize("host_name", sorted(HOSTS))
def test_auto_picks_only_llama_and_mlx_on_any_hardware_class(host_name: str) -> None:
    host = HOSTS[host_name]
    for role, fmt in itertools.product(ROLES, ModelFormat):
        pick = _auto(host, role, fmt)
        assert pick is None or pick in AUTO_PICKS, (role, fmt, pick)
        if pick is EngineName.MLX:
            assert host_name in MLX_HOSTS, (role, fmt)


def _without_a_record_for_mlx() -> tuple[Mapping[str, Any], ...]:
    """The coverage table with the one Apple GPU row of mlx turned into a device with no run."""
    return tuple(
        {**row, "status": "deferred"} if row["engine"] == EngineName.MLX else row
        for row in coverage()
    )


def test_auto_leaves_out_an_engine_that_has_no_record_on_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mlx is the one engine besides llama.cpp in the order; with no record it is left out."""
    host = HOSTS["mac-m1pro"]
    assert _auto(host, WorkerRole.CHAT, ModelFormat.MLX) is EngineName.MLX
    monkeypatch.setattr(proof, "coverage", _without_a_record_for_mlx)
    assert _auto(host, WorkerRole.CHAT, ModelFormat.MLX) is None
    assert _auto(host, WorkerRole.CHAT, ModelFormat.GGUF) is EngineName.LLAMA


LLAMA_FOR_EVERY_ROLE = {role.value: {"gguf": "llama.cpp"} for role in WorkerRole}


# The default for each role, by format, on each fixture file. Mac M1 Pro reads macOS, so mlx models
# go to mlx; mac-arm64 has no macOS version read, so it skips mlx. Every other class gets llama.cpp.
DEFAULTS: dict[str, dict[str, dict[str, str]]] = {
    "mac-m1pro": {
        "embed": {"gguf": "llama.cpp", "mlx": "mlx"},
        "rerank": {"gguf": "llama.cpp", "mlx": "mlx"},
        "chat": {"gguf": "llama.cpp", "mlx": "mlx"},
        "vision": {"gguf": "llama.cpp"},
    },
    "home-xeon-1070ti": LLAMA_FOR_EVERY_ROLE,
    "gcp-epyc-nogpu": LLAMA_FOR_EVERY_ROLE,
    "mac-arm64": LLAMA_FOR_EVERY_ROLE,
    "windows-amd64": LLAMA_FOR_EVERY_ROLE,
    "no-accel-linux": LLAMA_FOR_EVERY_ROLE,
    "intel-core-ultra-linux": LLAMA_FOR_EVERY_ROLE,
    "amd-xdna2-linux": LLAMA_FOR_EVERY_ROLE,
    "amd-generation-unread-linux": LLAMA_FOR_EVERY_ROLE,
    "qualcomm-datacentre-linux": LLAMA_FOR_EVERY_ROLE,
}


@pytest.mark.parametrize("fixture", sorted(p.name.split(".")[0] for p in FIXTURES.glob("*.json")))
def test_the_default_on_each_hardware_class(fixture: str) -> None:
    """A fixture with no row in DEFAULTS fails here; it does not pass by being left out."""
    assert fixture in DEFAULTS, f"no expected default for the fixture {fixture}"
    assert _defaults(HOSTS[fixture]) == DEFAULTS[fixture]


def test_every_row_of_the_defaults_table_is_a_fixture() -> None:
    assert set(DEFAULTS) == set(HOSTS)
