"""The engine that auto picks on each hardware class of the host fixtures, every engine installed.

The hosts come from tests/fixtures/host_caps: a probe file is a reading of a real machine, an accel
file is written from the PCI id tables. Every shipped adapter counts as installed, so the rule has
every engine to pick from and only its own order and the host gates decide.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    AUTO,
    choose_engine,
    host_answer,
    serving_engines,
)
from lilbee.providers.fleet.engines.proof import is_proven
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


HOSTS: dict[str, HostCaps] = {
    "mac-m1pro": _recorded("mac-m1pro"),
    "home-xeon-1070ti": _recorded("home-xeon-1070ti"),
    "gcp-epyc-nogpu": _recorded("gcp-epyc-nogpu"),
    "mac-arm64": _documented("mac-arm64"),
    "windows-amd64": _documented("windows-amd64"),
    "no-accel-linux": _documented("no-accel-linux"),
    "intel-core-ultra-linux": _documented("intel-core-ultra-linux"),
    "amd-xdna2-linux": _documented("amd-xdna2-linux"),
    "amd-generation-unread-linux": _documented("amd-generation-unread-linux"),
    "qualcomm-datacentre-linux": _documented("qualcomm-datacentre-linux"),
}

EVERY_ENGINE = {name: adapter.formats for name, adapter in ADAPTERS.items()}

# An engine the user must choose: it is not in the package and has a device with no record.
OPT_IN = frozenset(
    name for name, adapter in ADAPTERS.items() if not adapter.bundled and not is_proven(name)
)

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
def test_auto_never_picks_an_opt_in_engine_on_any_hardware_class(host_name: str) -> None:
    host = HOSTS[host_name]
    for role, fmt in itertools.product(ROLES, ModelFormat):
        assert _auto(host, role, fmt) not in OPT_IN, (role, fmt)


LLAMA_FOR_EVERY_ROLE = {role.value: {"gguf": "llama.cpp"} for role in WorkerRole}


def test_the_default_on_the_mac_puts_mlx_models_on_mlx() -> None:
    assert _defaults(HOSTS["mac-m1pro"]) == {
        "embed": {"gguf": "llama.cpp", "mlx": "mlx"},
        "rerank": {"gguf": "llama.cpp", "mlx": "mlx"},
        "chat": {"gguf": "llama.cpp", "mlx": "mlx"},
        "vision": {"gguf": "llama.cpp"},
    }


def test_the_default_on_the_desktop_with_a_gtx_1070_ti_is_llama() -> None:
    assert _defaults(HOSTS["home-xeon-1070ti"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_a_cloud_cpu_with_no_gpu_is_llama() -> None:
    assert _defaults(HOSTS["gcp-epyc-nogpu"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_a_mac_with_an_unread_macos_version_skips_mlx() -> None:
    assert _defaults(HOSTS["mac-arm64"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_windows_is_llama() -> None:
    assert _defaults(HOSTS["windows-amd64"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_linux_with_no_accelerator_is_llama() -> None:
    assert _defaults(HOSTS["no-accel-linux"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_an_intel_npu_host_is_llama() -> None:
    assert _defaults(HOSTS["intel-core-ultra-linux"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_an_amd_xdna2_npu_host_is_llama() -> None:
    assert _defaults(HOSTS["amd-xdna2-linux"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_an_amd_npu_of_an_unread_generation_is_llama() -> None:
    assert _defaults(HOSTS["amd-generation-unread-linux"]) == LLAMA_FOR_EVERY_ROLE


def test_the_default_on_a_qualcomm_datacentre_card_is_llama() -> None:
    assert _defaults(HOSTS["qualcomm-datacentre-linux"]) == LLAMA_FOR_EVERY_ROLE


def test_every_hardware_class_has_a_default_test() -> None:
    names = {p.name.split(".")[0] for p in FIXTURES.glob("*.json")}
    assert names == set(HOSTS)
