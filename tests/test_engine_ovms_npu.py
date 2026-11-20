"""The OpenVINO Model Server adapter on an Intel NPU: when the target is offered and its argv.

The NPU lists come from the recorded host fixtures. No NPU run is recorded yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.roles import RoleTarget, WorkerRole

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"
BINARY = Path("/engines/bin/ovms")
INTEL = 0x8086


def _host(name: str, display: tuple[int, ...] = (INTEL,)) -> HostCaps:
    data = json.loads((FIXTURES / f"{name}.accel.json").read_text(encoding="utf-8"))
    accel = data["accel"]
    return host_caps.collect(
        system=data["system"],
        machine=data["machine"],
        cpu_info={},
        nvidia=None,
        accel_devices=None if accel is None else [tuple(pair) for pair in accel],
        display_vendors=display,
    )


def _targets(monkeypatch: pytest.MonkeyPatch, host: HostCaps) -> frozenset[RoleTarget]:
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    return OVMS.device_targets(BINARY)


def _argv(role: WorkerRole, target: RoleTarget) -> list[str]:
    request = LaunchRequest(
        binary=BINARY,
        model_path=Path("/models/bge-reranker-base-int8-ov"),
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
        target=target,
    )
    return OVMS.launch_argv(OVMS.role_spec(role, None, None), request)


def test_an_intel_npu_host_is_offered_the_npu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    targets = _targets(monkeypatch, _host("intel-core-ultra-linux"))
    assert targets == {RoleTarget.CPU, RoleTarget.GPU, RoleTarget.NPU}


def test_an_intel_npu_host_without_a_graphics_card_is_offered_cpu_and_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets(monkeypatch, _host("intel-core-ultra-linux", display=(0x10DE,)))
    assert targets == {RoleTarget.CPU, RoleTarget.NPU}


@pytest.mark.parametrize("name", ["no-accel-linux", "amd-xdna2-linux"])
def test_a_host_without_an_intel_npu_is_not_offered_the_npu_target(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    assert RoleTarget.NPU not in _targets(monkeypatch, _host(name))


def test_an_unread_npu_list_is_not_offered_the_npu_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = host_caps.collect(
        system="Linux", machine="x86_64", cpu_info={}, nvidia=None, display_vendors=(INTEL,)
    )
    assert RoleTarget.NPU not in _targets(monkeypatch, host)


@pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
def test_the_npu_target_sets_the_npu_device(role: WorkerRole) -> None:
    argv = _argv(role, RoleTarget.NPU)
    assert argv[argv.index("--target_device") + 1] == "NPU"
    assert argv[argv.index("--task") + 1] == {"embed": "embeddings", "rerank": "rerank"}[role.value]
