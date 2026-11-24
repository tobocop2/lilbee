"""The OpenVINO Model Server adapter on an Intel NPU: when the target is offered and its argv.

The NPU lists come from the host fixtures, which are written from PCI id tables in the docs and
not recorded on a machine. No NPU run exists yet.
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


def _argv(
    role: WorkerRole,
    target: RoleTarget,
    ctx_per_slot: int = 512,
    model_path: Path = Path("/models/bge-reranker-base-int8-ov"),
) -> list[str]:
    request = LaunchRequest(
        binary=BINARY,
        model_path=model_path,
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=ctx_per_slot,
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


def test_the_npu_embed_launch_bounds_the_input() -> None:
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU, ctx_per_slot=648)
    assert argv[argv.index("--max_length") + 1] == "648"


def test_the_npu_embed_bound_follows_the_window() -> None:
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU, ctx_per_slot=256)
    assert argv[argv.index("--max_length") + 1] == "256"


@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_the_npu_bound_is_not_set_on_cpu_or_gpu(target: RoleTarget) -> None:
    assert "--max_length" not in _argv(WorkerRole.EMBED, target)


def test_the_npu_rerank_launch_has_no_bound() -> None:
    assert "--max_length" not in _argv(WorkerRole.RERANK, RoleTarget.NPU)


def _model_folder(tmp_path: Path, config: str | None) -> Path:
    if config is not None:
        (tmp_path / "config.json").write_text(config, encoding="utf-8")
    return tmp_path


def test_the_npu_embed_bound_stops_at_the_position_limit_of_the_model(tmp_path: Path) -> None:
    folder = _model_folder(tmp_path, json.dumps({"max_position_embeddings": 512}))
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU, ctx_per_slot=648, model_path=folder)
    assert argv[argv.index("--max_length") + 1] == "512"


def test_the_npu_embed_bound_keeps_a_window_below_the_position_limit(tmp_path: Path) -> None:
    folder = _model_folder(tmp_path, json.dumps({"max_position_embeddings": 8192}))
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU, ctx_per_slot=648, model_path=folder)
    assert argv[argv.index("--max_length") + 1] == "648"


@pytest.mark.parametrize(
    "config",
    [
        None,
        "not json",
        "[]",
        json.dumps({"hidden_size": 384}),
        json.dumps({"max_position_embeddings": "512"}),
        json.dumps({"max_position_embeddings": True}),
        json.dumps({"max_position_embeddings": 0}),
    ],
    ids=["no-file", "bad-json", "not-an-object", "no-key", "string", "bool", "zero"],
)
def test_the_npu_embed_bound_keeps_the_window_when_the_model_names_no_limit(
    tmp_path: Path, config: str | None
) -> None:
    folder = _model_folder(tmp_path, config)
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU, ctx_per_slot=648, model_path=folder)
    assert argv[argv.index("--max_length") + 1] == "648"


def test_the_npu_embed_launch_cuts_an_over_long_input() -> None:
    argv = _argv(WorkerRole.EMBED, RoleTarget.NPU)
    assert argv[argv.index("--truncate") + 1] == "true"


@pytest.mark.parametrize(
    ("role", "target"),
    [
        (WorkerRole.EMBED, RoleTarget.CPU),
        (WorkerRole.EMBED, RoleTarget.GPU),
        (WorkerRole.RERANK, RoleTarget.NPU),
    ],
)
def test_the_truncation_is_set_on_the_npu_embed_launch_only(
    role: WorkerRole, target: RoleTarget
) -> None:
    assert "--truncate" not in _argv(role, target)
