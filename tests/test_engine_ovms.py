"""The OpenVINO Model Server adapter: gate, launch argv, routes and the rerank shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.catalog import runnable
from lilbee.catalog.types import ModelFormat
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import choice, proof
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RerankShape,
)
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.roles import RoleTarget, WorkerRole

BINARY = Path("/engines/bin/ovms")
MODEL = Path("/models/bge-base-en-v1.5-int8-ov")
INTEL = 0x8086
NVIDIA = 0x10DE


def _host(
    system: str = "Linux", machine: str = "x86_64", display: tuple[int, ...] = ()
) -> HostCaps:
    return host_caps.collect(
        system=system, machine=machine, cpu_info={}, nvidia=None, display_vendors=display
    )


def _request(target: RoleTarget = RoleTarget.CPU) -> LaunchRequest:
    return LaunchRequest(
        binary=BINARY,
        model_path=MODEL,
        devices=(),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=2048,
        target=target,
    )


def _argv(role: WorkerRole, target: RoleTarget = RoleTarget.CPU) -> list[str]:
    return OVMS.launch_argv(OVMS.role_spec(role, None, None), _request(target))


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.OVMS] is OVMS


def test_the_engine_loads_openvino_ir_and_serves_three_roles() -> None:
    assert OVMS.formats == {ModelFormat.OPENVINO}
    assert OVMS.served_roles == {WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK}


def test_the_http_port_is_rest_port_and_the_grpc_port_is_not_set() -> None:
    argv = _argv(WorkerRole.CHAT)
    assert argv[argv.index("--rest_port") + 1] == PORT_SLOT
    assert "--port" not in argv


def test_each_role_names_its_task() -> None:
    tasks = {
        role: (lambda a: a[a.index("--task") + 1])(_argv(role))
        for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)
    }
    assert tasks == {
        WorkerRole.CHAT: "text_generation",
        WorkerRole.EMBED: "embeddings",
        WorkerRole.RERANK: "rerank",
    }


def test_the_launch_binds_loopback_and_serves_the_one_model_name() -> None:
    argv = _argv(WorkerRole.EMBED)
    assert argv[0] == str(BINARY)
    assert argv[argv.index("--model_path") + 1] == str(MODEL)
    assert argv[argv.index("--model_name") + 1] == "default_model"
    assert argv[argv.index("--rest_bind_address") + 1] == "127.0.0.1"


def test_the_target_sets_the_device() -> None:
    cpu = _argv(WorkerRole.CHAT, RoleTarget.CPU)
    gpu = _argv(WorkerRole.CHAT, RoleTarget.GPU)
    assert cpu[cpu.index("--target_device") + 1] == "CPU"
    assert gpu[gpu.index("--target_device") + 1] == "GPU"


def test_a_role_the_engine_does_not_serve_is_refused() -> None:
    with pytest.raises(ProviderError):
        OVMS.role_spec(WorkerRole.VISION, None, None)


def test_the_gate_offers_linux_x86_64() -> None:
    gate = OVMS.host_gate(_host())
    assert (gate.ok, gate.known) == (True, True)


@pytest.mark.parametrize(("system", "machine"), [("Darwin", "arm64"), ("Windows", "AMD64")])
def test_the_gate_refuses_another_platform_in_user_words(system: str, machine: str) -> None:
    gate = OVMS.host_gate(_host(system, machine))
    assert not gate.ok
    assert gate.reason.startswith("OpenVINO Model Server is not offered on ")


def test_an_unread_platform_is_unknown() -> None:
    gate = OVMS.host_gate(
        host_caps.collect(system="Plan9", machine="mips", cpu_info={}, nvidia=None)
    )
    assert gate.known is False


def test_the_targets_follow_the_graphics_cards_of_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host(display=(INTEL,)))
    assert OVMS.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}
    monkeypatch.setattr(runnable, "session_host", lambda: _host(display=(NVIDIA,)))
    assert OVMS.device_targets(BINARY) == {RoleTarget.CPU}


def test_the_swap_entry_checks_the_ready_route_and_names_the_model() -> None:
    extras = OVMS.swap_extras(None)  # type: ignore[arg-type]
    assert extras == {"checkEndpoint": "/v2/health/ready", "useModelName": "default_model"}


def test_the_rerank_profile_uses_the_cohere_shape_under_v3() -> None:
    profile = OVMS.request_profile(WorkerRole.RERANK)
    assert (profile.rerank_path, profile.rerank_shape) == ("/v3/rerank", RerankShape.DOCUMENTS)
    assert profile.rerank_model == "default_model"
    assert profile.batch_limit == 100


def test_the_other_roles_have_no_rerank_route() -> None:
    assert OVMS.request_profile(WorkerRole.EMBED).rerank_path is None
    assert OVMS.request_profile(WorkerRole.CHAT).rerank_path is None


def test_a_pack_for_the_engine_is_not_published_yet() -> None:
    record = OVMS.pin_record()
    assert (record["platforms"], record["sha256"]) == ({}, None)


def test_the_binary_is_the_store_tool_or_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: _host())
    monkeypatch.setattr(choice, "store_tool", lambda pin, tool, host: BINARY)
    assert OVMS.binary() == BINARY
    assert OVMS.identity(BINARY) == f"ovms 2026.4.1 {BINARY}"
    monkeypatch.setattr(choice, "store_tool", lambda pin, tool, host: None)
    with pytest.raises(ProviderError, match="OpenVINO Model Server engine is not installed"):
        OVMS.binary()


def test_the_engine_has_no_optional_features_to_probe() -> None:
    assert OVMS.supports_architecture("anything") is True
    probe = OVMS.discover_devices(BINARY)
    assert probe.devices == []
    assert not any(
        (
            OVMS.check_readback(BINARY),
            OVMS.supports_fit_target(BINARY),
            OVMS.supports_shared_kv(BINARY),
        )
    )
    assert OVMS.model_meta(MODEL) is None
    assert OVMS.assert_loadable("repo", "file", None) is None  # type: ignore[arg-type]
    assert OVMS.runtime_env() == {}
    assert OVMS.target_env(RoleTarget.CPU, 1) == {}


def test_the_memory_estimate_charges_the_weights_of_the_ir_folder(tmp_path: Path) -> None:
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "openvino_model.bin").write_bytes(b"\0" * 4096)
    request = MemoryRequest(
        model_path=tmp_path,
        ctx=2048,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    estimate = OVMS.estimate_memory(request)
    assert estimate.device_bytes >= 4096


def test_the_pin_record_names_the_licences_of_the_linked_code() -> None:
    assert OVMS.pin_record()["licenses"] == [
        "Apache-2.0",
        "BSD-3-Clause",
        "GPL-2.0-only WITH GCC-exception-2.0",
    ]


def test_the_proof_table_lists_every_device_as_not_run() -> None:
    devices = ("Intel CPU", "Intel integrated graphics", "Intel NPU")
    assert proof.not_run_on("ovms") == devices
    assert not proof.is_proven("ovms")


def test_rerank_bodies_and_scores_follow_the_shape() -> None:
    profile = OVMS.request_profile(WorkerRole.RERANK)
    assert profile.rerank_body("q", ["a", "b"]) == {
        "query": "q",
        "documents": ["a", "b"],
        "top_n": 2,
        "model": "default_model",
    }
    payload = {
        "results": [{"index": 1, "relevance_score": 0.7}, {"index": 0, "relevance_score": 0.2}]
    }
    assert profile.rerank_scores(payload) == {0: 0.2, 1: 0.7}
