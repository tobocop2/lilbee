"""The OpenVINO adapter: gate, per-process device, missing graphics runtime, proof rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import choice, proof
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.openvino import GPU_MISSING, OPENVINO, OpenvinoAdapter
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.roles import RoleTarget, WorkerRole

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"
BINARY = Path("/engines/bin/llama-server")


def _host(name: str) -> HostCaps:
    data = json.loads((FIXTURES / f"{name}.accel.json").read_text(encoding="utf-8"))
    accel = data["accel"]
    return host_caps.collect(
        system=data["system"],
        machine=data["machine"],
        cpu_info={},
        nvml=None,
        accel_devices=None if accel is None else [tuple(pair) for pair in accel],
    )


def _request(target: RoleTarget) -> LaunchRequest:
    return LaunchRequest(
        binary=BINARY,
        model_path=Path("/models/m.gguf"),
        devices=(),
        n_gpu_layers=99,
        slots=1,
        ctx_per_slot=2048,
        target=target,
    )


def _argv(target: RoleTarget) -> list[str]:
    return OPENVINO.launch_argv(OPENVINO.role_spec(WorkerRole.CHAT, None, None), _request(target))


@pytest.fixture
def both_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        OpenvinoAdapter, "_listed", staticmethod(lambda _b: frozenset({"CPU", "GPU"}))
    )


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenvinoAdapter, "_listed", staticmethod(lambda _b: frozenset({"CPU"})))


def test_a1_a_cpu_target_sets_the_cpu_device_for_that_process(both_devices: None) -> None:
    argv = _argv(RoleTarget.CPU)
    assert argv[:3] == ["/usr/bin/env", "GGML_OPENVINO_DEVICE=CPU", str(BINARY)]
    assert argv[argv.index("--n-gpu-layers") + 1] == "0"


def test_a1_a_gpu_target_sets_the_gpu_device_and_keeps_the_llama_form(both_devices: None) -> None:
    argv = _argv(RoleTarget.GPU)
    assert argv[:3] == ["/usr/bin/env", "GGML_OPENVINO_DEVICE=GPU", str(BINARY)]
    assert argv[argv.index("--model") + 1] == "/models/m.gguf"


def test_a1_the_device_is_not_in_the_shared_environment() -> None:
    assert OPENVINO.runtime_env() == {}


def test_a2_an_intel_npu_host_passes_and_says_the_npu_stays_unused() -> None:
    gate = OPENVINO.host_gate(_host("intel-core-ultra-linux"))
    assert (gate.ok, gate.known) == (True, True)
    assert "NPU" in gate.reason


def test_a2_a_host_with_no_npu_passes_with_no_remark() -> None:
    gate = OPENVINO.host_gate(_host("no-accel-linux"))
    assert (gate.ok, gate.reason) == (True, "")


def test_s12_a_wrong_platform_is_refused_before_any_stage() -> None:
    for name in ("mac-arm64", "windows-amd64"):
        gate = OPENVINO.host_gate(_host(name))
        assert not gate.ok
        assert "not offered" in gate.reason


def test_s12_an_unread_platform_is_unknown() -> None:
    gate = OPENVINO.host_gate(
        host_caps.collect(system="Plan9", machine="mips", cpu_info={}, nvml=None)
    )
    assert gate.known is False


def test_s4_a_gpu_target_without_the_runtime_refuses_in_user_words(cpu_only: None) -> None:
    with pytest.raises(ProviderError) as raised:
        _argv(RoleTarget.GPU)
    assert str(raised.value) == GPU_MISSING
    assert "Intel compute runtime" in GPU_MISSING
    assert "does not switch to the CPU" in GPU_MISSING


def test_s4_a_cpu_target_still_launches_without_the_runtime(cpu_only: None) -> None:
    assert _argv(RoleTarget.CPU)[1] == "GGML_OPENVINO_DEVICE=CPU"


def test_s4_the_targets_follow_the_device_list(cpu_only: None) -> None:
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}


def test_s4_both_devices_give_both_targets(both_devices: None) -> None:
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}


def test_s4_the_listing_runs_the_binary_with_list_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake(argv: list[str], **_kw: object) -> tuple[str, int]:
        seen.append(argv)
        return "Available devices:\n  OPENVINO0: GGML_OPENVINO_DEVICE=CPU (selected) - x\n", 0

    monkeypatch.setattr("lilbee.providers.fleet.engines.openvino.run_bounded", fake)
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}
    assert seen == [[str(BINARY), "--list-devices"]]


def test_s9_openvino_serves_chat_only() -> None:
    assert OPENVINO.served_roles == {WorkerRole.CHAT}
    for role in (WorkerRole.EMBED, WorkerRole.RERANK):
        with pytest.raises(ProviderError):
            OPENVINO.role_spec(role, None, None)


def test_proof_the_npu_is_not_run_so_auto_never_picks_openvino() -> None:
    assert proof.not_run_on("openvino") == ("Intel NPU",)
    assert not proof.is_proven("openvino")


def test_s12_a_pack_for_the_engine_is_not_published_yet() -> None:
    assert OPENVINO.pin_record()["platforms"] == {}
    assert OPENVINO.pin_record()["sha256"] is None


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.OPENVINO] is OPENVINO
