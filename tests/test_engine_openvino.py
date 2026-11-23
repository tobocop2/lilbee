"""The OpenVINO adapter: gate, per-process device, missing graphics runtime, proof rows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lilbee.catalog import runnable
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import engine_manifest, engine_store, host_caps
from lilbee.providers.fleet.engines import choice, proof
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.llama import _WHEEL_VERSION
from lilbee.providers.fleet.engines.openvino import (
    GPU_MISSING,
    NPU_MISSING,
    OPENVINO,
    OpenvinoAdapter,
    parse_devices,
)
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._published import render_table, wheel_name

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"
BINARY = Path("/engines/bin/llama-server")
TWO_GPU_LIST = (
    Path(__file__).parent
    / "conformance"
    / "fixtures"
    / "engines"
    / "openvino-two-gpu.list-devices.log"
).read_text(encoding="utf-8")
ONE_GPU_LIST = (
    "Available devices:\n"
    "  OPENVINO0: GGML_OPENVINO_DEVICE=CPU - cpu (1 MiB, 1 MiB free)\n"
    "  OPENVINO1: GGML_OPENVINO_DEVICE=GPU (selected) - Intel UHD (iGPU) (1 MiB, 1 MiB free)\n"
)
INTEL_AND_NVIDIA_LIST = (
    "Available devices:\n"
    "  OPENVINO0: GGML_OPENVINO_DEVICE=CPU - cpu (1 MiB, 1 MiB free)\n"
    "  OPENVINO1: GGML_OPENVINO_DEVICE=GPU.0 - NVIDIA GeForce RTX 4050 (dGPU) (1 MiB, 1 MiB free)\n"
    "  OPENVINO2: GGML_OPENVINO_DEVICE=GPU.1 - Intel(R) Arc(TM) A380 (dGPU) (1 MiB, 1 MiB free)\n"
)
MANY_INTEL_LIST = (
    "Available devices:\n"
    "  OPENVINO0: GGML_OPENVINO_DEVICE=GPU.10 - Intel(R) Arc(TM) A (dGPU) (1 MiB, 1 MiB free)\n"
    "  OPENVINO1: GGML_OPENVINO_DEVICE=GPU.2 - Intel(R) Arc(TM) B (dGPU) (1 MiB, 1 MiB free)\n"
)
LOOKALIKE_LIST = (
    "Available devices:\n"
    "  OPENVINO0: GGML_OPENVINO_DEVICE=CPU - cpu (1 MiB, 1 MiB free)\n"
    "  OPENVINO1: GGML_OPENVINO_DEVICE=GPUX - Intel(R) thing (1 MiB, 1 MiB free)\n"
)


def _host(name: str) -> HostCaps:
    data = json.loads((FIXTURES / f"{name}.accel.json").read_text(encoding="utf-8"))
    accel = data["accel"]
    return host_caps.collect(
        system=data["system"],
        machine=data["machine"],
        cpu_info={},
        nvidia=None,
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
        OpenvinoAdapter, "_listed", staticmethod(lambda _b: parse_devices(ONE_GPU_LIST))
    )


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenvinoAdapter, "_listed", staticmethod(lambda _b: parse_devices("")))


def test_a1_a_cpu_target_sets_the_cpu_device_for_that_process(both_devices: None) -> None:
    argv = _argv(RoleTarget.CPU)
    assert argv[:3] == ["/usr/bin/env", "GGML_OPENVINO_DEVICE=CPU", str(BINARY)]
    assert argv[argv.index("--n-gpu-layers") + 1] == "0"


def test_a1_a_gpu_target_sets_the_gpu_device_and_keeps_the_llama_form(both_devices: None) -> None:
    argv = _argv(RoleTarget.GPU)
    assert argv[:3] == ["/usr/bin/env", "GGML_OPENVINO_DEVICE=GPU", str(BINARY)]
    assert argv[argv.index("--model") + 1] == str(Path("/models/m.gguf"))


def test_a1_the_device_is_not_in_the_shared_environment() -> None:
    assert OPENVINO.runtime_env() == {}


INTEL = 0x8086
NVIDIA = 0x10DE


def _linux(display: tuple[int, ...] = (), accel: list[tuple[str, str]] | None = None) -> HostCaps:
    return host_caps.collect(
        system="Linux",
        machine="x86_64",
        cpu_info={},
        nvidia=None,
        accel_devices=accel,
        display_vendors=display,
    )


def test_gate_an_intel_npu_host_is_offered() -> None:
    gate = OPENVINO.host_gate(_host("intel-core-ultra-linux"))
    assert (gate.ok, gate.known, gate.reason) == (True, True, "")


def test_gate_an_intel_gpu_host_is_offered() -> None:
    gate = OPENVINO.host_gate(_linux(display=(INTEL,), accel=[]))
    assert (gate.ok, gate.known, gate.reason) == (True, True, "")


def test_gate_a_host_with_only_another_vendor_gpu_is_refused() -> None:
    gate = OPENVINO.host_gate(_linux(display=(NVIDIA,), accel=[]))
    assert not gate.ok
    assert gate.reason == (
        "OpenVINO is offered on computers with an Intel GPU or an Intel NPU, "
        "and this computer has neither."
    )


def test_gate_a_host_that_lists_no_graphics_card_is_unknown_not_refused() -> None:
    gate = OPENVINO.host_gate(_host("no-accel-linux"))
    assert (gate.ok, gate.known) == (True, False)
    assert "could not be read" in gate.reason


def test_gate_an_unread_npu_list_beside_another_vendor_gpu_is_unknown() -> None:
    gate = OPENVINO.host_gate(_linux(display=(NVIDIA,), accel=None))
    assert (gate.ok, gate.known) == (True, False)


def test_gate_an_amd_npu_host_is_refused_when_it_has_no_intel_graphics() -> None:
    gate = OPENVINO.host_gate(_linux(display=(0x1002,), accel=[("0x1022", "0x17f0")]))
    assert not gate.ok


def test_s12_a_wrong_platform_is_refused_before_any_stage() -> None:
    for name in ("mac-arm64", "windows-amd64"):
        gate = OPENVINO.host_gate(_host(name))
        assert not gate.ok
        assert "not offered" in gate.reason


def test_s12_an_unread_platform_is_unknown() -> None:
    gate = OPENVINO.host_gate(
        host_caps.collect(system="Plan9", machine="mips", cpu_info={}, nvidia=None)
    )
    assert gate.known is False


def test_s4_a_gpu_target_without_the_runtime_refuses_in_user_words(cpu_only: None) -> None:
    with pytest.raises(ProviderError) as raised:
        _argv(RoleTarget.GPU)
    assert str(raised.value) == GPU_MISSING
    assert "Intel graphics only" in GPU_MISSING
    assert "Intel compute runtime" in GPU_MISSING
    assert "does not switch to the CPU" in GPU_MISSING


def test_s4_a_cpu_target_still_launches_without_the_runtime(cpu_only: None) -> None:
    assert _argv(RoleTarget.CPU)[1] == "GGML_OPENVINO_DEVICE=CPU"


def test_s4_the_targets_follow_the_device_list(cpu_only: None) -> None:
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}


def test_s4_both_devices_give_both_targets(both_devices: None) -> None:
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}


# Written from the line format of the recorded lists, not recorded on a Core Ultra machine.
# The live proof is owed on a Core Ultra host.
NPU_LIST = ONE_GPU_LIST + (
    "  OPENVINO2: GGML_OPENVINO_DEVICE=NPU - Intel(R) AI Boost (1 MiB, 1 MiB free)\n"
)


def test_npu_a_listed_npu_gives_the_npu_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_list(monkeypatch, NPU_LIST)
    assert OPENVINO.device_targets(BINARY) == {
        RoleTarget.CPU,
        RoleTarget.GPU,
        RoleTarget.NPU,
    }


def test_npu_a_list_without_an_npu_gives_no_npu_target(both_devices: None) -> None:
    assert RoleTarget.NPU not in OPENVINO.device_targets(BINARY)


def test_npu_an_npu_target_sets_the_npu_device_one_slot_and_a_small_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_list(monkeypatch, NPU_LIST)
    request = replace(_request(RoleTarget.NPU), slots=4, ctx_per_slot=8192)
    argv = OPENVINO.launch_argv(OPENVINO.role_spec(WorkerRole.CHAT, None, None), request)
    assert argv[:3] == ["/usr/bin/env", "GGML_OPENVINO_DEVICE=NPU", str(BINARY)]
    assert argv[argv.index("--parallel") + 1] == "1"
    assert argv[argv.index("--ctx-size") + 1] == "512"


def test_npu_a_context_below_the_cap_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_list(monkeypatch, NPU_LIST)
    request = replace(_request(RoleTarget.NPU), ctx_per_slot=256)
    argv = OPENVINO.launch_argv(OPENVINO.role_spec(WorkerRole.CHAT, None, None), request)
    assert argv[argv.index("--ctx-size") + 1] == "256"


def test_npu_a_target_without_an_npu_refuses_in_user_words(both_devices: None) -> None:
    with pytest.raises(ProviderError) as raised:
        _argv(RoleTarget.NPU)
    assert str(raised.value) == NPU_MISSING


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


def test_proof_the_npu_and_arc_are_not_run_so_auto_never_picks_openvino() -> None:
    assert proof.experimental_devices("openvino") == ("Intel Arc graphics", "Intel NPU")
    assert not proof.is_proven("openvino")


def test_s12_a_pack_for_the_engine_is_not_published_yet() -> None:
    assert OPENVINO.pin_record()["platforms"] == {}
    assert OPENVINO.pin_record()["sha256"] is None


_OPENVINO_CELL = "linux-x86_64-openvino"
# The Intel term of the RHEL 8 archive. The PyPI wheel the pack ships is Apache-2.0.
_INTEL_RUNTIME_TERM = "LicenseRef-Intel-OpenVINO-Distribution-License"
# The terms of the files the pack ships: llama.cpp and its vendored files (MIT, ISC, MIT-0,
# Unlicense, BSD-3-Clause, BSD-2-Clause), OpenVINO, oneTBB and oneDNN (Apache-2.0), hwloc
# (BSD-3-Clause) and the OpenCL loader (BSD-2-Clause).
_PACK_EXPRESSION = (
    "MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND MIT-0 AND Unlicense"
)


def _publish_llama_cells(monkeypatch: pytest.MonkeyPatch, cells: dict[str, str]) -> None:
    """Publish a llama.cpp wheel per cell at the version of the llama.cpp packs, on Linux x86_64."""
    rows = [
        (wheel_name(EngineName.LLAMA, _WHEEL_VERSION, key, "2.34"), sha, 1)
        for key, sha in cells.items()
    ]
    monkeypatch.setattr(engine_manifest, "MANIFESTS", render_table(rows))
    monkeypatch.setattr(runnable, "session_host", lambda: _host("intel-core-ultra-linux"))


def test_pack_the_openvino_cell_of_the_llama_cpp_pack_is_the_engine_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_llama_cells(monkeypatch, {_OPENVINO_CELL: "a" * 64})
    pin = OPENVINO.pin_record()
    assert list(pin["platforms"]) == [_OPENVINO_CELL]
    assert pin["sha256"] == "a" * 64
    assert pin["platforms"][_OPENVINO_CELL]["min_os"] == "2.34"


def test_pack_the_version_is_the_version_of_the_llama_cpp_packs() -> None:
    assert OPENVINO.pin_record()["version"] == _WHEEL_VERSION


def test_pack_a_vulkan_or_cpu_llama_cpp_pack_is_not_the_openvino_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_llama_cells(
        monkeypatch, {"linux-x86_64-vulkan": "b" * 64, "linux-x86_64-cpu": "c" * 64}
    )
    pin = OPENVINO.pin_record()
    assert (pin["platforms"], pin["sha256"]) == ({}, None)


def test_licence_the_pack_declares_the_terms_of_the_files_it_ships() -> None:
    (expression,) = OPENVINO.pin_record()["licenses"]
    assert expression == _PACK_EXPRESSION
    assert _INTEL_RUNTIME_TERM not in expression


def test_licence_a_published_pack_passes_the_licence_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_llama_cells(monkeypatch, {_OPENVINO_CELL: "a" * 64})
    lock = choice.pin_lock(OPENVINO.pin_record())
    engine_store._check_licences(lock)
    assert engine_store.licence_permitted(_PACK_EXPRESSION) is True


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.OPENVINO] is OPENVINO


def _with_list(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    monkeypatch.setattr(OpenvinoAdapter, "_listed", staticmethod(lambda _b: parse_devices(output)))


def test_the_two_gpu_list_keeps_both_full_names() -> None:
    assert [d.name for d in parse_devices(TWO_GPU_LIST)] == ["CPU", "GPU.0", "GPU.1"]


def test_a_host_with_no_intel_gpu_reports_the_gpu_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_list(monkeypatch, TWO_GPU_LIST)
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}
    with pytest.raises(ProviderError) as raised:
        _argv(RoleTarget.GPU)
    assert str(raised.value) == GPU_MISSING


def test_an_intel_gpu_next_to_an_nvidia_card_is_the_one_launched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_list(monkeypatch, INTEL_AND_NVIDIA_LIST)
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}
    assert _argv(RoleTarget.GPU)[1] == "GGML_OPENVINO_DEVICE=GPU.1"


@pytest.mark.parametrize(
    "description",
    [
        "Intel(R) UHD Graphics 630 (iGPU)",
        "Intel(R) Arc(TM) A770 Graphics (dGPU)",
        "Intel(R) Iris(R) Xe Graphics (iGPU)",
    ],
)
def test_real_intel_descriptions_are_offered(
    monkeypatch: pytest.MonkeyPatch, description: str
) -> None:
    _with_list(
        monkeypatch,
        f"  OPENVINO0: GGML_OPENVINO_DEVICE=GPU.0 - {description} (1 MiB, 1 MiB free)\n",
    )
    assert _argv(RoleTarget.GPU)[1] == "GGML_OPENVINO_DEVICE=GPU.0"


def test_a_card_that_only_mentions_intel_is_not_an_intel_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_list(
        monkeypatch,
        "  OPENVINO0: GGML_OPENVINO_DEVICE=GPU.0 - Foo (Intel-compatible) (1 MiB, 1 MiB free)\n",
    )
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}


def test_the_lowest_numbered_intel_gpu_wins_and_ten_follows_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_list(monkeypatch, MANY_INTEL_LIST)
    assert _argv(RoleTarget.GPU)[1] == "GGML_OPENVINO_DEVICE=GPU.2"


def test_a_name_that_only_starts_with_gpu_is_not_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_list(monkeypatch, LOOKALIKE_LIST)
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU}


def test_one_plain_gpu_still_launches_as_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_list(monkeypatch, ONE_GPU_LIST)
    assert _argv(RoleTarget.GPU)[1] == "GGML_OPENVINO_DEVICE=GPU"
    assert OPENVINO.device_targets(BINARY) == {RoleTarget.CPU, RoleTarget.GPU}


_PARTIAL_LIST = "Available devices:\n  OPENVINO0: GGML_OPENVINO_DEVICE=CPU (selected) - x\n"


def _probe_exits(
    monkeypatch: pytest.MonkeyPatch, output: str, code: int
) -> list[dict[str, object]]:
    """Fake the probe run; the returned list collects the keyword arguments of each run."""
    seen: list[dict[str, object]] = []

    def fake(_argv: list[str], **kw: object) -> tuple[str, int]:
        seen.append(kw)
        return output, code

    monkeypatch.setattr("lilbee.providers.fleet.engines.openvino.run_bounded", fake)
    return seen


def _expected(how: str) -> str:
    return (
        f"The device probe ({BINARY.name} --list-devices) {how}, so its device list "
        "may be incomplete and the engine cannot be planned from it. A GPU driver fault, "
        "the out-of-memory killer or a service manager can end the probe.\n"
        f"The probe reported: {_PARTIAL_LIST.strip()}"
    )


def test_a_listing_that_exits_non_zero_raises_with_the_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_exits(monkeypatch, _PARTIAL_LIST, 3)
    with pytest.raises(ProviderError) as raised:
        OPENVINO.device_targets(BINARY)
    assert str(raised.value) == _expected("exited with status 3")
    assert raised.value.kind is ProviderErrorKind.SERVER
    assert raised.value.provider == "llama-server"


def test_a_listing_killed_by_a_signal_raises_with_the_signal_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Python on Windows names SIGTERM, but not SIGKILL.
    _probe_exits(monkeypatch, _PARTIAL_LIST, -15)
    with pytest.raises(ProviderError) as raised:
        OPENVINO.device_targets(BINARY)
    assert str(raised.value) == _expected("was stopped by signal 15 (SIGTERM)")


def test_a_listing_names_an_unknown_signal_as_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _probe_exits(monkeypatch, "", -250)
    with pytest.raises(ProviderError, match=r"signal 250 \(unknown signal\)") as raised:
        OPENVINO.device_targets(BINARY)
    assert str(raised.value).endswith("The probe reported: (nothing)")


def test_the_listing_probe_is_bounded_and_merges_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _probe_exits(monkeypatch, _PARTIAL_LIST, 0)
    OPENVINO.device_targets(BINARY)
    assert seen == [{"timeout_s": 60.0, "kill_wait_s": 5.0, "merge_stderr": True}]


def test_a_gpu_launch_does_not_plan_from_a_listing_that_crashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _probe_exits(monkeypatch, _PARTIAL_LIST, 139)
    with pytest.raises(ProviderError, match=r"exited with status 139"):
        _argv(RoleTarget.GPU)
