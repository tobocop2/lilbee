"""The Lemonade adapter: XDNA2 gate, fixed ports, backend download notice, proof rows."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import choice, lemonade, proof
from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.roles import WorkerRole

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"
BINARY = Path("/engines/bin/lemond")


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


def _launch() -> list[str]:
    request = LaunchRequest(
        binary=BINARY,
        model_path=Path("/models/m.gguf"),
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=2048,
    )
    return LEMONADE.launch_argv(LEMONADE.role_spec(WorkerRole.CHAT, None, None), request)


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")


def test_b1_an_xdna2_host_passes() -> None:
    gate = LEMONADE.host_gate(_host("amd-xdna2-linux"))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, True, True)


def test_b1_a_host_with_no_npu_is_refused_and_the_reason_names_xdna2() -> None:
    gate = LEMONADE.host_gate(_host("no-accel-linux"))
    assert not gate.ok
    assert "XDNA2" in gate.reason


def test_b1_an_older_xdna_npu_is_refused() -> None:
    caps = host_caps.collect(
        system="Linux",
        machine="x86_64",
        cpu_info={},
        nvml=None,
        accel_devices=[("0x1022", "0x1502")],
    )
    assert not LEMONADE.host_gate(caps).ok


def test_b1_an_unread_generation_is_unknown_and_not_auto_selectable() -> None:
    gate = LEMONADE.host_gate(_host("amd-generation-unread-linux"))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_b1_windows_is_not_offered() -> None:
    assert not LEMONADE.host_gate(_host("windows-amd64")).ok


def test_b2_the_memory_rule_is_a_fixed_reserve() -> None:
    assert LEMONADE.memory_kind is MemoryKind.FIXED_RESERVE
    assert LEMONADE.pin_record()["memory"]["rule"] == "fixed_reserve"


def test_s7_the_command_starts_lemond_with_its_own_data_folders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lemonade, "busy_ports", tuple)
    argv = _launch()
    cache, config = lemonade.data_dirs()
    assert argv[:2] == [str(BINARY), "--port"]
    assert PORT_SLOT in argv
    assert argv[-2:] == [str(cache), str(config)]
    assert "--no-broadcast" in argv


def test_ports_a_busy_port_refuses_the_launch_and_names_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lemonade, "port_is_busy", lambda port: port == 8002)
    with pytest.raises(ProviderError) as raised:
        _launch()
    assert "port 8002 is in use" in str(raised.value)
    assert "9000" in str(raised.value)


def test_ports_all_free_gives_no_busy_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lemonade, "port_is_busy", lambda _port: False)
    assert lemonade.busy_ports() == ()


def test_ports_a_listening_socket_reads_as_busy() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert lemonade.port_is_busy(port)
    assert not lemonade.port_is_busy(port)


def test_s8_the_first_launch_reports_the_backend_download(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lemonade, "busy_ports", tuple)
    with caplog.at_level(logging.WARNING, logger=lemonade.log.name):
        _launch()
    assert "downloads its llama.cpp backend" in caplog.text


def test_s8_no_report_once_the_backend_is_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lemonade, "busy_ports", tuple)
    (lemonade.data_dirs()[0] / "bin" / "llamacpp" / "cpu").mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger=lemonade.log.name):
        _launch()
    assert "downloads" not in caplog.text


def test_s14_a_role_the_server_does_not_answer_is_refused() -> None:
    for role in (WorkerRole.VISION,):
        with pytest.raises(ProviderError):
            LEMONADE.role_spec(role, None, None)


def test_s9_a_store_never_takes_lemonade_vectors_by_default() -> None:
    assert not proof.is_proven("lemonade")
    assert proof.not_run_on("lemonade") == ("AMD XDNA2 NPU",)


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.LEMONADE] is LEMONADE
