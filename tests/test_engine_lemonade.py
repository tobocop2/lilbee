"""The Lemonade adapter: XDNA2 gate, role folders, model link, swap entry, proof rows."""

from __future__ import annotations

import json
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lilbee.app import engine_service
from lilbee.catalog import xdna
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import choice, lemonade, proof
from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryKind
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import RoleTarget, WorkerRole

FIXTURES = Path(__file__).parent / "fixtures" / "host_caps"
BINARY = Path("/engines/bin/lemond")


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


MODES = {
    WorkerRole.CHAT: "chat",
    WorkerRole.EMBED: "embeddings",
}
TAGS = {WorkerRole.CHAT: "qwen3:0.6b", WorkerRole.EMBED: "embed-gemma:300m"}


def _request(model: Path, replica: int = 0, slots: int = 1, ctx: int = 2048) -> LaunchRequest:
    return LaunchRequest(
        binary=BINARY,
        model_path=model,
        devices=(),
        n_gpu_layers=0,
        slots=slots,
        ctx_per_slot=ctx,
        replica=replica,
    )


def _launch(
    role: WorkerRole = WorkerRole.CHAT,
    model: Path | None = None,
    replica: int = 0,
    slots: int = 1,
) -> list[str]:
    ticket = model or xdna.write_ticket(cfg.models_dir, TAGS[role])
    request = _request(ticket, replica, slots)
    return LEMONADE.launch_argv(LEMONADE.role_spec(role, None, None), request)


def _config_file(role: WorkerRole, name: str, replica: int = 0) -> dict[str, object]:
    path = lemonade.role_dir(role, replica) / "config" / name
    return json.loads(path.read_text(encoding="utf-8"))


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
        nvidia=None,
        accel_devices=[("0x1022", "0x1502")],
    )
    assert not LEMONADE.host_gate(caps).ok


def test_b1_an_unread_generation_is_unknown_and_not_auto_selectable() -> None:
    gate = LEMONADE.host_gate(_host("amd-generation-unread-linux"))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_b1_windows_is_unknown_and_never_automatic() -> None:
    gate = LEMONADE.host_gate(_host("windows-amd64"))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_b1_a_mac_is_refused() -> None:
    gate = LEMONADE.host_gate(_host("mac-arm64"))
    assert not gate.ok
    assert "not offered on macos" in gate.reason


def test_b1_windows_on_arm_is_refused() -> None:
    caps = host_caps.collect(system="Windows", machine="arm64", cpu_info={}, nvidia=None)
    assert not LEMONADE.host_gate(caps).ok


def test_b3_the_engine_serves_chat_and_embed_on_the_npu() -> None:
    assert LEMONADE.served_roles == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert LEMONADE.device_targets(BINARY) == {RoleTarget.NPU}
    assert LEMONADE.formats == {ModelFormat.XDNA}


def test_b3_rerank_is_refused_because_no_npu_recipe_serves_it() -> None:
    with pytest.raises(ProviderError, match="does not serve"):
        LEMONADE.role_spec(WorkerRole.RERANK, None, None)


def test_b2_the_memory_rule_is_a_fixed_reserve() -> None:
    assert LEMONADE.memory_kind is MemoryKind.FIXED_RESERVE
    assert LEMONADE.pin_record()["memory"]["rule"] == "fixed_reserve"


def test_s7_the_command_starts_lemond_with_its_own_data_folders() -> None:
    argv = _launch()
    root = lemonade.role_dir(WorkerRole.CHAT, 0)
    assert argv[:2] == [str(BINARY), "--port"]
    assert PORT_SLOT in argv
    assert argv[-2:] == [str(root / "cache"), str(root / "config")]
    assert "--no-broadcast" in argv


@pytest.mark.parametrize("role", list(MODES))
def test_r1_each_role_registers_one_model_on_the_npu_recipe(role: WorkerRole) -> None:
    _launch(role)
    root = lemonade.role_dir(role, 0)
    assert (root / "cache").is_dir()
    registered = _config_file(role, "user_models.json")
    entry = registered[f"lilbee-{role.value}"]
    assert entry["recipe"] == "flm"
    assert entry["checkpoint"] == TAGS[role]
    assert entry.get("labels") == (["embeddings"] if role is WorkerRole.EMBED else None)
    assert _config_file(role, "recipe_options.json") == {
        f"user.lilbee-{role.value}": {"ctx_size": 2048}
    }


@pytest.mark.parametrize("role", list(MODES))
def test_r1_the_config_names_no_model_folder_and_no_llama_backend(role: WorkerRole) -> None:
    _launch(role)
    assert _config_file(role, "config.json") == {"websocket_port": "auto"}


def test_r1_the_context_scales_with_the_slots() -> None:
    _launch(slots=3)
    options = _config_file(WorkerRole.CHAT, "recipe_options.json")
    assert options["user.lilbee-chat"] == {"ctx_size": 6144}


def test_r2_a_second_call_changes_nothing() -> None:
    first = _launch()
    root = lemonade.role_dir(WorkerRole.CHAT, 0)
    files = sorted((root / "config").iterdir())
    before = [f.read_bytes() for f in files]
    assert _launch() == first
    assert sorted((root / "config").iterdir()) == files
    assert [f.read_bytes() for f in files] == before


def test_r2_a_new_ticket_changes_the_registered_tag() -> None:
    _launch()
    ticket = xdna.write_ticket(cfg.models_dir, "llama3.2:1b")
    _launch(model=ticket)
    entry = _config_file(WorkerRole.CHAT, "user_models.json")["lilbee-chat"]
    assert entry["checkpoint"] == "llama3.2:1b"


def test_r2_a_model_path_that_is_not_a_ticket_is_refused(tmp_path: Path) -> None:
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="no NPU model tag"):
        _launch(model=weights)


def test_r3_the_swap_entry_names_the_model_and_checks_live() -> None:
    for role in MODES:
        launch = InstanceLaunch(
            role=role, argv=_launch(role), env_overrides={}, model="m", replica=0
        )
        extras = LEMONADE.swap_extras(launch)
        assert extras == {"checkEndpoint": "/live", "useModelName": f"user.lilbee-{role.value}"}
        launch = replace(launch, swap_extras=extras)
        config = json.loads(build_swap_config([launch], {launch.model_id: 5800}))
        entry = config["models"][launch.model_id]
        assert entry["useModelName"] == f"user.lilbee-{role.value}"
        assert entry["checkEndpoint"] == "/live"


def test_r4_held_lemonade_ports_do_not_refuse_the_launch() -> None:
    held: list[socket.socket] = []
    try:
        for port in (8001, 9000):
            server = socket.socket()
            try:
                server.bind(("127.0.0.1", port))
            except OSError:
                server.close()
                continue
            server.listen()
            held.append(server)
        assert _launch()[0] == str(BINARY)
    finally:
        for server in held:
            server.close()


def test_r4_the_fixed_port_helpers_are_gone() -> None:
    for name in ("CHILD_PORTS", "WEBSOCKET_PORT", "busy_ports", "port_is_busy"):
        assert not hasattr(lemonade, name)


def test_r6_every_role_and_replica_has_its_own_folder() -> None:
    folders = {lemonade.role_dir(role, r) for role in MODES for r in (0, 1)}
    assert len(folders) == 4
    _launch(WorkerRole.CHAT, replica=0)
    _launch(WorkerRole.CHAT, replica=1)
    assert (lemonade.role_dir(WorkerRole.CHAT, 0) / "config" / "user_models.json").is_file()
    assert (lemonade.role_dir(WorkerRole.CHAT, 1) / "config" / "user_models.json").is_file()


def test_the_adapter_names_no_model_folder_and_no_llama_server() -> None:
    for name in ("resolve_llama_server", "_place_model", "_links_to", "read_gguf_metadata"):
        assert not hasattr(lemonade, name)


def test_s14_a_role_the_server_does_not_answer_is_refused() -> None:
    for role in (WorkerRole.VISION,):
        with pytest.raises(ProviderError):
            LEMONADE.role_spec(role, None, None)


def test_s9_a_store_never_takes_lemonade_vectors_by_default() -> None:
    assert not proof.is_proven("lemonade")
    assert proof.not_run_on("lemonade") == ("AMD XDNA2 NPU",)


def test_the_adapter_ships_in_the_choice_table() -> None:
    assert choice.ADAPTERS[EngineName.LEMONADE] is LEMONADE


INSTALL_GUIDE = "https://lemonade-server.ai/docs/guide/install/"
EMBEDDABLE = "https://github.com/lemonade-sdk/lemonade/releases/tag/v2026.44.1"
USER_LEMOND = Path("/opt/lemonade/lemond")


def _no_store_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(choice, "store_tool", lambda *_a: None)


def test_g3_a_lemond_on_path_is_the_binary_when_the_store_has_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upstream embeddable README names the ``lemond`` executable (``lemond.exe`` on
    Windows) as the Lemonade instance:
    https://github.com/lemonade-sdk/lemonade/blob/v2026.44.1/docs/embeddable/README.md
    """
    _no_store_build(monkeypatch)
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: str(USER_LEMOND))
    assert LEMONADE.binary() == USER_LEMOND


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bit")
def test_g3_a_lemond_file_that_is_not_executable_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real mode-644 ``lemond`` on PATH is no binary; ``shutil.which`` is not mocked."""
    _no_store_build(monkeypatch)
    fake = tmp_path / "lemond"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o644)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert LEMONADE.user_binary() is None
    assert EngineName.LEMONADE not in choice.installed_engines()
    fake.chmod(0o755)
    assert LEMONADE.user_binary() == fake


def test_g3_a_store_build_wins_over_the_one_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(choice, "store_tool", lambda *_a: BINARY)
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: str(USER_LEMOND))
    assert LEMONADE.binary() == BINARY


def test_g3_the_lookup_asks_for_lemond(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: asked.append(name))
    assert LEMONADE.user_binary() is None
    assert asked == ["lemond"]


def test_g3_a_missing_install_says_where_to_get_lemonade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_store_build(monkeypatch)
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: None)
    with pytest.raises(ProviderError) as exc:
        LEMONADE.binary()
    text = str(exc.value)
    assert INSTALL_GUIDE in text
    assert EMBEDDABLE in text
    assert "PATH" in text
    assert "lilbee engine install" not in text


def test_g3_a_lemond_on_path_counts_as_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: str(USER_LEMOND))
    assert EngineName.LEMONADE in choice.installed_engines()
    monkeypatch.setattr(lemonade.shutil, "which", lambda name: None)
    assert EngineName.LEMONADE not in choice.installed_engines()


def test_g3_install_refuses_with_the_user_install_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Lemonade pack is published, so ``engine install`` points at the upstream install."""
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: _host("amd-xdna2-linux"))
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install("lemonade")
    text = str(exc.value)
    assert "No pack of lemonade is published yet." in text
    assert INSTALL_GUIDE in text
