"""The Lemonade adapter: XDNA2 gate, role folders, model link, swap entry, proof rows."""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lilbee.app import engine_service
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
        nvml=None,
        accel_devices=None if accel is None else [tuple(pair) for pair in accel],
    )


LLAMA_SERVER = Path("/engines/llama/llama-server")
MODES = {
    WorkerRole.CHAT: "chat",
    WorkerRole.EMBED: "embeddings",
}


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
    request = _request(model or Path("/models/m.gguf"), replica, slots)
    return LEMONADE.launch_argv(LEMONADE.role_spec(role, None, None), request)


def _config(role: WorkerRole = WorkerRole.CHAT, replica: int = 0) -> dict[str, object]:
    path = lemonade.role_dir(role, replica) / "config" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(lemonade, "resolve_llama_server", lambda: LLAMA_SERVER)


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


def test_b1_windows_is_unknown_and_never_automatic() -> None:
    gate = LEMONADE.host_gate(_host("windows-amd64"))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_b1_a_mac_is_refused() -> None:
    gate = LEMONADE.host_gate(_host("mac-arm64"))
    assert not gate.ok
    assert "not offered on macos" in gate.reason


def test_b1_windows_on_arm_is_refused() -> None:
    caps = host_caps.collect(system="Windows", machine="arm64", cpu_info={}, nvml=None)
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
def test_r1_each_role_gets_folders_a_config_and_a_link_to_the_model(role: WorkerRole) -> None:
    model = Path("/models/m.gguf")
    _launch(role, model)
    root = lemonade.role_dir(role, 0)
    link = root / "view" / MODES[role] / f"lilbee-{role.value}.gguf"
    assert link.is_symlink()
    assert link.readlink() == model
    assert (root / "cache").is_dir()
    config = _config(role)
    assert config["extra_models_dir"] == str(root / "view")
    assert config["websocket_port"] == "auto"
    assert config["llamacpp"] == {"backend": "cpu", "cpu_bin": str(LLAMA_SERVER.parent)}
    options = json.loads((root / "config" / "recipe_options.json").read_text(encoding="utf-8"))
    assert options == {
        f"extra.lilbee-{role.value}": {"ctx_size": 2048, "llamacpp_args": "--parallel 1"}
    }


def test_r1_the_recipe_scales_context_and_parallel_with_the_slots() -> None:
    _launch(slots=3)
    path = lemonade.role_dir(WorkerRole.CHAT, 0) / "config" / "recipe_options.json"
    options = json.loads(path.read_text(encoding="utf-8"))["extra.lilbee-chat"]
    assert options == {"ctx_size": 6144, "llamacpp_args": "--parallel 3"}


def test_r2_a_second_call_changes_nothing() -> None:
    _launch()
    root = lemonade.role_dir(WorkerRole.CHAT, 0)
    files = [root / "config" / "config.json", root / "config" / "recipe_options.json"]
    link = root / "view" / "chat" / "lilbee-chat.gguf"
    before = [f.read_bytes() for f in files]
    link_before = (link.lstat().st_ino, link.lstat().st_mtime_ns)
    assert _launch() == _launch()
    assert [f.read_bytes() for f in files] == before
    assert (link.lstat().st_ino, link.lstat().st_mtime_ns) == link_before


def test_r2_a_repeat_call_never_removes_the_live_link(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Path("/models/m.gguf")
    _launch(model=model)
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    gone: list[Path] = []
    real_unlink = Path.unlink

    def watch(self: Path, missing_ok: bool = False) -> None:
        gone.append(self)
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", watch)
    _launch(model=Path("/models/other.gguf"))
    assert link not in gone
    assert link.readlink() == Path("/models/other.gguf")


def test_r2_a_new_model_path_moves_the_link() -> None:
    _launch(model=Path("/models/a.gguf"))
    _launch(model=Path("/models/b.gguf"))
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    assert link.readlink() == Path("/models/b.gguf")


def test_r8_the_embed_recipe_records_the_context_lemond_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bge-m3 asked for 4096 in the recording and ran with the 8192 it was trained on."""
    served = int(_p4_embed_ctx())
    monkeypatch.setattr(lemonade, "read_gguf_metadata", lambda _p: {"context_length": str(served)})
    _launch(WorkerRole.EMBED)
    path = lemonade.role_dir(WorkerRole.EMBED, 0) / "config" / "recipe_options.json"
    options = json.loads(path.read_text(encoding="utf-8"))["extra.lilbee-embed"]
    assert options["ctx_size"] == served


def test_r8_chat_records_the_context_it_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lemonade, "read_gguf_metadata", lambda _p: {"context_length": "32768"})
    _launch(WorkerRole.CHAT)
    path = lemonade.role_dir(WorkerRole.CHAT, 0) / "config" / "recipe_options.json"
    assert json.loads(path.read_text(encoding="utf-8"))["extra.lilbee-chat"]["ctx_size"] == 2048


def test_r8_an_unreadable_embed_model_keeps_the_asked_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lemonade, "read_gguf_metadata", lambda _p: None)
    _launch(WorkerRole.EMBED)
    path = lemonade.role_dir(WorkerRole.EMBED, 0) / "config" / "recipe_options.json"
    assert json.loads(path.read_text(encoding="utf-8"))["extra.lilbee-embed"]["ctx_size"] == 2048


def _p4_embed_ctx() -> str:
    from tests.conformance.test_f3_recordings import _p4_children

    return _p4_children()["embed"]["--ctx-size"]


def test_r3_the_swap_entry_names_the_model_and_checks_live() -> None:
    for role in MODES:
        launch = InstanceLaunch(
            role=role, argv=_launch(role), env_overrides={}, model="m", replica=0
        )
        extras = LEMONADE.swap_extras(launch)
        assert extras == {"checkEndpoint": "/live", "useModelName": f"lilbee-{role.value}"}
        launch = replace(launch, swap_extras=extras)
        config = json.loads(build_swap_config([launch], {launch.model_id: 5800}))
        entry = config["models"][launch.model_id]
        assert entry["useModelName"] == f"lilbee-{role.value}"
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


def test_r5_a_failed_symlink_falls_back_to_a_hard_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")
    monkeypatch.setattr(os, "symlink", _refuse)
    _launch(model=model)
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    assert not link.is_symlink()
    assert os.path.samefile(link, model)


def test_r5_a_failed_hard_link_falls_back_to_a_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")
    monkeypatch.setattr(os, "symlink", _refuse)
    monkeypatch.setattr(os, "link", _refuse)
    _launch(model=model)
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    assert link.read_bytes() == b"weights"
    assert not os.path.samefile(link, model)


def test_r5_a_copy_is_replaced_when_the_model_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "a.gguf", tmp_path / "b.gguf"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    monkeypatch.setattr(os, "symlink", _refuse)
    monkeypatch.setattr(os, "link", _refuse)
    _launch(model=first)
    _launch(model=second)
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    assert link.read_bytes() == b"two"


def test_r5_no_way_to_place_the_model_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "symlink", _refuse)
    monkeypatch.setattr(os, "link", _refuse)
    monkeypatch.setattr(lemonade.shutil, "copyfile", _refuse)
    with pytest.raises(ProviderError, match="cannot link the model"):
        _launch(model=tmp_path / "m.gguf")


def test_r5_a_plain_file_in_place_of_the_link_is_replaced_when_the_model_is_missing() -> None:
    link = lemonade.role_dir(WorkerRole.CHAT, 0) / "view" / "chat" / "lilbee-chat.gguf"
    link.parent.mkdir(parents=True)
    link.write_bytes(b"old")
    _launch(model=Path("/models/gone.gguf"))
    assert link.readlink() == Path("/models/gone.gguf")


def test_r6_every_role_and_replica_has_its_own_folder() -> None:
    folders = {lemonade.role_dir(role, r) for role in MODES for r in (0, 1)}
    assert len(folders) == 4
    _launch(WorkerRole.CHAT, replica=0)
    _launch(WorkerRole.CHAT, replica=1)
    assert (
        _config(WorkerRole.CHAT, 0)["extra_models_dir"]
        != _config(WorkerRole.CHAT, 1)["extra_models_dir"]
    )


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise OSError("not allowed")


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
