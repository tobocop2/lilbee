"""The mlx-lm adapter: macOS gate, argv, profile, model metadata and the oversize refusal."""

from __future__ import annotations

import hashlib
import json
import shlex
import stat
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import host_caps, swap_config
from lilbee.providers.fleet.engine_store import (
    EngineConflictError,
    EngineStore,
    PinLock,
    engines_root,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import (
    NoPublishedPackError,
    installed_engines,
    pin_lock,
)
from lilbee.providers.fleet.engines.mlx import MACOS_FLOOR, MLX
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.roles import RoleTarget, WorkerRole
from tests._published import publish

CONTEXT = 40960
PROMPT_TOKENS = 60008


def _floor_text(floor: tuple[int, ...]) -> str:
    return ".".join(map(str, floor))


def host(system: str, machine: str, mac_ver: str = "") -> host_caps.HostCaps:
    return host_caps.collect(
        system=system, machine=machine, cpu_info={}, nvml=None, mac_ver=mac_ver
    )


@pytest.mark.parametrize(
    ("caps", "ok", "known", "words"),
    [
        (host("Darwin", "arm64", "14.0"), True, True, ""),
        (host("Darwin", "arm64", "14"), True, True, ""),
        (host("Darwin", "arm64", "15.2"), True, True, ""),
        (host("Darwin", "arm64", "26.0"), True, True, ""),
        (host("Darwin", "arm64", "13.6"), False, True, "needs macOS 14.0 or later"),
        (host("Darwin", "arm64", "13.6"), False, True, "this computer has 13.6"),
        (host("Darwin", "x86_64", "14.0"), False, True, "run lilbee natively"),
        (host("Linux", "x86_64"), False, True, "not offered on linux x86_64"),
        (host("Linux", "aarch64"), False, True, "not offered on linux arm64"),
        (host("Windows", "AMD64"), False, True, "not offered on windows x86_64"),
        (host("Darwin", "arm64", ""), True, False, "could not be read"),
    ],
)
def test_the_gate_passes_apple_silicon_on_macos_14_and_refuses_the_rest(
    caps: host_caps.HostCaps, ok: bool, known: bool, words: str
) -> None:
    gate = MLX.host_gate(caps)
    assert (gate.ok, gate.known) == (ok, known)
    assert words in gate.reason
    assert gate.auto_selectable is (ok and known)


def test_a_host_whose_platform_cannot_be_read_neither_passes_nor_refuses() -> None:
    gate = MLX.host_gate(host("", ""))
    assert (gate.ok, gate.known) == (True, False)


def test_the_adapter_is_found_by_module_and_serves_chat_only() -> None:
    assert choice.ADAPTERS[EngineName.MLX] is MLX
    assert (MLX.bundled, MLX.formats) == (False, frozenset({"mlx"}))
    assert MLX.served_roles == frozenset({WorkerRole.CHAT})


@pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION])
def test_a_role_other_than_chat_is_refused(role: WorkerRole) -> None:
    with pytest.raises(ProviderError, match=f"does not serve {role}"):
        MLX.role_spec(role, None, None)


def request(tmp_path: Path, **overrides: Any) -> LaunchRequest:
    fields: dict[str, Any] = {
        "binary": tmp_path / "mlx-lm-server",
        "model_path": tmp_path / "model",
        "devices": (),
        "n_gpu_layers": 99,
        "slots": 1,
        "ctx_per_slot": 4096,
    }
    return LaunchRequest(**{**fields, **overrides})


def test_the_argv_names_the_model_folder_and_leaves_the_port_to_the_swap(tmp_path: Path) -> None:
    argv = MLX.launch_argv(MLX.role_spec(WorkerRole.CHAT, None, None), request(tmp_path))
    assert argv[0] == str(tmp_path / "mlx-lm-server")
    assert argv[argv.index("--model") + 1] == str(tmp_path / "model")
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == PORT_SLOT
    assert argv.count("--port") == 1
    words = shlex.split(swap_config._command_line(argv, 4321))
    assert words[words.index("--port") + 1] == "4321"
    assert "--prompt-cache-bytes" in argv


def test_the_swap_names_the_default_model_for_every_request() -> None:
    assert MLX.swap_extras(MagicMock()) == {"useModelName": "default_model"}


def test_the_profile_matches_the_recorded_server() -> None:
    profile = MLX.request_profile(WorkerRole.CHAT)
    assert (profile.tokenize_path, profile.detokenize_path, profile.rerank_path) == (
        None,
        None,
        None,
    )
    assert profile.apply_template_path == "/apply-template"
    assert profile.reasoning_field == "reasoning"
    assert profile.label == "mlx-lm"
    assert profile.batch_overflow is None


def test_the_recorded_hang_is_not_read_as_a_context_overflow() -> None:
    fixture = Path(__file__).parents[2] / "conformance/fixtures/presets/mlxlm-chat.json"
    recorded = json.loads(fixture.read_text(encoding="utf-8"))["exchanges"]
    bodies = [e["body"] for e in recorded if e["status"] != 200]
    assert bodies
    assert not any(MLX.request_profile(WorkerRole.CHAT).context_overflow.search(b) for b in bodies)
    replies = [json.loads(e["body"]) for e in recorded if e["status"] == 200]
    message = replies[0]["choices"][0]["message"]
    assert message[MLX.request_profile(WorkerRole.CHAT).reasoning_field]


@pytest.mark.parametrize(
    "body", ["exceed_context_size", "request exceeds the available context size", "", "context"]
)
def test_no_server_reply_is_read_as_a_context_overflow(body: str) -> None:
    """The server never reports an overflow; the provider refuses before the request."""
    assert MLX.request_profile(WorkerRole.CHAT).context_overflow.search(body) is None


def model_dir(tmp_path: Path, config: object) -> Path:
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def test_model_meta_reads_the_window_from_config_json(tmp_path: Path) -> None:
    folder = model_dir(tmp_path, {"max_position_embeddings": CONTEXT})
    assert MLX.model_meta(folder) == {"context_length": str(CONTEXT)}


def test_model_meta_reads_the_window_of_a_nested_text_config(tmp_path: Path) -> None:
    folder = model_dir(tmp_path, {"text_config": {"max_position_embeddings": 8192}})
    assert MLX.model_meta(folder) == {"context_length": "8192"}


@pytest.mark.parametrize("config", [{}, {"max_position_embeddings": "big"}, [1], "text"])
def test_model_meta_gives_nothing_for_a_config_without_a_window(
    tmp_path: Path, config: object
) -> None:
    assert MLX.model_meta(model_dir(tmp_path, config)) is None


def test_model_meta_gives_nothing_for_a_missing_or_broken_file(tmp_path: Path) -> None:
    assert MLX.model_meta(tmp_path / "absent") is None
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "config.json").write_text("{not json", encoding="utf-8")
    assert MLX.model_meta(folder) is None


def test_the_memory_estimate_is_a_fixed_reserve_of_the_weights(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x" * 1000)
    estimate = MLX.estimate_memory(MemoryRequest(folder, 4096, 1, 99, True, cfg.kv_cache_type))
    assert MLX.memory_kind is MemoryKind.FIXED_RESERVE
    assert (estimate.kind, estimate.device_bytes, estimate.shared_bytes) == (
        MemoryKind.FIXED_RESERVE,
        1000,
        1000,
    )


def test_the_probes_answer_without_running_the_binary(tmp_path: Path) -> None:
    binary = tmp_path / "absent"
    assert MLX.discover_devices(binary).devices == []
    assert MLX.device_targets(binary) == frozenset({RoleTarget.GPU})
    assert not (
        MLX.check_readback(binary)
        or MLX.supports_fit_target(binary)
        or MLX.supports_shared_kv(binary)
    )


def test_an_oversize_prompt_is_refused_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_fleet_provider import _provider_with_clients

    folder = model_dir(tmp_path, {"max_position_embeddings": CONTEXT})
    monkeypatch.setattr(cfg, "chat_n_ctx_target", CONTEXT)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    ctx = engine_params.resolve_chat_ctx(folder, MLX.model_meta(folder), engine=MLX)
    assert ctx == CONTEXT
    client = MagicMock()
    client.in_flight = 0
    provider = _provider_with_clients({WorkerRole.CHAT: [client]})
    provider._chat_ctx = ctx
    monkeypatch.setattr(cfg, "chat_model", "org/repo#mlx")
    prompt = "word " * PROMPT_TOKENS
    with pytest.raises(ProviderError) as refused:
        provider.chat([{"role": "user", "content": prompt}], model="org/repo#mlx")
    assert refused.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert client.method_calls == []


def test_a_pin_with_no_digest_refuses_with_the_named_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: host("Darwin", "arm64", "15.2"))
    pin = MLX.pin_record()
    assert (pin["engine"], pin["backend"], pin["tool"]) == ("mlx", "metal", "mlx-lm-server")
    assert pin["sha256"] is None
    with pytest.raises(NoPublishedPackError, match=r"No pack of mlx is published yet\."):
        pin_lock(pin)


def test_install_refuses_without_a_digest_and_names_the_stay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caps = host("Darwin", "arm64", "15.2")
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: caps)
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: caps)
    with pytest.raises(engine_service.EngineRefusedError, match=r"No pack of mlx is published"):
        engine_service.install("mlx")


def test_install_on_macos_13_is_refused_by_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caps = host("Darwin", "arm64", "13.6")
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: caps)
    with pytest.raises(engine_service.EngineRefusedError, match=r"needs macOS 14\.0 or later"):
        engine_service.install("mlx")


def test_binary_of_an_engine_with_no_install_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: host("Darwin", "arm64", "15.2"))
    with pytest.raises(ProviderError, match=r"lilbee engine install mlx"):
        MLX.binary()


def test_the_engine_list_names_mlx_only_on_apple_silicon_with_macos_14(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    for system, machine, version, named in [
        ("Darwin", "arm64", "15.2", True),
        ("Darwin", "arm64", "13.6", False),
        ("Darwin", "x86_64", "14.0", False),
        ("Linux", "x86_64", "", False),
        ("Windows", "AMD64", "", False),
    ]:
        caps = host(system, machine, version)
        monkeypatch.setattr(engine_service, "session_host", lambda caps=caps: caps)
        assert ("mlx" in engine_service.not_installed()) is named, (system, machine, version)


def make_pack(path: Path) -> str:
    """A wheel with one executable under pkg/bin and an MIT licence; returns its sha256."""
    with zipfile.ZipFile(path, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/mlx-lm-server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_published_pack_states_the_gate_floor_and_installs_its_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: host("Darwin", "arm64", "15.2"))
    assert EngineName.MLX not in installed_engines()
    sha = make_pack(tmp_path / "pack.whl")
    publish(monkeypatch, EngineName.MLX, sha)
    pin = MLX.pin_record()
    assert {row["min_os"] for row in pin["platforms"].values()} == {_floor_text(MACOS_FLOOR)}
    lock = PinLock(
        pin["engine"], pin["version"], pin["backend"], {}, ("MIT",), sha, "mlx-lm-server"
    )
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, tmp_path / "pack.whl")
    assert EngineName.MLX in installed_engines()
    assert MLX.binary().parts[-3:] == ("pkg", "bin", "mlx-lm-server")
    assert "mlx 0.32.2" in MLX.identity(MLX.binary())


def test_the_members_that_need_no_reading_answer_plainly(tmp_path: Path) -> None:
    assert MLX.supports_architecture("anything")
    assert MLX.assert_loadable("repo", "file", None) is None
    assert MLX.runtime_env() == {}


def test_binary_reports_two_builds_of_one_pin_as_a_provider_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def conflict(self: EngineStore, lock: PinLock) -> Path:
        raise EngineConflictError("two builds of mlx are installed")

    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: host("Darwin", "arm64", "15.2"))
    sha = make_pack(tmp_path / "pack.whl")
    publish(monkeypatch, EngineName.MLX, sha)
    monkeypatch.setattr(EngineStore, "complete_build", conflict)
    with pytest.raises(ProviderError, match="two builds of mlx"):
        MLX.binary()


def test_a_planner_launch_for_mlx_gives_a_provider_that_refuses_an_oversize_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lilbee.providers.fleet import planning
    from lilbee.providers.fleet.provider import FleetProvider, SwapGroup
    from tests.test_fleet_provider import _FakeSwap

    folder = model_dir(tmp_path, {"max_position_embeddings": CONTEXT})
    (folder / "model.safetensors").write_bytes(b"\0" * 1024)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", CONTEXT)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    monkeypatch.setattr(cfg, "placement", "")
    monkeypatch.setattr(MLX, "binary", lambda: tmp_path / "mlx-lm-server")
    launch = planning.build_single_role_launch(WorkerRole.CHAT, folder, engine=MLX)
    assert launch.ctx == CONTEXT

    provider = FleetProvider()
    monkeypatch.setattr(provider, "_adopt_engine", lambda role: MLX)

    class Swap(_FakeSwap):
        def endpoint(self) -> str:
            return "http://127.0.0.1:1"

    swap: Any = Swap()
    calls: list[str] = []
    for method in ("chat", "chat_result", "chat_stream_items", "chat_tools"):
        monkeypatch.setattr(
            "lilbee.providers.fleet.client.LlamaServerClient." + method,
            lambda *a, **k: calls.append("http"),
        )
    provider._adopt_group(SwapGroup("chat"), swap, [launch])
    assert provider._chat_ctx == CONTEXT
    monkeypatch.setattr(cfg, "chat_model", "org/repo#mlx")
    prompt = "word " * PROMPT_TOKENS
    with pytest.raises(ProviderError) as refused:
        provider.chat([{"role": "user", "content": prompt}], model="org/repo#mlx")
    assert refused.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert calls == []
