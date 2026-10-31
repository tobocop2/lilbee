"""From a role to the written llama-swap config, each engine launches its own binary."""

from __future__ import annotations

import hashlib
import json
import shlex
import stat
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.app.engine_choice import EngineBook
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning, swap_config, swap_manager
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines import ik_llama, llama, mistral, tei
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import EngineBackend, WorkerRole
from tests._fleet_engines import ROLE_FIELDS, Role, cards_of, prepare_host
from tests.providers.fleet.test_mistral_adapter import caps

_GIB = 24
_CARD = ("NVIDIA L4", (8, 9))
_BUNDLED = "llama-server"
_STORE_TOOLS = {
    EngineName.MISTRAL: (mistral, "mistralrs"),
    EngineName.TEI: (tei, "text-embeddings-router"),
    EngineName.IK_LLAMA: (ik_llama, "llama-server"),
}


@dataclass(frozen=True)
class _Fleet:
    """The engine of each role; a role not named here has no model."""

    engines: dict[WorkerRole, EngineName]


_LLAMA_ALL = _Fleet(
    {
        WorkerRole.CHAT: EngineName.LLAMA,
        WorkerRole.EMBED: EngineName.LLAMA,
        WorkerRole.RERANK: EngineName.LLAMA,
    }
)
_MISTRAL_WITH_LLAMA_RERANK = _Fleet(
    {
        WorkerRole.CHAT: EngineName.MISTRAL,
        WorkerRole.EMBED: EngineName.MISTRAL,
        WorkerRole.VISION: EngineName.MISTRAL,
        WorkerRole.RERANK: EngineName.LLAMA,
    }
)
_MISTRAL_ALONE = _Fleet({WorkerRole.CHAT: EngineName.MISTRAL})
_TEI_WITH_LLAMA_CHAT = _Fleet(
    {
        WorkerRole.CHAT: EngineName.LLAMA,
        WorkerRole.EMBED: EngineName.TEI,
        WorkerRole.RERANK: EngineName.TEI,
    }
)
_IK_WITH_LLAMA_EMBED = _Fleet(
    {WorkerRole.CHAT: EngineName.IK_LLAMA, WorkerRole.EMBED: EngineName.LLAMA}
)

_ROWS = [
    pytest.param(fleet, role, id=f"{name}-{role.value}")
    for name, fleet in (
        ("llama", _LLAMA_ALL),
        ("mistral-with-llama-rerank", _MISTRAL_WITH_LLAMA_RERANK),
        ("mistral-alone", _MISTRAL_ALONE),
        ("tei-with-llama-chat", _TEI_WITH_LLAMA_CHAT),
        ("ik-with-llama-embed", _IK_WITH_LLAMA_EMBED),
    )
    for role in fleet.engines
]


def _ref(role: WorkerRole, engine: EngineName) -> str:
    """The configured model of *role*: TEI loads safetensors, the others a GGUF file."""
    return (
        f"org/{role.value}#safetensors"
        if engine is EngineName.TEI
        else f"org/repo/{role.value}.gguf"
    )


def _install_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: EngineName) -> Path:
    """Install a pack of *name* through the real store and return its tool's path."""
    module, tool = _STORE_TOOLS[name]
    wheel = tmp_path / f"{name.value}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        info = zipfile.ZipInfo(f"pkg/bin/{tool}")
        info.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(info, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "PACK_SHA256", sha)
    pin = ADAPTERS[name].pin_record()
    lock = PinLock(pin["engine"], pin["version"], pin["backend"], {}, ("MIT",), sha)
    EngineStore(engines_root(cfg.models_dir)).install_from_wheel(lock, wheel)
    return ADAPTERS[name].binary()


@pytest.fixture(autouse=True)
def _clean_snapshot() -> Iterator[None]:
    """The plan snapshot is process state, and the provider test writes it."""
    planning.clear_plan_probe()
    yield
    planning.clear_plan_probe()


def _install_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session: HostCaps
) -> dict[EngineName, Path]:
    """*session* with the bundled llama-server and every store engine installed."""
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setattr(runnable, "session_host", lambda: session)
    # The real read builds the process-wide services, and those start a warm-up thread.
    monkeypatch.setattr(swap_config, "_concurrency_limit", lambda *_a, **_k: 4)
    bundled = tmp_path / "lilbee_engine" / "bin" / _BUNDLED
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(llama, "resolve_llama_server", lambda: bundled)
    monkeypatch.setattr(
        llama,
        "probe_devices",
        lambda _binary: DeviceProbe(
            devices=cards_of((_GIB,)),
            output="",
            spoke_protocol=False,
            backend=EngineBackend.CUDA,
        ),
    )
    installed = {name: _install_pack(tmp_path, monkeypatch, name) for name in _STORE_TOOLS}
    return {EngineName.LLAMA: bundled, **installed}


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[EngineName, Path]:
    """A Linux host with one CUDA card."""
    return _install_host(tmp_path, monkeypatch, caps("Linux", "x86_64", gpus=[_CARD]))


@pytest.fixture
def cpu_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[EngineName, Path]:
    """A Linux host with no NVIDIA card, so mistral.rs runs its CPU build."""
    return _install_host(tmp_path, monkeypatch, caps("Linux", "x86_64"))


def _configure(fleet: _Fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set each role's model and the stored engine choice, as ``lilbee engine use`` does."""
    prepare_host(monkeypatch, tmp_path, tuple(Role(role, 1.0) for role in fleet.engines), 64 << 30)
    refs = {role: _ref(role, engine) for role, engine in fleet.engines.items()}
    for role, ref in refs.items():
        monkeypatch.setattr(cfg, ROLE_FIELDS[role], ref)
    monkeypatch.setattr(
        planning.engine_params,
        "resolve_model_path",
        lambda ref: tmp_path / f"{ref.rsplit('/', 1)[-1].split('#')[0].removesuffix('.gguf')}.gguf",
    )
    monkeypatch.setattr(
        cfg,
        "engine_choices",
        json.dumps({refs[role]: engine.value for role, engine in fleet.engines.items()}),
    )


def _first_word_of_each_role(configs: list[str]) -> dict[WorkerRole, str]:
    """The first word of each role's command, from the text of the written configs."""
    words: dict[WorkerRole, str] = {}
    for text in configs:
        for model_id, entry in json.loads(text)["models"].items():
            parts = shlex.split(entry["cmd"], posix=sys.platform != "win32")
            words[WorkerRole(model_id.rsplit("-", 1)[0])] = parts[0].strip('"')
    assert words, "no role reached a written config"
    return words


def _config_from_the_planner() -> list[str]:
    """The config rendered from a plan alone, with no snapshot of the clean box."""
    plan = planning.plan_all_launches(engines=EngineBook(cfg).adapters())
    ports = {launch.model_id: 40000 + i for i, launch in enumerate(plan.launches)}
    return [swap_config.build_swap_config(list(plan.launches), ports, swap=False, ttl_seconds=0)]


def _config_from_the_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The files a server's start writes: the real provider path up to the llama-swap spawn."""
    data_dir = tmp_path / "engine"
    monkeypatch.setattr(swap_manager, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
    monkeypatch.setattr(swap_manager, "spawn_bound_child", lambda *a, **k: MagicMock(pid=1))
    monkeypatch.setattr(swap_manager.SwapManager, "_await_health", lambda _self: None)
    monkeypatch.setattr(swap_manager, "ensure_engine_reaper", lambda _dir: True)
    monkeypatch.setattr(swap_manager, "_stop_own_fleet", lambda _cfg, _ports: None)
    provider = FleetProvider()
    assert provider._plan_and_spawn(data_dir)
    texts = [
        path.read_text(encoding="utf-8") for path in data_dir.glob(swap_manager._CONFIG_FILE_GLOB)
    ]
    for swap in provider._swaps.values():
        swap._close_log()
    return texts


@pytest.mark.parametrize(("fleet", "role"), _ROWS)
def test_a_plan_launches_the_binary_of_the_engine_of_the_role(
    fleet: _Fleet,
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(fleet, tmp_path, monkeypatch)
    words = _first_word_of_each_role(_config_from_the_planner())
    assert words[role] == str(host[fleet.engines[role]])


@pytest.mark.parametrize(("fleet", "role"), _ROWS)
def test_a_server_start_writes_the_binary_of_the_engine_of_the_role(
    fleet: _Fleet,
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(fleet, tmp_path, monkeypatch)
    words = _first_word_of_each_role(_config_from_the_provider(tmp_path, monkeypatch))
    assert words[role] == str(host[fleet.engines[role]])


def _entries_by_role(configs: list[str]) -> dict[WorkerRole, dict[str, object]]:
    """The written model entry of each role, from the text of the written configs."""
    entries: dict[WorkerRole, dict[str, object]] = {}
    for text in configs:
        for model_id, entry in json.loads(text)["models"].items():
            entries[WorkerRole(model_id.rsplit("-", 1)[0])] = entry
    assert entries, "no role reached a written config"
    return entries


def _words(entry: dict[str, object]) -> list[str]:
    return shlex.split(str(entry["cmd"]), posix=sys.platform != "win32")


@pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION])
def test_a_cuda_launch_of_mistral_turns_paged_attention_off(
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    words = _words(_entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))[role])
    assert words[words.index("--paged-attn") + 1] == "off"
    assert "--cpu" not in words
    assert "--gpu-memory-utilization" not in words


def test_a_cpu_launch_of_mistral_forces_the_cpu(
    cpu_host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_ALONE, tmp_path, monkeypatch)
    words = _words(
        _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))[WorkerRole.CHAT]
    )
    assert "--cpu" in words
    assert "--paged-attn" not in words
