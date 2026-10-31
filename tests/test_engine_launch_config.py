"""From a role to the written llama-swap config, each engine launches its own binary."""

from __future__ import annotations

import hashlib
import json
import shlex
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lilbee.app.engine_choice import EngineBook
from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import devices, planning, swap_config, swap_manager
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines import ik_llama, llama, mistral, tei
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.host_caps import CardMemory, HostCaps
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


def _configure(
    fleet: _Fleet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gib: float = 1.0
) -> None:
    """Set each role's model and the stored engine choice, as ``lilbee engine use`` does."""
    prepare_host(monkeypatch, tmp_path, tuple(Role(role, gib) for role in fleet.engines), 64 << 30)
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


def _started_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FleetProvider:
    """A provider past a server's start: the real path up to the llama-swap spawn."""
    data_dir = tmp_path / "engine"
    monkeypatch.setattr(swap_manager, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
    monkeypatch.setattr(swap_manager, "spawn_bound_child", lambda *a, **k: MagicMock(pid=1))
    monkeypatch.setattr(swap_manager.SwapManager, "_await_health", lambda _self: None)
    monkeypatch.setattr(swap_manager, "ensure_engine_reaper", lambda _dir: True)
    monkeypatch.setattr(swap_manager, "_stop_own_fleet", lambda _cfg, _ports: None)
    provider = FleetProvider()
    assert provider._plan_and_spawn(data_dir)
    return provider


def _config_from_the_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The files a server's start writes."""
    provider = _started_provider(tmp_path, monkeypatch)
    texts = [
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "engine").glob(swap_manager._CONFIG_FILE_GLOB)
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


@pytest.mark.parametrize(("fleet", "role"), _ROWS)
def test_only_mistral_roles_are_written_with_the_served_model_name(
    fleet: _Fleet,
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(fleet, tmp_path, monkeypatch)
    entry = _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))[role]
    if fleet.engines[role] is EngineName.MISTRAL:
        assert entry["useModelName"] == "default"
    else:
        assert "useModelName" not in entry
    assert "cmd" in entry


@pytest.mark.parametrize(("fleet", "role"), _ROWS)
def test_no_role_is_written_with_a_model_cache_variable(
    fleet: _Fleet,
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(fleet, tmp_path, monkeypatch)
    entry = _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))[role]
    assert not [item for item in entry.get("env", []) if item.startswith("HF_HUB_CACHE=")]
    assert "cmd" in entry


_FREE = 20 << 30
_LOW = 2 << 30
_TOTAL = 24 << 30
_FLEETS = [
    pytest.param(_LLAMA_ALL, id="llama"),
    pytest.param(_MISTRAL_WITH_LLAMA_RERANK, id="mistral-with-llama-rerank"),
    pytest.param(_MISTRAL_ALONE, id="mistral-alone"),
    pytest.param(_TEI_WITH_LLAMA_CHAT, id="tei-with-llama-chat"),
    pytest.param(_IK_WITH_LLAMA_EMBED, id="ik-with-llama-embed"),
]


class _Cards:
    """One card that every engine reads, with a free figure a test can lower."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, backend: str = "CUDA") -> None:
        self.free = _FREE
        self.driver_free = 0
        self.llama_reads = 0
        self.driver_reads = 0
        self._backend = backend
        monkeypatch.setattr(llama, "probe_devices", self._llama_probe)
        monkeypatch.setattr(devices, "read_card_memory", self._driver_read)

    def _llama_probe(self, _binary: Path) -> DeviceProbe:
        self.llama_reads += 1
        card = FleetDevice(self._backend, 0, "gpu0", _TOTAL, self.free)
        return DeviceProbe(devices=[card], output="", spoke_protocol=True)

    def _driver_read(self) -> list[CardMemory]:
        self.driver_reads += 1
        return [CardMemory("GPU-a", "gpu0", _TOTAL, self.driver_free or self.free)]


def _plan_of_the_reload(
    provider: FleetProvider, monkeypatch: pytest.MonkeyPatch
) -> planning.FleetPlan:
    """The plan a reload of the running provider computes."""
    real = planning.plan_all_launches
    plans: list[planning.FleetPlan] = []

    def _recording(*, engines: Any) -> planning.FleetPlan:
        plans.append(real(engines=engines))
        return plans[-1]

    monkeypatch.setattr(planning, "plan_all_launches", _recording)
    provider._reload_pass()
    return plans[-1]


@pytest.mark.parametrize("fleet", _FLEETS)
def test_a_reload_with_a_low_card_read_keeps_each_engines_clean_figure(
    fleet: _Fleet, host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(fleet, tmp_path, monkeypatch, gib=6.0)
    cards = _Cards(monkeypatch)
    provider = _started_provider(tmp_path, monkeypatch)
    cards.free = _LOW
    plan = _plan_of_the_reload(provider, monkeypatch)
    assert {launch.model_id.rsplit("-", 1)[0] for launch in plan.launches} == {
        role.value for role in fleet.engines
    }
    for name in set(fleet.engines.values()):
        probe = planning._plan_probe_store.get(name)
        assert probe is not None
        assert [d.free_bytes for d in probe.devices] == [_FREE]
    for swap in provider._swaps.values():
        swap._close_log()


def test_one_plan_reads_each_engine_once_and_chooses_once(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    cards = _Cards(monkeypatch)
    engines = EngineBook(cfg).adapters()
    planning.capture_plan_probe(engines)
    assert (cards.llama_reads, cards.driver_reads) == (1, 1)
    chosen = planning.snapshot_engine(engines)
    with planning._one_engine_per_pass(engine=chosen):
        plan = planning.plan_all_launches(engines=engines)
        other = next(e for e in engines.values() if e.name != chosen.name)
        assert planning.snapshot_engine({WorkerRole.CHAT: other}) is chosen
    assert plan.launches
    assert (cards.llama_reads, cards.driver_reads) == (1, 1)


def test_every_card_refused_keeps_the_cpu_pin_of_llama_beside_mistral(
    cpu_host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    real = planning._probe_engine_devices

    def _read(engine: Any) -> planning.DeviceReading:
        if engine.name is EngineName.LLAMA:
            return planning.DeviceReading(
                devices=[], backend=EngineBackend.UNKNOWN, refused_all=True
            )
        return real(engine)

    monkeypatch.setattr(planning, "_probe_engine_devices", _read)
    entries = _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))
    assert "--device none" in entries[WorkerRole.RERANK]["cmd"]
    assert "--device" not in entries[WorkerRole.CHAT]["cmd"]


@pytest.mark.parametrize(
    ("fleet", "role"),
    [
        pytest.param(_MISTRAL_ALONE, WorkerRole.CHAT, id="L7-mistral-alone"),
        pytest.param(
            _Fleet({WorkerRole.EMBED: EngineName.TEI, WorkerRole.RERANK: EngineName.TEI}),
            WorkerRole.EMBED,
            id="L8-tei-alone",
        ),
    ],
)
def test_an_engine_alone_gets_a_card_pin(
    fleet: _Fleet,
    role: WorkerRole,
    host: dict[EngineName, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(fleet, tmp_path, monkeypatch)
    _Cards(monkeypatch)
    entry = _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))[role]
    assert "CUDA_VISIBLE_DEVICES=0" in entry["env"]


def test_each_role_of_a_mixed_fleet_pins_in_its_own_engines_read(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    _Cards(monkeypatch, backend="Vulkan")
    entries = _entries_by_role(_config_from_the_provider(tmp_path, monkeypatch))
    assert "--device Vulkan0" in entries[WorkerRole.RERANK]["cmd"]
    assert "CUDA_VISIBLE_DEVICES=0" in entries[WorkerRole.CHAT]["env"]


def test_an_engine_chosen_after_the_capture_keeps_the_clean_free_figure(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = _Cards(monkeypatch, backend="Vulkan")
    planning.capture_plan_probe({WorkerRole.CHAT: llama.LLAMA})
    cards.free = _LOW
    probe = planning._current_plan_probe(mistral.MISTRAL)
    assert probe is not None
    assert [(d.backend, d.free_bytes) for d in probe.devices] == [("CUDA", _FREE)]
    assert [d.backend for d in planning._current_plan_probe(llama.LLAMA).devices] == ["Vulkan"]


def test_a_failed_read_of_one_engine_names_it_and_a_lone_engine_raises_as_it_is(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(_engine: Any) -> planning.DeviceReading:
        raise OSError("no driver")

    monkeypatch.setattr(planning, "_probe_engine_devices", _broken)
    with pytest.raises(OSError, match="no driver"):
        planning.capture_plan_probe({WorkerRole.CHAT: llama.LLAMA})
    both = {WorkerRole.CHAT: llama.LLAMA, WorkerRole.EMBED: mistral.MISTRAL}
    with pytest.raises(ProviderError, match="Cannot read the GPUs of the llama"):
        planning.capture_plan_probe(both)


def test_a_card_read_starts_no_process(
    host: dict[EngineName, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_process(*_a: Any, **_k: Any) -> None:
        raise AssertionError("a card read started a process")

    monkeypatch.setattr(subprocess, "run", _no_process)
    monkeypatch.setattr(subprocess, "Popen", _no_process)
    monkeypatch.setattr(devices, "read_card_memory", lambda: [])
    assert tei.TEI.discover_devices(host[EngineName.TEI]).devices == []
    assert mistral.MISTRAL.discover_devices(host[EngineName.MISTRAL]).devices == []


@pytest.mark.parametrize(
    ("mask", "indices"),
    [("", []), ("1", [1]), ("GPU-a,0", [0, 0]), ("1,7,0", [1])],
)
def test_the_card_read_follows_the_visible_devices_mask(
    mask: str, indices: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    cards = [
        CardMemory("GPU-a", "card0", _TOTAL, _FREE),
        CardMemory("GPU-b", "card1", _TOTAL, _FREE),
    ]
    monkeypatch.setattr(devices, "read_card_memory", lambda: cards)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    found = devices.visible_nvidia_cards()
    assert [d.name for d in found] == [f"card{i}" for i in indices]
    assert [d.index for d in found] == list(range(len(indices)))


def test_a_cpu_tei_build_reads_no_card(
    cpu_host: dict[EngineName, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[int] = []
    monkeypatch.setattr(devices, "read_card_memory", lambda: called.append(1) or [])
    assert tei.TEI.discover_devices(cpu_host[EngineName.TEI]).devices == []
    assert not called


def test_a_start_keeps_the_figure_each_engine_read_itself(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    cards = _Cards(monkeypatch)
    cards.driver_free = _LOW
    provider = _started_provider(tmp_path, monkeypatch)
    llama_read = planning._plan_probe_store.get(EngineName.LLAMA)
    mistral_read = planning._plan_probe_store.get(EngineName.MISTRAL)
    assert llama_read is not None
    assert mistral_read is not None
    assert [d.free_bytes for d in llama_read.devices] == [_FREE]
    assert [d.free_bytes for d in mistral_read.devices] == [_LOW]
    for swap in provider._swaps.values():
        swap._close_log()


def test_two_broken_engines_each_keep_their_failure_wait() -> None:
    store = planning._PlanProbeStore()
    store.note_failed_probe("broken-a")
    store.note_failed_probe("broken-b")
    assert store.probe_failed_recently("broken-a")
    assert store.probe_failed_recently("broken-b")
    assert not store.probe_failed_recently("healthy")


def test_a_start_reads_each_engine_and_a_reload_reads_each_again(
    host: dict[EngineName, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(_MISTRAL_WITH_LLAMA_RERANK, tmp_path, monkeypatch)
    cards = _Cards(monkeypatch)
    provider = _started_provider(tmp_path, monkeypatch)
    for name in _MISTRAL_WITH_LLAMA_RERANK.engines.values():
        assert planning._plan_probe_store.get(name) is not None
    before = (cards.llama_reads, cards.driver_reads)
    _plan_of_the_reload(provider, monkeypatch)
    assert cards.llama_reads > before[0]
    assert cards.driver_reads > before[1]
    for swap in provider._swaps.values():
        swap._close_log()
