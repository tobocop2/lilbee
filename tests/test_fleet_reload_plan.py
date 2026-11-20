"""A settings write restarts only the group whose launch changed, and no request sees a gap."""

from __future__ import annotations

import os
import stat as stat_module
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee.app.settings import _reload_changed_roles
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.client import LlamaServerClient, StreamInterruptedError
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.roles import WorkerRole
from tests._async_wait import poll_until
from tests._fake_engine_server import FakeEngine
from tests._fleet_engines import engines_for_all_roles
from tests.test_fleet_provider import _fake_client, _FakeSwap
from tests.test_fleet_reload_gap import _SignalCondition

_JOIN_S = 10.0
_GB = 1024**3
_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
_OTHER_CHAT_REF = "Qwen/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
_DRAIN_LIMIT_S = 0.3
_QUESTION = [{"role": "user", "content": "hi"}]
_NOTES_CHUNK_SIZE = 384
_NOTES_CHUNK_OVERLAP = 64
_EMBED_WINDOW_AT_512 = 2048
_EMBED_WINDOW_AT_384 = 1544


@pytest.fixture
def real_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The real planner, with fakes only at the engine binary, device probe and GGUF read."""
    files = {
        _EMBED_REF: tmp_path / "embed.gguf",
        _CHAT_REF: tmp_path / "chat.gguf",
        _OTHER_CHAT_REF: tmp_path / "other-chat.gguf",
    }
    for path in files.values():
        path.write_bytes(b"weights")
    meta = {
        files[_EMBED_REF]: {
            "general.architecture": "nomic-bert",
            "nomic-bert.context_length": "2048",
        },
        files[_CHAT_REF]: {"general.architecture": "qwen3", "qwen3.context_length": "32768"},
        files[_OTHER_CHAT_REF]: {"general.architecture": "qwen3", "qwen3.context_length": "32768"},
    }
    device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
    monkeypatch.setattr(llama_mod, "resolve_llama_server", lambda: Path("/no/llama-server"))
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: files[ref])
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda path: meta[Path(path)]
    )
    monkeypatch.setattr(
        llama_mod,
        "probe_devices",
        lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
    )
    monkeypatch.setattr(cfg, "embedding_model", _EMBED_REF)
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "ocr", "auto")
    monkeypatch.setattr(cfg, "chunk_size", 512)
    monkeypatch.setattr(cfg, "chunk_overlap", 100)
    planning_mod.clear_plan_probe()
    planning_mod.clear_read_device_cache()
    planning_mod.capture_plan_probe({WorkerRole.CHAT: LLAMA})
    yield
    planning_mod.clear_plan_probe()
    planning_mod.clear_read_device_cache()


def _running_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FleetProvider:
    """A provider serving the current real plan, with fake servers and fake clients."""
    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _FakeSwap())
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _dir, **_kw: None)
    monkeypatch.setattr(prov_mod, "LlamaServerClient", lambda _e, _m, **_kw: _fake_client())
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    with provider._lock:
        for group, launches in _launches_by_group(
            planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA))
        ).items():
            provider._adopt_group(group, _FakeSwap(), list(launches))
            provider._group_dirs[group] = tmp_path
    return provider


def _join_fleet_threads() -> None:
    """Wait for every background thread the fleet started, so a late teardown is seen."""
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.name.startswith("fleet"):
            thread.join(_JOIN_S)
            assert not thread.is_alive()


def _apply_settings_write(monkeypatch, provider: FleetProvider, keys: set[str]) -> None:
    """Run the settings reload path for *keys* against *provider* and wait for it to settle."""
    services = SimpleNamespace(provider=provider, reload_role=provider.reload_role)
    monkeypatch.setattr("lilbee.app.services.peek_services", lambda: services)
    _reload_changed_roles(keys)
    _join_fleet_threads()


def test_an_ocr_change_with_no_vision_model_leaves_every_launch_identical(real_planner) -> None:
    before = planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches
    cfg.ocr = "off"
    assert planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches == before


def test_an_ocr_change_with_a_vision_model_drops_only_the_vision_launch(
    real_planner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"projector")
    monkeypatch.setattr(planning_mod, "vision_mmproj", lambda _ref: projector)
    monkeypatch.setattr(cfg, "vision_model", _CHAT_REF)
    cfg.ocr = "all"
    before = {
        launch.role: launch
        for launch in planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches
    }
    cfg.ocr = "off"
    after = {
        launch.role: launch
        for launch in planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches
    }
    assert WorkerRole.VISION in before
    assert WorkerRole.VISION not in after
    assert before[WorkerRole.EMBED] == after[WorkerRole.EMBED]
    assert before[WorkerRole.CHAT] == after[WorkerRole.CHAT]


def test_a_chunk_size_change_moves_only_the_embed_window(real_planner) -> None:
    before = {
        launch.role: launch
        for launch in planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches
    }
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    after = {
        launch.role: launch
        for launch in planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).launches
    }
    assert before[WorkerRole.CHAT] == after[WorkerRole.CHAT]
    assert before[WorkerRole.EMBED].ctx == _EMBED_WINDOW_AT_512
    assert after[WorkerRole.EMBED].ctx == _EMBED_WINDOW_AT_384


def test_reload_for_ocr_keeps_the_embed_client_without_a_drop(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    embed_client = provider._clients[WorkerRole.EMBED][0]
    drop = MagicMock(wraps=provider._drop_group)
    monkeypatch.setattr(provider, "_drop_group", drop)
    cfg.ocr = "off"
    provider.reload_role(WorkerRole.VISION, wait=True)
    assert provider._clients[WorkerRole.EMBED][0] is embed_client
    drop.assert_not_called()


def test_a_preset_write_restarts_embed_and_keeps_chat_serving(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    chat_client = provider._clients[WorkerRole.CHAT][0]
    embed_client = provider._clients[WorkerRole.EMBED][0]
    teardown = MagicMock(wraps=provider._drop_swap_refs)
    monkeypatch.setattr(provider, "_drop_swap_refs", teardown)
    cfg.ocr = "off"
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    cfg.chunk_overlap = _NOTES_CHUNK_OVERLAP
    _apply_settings_write(monkeypatch, provider, {"chunk_size", "chunk_overlap", "ocr"})
    teardown.assert_not_called()
    assert provider._clients[WorkerRole.CHAT][0] is chat_client
    assert provider._require_clients(WorkerRole.EMBED)[0] is not embed_client


def test_a_chunk_size_write_keeps_the_chat_server(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    chat_client = provider._clients[WorkerRole.CHAT][0]
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    _apply_settings_write(monkeypatch, provider, {"chunk_size"})
    assert provider._clients[WorkerRole.CHAT][0] is chat_client
    assert provider._role_launches(WorkerRole.EMBED)[0].ctx == _EMBED_WINDOW_AT_384


def test_a_request_during_a_real_embed_restart_gets_the_new_server(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    provider = _running_provider(monkeypatch, tmp_path)
    gate = _SignalCondition(provider._lock)
    provider._reload_done = gate
    old_client = provider._clients[WorkerRole.EMBED][0]
    starting = threading.Event()
    release = threading.Event()

    class _HeldSwap(_FakeSwap):
        def start(self, *args, **kwargs) -> None:
            starting.set()
            assert release.wait(_JOIN_S)
            super().start(*args, **kwargs)

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _HeldSwap())
    cfg.chunk_size = _NOTES_CHUNK_SIZE
    provider.reload_role(WorkerRole.VISION)
    assert starting.wait(_JOIN_S)

    results: list[object] = []
    request = threading.Thread(
        target=lambda: results.append(provider._require_clients(WorkerRole.EMBED)), daemon=True
    )
    request.start()
    assert gate.entered.wait(_JOIN_S)
    release.set()
    request.join(_JOIN_S)
    _join_fleet_threads()
    assert len(results) == 1
    assert results[0][0] is not old_client  # type: ignore[index]


def _reporting_size(path: Path, size: int) -> Path:
    """A path to the small file at *path* whose stat reports *size* bytes.

    A truncated file is not sparse on NTFS, so a real several-GB file would write zeros.
    Every size read in the planner goes through ``stat``, so one override covers them all.
    """

    class _SizedPath(type(path)):  # type: ignore[misc]
        def stat(self, **kwargs: object) -> os.stat_result:
            real = super().stat(**kwargs)  # type: ignore[arg-type]
            fields = list(real)
            fields[stat_module.ST_SIZE] = size
            return os.stat_result(fields)

    return _SizedPath(path)


@pytest.mark.parametrize(
    ("free_gb", "captured", "names_other_processes"),
    [(1, True, True), (7, True, False), (1, False, False)],
    ids=["1gb-free", "7gb-free", "1gb-free-no-clean-box-snapshot"],
)
def test_a_small_card_refusal_names_the_memory_other_processes_hold(
    real_planner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    free_gb: int,
    captured: bool,
    names_other_processes: bool,
) -> None:
    small = tmp_path / "big-chat.gguf"
    small.write_bytes(b"weights")
    chat = _reporting_size(small, 6 * _GB)
    resolve = engine_params.resolve_model_path
    monkeypatch.setattr(
        engine_params, "resolve_model_path", lambda ref: chat if ref == _CHAT_REF else resolve(ref)
    )
    read = llama_mod.read_gguf_metadata
    chat_meta = {"general.architecture": "qwen3", "qwen3.context_length": "32768"}
    monkeypatch.setattr(
        llama_mod, "read_gguf_metadata", lambda p: chat_meta if Path(p) == chat else read(p)
    )
    device = FleetDevice("CUDA", 0, "gpu", 8 * _GB, free_gb * _GB)
    monkeypatch.setattr(
        llama_mod,
        "probe_devices",
        lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
    )
    planning_mod.clear_plan_probe()
    planning_mod.clear_read_device_cache()
    if captured:
        planning_mod.capture_plan_probe({WorkerRole.CHAT: LLAMA})

    reason = planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA)).refused_roles[
        WorkerRole.CHAT
    ]

    assert ("Other processes hold" in reason) is names_other_processes
    if names_other_processes:
        assert f"Other processes hold {8 - free_gb} GiB of the card's 8 GiB." in reason


class _EngineSwap(_FakeSwap):
    """A swap fronting a fake engine; it records how many answers the engine had at its stop."""

    def __init__(self, *, answers_at_once: bool) -> None:
        super().__init__()
        self.engine = FakeEngine()
        if answers_at_once:
            self.engine.release.set()
        self.answers_at_stop: int | None = None

    def endpoint(self) -> str:
        return self.engine.url

    def role_ready(self, role: WorkerRole) -> bool:
        return True

    def shutdown(self) -> None:
        self.answers_at_stop = self.engine.completed
        self.engine.stop()
        super().shutdown()


@pytest.fixture
def engine_fleet(
    real_planner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[FleetProvider, list[_EngineSwap]]]:
    """A provider serving the real plan from fake engines, with every engine it starts listed.

    The engines of the first fleet hold each chat request until the test releases it. The
    engines a reload starts answer at once.
    """
    yield from _serve_fake_engines(monkeypatch, tmp_path)


@pytest.fixture
def mlx_engine_fleet(
    real_planner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[FleetProvider, list[_EngineSwap]]]:
    """The engine fleet of ``engine_fleet``, with each chat server read as an MLX server."""

    def adapter_of(_provider: FleetProvider, role: WorkerRole) -> object:
        return MLX if role is WorkerRole.CHAT else LLAMA

    monkeypatch.setattr(FleetProvider, "_adopt_engine", adapter_of)
    yield from _serve_fake_engines(monkeypatch, tmp_path)


def _serve_fake_engines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[FleetProvider, list[_EngineSwap]]]:
    """Serve the real plan from fake engines; see ``engine_fleet``."""
    swaps: list[_EngineSwap] = []
    fleet_up = threading.Event()

    def make_swap(_dir: Path, _group: object) -> _EngineSwap:
        swap = _EngineSwap(answers_at_once=fleet_up.is_set())
        swaps.append(swap)
        return swap

    monkeypatch.setattr(prov_mod, "SwapManager", make_swap)
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _dir, **_kw: None)
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_preload_roles", lambda roles=None: None)
    with provider._lock:
        for group, launches in _launches_by_group(
            planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA))
        ).items():
            provider._adopt_group(group, make_swap(tmp_path, group), list(launches))
            provider._group_dirs[group] = tmp_path
    fleet_up.set()
    yield provider, swaps
    for swap in swaps:
        swap.engine.stop()


def _reload_chat_model(provider: FleetProvider) -> None:
    """Change the chat model and start the reload; return once the old chat client is retired."""
    old_client = provider._clients[WorkerRole.CHAT][0]
    cfg.chat_model = _OTHER_CHAT_REF
    provider.reload_role(WorkerRole.CHAT)
    assert poll_until(lambda: old_client not in provider._clients.get(WorkerRole.CHAT, []))


def test_a_reload_stops_the_old_engine_after_a_running_chat_is_answered(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
) -> None:
    provider, swaps = engine_fleet
    old = provider._swap_for(WorkerRole.CHAT)
    answers: list[object] = []
    caller = threading.Thread(target=lambda: answers.append(provider.chat(_QUESTION)), daemon=True)
    caller.start()
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert [answer.text for answer in answers] == ["the answer"]
    assert old.answers_at_stop == 1
    assert old.engine.requests == 1
    assert swaps[-1].engine.requests == 0


def test_a_reload_with_no_time_left_asks_each_client_for_no_wait(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", 0.0)
    waits: list[float] = []
    wait_idle = LlamaServerClient.wait_idle

    def _record(client: LlamaServerClient, timeout: float) -> bool:
        waits.append(timeout)
        return wait_idle(client, timeout)

    monkeypatch.setattr(LlamaServerClient, "wait_idle", _record)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    assert poll_until(lambda: old.engine.stopped.is_set())
    with pytest.raises(StreamInterruptedError, match="answer is cut"):
        list(stream)
    _join_fleet_threads()

    assert waits
    assert set(waits) == {0.0}


def test_a_chat_cut_by_the_drain_limit_is_sent_again_and_ends_on_the_new_engine_only(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    answers: list[object] = []
    caller = threading.Thread(target=lambda: answers.append(provider.chat(_QUESTION)), daemon=True)
    caller.start()
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert [answer.text for answer in answers] == ["the answer"]
    assert old.engine.requests == 1
    assert old.answers_at_stop == 0
    assert swaps[-1].engine.requests == 1
    assert swaps[-1].engine.completed == 1


def test_a_timeout_restart_on_mlx_lets_the_other_chat_finish_on_the_old_engine(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One chat times out while another runs: the restart waits for it, past the drain limit."""
    provider, swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    client = provider._clients[WorkerRole.CHAT][0]
    answers: list[object] = []
    caller = threading.Thread(target=lambda: answers.append(provider.chat(_QUESTION)), daemon=True)
    caller.start()
    assert old.engine.started.wait(_JOIN_S)

    # httpx fixes a request's timeout when it sends it, so only the next request times out.
    client._http.timeout = httpx.Timeout(_DRAIN_LIMIT_S)
    with pytest.raises(ProviderError, match="not responding"):
        provider.chat(_QUESTION)
    assert poll_until(lambda: client not in provider._clients.get(WorkerRole.CHAT, []))
    assert not poll_until(lambda: old.engine.stopped.is_set(), timeout=3 * _DRAIN_LIMIT_S)

    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert [answer.text for answer in answers] == ["the answer"]
    assert old.engine.stopped.is_set()
    assert old.engine.requests == 2
    assert swaps[-1] is not old
    assert swaps[-1].engine.requests == 0


def test_a_reload_lets_a_started_stream_finish_before_it_stops_the_old_engine(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
) -> None:
    provider, _swaps = engine_fleet
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    old.engine.release.set()
    frames = list(stream)
    _join_fleet_threads()

    assert "first" in frames
    assert old.answers_at_stop == 1


def test_a_stream_cut_by_the_drain_limit_ends_with_a_message_for_the_user(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    assert poll_until(lambda: old.engine.stopped.is_set())
    with pytest.raises(StreamInterruptedError, match="answer is cut"):
        list(stream)
    _join_fleet_threads()

    assert old.answers_at_stop == 0


def test_a_shutdown_during_the_drain_stops_the_old_engine_and_starts_no_new_one(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
) -> None:
    provider, swaps = engine_fleet
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    assert old.engine.started.wait(_JOIN_S)
    started_before = len(swaps)

    _reload_chat_model(provider)
    with provider._lock:
        provider._shut_down = True

    assert poll_until(lambda: old.engine.stopped.is_set())
    _join_fleet_threads()
    assert len(swaps) == started_before
    assert provider._reload_owed == frozenset()
    stream.close()


def test_a_reload_waits_for_a_running_request_on_every_replica(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
) -> None:
    provider, _swaps = engine_fleet
    old = provider._swap_for(WorkerRole.CHAT)
    replica = LlamaServerClient(old.engine.url, "m", profile=LLAMA.request_profile(WorkerRole.CHAT))
    replica.reserve()
    with provider._lock:
        provider._clients[WorkerRole.CHAT].append(replica)

    _reload_chat_model(provider)
    assert not old.engine.stopped.wait(_DRAIN_LIMIT_S)
    replica.release()

    assert old.engine.stopped.wait(_JOIN_S)
    _join_fleet_threads()


class _WaitSeen(Exception):
    """Raised by the recorded wait, so the request stops where it would have waited."""


class _RecordedWait(threading.Condition):
    """A condition that records the timeout of a wait and ends the request there."""

    def __init__(self, lock: threading.RLock, timeouts: list[float | None]) -> None:
        super().__init__(lock)
        self._timeouts = timeouts

    def wait_for(self, predicate, timeout=None):  # type: ignore[no-untyped-def]
        self._timeouts.append(timeout)
        raise _WaitSeen


def test_a_request_waits_for_a_reload_as_long_as_the_drain_and_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FleetProvider()
    timeouts: list[float | None] = []
    provider._reload_done = _RecordedWait(provider._lock, timeouts)
    monkeypatch.setattr(provider, "_ensure_fleet", lambda: True)

    with pytest.raises(_WaitSeen):
        provider._require_clients(WorkerRole.CHAT)

    assert timeouts == [prov_mod._SHUTDOWN_WORK_WAIT_S + prov_mod._RELOAD_START_WAIT_S]


def test_a_caller_that_waits_for_a_reload_waits_as_long_as_the_drain_and_the_start() -> None:
    provider = FleetProvider()
    work = MagicMock()
    work.wait_unless_woken.return_value = True
    provider._fleet_work = work

    provider.reload_role(WorkerRole.CHAT, wait=True)

    work.wait_unless_woken.assert_called_once_with(
        prov_mod._SHUTDOWN_WORK_WAIT_S + prov_mod._RELOAD_START_WAIT_S
    )
