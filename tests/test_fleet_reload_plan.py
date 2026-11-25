"""A settings write restarts only the group whose launch changed, and no request sees a gap."""

from __future__ import annotations

import os
import stat as stat_module
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from filelock import Timeout as FileLockTimeout

from lilbee.app.settings import _reload_changed_roles
from lilbee.core.config import cfg
from lilbee.providers import engine_params, model_cache
from lilbee.providers.base import SERVER_STOPPING_MESSAGE, ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.client import LlamaServerClient, StreamInterruptedError
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.roles import WorkerRole
from lilbee.runtime import engine_lock as engine_lock_mod
from lilbee.runtime.engine_lock import build_lock
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
# A drain limit that the test ends first, by releasing the held chat.
_LONG_DRAIN_S = 20.0
# How long a peer waits for the build lock in these tests (90 s in production).
_PEER_LOCK_WAIT_S = 1.0
_RESTARTED = "The chat model server stopped answering and was restarted"
_RELOADING = (
    "The chat model is reloading after a settings change, so this request was cut. "
    "Send your request again."
)
_QUESTION = [{"role": "user", "content": "hi"}]
_START_REFUSED = "The engine did not start."
_NOTES_CHUNK_SIZE = 384
_NOTES_CHUNK_OVERLAP = 64
# A chunk of tokens plus the 128-token input headroom and the 8-token margin.
_EMBED_WINDOW_AT_512 = 648
_EMBED_WINDOW_AT_384 = 520
# The plan shares the host RAM out as prompt caches. With this much RAM every cache gets its
# ceiling, so a role that starts or stops does not resize the cache of another role.
_HOST_RAM = 64 * _GB


@pytest.fixture
def real_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The real planner, with fakes only at the engine binary, device probe, GGUF read and RAM."""
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
    monkeypatch.setattr(model_cache, "total_system_memory", lambda: _HOST_RAM)
    monkeypatch.setattr(planning_mod, "total_system_memory", lambda: _HOST_RAM)
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
        assert f"Other processes hold {8 - free_gb}.0 GiB of the card's 8.0 GiB." in reason


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
    yield from _serve_mlx_chat(monkeypatch, tmp_path)


def _serve_mlx_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[FleetProvider, list[_EngineSwap]]]:
    """Serve the real plan from fake engines, each chat server read as an MLX server."""

    def adapter_of(_provider: FleetProvider, launch: InstanceLaunch) -> object:
        return MLX if launch.role is WorkerRole.CHAT else LLAMA

    monkeypatch.setattr(FleetProvider, "_engine_of", adapter_of)
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
    with pytest.raises(ProviderError, match="is reloading after a settings change"):
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


def _time_out_a_chat(provider: FleetProvider) -> LlamaServerClient:
    """Send a chat that times out on the MLX chat server, which starts its restart.

    Returns the timed-out client once the restart has taken it out of dispatch.
    """
    client = provider._clients[WorkerRole.CHAT][0]
    # httpx fixes a request's timeout when it sends it, so only the next request times out.
    client._http.timeout = httpx.Timeout(_DRAIN_LIMIT_S)
    with pytest.raises(ProviderError, match="not responding"):
        provider.chat(_QUESTION)
    assert poll_until(lambda: client not in provider._clients.get(WorkerRole.CHAT, []))
    return client


def _held_chat(provider: FleetProvider, answers: list[object]) -> threading.Thread:
    """Start a chat that the first engine holds; its answer or error goes to *answers*."""

    def ask() -> None:
        try:
            answers.append(provider.chat(_QUESTION))
        except ProviderError as err:
            answers.append(err)

    caller = threading.Thread(target=ask, daemon=True)
    caller.start()
    assert provider._swap_for(WorkerRole.CHAT).engine.started.wait(_JOIN_S)
    return caller


def _texts(answers: list[object]) -> list[str]:
    """The text of each answer, or the message of each error."""
    return [
        str(answer) if isinstance(answer, ProviderError) else answer.text  # type: ignore[attr-defined]
        for answer in answers
    ]


def _peer_takes_the_build_lock(engine_dir: Path) -> bool:
    """Whether a peer gets the build lock of *engine_dir* within its wait."""
    try:
        with build_lock(engine_dir):
            return True
    except FileLockTimeout:
        return False


def test_a_peer_takes_the_build_lock_while_a_timeout_restart_drains(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain runs before the build lock, so a peer process is not shut out for its length."""
    provider, swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    monkeypatch.setattr(engine_lock_mod, "_BUILD_LOCK_TIMEOUT_S", _PEER_LOCK_WAIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    engine_dir = provider._reload_dir()
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _time_out_a_chat(provider)

    took_the_lock = _peer_takes_the_build_lock(engine_dir)
    still_draining = not old.engine.stopped.is_set()
    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert took_the_lock
    assert still_draining
    assert _texts(answers) == ["the answer"]
    assert old.engine.requests == 2
    assert swaps[-1] is not old
    assert swaps[-1].engine.requests == 0


def test_a_chat_with_no_answer_at_the_drain_limit_is_sent_once_to_the_new_server(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    embed = provider._swap_for(WorkerRole.EMBED)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _time_out_a_chat(provider)

    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert _texts(answers) == ["the answer"]
    assert old.engine.stopped.is_set()
    assert old.engine.requests == 2
    assert swaps[-1] is not old
    assert swaps[-1].engine.requests == 1
    assert swaps[-1].engine.completed == 1
    assert provider._swap_for(WorkerRole.EMBED) is embed


def test_a_stream_that_sent_text_ends_with_the_restart_error_at_the_drain_limit(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    try:
        assert old.engine.started.wait(_JOIN_S)
        _time_out_a_chat(provider)
        assert poll_until(lambda: old.engine.stopped.is_set())
        with pytest.raises(ProviderError) as caught:
            list(stream)
        _join_fleet_threads()
    finally:
        stream.close()

    assert str(caught.value).startswith(_RESTARTED)
    assert "settings change" not in str(caught.value)
    assert swaps[-1] is not old
    assert swaps[-1].engine.requests == 0


def test_a_request_that_waits_out_a_timeout_restart_gets_the_restart_error(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    monkeypatch.setattr(prov_mod, "_RELOAD_GAP_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _time_out_a_chat(provider)

    try:
        with pytest.raises(ProviderError) as caught:
            provider.chat(_QUESTION)
    finally:
        old.engine.release.set()
        caller.join(_JOIN_S)
        _join_fleet_threads()

    assert str(caught.value).startswith(_RESTARTED)
    assert "settings change" not in str(caught.value)


def test_a_restart_after_a_timeout_restart_keeps_the_restart_error(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead-server rebuild asked before the pass runs does not undo the timeout restart."""
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    try:
        assert old.engine.started.wait(_JOIN_S)
        # The fleet work waits for the lock, so it reads both restarts at once.
        with provider._lock:
            provider._force_restart(WorkerRole.CHAT, stopped_answering=True)
            provider._force_restart(WorkerRole.CHAT, stopped_answering=False)
        assert poll_until(lambda: old.engine.stopped.is_set())
        with pytest.raises(ProviderError) as caught:
            list(stream)
        _join_fleet_threads()
    finally:
        stream.close()

    assert str(caught.value).startswith(_RESTARTED)


def test_a_timeout_restart_whose_plan_fails_gives_the_server_back(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed pass leaves the running fleet as it was, so the next timeout restarts again."""
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)

    def no_plan(**_kwargs: object) -> object:
        raise ProviderError("The engine binary is missing.")

    monkeypatch.setattr(planning_mod, "plan_all_launches", no_plan)
    old = provider._swap_for(WorkerRole.CHAT)
    client = provider._clients[WorkerRole.CHAT][0]
    client._http.timeout = httpx.Timeout(_DRAIN_LIMIT_S)
    with pytest.raises(ProviderError, match="not responding"):
        provider.chat(_QUESTION)
    _join_fleet_threads()

    assert provider._clients.get(WorkerRole.CHAT) == [client]
    assert client not in provider._retiring_clients
    assert client._on_idle is None
    assert client.restart_message is None
    assert provider._swap_for(WorkerRole.CHAT) is old
    assert not old.engine.stopped.is_set()


def test_a_cancel_during_a_timeout_drain_ends_a_stream_on_the_stuck_server_at_once(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    try:
        assert old.engine.started.wait(_JOIN_S)
        _time_out_a_chat(provider)
        provider.cancel_inference()
        with pytest.raises(ProviderError) as caught:
            list(stream)
        draining = not old.engine.stopped.is_set()
    finally:
        old.engine.release.set()
        stream.close()
        _join_fleet_threads()

    assert draining
    assert isinstance(caught.value, StreamInterruptedError)
    assert not str(caught.value).startswith(_RESTARTED)


def test_a_shutdown_during_a_timeout_drain_closes_the_stuck_clients(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pass outruns the shutdown's wait, so only the shutdown can close the stuck clients."""
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    plan_may_run = threading.Event()
    real_plan = planning_mod.plan_all_launches

    def held_plan(**kwargs: object) -> object:
        plan_may_run.wait(_JOIN_S)
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", held_plan)
    old = provider._swap_for(WorkerRole.CHAT)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    stuck = _time_out_a_chat(provider)
    stopper = threading.Thread(target=provider.shutdown, daemon=True)
    try:
        stopper.start()
        closed = poll_until(lambda: stuck._http.is_closed)
    finally:
        plan_may_run.set()
        old.engine.release.set()
        stopper.join(_JOIN_S)
        caller.join(_JOIN_S)
        _join_fleet_threads()

    assert closed


def test_a_shutdown_ends_the_warm_that_a_reload_started(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new chat engine holds the warm request, so only the shutdown can end the warm."""
    provider, swaps = engine_fleet
    monkeypatch.delattr(provider, "_preload_roles")
    first_new = len(swaps)
    # Only the warms this test starts count; another test's thread is not this one's leak.
    before = set(threading.enumerate())

    def held_swap(_dir: Path, _group: object) -> _EngineSwap:
        swap = _EngineSwap(answers_at_once=False)
        swaps.append(swap)
        return swap

    monkeypatch.setattr(prov_mod, "SwapManager", held_swap)
    cfg.chat_model = _OTHER_CHAT_REF
    provider.reload_role(WorkerRole.CHAT)
    assert poll_until(lambda: any(swap.engine.started.is_set() for swap in swaps[first_new:]))

    provider.shutdown()

    warms = [
        thread
        for thread in threading.enumerate()
        if thread.name == "fleet-reload-warm" and thread not in before
    ]
    assert not [thread for thread in warms if thread.is_alive()]


def test_a_retry_that_fails_on_the_new_server_starts_no_second_server(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart retry runs once: a new server that dies at once is not restarted again."""
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    embed = provider._swap_for(WorkerRole.EMBED)
    starts: list[_EngineSwap] = []
    real_start = _EngineSwap.start

    def start_then_die(self: _EngineSwap, launches: list[object], **kwargs: object) -> None:
        real_start(self, launches, **kwargs)
        starts.append(self)
        self.engine.stop()

    monkeypatch.setattr(_EngineSwap, "start", start_then_die)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _time_out_a_chat(provider)
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert not caller.is_alive()
    assert len(answers) == 1
    assert isinstance(answers[0], ProviderError)
    assert old.engine.requests == 2
    assert len(starts) == 1
    assert provider._swap_for(WorkerRole.EMBED) is embed


def _fail_first_starts(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN,
) -> list[_EngineSwap]:
    """Make the first *count* server starts fail with *kind*; returns each swap asked to start."""
    return _fail_starts(monkeypatch, lambda number: number <= count, kind)


def _fail_starts(
    monkeypatch: pytest.MonkeyPatch,
    fails: Callable[[int], bool],
    kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN,
) -> list[_EngineSwap]:
    """Make server start number n (from 1) fail when *fails* (n); returns each swap asked."""
    starts: list[_EngineSwap] = []
    real_start = _EngineSwap.start

    def start(self: _EngineSwap, launches: list[object], **kwargs: object) -> None:
        starts.append(self)
        if fails(len(starts)):
            raise ProviderError(_START_REFUSED, kind=kind)
        real_start(self, launches, **kwargs)

    monkeypatch.setattr(_EngineSwap, "start", start)
    return starts


def _ask(provider: FleetProvider) -> object:
    """One chat: its answer, or the error it raised."""
    try:
        return provider.chat(_QUESTION)
    except ProviderError as err:
        return err


def test_a_chat_after_a_restart_that_did_not_start_names_the_failed_start(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restart stops the chat server and cannot start it: the next chat says why."""
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    starts = _fail_first_starts(monkeypatch, 2)
    _time_out_a_chat(provider)
    _join_fleet_threads()

    with pytest.raises(ProviderError) as heard:
        provider.chat(_QUESTION)
    _join_fleet_threads()

    assert str(heard.value).startswith(
        f"The chat model server was restarted and did not start again: {_START_REFUSED}"
    )
    assert "Send your request again" in str(heard.value)
    assert old.engine.stopped.is_set()
    assert len(starts) == 2


def test_a_chat_after_a_restart_that_did_not_start_starts_the_server_again(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    embed = provider._swap_for(WorkerRole.EMBED)
    starts = _fail_first_starts(monkeypatch, 1)
    _time_out_a_chat(provider)
    _join_fleet_threads()
    assert provider._swap_for(WorkerRole.CHAT) is None

    answers = [_ask(provider)]
    _join_fleet_threads()

    assert _texts(answers) == ["the answer"]
    assert len(starts) == 2
    assert provider._swap_for(WorkerRole.CHAT) is starts[-1]
    assert provider._swap_for(WorkerRole.EMBED) is embed


def test_a_chat_after_a_reload_that_did_not_start_starts_the_server_again(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settings reload whose start fails leaves the same gap; the next chat closes it."""
    provider, _swaps = engine_fleet
    starts = _fail_first_starts(monkeypatch, 1)
    _reload_chat_model(provider)
    _join_fleet_threads()
    assert provider._swap_for(WorkerRole.CHAT) is None

    answers = [_ask(provider)]
    _join_fleet_threads()

    assert _texts(answers) == ["the answer"]
    assert provider._swap_for(WorkerRole.CHAT) is starts[-1]
    assert provider._launches[provider._role_group[WorkerRole.CHAT]][0].model == _OTHER_CHAT_REF


def test_a_server_that_started_again_drops_its_failed_start(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the start succeeds, a later failed rebuild reports its own cause, not the old start."""
    provider, _swaps = engine_fleet
    _fail_first_starts(monkeypatch, 1)
    _reload_chat_model(provider)
    _join_fleet_threads()
    assert _texts([_ask(provider)]) == ["the answer"]
    _join_fleet_threads()
    assert provider._failed_starts == {}

    def no_plan(**_kwargs: object) -> object:
        raise ProviderError("The engine binary is missing.")

    monkeypatch.setattr(planning_mod, "plan_all_launches", no_plan)
    with pytest.raises(ProviderError) as heard:
        provider._rebuild_role(WorkerRole.CHAT)
    _join_fleet_threads()

    assert str(heard.value) == "The engine binary is missing."


class _LadderReached(Exception):
    """The build got past its demand plan."""


def test_a_build_drops_an_earlier_failed_start(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A build plans every role afresh, so a role it does not serve keeps no failed start."""
    provider, _swaps = engine_fleet
    provider._failed_starts = {WorkerRole.CHAT: ProviderError(_START_REFUSED)}

    def reached() -> Path:
        raise _LadderReached

    monkeypatch.setattr(prov_mod, "machine_engine_dir", reached)
    with pytest.raises(_LadderReached):
        provider._acquire_engine(tmp_path)

    assert provider._failed_starts == {}


# A request wait bound that the held plan of the next start always outlasts.
_SHORT_FLEET_WAIT_S = 0.05


def _restart_that_does_not_start(
    provider: FleetProvider, monkeypatch: pytest.MonkeyPatch, fails: Callable[[int], bool]
) -> list[_EngineSwap]:
    """Time out a chat so its restart stops the chat server; starts fail when *fails* (n)."""
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    starts = _fail_starts(monkeypatch, fails)
    _time_out_a_chat(provider)
    _join_fleet_threads()
    assert WorkerRole.CHAT in provider._failed_starts
    return starts


def _hold_the_next_plan(
    monkeypatch: pytest.MonkeyPatch, release: threading.Event, at_hold: Callable[[], None]
) -> None:
    """Hold the next pass before its plan until *release*; *at_hold* runs when it holds."""
    real_refresh = planning_mod.refresh_plan_devices

    def held(*args: object, **kwargs: object) -> None:
        at_hold()
        assert release.wait(_JOIN_S)
        real_refresh(*args, **kwargs)

    monkeypatch.setattr(planning_mod, "refresh_plan_devices", held)


def test_a_chat_that_waits_out_the_next_start_hears_that_it_is_starting(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next start still runs at the request's wait bound: that start has not failed."""
    provider, _swaps = mlx_engine_fleet
    starts = _restart_that_does_not_start(provider, monkeypatch, lambda number: number == 1)
    release = threading.Event()
    _hold_the_next_plan(monkeypatch, release, lambda: None)
    monkeypatch.setattr(prov_mod, "_FLEET_WAIT_S", _SHORT_FLEET_WAIT_S)
    try:
        heard = _ask(provider)
    finally:
        release.set()
    _join_fleet_threads()

    assert isinstance(heard, ProviderError)
    assert str(heard).startswith("The engines are still starting after ")
    assert heard.kind is ProviderErrorKind.STARTING
    assert _texts([_ask(provider)]) == ["the answer"]
    assert len(starts) == 2


def test_a_chat_whose_next_plan_fails_after_a_failed_start_hears_the_plan_error(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed plan is not a failed start; the role stays recorded, so the next chat starts it."""
    provider, _swaps = mlx_engine_fleet
    _restart_that_does_not_start(provider, monkeypatch, lambda number: number == 1)
    real_plan = planning_mod.plan_all_launches
    plans: list[object] = []

    def first_plan_fails(**kwargs: object) -> object:
        plans.append(kwargs)
        if len(plans) == 1:
            raise ProviderError("The engine binary is missing.")
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", first_plan_fails)
    heard = _ask(provider)
    _join_fleet_threads()

    assert isinstance(heard, ProviderError)
    assert str(heard) == "The engine binary is missing."
    assert _texts([_ask(provider)]) == ["the answer"]


def test_a_stop_during_the_next_start_after_a_failed_start_says_the_server_is_stopping(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    _restart_that_does_not_start(provider, monkeypatch, lambda number: number == 1)
    release = threading.Event()
    _hold_the_next_plan(monkeypatch, release, provider.stop_waiting)
    try:
        heard = _ask(provider)
    finally:
        release.set()
    _join_fleet_threads()

    assert isinstance(heard, ProviderError)
    assert str(heard) == SERVER_STOPPING_MESSAGE
    assert heard.kind is ProviderErrorKind.STARTING


def _force_chat_and_embed(provider: FleetProvider) -> None:
    """Run one forced pass over the chat and embed groups; it raises the failed start."""
    with provider._lock:
        assert provider._role_group[WorkerRole.CHAT] is not provider._role_group[WorkerRole.EMBED]
        provider._forced_roles[WorkerRole.CHAT] = False
        provider._forced_roles[WorkerRole.EMBED] = False
        work = provider._start_fleet_work()
    with pytest.raises(ProviderError, match=_START_REFUSED):
        provider._join_fleet_work(work)
    _join_fleet_threads()


def test_a_failed_start_records_every_group_the_pass_did_not_start(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first start fails, so the later group has no server either; the next chat starts both."""
    provider, _swaps = engine_fleet
    starts = _fail_starts(monkeypatch, lambda number: number == 1)
    _force_chat_and_embed(provider)

    assert len(starts) == 1
    assert {WorkerRole.CHAT, WorkerRole.EMBED} <= set(provider._failed_starts)
    assert _texts([_ask(provider)]) == ["the answer"]
    _join_fleet_threads()
    assert provider._failed_starts == {}
    assert provider._clients.get(WorkerRole.EMBED)


def test_a_failed_start_does_not_record_a_group_the_pass_started(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = engine_fleet
    _fail_starts(monkeypatch, lambda number: number == 2)
    _force_chat_and_embed(provider)

    up = {role for role in (WorkerRole.CHAT, WorkerRole.EMBED) if provider._clients.get(role)}
    assert len(up) == 1
    assert not up & set(provider._failed_starts)
    assert {WorkerRole.CHAT, WorkerRole.EMBED} - up <= set(provider._failed_starts)


def test_the_failed_start_message_keeps_the_kind_of_its_cause(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, _swaps = mlx_engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    _fail_first_starts(monkeypatch, 2, ProviderErrorKind.SERVER)
    _time_out_a_chat(provider)
    _join_fleet_threads()

    heard = _ask(provider)
    _join_fleet_threads()

    assert isinstance(heard, ProviderError)
    assert "did not start again" in str(heard)
    assert heard.kind is ProviderErrorKind.SERVER


def test_two_chats_after_a_failed_start_share_one_start(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second chat waits for the start the first chat drives, and starts no server itself."""
    provider, _swaps = mlx_engine_fleet
    starts = _restart_that_does_not_start(provider, monkeypatch, lambda number: number == 1)
    gate = _SignalCondition(provider._lock)
    provider._reload_done = gate
    starting = threading.Event()
    release = threading.Event()
    failing_start = _EngineSwap.start

    def held_start(self: _EngineSwap, launches: list[object], **kwargs: object) -> None:
        starting.set()
        assert release.wait(_JOIN_S)
        failing_start(self, launches, **kwargs)

    monkeypatch.setattr(_EngineSwap, "start", held_start)
    answers: list[object] = []
    chats = [threading.Thread(target=lambda: answers.append(_ask(provider))) for _ in range(2)]
    try:
        chats[0].start()
        assert starting.wait(_JOIN_S)
        chats[1].start()
        assert gate.entered.wait(_JOIN_S)
    finally:
        release.set()
        for chat in [chat for chat in chats if chat.ident is not None]:
            chat.join(_JOIN_S)
    _join_fleet_threads()

    assert _texts(answers) == ["the answer", "the answer"]
    assert len(starts) == 2


def test_a_late_timeout_from_a_replaced_server_restarts_nothing(
    mlx_engine_fleet: tuple[FleetProvider, list[_EngineSwap]],
) -> None:
    provider, swaps = mlx_engine_fleet
    old = provider._swap_for(WorkerRole.CHAT)
    provider._clients[WorkerRole.CHAT][0]._http.timeout = httpx.Timeout(_DRAIN_LIMIT_S)
    with pytest.raises(ProviderError, match="not responding"):
        provider.chat(_QUESTION)
    _join_fleet_threads()
    fresh = provider._swap_for(WorkerRole.CHAT)
    assert fresh is not old

    sent = httpx.Request("POST", f"{old.engine.url}/v1/chat/completions")

    def late_timeout() -> None:
        raise httpx.ReadTimeout("late", request=sent)

    with pytest.raises(ProviderError, match="not responding"):
        provider._with_rediscover(late_timeout, role=WorkerRole.CHAT)
    _join_fleet_threads()

    assert not fresh.engine.stopped.is_set()
    assert fresh is swaps[-1]


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
    with pytest.raises(ProviderError) as caught:
        list(stream)
    _join_fleet_threads()

    assert str(caught.value) == _RELOADING
    assert caught.value.kind is ProviderErrorKind.CONNECTION
    assert old.answers_at_stop == 0


def test_a_stream_cut_after_its_change_was_undone_still_hears_of_the_reload(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chat model is set back during the drain, so the last plan no longer differs."""
    provider, _swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _DRAIN_LIMIT_S)
    real_plan = planning_mod.plan_all_launches
    plans: list[int] = []

    def undo_before_the_second_plan(**kwargs: object) -> object:
        plans.append(1)
        if len(plans) == 2:
            cfg.chat_model = _CHAT_REF
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", undo_before_the_second_plan)
    old = provider._swap_for(WorkerRole.CHAT)
    stream = provider.chat(_QUESTION, stream=True)
    assert old.engine.started.wait(_JOIN_S)

    _reload_chat_model(provider)
    assert poll_until(lambda: old.engine.stopped.is_set())
    with pytest.raises(ProviderError) as caught:
        list(stream)
    _join_fleet_threads()

    assert len(plans) >= 2
    assert str(caught.value) == _RELOADING


_PINNED_CHAT_CTX = 4096


def test_a_group_that_changes_again_during_its_drain_starts_once_from_the_last_plan(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second write lands during the drain: the chat group drains once and starts once."""
    provider, swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    real_plan = planning_mod.plan_all_launches
    plans: list[int] = []

    def pin_the_window_before_the_second_plan(**kwargs: object) -> object:
        plans.append(1)
        if len(plans) == 2:
            monkeypatch.setattr(cfg, "num_ctx", _PINNED_CHAT_CTX)
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", pin_the_window_before_the_second_plan)
    old = provider._swap_for(WorkerRole.CHAT)
    started = len(swaps)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _reload_chat_model(provider)

    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert _texts(answers) == ["the answer"]
    assert len(plans) == 2
    assert old.shutdowns == 1
    assert old.answers_at_stop == 1
    assert len(swaps) == started + 1
    new = provider._swap_for(WorkerRole.CHAT)
    assert new is swaps[-1]
    [launches] = new.started
    assert {launch.ctx for launch in launches if launch.role is WorkerRole.CHAT} == {
        _PINNED_CHAT_CTX
    }
    assert {launch.model for launch in launches if launch.role is WorkerRole.CHAT} == {
        _OTHER_CHAT_REF
    }


def test_a_peer_takes_the_build_lock_while_a_settings_reload_drains(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No hold of the build lock waits for a request, so the drain shuts no peer process out."""
    provider, swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    monkeypatch.setattr(engine_lock_mod, "_BUILD_LOCK_TIMEOUT_S", _PEER_LOCK_WAIT_S)
    old = provider._swap_for(WorkerRole.CHAT)
    engine_dir = provider._reload_dir()
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _reload_chat_model(provider)

    took_the_lock = _peer_takes_the_build_lock(engine_dir)
    still_draining = not old.engine.stopped.is_set()
    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert took_the_lock
    assert still_draining
    assert _texts(answers) == ["the answer"]
    assert old.answers_at_stop == 1
    assert swaps[-1] is not old
    assert swaps[-1].engine.requests == 0


def test_a_reload_whose_plan_fails_after_the_drain_gives_the_server_back(
    engine_fleet: tuple[FleetProvider, list[_EngineSwap]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan of the hold that follows the drain fails, so the fleet stays as it was."""
    provider, swaps = engine_fleet
    monkeypatch.setattr(prov_mod, "_SHUTDOWN_WORK_WAIT_S", _LONG_DRAIN_S)
    real_plan = planning_mod.plan_all_launches
    plans: list[int] = []

    def plan_once(**kwargs: object) -> object:
        plans.append(1)
        if len(plans) > 1:
            raise ProviderError("The engine binary is missing.")
        return real_plan(**kwargs)

    monkeypatch.setattr(planning_mod, "plan_all_launches", plan_once)
    old = provider._swap_for(WorkerRole.CHAT)
    client = provider._clients[WorkerRole.CHAT][0]
    started = len(swaps)
    answers: list[object] = []
    caller = _held_chat(provider, answers)
    _reload_chat_model(provider)

    old.engine.release.set()
    caller.join(_JOIN_S)
    _join_fleet_threads()

    assert _texts(answers) == ["the answer"]
    assert len(plans) == 2
    assert provider._clients.get(WorkerRole.CHAT) == [client]
    assert client not in provider._retiring_clients
    assert client.restart_message is None
    assert provider._swap_for(WorkerRole.CHAT) is old
    assert not old.engine.stopped.is_set()
    assert len(swaps) == started


class _StateFileHeld(OSError):
    """A peer holds the state file of a server open, so its stop cannot remove the file."""


def _refuse_a_closed_embed(client: MagicMock) -> None:
    """Make *client* refuse an embed once it is closed, as its HTTP client does."""

    def embed(texts: list[str]) -> list[list[float]]:
        if client.close.called:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return [[0.0] for _ in texts]

    client.embed.side_effect = embed


def test_a_failed_stop_of_one_group_leaves_the_next_group_open_in_dispatch(
    real_planner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chat server stops first and its stop raises; the embed server still serves."""
    provider = _running_provider(monkeypatch, tmp_path)
    embed_clients = list(provider._clients[WorkerRole.EMBED])
    for client in embed_clients:
        _refuse_a_closed_embed(client)

    def refuse_stop() -> None:
        raise _StateFileHeld("The state file is in use.")

    monkeypatch.setattr(provider._swap_for(WorkerRole.CHAT), "shutdown", refuse_stop)
    cfg.chat_model = _OTHER_CHAT_REF
    cfg.chunk_size = _NOTES_CHUNK_SIZE

    with pytest.raises(_StateFileHeld):
        provider._reload_pass()

    assert provider._clients[WorkerRole.EMBED] == embed_clients
    for client in embed_clients:
        client.close.assert_not_called()
    assert provider.embed(["a note"]) == [[0.0]]


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
