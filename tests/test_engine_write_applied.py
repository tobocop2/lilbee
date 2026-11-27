"""An engine write tells the truth about the running engine, on a fleet of real engine processes.

Each swap group is an OS process running the fake engine; only llama-swap is replaced. The
engine book, the plan, the reload and the engine service are the product's own.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import psutil
import pytest
from typer.testing import CliRunner

from lilbee.app import engine_service
from lilbee.app import services as services_mod
from lilbee.app.engine_choice import resolve_engines
from lilbee.app.services import Services, build_services
from lilbee.cli import app as cli_app
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning
from lilbee.providers.fleet import provider as fleet_provider
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.provider import RELOAD_FAILED_NOTICE, FleetProvider
from lilbee.providers.roles import WorkerRole
from tests.conftest import CALLER_THREAD
from tests.test_engine_real_fleet import _GIB, _MESSAGES, GGUF_CHAT, GGUF_EMBED, MLX_REF, _Gguf

_REPO = Path(__file__).resolve().parents[1]
_WAIT_S = 10.0
_ENGINE_SCRIPT = """\
import os
import sys
import threading

sys.path.insert(0, sys.argv[1])
from tests.conformance import fake_engine


def _exit_with_parent() -> None:
    sys.stdin.read()
    os._exit(0)


threading.Thread(target=_exit_with_parent, daemon=True).start()
server = fake_engine._Server(fake_engine.Behaviour())
print(server.server_address[1], flush=True)
server.serve_forever()
"""
_PARENT_SCRIPT = """\
import os
import subprocess
import sys

child = subprocess.Popen(
    [sys.executable, "-B", sys.argv[1], sys.argv[2]],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
port = child.stdout.readline().strip()
print(child.pid, port, flush=True)
os._exit(0)
"""


class _Mlx(_Gguf):
    name = "mlx"
    formats = frozenset({"gguf", "mlx"})

    def __hash__(self) -> int:
        return hash(self.name)

    def launch_argv(self, spec: Any, request: Any) -> list[str]:
        return [*super().launch_argv(spec, request), "--mlx"]


class _ProcessSwap:
    """Stands in for llama-swap only: each group is a real fake-engine process."""

    script: ClassVar[Path]
    live: ClassVar[dict[Any, _ProcessSwap]] = {}
    started: ClassVar[list[_ProcessSwap]] = []
    start_failure: ClassVar[BaseException | None] = None
    start_gate: ClassVar[threading.Event | None] = None
    starting: ClassVar[threading.Event] = threading.Event()

    def __init__(self, data_dir: Path, group: Any) -> None:
        self.group = group
        self.log_path = data_dir / "swap.log"
        self.process: subprocess.Popen[str] | None = None
        self.port = 0

    def start(self, launches: list[Any], **kwargs: Any) -> None:
        type(self).starting.set()
        if type(self).start_gate is not None:
            type(self).start_gate.wait()
        if type(self).start_failure is not None:
            raise type(self).start_failure
        self.process = subprocess.Popen(
            [sys.executable, "-B", str(self.script), str(_REPO)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        type(self).started.append(self)
        assert self.process.stdout is not None
        self.port = int(self.process.stdout.readline())
        type(self).live[self.group] = self

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def shutdown(self, *_a: Any, **_k: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(_WAIT_S)

    def __getattr__(self, name: str) -> Any:
        if name == "health_warnings":
            return lambda *a, **k: []
        if name in {"role_ready", "is_live", "running"}:
            return lambda *a, **k: True
        if name == "bound":
            return False
        return lambda *a, **k: None


def _alive(swap: _ProcessSwap) -> bool:
    assert swap.process is not None
    return swap.process.poll() is None


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Services]:
    """A running server: chat and embed on llama.cpp, mlx installed, engines are real processes."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Gguf())
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, _Mlx())
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 64 * _GIB)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: 64 * _GIB)
    settings: dict[str, Any] = {
        "system_memory_reserve_gb": 0,
        "gpu_memory_fraction": 1.0,
        "num_ctx": None,
        "placement": None,
        "reranker_model": "",
        "vision_model": "",
        "data_root": tmp_path,
        "documents_dir": tmp_path / "documents",
        "data_dir": tmp_path / "data",
        "lancedb_dir": tmp_path / "data" / "lancedb",
        "worker_pool_eager_start": False,
        "chat_model": GGUF_CHAT,
        "embedding_model": GGUF_EMBED,
        "preferred_engine": EngineSetting.LLAMA,
        "engine_choices": None,
    }
    for key, value in settings.items():
        monkeypatch.setattr(cfg, key, value)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "engine.py"
    script.write_text(_ENGINE_SCRIPT, encoding="utf-8")
    monkeypatch.setattr(_ProcessSwap, "script", script, raising=False)
    monkeypatch.setattr(_ProcessSwap, "live", {})
    monkeypatch.setattr(_ProcessSwap, "started", [])
    monkeypatch.setattr(_ProcessSwap, "start_failure", None)
    monkeypatch.setattr(_ProcessSwap, "start_gate", None)
    monkeypatch.setattr(_ProcessSwap, "starting", threading.Event())
    monkeypatch.setattr(fleet_provider, "SwapManager", _ProcessSwap)
    services = build_services(cfg)
    monkeypatch.setattr(services_mod._state, "singleton", services)
    try:
        assert services.provider.embed(["hi"])
        assert services.provider.chat(_MESSAGES)
        assert len(_ProcessSwap.live) > 0
        _ProcessSwap.starting.clear()
        yield services
    finally:
        _stop_every_engine(services)


def _stop_every_engine(services: Services) -> None:
    """Stop each engine process the fixture started, even when the provider's own stop fails."""
    try:
        services.provider.shutdown()
    finally:
        for swap in _ProcessSwap.started:
            if swap.process is not None and swap.process.poll() is None:
                swap.process.kill()
                swap.process.wait(_WAIT_S)
        services.store.close()


def _local(services: Services) -> FleetProvider:
    local = services.provider._local
    assert isinstance(local, FleetProvider)
    return local


def _wait_for(condition: Any) -> None:
    deadline = time.monotonic() + _WAIT_S
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


def _running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_an_engine_process_exits_when_its_parent_exits_without_stopping_it(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine.py"
    engine.write_text(_ENGINE_SCRIPT, encoding="utf-8")
    parent = tmp_path / "parent.py"
    parent.write_text(_PARENT_SCRIPT, encoding="utf-8")
    ran = subprocess.run(
        [sys.executable, "-B", str(parent), str(engine), str(_REPO)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=_WAIT_S * 3,
        check=True,
    )
    pid, port = (int(field) for field in ran.stdout.split())
    assert port > 0
    try:
        _wait_for(lambda: not _running(pid))
    finally:
        if _running(pid):
            psutil.Process(pid).kill()


def _swap_with_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> _ProcessSwap:
    script = tmp_path / "engine.py"
    script.write_text(source, encoding="utf-8")
    monkeypatch.setattr(_ProcessSwap, "script", script, raising=False)
    monkeypatch.setattr(_ProcessSwap, "live", {})
    monkeypatch.setattr(_ProcessSwap, "started", [])
    monkeypatch.setattr(_ProcessSwap, "start_failure", None)
    monkeypatch.setattr(_ProcessSwap, "start_gate", None)
    monkeypatch.setattr(_ProcessSwap, "starting", threading.Event())
    return _ProcessSwap(tmp_path, "group")


def test_an_engine_process_exits_when_its_swap_closes_the_pipe_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swap = _swap_with_script(tmp_path, monkeypatch, _ENGINE_SCRIPT)
    swap.start([])
    assert swap.process is not None
    assert swap.process.stdin is not None
    try:
        swap.process.stdin.close()
        assert swap.process.wait(_WAIT_S) == 0
    finally:
        swap.shutdown()


def test_an_engine_that_prints_no_port_is_still_listed_as_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swap = _swap_with_script(tmp_path, monkeypatch, "raise SystemExit(0)\n")
    with pytest.raises(ValueError, match="invalid literal"):
        swap.start([])
    assert _ProcessSwap.started == [swap]


def test_a_failing_provider_shutdown_still_stops_every_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swap = _swap_with_script(tmp_path, monkeypatch, _ENGINE_SCRIPT)
    swap.start([])
    assert swap.process is not None
    closed: list[bool] = []

    class _Provider:
        def shutdown(self) -> None:
            raise RuntimeError("provider shutdown failed")

    class _Store:
        def close(self) -> None:
            closed.append(True)

    services: Any = SimpleNamespace(provider=_Provider(), store=_Store())
    with pytest.raises(RuntimeError, match="provider shutdown failed"):
        _stop_every_engine(services)
    assert swap.process.poll() is not None
    assert closed == [True]


def test_a_write_that_cannot_find_the_engine_binary_is_not_reported_as_applied(
    fleet: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {group: swap for group, swap in _ProcessSwap.live.items()}

    def _no_binary(*, engines: Any) -> None:
        raise ProviderError(
            "llama-server binary not found.",
            kind=ProviderErrorKind.NOT_FOUND,
        )

    monkeypatch.setattr(planning, "plan_all_launches", _no_binary)
    with pytest.raises(engine_service.EngineNotAppliedError, match="binary not found"):
        engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, "mlx")
    assert cfg.engine_choices is not None
    assert all(_alive(swap) for swap in before.values())
    assert _local(fleet).reload_notice() == RELOAD_FAILED_NOTICE


@pytest.mark.parametrize(
    "first_write",
    [
        lambda: engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, "mlx"),
        lambda: engine_service.prefer("mlx"),
    ],
    ids=["one-group-still-up", "every-group-stopped"],
)
def test_a_write_that_joins_a_reload_that_fails_reports_the_failure(
    fleet: Services, first_write: Any, joined: Callable[[], None]
) -> None:
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    _ProcessSwap.start_failure = ProviderError("engine did not start")
    outcomes: dict[str, BaseException | str] = {}

    def write(name: str, action: Any) -> None:
        try:
            outcomes[name] = action()
        except engine_service.EngineNotAppliedError as exc:
            outcomes[name] = exc

    first = threading.Thread(target=write, args=("first", first_write), name=CALLER_THREAD)
    first.start()
    _ProcessSwap.starting.wait()
    second = threading.Thread(
        target=write,
        args=("second", lambda: engine_service.prefer("auto")),
        name=CALLER_THREAD,
    )
    second.start()
    joined()
    gate.set()
    first.join()
    second.join()
    assert {name: type(value) for name, value in outcomes.items()} == {
        "first": engine_service.EngineNotAppliedError,
        "second": engine_service.EngineNotAppliedError,
    }


def test_vision_at_the_bound_fails_the_call_and_leaves_no_vision_role(
    fleet: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "vision_model", GGUF_CHAT)
    monkeypatch.setattr(cfg, "ocr", "off")
    monkeypatch.setattr(fleet_provider, "_FLEET_WAIT_S", 0)
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    first = threading.Thread(target=fleet.provider.replan_fleet)
    local = _local(fleet)
    monkeypatch.setattr(cfg, "preferred_engine", EngineSetting.MLX)
    first.start()
    _ProcessSwap.starting.wait()
    work = local._fleet_work
    assert work is not None
    wanted = local._writes_wanted
    with pytest.raises(ProviderError, match=r"still starting after 0 s") as raised:
        local._serve_vision_on_request()
    assert raised.value.kind is ProviderErrorKind.STARTING
    assert not planning.vision_role_wanted(GGUF_CHAT)
    # One write asked for the grant; a second, after the revoke, plans without it.
    assert local._writes_wanted == wanted + 2
    gate.set()
    first.join()
    work.wait()
    assert local._fleet_work is None
    assert local._writes_done == local._writes_wanted
    assert WorkerRole.VISION not in {
        launch.role for group in local._launches.values() for launch in group
    }


def test_a_write_while_a_reload_has_every_group_stopped_reaches_the_engines(
    fleet: Services, joined: Callable[[], None]
) -> None:
    gate = threading.Event()
    _ProcessSwap.start_gate = gate
    first = threading.Thread(target=engine_service.prefer, args=("mlx",), name=CALLER_THREAD)
    first.start()
    _ProcessSwap.starting.wait()
    local = _local(fleet)
    assert local._swaps == {}
    second = threading.Thread(
        target=engine_service.use_for_role,
        args=(WorkerRole.CHAT, GGUF_CHAT, "llama.cpp"),
        name=CALLER_THREAD,
    )
    second.start()
    joined()
    gate.set()
    first.join()
    second.join()
    launches = {launch.role: launch.argv for group in local._launches.values() for launch in group}
    assert set(launches) == {WorkerRole.CHAT, WorkerRole.EMBED}
    assert "--mlx" not in launches[WorkerRole.CHAT]
    assert "--mlx" in launches[WorkerRole.EMBED]


def test_a_reload_that_fails_on_the_disk_is_not_applied(fleet: Services) -> None:
    _ProcessSwap.start_failure = OSError("no space left on the device")
    with pytest.raises(engine_service.EngineNotAppliedError, match="no space left"):
        engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, "mlx")


def test_the_failed_reload_notice_and_a_role_fallback_show_together(
    fleet: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ProcessSwap.start_failure = ProviderError("engine did not start")
    with pytest.raises(engine_service.EngineNotAppliedError):
        engine_service.use_for_role(WorkerRole.CHAT, GGUF_CHAT, "mlx")
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    monkeypatch.setattr(cfg, "engine_choices", None)
    monkeypatch.setattr(cfg, "preferred_engine", EngineSetting.MLX)
    monkeypatch.setattr(cfg, "chat_model", MLX_REF)
    fallbacks = [c.notice for c in resolve_engines(cfg).choices.values() if c.notice is not None]
    assert len(fallbacks) > 0
    messages = [w.message for w in fleet.provider.health_warnings()]
    assert RELOAD_FAILED_NOTICE in messages
    assert all(fallback in messages for fallback in fallbacks)


@pytest.mark.parametrize(
    "failure",
    [
        ProviderError("engine did not start"),
        OSError("no space left on the device"),
    ],
    ids=["provider-error", "disk-error"],
)
def test_a_background_reload_failure_is_logged_with_its_text_and_raises_no_trace(
    fleet: Services,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: BaseException,
) -> None:
    def _fail(*, engines: Any) -> None:
        raise failure

    unhandled: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(planning, "plan_all_launches", _fail)
    monkeypatch.setattr(threading, "excepthook", unhandled.append)
    local = _local(fleet)
    with caplog.at_level("ERROR", logger=fleet_provider.log.name):
        local.replan_fleet(wait=False)
        _wait_for(lambda: any(r.levelname == "ERROR" for r in caplog.records))
        for thread in threading.enumerate():
            if thread.name == "fleet-work":
                thread.join(_WAIT_S)
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == [f"The engine reload failed: {failure}"]
    assert unhandled == []
    assert local._reload_failed
    assert RELOAD_FAILED_NOTICE in [w.message for w in fleet.provider.health_warnings()]


def test_engine_use_on_the_command_line_shows_a_not_applied_choice_and_exits_one(
    fleet: Services,
) -> None:
    _ProcessSwap.start_failure = OSError("no space left on the device")
    result = CliRunner().invoke(cli_app, ["--json", "engine", "use", "chat", GGUF_CHAT, "mlx"])
    assert result.exit_code == 1
    body = json.loads(result.stdout.strip().splitlines()[-1])
    assert body["command"] == "engine use"
    assert "no space left" in body["error"]
    assert "Restart lilbee" in body["error"]
    assert cfg.engine_choices is not None
