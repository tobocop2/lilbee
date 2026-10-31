"""Real fleet planning and real clients against a fake engine server; one role has no engine.

Only the llama-swap process is replaced. The engine book, the plan, the launches and the
clients are the product's own, so a role with no engine is proved on the path a request takes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from lilbee.app import services as services_mod
from lilbee.app.services import Services, build_services
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet import provider as fleet_provider
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind
from lilbee.providers.roles import EngineBackend, WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from tests.conformance.fake_adapter import BundledFakeAdapter
from tests.conformance.fake_engine import FakeEngine

MLX_REF = "org/repo#mlx"
GGUF_CHAT = "org/repo/chat.gguf"
GGUF_EMBED = "org/repo/embed.gguf"
_GIB = 1 << 30
_MESSAGES = [{"role": "user", "content": "hi"}]


class _Gguf(BundledFakeAdapter):
    """A gguf engine that ships in the package, plans one GPU and sizes every model at one GiB."""

    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"fakearch"})
    served_roles = frozenset(WorkerRole)

    def __hash__(self) -> int:
        return hash(self.name)

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(
            devices=[FleetDevice("CUDA", 0, "gpu0", 24 * _GIB, 24 * _GIB)],
            output="",
            spoke_protocol=False,
            backend=EngineBackend.CUDA,
        )

    def estimate_memory(self, request: Any) -> MemoryEstimate:
        return MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=_GIB,
            host_bytes=0,
            shared_bytes=_GIB,
            per_device=(_GIB,),
            per_device_shared=(_GIB,),
        )

    def model_meta(self, path: Path) -> dict[str, str]:
        return {"architecture": "fakearch", "context_length": "4096"}

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def runtime_env(self) -> dict[str, str]:
        return {}

    def swap_extras(self, launch: Any) -> dict[str, str]:
        return {}


class _StubSwap:
    """Stands in for the llama-swap process only; planning and clients are the real ones."""

    started: ClassVar[list[list[tuple[WorkerRole, str]]]] = []
    url = ""

    def __init__(self, data_dir: Path, group: Any) -> None:
        self.log_path = data_dir / "swap.log"

    def start(self, launches: list[Any], **kwargs: Any) -> None:
        type(self).started.append([(launch.role, launch.model_id) for launch in launches])

    def endpoint(self) -> str:
        return type(self).url

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name == "health_warnings":
            return lambda *a, **k: []
        if name in {"role_ready", "is_live", "running", "bound"}:
            return lambda *a, **k: True
        return lambda *a, **k: None


@pytest.fixture(params=[WorkerRole.CHAT, WorkerRole.EMBED], ids=lambda role: role.value)
def real(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[WorkerRole, Services, FakeEngine]]:
    """A container whose *broken* role names a format no installed engine loads."""
    broken: WorkerRole = request.param
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX, raising=False)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Gguf())
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: 64 * _GIB)
        monkeypatch.setattr(module, "free_system_memory", lambda: 64 * _GIB)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "placement", None)
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "data" / "lancedb")
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "chat_model", MLX_REF if broken is WorkerRole.CHAT else GGUF_CHAT)
    monkeypatch.setattr(
        cfg, "embedding_model", MLX_REF if broken is WorkerRole.EMBED else GGUF_EMBED
    )
    monkeypatch.setattr(fleet_provider, "SwapManager", _StubSwap)
    _StubSwap.started = []
    with FakeEngine() as engine:
        _StubSwap.url = engine.url
        services = build_services(cfg)
        monkeypatch.setattr(services_mod._state, "singleton", services)
        yield broken, services, engine
        services.store.close()


def _launched_roles() -> set[WorkerRole]:
    return {role for batch in _StubSwap.started for role, _model in batch}


def test_the_other_role_runs_on_the_real_fleet(
    real: tuple[WorkerRole, Services, FakeEngine],
) -> None:
    broken, services, engine = real
    provider = services.provider
    if broken is WorkerRole.CHAT:
        vectors = provider.embed(["hi"])
        assert len(vectors) == 1
        assert len(vectors[0]) > 0
        with pytest.raises(NoEngineError, match="For chat: No installed engine loads the mlx"):
            provider.chat(_MESSAGES)
        served = WorkerRole.EMBED
    else:
        assert provider.chat(_MESSAGES)
        with pytest.raises(NoEngineError, match="For embed: No installed engine loads the mlx"):
            provider.embed(["hi"])
        served = WorkerRole.CHAT
    assert _launched_roles() == {served}
    assert engine.seen


def test_two_threads_resolve_different_roles_at_once(
    real: tuple[WorkerRole, Services, FakeEngine],
) -> None:
    broken, _services, _engine = real

    def embed(provider: RoutingProvider) -> Any:
        return provider.embed(["hi"])

    def chat(provider: RoutingProvider) -> Any:
        return provider.chat(_MESSAGES)

    good, bad = (embed, chat) if broken is WorkerRole.CHAT else (chat, embed)
    rounds = 0
    for _ in range(10):
        results = _race(RoutingProvider(), good, bad)
        assert not isinstance(results["good"], Exception), results["good"]
        assert isinstance(results["bad"], NoEngineError), results["bad"]
        rounds += 1
    assert rounds == 10


def _race(
    provider: RoutingProvider,
    good: Callable[[RoutingProvider], Any],
    bad: Callable[[RoutingProvider], Any],
) -> dict[str, Any]:
    """Call *good* and *bad* on one fresh provider from two threads released together."""
    barrier = threading.Barrier(2)
    results: dict[str, Any] = {}

    def run(name: str, call: Callable[[RoutingProvider], Any]) -> None:
        barrier.wait()
        try:
            results[name] = call(provider)
        except Exception as exc:
            results[name] = exc

    threads = [
        threading.Thread(target=run, args=("good", good)),
        threading.Thread(target=run, args=("bad", bad)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    provider.shutdown()
    return results
