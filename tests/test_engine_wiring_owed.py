"""Tests that fail when a caller reaches for the llama adapter, not the one it was given."""

from __future__ import annotations

import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import MemoryEstimate, MemoryKind, MemoryRequest
from tests.conformance.fake_adapter import FAKE, FakeAdapter

_REQUEST = MemoryRequest(
    model_path=Path("model.bin"),
    ctx=4096,
    slots=1,
    gpu_layers=-1,
    flash_attn=True,
    kv_cache_type=KvCacheType.F16,
)


class _SizedEngine(FakeAdapter):
    """A sized engine whose estimate fails the test when anything asks for it."""

    memory_kind = MemoryKind.SIZED

    def __init__(self) -> None:
        self.estimate_calls = 0

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        self.estimate_calls += 1
        raise AssertionError("a sized engine owes no estimate to a launch")


class _CountingFixedEngine(FakeAdapter):
    def __init__(self) -> None:
        self.estimate_calls = 0

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        self.estimate_calls += 1
        return super().estimate_memory(request)


def test_sized_engine_runs_no_memory_estimate() -> None:
    engine = _SizedEngine()
    assert planning._declared_reserve(engine, _REQUEST, unified=False) == 0
    assert planning._declared_reserve(engine, _REQUEST, unified=True) == 0
    assert engine.estimate_calls == 0


def test_fixed_reserve_engine_charges_its_declared_bytes_once_per_ask() -> None:
    engine = _CountingFixedEngine()
    expected = FAKE.estimate_memory(_REQUEST)
    assert planning._declared_reserve(engine, _REQUEST, unified=True) == expected.shared_bytes
    assert planning._declared_reserve(engine, _REQUEST, unified=False) == expected.device_bytes
    assert engine.estimate_calls == 2
    assert expected.shared_bytes > 0


class _DeviceEngine(FakeAdapter):
    """Lists one device and records the binary it was asked about."""

    def __init__(self) -> None:
        self.asked: list[Path] = []

    def discover_devices(self, binary: Path) -> DeviceProbe:
        self.asked.append(binary)
        device = FleetDevice(
            backend="FAKE", index=7, name="fake card", total_bytes=1 << 30, free_bytes=1 << 29
        )
        return DeviceProbe(devices=[device], output="listed", spoke_protocol=False)


def test_device_group_asks_the_engine_it_was_given(tmp_path: Path) -> None:
    engine = _DeviceEngine()
    binary = tmp_path / "engine-bin"
    devices = planning.resolve_devices(binary, engine)
    assert engine.asked == [binary]
    assert [(d.backend, d.index) for d in devices] == [("FAKE", 7)]


def test_llama_adapter_survives_a_pickle_round_trip() -> None:
    restored = pickle.loads(pickle.dumps(LLAMA))  # noqa: S301  # round trip of our own adapter
    assert type(restored) is type(LLAMA)
    assert restored.name == LLAMA.name
    assert restored.formats == LLAMA.formats


def _report_engine_name(engine: Any, queue: Any) -> None:
    queue.put((engine.name, sorted(engine.formats)))


def test_adapter_reaches_a_spawned_child_intact() -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_report_engine_name, args=(LLAMA, queue))
    child.start()
    try:
        name, formats = queue.get(timeout=60)
    finally:
        child.join(timeout=60)
    assert (name, formats) == (LLAMA.name, sorted(LLAMA.formats))
    assert child.exitcode == 0


def test_download_child_is_started_with_the_callers_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.catalog import download_process

    seen: dict[str, Any] = {}

    class _Context:
        def Pipe(self, duplex: bool) -> tuple[Any, Any]:
            return _Closable(), _Closable()

        def Process(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return _Closable()

    class _Closable:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(download_process.multiprocessing, "get_context", lambda _m: _Context())
    entry = type("Entry", (), {"hf_repo": "org/repo"})()
    download_process._start_worker(entry, tmp_path, None, FAKE)  # type: ignore[arg-type]
    assert seen["args"][-1] is FAKE
    assert pickle.loads(pickle.dumps(seen["args"][-1])) is not None  # noqa: S301  # our own adapter
