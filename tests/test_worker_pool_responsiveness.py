"""End-to-end responsiveness check for the persistent worker pool.

The reason the pool exists at all is to keep the asyncio loop responsive
under load. This test bolts a stubbed embed worker (no real Llama load,
so it runs in CI) onto the real pool and asserts that asyncio sleeps
running concurrently with a long sequence of embed calls do not stall
beyond a tight threshold. If the pool ever regresses to running embed
inference inline on the asyncio loop, this assertion fires.

Latency budget: 200ms p95 for an asyncio.sleep(0.05) tick during pool
embed activity. The threshold is generous so transient CI noise does
not flake the test; tighten only if the wall-clock perf shows headroom.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import pytest

from lilbee.core.config import cfg
from lilbee.providers.worker.transport import RoleConfig

pytestmark = [pytest.mark.xdist_group("worker_pool_responsiveness")]


# 200ms p95 budget for one asyncio.sleep(0.05) tick under concurrent
# pool embed activity. Rationale: a real user types in the TUI at sub-
# second cadence; if embed ever runs back on the asyncio loop, ticks
# spike into multi-second territory (we measured >2s prior to the
# pool). 200ms gives generous headroom over the ~50ms tick target so
# loaded CI runners do not flake while still catching any inline
# regression by 10x. The test is xdist-grouped so concurrent files do
# not eat the budget.
_LATENCY_P95_BUDGET_S = 0.200
_TICK_INTERVAL_S = 0.05
_TICK_COUNT = 30
_EMBED_BATCH_COUNT = 12


def _slow_stub_load(_self) -> Any:
    """Stub Llama load that simulates real embed-call wall time."""

    class _SlowLlama:
        n_batch = 8192

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0] * max(1, len(text))

        def create_embedding(self, *, input: list[str]) -> dict[str, Any]:
            # Simulate ~30 ms of CPU-bound inference per call. Real
            # llama-cpp embed runs much longer; this is just enough to
            # demonstrate the routing keeps the asyncio loop unblocked.
            time.sleep(0.03)
            return {"data": [{"embedding": [float(len(t))]} for t in input]}

    return _SlowLlama()


def _patched_embed_worker_main(
    data_conn: Any, health_conn: Any, abort_flag: Any, role_config: RoleConfig
) -> None:
    """Real embed worker entrypoint with the load step swapped for the slow stub."""
    from lilbee.providers.worker import embed_worker

    embed_worker._EmbedSession._load = _slow_stub_load  # type: ignore[method-assign]
    embed_worker.embed_worker_main(data_conn, health_conn, abort_flag, role_config)


@pytest.fixture()
def pool_provider(monkeypatch, tmp_path):
    cfg.worker_pool_call_timeout_s = 30.0
    cfg.embedding_model = "stub/model"
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    fake_path = tmp_path / "models" / "stub.gguf"
    fake_path.write_bytes(b"")
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        lambda _name: fake_path,
    )
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.provider.embed_worker_main",
        _patched_embed_worker_main,
    )

    from lilbee.core.services import set_services
    from lilbee.providers.llama_cpp.provider import LlamaCppProvider
    from tests.conftest import make_mock_services

    provider = LlamaCppProvider()
    set_services(make_mock_services(provider=provider))
    try:
        yield provider
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_asyncio_loop_stays_responsive_during_pool_embed(pool_provider) -> None:
    """The asyncio loop continues to schedule sleeps during pool-driven embed calls.

    Drives pool embed calls from a worker thread (mirroring how
    LlamaCppProvider is called from the in-process embed thread in
    production) while the asyncio loop runs a steady sleep tick. If the
    pool routing ever regresses to running embed inline on the asyncio
    thread, the per-tick latency p95 exceeds the budget.
    """
    loop = asyncio.get_running_loop()
    latencies: list[float] = []
    embed_done = asyncio.Event()

    def _drive_embeds() -> None:
        try:
            for i in range(_EMBED_BATCH_COUNT):
                pool_provider.embed([f"text-{i}-{i}"])
        finally:
            loop.call_soon_threadsafe(embed_done.set)

    async def _tick() -> None:
        for _ in range(_TICK_COUNT):
            t0 = time.monotonic()
            await asyncio.sleep(_TICK_INTERVAL_S)
            latencies.append(time.monotonic() - t0)
            if embed_done.is_set():
                return

    embed_task = loop.run_in_executor(None, _drive_embeds)
    tick_task = asyncio.create_task(_tick())

    await asyncio.gather(embed_task, tick_task)

    # All embed calls completed (no exceptions raised).
    from lilbee.core.services import get_services

    assert "embed" in get_services().worker_pool.registered_roles

    # Latency budget: p95 under threshold. Sort + index avoids importing
    # a percentile helper just for this assertion.
    if not latencies:
        pytest.fail("Tick task collected zero latency samples")
    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
    p95 = sorted_latencies[p95_index]
    median = statistics.median(latencies)
    budget_ms = _LATENCY_P95_BUDGET_S * 1000
    assert p95 < _LATENCY_P95_BUDGET_S, (
        f"asyncio tick p95 latency {p95 * 1000:.1f}ms exceeded the {budget_ms:.0f}ms budget "
        f"while pool was active (median={median * 1000:.1f}ms, samples={len(latencies)})"
    )
