"""A role pinned to the CPU fits its slots against system memory, not against a card."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import model_memory as model_memory_mod
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_GB = 10**9
_BASE = 2 * _GB
_PER_SLOT = _GB


def _footprint(slots: int) -> int:
    return _BASE + _PER_SLOT * slots


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A chat model whose footprint grows by one gigabyte a slot, and a host to size it on."""
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a, engine: 16)
    monkeypatch.setattr(cfg, "num_ctx", None, raising=False)

    def estimate(_path, *, slots: int, **_kwargs) -> GgufVramEstimate:
        total = _footprint(slots)
        return GgufVramEstimate(vram_bytes=total, ram_bytes=0, unified_bytes=total)

    monkeypatch.setattr(model_memory_mod, "estimate_instance_footprint", estimate)
    monkeypatch.setattr(planning_mod, "_system_memory_floor", lambda: 0)
    # The CPU estimate reads the header for the repacked weight copy and the attention shape;
    # the fake file has no header, so both reads are stubbed as in test_cpu_repack_charge.py.
    monkeypatch.setattr(model_memory_mod, "repacked_weight_bytes", lambda _p: 0)
    monkeypatch.setattr(model_memory_mod, "attention_shape", lambda _p: None)
    # A launch reads the binary from its adapter, so the host has the bundled one.
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server",
        lambda: Path("/bin/llama-server"),
    )

    def set_host(*, card_bytes: int, free_system: int) -> FleetDevice:
        card = FleetDevice("CUDA", 0, "gpu", card_bytes, card_bytes)
        monkeypatch.setattr(planning_mod, "plan_sizing_budget", lambda _e, _d=None: card_bytes)
        monkeypatch.setattr(planning_mod, "_plan_free_system_memory", lambda _e: free_system)
        return card

    return set_host


def _cpu_chat_charge() -> int:
    """The system-memory charge the planner gives a CPU-pinned chat role."""
    sized = planning_mod._estimate_cpu_role(
        WorkerRole.CHAT, "ref", engine=LLAMA, skipped_not_installed={}
    )
    assert sized is not None
    assert sized.est_vram_bytes == 0
    return sized.est_ram_bytes


def _launched_slots(card: FleetDevice) -> int:
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(), target=RoleTarget.CPU)
    launch = planning_mod._launch_for(plan, "ref", {card.index: card}, engine=LLAMA)
    return int(launch.argv[launch.argv.index("--parallel") + 1])


class TestCpuRoleSlotFit:
    def test_a_small_card_does_not_cut_the_slots_of_a_cpu_role(self, host) -> None:
        host(card_bytes=3 * _GB, free_system=100 * _GB)
        # The card holds one slot (3 GB); system memory holds all four (6 GB).
        assert _cpu_chat_charge() == _footprint(4)

    def test_a_large_card_does_not_lift_the_slots_of_a_cpu_role(self, host) -> None:
        host(card_bytes=100 * _GB, free_system=4 * _GB)
        # The card would hold four slots; 4 GB of system memory holds two.
        assert _cpu_chat_charge() == _footprint(2)

    def test_the_launch_of_a_cpu_role_reads_system_memory_for_its_slots(self, host) -> None:
        card = host(card_bytes=3 * _GB, free_system=100 * _GB)
        assert _launched_slots(card) == 4

    def test_the_launch_of_a_cpu_role_is_cut_by_small_system_memory(self, host) -> None:
        card = host(card_bytes=100 * _GB, free_system=4 * _GB)
        assert _launched_slots(card) == 2

    def test_an_llm_reranker_on_the_cpu_fans_out_by_system_memory(self, host) -> None:
        from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY

        card = host(card_bytes=3 * _GB, free_system=100 * _GB)
        path = Path("/m/r.gguf")
        mode = RerankMode.LLM
        on_card = planning_mod._slots_for(
            WorkerRole.RERANK, path, 16, engine=LLAMA, rerank_mode=mode, device=card
        )
        on_cpu = planning_mod._slots_for(
            WorkerRole.RERANK,
            path,
            16,
            engine=LLAMA,
            rerank_mode=mode,
            device=card,
            cpu_target=True,
        )
        assert (on_card, on_cpu) == (1, LLM_RERANK_CONCURRENCY)

    def test_a_vision_role_on_the_cpu_fans_out_by_system_memory(
        self, host, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        card = host(card_bytes=3 * _GB, free_system=100 * _GB)
        monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
        path = Path("/m/v.gguf")
        on_card = planning_mod._slots_for(WorkerRole.VISION, path, 16, engine=LLAMA, device=card)
        on_cpu = planning_mod._slots_for(
            WorkerRole.VISION, path, 16, engine=LLAMA, device=card, cpu_target=True
        )
        assert (on_card, on_cpu) == (1, 4)

    def test_a_role_on_a_card_still_reads_the_card(self, host) -> None:
        card = host(card_bytes=3 * _GB, free_system=100 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = planning_mod._launch_for(plan, "ref", {0: card}, engine=LLAMA)
        assert int(launch.argv[launch.argv.index("--parallel") + 1]) == 1

    def test_the_reserve_for_the_os_comes_off_the_system_budget(
        self, host, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        host(card_bytes=100 * _GB, free_system=5 * _GB)
        monkeypatch.setattr(planning_mod, "_system_memory_floor", lambda: _GB)
        assert _cpu_chat_charge() == _footprint(2)


class TestCpuRoleSlotRequest:
    """The request that sizes a slot count says what a CPU role loads."""

    @pytest.fixture
    def requests(self, host, monkeypatch: pytest.MonkeyPatch) -> list[MemoryRequest]:
        host(card_bytes=3 * _GB, free_system=100 * _GB)
        seen: list[MemoryRequest] = []
        real = type(LLAMA).estimate_memory

        def spy(self, request: MemoryRequest):
            seen.append(request)
            return real(self, request)

        # Patch the class: a patch on the instance leaves an attribute behind on undo.
        monkeypatch.setattr(type(LLAMA), "estimate_memory", spy)
        return seen

    def test_a_cpu_chat_role_sizes_its_slots_with_no_layer_on_a_card(
        self, requests: list[MemoryRequest], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "_role_gpu_layers", lambda _r: 99)
        planning_mod._slots_for(
            WorkerRole.CHAT, Path("/m/c.gguf"), 16, engine=LLAMA, cpu_target=True
        )
        assert requests
        assert {r.gpu_layers for r in requests} == {0}

    def test_a_chat_role_on_a_card_sizes_its_slots_with_the_configured_layers(
        self, requests: list[MemoryRequest], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "_role_gpu_layers", lambda _r: 99)
        planning_mod._slots_for(WorkerRole.CHAT, Path("/m/c.gguf"), 16, engine=LLAMA)
        assert requests
        assert {r.gpu_layers for r in requests} == {99}

    def test_a_cpu_chat_role_is_sized_as_one_system_memory_pool(
        self, requests: list[MemoryRequest]
    ) -> None:
        planning_mod._slots_for(
            WorkerRole.CHAT, Path("/m/c.gguf"), 16, engine=LLAMA, cpu_target=True
        )
        assert requests
        assert all(r.on_cpu for r in requests)

    def test_a_cpu_vision_role_is_sized_as_one_system_memory_pool(
        self, requests: list[MemoryRequest], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
        planning_mod._slots_for(
            WorkerRole.VISION, Path("/m/v.gguf"), 16, engine=LLAMA, cpu_target=True
        )
        assert requests
        assert all(r.on_cpu for r in requests)

    def test_a_cpu_llm_reranker_is_sized_as_one_system_memory_pool(
        self, requests: list[MemoryRequest]
    ) -> None:
        planning_mod._slots_for(
            WorkerRole.RERANK,
            Path("/m/r.gguf"),
            16,
            engine=LLAMA,
            rerank_mode=RerankMode.LLM,
            cpu_target=True,
        )
        assert requests
        assert all(r.on_cpu for r in requests)
