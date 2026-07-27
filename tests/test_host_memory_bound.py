"""The host-memory bound has to be wrong in neither direction.

Refusing a load that would have worked is as bad as admitting a plan the host
cannot hold, and the same figure was doing both: refusing on bytes that are
evictable page cache, and checking each role against the whole machine as though
it were the only one.
"""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_REF = "some-org/some-model"


@pytest.fixture(autouse=True)
def _offload_on(monkeypatch):
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "cpu_moe", True, raising=False)
    monkeypatch.setattr(planning, "total_system_memory", lambda: 64 * _GB)
    monkeypatch.setattr(planning, "free_system_memory", lambda: 32 * _GB)


class TestMappedWeightsAreNotResident:
    def test_a_mapped_model_larger_than_ram_is_not_refused(self, monkeypatch) -> None:
        # gguf-parser's host figure counts mmap pages, which llama.cpp maps over
        # rather than allocating. A 200 GB MoE streaming experts from local NVMe
        # is a practiced setup, and stock llama-server loads and generates.
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda _r, _f: False)
        assert not planning._host_memory_refuses(WorkerRole.CHAT, _REF, 200 * _GB, 0)

    def test_an_unmapped_model_larger_than_ram_is_refused(self, monkeypatch) -> None:
        # With --no-mmap the engine buffers the whole thing, so the bytes really
        # do have to fit.
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda _r, _f: True)
        assert planning._host_memory_refuses(WorkerRole.CHAT, _REF, 200 * _GB, 0)


class TestTheBoundIsAgainstThePlanNotOneRole:
    def test_two_roles_that_each_fit_but_do_not_together_are_caught(self, monkeypatch) -> None:
        monkeypatch.setattr(planning, "_host_bytes_must_be_resident", lambda _r, _f: True)
        # 40 GB alone fits a 64 GB host; a second 40 GB does not.
        assert not planning._host_memory_refuses(WorkerRole.CHAT, _REF, 40 * _GB, 0)
        assert planning._host_memory_refuses(WorkerRole.VISION, _REF, 40 * _GB, 40 * _GB)

    def test_what_is_already_committed_is_carried_into_the_next_role(self) -> None:
        from lilbee.providers.fleet.placement import ModelPlacementInput

        admitted = {
            WorkerRole.CHAT: ModelPlacementInput(WorkerRole.CHAT, 1, est_ram_bytes=30 * _GB),
            WorkerRole.EMBED: ModelPlacementInput(WorkerRole.EMBED, 1, est_ram_bytes=5 * _GB),
        }
        assert planning._host_committed(admitted) == 35 * _GB


def test_the_estimator_fallback_is_bound_too(monkeypatch) -> None:
    # A model the estimator could not size charged its host half to VRAM and
    # skipped the host bound entirely.
    seen: list[int] = []
    monkeypatch.setattr(planning, "_role_weights_bytes", lambda *_a: 1)
    monkeypatch.setattr(planning, "_weights_exceed_hardware", lambda *_a, **_k: False)
    monkeypatch.setattr(
        planning, "_host_memory_refuses", lambda _r, _f, ram, committed: bool(seen.append(ram))
    )
    monkeypatch.setattr(planning, "_fallback_floor_for", lambda *_a: 8 * _GB)
    planning._sizing_failure_fallback(
        WorkerRole.CHAT, _REF, RuntimeError("boom"), device_count=1, total_vram=24 * _GB
    )
    assert seen, "the fallback never consulted the host bound"
