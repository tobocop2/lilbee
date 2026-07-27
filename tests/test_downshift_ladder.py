"""The ladder's own contract: False means the retry would ask for the same thing.

A step that changes nothing is worse than no step, because the caller tears the
group down and respawns an identical launch on the strength of it.
"""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.roles import WorkerRole


@pytest.fixture(autouse=True)
def _auto_ctx(monkeypatch):
    monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
    planning.clear_ctx_downshift()
    yield
    planning.clear_ctx_downshift()


def _walk(base: int, role: WorkerRole = WorkerRole.CHAT) -> list[int]:
    """Every context the ladder actually serves for *base*, first to last."""
    seen = [planning.apply_ctx_downshift(role, base)]
    while planning.record_ctx_downshift(role):
        seen.append(planning.apply_ctx_downshift(role, base))
        assert len(seen) < 30, "the ladder has to terminate"
    return seen


class TestAGrantedStepAlwaysChangesTheContext:
    def test_a_base_just_above_the_floor_grants_one_step_only(self) -> None:
        # 8192 halves once to the floor; a second step would serve 4096 again.
        assert _walk(8192) == [8192, 4096]

    def test_a_base_at_the_floor_grants_nothing(self) -> None:
        assert _walk(planning.MIN_DOWNSHIFT_CTX) == [planning.MIN_DOWNSHIFT_CTX]

    def test_a_base_below_the_floor_grants_nothing(self) -> None:
        assert _walk(512, WorkerRole.EMBED) == [512]


class TestTheLadderReachesTheFloorFromAnyBase:
    def test_a_long_window_walks_all_the_way_down(self) -> None:
        assert _walk(131072) == [131072, 65536, 32768, 16384, 8192, 4096]

    def test_the_last_rung_is_always_the_floor(self) -> None:
        for base in (16384, 32768, 65536, 131072):
            planning.clear_ctx_downshift()
            assert _walk(base)[-1] == planning.MIN_DOWNSHIFT_CTX


class TestASuccessfulLoadForgetsTheLadder:
    def test_a_role_that_loads_starts_from_full_size_again(self) -> None:
        # Otherwise a user who switches from a model that exhausted the ladder to
        # one that fits gets a quartered estimate and an immediate refusal.
        planning.record_ctx_downshift(WorkerRole.CHAT)
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) < 32768
        planning.clear_ctx_downshift(WorkerRole.CHAT)
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) == 32768

    def test_clearing_one_role_leaves_the_others(self) -> None:
        planning.record_ctx_downshift(WorkerRole.CHAT)
        planning.record_ctx_downshift(WorkerRole.EMBED)
        planning.clear_ctx_downshift(WorkerRole.CHAT)
        assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) == 32768
        assert planning.apply_ctx_downshift(WorkerRole.EMBED, 32768) < 32768


def test_a_ready_engine_clears_that_role(monkeypatch, tmp_path) -> None:
    # The clear has to be wired to something real, not only available.
    from lilbee.providers.fleet import swap_manager as sm

    planning.record_ctx_downshift(WorkerRole.CHAT)
    assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) < 32768

    mgr = sm.SwapManager.__new__(sm.SwapManager)
    mgr._estimate_checked = set()
    mgr._log_path = tmp_path / "swap.log"
    launch = type("L", (), {"role": WorkerRole.CHAT, "est_vram_bytes": 0})()
    mgr._launch_by_model = {"chat-0": launch}
    monkeypatch.setattr(sm, "report_missing_log", lambda *_a: False)
    mgr._check_estimates({"chat-0"})

    assert planning.apply_ctx_downshift(WorkerRole.CHAT, 32768) == 32768
