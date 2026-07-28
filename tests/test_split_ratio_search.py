"""How many tensor-split ratios the planner is willing to try.

One whole-GiB proportional ratio per card count means a split that would fit at
some other proportion is never found, and the model lands tight on one card as
if no split existed.
"""

from __future__ import annotations

from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


class TestTheRatioLadder:
    def test_the_proportional_ratio_is_tried_first(self) -> None:
        from lilbee.providers.fleet.placement import _split_ratio_candidates

        remaining = {0: 24.0 * _GB, 1: 12.0 * _GB}
        assert _split_ratio_candidates([0, 1], remaining)[0] == (24, 12)

    def test_candidates_are_distinct_and_match_the_declared_rung_count(self) -> None:
        # _MAX_RATIO_CANDIDATES is what the memo is sized against, so it has to
        # equal what the ladder actually produces when nothing deduplicates.
        from lilbee.providers.fleet.placement import _MAX_RATIO_CANDIDATES, _split_ratio_candidates

        remaining = {0: 24.0 * _GB, 1: 12.4 * _GB, 2: 7.9 * _GB}
        candidates = _split_ratio_candidates([0, 1, 2], remaining)
        assert len(candidates) == len(set(candidates))
        assert len(candidates) == _MAX_RATIO_CANDIDATES
        assert all(len(c) == 3 and all(part >= 1 for part in c) for c in candidates)

    def test_a_lone_card_has_nothing_to_shift_against(self) -> None:
        # A split needs two cards; the ladder must not invent a second share.
        from lilbee.providers.fleet.placement import _shifted_toward_roomiest

        assert _shifted_toward_roomiest([0], {0: 24.0 * _GB}, (24,)) == (24,)

    def test_equal_cards_need_only_the_even_ratio(self) -> None:
        # Nothing to shift when every card has the same room, so the ladder must
        # not spend estimator calls re-asking the same question.
        from lilbee.providers.fleet.placement import _split_ratio_candidates

        remaining = {0: 24.0 * _GB, 1: 24.0 * _GB}
        assert _split_ratio_candidates([0, 1], remaining) == ((24, 24),)


class TestASplitFoundOnlyAtASecondRatio:
    def test_a_model_that_fits_only_when_shifted_is_still_split(self) -> None:
        # The proportional shard overflows the smaller card while the pair has
        # room overall. Shifting load to the roomier card fits; without a second
        # ratio the model would land tight on one card instead.
        model = ModelPlacementInput(WorkerRole.EMBED, 30 * _GB)
        seen: list[tuple[int, ...]] = []

        def _peak(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
            seen.append(ratio)
            total = sum(ratio)
            share = [30 * _GB * part / total for part in ratio]
            return tuple(int(s) for s in share)

        plan = plan_placement([model], [(0, 26 * _GB), (1, 14 * _GB)], estimate_peak=_peak)
        instance = plan.instances[0]
        assert instance.devices == (0, 1), f"tried {seen}"
        assert plan.tight_roles == {}

    def test_the_sweep_stops_at_its_call_budget(self, monkeypatch) -> None:
        from lilbee.providers.fleet import placement as placement_mod

        monkeypatch.setattr(placement_mod, "_MAX_SPLIT_ESTIMATES", 3)
        calls = {"n": 0}

        def _peak(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
            calls["n"] += 1
            return tuple(999 * _GB for _ in ratio)  # never fits, so the sweep runs on

        model = ModelPlacementInput(WorkerRole.EMBED, 30 * _GB)
        devices = [(idx, (20 + idx) * _GB) for idx in range(6)]
        plan_placement([model], devices, estimate_peak=_peak)
        assert calls["n"] <= 3


def test_the_estimator_memo_outlasts_one_plan() -> None:
    # The sweep asks the estimator once per (card count, proportion) and the
    # context bisection asks again per candidate. A memo smaller than one plan's
    # working set evicts keys the next phase re-requests, turning a cache hit
    # into another subprocess.
    from lilbee.providers.fleet.placement import (
        _CTX_FIT_ESTIMATE_COST,
        _MAX_RATIO_CANDIDATES,
        _MAX_SPLIT_ESTIMATES,
    )
    from lilbee.providers.fleet.vram import _CACHE_SIZE
    from lilbee.providers.roles import ROLE_REGISTRY

    # The earlier arithmetic counted one key per candidate and missed the context
    # bisection entirely, which is what made the memo look big enough when it was
    # not. Each fitting candidate bisects, and every probe keys separately
    # because the key carries ctx.
    per_plan = _MAX_SPLIT_ESTIMATES * (1 + _CTX_FIT_ESTIMATE_COST) + _MAX_RATIO_CANDIDATES * len(
        ROLE_REGISTRY
    )
    assert per_plan <= _CACHE_SIZE
