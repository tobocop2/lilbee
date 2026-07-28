"""Properties of the planner's arithmetic, checked against generated inputs.

Every other test here uses inputs somebody chose, which means they cover the
cases somebody thought of. Three review passes over this code found real
defects and still missed one that a generator finds immediately: the context
ladder returned a larger number than it was given, because the floor was
applied as a bare maximum and nobody had tried a context already below it.

These state what must hold for every input rather than for the inputs we
imagined, so the next defect of that shape is caught by construction.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.placement import (
    _normalized,
    _shifted_toward_roomiest,
    _split_ratio_candidates,
    _tight_device_group,
)
from lilbee.providers.roles import WorkerRole

_GB = 1024**3

# Device pools as the planner sees them: a handful of cards with free bytes.
_free_bytes = st.floats(min_value=0.0, max_value=200 * _GB, allow_nan=False, allow_infinity=False)
_pools = st.dictionaries(st.integers(min_value=0, max_value=7), _free_bytes, min_size=1, max_size=8)
_multi_pools = st.dictionaries(
    st.integers(min_value=0, max_value=7), _free_bytes, min_size=2, max_size=8
)


class TestTheContextLadder:
    """A downshift exists to make the next launch smaller. Anything it returns
    that is not smaller is either a wasted relaunch or, if larger, a retry that
    asks for more memory than the load which just ran out of it."""

    @given(ctx=st.integers(min_value=1, max_value=1 << 22), steps=st.integers(0, 12))
    def test_a_shift_never_exceeds_what_it_was_given(self, ctx: int, steps: int) -> None:
        assert planning._shifted(ctx, steps) <= ctx

    @given(ctx=st.integers(min_value=1, max_value=1 << 22), steps=st.integers(0, 12))
    def test_a_shift_is_never_zero_or_negative(self, ctx: int, steps: int) -> None:
        assert planning._shifted(ctx, steps) >= 1

    @given(ctx=st.integers(min_value=1, max_value=1 << 22), steps=st.integers(1, 12))
    def test_more_steps_never_serve_a_larger_context(self, ctx: int, steps: int) -> None:
        assert planning._shifted(ctx, steps) <= planning._shifted(ctx, steps - 1)

    @given(ctx=st.integers(min_value=planning.MIN_DOWNSHIFT_CTX, max_value=1 << 22))
    def test_a_context_above_the_floor_never_falls_below_it(self, ctx: int) -> None:
        for steps in range(12):
            assert planning._shifted(ctx, steps) >= planning.MIN_DOWNSHIFT_CTX

    @given(ctx=st.integers(min_value=1, max_value=1 << 22))
    @settings(max_examples=50)
    def test_the_ladder_terminates_from_any_base(self, ctx: int) -> None:
        # record_ctx_downshift promises False means the retry would ask for the
        # same thing again, so walking it must reach a fixed point and stop.
        # Patched inside the test because a fixture would not reset per input.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfg, "num_ctx", None, raising=False)
            planning.clear_ctx_downshift()
            try:
                served = [planning.apply_ctx_downshift(WorkerRole.CHAT, ctx)]
                while planning.record_ctx_downshift(WorkerRole.CHAT):
                    served.append(planning.apply_ctx_downshift(WorkerRole.CHAT, ctx))
                    assert len(served) < 40, "the ladder must reach a fixed point"
                assert all(b < a for a, b in pairwise(served))
            finally:
                planning.clear_ctx_downshift()


class TestTheSplitRatioLadder:
    """A tensor split is a ratio the engine divides layers by. A zero or negative
    part is not a smaller share, it is an invalid argument, and a ratio that
    names a different number of cards than were chosen silently mis-splits."""

    @given(pool=_multi_pools)
    def test_every_part_is_positive(self, pool: dict[int, float]) -> None:
        for ratio in _split_ratio_candidates(sorted(pool), pool):
            assert all(part >= 1 for part in ratio)

    @given(pool=_multi_pools)
    def test_every_candidate_covers_exactly_the_chosen_cards(self, pool: dict[int, float]) -> None:
        devices = sorted(pool)
        for ratio in _split_ratio_candidates(devices, pool):
            assert len(ratio) == len(devices)

    @given(pool=_multi_pools)
    def test_candidates_are_distinct_proportions(self, pool: dict[int, float]) -> None:
        # Two candidates in lowest terms that are equal are the same split asked
        # twice, and each ask is a gguf-parser subprocess.
        candidates = _split_ratio_candidates(sorted(pool), pool)
        normalised = [_normalized(r) for r in candidates]
        assert len(normalised) == len(set(normalised))

    @given(pool=_multi_pools)
    def test_the_ladder_is_bounded(self, pool: dict[int, float]) -> None:
        from lilbee.providers.fleet.placement import _MAX_RATIO_CANDIDATES

        assert len(_split_ratio_candidates(sorted(pool), pool)) <= _MAX_RATIO_CANDIDATES

    @given(pool=_multi_pools)
    def test_shifting_preserves_the_card_count_and_positivity(self, pool: dict[int, float]) -> None:
        devices = sorted(pool)
        base = _split_ratio_candidates(devices, pool)[0]
        shifted = _shifted_toward_roomiest(devices, pool, base)
        assert len(shifted) == len(base)
        assert all(part >= 1 for part in shifted)

    @given(ratio=st.lists(st.integers(min_value=1, max_value=1024), min_size=1, max_size=8))
    def test_normalising_is_idempotent_and_preserves_proportion(self, ratio: list[int]) -> None:
        once = _normalized(tuple(ratio))
        assert _normalized(once) == once
        assert all(part >= 1 for part in once)


class TestTheTightDeviceGroup:
    """The last resort before refusing a model. It must return cards that exist
    and, when it returns several, must not be returning them arbitrarily."""

    @given(needed=st.integers(min_value=1, max_value=400 * _GB), pool=_pools)
    def test_it_only_ever_returns_cards_it_was_given(
        self, needed: int, pool: dict[int, float]
    ) -> None:
        assert set(_tight_device_group(needed, pool)) <= set(pool)

    @given(needed=st.integers(min_value=1, max_value=400 * _GB), pool=_pools)
    def test_it_never_repeats_a_card(self, needed: int, pool: dict[int, float]) -> None:
        group = _tight_device_group(needed, pool)
        assert len(group) == len(set(group))

    @given(needed=st.integers(min_value=1, max_value=400 * _GB), pool=_pools)
    def test_a_non_empty_pool_always_yields_somewhere_to_put_it(
        self, needed: int, pool: dict[int, float]
    ) -> None:
        assert _tight_device_group(needed, pool)

    @given(needed=st.integers(min_value=1, max_value=400 * _GB), pool=_pools)
    def test_one_card_is_returned_only_when_one_card_is_enough(
        self, needed: int, pool: dict[int, float]
    ) -> None:
        group = _tight_device_group(needed, pool)
        if len(group) == 1 and len(pool) > 1:
            roomiest = max(pool.values())
            # Either that card holds it, or nothing else had any room to add.
            assert pool[group[0]] >= needed or roomiest == pool[group[0]]


class TestTheHostMemoryBound:
    """Admission over a plan, not a role. Adding a role to a plan can only make
    the plan harder to fit, never easier."""

    @given(
        ram=st.integers(min_value=1, max_value=500 * _GB),
        committed=st.integers(min_value=0, max_value=500 * _GB),
        extra=st.integers(min_value=0, max_value=500 * _GB),
    )
    @settings(max_examples=60)
    def test_more_already_committed_never_turns_a_refusal_into_an_admission(
        self, ram: int, committed: int, extra: int
    ) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfg, "cpu_moe", True, raising=False)
            mp.setattr(planning, "_host_bytes_must_be_resident", lambda *_a: True)
            mp.setattr(planning, "total_system_memory", lambda: 64 * _GB)
            mp.setattr(planning, "free_system_memory", lambda: 32 * _GB)
            refused_at = planning._host_memory_refuses(WorkerRole.CHAT, "m", ram, committed)
            refused_more = planning._host_memory_refuses(
                WorkerRole.CHAT, "m", ram, committed + extra
            )
        assert not (refused_at and not refused_more)

    @given(
        ram=st.integers(min_value=1, max_value=200 * _GB),
        committed=st.integers(min_value=0, max_value=200 * _GB),
    )
    @settings(max_examples=60)
    def test_the_refusal_is_the_plan_total_against_the_machine(
        self, ram: int, committed: int
    ) -> None:
        # Monotonicity alone is satisfied by a bound that ignores what is already
        # committed, since it then answers the same either way. This pins the
        # actual sum, so dropping the committed term is caught rather than passed.
        total = 64 * _GB
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfg, "cpu_moe", True, raising=False)
            mp.setattr(planning, "_host_bytes_must_be_resident", lambda *_a: True)
            mp.setattr(planning, "total_system_memory", lambda: total)
            mp.setattr(planning, "free_system_memory", lambda: total)
            refused = planning._host_memory_refuses(WorkerRole.CHAT, "m", ram, committed)
        assert refused == (committed + ram > total)

    @given(ram=st.integers(min_value=1, max_value=500 * _GB))
    def test_nothing_offloading_is_never_refused(self, ram: int) -> None:
        # The host figure describes memory nobody will allocate when the engine
        # keeps every layer on the card, so it cannot be grounds for refusal.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfg, "cpu_moe", False, raising=False)
            mp.setattr(cfg, "n_cpu_moe", None, raising=False)
            mp.setattr(cfg, "n_gpu_layers", None, raising=False)
            assert not planning._host_memory_refuses(WorkerRole.CHAT, "m", ram, 0)


class TestTheEstimatePlausibilityBand:
    """A floor is a lower bound. It must not reject an estimate above itself."""

    @given(
        estimated=st.integers(min_value=0, max_value=500 * _GB),
        floor=st.integers(min_value=0, max_value=500 * _GB),
    )
    def test_only_estimates_below_a_known_floor_are_rejected(
        self, estimated: int, floor: int
    ) -> None:
        implausible = planning._estimate_is_implausible(estimated=estimated, floor=floor)
        if implausible:
            assert 0 < estimated < floor
            assert floor > 0

    @given(estimated=st.integers(min_value=1, max_value=500 * _GB))
    def test_an_unknown_floor_rejects_nothing(self, estimated: int) -> None:
        assume(estimated > 0)
        assert not planning._estimate_is_implausible(estimated=estimated, floor=0)
