"""Tests for the multi-GPU placement planner."""

from __future__ import annotations

from collections.abc import Sequence

from lilbee.providers.fleet.placement import (
    InstancePlan,
    ModelPlacementInput,
    PeakEstimator,
    Placement,
    plan_placement,
)
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_FOUR_24GB_CARDS = [(0, 24 * _GB), (1, 24 * _GB), (2, 24 * _GB), (3, 24 * _GB)]
_FOUR_24GB_FREE = {0: 24 * _GB, 1: 24 * _GB, 2: 24 * _GB, 3: 24 * _GB}


def _fits_any_count(model: ModelPlacementInput) -> PeakEstimator:
    """Estimator whose even per-device share fits a 24 GB card at every split count."""

    def estimate_peak(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(model.est_vram_bytes // len(ratio) for _ in ratio)

    return estimate_peak


class _CtxByCount:
    """Fake chat-context fitter returning a fixed served ctx per shard card count,
    recording the per-device free headroom it was asked about."""

    def __init__(self, served_by_count: dict[int, int]) -> None:
        self._served_by_count = served_by_count
        self.calls: list[list[int]] = []

    def __call__(self, ratio: tuple[int, ...], per_device_free_bytes: Sequence[int]) -> int:
        self.calls.append(list(per_device_free_bytes))
        return self._served_by_count[len(ratio)]


def _never(_role: WorkerRole, _ratio: tuple[int, ...]) -> tuple[int, ...]:
    """Estimator that fails if invoked: a placement that fits single cards never splits."""
    raise AssertionError(
        "estimate_peak called unexpectedly; this placement should not tensor-split"
    )


def _even(*models: ModelPlacementInput) -> PeakEstimator:
    """Fake estimator: each model's footprint split evenly across the cards, no overhead."""
    by_role = {m.role: m.est_vram_bytes for m in models}

    def estimate_peak(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(by_role[role] // len(ratio) for _ in ratio)

    return estimate_peak


class TestPlanPlacement:
    def test_single_model_fits_one_gpu(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            [(0, 24 * _GB)],
            estimate_peak=_never,
        )
        assert plan == Placement(
            instances=(InstancePlan(WorkerRole.CHAT, (0,)),),
            unplaceable_roles=(),
        )

    def test_search_role_placed_before_chat_on_constrained_host(self) -> None:
        """When chat and an essential search role can't both fit, the search role
        wins its placement instead of being starved by the larger chat model
        placed first (bb-7jg1.6)."""
        chat = ModelPlacementInput(WorkerRole.CHAT, 20 * _GB)
        embed = ModelPlacementInput(WorkerRole.EMBED, 18 * _GB)
        # One 24 GB card (21.6 GB usable): either model fits alone, not both.
        plan = plan_placement([chat, embed], [(0, 24 * _GB)], estimate_peak=_even(chat, embed))
        placed = {i.role for i in plan.instances}
        assert WorkerRole.EMBED in placed
        assert WorkerRole.CHAT in plan.unplaceable_roles

    def test_colocates_small_models_on_one_gpu(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
                ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
            ],
            [(0, 24 * _GB)],
            estimate_peak=_never,
        )
        assert plan.unplaceable_roles == ()
        assert {i.role for i in plan.instances} == {WorkerRole.EMBED, WorkerRole.RERANK}
        assert all(i.devices == (0,) for i in plan.instances)

    def test_tensor_splits_model_too_big_for_one_gpu(self) -> None:
        # 30 GB > 24*0.9 per GPU, but each card's even share fits -> split across both.
        model = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_even(model),
        )
        # Equal cards -> equal proportion (int(24*0.9 GiB) = 21 each).
        assert plan.instances == (InstancePlan(WorkerRole.CHAT, (0, 1), (21, 21)),)
        assert plan.unplaceable_roles == ()

    def test_tensor_split_is_proportional_on_unequal_gpus(self) -> None:
        # 28 GB splits across a 24 GB + 16 GB pair; the ratio must follow free VRAM
        # (int(24*0.9)=21, int(16*0.9)=14), not an even 1:1 that would OOM the small card.
        model = ModelPlacementInput(WorkerRole.CHAT, 28 * _GB)
        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 16 * _GB)],
            estimate_peak=_even(model),
        )
        assert plan.instances[0].devices == (0, 1)
        assert plan.instances[0].tensor_split == (21, 14)

    def test_unplaceable_when_model_fits_nowhere(self) -> None:
        model = ModelPlacementInput(WorkerRole.CHAT, 100 * _GB)
        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_even(model),
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_no_gpu_devices_places_every_role_on_cpu(self) -> None:
        # A GPU-less host (or a probe that found nothing): each role runs as a
        # single un-pinned CPU instance so a fleet-of-one works without a GPU.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 5 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ],
            [],
            estimate_peak=_never,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert all(i.devices == () and i.tensor_split == () for i in plan.instances)
        assert plan.unplaceable_roles == ()

    def test_unified_budget_places_roles_that_fit_unpinned(self) -> None:
        # No discrete GPU but a measured shared-RAM budget (Apple Silicon / CPU):
        # roles that fit run un-pinned; system RAM is the shared pool.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 8 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ],
            [],
            estimate_peak=_never,
            unified_budget=12 * _GB,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert all(i.devices == () for i in plan.instances)
        assert plan.unplaceable_roles == ()

    def test_unified_budget_marks_oversize_role_unplaceable(self) -> None:
        # A model larger than free system RAM must NOT load (it would force the OS
        # into an OOM livelock): it is unplaceable, gets no server, calls error.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 20 * _GB)],
            [],
            estimate_peak=_never,
            unified_budget=11 * _GB,
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_shared_pool_reserves_search_roles_before_chat(self) -> None:
        # Search-first: embed is reserved before the elastic chat, so a chat that
        # would consume the whole 12 GB pool is dropped instead of starving search
        # (the proven embed-starvation bug on Apple Silicon).
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 10 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 4 * _GB),
            ],
            [],
            estimate_peak=_never,
            unified_budget=12 * _GB,
        )
        assert {i.role for i in plan.instances} == {WorkerRole.EMBED}
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_search_role_placed_first_then_chat(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 5 * _GB),
                ModelPlacementInput(WorkerRole.CHAT, 20 * _GB),
            ],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_never,
        )
        by_role = {i.role: i.devices for i in plan.instances}
        # Search-first: embed claims device 0, then chat lands on the emptier
        # device 1. Each is a single-GPU instance (bb-7jg1.6).
        assert by_role == {WorkerRole.EMBED: (0,), WorkerRole.CHAT: (1,)}
        assert plan.unplaceable_roles == ()


class TestPerDeviceSplit:
    """The split decision fits and charges each card on its own share, not the summed pool."""

    def test_reserves_more_cards_when_per_device_peak_exceeds_combined_fit(self) -> None:
        # Two cards' combined 90% headroom (43.2 GiB) "fits" the 32 GB single-device
        # estimate, so a combined-scalar planner would cram onto 2 and OOM device 0.
        # The replicated compute buffer makes each card's real share overflow at 2;
        # only a 3-way split fits -> the per-device planner must reserve 3 cards.
        model = ModelPlacementInput(WorkerRole.CHAT, 32 * _GB)

        def peak(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(18 * _GB // len(ratio) + 14 * _GB for _ in ratio)

        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 24 * _GB), (2, 24 * _GB)],
            estimate_peak=peak,
        )
        assert plan.unplaceable_roles == ()
        assert plan.instances[0].devices == (0, 1, 2)

    def test_charges_each_card_its_own_share(self) -> None:
        # Chat splits with an uneven per-device charge (20 GiB, 2 GiB). The elastic
        # vision replica, placed after chat, fits only the less-charged card. A summed
        # charge (22 GiB on both) would leave it nowhere to go.
        chat = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        vision = ModelPlacementInput(WorkerRole.VISION, 6 * _GB, replicas=2)

        def peak(_role: WorkerRole, _ratio: tuple[int, ...]) -> tuple[int, ...]:
            return (20 * _GB, 2 * _GB)

        plan = plan_placement(
            [chat, vision],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=peak,
        )
        elastic = next(i for i in plan.instances if i.role is WorkerRole.VISION and i.replica == 1)
        assert elastic.devices == (0,)  # card 0 (charged 2) has room; card 1 (charged 20) does not
        assert plan.co_tenants == frozenset()

    def test_unplaceable_when_per_device_never_fits(self) -> None:
        model = ModelPlacementInput(WorkerRole.CHAT, 40 * _GB)

        def peak(_role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(30 * _GB for _ in ratio)  # 30 > 24*0.9 at any card count

        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=peak,
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)

    def test_skips_count_when_estimator_returns_wrong_cardinality(self) -> None:
        # A malformed estimate (cardinality != device count) is skipped, not crashed.
        model = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)

        def peak(_role: WorkerRole, _ratio: tuple[int, ...]) -> tuple[int, ...]:
            return (5 * _GB,)  # always one entry, never matching a 2-card split

        plan = plan_placement(
            [model],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=peak,
        )
        assert plan.unplaceable_roles == (WorkerRole.CHAT,)


class TestContextAwareChatSplit:
    """A chat tensor-split widens onto idle cards when a tighter shard would starve
    KV below the context target, instead of always bin-packing onto the fewest GPUs."""

    def test_widens_onto_idle_cards_when_fewest_starves_context(self) -> None:
        # 60 GB chat: a 2-card split overflows, 3 cards fits but only serves 4096,
        # 4 cards serves 16384. Target 8192 -> the planner widens to all four cards
        # rather than stopping at the first (3-card) shard that merely fits.
        chat = ModelPlacementInput(WorkerRole.CHAT, 60 * _GB)
        plan = plan_placement(
            [chat],
            _FOUR_24GB_CARDS,
            estimate_peak=_fits_any_count(chat),
            chat_ctx_fit=_CtxByCount({3: 4096, 4: 16384}),
            chat_ctx_target=8192,
            free_headroom=_FOUR_24GB_FREE,
        )
        assert plan.instances[0].devices == (0, 1, 2, 3)
        assert plan.unplaceable_roles == ()

    def test_stays_at_fewest_when_context_is_already_usable(self) -> None:
        # A 2-card split already serves 16384 >= target, so it does NOT over-spread
        # onto the idle cards (preserving inference speed and residual VRAM).
        chat = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        plan = plan_placement(
            [chat],
            _FOUR_24GB_CARDS,
            estimate_peak=_fits_any_count(chat),
            chat_ctx_fit=_CtxByCount({2: 16384, 3: 24576, 4: 32768}),
            chat_ctx_target=8192,
            free_headroom=_FOUR_24GB_FREE,
        )
        assert plan.instances[0].devices == (0, 1)

    def test_maximizes_context_when_target_unreachable(self) -> None:
        # No shard reaches the 8192 target; the planner falls back to the shard that
        # serves the MOST context (all cards) rather than the fewest.
        chat = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        plan = plan_placement(
            [chat],
            _FOUR_24GB_CARDS,
            estimate_peak=_fits_any_count(chat),
            chat_ctx_fit=_CtxByCount({2: 512, 3: 1024, 4: 2048}),
            chat_ctx_target=8192,
            free_headroom=_FOUR_24GB_FREE,
        )
        assert plan.instances[0].devices == (0, 1, 2, 3)

    def test_fitter_receives_chosen_cards_live_free_headroom(self) -> None:
        # The fitter is sized against the per-device LIVE free VRAM, not total
        # capacity, so a fleet whose cards are partly occupied widens correctly.
        chat = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        fit = _CtxByCount({2: 16384})
        plan_placement(
            [chat],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_fits_any_count(chat),
            chat_ctx_fit=fit,
            chat_ctx_target=8192,
            free_headroom={0: 9 * _GB, 1: 7 * _GB},
        )
        assert fit.calls[0] == [9 * _GB, 7 * _GB]

    def test_non_chat_split_ignores_the_chat_fitter(self) -> None:
        # The context-aware widening is chat-only: a vision split still takes the
        # fewest fitting cards even when a chat fitter is passed.
        vision = ModelPlacementInput(WorkerRole.VISION, 30 * _GB)
        plan = plan_placement(
            [vision],
            _FOUR_24GB_CARDS,
            estimate_peak=_fits_any_count(vision),
            chat_ctx_fit=_CtxByCount({2: 1, 3: 1, 4: 1}),
            chat_ctx_target=8192,
            free_headroom=_FOUR_24GB_FREE,
        )
        assert plan.instances[0].devices == (0, 1)  # fewest, unaffected by the fitter

    def test_default_no_fitter_keeps_first_fit_behavior(self) -> None:
        # Without a fitter (the planner's generic callers / tests), a chat split
        # still takes the fewest fitting cards exactly as before.
        chat = ModelPlacementInput(WorkerRole.CHAT, 30 * _GB)
        plan = plan_placement([chat], _FOUR_24GB_CARDS, estimate_peak=_fits_any_count(chat))
        assert plan.instances[0].devices == (0, 1)


class TestReplicas:
    """Data-parallel replicas: N instances of a role spread one-per-GPU for throughput."""

    def _embeds(self, plan: Placement) -> list[InstancePlan]:
        return [i for i in plan.instances if i.role is WorkerRole.EMBED]

    def test_replicas_spread_one_per_distinct_gpu(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=3)],
            [(0, 24 * _GB), (1, 24 * _GB), (2, 24 * _GB)],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 3
        assert {i.devices[0] for i in embeds} == {0, 1, 2}  # one per distinct card
        assert {i.replica for i in embeds} == {0, 1, 2}  # distinct replica indices
        assert plan.unplaceable_roles == ()

    def test_replicas_wrap_to_colocate_when_more_than_gpus(self) -> None:
        # embed ~10GB, 2x24GB (21.6 usable) fits 2 per card -> 4 of 8 requested place.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 10 * _GB, replicas=8)],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 4  # two per card, then no room
        assert sorted(i.devices[0] for i in embeds) == [0, 0, 1, 1]
        assert {i.replica for i in embeds} == {0, 1, 2, 3}

    def test_replicated_role_unplaceable_when_no_gpu_fits_one(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 100 * _GB, replicas=2)],
            [(0, 24 * _GB)],
            estimate_peak=_never,
        )
        assert plan.instances == ()
        assert plan.unplaceable_roles == (WorkerRole.EMBED,)

    def test_chat_placed_before_replicas_then_replicas_fill_remaining(self) -> None:
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 20 * _GB),  # single, claims a card first
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=2),
            ],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_never,
        )
        assert len([i for i in plan.instances if i.role is WorkerRole.CHAT]) == 1
        assert len(self._embeds(plan)) == 2
        assert plan.unplaceable_roles == ()

    def test_unified_budget_runs_replicas_as_coresident_processes(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 2 * _GB, replicas=3)],
            [],
            estimate_peak=_never,
            unified_budget=10 * _GB,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 3  # 3x2GB <= 10GB
        assert all(i.devices == () for i in embeds)
        assert {i.replica for i in embeds} == {0, 1, 2}

    def test_unified_budget_caps_replicas_to_what_fits(self) -> None:
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 4 * _GB, replicas=5)],
            [],
            estimate_peak=_never,
            unified_budget=10 * _GB,
        )
        assert len(self._embeds(plan)) == 2  # 2x4=8<=10; a third (12) overruns

    def test_no_gpu_no_budget_expands_every_replica(self) -> None:
        # GPU-less + ungated legacy path: each replica becomes an un-pinned instance.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=2)],
            [],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 2
        assert {i.replica for i in embeds} == {0, 1}


class TestPersistentSingleElasticSplit:
    """A replicated embed/vision role reserves replica 0 as a persistent single in the
    singles phase, then places the extra replicas into the residual VRAM."""

    def _embeds(self, plan: Placement) -> list[InstancePlan]:
        return [i for i in plan.instances if i.role is WorkerRole.EMBED]

    def test_replica_zero_is_a_persistent_single_before_the_elastic_batch(self) -> None:
        # embed replicas=3: replica 0 is reserved as the persistent query embedder,
        # alongside chat/rerank, and replicas 1..2 are the elastic ingest pool.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 5 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=3),
            ],
            [(0, 24 * _GB), (1, 24 * _GB), (2, 24 * _GB)],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert {i.replica for i in embeds} == {0, 1, 2}
        # All three placed: one persistent single (replica 0) + two elastic.
        assert len(embeds) == 3
        assert plan.unplaceable_roles == ()

    def test_query_embedder_persists_when_chat_claims_the_reserved_room(self) -> None:
        # One card. Persistent fleet (chat + embed-0) is reserved first, so the
        # query embedder always exists; the elastic replicas get only the residual,
        # which here is exhausted, so embed-1/embed-2 never place.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.EMBED, 8 * _GB, replicas=3),
                ModelPlacementInput(WorkerRole.CHAT, 10 * _GB),
            ],
            [(0, 24 * _GB)],  # 21.6 usable: embed-0 8 + chat 10 = 18, residual 3.6 < 8
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert {i.replica for i in embeds} == {0}  # only the persistent query embedder
        chat = [i for i in plan.instances if i.role is WorkerRole.CHAT]
        assert len(chat) == 1
        assert plan.unplaceable_roles == ()

    def test_elastic_batch_capped_by_residual_vram(self) -> None:
        # 2 cards (21.6 usable each). embed-0 single + chat take card room; the
        # elastic batch fills only what residual VRAM is left, not all replicas.
        plan = plan_placement(
            [
                ModelPlacementInput(WorkerRole.CHAT, 18 * _GB),
                ModelPlacementInput(WorkerRole.EMBED, 10 * _GB, replicas=4),
            ],
            [(0, 24 * _GB), (1, 24 * _GB)],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        # embed-0 single lands on one card; chat lands on the other (18 fits, 21.6);
        # residual on embed-0's card is 11.6 -> one elastic 10GB replica fits there.
        assert 0 in {i.replica for i in embeds}  # persistent single always present
        assert len(embeds) < 4  # capped by residual, not all 4 requested
        assert plan.unplaceable_roles == ()

    def test_replicated_role_unplaceable_when_persistent_single_does_not_fit(self) -> None:
        # If replica 0 (the persistent single) fits nowhere, the role is unplaceable.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 100 * _GB, replicas=2)],
            [(0, 24 * _GB)],
            estimate_peak=_never,
        )
        assert self._embeds(plan) == []
        assert plan.unplaceable_roles == (WorkerRole.EMBED,)

    def test_single_replica_role_is_unchanged(self) -> None:
        # replicas=1 still places exactly one replica-0 single, no elastic batch.
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB, replicas=1)],
            [(0, 24 * _GB)],
            estimate_peak=_never,
        )
        embeds = self._embeds(plan)
        assert len(embeds) == 1
        assert embeds[0].replica == 0


class TestChatVisionCoTenancy:
    """A chat model that crowds out vision makes the pair swap tenants, not unplaceable."""

    def _search(self) -> list[ModelPlacementInput]:
        return [
            ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
        ]

    def _roles(self, plan: Placement) -> set[WorkerRole]:
        return {i.role for i in plan.instances}

    def test_chat_giant_makes_vision_a_co_tenant_instead_of_unplaceable(self) -> None:
        # One 24GB card (21.6 usable): embed+rerank pin 2GB, vision needs 4, chat 18.
        # Chat alone leaves no room for vision, so the pair share a swap group.
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 18 * _GB),
            *self._search(),
            ModelPlacementInput(WorkerRole.VISION, 4 * _GB),
        ]
        plan = plan_placement(models, [(0, 24 * _GB)], estimate_peak=_never)

        assert plan.unplaceable_roles == ()
        assert self._roles(plan) == {
            WorkerRole.CHAT,
            WorkerRole.EMBED,
            WorkerRole.RERANK,
            WorkerRole.VISION,
        }
        assert plan.co_tenants == frozenset({WorkerRole.CHAT, WorkerRole.VISION})

    def test_chat_and_vision_that_both_fit_stay_pinned(self) -> None:
        # An 80GB card holds everything at once: no swap group, no eviction.
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 18 * _GB),
            *self._search(),
            ModelPlacementInput(WorkerRole.VISION, 4 * _GB),
        ]
        plan = plan_placement(models, [(0, 80 * _GB)], estimate_peak=_never)

        assert plan.unplaceable_roles == ()
        assert plan.co_tenants == frozenset()

    def test_vision_too_big_for_the_search_tier_is_still_unplaceable(self) -> None:
        # Vision does not fit even before chat is charged: a real "use a smaller model".
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 3 * _GB),
            *self._search(),
            ModelPlacementInput(WorkerRole.VISION, 6 * _GB),
        ]
        plan = plan_placement(models, [(0, 8 * _GB)], estimate_peak=_never)

        assert plan.unplaceable_roles == (WorkerRole.VISION,)
        assert plan.co_tenants == frozenset()

    def test_co_tenant_vision_runs_a_single_replica(self) -> None:
        # swap:true evicts same-group siblings, so a second vision replica in the
        # co-tenant group would evict the first. The planner must not emit one.
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 18 * _GB),
            *self._search(),
            ModelPlacementInput(WorkerRole.VISION, 4 * _GB, replicas=2),
        ]
        plan = plan_placement(models, [(0, 24 * _GB)], estimate_peak=_never)

        visions = [i for i in plan.instances if i.role is WorkerRole.VISION]
        assert plan.co_tenants == frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert len(visions) == 1
        assert visions[0].replica == 0

    def test_chat_that_fits_nowhere_is_unplaceable_and_vision_stays_pinned(self) -> None:
        # Refunding vision still does not make room: chat is the genuinely oversize one.
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 200 * _GB),
            *self._search(),
            ModelPlacementInput(WorkerRole.VISION, 4 * _GB),
        ]
        plan = plan_placement(models, [(0, 24 * _GB)], estimate_peak=_never)

        assert plan.unplaceable_roles == (WorkerRole.CHAT,)
        assert plan.co_tenants == frozenset()
        assert WorkerRole.VISION in self._roles(plan)


class TestSharedMemoryCoTenancy:
    """The unified-memory path reaches the same verdict as the discrete-GPU path."""

    def test_chat_giant_co_tenants_with_vision_in_shared_memory(self) -> None:
        models = [
            ModelPlacementInput(WorkerRole.CHAT, 18 * _GB),
            ModelPlacementInput(WorkerRole.EMBED, 1 * _GB),
            ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
            ModelPlacementInput(WorkerRole.VISION, 4 * _GB),
        ]
        plan = plan_placement(models, [], estimate_peak=_never, unified_budget=22 * _GB)

        assert plan.unplaceable_roles == ()
        assert plan.co_tenants == frozenset({WorkerRole.CHAT, WorkerRole.VISION})
