"""Tests for fleet launch planning: VRAM estimate, placement, argv, device probe."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType, RerankerType
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice, visible_env
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import RerankMode, WorkerRole

_GB = 1024**3


def _fixed_estimator(*, vram: int = 1024, unified: int | None = None):
    """A gguf-parser stand-in returning a constant footprint (no subprocess)."""
    total_unified = vram if unified is None else unified

    def _est(model_path, **_kwargs) -> GgufVramEstimate:
        return GgufVramEstimate(vram_bytes=vram, ram_bytes=0, unified_bytes=total_unified)

    return _est


def _slotted_estimator(*, base: int, per_slot: int):
    """A gguf-parser stand-in whose footprint grows with the slot count, so the
    slot-fit loop steps down deterministically under a fixed budget."""

    def _est(model_path, *, slots: int, **_kwargs) -> GgufVramEstimate:
        total = base + per_slot * slots
        return GgufVramEstimate(vram_bytes=total, ram_bytes=0, unified_bytes=total)

    return _est


@pytest.fixture(autouse=True)
def _stub_estimator(monkeypatch) -> None:
    """Default: a cheap fixed footprint so tests never shell out to gguf-parser.
    Tests that exercise sizing override this with their own estimator."""
    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _fixed_estimator())


def test_vision_mmproj_returns_path_when_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.find_mmproj_for_model",
        lambda _p: Path("/m/mmproj.gguf"),
    )
    assert planning_mod._vision_mmproj("ref") == Path("/m/mmproj.gguf")


def test_vision_mmproj_returns_none_when_absent(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )

    def _raise(_p: Path) -> Path:
        raise ProviderError("no mmproj")

    monkeypatch.setattr("lilbee.providers.gguf_meta.find_mmproj_for_model", _raise)
    assert planning_mod._vision_mmproj("ref") is None


def test_role_ctx_chat_honors_configured_num_ctx(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "num_ctx", 16384)
    assert planning_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 16384


def test_role_ctx_chat_uses_dynamic_picker_when_unset(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_chat_ctx", lambda _p, _m, *_a: 4096)
    assert planning_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 4096


def test_role_ctx_embed_caps_to_chunk_size(monkeypatch) -> None:
    # A 32K-trained embedder is sized to the chunk length, not its full context, so its
    # placement estimate doesn't balloon (200GB+) and starve the role alongside a giant.
    monkeypatch.setattr(
        "lilbee.providers.engine_params.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 32768,
    )
    monkeypatch.setattr(cfg, "chunk_size", 512)
    assert planning_mod._role_ctx(WorkerRole.EMBED, Path("/m/e.gguf"), {}) == 512
    assert planning_mod._role_ctx(WorkerRole.RERANK, Path("/m/r.gguf"), {}) == 512


def test_role_ctx_embed_uses_train_ctx_when_below_chunk_size(monkeypatch) -> None:
    # A small-context embedder caps at what it was trained for, never above it.
    monkeypatch.setattr(
        "lilbee.providers.engine_params.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 256,
    )
    monkeypatch.setattr(cfg, "chunk_size", 512)
    assert planning_mod._role_ctx(WorkerRole.EMBED, Path("/m/e.gguf"), {}) == 256


def test_role_gpu_layers_marks_embed_roles(monkeypatch) -> None:
    seen: dict[str, bool] = {}

    def _fake(*, embedding: bool) -> int:
        seen["embedding"] = embedding
        return 7

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_n_gpu_layers", _fake)
    assert planning_mod._role_gpu_layers(WorkerRole.RERANK) == 7
    assert seen["embedding"] is True  # rerank is embedding-class
    assert planning_mod._role_gpu_layers(WorkerRole.CHAT) == 7
    assert seen["embedding"] is False  # chat honors cfg.n_gpu_layers


def test_role_gpu_layers_vision_offloads_all_layers(monkeypatch) -> None:
    # The mtmd vision loader hardcodes n_gpu_layers=-1; the fleet must too,
    # not honor cfg.n_gpu_layers for vision.
    seen: dict[str, bool] = {}

    def _fake(*, embedding: bool) -> int:
        seen["embedding"] = embedding
        return -1

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_n_gpu_layers", _fake)
    assert planning_mod._role_gpu_layers(WorkerRole.VISION) == -1
    assert seen["embedding"] is True  # vision => all layers


def test_role_ctx_vision_uses_vision_picker(monkeypatch) -> None:
    # Vision must use the vision loader's training-ctx picker, not cfg.num_ctx
    # or the chat-ctx dynamic picker.
    monkeypatch.setattr(cfg, "num_ctx", 16384)  # would be wrong for vision
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_vision_ctx", lambda _p: 4321)
    assert planning_mod._role_ctx(WorkerRole.VISION, Path("/m/v.gguf"), {}) == 4321


def test_flash_attn_flag_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", None)
    assert planning_mod._flash_attn_flag() == "on"
    monkeypatch.setattr(cfg, "flash_attention", True)
    assert planning_mod._flash_attn_flag() == "on"


def test_flash_attn_flag_off_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", False)
    assert planning_mod._flash_attn_flag() == "off"


def test_slots_for_aux_roles_are_single_slot() -> None:
    # Embed/rerank batch request-side, so their server is always single-slot
    # (and never invoke the estimator).
    assert planning_mod._slots_for(WorkerRole.EMBED, Path("/m/e.gguf"), 0) == 1
    assert planning_mod._slots_for(WorkerRole.RERANK, Path("/m/r.gguf"), 0) == 1


def test_slots_for_chat_is_vram_aware(monkeypatch) -> None:
    # Chat is no longer a fixed 4: a giant on a ~24GB Metal budget steps down.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 24 * 10**9)
    # budget = 24e9 * _CHAT_VRAM_FRACTION(0.8) = 19.2e9; 17e9 + 2e9/slot never fits >1.
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=2 * 10**9),
    )
    assert planning_mod._slots_for(WorkerRole.CHAT, Path("/m/c.gguf"), 65536) == 1


def test_resolve_chat_slots_uses_ceiling_when_vram_is_ample(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 10**12)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536) == 4


def test_resolve_chat_slots_drops_to_one_on_constrained_gpu(monkeypatch) -> None:
    # 17 GB base footprint at >1 slots overruns a ~24GB Metal budget (19.2e9) -> 1.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 24 * 10**9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=2 * 10**9),
    )
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536) == 1


def test_resolve_chat_slots_steps_down_to_fit_unified_budget(monkeypatch) -> None:
    # Ample VRAM keeps 4 slots, but a tight free-RAM budget forces the count down
    # so the model loads at fewer slots instead of being refused at placement.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 64 * 10**9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536) == 4
    assert (
        planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, unified_budget=13 * 10**9) == 1
    )


def test_resolve_chat_slots_reservation_shrinks_budget(monkeypatch) -> None:
    # The search reservation is subtracted from the chat budget, so a chat that
    # fits 4 slots with no reservation steps down once embed/rerank are held back.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 50 * 10**9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=30 * 10**9, per_slot=2 * 10**9),
    )
    # Budget = 50e9 * 0.8 = 40e9. No reservation: 4 slots (38e9) fits.
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536) == 4
    # Reserve 9e9 for search -> budget 31e9; even 2 slots (34e9) overruns -> 1.
    assert (
        planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, chat_reservation=9 * 10**9) == 1
    )


def test_unified_memory_budget_none_when_discrete_gpu_present() -> None:
    # Discrete GPUs load into dedicated VRAM, so system RAM isn't the gate.
    gpu = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 24 * _GB)
    assert planning_mod._unified_memory_budget([gpu]) is None


def test_unified_memory_budget_subtracts_os_floor_when_no_gpu(monkeypatch) -> None:
    # No enumerated GPU (Apple Silicon / CPU): budget = free RAM minus the OS floor.
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 20 * 10**9)
    assert (
        planning_mod._unified_memory_budget([])
        == 20 * 10**9 - planning_mod._SYSTEM_MEMORY_FLOOR_BYTES
    )


def test_unified_memory_budget_floors_at_zero_under_pressure(monkeypatch) -> None:
    # Less free than the floor -> budget 0 -> every model unplaceable (no freeze).
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1 * 10**9)
    assert planning_mod._unified_memory_budget([]) == 0


def test_resolve_vision_slots_uses_ceiling_when_vram_is_ample(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 10**12)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=2 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384) == 4


def test_resolve_vision_slots_drops_to_one_on_small_gpu(monkeypatch) -> None:
    # A tiny VRAM budget can't fit multiple slots of vision footprint -> falls back to 1.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 3 * 10**9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=2 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384) == 1


def test_resolve_vision_slots_ceiling_one_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)
    # Even with huge VRAM, a ceiling of 1 means strictly sequential OCR.
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 10**12)
    assert planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384) == 1


def test_cache_type_flag_none_for_f16(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)
    assert planning_mod._cache_type_flag() is None


def test_cache_type_flag_uses_enum_value(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    assert planning_mod._cache_type_flag() == "q8_0"


def test_server_model_inputs_filters_to_requested_roles(monkeypatch) -> None:
    monkeypatch.setattr(
        planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
    )
    monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    # Only EMBED requested -> chat is filtered out even though it is configured.
    _inputs, refs, _res = planning_mod._server_model_inputs((WorkerRole.EMBED,))
    assert set(refs) == {WorkerRole.EMBED}


def test_server_model_inputs_reserves_search_before_chat_on_shared_host(monkeypatch) -> None:
    # The blocker fix: on a shared-memory host, chat is sized against the budget
    # minus the embed+rerank footprint so a large chat can never starve search.
    monkeypatch.setattr(cfg, "chat_model", "org/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/embed.gguf")
    monkeypatch.setattr(cfg, "reranker_model", "org/rerank.gguf")
    monkeypatch.setattr(cfg, "vision_model", "")
    seen: dict[str, int] = {}
    sizes = {WorkerRole.EMBED: 2 * _GB, WorkerRole.RERANK: 3 * _GB}

    def _estimate(role, ref, *, unified_budget=None, chat_reservation=0):
        if role is WorkerRole.CHAT:
            seen["chat_reservation"] = chat_reservation
        return ModelPlacementInput(role, sizes.get(role, 10 * _GB))

    monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
    _inputs, _refs, reservation = planning_mod._server_model_inputs(unified_budget=20 * _GB)
    assert reservation == 5 * _GB  # embed (2) + rerank (3)
    assert seen["chat_reservation"] == 5 * _GB


def test_server_model_inputs_no_reservation_on_discrete_gpu(monkeypatch) -> None:
    # Discrete GPUs pin each role to its own VRAM and pack independently, so chat
    # is sized with no search reservation (the FFD bin-pack handles co-location).
    monkeypatch.setattr(cfg, "chat_model", "org/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/embed.gguf")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    seen: dict[str, int] = {}

    def _estimate(role, ref, *, unified_budget=None, chat_reservation=0):
        if role is WorkerRole.CHAT:
            seen["chat_reservation"] = chat_reservation
        return ModelPlacementInput(role, 2 * _GB)

    monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
    _inputs, _refs, reservation = planning_mod._server_model_inputs(unified_budget=None)
    assert reservation == 0
    assert seen["chat_reservation"] == 0


def test_replica_count_reads_per_role_knobs(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 3)
    monkeypatch.setattr(cfg, "vision_replicas", 2)
    assert planning_mod._replica_count(WorkerRole.EMBED) == 3
    assert planning_mod._replica_count(WorkerRole.VISION) == 2
    assert planning_mod._replica_count(WorkerRole.CHAT) == 1  # chat never replicates
    assert planning_mod._replica_count(WorkerRole.RERANK) == 1  # rerank never replicates


def test_estimate_role_carries_replica_count(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 4)
    model = tmp_path / "e.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=10))
    inp = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1)
    assert inp.replicas == 4


def test_search_reservation_scales_with_replicas() -> None:
    inputs = {
        WorkerRole.EMBED: ModelPlacementInput(WorkerRole.EMBED, 2 * _GB, replicas=3),
        WorkerRole.RERANK: ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
    }
    # 3 embed replicas + 1 rerank are all reserved ahead of chat.
    assert planning_mod._search_reservation(inputs) == 3 * 2 * _GB + 1 * _GB


class TestBuildFleetWiring:
    def test_server_model_inputs_skips_unconfigured_optional_roles(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod,
            "_estimate_role",
            lambda role, ref, **_k: ModelPlacementInput(role, 5 * _GB),
        )
        monkeypatch.setattr(cfg, "reranker_model", "")  # unconfigured -> skipped
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs, _res = planning_mod._server_model_inputs()
        assert {i.role for i in inputs} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert set(refs) == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_server_model_inputs_skips_role_whose_model_is_not_installed(self, monkeypatch) -> None:
        # Search-only indexing must not require an installed chat model: a
        # configured-but-missing chat model is skipped, not fatal, so the embed
        # server still gets planned.
        from lilbee.providers.base import ProviderError

        def _estimate(role, ref, **_k):
            if role is WorkerRole.CHAT:
                raise ProviderError("not installed", provider="llama-server")
            return ModelPlacementInput(role, _GB)

        monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/missing-chat.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs, _res = planning_mod._server_model_inputs()
        assert WorkerRole.CHAT not in refs
        assert {i.role for i in inputs} == {WorkerRole.EMBED}

    def test_server_model_inputs_includes_configured_rerank(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "some/reranker.gguf")
        monkeypatch.setattr(cfg, "vision_model", "")
        _inputs, refs, _res = planning_mod._server_model_inputs()
        assert WorkerRole.RERANK in refs

    def test_server_model_inputs_includes_vision_only_with_mmproj(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "some/vision.gguf")

        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: None)
        assert WorkerRole.VISION not in planning_mod._server_model_inputs()[1]

        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        assert WorkerRole.VISION in planning_mod._server_model_inputs()[1]

    def test_estimate_role_vision_forwards_mmproj(self, tmp_path, monkeypatch) -> None:
        # gguf-parser counts the projector, so the estimator must receive its path.
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"y" * 500)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: mmproj)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
        seen: dict[str, object] = {}

        def _est(model_path, *, mmproj_path=None, **_k):
            seen["mmproj"] = mmproj_path
            return GgufVramEstimate(vram_bytes=1500, ram_bytes=0, unified_bytes=1500)

        monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
        inp = planning_mod._estimate_role(WorkerRole.VISION, "ref", slots=1)
        assert seen["mmproj"] == mmproj
        assert inp.est_vram_bytes == 1500

    def test_estimate_role_resolves_slots_when_unspecified(self, tmp_path, monkeypatch) -> None:
        # With no explicit slots, the estimate sizes them via _slots_for (vision
        # is memory-aware), so placement and the launched --parallel stay consistent.
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: None)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
        monkeypatch.setattr(
            planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=4242)
        )
        inp = planning_mod._estimate_role(WorkerRole.VISION, "ref")  # slots resolved internally
        assert inp.role is WorkerRole.VISION
        assert inp.est_vram_bytes == 4242

    def test_estimate_role_aux_uses_f16_kv_and_no_flash(self, tmp_path, monkeypatch) -> None:
        # Aux roles run f16 KV regardless of cfg.kv_cache_type and apply no flash
        # attention, so the estimator must be told so (only chat passes --cache-type).
        model = tmp_path / "e.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 512)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)  # would be wrong for embed
        seen: dict[str, object] = {}

        def _est(model_path, *, kv_cache_type, flash_attn, **_k):
            seen["kv"] = kv_cache_type
            seen["flash"] = flash_attn
            return GgufVramEstimate(vram_bytes=10, ram_bytes=0, unified_bytes=10)

        monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
        planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1)
        assert seen["kv"] is KvCacheType.F16
        assert seen["flash"] is False

    def test_estimate_role_charges_unified_footprint_on_shared_host(
        self, tmp_path, monkeypatch
    ) -> None:
        # A shared-memory host charges the unified footprint; a discrete GPU the VRAM one.
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
        monkeypatch.setattr(
            planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=9000, unified=900)
        )
        shared = planning_mod._estimate_role(WorkerRole.CHAT, "ref", slots=1, unified_budget=10**9)
        discrete = planning_mod._estimate_role(WorkerRole.CHAT, "ref", slots=1)
        assert shared.est_vram_bytes == 900
        assert discrete.est_vram_bytes == 9000

    def test_launch_for_vision_passes_mmproj(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: model,
        )
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        plan = InstancePlan(role=WorkerRole.VISION, devices=(0,))
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        launch = planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert "--mmproj" in launch.argv
        assert str(Path("/m/mmproj.gguf")) in launch.argv

    def test_estimate_role_uses_gguf_parser_footprint(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
        monkeypatch.setattr(
            planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=7777)
        )
        inp = planning_mod._estimate_role(WorkerRole.CHAT, "ref", slots=2)
        assert inp.role == WorkerRole.CHAT
        assert inp.est_vram_bytes == 7777

    def test_launch_for_builds_instance_with_pinning(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda ref: model,
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert launch.role == WorkerRole.CHAT
        assert launch.env_overrides == visible_env((device,))
        # port file is stamped with the owning pid so reaping is instance-safe
        assert launch.port_file == Path(f"/data/llama-server-chat-0-{os.getpid()}.port")
        assert "--model" in launch.argv
        assert "--port" not in launch.argv  # claimed at spawn, not here
        assert launch.weights_bytes == 2048  # model file size scales the ready timeout

    def _launch_role(self, tmp_path, monkeypatch, role: WorkerRole, ctx: int = 4096) -> list[str]:
        return self._launch_for_role(tmp_path, monkeypatch, role, ctx).argv

    def test_launch_for_chat_sets_flash_and_cache_type(self, tmp_path, monkeypatch) -> None:
        from lilbee.core.config.enums import KvCacheType

        monkeypatch.setattr(cfg, "flash_attention", None)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.CHAT)
        assert argv[argv.index("--flash-attn") + 1] == "on"
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
        assert "--batch-size" not in argv  # chat is not an embedding role
        assert "--threads" not in argv

    def test_launch_for_chat_f16_omits_cache_type(self, tmp_path, monkeypatch) -> None:
        from lilbee.core.config.enums import KvCacheType

        monkeypatch.setattr(cfg, "flash_attention", False)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.CHAT)
        assert argv[argv.index("--flash-attn") + 1] == "off"
        assert "--cache-type-k" not in argv

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_raise_batch_to_ctx(self, tmp_path, monkeypatch, role) -> None:
        argv = self._launch_role(tmp_path, monkeypatch, role, ctx=8192)
        # full-context embeddings: both batch and ubatch raised (server caps at ubatch)
        assert argv[argv.index("--batch-size") + 1] == "8192"
        assert argv[argv.index("--ubatch-size") + 1] == "8192"
        assert "--flash-attn" not in argv  # embedding path applies no flash attn
        assert "--cache-type-k" not in argv

    def _launch_for_role(self, tmp_path, monkeypatch, role: WorkerRole, ctx: int = 4096):
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: model,
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: ctx)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=role, devices=(0,))
        return planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_set_token_cap(self, tmp_path, monkeypatch, role) -> None:
        launch = self._launch_for_role(tmp_path, monkeypatch, role, ctx=8192)
        # Truncate a few tokens below the per-slot ctx so the server's re-added BOS fits.
        assert launch.token_cap == 8192 - planning_mod._EMBED_CTX_MARGIN

    @pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.VISION])
    def test_launch_for_non_embed_roles_have_no_token_cap(
        self, tmp_path, monkeypatch, role
    ) -> None:
        launch = self._launch_for_role(tmp_path, monkeypatch, role)
        assert launch.token_cap is None

    def test_launch_for_vision_sets_full_core_threads(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod.os, "cpu_count", lambda: 12)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--threads") + 1] == "12"
        assert argv[argv.index("--threads-batch") + 1] == "12"
        assert "--batch-size" not in argv

    def test_launch_for_vision_enables_flash_attn(self, tmp_path, monkeypatch) -> None:
        # Vision OCR pages are slow without flash attention; the in-process path
        # enables it for vision, so the fleet must too (no KV quant either side).
        monkeypatch.setattr(cfg, "flash_attention", None)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--flash-attn") + 1] == "on"
        assert "--cache-type-k" not in argv  # vision applies no KV quant, like the oracle
        monkeypatch.setattr(cfg, "flash_attention", False)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--flash-attn") + 1] == "off"

    def _launch_rerank(self, tmp_path, monkeypatch, arch: str | None):
        model = tmp_path / "r.gguf"
        model.write_bytes(b"x" * 1000)
        meta = {"architecture": arch} if arch is not None else {}
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: meta)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.RERANK, devices=(0,))
        return planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )

    def test_launch_for_rerank_decoder_arch_serves_generatively(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.AUTO)
        monkeypatch.setattr(cfg, "flash_attention", None)
        launch = self._launch_rerank(tmp_path, monkeypatch, "qwen3")
        assert launch.rerank_mode is RerankMode.LLM
        assert "--jinja" in launch.argv
        assert "--pooling" not in launch.argv
        assert "--batch-size" not in launch.argv  # generative, not pooled embeddings
        assert launch.token_cap is None  # LLM path relies on ctx headroom, no truncation
        assert launch.argv[launch.argv.index("--flash-attn") + 1] == "on"

    def test_launch_for_rerank_encoder_arch_stays_cross_encoder(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.AUTO)
        launch = self._launch_rerank(tmp_path, monkeypatch, "bert")
        assert launch.rerank_mode is RerankMode.CROSS_ENCODER
        assert launch.argv[launch.argv.index("--pooling") + 1] == "rank"
        assert "--batch-size" in launch.argv
        assert "--jinja" not in launch.argv

    def test_launch_for_rerank_override_forces_llm_on_encoder(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.LLM)
        launch = self._launch_rerank(tmp_path, monkeypatch, "bert")
        assert launch.rerank_mode is RerankMode.LLM
        assert "--jinja" in launch.argv

    def test_role_ctx_rerank_llm_uses_llm_rerank_ctx(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.LLM)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_llm_rerank_ctx", lambda _m, _p: 1024
        )
        ctx = planning_mod._role_ctx(
            WorkerRole.RERANK, tmp_path / "r.gguf", {"architecture": "qwen3"}
        )
        assert ctx == 1024

    def test_resolve_llm_rerank_ctx_adds_query_headroom(self, tmp_path, monkeypatch) -> None:
        from lilbee.providers import engine_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(engine_params, "train_ctx_from_meta", lambda *a, **k: 40960)
        ctx = engine_params.resolve_llm_rerank_ctx({"architecture": "qwen3"}, tmp_path / "m.gguf")
        assert ctx == 512 + engine_params._LLM_RERANK_HEADROOM

    def test_resolve_llm_rerank_ctx_capped_by_train_ctx(self, tmp_path, monkeypatch) -> None:
        from lilbee.providers import engine_params

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(engine_params, "train_ctx_from_meta", lambda *a, **k: 600)
        assert engine_params.resolve_llm_rerank_ctx({}, tmp_path / "m.gguf") == 600

    def test_launch_for_vision_threads_floor_when_cpu_count_unknown(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(planning_mod.os, "cpu_count", lambda: None)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--threads") + 1] == str(planning_mod._DEFAULT_THREADS)

    def test_plan_all_launches_resolves_devices_and_plans(self, monkeypatch) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        monkeypatch.setattr(planning_mod, "probe_devices", lambda _binary: [device])
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
                0,
            ),
        )
        monkeypatch.setattr(
            planning_mod,
            "plan_placement",
            lambda inputs, devices, *, unified_budget=None: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),), unplaceable_roles=()
            ),
        )
        sentinel = MagicMock()
        monkeypatch.setattr(planning_mod, "_launch_for", lambda *a, **kw: sentinel)
        assert planning_mod.plan_all_launches() == [sentinel]

    def test_plan_all_launches_falls_back_to_vulkan_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        monkeypatch.setattr(planning_mod, "probe_devices", lambda _binary: [])  # can't enumerate
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_select.enumerate_gpu_vram",
            lambda: [(0, 24 * _GB)],
        )
        seen: dict[str, list] = {}
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
                0,
            ),
        )

        def _capture(inputs, devices, *, unified_budget=None):
            seen["devices"] = devices
            return Placement(instances=(), unplaceable_roles=(WorkerRole.CHAT,))

        monkeypatch.setattr(planning_mod, "plan_placement", _capture)
        planning_mod.plan_all_launches()
        assert seen["devices"] == [(0, 24 * _GB)]  # synthesized from the Vulkan fallback
