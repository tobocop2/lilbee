"""Tests for fleet launch planning: VRAM estimate, placement, argv, device probe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType, RerankerType
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import (
    VULKAN_BACKEND,
    DeviceProbe,
    FleetDevice,
    visible_env,
)
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import RerankMode, WorkerRole

_GB = 1024**3


def _card(total_bytes: int, *, index: int = 0) -> FleetDevice:
    """A discrete card of *total_bytes*, the memory sizing budgets come from."""
    return FleetDevice("CUDA", index, f"gpu{index}", total_bytes, total_bytes)


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
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_chat_ctx", lambda _p, _m, **_kw: 4096
    )
    assert planning_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 4096


def test_role_ctx_embed_covers_chunk_size_plus_margin(monkeypatch) -> None:
    # A 32K-trained embedder (and a plain non-LLM reranker) is sized to the chunker's
    # character budget plus the truncation margin (chunk_size * CHARS_PER_TOKEN + margin
    # -- the provable token ceiling for a full chunk), not its full context, so its
    # placement estimate doesn't balloon (200GB+) and starve the role alongside a giant.
    monkeypatch.setattr(
        "lilbee.providers.engine_params.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 32768,
    )
    monkeypatch.setattr(cfg, "chunk_size", 512)
    assert planning_mod._role_ctx(WorkerRole.EMBED, Path("/m/e.gguf"), {}) == 2056
    assert planning_mod._role_ctx(WorkerRole.RERANK, Path("/m/r.gguf"), {}) == 2056


def test_embed_ctx_token_cap_fits_full_chunk(monkeypatch) -> None:
    # The embed input truncates at ctx - _EMBED_CTX_MARGIN, so the server must
    # be sized so a full chunk_size input survives (token_cap >= chunk_size), not 8 short.
    from lilbee.providers.engine_params import _EMBED_CTX_MARGIN, resolve_embed_ctx

    monkeypatch.setattr(
        "lilbee.providers.engine_params.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 32768,
    )
    monkeypatch.setattr(cfg, "chunk_size", 512)
    ctx = resolve_embed_ctx({}, Path("/m/e.gguf"))
    assert ctx - _EMBED_CTX_MARGIN >= cfg.chunk_size


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


def testflash_attn_flag_on_by_default(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", None)
    assert planning_mod.flash_attn_flag() == "on"
    monkeypatch.setattr(cfg, "flash_attention", True)
    assert planning_mod.flash_attn_flag() == "on"


def testflash_attn_flag_off_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "flash_attention", False)
    assert planning_mod.flash_attn_flag() == "off"


def test_slots_for_aux_roles_are_single_slot() -> None:
    # Embed and cross-encoder rerank batch request-side, so their server is
    # single-slot (and never invoke the estimator).
    assert planning_mod._slots_for(WorkerRole.EMBED, Path("/m/e.gguf"), 0) == 1
    assert planning_mod._slots_for(WorkerRole.RERANK, Path("/m/r.gguf"), 0) == 1


def test_slots_for_cross_encoder_rerank_stays_single_slot() -> None:
    n = planning_mod._slots_for(
        WorkerRole.RERANK, Path("/m/r.gguf"), 1024, rerank_mode=RerankMode.CROSS_ENCODER
    )
    assert n == 1


def test_slots_for_llm_rerank_uses_full_fanout_when_vram_ample(monkeypatch) -> None:
    from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY

    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(
        planning_mod, "estimate_instance_footprint", _slotted_estimator(base=10**8, per_slot=10**7)
    )
    n = planning_mod._slots_for(
        WorkerRole.RERANK,
        Path("/m/r.gguf"),
        1024,
        rerank_mode=RerankMode.LLM,
        device=_card(10**12),
    )
    assert n == LLM_RERANK_CONCURRENCY


def test_slots_for_llm_rerank_steps_down_when_vram_tight(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    # budget = 1e9 * 0.75 * _LLM_RERANK_VRAM_FRACTION(0.5) = 3.75e8; 4e8 base fits only 1.
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=4 * 10**8, per_slot=2 * 10**8),
    )
    n = planning_mod._slots_for(
        WorkerRole.RERANK,
        Path("/m/r.gguf"),
        1024,
        rerank_mode=RerankMode.LLM,
        device=_card(10**9),
    )
    assert n == 1


def test_slots_for_chat_is_vram_aware(monkeypatch) -> None:
    # Chat is no longer a fixed 4: a giant on a ~24GB Metal budget steps down.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    # budget = 24e9 * 0.75 * _CHAT_VRAM_FRACTION(0.8) = 14.4e9; 17e9 base never fits.
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=2 * 10**9),
    )
    assert (
        planning_mod._slots_for(WorkerRole.CHAT, Path("/m/c.gguf"), 65536, device=_card(24 * 10**9))
        == 1
    )


def test_resolve_chat_slots_uses_ceiling_when_vram_is_ample(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, device=_card(10**12)) == 4


def test_resolve_chat_slots_drops_to_one_on_constrained_gpu(monkeypatch) -> None:
    # 17 GB base footprint at >1 slots overruns a ~24GB Metal budget (19.2e9) -> 1.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=2 * 10**9),
    )
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, device=_card(24 * 10**9)) == 1


def test_resolve_chat_slots_steps_down_to_fit_unified_budget(monkeypatch) -> None:
    # Ample VRAM keeps 4 slots, but a tight free-RAM budget forces the count down
    # so the model loads at fewer slots instead of being refused at placement.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=17 * 10**9, per_slot=10**9),
    )
    card = _card(64 * 10**9)
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, device=card) == 4
    assert (
        planning_mod._resolve_chat_slots(
            Path("/m/c.gguf"), 65536, unified_budget=13 * 10**9, device=card
        )
        == 1
    )


def test_resolve_chat_slots_reservation_shrinks_budget(monkeypatch) -> None:
    # The search reservation is subtracted from the chat budget, so a chat that
    # fits 4 slots with no reservation steps down once embed/rerank are held back.
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=30 * 10**9, per_slot=2 * 10**9),
    )
    # Budget = 64e9 * 0.75 * 0.8 = 38.4e9. No reservation: 4 slots (38e9) fits.
    card = _card(64 * 10**9)
    assert planning_mod._resolve_chat_slots(Path("/m/c.gguf"), 65536, device=card) == 4
    # Reserve 9e9 for search -> budget 29.4e9; even 2 slots (34e9) overruns -> 1.
    assert (
        planning_mod._resolve_chat_slots(
            Path("/m/c.gguf"), 65536, chat_reservation=9 * 10**9, device=card
        )
        == 1
    )


def test_unified_memory_budget_none_when_discrete_gpu_present() -> None:
    # Discrete GPUs load into dedicated VRAM, so system RAM isn't the gate.
    gpu = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 24 * _GB)
    assert planning_mod._unified_memory_budget([gpu]) is None


def test_unified_memory_budget_subtracts_os_floor_when_no_gpu(monkeypatch) -> None:
    # No enumerated GPU (Apple Silicon / CPU): budget = free RAM minus the OS floor.
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 20 * 10**9)
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 64 * 10**9)
    assert (
        planning_mod._unified_memory_budget([])
        == 20 * 10**9 - planning_mod._SYSTEM_MEMORY_FLOOR_CAP_BYTES
    )


def test_unified_memory_budget_scales_floor_down_on_small_hosts(monkeypatch) -> None:
    # An 8 GB host keeps a quarter of RAM for the OS, not the full 4 GiB cap;
    # otherwise a CI-runner-sized machine refuses to serve even an 80 MB model.
    total = 8 * 10**9
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 4 * 10**9)
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: total)
    assert planning_mod._unified_memory_budget([]) == 4 * 10**9 - total // 4


def test_unified_memory_budget_floors_at_zero_under_pressure(monkeypatch) -> None:
    # Less free than the floor -> budget 0 -> every model unplaceable (no freeze).
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1 * 10**9)
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 64 * 10**9)
    assert planning_mod._unified_memory_budget([]) == 0


def test_resolve_vision_slots_uses_ceiling_when_vram_is_ample(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=2 * 10**9, per_slot=10**9),
    )
    assert planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384, device=_card(10**12)) == 4


def test_resolve_vision_slots_drops_to_one_on_small_gpu(monkeypatch) -> None:
    # A tiny VRAM budget can't fit multiple slots of vision footprint -> falls back to 1.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr(
        planning_mod,
        "estimate_instance_footprint",
        _slotted_estimator(base=2 * 10**9, per_slot=10**9),
    )
    assert (
        planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384, device=_card(3 * 10**9)) == 1
    )


def test_resolve_vision_slots_ceiling_one_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)
    # Even with huge VRAM, a ceiling of 1 means strictly sequential OCR.
    assert planning_mod._resolve_vision_slots(Path("/m/v.gguf"), 16384, device=_card(10**12)) == 1


def test_cache_type_flags_are_absent_for_f16(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)

    assert planning_mod.chat_cache_type_flags() == (None, None)


def test_cache_type_flags_use_the_enum_value(monkeypatch) -> None:
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    monkeypatch.setattr(cfg, "flash_attention", None)

    assert planning_mod.chat_cache_type_flags() == ("q8_0", "q8_0")


def test_only_the_v_cache_falls_back_without_flash_attention(monkeypatch) -> None:
    # llama.cpp refuses a quantized V cache without flash attention; K needs nothing.
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    monkeypatch.setattr(cfg, "flash_attention", False)

    assert planning_mod.chat_cache_type_flags() == ("q8_0", None)


def test_estimator_kv_type_matches_the_launch_without_flash_attention(monkeypatch) -> None:
    # The estimate must match the KV type actually launched.
    from lilbee.core.config.enums import KvCacheType

    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    monkeypatch.setattr(cfg, "flash_attention", False)
    assert planning_mod._role_kv_cache_type(WorkerRole.CHAT) is KvCacheType.Q8_0
    assert planning_mod._role_kv_cache_type_v(WorkerRole.CHAT) is KvCacheType.F16


def test_server_model_inputs_filters_to_requested_roles(monkeypatch) -> None:
    monkeypatch.setattr(
        planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
    )
    monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    # Only EMBED requested -> chat is filtered out even though it is configured.
    _inputs, refs, _res, _skipped = planning_mod._server_model_inputs((WorkerRole.EMBED,))
    assert set(refs) == {WorkerRole.EMBED}


def test_server_model_inputs_skips_sdk_routed_roles(monkeypatch, caplog) -> None:
    """A remote-ref role gets no local server plan and no misleading warning.

    Regression: a cloud chat model (API key set) was fed to the local planner,
    which warned 'is not installed' and left chat unplaced, so the chat surface
    reported an engine error for a model that never needed the engine.
    """
    import logging

    monkeypatch.setattr(
        planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
    )
    monkeypatch.setattr(cfg, "chat_model", "gemini/gemini-2.0-flash")
    monkeypatch.setattr(cfg, "reranker_model", "ollama/bge-reranker")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
        _inputs, refs, _res, skipped = planning_mod._server_model_inputs(
            (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)
        )
    assert WorkerRole.CHAT not in refs
    assert WorkerRole.CHAT not in skipped
    assert WorkerRole.RERANK not in refs  # ollama-managed server, not the fleet's
    assert WorkerRole.RERANK not in skipped
    assert WorkerRole.EMBED in refs
    assert "is not installed" not in caplog.text


def test_server_model_inputs_reserves_search_before_chat_on_shared_host(monkeypatch) -> None:
    # The blocker fix: on a shared-memory host, chat is sized against the budget
    # minus the embed+rerank footprint so a large chat can never starve search.
    monkeypatch.setattr(cfg, "chat_model", "org/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/embed.gguf")
    monkeypatch.setattr(cfg, "reranker_model", "org/rerank.gguf")
    monkeypatch.setattr(cfg, "vision_model", "")
    seen: dict[str, int] = {}
    sizes = {WorkerRole.EMBED: 2 * _GB, WorkerRole.RERANK: 3 * _GB}

    def _estimate(role, ref, *, unified_budget=None, chat_reservation=0, device_count=0):
        if role is WorkerRole.CHAT:
            seen["chat_reservation"] = chat_reservation
        return ModelPlacementInput(role, sizes.get(role, 10 * _GB))

    monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
    _inputs, _refs, reservation, _skipped = planning_mod._server_model_inputs(
        unified_budget=20 * _GB
    )
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

    def _estimate(role, ref, *, unified_budget=None, chat_reservation=0, device_count=0):
        if role is WorkerRole.CHAT:
            seen["chat_reservation"] = chat_reservation
        return ModelPlacementInput(role, 2 * _GB)

    monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
    _inputs, _refs, reservation, _skipped = planning_mod._server_model_inputs(unified_budget=None)
    assert reservation == 0
    assert seen["chat_reservation"] == 0


def test_replica_count_reads_per_role_knobs(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 3)
    monkeypatch.setattr(cfg, "vision_replicas", 2)
    assert planning_mod._replica_count(WorkerRole.EMBED, device_count=4) == 3
    assert planning_mod._replica_count(WorkerRole.VISION, device_count=4) == 2
    assert (
        planning_mod._replica_count(WorkerRole.CHAT, device_count=4) == 1
    )  # chat never replicates
    assert planning_mod._replica_count(WorkerRole.RERANK, device_count=4) == 1  # rerank never


def test_replica_count_auto_uses_device_count(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 0)  # 0 = auto = one per GPU
    monkeypatch.setattr(cfg, "vision_replicas", 0)
    assert planning_mod._replica_count(WorkerRole.EMBED, device_count=4) == 4
    assert planning_mod._replica_count(WorkerRole.VISION, device_count=3) == 3
    # An explicit positive knob wins over the auto device count.
    monkeypatch.setattr(cfg, "embed_replicas", 2)
    assert planning_mod._replica_count(WorkerRole.EMBED, device_count=4) == 2
    # Auto on a GPU-less host still resolves to at least one instance.
    monkeypatch.setattr(cfg, "embed_replicas", 0)
    assert planning_mod._replica_count(WorkerRole.EMBED, device_count=0) == 1
    assert planning_mod._replica_count(WorkerRole.CHAT, device_count=4) == 1


def test_estimate_role_carries_replica_count(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 4)
    model = tmp_path / "e.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=10))
    inp = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1, device_count=2)
    assert inp.replicas == 4  # explicit knob wins over the device count


def test_estimate_role_auto_replicas_follow_device_count(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cfg, "embed_replicas", 0)  # 0 = auto = one per GPU
    model = tmp_path / "e.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=10))
    inp = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1, device_count=3)
    assert inp.replicas == 3


def test_search_reservation_scales_with_replicas() -> None:
    inputs = {
        WorkerRole.EMBED: ModelPlacementInput(WorkerRole.EMBED, 2 * _GB, replicas=3),
        WorkerRole.RERANK: ModelPlacementInput(WorkerRole.RERANK, 1 * _GB),
    }
    # 3 embed replicas + 1 rerank are all reserved ahead of chat.
    assert planning_mod._search_reservation(inputs) == 3 * 2 * _GB + 1 * _GB


def test_placement_estimate_ctx_chat_reserves_target(monkeypatch) -> None:
    """A long-context model reserves the chat ctx target, not its full trained ceiling.

    Reserving the full ceiling over-charges KV and can wrongly reject a split; the
    target is the context we intend to serve, capped by the model and floored at a
    usable minimum.
    """
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    monkeypatch.setattr("lilbee.providers.engine_params.chat_ctx_ceiling", lambda _m, _p: 131072)
    assert planning_mod._placement_estimate_ctx(WorkerRole.CHAT, Path("/m.gguf"), {}) == 24576


def test_placement_estimate_ctx_chat_floored_when_target_tiny(monkeypatch) -> None:
    """A tiny target is floored at the usable minimum so placement still reserves room."""
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 1024)
    monkeypatch.setattr("lilbee.providers.engine_params.chat_ctx_ceiling", lambda _m, _p: 131072)
    assert (
        planning_mod._placement_estimate_ctx(WorkerRole.CHAT, Path("/m.gguf"), {})
        == planning_mod._MIN_USABLE_CHAT_CTX
    )


def test_placement_estimate_ctx_chat_capped_by_short_ceiling(monkeypatch) -> None:
    """A short-context model reserves only its ceiling, below the usable floor."""
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr("lilbee.providers.engine_params.chat_ctx_ceiling", lambda _m, _p: 2048)
    assert planning_mod._placement_estimate_ctx(WorkerRole.CHAT, Path("/m.gguf"), {}) == 2048


def test_placement_estimate_ctx_chat_honors_num_ctx_pin(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "num_ctx", 16384)
    assert planning_mod._placement_estimate_ctx(WorkerRole.CHAT, Path("/m.gguf"), {}) == 16384


def test_estimate_role_chat_footprint_sized_at_target_ctx(monkeypatch) -> None:
    """The single-instance chat footprint reserves KV for the target ctx, not the
    single-card dynamic ctx (which collapses when a big model barely fits one card).

    Regression for bb-9rn: sizing at the target is what lets a too-tight single-card
    placement fall through to a tensor-split instead of a 512-token corner.
    """
    captured: dict[str, int] = {}

    def _est(model_path, *, ctx, slots, **_kwargs) -> GgufVramEstimate:
        captured["ctx"] = ctx
        return GgufVramEstimate(vram_bytes=10**8, ram_bytes=0, unified_bytes=10**8)

    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
    monkeypatch.setattr(planning_mod, "_slots_for", lambda *a, **k: 1)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/c.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr("lilbee.providers.engine_params.chat_ctx_ceiling", lambda _m, _p: 262144)

    planning_mod._estimate_role(WorkerRole.CHAT, "org/chat.gguf")
    assert captured["ctx"] == 24576


def test_placement_estimate_ctx_non_chat_delegates_to_role_ctx(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 512)
    assert planning_mod._placement_estimate_ctx(WorkerRole.EMBED, Path("/m.gguf"), {}) == 512


def test_placement_estimate_slots_per_role(monkeypatch) -> None:
    # A tensor-split chat reserves for one full-context sequence, not _CHAT_SLOTS.
    assert (
        planning_mod._placement_estimate_slots(WorkerRole.CHAT, {})
        == planning_mod._SPLIT_CHAT_SLOTS
    )
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 3)
    assert planning_mod._placement_estimate_slots(WorkerRole.VISION, {}) == 3
    assert planning_mod._placement_estimate_slots(WorkerRole.EMBED, {}) == planning_mod._AUX_SLOTS


def test_placement_estimate_slots_rerank_modes(monkeypatch) -> None:
    from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY

    monkeypatch.setattr(planning_mod, "_rerank_mode_for", lambda _m: RerankMode.LLM)
    assert planning_mod._placement_estimate_slots(WorkerRole.RERANK, {}) == LLM_RERANK_CONCURRENCY
    monkeypatch.setattr(planning_mod, "_rerank_mode_for", lambda _m: RerankMode.CROSS_ENCODER)
    assert planning_mod._placement_estimate_slots(WorkerRole.RERANK, {}) == planning_mod._AUX_SLOTS


def test_peak_estimator_returns_per_device_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/c.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_placement_estimate_slots", lambda _r, _m: 2)
    monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda _r, _p, _m: 1000)
    seen: dict = {}

    def _est(_path, *, ctx, slots, tensor_split, mmproj_path=None, **_k) -> GgufVramEstimate:
        seen.update(ctx=ctx, slots=slots, ratio=tensor_split, mmproj=mmproj_path)
        return GgufVramEstimate(
            vram_bytes=0, ram_bytes=0, unified_bytes=0, per_device_vram=(11, 22)
        )

    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
    estimate = planning_mod._peak_estimator({WorkerRole.CHAT: "ref"})
    assert estimate(WorkerRole.CHAT, (1, 1)) == (11, 22)
    # Per-slot context in; the estimator turns it into a total for the parser.
    assert seen["ctx"] == 1000 and seen["slots"] == 2 and seen["ratio"] == (1, 1)
    assert seen["mmproj"] is None  # chat carries no projector


def test_peak_estimator_vision_passes_mmproj(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/v.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
    monkeypatch.setattr(planning_mod, "_placement_estimate_slots", lambda _r, _m: 1)
    monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda _r, _p, _m: 100)
    seen: dict = {}

    def _est(_path, *, mmproj_path=None, **_k) -> GgufVramEstimate:
        seen["mmproj"] = mmproj_path
        return GgufVramEstimate(vram_bytes=0, ram_bytes=0, unified_bytes=0, per_device_vram=(5, 5))

    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
    estimate = planning_mod._peak_estimator({WorkerRole.VISION: "ref"})
    assert estimate(WorkerRole.VISION, (1, 1)) == (5, 5)
    assert seen["mmproj"] == Path("/m/mmproj.gguf")


def test_peak_estimator_reserves_single_sequence_total_for_split_chat(monkeypatch) -> None:
    # Regression (bb-xly): a tensor-split chat reserves the per-sequence ceiling as the
    # total --ctx-size (slots=1), not ceiling x _CHAT_SLOTS, which would over-reserve KV
    # no launch allocates and wrongly mark a large-context giant unplaceable.
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/c.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda _r, _p, _m: 196608)
    seen: dict = {}

    def _est(_path, *, ctx, slots, tensor_split, **_k) -> GgufVramEstimate:
        seen.update(ctx=ctx, slots=slots)
        return GgufVramEstimate(
            vram_bytes=0, ram_bytes=0, unified_bytes=0, per_device_vram=(5, 5, 5)
        )

    monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _est)
    planning_mod._peak_estimator({WorkerRole.CHAT: "ref"})(WorkerRole.CHAT, (1, 1, 1))
    assert seen["slots"] == planning_mod._SPLIT_CHAT_SLOTS
    assert seen["ctx"] == 196608  # ceiling x 1, not ceiling x _CHAT_SLOTS


class TestBuildFleetWiring:
    def test_server_model_inputs_skips_unconfigured_optional_roles(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod,
            "_estimate_role",
            lambda role, ref, **_k: ModelPlacementInput(role, 5 * _GB),
        )
        monkeypatch.setattr(cfg, "reranker_model", "")  # unconfigured -> skipped
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs, _res, _skipped = planning_mod._server_model_inputs()
        assert {i.role for i in inputs} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert set(refs) == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_server_model_inputs_skips_role_whose_model_is_not_installed(self, monkeypatch) -> None:
        # Search-only indexing must not require an installed chat model: a
        # configured-but-missing chat model is skipped, not fatal, so the embed
        # server still gets planned.
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def _estimate(role, ref, **_k):
            if role is WorkerRole.CHAT:
                raise ProviderError(
                    "not installed", provider="llama-server", kind=ProviderErrorKind.NOT_FOUND
                )
            return ModelPlacementInput(role, _GB)

        monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/missing-chat.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs, _res, skipped = planning_mod._server_model_inputs()
        assert WorkerRole.CHAT not in refs
        assert {i.role for i in inputs} == {WorkerRole.EMBED}
        # The missing chat model is reported as not-installed so a surface can say so.
        assert skipped == {WorkerRole.CHAT: "org/repo/missing-chat.gguf"}

    def test_server_model_inputs_distinguishes_sizing_failure_from_missing(
        self, monkeypatch, caplog
    ) -> None:
        # A sizing failure (estimator errored) must not be reported as "not
        # installed" -- that misdirects debugging toward the registry when the
        # real fault is the memory estimator.
        import logging

        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def _estimate(role, ref, **_k):
            if role is WorkerRole.EMBED:
                raise ProviderError(
                    "no file for ref",
                    provider="llama-server",
                    kind=ProviderErrorKind.NOT_FOUND,
                )
            raise ProviderError(
                "unexpected estimator output",
                provider="llama-server",
                kind=ProviderErrorKind.SERVER,
            )

        monkeypatch.setattr(planning_mod, "_estimate_role", _estimate)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        with caplog.at_level(logging.WARNING):
            inputs, refs, _res, _skipped = planning_mod._server_model_inputs()
        assert not refs and not inputs
        # The genuinely-missing embed model says so; the chat sizing failure
        # names the estimator instead of misdirecting toward the registry.
        assert "model 'org/repo/embed.gguf' is not installed" in caplog.text
        assert "could not size model 'org/repo/chat.gguf'" in caplog.text
        assert "could not size model 'org/repo/embed.gguf'" not in caplog.text
        assert "model 'org/repo/chat.gguf' is not installed" not in caplog.text

    def test_server_model_inputs_includes_configured_rerank(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "some/reranker.gguf")
        monkeypatch.setattr(cfg, "vision_model", "")
        _inputs, refs, _res, _skipped = planning_mod._server_model_inputs()
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
        shared = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1, unified_budget=10**9)
        discrete = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1)
        assert shared.est_vram_bytes == 900
        assert discrete.est_vram_bytes == 9000

    def test_estimate_role_chat_charged_at_serve_budget(self, monkeypatch) -> None:
        """A chat instance's placement footprint is scaled up by the placement/serve
        budget ratio, so a model that would starve its KV on one card is tensor-split.

        Regression for bb-9rn: without this, a 17GB model fits one 24GB card at the 0.9
        placement headroom but its served ctx (sized at 0.75) collapses to ~512 tokens.
        """
        monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path", lambda _r: Path("/m/c.gguf")
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 16)
        monkeypatch.setattr(planning_mod, "_slots_for", lambda *a, **k: 1)
        monkeypatch.setattr(
            planning_mod, "estimate_instance_footprint", _fixed_estimator(vram=10000)
        )

        chat = planning_mod._estimate_role(WorkerRole.CHAT, "ref")
        embed = planning_mod._estimate_role(WorkerRole.EMBED, "ref")
        # chat charged at the serve budget: 10000 * (USABLE_VRAM_FRACTION / 0.75); embed raw.
        assert chat.est_vram_bytes == int(10000 * (planning_mod.USABLE_VRAM_FRACTION / 0.75))
        assert embed.est_vram_bytes == 10000

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
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})
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
        inp = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=2)
        assert inp.role == WorkerRole.EMBED
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
        # Pin the runtime env to empty: the real one reflects whatever CUDA
        # wheels the host venv has installed, and the assertion below checks
        # exact env equality for the pinning keys.
        monkeypatch.setattr(planning_mod, "llama_server_runtime_env", lambda: {})
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})
        assert launch.role == WorkerRole.CHAT
        assert launch.env_overrides == visible_env((device,))
        assert "--model" in launch.argv
        assert "--port" not in launch.argv  # claimed at spawn, not here
        assert launch.weights_bytes == 2048  # model file size scales the ready timeout

    def test_launch_for_split_chat_sizes_ctx_against_per_device_headroom(
        self, tmp_path, monkeypatch
    ) -> None:
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", None)
        seen: dict = {}

        def _fit(_model_path, *, slots, ratio, per_device_free_bytes, **_k) -> int:
            # Only one full window fits (a tight split): more slots shrink the fit,
            # so the chooser keeps a single full-context sequence.
            seen.update(ratio=ratio, free=per_device_free_bytes)
            return 5000 if slots == 1 else 4000

        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", _fit)
        d0 = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 70 * _GB)
        d1 = FleetDevice("CUDA", 1, "gpu", 80 * _GB, 60 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: d0, 1: d1})
        assert launch.ctx == 5000
        assert seen["ratio"] == (1, 1)
        assert seen["free"] == [70 * _GB, 60 * _GB]  # per-device free, not the summed pool
        assert launch.slots == 1  # tight split: one full-context sequence
        assert launch.argv[launch.argv.index("--ctx-size") + 1] == str(5000)

    def test_launch_for_split_chat_serves_multiple_slots_when_headroom_holds_them(
        self, tmp_path, monkeypatch
    ) -> None:
        # A split whose cards hold several full windows serves that many agents
        # concurrently: the reel/multi-agent case on 2 big cards.
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", None)

        # Every slot count still reaches the full window: 4 agents fit.
        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", lambda *_a, **_k: 65536)
        d0 = FleetDevice("CUDA", 0, "gpu", 143 * _GB, 130 * _GB)
        d1 = FleetDevice("CUDA", 1, "gpu", 143 * _GB, 130 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: d0, 1: d1})
        assert launch.slots == planning_mod._CHAT_SLOTS  # all four full windows fit
        assert launch.ctx == 65536  # each agent keeps the full window
        # --ctx-size is the per-slot window times the slot count.
        total = 65536 * planning_mod._CHAT_SLOTS
        assert launch.argv[launch.argv.index("--ctx-size") + 1] == str(total)

    def test_launch_for_warns_on_oversize_network_fs_chat(self, tmp_path, monkeypatch, caplog):
        # A chat model served from a network volume that can't fit host RAM keeps
        # mmap, which can hang the load; warn to advise local staging.
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        monkeypatch.setattr(planning_mod, "is_network_path", lambda _p: True)
        monkeypatch.setattr(planning_mod, "_weights_bytes", lambda _p: 200 * 10**9)
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 100 * 10**9)
        device = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 70 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        with caplog.at_level("WARNING", logger="lilbee.providers.fleet.planning"):
            launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})
        assert "network filesystem" in caplog.text.lower()
        assert "--no-mmap" not in launch.argv  # too big to malloc; mmap kept

    def test_launch_for_split_chat_subtracts_reserved_headroom(self, tmp_path, monkeypatch) -> None:
        # An embed/rerank server on a shared card leaves less room for the chat KV
        # than the card's raw free VRAM; sizing the split against raw free over-commits
        # and OOMs at launch. The reservation is subtracted per device.
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", None)
        seen: dict = {}

        def _fit(_model_path, *, per_device_free_bytes, **_k) -> int:
            seen["free"] = per_device_free_bytes
            return 5000

        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", _fit)
        d0 = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 70 * _GB)
        d1 = FleetDevice("CUDA", 1, "gpu", 80 * _GB, 60 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        planning_mod._launch_for(
            plan,
            "ref",
            Path("/bin/llama-server"),
            {0: d0, 1: d1},
            reserved_by_device={0: 10 * _GB},  # an embed server sits on card 0
        )
        assert seen["free"] == [60 * _GB, 60 * _GB]  # card 0 reduced by the 10 GiB embed

    def test_launch_for_warns_on_pcie_split_chat(self, tmp_path, monkeypatch, caplog):
        # A chat model tensor-split across GPUs with no NVLink is all-reduce bound;
        # warn so the slow-generation cause is visible.
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", None)
        monkeypatch.setattr(planning_mod, "host_lacks_nvlink", lambda: True)
        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", lambda *_a, **_k: 5000)
        d0 = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 70 * _GB)
        d1 = FleetDevice("CUDA", 1, "gpu", 80 * _GB, 60 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        with caplog.at_level("WARNING", logger="lilbee.providers.fleet.planning"):
            planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: d0, 1: d1})
        assert "nvlink" in caplog.text.lower()

    def test_non_chat_reservation_sums_per_device(self) -> None:
        instances = [
            InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1)),
            InstancePlan(role=WorkerRole.EMBED, devices=(0,)),
            InstancePlan(role=WorkerRole.EMBED, devices=(0,), replica=1),
            InstancePlan(role=WorkerRole.RERANK, devices=(1,)),
        ]
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 40 * _GB),
            ModelPlacementInput(WorkerRole.EMBED, 3 * _GB),
            ModelPlacementInput(WorkerRole.RERANK, 2 * _GB),
        ]
        reserved = planning_mod._non_chat_reservation(instances, inputs)
        # Chat is excluded (it sizes its own weights); two embed replicas stack on card 0.
        assert reserved == {0: 6 * _GB, 1: 2 * _GB}

    def test_non_chat_reservation_excludes_chats_co_tenants(self) -> None:
        # A co-tenant vision is evicted while chat is resident, so its VRAM must not
        # be held back from the chat shard's KV; only the pinned embed is reserved.
        instances = [
            InstancePlan(role=WorkerRole.CHAT, devices=(0,)),
            InstancePlan(role=WorkerRole.VISION, devices=(0,)),
            InstancePlan(role=WorkerRole.EMBED, devices=(0,)),
        ]
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 40 * _GB),
            ModelPlacementInput(WorkerRole.VISION, 6 * _GB),
            ModelPlacementInput(WorkerRole.EMBED, 3 * _GB),
        ]
        reserved = planning_mod._non_chat_reservation(
            instances, inputs, frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        )
        assert reserved == {0: 3 * _GB}

    def test_non_chat_reservation_charges_a_co_tenant_group_without_chat(self) -> None:
        # A vision/rerank swap group that excludes chat runs behind its own process
        # and can be resident beside a chat shard, so its members are charged (not
        # treated as chat's to reclaim); only chat itself is excluded.
        instances = [
            InstancePlan(role=WorkerRole.CHAT, devices=(0,)),
            InstancePlan(role=WorkerRole.VISION, devices=(0,)),
            InstancePlan(role=WorkerRole.RERANK, devices=(0,)),
            InstancePlan(role=WorkerRole.EMBED, devices=(0,)),
        ]
        inputs = [
            ModelPlacementInput(WorkerRole.CHAT, 40 * _GB),
            ModelPlacementInput(WorkerRole.VISION, 6 * _GB),
            ModelPlacementInput(WorkerRole.RERANK, 2 * _GB),
            ModelPlacementInput(WorkerRole.EMBED, 3 * _GB),
        ]
        reserved = planning_mod._non_chat_reservation(
            instances, inputs, frozenset({WorkerRole.VISION, WorkerRole.RERANK})
        )
        assert reserved == {0: (6 + 2 + 3) * _GB}

    def test_launch_for_pinned_multi_card_chat_runs_one_slot(self, tmp_path, monkeypatch) -> None:
        # A cfg.num_ctx pin skips the fit, but a multi-card chat still serves one slot
        # so --ctx-size matches the single-sequence footprint the planner reserved.
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", 8192)

        def _fail(*_a, **_k) -> int:
            raise AssertionError("fit_split_ctx must not run when cfg.num_ctx pins the context")

        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", _fail)
        d0 = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 70 * _GB)
        d1 = FleetDevice("CUDA", 1, "gpu", 80 * _GB, 60 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: d0, 1: d1})
        assert launch.ctx == 8192
        assert launch.slots == planning_mod._SPLIT_CHAT_SLOTS

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
        return planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_set_token_cap(self, tmp_path, monkeypatch, role) -> None:
        from lilbee.providers.engine_params import _EMBED_CTX_MARGIN

        launch = self._launch_for_role(tmp_path, monkeypatch, role, ctx=8192)
        # Truncate a few tokens below the per-slot ctx so the server's re-added BOS fits.
        assert launch.token_cap == 8192 - _EMBED_CTX_MARGIN

    @pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.VISION])
    def test_launch_for_non_embed_roles_have_no_token_cap(
        self, tmp_path, monkeypatch, role
    ) -> None:
        launch = self._launch_for_role(tmp_path, monkeypatch, role)
        assert launch.token_cap is None

    def test_launch_for_vision_leaves_the_thread_count_to_the_engine(
        self, tmp_path, monkeypatch
    ) -> None:
        """os.cpu_count() counts SMT siblings, efficiency cores and the host's cores
        inside a cgroup-limited container. llama.cpp counts physical math cores and
        skips efficiency cores deliberately, so the override was strictly worse
        informed than the default it replaced.
        """
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert "--threads" not in argv
        assert "--threads-batch" not in argv
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
        return planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})

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

    def test_launch_for_llm_rerank_parallel_matches_fanout(self, tmp_path, monkeypatch) -> None:
        from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY

        # a 24 GiB card + the autouse fixed-footprint estimator => the full fan-out fits
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.AUTO)
        launch = self._launch_rerank(tmp_path, monkeypatch, "qwen3")
        assert launch.argv[launch.argv.index("--parallel") + 1] == str(LLM_RERANK_CONCURRENCY)

    def test_estimate_role_rerank_threads_llm_mode_to_slots(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "r.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "qwen3"}
        )
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda *a, **k: 1024)
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.AUTO)
        captured: dict[str, object] = {}

        def _fake_slots(role, path, ctx, **kw):
            captured["rerank_mode"] = kw.get("rerank_mode")
            return 8

        monkeypatch.setattr(planning_mod, "_slots_for", _fake_slots)
        planning_mod._estimate_role(WorkerRole.RERANK, "ref")
        assert captured["rerank_mode"] is RerankMode.LLM

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

    def test_plan_all_launches_resolves_devices_and_plans(self, monkeypatch) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
                0,
                {},
            ),
        )
        monkeypatch.setattr(
            planning_mod,
            "plan_placement",
            lambda inputs, devices, *, estimate_peak, unified_budget=None, **_kw: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),), unplaceable_roles=()
            ),
        )
        sentinel = MagicMock()
        monkeypatch.setattr(planning_mod, "_launch_for", lambda *a, **kw: sentinel)
        assert planning_mod.plan_all_launches() == planning_mod.FleetPlan((sentinel,))

    def test_plan_all_launches_carries_skipped_not_installed(self, monkeypatch) -> None:
        # The plan surfaces a configured-but-missing chat model so the provider's
        # warm path can fail with a named reason instead of spinning.
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.EMBED, 1 * _GB)],
                {WorkerRole.EMBED: "eref"},
                0,
                {WorkerRole.CHAT: "org/repo/missing-chat.gguf"},
            ),
        )
        monkeypatch.setattr(
            planning_mod,
            "plan_placement",
            lambda inputs, devices, *, estimate_peak, unified_budget=None, **_kw: Placement(
                instances=(InstancePlan(WorkerRole.EMBED, (0,)),), unplaceable_roles=()
            ),
        )
        monkeypatch.setattr(planning_mod, "_launch_for", lambda *a, **kw: MagicMock())
        plan = planning_mod.plan_all_launches()
        assert plan.skipped_not_installed == {WorkerRole.CHAT: "org/repo/missing-chat.gguf"}

    def test_plan_launches_reports_co_tenant_roles(self, monkeypatch, caplog) -> None:
        # Co-tenancy changes how the box behaves (one model resident at a time), so it
        # is stated in the log rather than being inferred from a silent plan.
        import logging

        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref", WorkerRole.VISION: "vref"},
                0,
                {},
            ),
        )
        monkeypatch.setattr(
            planning_mod,
            "plan_placement",
            lambda inputs, devices, *, estimate_peak, unified_budget=None, **_kw: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),),
                unplaceable_roles=(),
                co_tenants=frozenset({WorkerRole.CHAT, WorkerRole.VISION}),
            ),
        )
        monkeypatch.setattr(planning_mod, "_launch_for", lambda *a, **kw: MagicMock())

        with caplog.at_level(logging.INFO):
            plan = planning_mod.plan_all_launches()

        assert plan.co_tenants == frozenset({WorkerRole.CHAT, WorkerRole.VISION})
        assert "chat, vision" in caplog.text
        assert "only one is resident" in caplog.text

    def test_plan_all_launches_falls_back_to_vulkan_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/llama-server"))
        # The engine could not answer at all, which is the only case the fallback
        # is for; a clean run listing nothing is a verdict and gets believed.
        monkeypatch.setattr(
            planning_mod, "probe_devices", lambda _binary: DeviceProbe([], "", spoke_protocol=False)
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_select.enumerate_gpu_vram",
            lambda: [(0, 24 * _GB, 20 * _GB)],
        )
        seen: dict[str, list] = {}
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles, **_kw: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
                0,
                {},
            ),
        )

        def _capture(inputs, devices, *, estimate_peak, unified_budget=None, **_kw):
            seen["devices"] = devices
            return Placement(instances=(), unplaceable_roles=(WorkerRole.CHAT,))

        monkeypatch.setattr(planning_mod, "plan_placement", _capture)
        planning_mod.plan_all_launches()
        assert seen["devices"] == [(0, 24 * _GB)]  # synthesized from the Vulkan fallback


class TestChatSplitCtxObjective:
    """The chat split's context fitter/target wiring and the bb-a8f charging basis."""

    def test_no_chat_model_yields_no_objective(self) -> None:
        fit, target = planning_mod._chat_split_ctx_objective({WorkerRole.EMBED: "e"})
        assert fit is None
        assert target == 0

    def test_fitter_delegates_to_fit_split_ctx_with_launch_sizing(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path", lambda _ref: Path("/m.gguf")
        )
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"arch": "x"}
        )
        monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda *_a: 8192)
        captured: dict[str, object] = {}

        def _fake_fit(model_path, **kw):
            captured.update(kw)
            captured["model_path"] = model_path
            return 4096

        monkeypatch.setattr("lilbee.providers.fleet.ctx.fit_split_ctx", _fake_fit)
        fit, target = planning_mod._chat_split_ctx_objective({WorkerRole.CHAT: "ref"})
        assert target == 8192
        assert fit is not None
        assert fit((21, 21), [24 * _GB, 18 * _GB]) == 4096
        assert captured["model_path"] == Path("/m.gguf")
        assert captured["ratio"] == (21, 21)
        assert captured["per_device_free_bytes"] == [24 * _GB, 18 * _GB]
        assert captured["slots"] == planning_mod._SPLIT_CHAT_SLOTS
        # The fit is bounded by the planned working context (bb-ev9), not the model max.
        assert captured["ctx_ceiling"] == 8192

    def test_resolve_placement_wires_objective_and_keeps_total_charging(self, monkeypatch) -> None:
        # The chat fitter is sized against LIVE free VRAM, but charging stays on TOTAL
        # capacity (bb-a8f), so a warm fleet's own residents aren't double-counted.
        devices = [
            FleetDevice("CUDA", 0, "gpu", 24 * _GB, 20 * _GB),
            FleetDevice("CUDA", 1, "gpu", 24 * _GB, 18 * _GB),
        ]
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path", lambda _ref: Path("/m.gguf")
        )
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"arch": "x"}
        )
        monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda *_a: 8192)
        monkeypatch.setattr(planning_mod, "_peak_estimator", lambda _refs: lambda *_a: (1,))
        captured: dict[str, object] = {}

        def _capture(inputs, devs, *, estimate_peak, unified_budget=None, **kw):
            captured.update(kw)
            captured["devs"] = devs
            return Placement(instances=(), unplaceable_roles=())

        monkeypatch.setattr(planning_mod, "plan_placement", _capture)
        planning_mod._resolve_placement(
            None, [], {WorkerRole.CHAT: "ref"}, devices, unified_budget=None
        )
        assert captured["chat_ctx_target"] == 8192
        assert callable(captured["chat_ctx_fit"])
        assert captured["free_headroom"] == {0: 20 * _GB, 1: 18 * _GB}
        assert captured["devs"] == [(0, 24 * _GB), (1, 24 * _GB)]


class TestResolveDevicesProbeFailureWarning:
    def test_warns_when_probe_finds_nothing_on_an_nvidia_host(self, monkeypatch, caplog) -> None:
        # A driver hiccup on a CUDA pod must not silently fall into the unified
        # shared-memory path; the operator needs a loud signal of what to check.
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
        monkeypatch.setattr("lilbee.providers.fleet.gpu_select.enumerate_gpu_vram", lambda: [])
        with caplog.at_level("WARNING", logger=planning_mod.__name__):
            devices = planning_mod.resolve_devices(Path("/bin/llama-server"))
        assert devices == []
        assert any("shared-memory mode" in record.message for record in caplog.records)

    def test_no_warning_without_an_nvidia_gpu(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: False)
        monkeypatch.setattr("lilbee.providers.fleet.gpu_select.enumerate_gpu_vram", lambda: [])
        with caplog.at_level("WARNING", logger=planning_mod.__name__):
            planning_mod.resolve_devices(Path("/bin/llama-server"))
        assert not any("shared-memory mode" in record.message for record in caplog.records)

    def test_no_warning_when_probe_succeeds(self, monkeypatch, caplog) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([device], "Available devices:\n", spoke_protocol=True),
        )
        with caplog.at_level("WARNING", logger=planning_mod.__name__):
            assert planning_mod.resolve_devices(Path("/bin/llama-server")) == [device]
        assert not caplog.records

    def test_raises_when_cuda_build_enumerates_no_device(self, monkeypatch) -> None:
        # The bb-3xnx failure: a CUDA-linked engine + an NVIDIA GPU, but the probe
        # sees nothing (a runtime the probe can't load). Must hard-fail, not fall back.
        from lilbee.providers.base import ProviderError
        from lilbee.providers.fleet import cuda_runtime

        monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe([], "Available devices:\n", spoke_protocol=True),
        )
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: True)
        with pytest.raises(ProviderError, match="no CUDA-capable device"):
            planning_mod.resolve_devices(Path("/bin/llama-server"))

    def test_an_engine_that_never_answered_is_not_accused_of_a_broken_driver(
        self, monkeypatch
    ) -> None:
        """The fail-loud guard reads an empty device list as a driver that would
        not initialize. A binary with no --list-devices support enumerated
        nothing because it was never asked, so accusing its driver is both wrong
        and fatal, and it pre-empts the fallback that rescues those engines."""
        from lilbee.providers.fleet import cuda_runtime, gpu_select

        monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
        monkeypatch.setattr(
            planning_mod,
            "probe_devices",
            lambda _binary: DeviceProbe(
                [], "error: invalid argument: --list-devices\n", spoke_protocol=False
            ),
        )
        monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda *_a: True)
        monkeypatch.setattr(gpu_select, "enumerate_gpu_vram", lambda: [(0, 8 * _GB, 8 * _GB)])
        monkeypatch.setattr(gpu_select, "integrated_vulkan_indices", frozenset)

        devices = planning_mod.resolve_devices(Path("/bin/llama-server"))

        assert [(d.backend, d.index) for d in devices] == [("Vulkan", 0)]

    def test_probe_timeout_propagates_without_vulkan_fallback(self, monkeypatch) -> None:
        # A wedged probe raises; falling into the in-process Vulkan probe there
        # could hang this thread unkillably against the same wedged driver.
        from lilbee.providers.base import ProviderError

        def _wedged(_binary: Path) -> DeviceProbe:
            raise ProviderError("The GPU device probe did not respond")

        def _must_not_run() -> list[tuple[int, int]]:
            raise AssertionError("Vulkan fallback must not run after a probe timeout")

        monkeypatch.setattr(planning_mod, "probe_devices", _wedged)
        monkeypatch.setattr("lilbee.providers.fleet.gpu_select.enumerate_gpu_vram", _must_not_run)
        with pytest.raises(ProviderError, match="did not respond"):
            planning_mod.resolve_devices(Path("/bin/llama-server"))


class TestReadDeviceCacheFailure:
    """A probe failure is cached and re-raised so a wedged host is not re-probed per poll."""

    def _cache(self) -> planning_mod._ReadDeviceCache:
        return planning_mod._ReadDeviceCache(ttl_s=0.0, failure_ttl_s=60.0)

    def test_failure_is_cached_and_reraised(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        calls = {"n": 0}

        def _wedged(_binary: Path) -> list[FleetDevice]:
            calls["n"] += 1
            raise ProviderError("probe wedged")

        monkeypatch.setattr(planning_mod, "resolve_devices", _wedged)
        cache = self._cache()
        for _ in range(3):
            with pytest.raises(ProviderError, match="probe wedged"):
                cache.get(Path("/bin/llama-server"))
        assert calls["n"] == 1  # later reads served from the cached failure

    def test_clear_drops_the_cached_failure(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        outcomes: list[object] = [ProviderError("probe wedged"), [device]]

        def _next(_binary: Path) -> list[FleetDevice]:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(planning_mod, "resolve_devices", _next)
        cache = self._cache()
        with pytest.raises(ProviderError):
            cache.get(Path("/bin/llama-server"))
        cache.clear()
        assert cache.get(Path("/bin/llama-server")) == [device]

    def test_success_after_expired_failure_clears_it(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        outcomes: list[object] = [ProviderError("probe wedged"), [device]]

        def _next(_binary: Path) -> list[FleetDevice]:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(planning_mod, "resolve_devices", _next)
        cache = planning_mod._ReadDeviceCache(ttl_s=0.0, failure_ttl_s=0.0)
        with pytest.raises(ProviderError):
            cache.get(Path("/bin/llama-server"))
        assert cache.get(Path("/bin/llama-server")) == [device]


def _parse_flags(argv: list[str]) -> dict[str, str | None]:
    """Map each ``--flag`` in *argv* to its value token (None for bare flags)."""
    flags: dict[str, str | None] = {}
    for position, token in enumerate(argv):
        if not token.startswith("--"):
            continue
        following = argv[position + 1] if position + 1 < len(argv) else None
        flags[token] = None if following is None or following.startswith("--") else following
    return flags


# Launch flags with no memory-sizing effect; everything else must map below.
_NON_SIZING_LAUNCH_FLAGS = {
    "--model",
    "--host",
    "--cont-batching",
    "--jinja",
    "--no-mmap",
    "--no-prefill-assistant",
    "--reasoning-format",
    "--embeddings",
    "--pooling",
}
# Sizing-relevant launch flag -> the gguf-parser flag that must carry the same value.
_SIZING_FLAG_TO_ESTIMATOR_FLAG = {
    "--ctx-size": "--ctx-size",
    "--parallel": "--parallel",
    "--n-gpu-layers": "--gpu-layers",
    "--batch-size": "--batch-size",
    "--ubatch-size": "--ubatch-size",
    "--tensor-split": "--tensor-split",
    "--cache-type-k": "--cache-type-k",
    "--cache-type-v": "--cache-type-v",
    "--mmproj": "--mmproj-path",
}
_FLASH_LAUNCH_FLAG = "--flash-attn"


class TestEstimateLaunchParity:
    """Every sizing-relevant flag the launch argv carries must be reflected in the
    gguf-parser argv the placement estimate ran with, or the estimate diverges
    from what the server allocates (the embed/rerank ubatch OOM)."""

    @pytest.mark.parametrize(
        "role",
        [WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION],
    )
    @pytest.mark.parametrize("backend", ["CUDA", "Vulkan"])
    def test_launch_sizing_flags_reflected_in_estimator_argv(
        self, role: WorkerRole, backend: str, tmp_path, monkeypatch
    ) -> None:
        from lilbee.providers.fleet import vram as vram_mod

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 64)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "bert"}
        )
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        # Vulkan leaves V unquantized, so K and V differ; the estimate has to
        # carry that difference or it sizes a cache the launch will not allocate.
        monkeypatch.setattr(planning_mod, "_fleet_backend", lambda: backend)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        mmproj = Path("/m/mmproj.gguf") if role is WorkerRole.VISION else None
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: mmproj)
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        captured: dict[str, object] = {}

        def _capture(path, **kwargs) -> GgufVramEstimate:
            captured.update(kwargs, path=path)
            return GgufVramEstimate(
                vram_bytes=1, ram_bytes=0, unified_bytes=1, per_device_vram=(1, 1)
            )

        monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _capture)
        is_chat = role is WorkerRole.CHAT
        ratio = (1, 1) if is_chat else ()
        planning_mod._peak_estimator({role: "ref"})(role, ratio)
        devices = (0, 1) if is_chat else (0,)
        by_index = {
            0: FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB),
            1: FleetDevice("CUDA", 1, "gpu", 24 * _GB, 23 * _GB),
        }
        plan = InstancePlan(role=role, devices=devices, tensor_split=ratio)
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), by_index)
        estimator_argv = vram_mod.estimator_argv(
            str(captured["path"]),
            ctx=captured["ctx"],  # type: ignore[arg-type]
            slots=captured["slots"],  # type: ignore[arg-type]
            gpu_layers=captured["gpu_layers"],  # type: ignore[arg-type]
            flash_attn=captured["flash_attn"],  # type: ignore[arg-type]
            kv_cache_type=captured["kv_cache_type"].value,  # type: ignore[union-attr]
            kv_cache_type_v=captured["kv_cache_type_v"].value,  # type: ignore[union-attr]
            mmproj=str(captured["mmproj_path"]) if captured.get("mmproj_path") else None,
            tensor_split=captured.get("tensor_split", ()),  # type: ignore[arg-type]
            batch_size=captured.get("batch_size"),  # type: ignore[arg-type]
        )
        launch_flags = _parse_flags(launch.argv)
        estimator_flags = _parse_flags(estimator_argv)
        for flag, value in launch_flags.items():
            if flag in _NON_SIZING_LAUNCH_FLAGS:
                continue
            if flag == _FLASH_LAUNCH_FLAG:
                expected = "--flash-attention" if value == "on" else "--no-flash-attention"
                assert expected in estimator_argv, f"{role}: flash mismatch"
                continue
            estimator_flag = _SIZING_FLAG_TO_ESTIMATOR_FLAG.get(flag)
            assert estimator_flag is not None, (
                f"{role}: launch flag {flag} is unclassified; add it to the sizing map "
                "(and thread it into the gguf-parser estimate) or the non-sizing set"
            )
            assert estimator_flags.get(estimator_flag) == value, (
                f"{role}: launch {flag}={value} not reflected as estimator "
                f"{estimator_flag}={estimator_flags.get(estimator_flag)}"
            )
        # The loop above walks launch flags, so it cannot see an estimator flag
        # with no launch counterpart. That direction matters for the KV types:
        # an estimator sizing a q8_0 V cache the launch leaves at f16 reserves
        # less memory than the server will allocate.
        for kv_flag in ("--cache-type-k", "--cache-type-v"):
            assert estimator_flags.get(kv_flag, "f16") == launch_flags.get(kv_flag, "f16"), (
                f"{role}/{backend}: estimator {kv_flag}={estimator_flags.get(kv_flag)} "
                f"but launch {kv_flag}={launch_flags.get(kv_flag)}"
            )

    def test_multi_slot_context_is_charged_once_on_both_sides(self, tmp_path, monkeypatch) -> None:
        """A single-slot fleet cannot tell the two conventions apart.

        With one slot the per-slot context and the total are the same number, so
        the comparison above holds whichever side does the multiply. Run a role
        that really batches, where a missing multiply reserves an eighth of the
        KV the server allocates.
        """
        from lilbee.providers.fleet import vram as vram_mod
        from lilbee.providers.fleet.adapters import LLM_RERANK_CONCURRENCY

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 64)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "qwen3"}
        )
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.LLM)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
        captured: dict[str, object] = {}

        def _capture(path, **kwargs) -> GgufVramEstimate:
            captured.update(kwargs, path=path)
            return GgufVramEstimate(vram_bytes=1, ram_bytes=0, unified_bytes=1)

        monkeypatch.setattr(planning_mod, "estimate_instance_footprint", _capture)
        device = FleetDevice("CUDA", 0, "gpu", 80 * _GB, 80 * _GB)
        plan = InstancePlan(role=WorkerRole.RERANK, devices=(0,))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})

        assert launch.slots == LLM_RERANK_CONCURRENCY > 1, "need a batching role to tell them apart"
        estimator_argv = vram_mod.estimator_argv(
            str(captured["path"]),
            ctx=captured["ctx"],  # type: ignore[arg-type]
            slots=captured["slots"],  # type: ignore[arg-type]
            gpu_layers=captured["gpu_layers"],  # type: ignore[arg-type]
            flash_attn=captured["flash_attn"],  # type: ignore[arg-type]
            kv_cache_type=captured["kv_cache_type"].value,  # type: ignore[union-attr]
            kv_cache_type_v=captured["kv_cache_type_v"].value,  # type: ignore[union-attr]
            mmproj=None,
            tensor_split=(),
            batch_size=captured.get("batch_size"),  # type: ignore[arg-type]
        )
        launch_flags = _parse_flags(launch.argv)
        estimator_flags = _parse_flags(estimator_argv)
        assert launch_flags["--ctx-size"] == str(4096 * LLM_RERANK_CONCURRENCY)
        assert estimator_flags["--ctx-size"] == launch_flags["--ctx-size"]
        assert estimator_flags["--parallel"] == launch_flags["--parallel"]


class TestWeightsBytes:
    """The cold-load timeout scales with the model's total on-disk weights."""

    def test_single_file_uses_its_own_size(self, tmp_path) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 100)
        assert planning_mod._weights_bytes(model) == 100

    def test_split_gguf_sums_all_sibling_shards(self, tmp_path) -> None:
        first = tmp_path / "big-00001-of-00003.gguf"
        first.write_bytes(b"x" * 100)
        (tmp_path / "big-00002-of-00003.gguf").write_bytes(b"x" * 200)
        (tmp_path / "big-00003-of-00003.gguf").write_bytes(b"x" * 300)
        (tmp_path / "other-00001-of-00002.gguf").write_bytes(b"x" * 999)  # different model
        (tmp_path / "big.gguf").write_bytes(b"x" * 50)  # not a shard
        assert planning_mod._weights_bytes(first) == 600

    def test_launch_for_split_model_sums_shards(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "chat-00001-of-00002.gguf"
        model.write_bytes(b"x" * 1024)
        (tmp_path / "chat-00002-of-00002.gguf").write_bytes(b"x" * 1024)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _ref: model,
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: 4096)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})
        assert launch.weights_bytes == 2048  # both shards, not just the served file


def test_server_spec_embed_decoder_forces_last_pooling() -> None:
    spec = planning_mod._server_spec(WorkerRole.EMBED, None, {"architecture": "qwen3"})
    assert spec.extra_args == ("--embeddings", "--pooling", "last")


def test_server_spec_embed_honors_declared_pooling() -> None:
    spec = planning_mod._server_spec(
        WorkerRole.EMBED, None, {"architecture": "qwen3", "pooling_type": "1"}
    )
    assert spec.extra_args == ("--embeddings", "--pooling", "mean")


def test_server_spec_embed_without_meta_is_plain() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS

    spec = planning_mod._server_spec(WorkerRole.EMBED, None, None)
    assert spec is ROLE_SPECS[WorkerRole.EMBED]


def test_server_spec_rerank_uses_rerank_spec() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS

    spec = planning_mod._server_spec(WorkerRole.RERANK, RerankMode.CROSS_ENCODER, None)
    assert spec is ROLE_SPECS[WorkerRole.RERANK]


def test_server_spec_other_role_uses_role_default() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS

    spec = planning_mod._server_spec(WorkerRole.CHAT, None, None)
    assert spec is ROLE_SPECS[WorkerRole.CHAT]


class TestChatNoMmap:
    def test_local_disk_always_mmaps(self, monkeypatch) -> None:
        # Local disk keeps mmap regardless of size: --no-mmap's buffered read only
        # wins on an already-hot cache and pessimizes the cold-start first token.
        monkeypatch.setattr(
            "lilbee.providers.model_cache.total_system_memory", lambda: 1000 * 10**9
        )
        assert planning_mod._chat_no_mmap(112 * 10**9) is False
        assert planning_mod._chat_no_mmap(20 * 10**9) is False

    def test_network_fs_uses_no_mmap_only_when_host_copy_fits(self, monkeypatch) -> None:
        # A network filesystem prefers a buffered read (mmap page faults over the
        # wire can wedge the loader), but only when the malloc'd copy fits the RAM
        # fraction; an oversized model still mmaps. The same model on local disk
        # keeps mmap either way.
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 100 * 10**9)
        assert planning_mod._chat_no_mmap(70 * 10**9, on_network_fs=True) is True
        assert planning_mod._chat_no_mmap(90 * 10**9, on_network_fs=True) is False
        assert planning_mod._chat_no_mmap(70 * 10**9) is False


class TestPlanProbe:
    """The clean-box plan snapshot: reloads size against it, never a live probe."""

    @pytest.fixture(autouse=True)
    def _reset_probe(self):
        planning_mod.clear_plan_probe()
        yield
        planning_mod.clear_plan_probe()

    @staticmethod
    def _live_card(monkeypatch, total: int, free: int) -> None:
        """What a probe run right now would report, snapshot or no snapshot."""
        card = FleetDevice("CUDA", 0, "A", total, free)
        monkeypatch.setattr(planning_mod._read_device_cache, "get", lambda _b: [card])
        monkeypatch.setattr(planning_mod, "resolve_devices", lambda _b: [card])

    def _capture(self, monkeypatch, *, total_vram: int, free_vram: int, free_ram: int) -> None:
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda: None
        )
        # The capture takes devices and the all-refused fact from one probe run.
        monkeypatch.setattr(
            planning_mod,
            "_resolve_devices_and_refusal",
            lambda _b: ([FleetDevice("CUDA", 0, "A", total_vram, free_vram)], False),
        )
        self._live_card(monkeypatch, total_vram, free_vram)
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: free_ram)
        planning_mod.capture_plan_probe()

    def test_readers_serve_the_snapshot_not_the_live_state(self, monkeypatch) -> None:
        self._capture(monkeypatch, total_vram=24 * _GB, free_vram=20 * _GB, free_ram=64 * _GB)
        # The box "fills up" (a loaded fleet) and the card it reports shrinks...
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1 * _GB)
        self._live_card(monkeypatch, 8 * _GB, 1 * _GB)
        # ...but the plan paths keep the clean-box numbers.
        assert planning_mod.plan_sizing_budget() == int(24 * _GB * cfg.gpu_memory_fraction)
        assert planning_mod._plan_free_system_memory() == 64 * _GB
        assert planning_mod._plan_devices(Path("/bin/srv"))[0].free_bytes == 20 * _GB

    def test_clear_returns_readers_to_live_state(self, monkeypatch) -> None:
        self._capture(monkeypatch, total_vram=24 * _GB, free_vram=20 * _GB, free_ram=64 * _GB)
        planning_mod.clear_plan_probe()
        self._live_card(monkeypatch, 8 * _GB, 3 * _GB)
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 2 * _GB)
        assert planning_mod.plan_sizing_budget() == int(8 * _GB * cfg.gpu_memory_fraction)
        assert planning_mod._plan_free_system_memory() == 2 * _GB

    def test_uncaptured_readers_pass_through_live_state(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        self._live_card(monkeypatch, 8 * _GB, 7 * _GB)
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 5 * _GB)
        assert planning_mod.plan_sizing_budget() == int(8 * _GB * cfg.gpu_memory_fraction)
        assert planning_mod._plan_free_system_memory() == 5 * _GB

    def test_an_unreadable_probe_sizes_against_system_memory(self, monkeypatch) -> None:
        # No snapshot and no engine to ask: the fleet runs on the CPU, where system
        # memory is the honest budget rather than a stale guess at a card.
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))

        def _no_engine(_binary):
            raise OSError("no engine")

        monkeypatch.setattr(planning_mod._read_device_cache, "get", _no_engine)
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 32 * _GB)
        assert planning_mod.plan_sizing_budget() == int(32 * _GB * cfg.gpu_memory_fraction)


class TestPlacementChargesAgainstFreeMemory:
    """VRAM another process is holding is not headroom the fleet can plan into."""

    @pytest.fixture(autouse=True)
    def _reset_probe(self):
        planning_mod.clear_plan_probe()
        yield
        planning_mod.clear_plan_probe()

    @staticmethod
    def _snapshot(monkeypatch, devices: list[FleetDevice]) -> None:
        """Capture the clean-box snapshot, where free bytes exclude only other tenants."""
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda: None
        )
        monkeypatch.setattr(
            planning_mod, "_resolve_devices_and_refusal", lambda _b: (devices, False)
        )
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 64 * _GB)
        planning_mod.capture_plan_probe()

    def test_a_tenant_holding_vram_is_not_planned_over(self, monkeypatch) -> None:
        # A desktop compositor and a browser hold 20 of the card's 24 GiB. Charged
        # against total capacity the card reads as empty, so a 10 GiB model is
        # planned onto 4 GiB of real headroom and OOMs at load.
        held = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 4 * _GB)
        self._snapshot(monkeypatch, [held])
        placement = planning_mod._resolve_placement(
            None,
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            {WorkerRole.CHAT: "ref"},
            [held],
            unified_budget=None,
            charge_against_free=True,
        )
        assert WorkerRole.CHAT in placement.tight_roles

    def test_an_empty_card_is_still_charged_at_its_usable_capacity(self, monkeypatch) -> None:
        idle = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 24 * _GB)
        self._snapshot(monkeypatch, [idle])
        placement = planning_mod._resolve_placement(
            None,
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            {WorkerRole.CHAT: "ref"},
            [idle],
            unified_budget=None,
            charge_against_free=True,
        )
        assert placement.tight_roles == {}
        assert placement.instances == (InstancePlan(WorkerRole.CHAT, (0,)),)

    def test_a_live_probe_keeps_charging_total_capacity(self, monkeypatch) -> None:
        """Without the clean-box snapshot, free bytes include the fleet's own models.

        The placement view re-resolves on a warm box, where the fleet's own
        residency is exactly what free memory is missing. Charging it there would
        report the running plan as unplaceable.
        """
        warm = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 1 * _GB)
        placement = planning_mod._resolve_placement(
            None,
            [ModelPlacementInput(WorkerRole.CHAT, 10 * _GB)],
            {WorkerRole.CHAT: "ref"},
            [warm],
            unified_budget=None,
            charge_against_free=False,
        )
        assert placement.tight_roles == {}


class TestSlotsAreChargedOnce:
    """The estimator is given a per-slot context and does the multiply itself.

    llama-server divides --ctx-size across --parallel slots, and the estimator
    ignores --parallel entirely, so whoever builds its command line has to carry
    the total. Leaving that to each caller had two of four sites passing the
    per-slot figure, which under-reserves KV by the whole slot count.
    """

    def test_estimator_argv_carries_the_total_context(self) -> None:
        from lilbee.providers.fleet import vram as vram_mod

        argv = vram_mod.estimator_argv(
            "/m/m.gguf",
            ctx=4096,
            slots=4,
            gpu_layers=-1,
            flash_attn=True,
            kv_cache_type="q8_0",
            kv_cache_type_v="q8_0",
            mmproj=None,
            tensor_split=(),
            batch_size=None,
        )
        assert argv[argv.index("--ctx-size") + 1] == "16384"
        assert argv[argv.index("--parallel") + 1] == "4"

    def test_the_slot_fit_can_step_down(self, tmp_path, monkeypatch) -> None:
        # Charged per slot, every probe in the descending search costs the same,
        # so the search can only ever answer its ceiling or 1.
        import json

        from lilbee.providers.fleet import vram as vram_mod

        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF")
        vram_mod._cached_footprint.cache_clear()
        monkeypatch.setattr(
            planning_mod, "estimate_instance_footprint", vram_mod.estimate_instance_footprint
        )
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))

        def _priced_by_context(argv: list[str], _path: str) -> str:
            total = 10**9 + int(argv[argv.index("--ctx-size") + 1]) * 10**5
            return json.dumps(
                {
                    "estimate": {
                        "items": [
                            {"ram": {"uma": 0, "nonuma": 0}, "vrams": [{"uma": 0, "nonuma": total}]}
                        ]
                    }
                }
            )

        monkeypatch.setattr(vram_mod, "_run_parser", _priced_by_context)
        # 1e9 of weights plus 1e5 per token: two slots of 4096 fit, three do not.
        assert (
            planning_mod._fit_slots(
                4,
                WorkerRole.CHAT,
                model,
                4096,
                mmproj_path=None,
                unified=False,
                budget=10**9 + 8192 * 10**5,
            )
            == 2
        )


class TestSizingBudgetComesFromTheDevice:
    """ctx and slots are sized against the GPU that will run the model."""

    @pytest.fixture(autouse=True)
    def _reset_probe(self):
        planning_mod.clear_plan_probe()
        yield
        planning_mod.clear_plan_probe()

    @staticmethod
    def _capture(monkeypatch, devices: list[FleetDevice], *, host_ram: int) -> None:
        """Snapshot *devices* on a host with *host_ram* bytes of system RAM."""
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda: None
        )
        monkeypatch.setattr(
            planning_mod, "_resolve_devices_and_refusal", lambda _b: (devices, False)
        )
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: host_ram)
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: host_ram)

        # Nothing here may fall back to the host-memory read; reaching it is the
        # defect these tests exist to catch, so make it loud rather than plausible.
        def _host_read(*_a, **_k):
            raise AssertionError("sizing budget fell back to the host-memory read")

        monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", _host_read)
        planning_mod.capture_plan_probe()

    def test_discrete_card_is_not_sized_against_host_ram(self, monkeypatch) -> None:
        # A 24 GiB AMD card in a 128 GiB host. There is no AMD VRAM query, so the
        # host-RAM read hands the planner a budget four times the card.
        self._capture(
            monkeypatch,
            [FleetDevice("ROCm", 0, "gfx1100", 24 * _GB, 24 * _GB)],
            host_ram=128 * _GB,
        )
        assert planning_mod.plan_sizing_budget() == int(24 * _GB * cfg.gpu_memory_fraction)

    def test_unified_device_is_sized_against_its_own_working_set(self, monkeypatch) -> None:
        # Metal reports recommendedMaxWorkingSetSize, which is below installed RAM.
        self._capture(
            monkeypatch,
            [FleetDevice("Metal", 0, "M3 Max", 48 * _GB, 48 * _GB, unified=True)],
            host_ram=64 * _GB,
        )
        assert planning_mod.plan_sizing_budget() == int(48 * _GB * cfg.gpu_memory_fraction)

    def test_host_ram_still_sizes_a_gpu_less_host(self, monkeypatch) -> None:
        self._capture(monkeypatch, [], host_ram=32 * _GB)
        assert planning_mod.plan_sizing_budget() == int(32 * _GB * cfg.gpu_memory_fraction)

    def test_launch_sizes_ctx_against_the_placed_card(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        small = FleetDevice("CUDA", 0, "3090", 24 * _GB, 24 * _GB)
        large = FleetDevice("CUDA", 1, "A6000", 48 * _GB, 48 * _GB)
        self._capture(monkeypatch, [small, large], host_ram=128 * _GB)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(cfg, "num_ctx", None)
        seen: list[int] = []

        def _record_ctx(_path, _meta, *, available_bytes=None):
            seen.append(available_bytes)
            return 4096

        monkeypatch.setattr("lilbee.providers.engine_params.resolve_chat_ctx", _record_ctx)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(1,))
        planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: small, 1: large})
        assert seen == [int(48 * _GB * cfg.gpu_memory_fraction)]

    def test_unified_budgets_cap_at_the_devices_own_memory(self, monkeypatch) -> None:
        # An iGPU addressing 8 GiB of a 64 GiB host cannot host 60 GiB of models
        # just because the host has the RAM.
        igpu = FleetDevice("Vulkan", 0, "Radeon 780M", 8 * _GB, 8 * _GB, unified=True)
        self._capture(monkeypatch, [igpu], host_ram=64 * _GB)
        assert planning_mod._unified_memory_budget([igpu]) == 8 * _GB
        assert planning_mod._unified_admission_budget([igpu]) == 8 * _GB


class TestPlacementFindingsLog:
    """plan_launches tells the user about tight and unservable placements."""

    def test_tight_role_logs_a_memory_is_tight_warning(self, caplog) -> None:
        import logging

        placement = Placement(
            instances=(InstancePlan(role=WorkerRole.VISION, devices=(0,)),),
            unplaceable_roles=(),
            tight_roles={WorkerRole.VISION: int(4.1 * _GB)},
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod._log_placement_findings(placement, {WorkerRole.VISION: "org/ocr.gguf"})
        assert "Memory is tight for the vision model org/ocr.gguf" in caplog.text
        assert "4.1 GiB" in caplog.text
        assert "will still load on demand" in caplog.text

    def test_unplaceable_shared_memory_role_still_warns_it_gets_no_server(self, caplog) -> None:
        import logging

        placement = Placement(
            instances=(),
            unplaceable_roles=(WorkerRole.CHAT,),
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod._log_placement_findings(placement, {WorkerRole.CHAT: "org/chat.gguf"})
        assert "will not be served" in caplog.text

    def test_tiny_shortfall_never_reads_as_zero(self, caplog) -> None:
        import logging

        placement = Placement(
            instances=(InstancePlan(role=WorkerRole.VISION, devices=(0,)),),
            unplaceable_roles=(),
            tight_roles={WorkerRole.VISION: 10 * 1024 * 1024},
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod._log_placement_findings(placement, {WorkerRole.VISION: "org/ocr.gguf"})
        assert "0.0 GiB" not in caplog.text
        assert "0.1 GiB" in caplog.text


class TestSizingFailureFallsBackToFileSize:
    """A model the estimator cannot size is enrolled at its weight bytes, so the
    load, not the estimator, decides. Weight bytes are also a physics bound:
    weights alone exceeding total VRAM refuses with a clear message."""

    @pytest.fixture
    def _sizing_boom(self, tmp_path, monkeypatch):
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        model = tmp_path / "unsizable.gguf"
        model.write_bytes(b"G" * 4096)

        def boom(role, ref, **_k):
            if role is WorkerRole.CHAT:
                raise ProviderError(
                    "unexpected estimator output",
                    provider="llama-server",
                    kind=ProviderErrorKind.SERVER,
                )
            return ModelPlacementInput(role, 512)

        monkeypatch.setattr(planning_mod, "_estimate_role", boom)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/unsizable.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        return model

    def test_unsizable_model_enrolls_at_its_file_size(self, _sizing_boom, caplog) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            inputs, _refs, _res, _skipped = planning_mod._server_model_inputs(total_vram=24 * _GB)
        by_role = {i.role: i for i in inputs}
        assert by_role[WorkerRole.CHAT].est_vram_bytes == 4096
        assert "Using its file size" in caplog.text

    def test_weights_beyond_total_vram_refuse_with_a_clear_message(
        self, _sizing_boom, caplog
    ) -> None:
        import logging

        with caplog.at_level(logging.WARNING):
            inputs, _refs, _res, _skipped = planning_mod._server_model_inputs(total_vram=1024)
        assert WorkerRole.CHAT not in {i.role for i in inputs}
        assert "weights alone" in caplog.text

    def test_weights_bound_stands_down_under_partial_offload(
        self, _sizing_boom, monkeypatch, caplog
    ) -> None:
        import logging

        monkeypatch.setattr(cfg, "n_gpu_layers", 10)
        with caplog.at_level(logging.WARNING):
            inputs, _refs, _res, _skipped = planning_mod._server_model_inputs(total_vram=1024)
        assert WorkerRole.CHAT in {i.role for i in inputs}
        assert "weights alone" not in caplog.text

    def test_unresolvable_file_still_skips(self, tmp_path, monkeypatch, caplog) -> None:
        import logging

        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def boom(role, ref, **_k):
            raise ProviderError(
                "unexpected estimator output",
                provider="llama-server",
                kind=ProviderErrorKind.SERVER,
            )

        def no_path(_r):
            raise ProviderError(
                "no file", provider="llama-server", kind=ProviderErrorKind.NOT_FOUND
            )

        monkeypatch.setattr(planning_mod, "_estimate_role", boom)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", no_path)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/ghost.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        with caplog.at_level(logging.WARNING):
            inputs, _refs, _res, _skipped = planning_mod._server_model_inputs()
        assert inputs == []
        assert "could not size" in caplog.text

    def test_unsizable_vision_model_counts_its_mmproj(self, tmp_path, monkeypatch, caplog) -> None:
        import logging

        from lilbee.providers.base import ProviderError, ProviderErrorKind

        model = tmp_path / "vl.gguf"
        model.write_bytes(b"G" * 4096)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"G" * 1024)

        def boom(role, ref, **_k):
            if role is WorkerRole.VISION:
                raise ProviderError(
                    "unexpected estimator output",
                    provider="llama-server",
                    kind=ProviderErrorKind.SERVER,
                )
            return ModelPlacementInput(role, 512)

        monkeypatch.setattr(planning_mod, "_estimate_role", boom)
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: mmproj)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
        monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "")
        monkeypatch.setattr(cfg, "vision_model", "org/repo/vl.gguf")
        with caplog.at_level(logging.WARNING):
            inputs, _refs, _res, _skipped = planning_mod._server_model_inputs(total_vram=24 * _GB)
        by_role = {i.role: i for i in inputs}
        assert by_role[WorkerRole.VISION].est_vram_bytes == 4096 + 1024

    def test_fit_slots_returns_single_slot_when_estimator_fails(
        self, tmp_path, monkeypatch
    ) -> None:
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def boom(*_a, **_k):
            raise ProviderError(
                "unparseable", provider="llama-server", kind=ProviderErrorKind.SERVER
            )

        monkeypatch.setattr(planning_mod, "estimate_instance_footprint", boom)
        slots = planning_mod._fit_slots(
            4,
            WorkerRole.CHAT,
            tmp_path / "m.gguf",
            2048,
            mmproj_path=None,
            unified=False,
            budget=8 * _GB,
        )
        assert slots == 1


def test_expert_offload_is_ignored_on_a_dense_model(monkeypatch) -> None:
    # No expert tensors to move, so the flag would be a silent no-op.
    monkeypatch.setattr(cfg, "cpu_moe", True)
    assert planning_mod.expert_offload_all({"architecture": "qwen3"}) is False


def test_expert_offload_applies_to_a_sparse_model(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "cpu_moe", True)
    assert planning_mod.expert_offload_all({"expert_count": "128"}) is True


def test_expert_offload_layer_count_applies_to_a_sparse_model(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "n_cpu_moe", 16)
    assert planning_mod.expert_offload_layers({"expert_count": "128"}) == 16


def test_expert_offload_survives_unparsable_expert_count(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "cpu_moe", True)
    assert planning_mod.expert_offload_all({"expert_count": "many"}) is False


def test_expert_offload_lets_a_sparse_model_bigger_than_vram_through(monkeypatch) -> None:
    # Oversize weights are legitimate once a sparse model's experts live in RAM.
    monkeypatch.setattr(cfg, "cpu_moe", True)
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    assert planning_mod._weights_exceed_hardware(80 * 1024**3, 24 * 1024**3, is_moe=True) is False


def test_expert_offload_still_refuses_a_dense_model_over_vram(monkeypatch) -> None:
    # A dense model gains no offload flags, so its weights really must fit: keep
    # the guided refusal instead of a raw load-time OOM.
    monkeypatch.setattr(cfg, "cpu_moe", True)
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    assert planning_mod._weights_exceed_hardware(80 * 1024**3, 24 * 1024**3, is_moe=False) is True


def test_oversize_model_is_still_refused_without_expert_offload(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "cpu_moe", False)
    monkeypatch.setattr(cfg, "n_cpu_moe", None)
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    assert planning_mod._weights_exceed_hardware(80 * 1024**3, 24 * 1024**3, is_moe=True) is True


def test_zero_n_cpu_moe_is_not_effective_offload(monkeypatch) -> None:
    # n_cpu_moe <= 0 would emit a no-op --n-cpu-moe 0; treat it as no offload, so
    # a dense OR sparse model over VRAM is still refused rather than silently OOM.
    monkeypatch.setattr(cfg, "cpu_moe", False)
    monkeypatch.setattr(cfg, "n_cpu_moe", 0)
    monkeypatch.setattr(cfg, "n_gpu_layers", None)
    assert planning_mod._expert_offload_configured() is False
    assert planning_mod._weights_exceed_hardware(80 * 1024**3, 24 * 1024**3, is_moe=True) is True
    assert planning_mod.expert_offload_layers({"expert_count": "128"}) is None


def test_oversize_sparse_model_is_told_to_offload_its_experts(monkeypatch, caplog) -> None:
    monkeypatch.setattr(planning_mod, "_ref_is_moe", lambda _ref: True)
    with caplog.at_level("WARNING"):
        planning_mod._warn_weights_exceed(WorkerRole.CHAT, "m/moe", 80 * 1024**3, 24 * 1024**3)
    assert "cpu_moe" in caplog.text
    assert "n_gpu_layers" not in caplog.text


def test_oversize_dense_model_is_still_told_to_cut_gpu_layers(monkeypatch, caplog) -> None:
    monkeypatch.setattr(planning_mod, "_ref_is_moe", lambda _ref: False)
    with caplog.at_level("WARNING"):
        planning_mod._warn_weights_exceed(WorkerRole.CHAT, "m/dense", 80 * 1024**3, 24 * 1024**3)
    assert "n_gpu_layers" in caplog.text
    assert "cpu_moe" not in caplog.text


def test_ref_is_moe_reads_the_model_metadata(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "_is_moe", lambda _meta: True)
    monkeypatch.setattr(
        "lilbee.providers.engine_params.resolve_model_path", lambda _ref: Path("/m/moe.gguf")
    )
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    assert planning_mod._ref_is_moe("m/moe") is True


def test_ref_is_moe_is_false_for_an_unresolvable_model(monkeypatch) -> None:
    # The caller is already reporting a failure; a metadata read must not raise
    # over the top of it.
    def _boom(_ref):
        raise OSError("gone")

    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _boom)
    assert planning_mod._ref_is_moe("m/missing") is False


def test_estimator_argv_has_no_override_tensor_without_offload(monkeypatch) -> None:
    from lilbee.providers.fleet import vram as vram_mod

    # Resolve no real binary: this pins argv construction, and CI runners have no
    # bundled gguf-parser (resolve would raise and fail the test environmentally).
    monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
    argv = vram_mod.estimator_argv(
        "/m/m.gguf",
        ctx=4096,
        slots=1,
        gpu_layers=-1,
        flash_attn=True,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        mmproj=None,
        tensor_split=(),
        batch_size=None,
    )
    assert "--override-tensor" not in argv


def test_estimator_argv_charges_offloaded_experts_to_cpu(monkeypatch) -> None:
    # Without this the estimate charges the GPU for experts the launch keeps in
    # system memory, and the planner sizes slots against a footprint that never exists.
    from lilbee.providers.fleet import vram as vram_mod

    # Resolve no real binary: this pins argv construction, and CI runners have no
    # bundled gguf-parser (resolve would raise and fail the test environmentally).
    monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))
    argv = vram_mod.estimator_argv(
        "/m/m.gguf",
        ctx=4096,
        slots=1,
        gpu_layers=-1,
        flash_attn=True,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        mmproj=None,
        tensor_split=(),
        batch_size=None,
        expert_offload=(r"blk\.0\.ffn_x", r"blk\.1\.ffn_x"),
    )
    value = argv[argv.index("--override-tensor") + 1]
    assert value == r"blk\.0\.ffn_x=CPU,blk\.1\.ffn_x=CPU"


def test_estimate_offload_matches_what_the_launch_offloads(monkeypatch, tmp_path: Path) -> None:
    # The estimate and the launch must move the same tensors or the planner's
    # budget describes a configuration that never runs.
    model = tmp_path / "moe.gguf"
    model.write_bytes(b"x" * 64)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"expert_count": "128"}
    )
    monkeypatch.setattr(cfg, "cpu_moe", True)
    monkeypatch.setattr(cfg, "n_cpu_moe", None)
    from lilbee.providers.fleet.adapters import expert_offload_patterns

    launched = expert_offload_patterns(cpu_moe=True, n_cpu_moe=None)
    assert planning_mod._role_expert_offload(model) == launched


def test_estimate_offloads_nothing_for_a_dense_model(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "dense.gguf"
    model.write_bytes(b"x" * 64)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {"architecture": "qwen3"}
    )
    monkeypatch.setattr(cfg, "cpu_moe", True)
    assert planning_mod._role_expert_offload(model) == ()


def test_resolve_split_chat_slots_uses_all_when_every_window_fits() -> None:
    # fit_fn returns the full window at every slot count -> take the max.
    slots, ctx = planning_mod._resolve_split_chat_slots(lambda _n: 65536)
    assert slots == planning_mod._CHAT_SLOTS
    assert ctx == 65536


def test_resolve_split_chat_slots_falls_to_one_on_a_tight_split() -> None:
    # More slots shrink the per-slot window below the single-slot full window.
    def fit(n: int) -> int:
        return 65536 if n == 1 else 20000

    slots, ctx = planning_mod._resolve_split_chat_slots(fit)
    assert slots == 1
    assert ctx == 65536


def test_resolve_split_chat_slots_bounds_the_expensive_probes() -> None:
    """Every fit_fn call is a full search whose probes shell out to gguf-parser.

    They run while this process holds the cross-process build lock other lilbee
    starts wait on without a deadline, so the count must grow with the log of
    the slot ceiling, not with the ceiling itself. The tight split is the case
    that used to pay for every count.
    """
    import math

    probes: list[int] = []

    def fit(n: int) -> int:
        probes.append(n)
        return 65536 if n == 1 else 20000  # nothing above one slot fits

    slots, _ctx = planning_mod._resolve_split_chat_slots(fit)
    assert slots == 1
    budget = 1 + math.ceil(math.log2(planning_mod._CHAT_SLOTS))
    assert len(probes) <= budget, f"{len(probes)} searches, expected at most {budget}"


def test_resolve_split_chat_slots_picks_the_largest_fitting_count() -> None:
    # Two full windows fit but not three or four.
    def fit(n: int) -> int:
        return 40000 if n <= 2 else 12000

    slots, _ctx = planning_mod._resolve_split_chat_slots(fit)
    assert slots == 2


def test_resolve_split_chat_slots_stays_single_when_the_fit_is_the_floor() -> None:
    # A degenerate floor fit is a give-up value, not a verified fit; never multiply it.
    from lilbee.providers.model_cache import _DYNAMIC_CTX_FLOOR

    slots, ctx = planning_mod._resolve_split_chat_slots(lambda _n: _DYNAMIC_CTX_FLOOR)
    assert slots == 1
    assert ctx == _DYNAMIC_CTX_FLOOR


def test_role_model_placeable_true_for_installed_fitting_model(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "_vision_without_mmproj", lambda _r, _ref: False)
    monkeypatch.setattr(planning_mod, "_role_weights_bytes", lambda _r, _ref: 10 * 1024**3)
    monkeypatch.setattr(planning_mod, "_weights_exceed_hardware", lambda _w, _v, **_k: False)
    monkeypatch.setattr(planning_mod, "_ref_is_moe", lambda _ref: False)
    _ref = type("R", (), {"is_remote": False})()
    monkeypatch.setattr(planning_mod, "parse_model_ref", lambda _r: _ref)
    assert planning_mod.role_model_placeable(WorkerRole.CHAT, "org/repo/m.gguf", 80 * 1024**3)


def test_role_model_placeable_false_when_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "_vision_without_mmproj", lambda _r, _ref: False)
    monkeypatch.setattr(planning_mod, "_role_weights_bytes", lambda _r, _ref: 0)  # not installed
    _ref = type("R", (), {"is_remote": False})()
    monkeypatch.setattr(planning_mod, "parse_model_ref", lambda _r: _ref)
    assert not planning_mod.role_model_placeable(WorkerRole.EMBED, "org/repo/m.gguf", 80 * 1024**3)


def test_role_model_placeable_false_when_weights_exceed_vram(monkeypatch) -> None:
    monkeypatch.setattr(planning_mod, "_vision_without_mmproj", lambda _r, _ref: False)
    monkeypatch.setattr(planning_mod, "_role_weights_bytes", lambda _r, _ref: 200 * 1024**3)
    monkeypatch.setattr(planning_mod, "_weights_exceed_hardware", lambda w, v, **_k: w > v)
    monkeypatch.setattr(planning_mod, "_ref_is_moe", lambda _ref: False)
    _ref = type("R", (), {"is_remote": False})()
    monkeypatch.setattr(planning_mod, "parse_model_ref", lambda _r: _ref)
    assert not planning_mod.role_model_placeable(WorkerRole.CHAT, "org/repo/m.gguf", 80 * 1024**3)


def test_role_model_placeable_false_for_a_remote_ref(monkeypatch) -> None:
    # A remote (SDK-routed) ref is never placed on the local engine.
    _ref = type("R", (), {"is_remote": True})()
    monkeypatch.setattr(planning_mod, "parse_model_ref", lambda _r: _ref)
    assert not planning_mod.role_model_placeable(WorkerRole.CHAT, "openrouter/gpt", 80 * 1024**3)


def test_placeable_total_vram_reuses_the_captured_probe(monkeypatch) -> None:
    _dev = type("D", (), {"total_bytes": 40 * 1024**3})
    probe = type("P", (), {"devices": [_dev(), _dev()]})()
    monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: probe)
    assert planning_mod.placeable_total_vram() == 80 * 1024**3


def test_placeable_total_vram_zero_when_unprobeable(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError

    monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: None)
    monkeypatch.setattr(planning_mod, "apply_fleet_gpu_env", lambda: None, raising=False)

    def _boom(*_a, **_k):
        raise ProviderError("no binary", provider="llama-server")

    monkeypatch.setattr(planning_mod, "resolve_llama_server", _boom)
    assert planning_mod.placeable_total_vram() == 0


def test_assert_engine_probeable_probes_without_capturing_a_snapshot(monkeypatch) -> None:
    # The build precondition enumerates devices (surfacing a wedge) but must not
    # store a plan snapshot; that stays with the clean-box capture after the stop.
    called: list[bool] = []
    monkeypatch.setattr(
        planning_mod, "_probe_engine_devices", lambda: (called.append(True), ([], False))[1]
    )
    planning_mod._plan_probe_store.clear()
    planning_mod.assert_engine_probeable()
    assert called == [True]  # it enumerated devices...
    assert planning_mod._plan_probe_store.get() is None  # ...but took no snapshot


def test_integrated_gpu_keeps_the_shared_ram_budget(monkeypatch) -> None:
    """An iGPU's reported total is system RAM, not headroom on top of it."""
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 10 * 10**9)
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 15 * 10**9)
    igpu = FleetDevice("Vulkan", 0, "Iris Xe", 15 * 10**9, 15 * 10**9, unified=True)
    assert planning_mod._unified_memory_budget([igpu]) is not None


def test_apple_silicon_keeps_the_shared_ram_budget(monkeypatch) -> None:
    """Metal reports a working-set slice of system RAM, so the same holds."""
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 20 * 10**9)
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: 34 * 10**9)
    metal = FleetDevice("MTL", 0, "Apple M1 Pro", 22 * 10**9, 22 * 10**9, unified=True)
    assert planning_mod._unified_memory_budget([metal]) is not None


def test_a_discrete_card_beside_an_igpu_still_lifts_the_budget() -> None:
    """One device with memory of its own is enough; VRAM is the constraint then."""
    igpu = FleetDevice("Vulkan", 0, "Iris Xe", 15 * 10**9, 15 * 10**9, unified=True)
    dgpu = FleetDevice("Vulkan", 1, "RTX 4090", 24 * 10**9, 24 * 10**9)
    assert planning_mod._unified_memory_budget([igpu, dgpu]) is None


def test_metal_devices_are_recognised_as_unified(monkeypatch) -> None:
    """The Apple case is decided by backend, since the size ratio cannot decide it."""
    from lilbee.providers.fleet import gpu_select
    from lilbee.providers.fleet.devices import _is_unified

    # The CUDA leg asks the host's real Vulkan loader otherwise, so this passed
    # on macOS and CI (no loader) and failed on any Linux box with mesa.
    monkeypatch.setattr(gpu_select, "vulkan_device_types_by_name", dict)

    assert _is_unified("MTL", "Apple M3 Max") is True
    assert _is_unified("Metal", "Apple M3 Max") is True
    assert _is_unified("CUDA", "NVIDIA RTX A5000") is False


def test_engine_reporting_no_devices_is_believed_over_the_host_loader(monkeypatch) -> None:
    """A CPU-only build on a desktop with mesa must plan as CPU, not as a GPU box.

    --list-devices prints every non-CPU device the engine can use, so an empty
    list is a fact. Fabricating devices from the host loader plans a fleet onto
    GPUs the engine cannot see: the pins are no-ops, the shared-RAM guard is off
    because the list looked non-empty, and every role loads full weights into
    RAM while running on the CPU anyway.
    """
    from pathlib import Path as _Path

    from lilbee.providers.fleet import gpu_select

    probe = SimpleNamespace(
        devices=[], output="Available devices:\n", spoke_protocol=True, refused_all=False
    )
    monkeypatch.setattr(planning_mod, "probe_devices", lambda _b: probe)
    monkeypatch.setattr(planning_mod.model_cache, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
    )
    # The host loader can see a GPU; the engine still cannot use it.
    monkeypatch.setattr(gpu_select, "enumerate_gpu_vram", lambda: [(0, 8 * 10**9, 8 * 10**9)])

    assert planning_mod.resolve_devices(_Path("/fake/llama-server")) == []


def test_a_probe_that_could_not_run_still_falls_back(monkeypatch) -> None:
    """No output means no information, which is the one case worth guessing in."""
    from pathlib import Path as _Path

    from lilbee.providers.fleet import gpu_select

    probe = SimpleNamespace(devices=[], output="", spoke_protocol=False, refused_all=False)
    monkeypatch.setattr(planning_mod, "probe_devices", lambda _b: probe)
    monkeypatch.setattr(planning_mod.model_cache, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
    )
    monkeypatch.setattr(gpu_select, "enumerate_gpu_vram", lambda: [(0, 8 * 10**9, 8 * 10**9)])
    monkeypatch.setattr(gpu_select, "integrated_vulkan_indices", frozenset)

    devices = planning_mod.resolve_devices(_Path("/fake/llama-server"))
    assert [(d.backend, d.index) for d in devices] == [("Vulkan", 0)]


def test_an_engine_that_does_not_know_the_flag_still_reaches_the_fallback(monkeypatch) -> None:
    """A build predating --list-devices prints usage text and exits non-zero.

    The probe merges stderr into stdout, so gating the fallback on "no output"
    read that usage text as an authoritative empty device list and planned a GPU
    host as CPU-only, which is a regression against what these users had.
    """
    from pathlib import Path as _Path

    from lilbee.providers.fleet import gpu_select

    probe = SimpleNamespace(
        devices=[],
        output="error: invalid argument: --list-devices\n",
        spoke_protocol=False,
        refused_all=False,
    )
    monkeypatch.setattr(planning_mod, "probe_devices", lambda _b: probe)
    monkeypatch.setattr(planning_mod.model_cache, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
    )
    monkeypatch.setattr(gpu_select, "enumerate_gpu_vram", lambda: [(0, 8 * 10**9, 8 * 10**9)])
    monkeypatch.setattr(gpu_select, "integrated_vulkan_indices", frozenset)

    devices = planning_mod.resolve_devices(_Path("/fake/llama-server"))

    assert [(d.backend, d.index) for d in devices] == [("Vulkan", 0)]


def test_vulkan_launches_pin_by_name_not_by_raw_index() -> None:
    """--device takes the names --list-devices printed, the space we parsed from."""
    from lilbee.providers.fleet.planning import _device_names

    vulkan = (FleetDevice("Vulkan", 0, "", 0, 0), FleetDevice("Vulkan", 2, "", 0, 0))
    assert _device_names(vulkan) == ("Vulkan0", "Vulkan2")


def test_non_vulkan_backends_keep_pinning_through_env() -> None:
    """CUDA, ROCm and SYCL compose their variables in the probe's own space."""
    from lilbee.providers.fleet.planning import _device_names

    assert _device_names((FleetDevice("CUDA", 0, "", 0, 0),)) == ()
    assert _device_names((FleetDevice("ROCm", 0, "", 0, 0),)) == ()
    assert _device_names(()) == ()


class TestFlashAttentionIsBackendAware:
    """Forcing --flash-attn on took the decision away from the engine everywhere.

    llama.cpp's Vulkan flash-attn coverage lags CUDA's and has been incomplete on
    Intel's mesa driver, which is what a Tiger Lake laptop reported as a chat
    slowdown. But 'auto' cannot be passed alone: llama.cpp refuses a quantized V
    cache without flash attention and the server never starts.
    """

    def _on_backend(self, monkeypatch, backend: str | None) -> None:
        monkeypatch.setattr(planning_mod, "_fleet_backend", lambda: backend)
        monkeypatch.setattr(cfg, "flash_attention", None)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)

    @pytest.mark.parametrize("backend", ["CUDA", "ROCm", "HIP", "Metal", "MTL"])
    def test_backends_with_full_coverage_are_unchanged(self, monkeypatch, backend: str) -> None:
        """Every host that works today must keep the argv it has now."""
        self._on_backend(monkeypatch, backend)
        assert planning_mod.flash_attn_flag() == "on"
        assert planning_mod.chat_cache_type_flags() == ("q8_0", "q8_0")

    def test_an_unknown_backend_keeps_todays_behaviour(self, monkeypatch) -> None:
        self._on_backend(monkeypatch, None)
        assert planning_mod.flash_attn_flag() == "on"
        assert planning_mod.chat_cache_type_flags() == ("q8_0", "q8_0")

    @pytest.mark.parametrize("backend", ["Vulkan", "SYCL"])
    def test_lagging_backends_defer_to_the_engine_and_leave_v_unquantized(
        self, monkeypatch, backend: str
    ) -> None:
        """K quantization needs nothing; only V requires flash attention."""
        self._on_backend(monkeypatch, backend)
        assert planning_mod.flash_attn_flag() == "auto"
        assert planning_mod.chat_cache_type_flags() == ("q8_0", None)

    def test_explicit_off_no_longer_asks_for_an_impossible_v_cache(self, monkeypatch) -> None:
        """flash_attention=false with a quantized KV type asked llama-server for
        'V cache quantization requires flash_attn', so it never started."""
        self._on_backend(monkeypatch, "CUDA")
        monkeypatch.setattr(cfg, "flash_attention", False)
        assert planning_mod.flash_attn_flag() == "off"
        assert planning_mod.chat_cache_type_flags() == ("q8_0", None)

    def test_the_estimate_does_not_assume_flash_attention_under_auto(self, monkeypatch) -> None:
        """The engine decides at load; assuming it would size KV below what the
        launch may need."""
        self._on_backend(monkeypatch, "Vulkan")
        assert planning_mod._role_flash(WorkerRole.CHAT) is False
        self._on_backend(monkeypatch, "CUDA")
        assert planning_mod._role_flash(WorkerRole.CHAT) is True


def test_clearing_the_device_cache_also_clears_what_the_loader_said(monkeypatch) -> None:
    """The loader's answers are held for the process lifetime otherwise, so a
    driver reload or a newly plugged eGPU would never be noticed."""
    from lilbee.providers.fleet import gpu_select

    readings = iter(
        [
            [gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.INTEGRATED_GPU, "iGPU", 0, 0)],
            [gpu_select.VulkanDevice(0, gpu_select.VkDeviceType.DISCRETE_GPU, "eGPU", 0, 0)],
        ]
    )
    monkeypatch.setattr(gpu_select, "_enumerate_vulkan_devices", lambda: next(readings))
    gpu_select.vulkan_device_types_by_name.cache_clear()
    gpu_select.integrated_vulkan_indices.cache_clear()
    try:
        assert gpu_select.vulkan_device_types_by_name() == {
            "iGPU": gpu_select.VkDeviceType.INTEGRATED_GPU
        }

        planning_mod.clear_read_device_cache()

        assert gpu_select.vulkan_device_types_by_name() == {
            "eGPU": gpu_select.VkDeviceType.DISCRETE_GPU
        }
    finally:
        gpu_select.vulkan_device_types_by_name.cache_clear()
        gpu_select.integrated_vulkan_indices.cache_clear()


class TestLoaderDerivedDevicesAreSizedAgainstButNeverPinned:
    """The fallback holds raw loader ordinals; --device speaks the engine's own
    post-filter naming.

    Loader order [llvmpipe, iGPU] leaves the iGPU at raw index 1, so a pin would
    say Vulkan1 while a Vulkan-capable engine names its only device Vulkan0: an
    invalid-device error, or the wrong adapter where there are more.
    """

    def test_a_loader_derived_device_is_not_pinned(self) -> None:
        from lilbee.providers.fleet.planning import _device_names

        loader = (FleetDevice(VULKAN_BACKEND, 1, "", 8 * _GB, 8 * _GB, from_loader=True),)

        assert _device_names(loader) == ()

    def test_engine_parsed_devices_are_still_pinned_by_name(self) -> None:
        from lilbee.providers.fleet.planning import _device_names

        parsed = (
            FleetDevice(VULKAN_BACKEND, 0, "Card A", 8 * _GB, 8 * _GB),
            FleetDevice(VULKAN_BACKEND, 2, "Card C", 8 * _GB, 8 * _GB),
        )

        assert _device_names(parsed) == ("Vulkan0", "Vulkan2")

    def test_the_fallback_marks_what_it_builds(self, monkeypatch) -> None:
        """The flag has to be set where the devices are invented, or the rule
        above is decoration."""
        from pathlib import Path as _Path

        from lilbee.providers.fleet import gpu_select

        probe = SimpleNamespace(devices=[], output="", spoke_protocol=False, refused_all=False)
        monkeypatch.setattr(planning_mod, "probe_devices", lambda _b: probe)
        monkeypatch.setattr(planning_mod.model_cache, "has_nvidia_gpu", lambda: False)
        monkeypatch.setattr(gpu_select, "enumerate_gpu_vram", lambda: [(1, 8 * _GB, 8 * _GB)])
        monkeypatch.setattr(gpu_select, "integrated_vulkan_indices", frozenset)

        devices = planning_mod.resolve_devices(_Path("/fake/llama-server"))

        assert [d.from_loader for d in devices] == [True]
        assert planning_mod._device_names(tuple(devices)) == ()


class TestARefusedDeviceIsAlsoKeptFromTheEngine:
    """A CPU-shaped plan does not make the engine use the CPU.

    Without a pin, ggml's fallback takes the first non-CPU adapter, i.e. the one
    lilbee refused, and offloads every layer onto it while placement budgeted
    against system RAM.
    """

    def test_the_launch_names_no_device(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod._plan_probe_store,
            "get",
            lambda: SimpleNamespace(engine_devices_all_refused=True),
        )

        assert planning_mod._cpu_pin_when_every_device_was_refused() == ("none",)

    def test_a_plain_cpu_host_is_left_unpinned(self, monkeypatch) -> None:
        """Nothing was refused, so there is nothing to keep the engine off."""
        monkeypatch.setattr(
            planning_mod._plan_probe_store,
            "get",
            lambda: SimpleNamespace(engine_devices_all_refused=False),
        )

        assert planning_mod._cpu_pin_when_every_device_was_refused() == ()

    def test_no_snapshot_means_no_claim(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: None)

        assert planning_mod._cpu_pin_when_every_device_was_refused() == ()

    def test_a_placed_gpu_still_wins_over_the_cpu_pin(self, monkeypatch) -> None:
        """The refusal pin is a fallback for an empty device list, not an override."""
        monkeypatch.setattr(
            planning_mod._plan_probe_store,
            "get",
            lambda: SimpleNamespace(engine_devices_all_refused=True),
        )
        chosen = (FleetDevice(VULKAN_BACKEND, 0, "Card A", 8 * _GB, 8 * _GB),)

        names = planning_mod._device_names(chosen) or (
            planning_mod._cpu_pin_when_every_device_was_refused()
        )

        assert names == ("Vulkan0",)


def test_capturing_the_plan_snapshot_probes_the_engine_once(monkeypatch) -> None:
    """--list-devices is a subprocess against a driver that may be wedged, with a
    minute-long timeout. Devices and the all-refused fact come from one run."""
    from lilbee.providers.fleet.devices import DeviceProbe

    runs: list[int] = []

    def _counting(_binary, **_kw) -> DeviceProbe:
        runs.append(1)
        return DeviceProbe(
            [FleetDevice("CUDA", 0, "A", 24 * _GB, 20 * _GB)],
            "Available devices:\n",
            spoke_protocol=True,
        )

    monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
    monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
    monkeypatch.setattr("lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda: None)
    monkeypatch.setattr(
        "lilbee.providers.fleet.cuda_runtime.assert_gpu_devices_usable", lambda *_a: None
    )
    monkeypatch.setattr(planning_mod, "probe_devices", _counting)
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 64 * _GB)
    planning_mod.clear_plan_probe()
    try:
        planning_mod.capture_plan_probe()
    finally:
        planning_mod.clear_plan_probe()

    assert len(runs) == 1, f"ran --list-devices {len(runs)} times for one snapshot"


class TestTheFleetBackendWithoutAPlanSnapshot:
    """Flash-attention and KV choices ask which backend the host plans onto.

    Outside a planning pass there is no snapshot, so the question falls to the
    read cache, and a probe that cannot run must answer "unknown" rather than
    propagate out of a flag decision.
    """

    def test_the_read_cache_answers_when_no_snapshot_exists(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: None)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr(
            planning_mod._read_device_cache,
            "get",
            lambda _b: [FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)],
        )

        assert planning_mod._fleet_backend() == "CUDA"

    def test_a_probe_that_cannot_run_yields_no_backend(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        def _boom(_b):
            raise ProviderError("probe wedged")

        monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: None)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr(planning_mod._read_device_cache, "get", _boom)

        assert planning_mod._fleet_backend() is None

    def test_a_host_with_no_devices_yields_no_backend(self, monkeypatch) -> None:
        monkeypatch.setattr(planning_mod._plan_probe_store, "get", lambda: None)
        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr(planning_mod._read_device_cache, "get", lambda _b: [])

        assert planning_mod._fleet_backend() is None


class TestTheFleetBackendFromThePlanSnapshot:
    """Inside a planning pass the backend comes from the snapshot, so a whole
    pass answers consistently even as the live probe changes underneath it."""

    def test_the_snapshot_names_the_backend(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod._plan_probe_store,
            "get",
            lambda: SimpleNamespace(devices=(FleetDevice("ROCm", 0, "AMD", 24 * _GB, 23 * _GB),)),
        )

        assert planning_mod._fleet_backend() == "ROCm"

    def test_a_snapshot_of_a_gpu_less_host_names_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod._plan_probe_store, "get", lambda: SimpleNamespace(devices=())
        )

        assert planning_mod._fleet_backend() is None

    def test_the_snapshot_wins_over_the_live_probe(self, monkeypatch) -> None:
        """A live read under a loaded fleet can disagree; the pass must not."""
        monkeypatch.setattr(
            planning_mod._plan_probe_store,
            "get",
            lambda: SimpleNamespace(
                devices=(FleetDevice("CUDA", 0, "NVIDIA", 24 * _GB, 23 * _GB),)
            ),
        )

        def _must_not_run(_b):
            raise AssertionError("the live probe was consulted despite a snapshot")

        monkeypatch.setattr(planning_mod._read_device_cache, "get", _must_not_run)

        assert planning_mod._fleet_backend() == "CUDA"


class TestABusyUnifiedHostStillServesASmallModel:
    """Admission and sizing ask different questions of the same RAM.

    Routing unified hosts to the refusing path made admission read instantaneous
    free RAM, so CI refused a 0.6B chat model on a macOS runner that was merely
    busy: "does not fit available memory and will not be served". The plan
    defines the whole intended residency, so what it must fit is the machine,
    not the machine's current moment.
    """

    def _host(self, monkeypatch, *, total: int, free: int) -> list[FleetDevice]:
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: total)
        monkeypatch.setattr(planning_mod, "_plan_free_system_memory", lambda: free)
        return [FleetDevice("MTL", 0, "Apple M2", total, free, unified=True)]

    def test_admission_charges_the_machine_not_the_moment(self, monkeypatch) -> None:
        devices = self._host(monkeypatch, total=16 * _GB, free=1 * _GB)

        assert planning_mod._unified_admission_budget(devices) > 8 * _GB

    def test_sizing_still_charges_what_is_free(self, monkeypatch) -> None:
        """Context and slots must fit what can actually be backed right now."""
        devices = self._host(monkeypatch, total=16 * _GB, free=1 * _GB)

        assert planning_mod._unified_memory_budget(devices) < 2 * _GB

    def test_a_dedicated_host_has_neither_budget(self, monkeypatch) -> None:
        devices = [FleetDevice("CUDA", 0, "NVIDIA", 24 * _GB, 23 * _GB)]

        assert planning_mod._unified_memory_budget(devices) is None
        assert planning_mod._unified_admission_budget(devices) is None

    def test_a_model_larger_than_the_machine_is_still_refused(self, monkeypatch) -> None:
        """The refusal the routing change exists for has to survive the fix."""
        from lilbee.providers.fleet.placement import ModelPlacementInput, plan_placement

        devices = self._host(monkeypatch, total=16 * _GB, free=8 * _GB)
        budget = planning_mod._unified_admission_budget(devices)

        assert budget is not None
        plan = plan_placement(
            [ModelPlacementInput(WorkerRole.CHAT, 40 * _GB)],
            [(0, 16 * _GB)],
            estimate_peak=lambda _r, ratio: tuple(40 * _GB for _ in ratio),
            unified_budget=budget,
        )

        assert WorkerRole.CHAT in plan.unplaceable_roles
