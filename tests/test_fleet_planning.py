"""Tests for fleet launch planning: VRAM estimate, placement, argv, device probe."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice, visible_env
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.roles import WorkerRole

_GB = 1024**3


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
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_chat_ctx", lambda _p, _m: 4096)
    assert planning_mod._role_ctx(WorkerRole.CHAT, Path("/m/c.gguf"), None) == 4096


def test_role_ctx_embed_uses_model_training_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.train_ctx_from_meta",
        lambda _meta, *, fallback, model_path: 512,
    )
    assert planning_mod._role_ctx(WorkerRole.EMBED, Path("/m/e.gguf"), {}) == 512


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


def test_slots_for_chat_and_aux_are_fixed() -> None:
    # Non-vision roles ignore the model args; chat batches, aux is single-slot.
    assert planning_mod._slots_for(WorkerRole.CHAT, 0, None, 0) == 4
    assert planning_mod._slots_for(WorkerRole.EMBED, 0, None, 0) == 1
    assert planning_mod._slots_for(WorkerRole.RERANK, 0, None, 0) == 1


_VISION_META = {"block_count": "24", "embedding_length": "2048"}


def test_resolve_vision_slots_uses_ceiling_when_vram_is_ample(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 10**12)
    assert planning_mod._resolve_vision_slots(2 * 10**9, _VISION_META, 16384) == 4


def test_resolve_vision_slots_drops_to_one_on_small_gpu(monkeypatch) -> None:
    # A tiny VRAM budget can't fit multiple slots of vision KV -> falls back to 1.
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 4)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.9)
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 3 * 10**9)
    assert planning_mod._resolve_vision_slots(2 * 10**9, _VISION_META, 16384) == 1


def test_resolve_vision_slots_ceiling_one_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "vision_ocr_concurrency", 1)
    # Even with huge VRAM, a ceiling of 1 means strictly sequential OCR.
    monkeypatch.setattr("lilbee.providers.model_cache.get_available_memory", lambda _f: 10**12)
    assert planning_mod._resolve_vision_slots(2 * 10**9, _VISION_META, 16384) == 1


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
    _inputs, refs = planning_mod._server_model_inputs((WorkerRole.EMBED,))
    assert set(refs) == {WorkerRole.EMBED}


class TestBuildFleetWiring:
    def test_server_model_inputs_skips_unconfigured_optional_roles(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod,
            "_estimate_role",
            lambda role, ref, **_k: ModelPlacementInput(role, 5 * _GB),
        )
        monkeypatch.setattr(cfg, "reranker_model", "")  # unconfigured -> skipped
        monkeypatch.setattr(cfg, "vision_model", "")
        inputs, refs = planning_mod._server_model_inputs()
        assert {i.role for i in inputs} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert set(refs) == {WorkerRole.CHAT, WorkerRole.EMBED}

    def test_server_model_inputs_skips_role_whose_model_is_not_installed(
        self, monkeypatch
    ) -> None:
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
        inputs, refs = planning_mod._server_model_inputs()
        assert WorkerRole.CHAT not in refs
        assert {i.role for i in inputs} == {WorkerRole.EMBED}

    def test_server_model_inputs_includes_configured_rerank(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod, "_estimate_role", lambda role, ref, **_k: ModelPlacementInput(role, _GB)
        )
        monkeypatch.setattr(cfg, "reranker_model", "some/reranker.gguf")
        monkeypatch.setattr(cfg, "vision_model", "")
        _inputs, refs = planning_mod._server_model_inputs()
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

    def test_estimate_role_vision_adds_mmproj_size(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        mmproj = tmp_path / "mmproj.gguf"
        mmproj.write_bytes(b"y" * 500)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: mmproj)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 16)
        inp = planning_mod._estimate_role(WorkerRole.VISION, "ref", slots=1)
        assert inp.est_vram_bytes >= 1500  # weights + mmproj counted

    def test_estimate_role_resolves_slots_when_unspecified(self, tmp_path, monkeypatch) -> None:
        # With no explicit slots, the estimate sizes them via _slots_for (vision
        # is VRAM-aware), so placement and the launched --parallel stay consistent.
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: None)
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 16)
        inp = planning_mod._estimate_role(WorkerRole.VISION, "ref")  # slots resolved internally
        assert inp.role is WorkerRole.VISION
        assert inp.est_vram_bytes >= 1000

    def test_estimate_role_aux_kv_uses_f16_not_configured_type(self, tmp_path, monkeypatch) -> None:
        # Aux roles run f16 KV regardless of cfg.kv_cache_type, so the estimate
        # must use f16 to match runtime (only chat passes --cache-type).
        from lilbee.core.config.enums import KV_CACHE_TYPE_BYTES, KvCacheType

        model = tmp_path / "e.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata",
            lambda _p: {"block_count": "8", "embedding_length": "16"},
        )
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 512)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)  # would be wrong for embed
        inp = planning_mod._estimate_role(WorkerRole.EMBED, "ref", slots=1)
        f16 = KV_CACHE_TYPE_BYTES[KvCacheType.F16]
        expected_kv = 2 * 8 * 16 * 512 * 1 * f16
        assert inp.est_vram_bytes == 1000 + expected_kv + 1024**3  # weights + f16 KV + overhead

    def test_launch_for_vision_passes_mmproj(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "v.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda _r: model,
        )
        monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 4096)
        plan = InstancePlan(role=WorkerRole.VISION, devices=(0,))
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        launch = planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert "--mmproj" in launch.argv
        assert str(Path("/m/mmproj.gguf")) in launch.argv

    def test_estimate_role_reads_weights_and_meta(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1000)
        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: model)
        monkeypatch.setattr(
            "lilbee.providers.gguf_meta.read_gguf_metadata",
            lambda _p: {"block_count": "4", "embedding_length": "8"},
        )
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 16)
        inp = planning_mod._estimate_role(WorkerRole.CHAT, "ref", slots=2)
        assert inp.role == WorkerRole.CHAT
        assert inp.est_vram_bytes > 1000  # weights + kv + overhead

    def test_launch_for_builds_instance_with_pinning(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "chat.gguf"
        model.write_bytes(b"x" * 2048)
        monkeypatch.setattr(
            "lilbee.providers.engine_params.resolve_model_path",
            lambda ref: model,
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: 4096)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
        launch = planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )
        assert launch.role == WorkerRole.CHAT
        assert launch.env_overrides == visible_env((device,))
        # port file is stamped with the owning pid so reaping is instance-safe
        assert launch.port_file == Path(f"/data/llama-server-chat-{os.getpid()}.port")
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
        monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m: ctx)
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        plan = InstancePlan(role=role, devices=(0,))
        return planning_mod._launch_for(
            plan, "ref", Path("/bin/llama-server"), Path("/data"), {0: device}
        )

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_launch_for_embed_roles_set_token_cap(self, tmp_path, monkeypatch, role) -> None:
        launch = self._launch_for_role(tmp_path, monkeypatch, role, ctx=8192)
        assert launch.token_cap == 8192  # embed/rerank truncate to per-slot ctx

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

    def test_launch_for_vision_threads_floor_when_cpu_count_unknown(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(planning_mod.os, "cpu_count", lambda: None)
        argv = self._launch_role(tmp_path, monkeypatch, WorkerRole.VISION)
        assert argv[argv.index("--threads") + 1] == str(planning_mod._DEFAULT_THREADS)

    def test_build_fleet_resolves_devices_plans_and_starts(self, monkeypatch) -> None:
        device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
        monkeypatch.setattr(
            planning_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(planning_mod, "probe_devices", lambda _binary: [device])
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
            ),
        )
        monkeypatch.setattr(
            planning_mod,
            "plan_placement",
            lambda inputs, devices: Placement(
                instances=(InstancePlan(WorkerRole.CHAT, (0,)),), unplaceable_roles=()
            ),
        )
        monkeypatch.setattr(planning_mod, "_launch_for", lambda *a: MagicMock())
        started = {"n": 0}
        monkeypatch.setattr(
            planning_mod.Fleet, "start", lambda self, launches: started.__setitem__("n", 1)
        )
        fleet = planning_mod.build_fleet()
        assert isinstance(fleet, planning_mod.Fleet)
        assert started["n"] == 1

    def test_build_fleet_falls_back_to_vulkan_probe(self, monkeypatch) -> None:
        monkeypatch.setattr(
            planning_mod, "resolve_llama_server_binary", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr(planning_mod, "probe_devices", lambda _binary: [])  # can't enumerate
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_select.enumerate_gpu_vram",
            lambda: [(0, 24 * _GB)],
        )
        seen: dict[str, list] = {}
        monkeypatch.setattr(
            planning_mod,
            "_server_model_inputs",
            lambda *_roles: (
                [ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)],
                {WorkerRole.CHAT: "ref"},
            ),
        )

        def _capture(inputs, devices):
            seen["devices"] = devices
            return Placement(instances=(), unplaceable_roles=(WorkerRole.CHAT,))

        monkeypatch.setattr(planning_mod, "plan_placement", _capture)
        monkeypatch.setattr(planning_mod.Fleet, "start", lambda self, launches: None)
        planning_mod.build_fleet()
        assert seen["devices"] == [(0, 24 * _GB)]  # synthesized from the Vulkan fallback
