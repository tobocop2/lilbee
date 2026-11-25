"""The plan sizes the engine's prompt cache and context checkpoints, and charges both.

llama-server keeps idle prompts (``--cache-ram``, 8 GiB by default) and context
checkpoints (``--ctx-checkpoints``, 32 a slot by default) in system memory. The plan
passes both bounds, sized from the room it leaves free, and records their charge.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType, RerankerType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import model_memory as model_memory_mod
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_gguf
from lilbee.providers.fleet.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import RoleTarget, WorkerRole

_MIB = 1024**2
_GIB = 1024**3
_SLOTS = 2
_HELD = 3 * _GIB
"""What the fake model holds in system memory at the launch's context and slots."""
_CHECKPOINT = 100 * _MIB
"""One context checkpoint of the fake sliding-window model."""


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A CPU host whose installed RAM the test sets, and a model of known size."""
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server",
        lambda: Path("/bin/llama-server"),
    )
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    monkeypatch.setattr(planning_mod, "_slots_for", lambda *_a, **_k: _SLOTS)
    monkeypatch.setattr(planning_mod, "_system_memory_floor", lambda: 0)
    monkeypatch.setattr(planning_mod, "loopback_reserve_bytes", lambda: 0)
    monkeypatch.setattr(planning_mod, "_plan_free_system_memory", lambda _e: 2 * _GIB)
    monkeypatch.setattr(model_memory_mod, "repacked_weight_bytes", lambda _p: 0)
    monkeypatch.setattr(model_memory_mod, "attention_shape", lambda _p: None)
    monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
    checkpoint = {"bytes": _CHECKPOINT}

    def estimate(_path, **_kwargs) -> GgufVramEstimate:
        return GgufVramEstimate(
            vram_bytes=0,
            ram_bytes=_HELD,
            unified_bytes=_HELD,
            checkpoint_bytes=checkpoint["bytes"],
        )

    monkeypatch.setattr(model_memory_mod, "estimate_instance_footprint", estimate)

    def set_host(*, total: int, checkpoint_bytes: int = _CHECKPOINT) -> None:
        checkpoint["bytes"] = checkpoint_bytes
        monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: total)

    return set_host


def _launch(role: WorkerRole = WorkerRole.CHAT) -> InstanceLaunch:
    plan = InstancePlan(role=role, devices=(), target=RoleTarget.CPU)
    return planning_mod._launch_for(plan, "ref", {}, unified_budget=1, engine=LLAMA)


def _flag(launch: InstanceLaunch, flag: str) -> int | None:
    if flag not in launch.argv:
        return None
    return int(launch.argv[launch.argv.index(flag) + 1])


def _total_for(cache_mib: int, checkpoints: int = 4) -> int:
    """Installed RAM that leaves exactly *cache_mib* of cache after the checkpoints."""
    return _HELD + _SLOTS * checkpoints * _CHECKPOINT + 2 * cache_mib * _MIB


class TestCheckpoints:
    def test_a_chat_launch_keeps_four_checkpoints_a_slot_when_the_room_holds_them(
        self, host
    ) -> None:
        host(total=_total_for(1000))
        assert _flag(_launch(), "--ctx-checkpoints") == 4

    def test_the_checkpoints_are_lowered_to_what_the_room_holds(self, host) -> None:
        host(total=_HELD + _SLOTS * 2 * _CHECKPOINT + _CHECKPOINT)
        assert _flag(_launch(), "--ctx-checkpoints") == 2

    def test_a_host_with_no_room_keeps_no_checkpoints(self, host) -> None:
        host(total=_HELD // 2)
        assert _flag(_launch(), "--ctx-checkpoints") == 0

    def test_a_model_that_makes_no_checkpoints_gets_the_flag_and_is_charged_only_its_cache(
        self, host
    ) -> None:
        host(total=_HELD + 2 * 500 * _MIB, checkpoint_bytes=0)
        launch = _launch()
        assert _flag(launch, "--ctx-checkpoints") == 4
        assert launch.est_host_cache_bytes == 500 * _MIB


class TestPromptCache:
    def test_the_cache_is_half_the_room_left_after_the_checkpoints(self, host) -> None:
        host(total=_total_for(1000))
        assert _flag(_launch(), "--cache-ram") == 1000

    def test_the_cache_never_passes_the_engine_default(self, host) -> None:
        host(total=64 * _GIB)
        assert _flag(_launch(), "--cache-ram") == 8192

    def test_the_cache_is_sized_from_installed_ram_not_free_ram(self, host, monkeypatch) -> None:
        host(total=_total_for(1000))
        first = _launch().argv
        monkeypatch.setattr(planning_mod, "_plan_free_system_memory", lambda _e: 9 * _GIB)
        assert _launch().argv == first

    @pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
    def test_a_pooled_role_gets_no_prompt_cache_and_no_checkpoint_flag(
        self, host, monkeypatch, role: WorkerRole
    ) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.CROSS_ENCODER)
        host(total=64 * _GIB)
        launch = _launch(role)
        assert _flag(launch, "--cache-ram") == 0
        assert "--ctx-checkpoints" not in launch.argv
        assert launch.est_host_cache_bytes == 0

    def test_an_llm_reranker_keeps_prompts_like_chat(self, host, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "reranker_type", RerankerType.LLM)
        host(total=_total_for(1000))
        launch = _launch(WorkerRole.RERANK)
        assert _flag(launch, "--cache-ram") == 1000
        assert _flag(launch, "--ctx-checkpoints") == 4


class TestCharge:
    def test_the_charge_is_the_cache_plus_every_slots_checkpoints(self, host) -> None:
        host(total=_total_for(1000))
        launch = _launch()
        assert launch.est_host_cache_bytes == 1000 * _MIB + _SLOTS * 4 * _CHECKPOINT

    def test_the_charge_stays_apart_from_the_host_buffer_estimate(self, host) -> None:
        host(total=_total_for(1000))
        assert _launch().est_host_bytes == 0

    def test_the_charge_survives_the_engine_state_file(self, host) -> None:
        host(total=_total_for(1000))
        launch = _launch()
        state = json.loads(json.dumps(launch.to_state()))
        assert InstanceLaunch.from_state(state).est_host_cache_bytes == launch.est_host_cache_bytes

    def test_a_build_that_lists_neither_flag_drops_both_and_is_charged_nothing(
        self, host, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "lilbee.providers.fleet.adapters.engine_help", lambda _b: "  --model X  the model\n"
        )
        host(total=_total_for(1000))
        launch = _launch()
        assert "--cache-ram" not in launch.argv
        assert "--ctx-checkpoints" not in launch.argv
        assert launch.est_host_cache_bytes == 0


def _lost_bound_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "starts without" in r.getMessage()]


class TestNoRoom:
    def test_a_host_with_no_room_launches_without_a_prompt_cache_and_warns_once(
        self, host, caplog
    ) -> None:
        host(total=_HELD // 2)
        with caplog.at_level(logging.WARNING, logger="lilbee"):
            launch = _launch()
        assert _flag(launch, "--cache-ram") == 0
        assert launch.est_host_cache_bytes == 0
        named = _lost_bound_warnings(caplog)
        assert len(named) == 1
        assert "without its prompt cache" in named[0]
        assert "Each turn reprocesses the whole conversation." in named[0]

    def test_a_lost_cache_with_its_checkpoints_kept_does_not_say_each_turn_is_reprocessed(
        self, host, caplog
    ) -> None:
        host(total=_HELD + _SLOTS * 4 * _CHECKPOINT + _MIB)
        with caplog.at_level(logging.WARNING, logger="lilbee"):
            launch = _launch()
        assert _flag(launch, "--cache-ram") == 0
        named = _lost_bound_warnings(caplog)
        assert len(named) == 1
        assert "Each turn" not in named[0]

    def test_lost_checkpoints_with_a_cache_kept_warn_once_that_each_turn_is_reprocessed(
        self, host, monkeypatch, caplog
    ) -> None:
        # 300 MiB of room and 4 slots of 116 MiB checkpoints: no checkpoint fits, and half
        # the room, 150 MiB, is the prompt cache.
        monkeypatch.setattr(planning_mod, "_slots_for", lambda *_a, **_k: 4)
        host(total=_HELD + 300 * _MIB, checkpoint_bytes=116 * _MIB)
        with caplog.at_level(logging.WARNING, logger="lilbee"):
            launch = _launch()
        assert _flag(launch, "--ctx-checkpoints") == 0
        assert _flag(launch, "--cache-ram") == 150
        named = _lost_bound_warnings(caplog)
        assert len(named) == 1
        assert "without context checkpoints" in named[0]
        assert "Each turn reprocesses the whole conversation." in named[0]

    def test_a_model_that_makes_no_checkpoints_does_not_warn_for_a_bound_of_zero(
        self, host, caplog
    ) -> None:
        bounds = planning_mod._HostCache(cache_ram_mib=10, ctx_checkpoints=0)
        launch = InstanceLaunch(
            role=WorkerRole.CHAT, argv=[], env_overrides={}, model="m.gguf", slots=1
        )
        with caplog.at_level(logging.WARNING, logger="lilbee"):
            planning_mod._warn_without_prompt_cache(launch, bounds, 0)
        assert not _lost_bound_warnings(caplog)

    def test_a_host_with_room_does_not_warn(self, host, caplog) -> None:
        host(total=_total_for(1000))
        with caplog.at_level(logging.WARNING, logger="lilbee"):
            _launch()
        assert not _lost_bound_warnings(caplog)


def _draft(role: WorkerRole, *, held: int, checkpoint: int, keeps: bool):
    launch = InstanceLaunch(
        role=role, argv=["/bin/llama-server"], env_overrides={}, model=f"{role}.gguf", slots=1
    )
    return SimpleNamespace(
        launch=launch, held_bytes=held, checkpoint_bytes=checkpoint, keeps_prompts=keeps
    )


class TestHeldMemory:
    _REQUEST = MemoryRequest(
        model_path=Path("m.gguf"),
        ctx=4096,
        slots=1,
        gpu_layers=0,
        flash_attn=True,
        kv_cache_type=KvCacheType.F16,
    )

    @pytest.mark.parametrize("error", [ProviderError("no parser"), OSError("gone")])
    def test_a_model_the_estimator_cannot_size_holds_its_weights_and_no_checkpoints(
        self, error: Exception
    ) -> None:
        def fail(_request):
            raise error

        engine = SimpleNamespace(estimate_memory=fail)
        held = planning_mod._held_host_memory(
            engine, self._REQUEST, one_pool=True, weights_bytes=7 * _GIB
        )
        assert held == (7 * _GIB, 0)

    def test_beside_a_card_an_instance_holds_only_its_host_half(self) -> None:
        estimate = SimpleNamespace(host_bytes=3, checkpoint_bytes=5)
        engine = SimpleNamespace(estimate_memory=lambda _r: estimate)
        held = planning_mod._held_host_memory(
            engine, self._REQUEST, one_pool=False, weights_bytes=9
        )
        assert held == (3, 5)


class TestPlanWideRoom:
    def test_the_room_is_what_every_instance_leaves_and_the_keepers_share_the_cache(
        self,
    ) -> None:
        drafts = [
            _draft(WorkerRole.CHAT, held=4 * _GIB, checkpoint=100 * _MIB, keeps=True),
            _draft(WorkerRole.EMBED, held=1 * _GIB, checkpoint=0, keeps=False),
            _draft(WorkerRole.VISION, held=2 * _GIB, checkpoint=0, keeps=True),
        ]
        room = 12 * _GIB - 7 * _GIB - 4 * 100 * _MIB
        bounds = planning_mod._host_cache_bounds(drafts, 12 * _GIB)
        assert [b.ctx_checkpoints for b in bounds] == [4, 0, 4]
        assert [b.cache_ram_mib for b in bounds] == [room // 2 // 2 // _MIB, 0, room // 4 // _MIB]

    def test_the_plan_loop_sizes_every_launch_against_the_whole_plan(
        self, host, monkeypatch
    ) -> None:
        host(total=_HELD * 2 + 2 * _SLOTS * 4 * _CHECKPOINT + 4 * 1000 * _MIB)
        monkeypatch.setattr(planning_mod, "_unusable_chat_ctx_reason", lambda *_a: None)
        roles = (WorkerRole.CHAT, WorkerRole.VISION)
        monkeypatch.setattr(planning_mod, "_text_only_vision_reason", lambda *_a: None)
        monkeypatch.setattr(planning_mod, "vision_mmproj", lambda _ref: None)
        placement = Placement(
            instances=tuple(InstancePlan(r, (), target=RoleTarget.CPU) for r in roles),
            unplaceable_roles=(),
        )
        launches, refused = planning_mod._launches_and_refusals(
            placement,
            [ModelPlacementInput(r, _HELD) for r in roles],
            {r: "ref" for r in roles},
            {},
            engines={r: LLAMA for r in roles},
            unified_budget=1,
            chat_reservation=0,
            capacity={},
            clean_box=False,
        )
        assert refused == {}
        assert [_flag(launch, "--cache-ram") for launch in launches] == [1000, 1000]
        assert [_flag(launch, "--ctx-checkpoints") for launch in launches] == [4, 4]


_GEMMA3_4B = {
    "architecture": "gemma3",
    "blockCount": 34,
    "attentionHeadCount": 8,
    "attentionHeadCountKV": 4,
    "attentionKeyLength": 256,
    "attentionValueLength": 256,
    "attentionSlidingWindow": 1024,
    "attentionSlidingWindowLayers": [i % 6 != 5 for i in range(34)],
}


def _report(architecture: dict) -> str:
    item = {"ram": {"uma": 600, "nonuma": 700}, "vrams": [{"uma": 0, "nonuma": 0}]}
    return json.dumps({"estimate": {"items": [item]}, "architecture": architecture})


class TestCheckpointBytes:
    def test_a_sliding_window_checkpoint_is_the_window_of_one_sequence_at_f16(self) -> None:
        # 29 windowed layers x 4 KV heads x (256 + 256) x 2 bytes x 1024: the 116.000 MiB that
        # the engine logged as 116.012 MiB for gemma-3-4b.
        parsed = vram_mod._parse_estimate(_report(_GEMMA3_4B), "m.gguf")
        assert parsed.checkpoint_bytes == 116 * _MIB

    def test_the_checkpoint_follows_the_kv_cache_type(self) -> None:
        # The engine logged 61.637 MiB for the same model with a q8_0 cache.
        assert vram_mod.checkpoint_bytes(_GEMMA3_4B, 34 / 32) == int(116 * _MIB * 34 / 64)

    def test_the_parser_run_passes_its_cache_type_to_the_checkpoint(self, monkeypatch) -> None:
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/bin/gguf-parser"))
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _argv, _p: _report(_GEMMA3_4B))
        parsed = vram_mod._cached_footprint.__wrapped__(
            "e", "m.gguf", 0, 4096, 1, 0, True, KvCacheType.Q4_0.value, None, 0, (), None, ()
        )
        assert parsed.checkpoint_bytes == int(116 * _MIB * 18 / 64)

    def test_a_model_without_a_window_or_recurrent_layers_makes_no_checkpoints(self) -> None:
        plain = {k: v for k, v in _GEMMA3_4B.items() if not k.startswith("attentionSliding")}
        assert vram_mod.checkpoint_bytes(plain, 2) == 0

    def test_a_window_without_a_layer_list_counts_every_block(self) -> None:
        every = {k: v for k, v in _GEMMA3_4B.items() if k != "attentionSlidingWindowLayers"}
        assert vram_mod.checkpoint_bytes(every, 2) == 34 * 4 * 512 * 2 * 1024

    def test_a_recurrent_checkpoint_is_the_f32_conv_and_ssm_state_of_every_block(self) -> None:
        mamba2 = {
            "blockCount": 2,
            "attentionRecurrent": True,
            "ssmConvolutionKernel": 4,
            "ssmInnerSize": 8,
            "ssmStateSize": 16,
            "ssmGroupCount": 1,
        }
        per_layer = 3 * (8 + 2 * 1 * 16) + 16 * 8
        assert vram_mod.checkpoint_bytes(mamba2, 2) == 2 * per_layer * 4

    @pytest.mark.parametrize(
        "architecture",
        [
            {"blockCount": 2, "attentionHybrid": True, "rwkvHeadSize": 64},
            {"blockCount": 2, "attentionSlidingWindow": 512},
        ],
    )
    def test_an_architecture_the_header_cannot_size_charges_one_slots_share(
        self, monkeypatch, architecture: dict
    ) -> None:
        assert vram_mod.checkpoint_bytes(architecture, 2) is None
        parsed = vram_mod._parse_estimate(_report(architecture), "m.gguf")
        monkeypatch.setattr(model_memory_mod, "gguf_is_incomplete", lambda _p: False)
        monkeypatch.setattr(
            model_memory_mod, "estimate_instance_footprint", lambda *_a, **_k: parsed
        )
        request = MemoryRequest(
            model_path=Path("m.gguf"),
            ctx=4096,
            slots=3,
            gpu_layers=0,
            flash_attn=True,
            kv_cache_type=KvCacheType.F16,
        )
        assert estimate_gguf(request).checkpoint_bytes == 600 // 3

    def test_a_projector_keeps_the_text_models_checkpoint(self) -> None:
        base = GgufVramEstimate(1, 1, 1, checkpoint_bytes=7)
        corrected = vram_mod._corrected_projector_estimate(
            base, replace(base, unified_bytes=5, checkpoint_bytes=9), 2
        )
        assert corrected.checkpoint_bytes == 7


class TestTheLaunchKvTypePricesTheCheckpoint:
    def test_a_q8_0_chat_launch_is_charged_q8_0_checkpoints(self, host, monkeypatch) -> None:
        # gemma-3-4b's header through the real parser path: one q8_0 checkpoint is
        # 29 x 4 x 512 x 1024 x 34/32 = 64618496 bytes. Two slots keep 4 each, and 64 GiB
        # leaves room for the 8192 MiB cache.
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/bin/gguf-parser"))
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _argv, _p: _report(_GEMMA3_4B))
        monkeypatch.setattr(
            model_memory_mod, "estimate_instance_footprint", vram_mod.estimate_instance_footprint
        )
        monkeypatch.setattr(model_memory_mod, "gguf_is_incomplete", lambda _p: False)
        vram_mod._cached_footprint.cache_clear()
        host(total=64 * _GIB)
        launch = _launch()
        vram_mod._cached_footprint.cache_clear()
        assert launch.argv[launch.argv.index("--cache-type-k") + 1] == "q8_0"
        # 8192 MiB of cache plus 2 x 4 x 64618496: at f16 it would be 9563013120.
        assert launch.est_host_cache_bytes == 9_106_882_560
