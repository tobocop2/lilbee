"""A one-pool charge counts the CPU build's repacked weight copy, and a shared file once."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import model_memory as model_memory_mod
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import vram as vram_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryRequest
from lilbee.providers.fleet.model_memory import CPU_RUNTIME_ALLOWANCE_BYTES
from lilbee.providers.fleet.placement import ModelPlacementInput
from lilbee.providers.fleet.vram import AttentionShape
from lilbee.providers.roles import EngineBackend, RerankMode, WorkerRole
from tests._fleet_engines import engines_for_all_roles

Q4_0, Q6_K, Q4_K, IQ4_NL = 2, 14, 12, 20
FILE_SIZE = 5000
UNIFIED = 700
# 2-D Q4_K [256, 4]: 4 blocks of 144 bytes. Q4_0 [64, 2]: 4 blocks of 18. IQ4_NL [32, 3]: 3 of 18.
REPACKED = 4 * 144 + 4 * 18 + 3 * 18
# A chat launch on the CPU also carries the runtime allowance; the shape of this model is empty.
CPU_CHARGE = UNIFIED + FILE_SIZE + REPACKED + CPU_RUNTIME_ALLOWANCE_BYTES


def _tensor(kind: int, dims: list[int]) -> dict[str, object]:
    return {"name": "t", "nDimensions": len(dims), "dimensions": dims, "type": kind}


RAW = {
    "tensorInfos": [
        _tensor(Q4_K, [256, 4]),
        _tensor(Q4_0, [64, 2]),
        _tensor(IQ4_NL, [32, 3]),
        _tensor(Q6_K, [256, 8]),
        _tensor(Q4_K, [256]),
    ]
}


@pytest.fixture
def model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    vram_mod._cached_footprint.cache_clear()
    vram_mod._cached_repacked_bytes.cache_clear()
    monkeypatch.setattr(
        model_memory_mod, "estimate_instance_footprint", vram_mod.estimate_instance_footprint
    )
    monkeypatch.setattr(model_memory_mod, "attention_shape", lambda _p: AttentionShape(0, 0, 0))
    monkeypatch.setattr(vram_mod, "resolve_gguf_parser", lambda: Path("/fake/gguf-parser"))

    def _parser(argv: list[str], _path: str) -> str:
        if "--raw" in argv:
            return json.dumps(RAW)
        return json.dumps(
            {
                "metadata": {"fileSize": FILE_SIZE},
                "estimate": {
                    "items": [
                        {
                            "ram": {"uma": 0, "nonuma": 0},
                            "vrams": [{"uma": UNIFIED, "nonuma": UNIFIED}],
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr(vram_mod, "_run_parser", _parser)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: path)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a, engine: 16)
    return path


def _request(path: Path, *, on_cpu: bool) -> MemoryRequest:
    return MemoryRequest(
        model_path=path,
        ctx=16,
        slots=1,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=cfg.kv_cache_type,
        on_cpu=on_cpu,
    )


class TestTheRepackedCopy:
    def test_it_is_the_two_dimensional_tensors_of_the_repacked_types(self, model: Path) -> None:
        assert vram_mod.repacked_weight_bytes(model) == REPACKED

    def test_unreadable_tensor_output_is_refused(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps({"header": {}}))
        with pytest.raises(ProviderError, match="unreadable output"):
            vram_mod.repacked_weight_bytes(model)

    def test_a_cpu_load_backs_the_file_and_the_copy(self, model: Path) -> None:
        cpu = LLAMA.estimate_memory(_request(model, on_cpu=True))
        card = LLAMA.estimate_memory(_request(model, on_cpu=False))
        assert cpu.resident_footprint(unified=True) == CPU_CHARGE
        assert card.resident_footprint(unified=True) == UNIFIED + FILE_SIZE

    def test_the_alternate_build_keeps_the_cpu_chat_figure_without_the_copy(
        self, model: Path
    ) -> None:
        from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA

        estimate = IK_LLAMA.estimate_memory(_request(model, on_cpu=True))
        assert estimate.resident_footprint(unified=True) == CPU_CHARGE - REPACKED

    def test_the_openvino_build_backs_the_file_without_the_copy_on_the_cpu(
        self, model: Path
    ) -> None:
        """llama-server b12032 on OpenVINO 2026.4.1, CPU device, Llama-3.2-1B at ctx 2048.

        Its anonymous memory after load is 335.4 MiB for Q4_0 and 331.3 MiB for Q8_0: the
        Q4_0 weights, which the llama.cpp CPU build repacks into 504.0 MiB, add nothing.
        """
        from lilbee.providers.fleet.engines.openvino import OPENVINO

        cpu = OPENVINO.estimate_memory(_request(model, on_cpu=True))
        card = OPENVINO.estimate_memory(_request(model, on_cpu=False))
        assert cpu.resident_footprint(unified=True) == UNIFIED + FILE_SIZE
        assert card.resident_footprint(unified=True) == UNIFIED + FILE_SIZE

    def test_the_alternate_build_on_a_card_is_charged_as_llama_cpp(self, model: Path) -> None:
        from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA

        request = _request(model, on_cpu=False)
        assert IK_LLAMA.estimate_memory(request) == LLAMA.estimate_memory(request)


COMPUTE_BUFFER = 4000
HOST_PART = 100
DEVICE_MODEL_INFLATION = 250
PROJECTOR_UNIFIED = 60
PROJECTOR_FILE = 80


class TestTheComputeBuffer:
    """A CPU load is sized at no offloaded layer, where the device holds only the buffer."""

    @pytest.fixture(autouse=True)
    def _parser_that_answers_by_layers(self, model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _parser(argv: list[str], _path: str) -> str:
            if "--raw" in argv:
                return json.dumps(RAW)
            on_card = argv[argv.index("--gpu-layers") + 1] != "0"
            projector = PROJECTOR_UNIFIED if "--mmproj-path" in argv else 0
            host = HOST_PART + projector + (DEVICE_MODEL_INFLATION if on_card else 0)
            device = COMPUTE_BUFFER + (FILE_SIZE if on_card else 0)
            return json.dumps(
                {
                    "metadata": {"fileSize": FILE_SIZE},
                    "estimate": {
                        "items": [
                            {
                                "ram": {"uma": host, "nonuma": HOST_PART},
                                "vrams": [{"uma": 0, "nonuma": device}],
                            }
                        ]
                    },
                }
            )

        monkeypatch.setattr(vram_mod, "_run_parser", _parser)

    @staticmethod
    def _pooled(path: Path, *, on_cpu: bool = True) -> MemoryRequest:
        return replace(_request(path, on_cpu=on_cpu), batch_size=16)

    def test_a_cpu_load_backs_the_buffer(self, model: Path) -> None:
        cpu = LLAMA.estimate_memory(self._pooled(model))
        assert cpu.resident_footprint(unified=True) == (
            HOST_PART + COMPUTE_BUFFER + FILE_SIZE + REPACKED
        )

    def test_the_alternate_build_backs_the_buffer_without_the_copy(self, model: Path) -> None:
        from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA

        cpu = IK_LLAMA.estimate_memory(self._pooled(model))
        assert cpu.resident_footprint(unified=True) == HOST_PART + COMPUTE_BUFFER + FILE_SIZE

    def test_a_cpu_chat_load_keeps_the_parser_figure(self, model: Path) -> None:
        chat = model_memory_mod.estimate_gguf(_request(model, on_cpu=True))
        assert chat.resident_footprint(unified=True) == (
            HOST_PART + DEVICE_MODEL_INFLATION + FILE_SIZE + REPACKED
        )

    def test_a_cpu_load_is_the_same_at_every_layer_count(self, model: Path) -> None:
        all_layers = LLAMA.estimate_memory(self._pooled(model))
        none = LLAMA.estimate_memory(replace(self._pooled(model), gpu_layers=0))
        assert all_layers.resident_footprint(unified=True) == none.resident_footprint(unified=True)

    def test_a_card_load_leaves_the_buffer_on_the_card(self, model: Path) -> None:
        card = LLAMA.estimate_memory(self._pooled(model, on_cpu=False))
        assert card.resident_footprint(unified=True) == (
            HOST_PART + DEVICE_MODEL_INFLATION + FILE_SIZE
        )
        assert card.device_bytes == COMPUTE_BUFFER + FILE_SIZE

    def test_a_projector_is_not_charged_through_the_buffer(
        self, model: Path, tmp_path: Path
    ) -> None:
        projector = tmp_path / "mmproj.gguf"
        projector.write_bytes(b"x" * PROJECTOR_FILE)
        plain = LLAMA.estimate_memory(self._pooled(model))
        vision = LLAMA.estimate_memory(replace(self._pooled(model), mmproj_path=projector))
        assert vision.resident_footprint(unified=True) - plain.resident_footprint(unified=True) == (
            PROJECTOR_UNIFIED + PROJECTOR_FILE
        )

    def test_a_cpu_only_host_embed_role_charges_the_buffer_once(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)
        charge = planning_mod._estimate_role(
            WorkerRole.EMBED, "ref", slots=1, engine=LLAMA, unified_budget=10**9
        ).est_vram_bytes
        assert charge == HOST_PART + COMPUTE_BUFFER + FILE_SIZE + REPACKED

    @pytest.mark.parametrize(
        ("mode", "charged"),
        [(RerankMode.CROSS_ENCODER, True), (RerankMode.LLM, False)],
        ids=["cross-encoder", "llm"],
    )
    def test_only_a_cross_encoder_rerank_is_charged_the_buffer(
        self, model: Path, monkeypatch: pytest.MonkeyPatch, mode: RerankMode, charged: bool
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)
        monkeypatch.setattr(planning_mod, "_role_rerank_mode", lambda _role, _meta: mode)
        charge = planning_mod._estimate_role(
            WorkerRole.RERANK, "ref", slots=1, engine=LLAMA, unified_budget=10**9
        ).est_vram_bytes
        with_buffer = HOST_PART + COMPUTE_BUFFER + FILE_SIZE + REPACKED
        assert (charge == with_buffer) is charged


class TestTheOnePoolCharge:
    @staticmethod
    def _charge(**kwargs: object) -> int:
        return planning_mod._estimate_role(
            WorkerRole.EMBED, "ref", slots=1, engine=LLAMA, **kwargs
        ).est_vram_bytes

    def test_a_cpu_host_charges_the_copy(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)
        assert self._charge(unified_budget=10**9) == UNIFIED + FILE_SIZE + REPACKED

    def test_a_unified_gpu_host_charges_no_copy(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.METAL)
        assert self._charge(unified_budget=10**9) == UNIFIED + FILE_SIZE

    def test_no_layer_on_a_card_and_no_budget_is_one_pool(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CUDA)
        assert self._charge(gpu_layers=0) == UNIFIED + FILE_SIZE + REPACKED

    def test_a_card_charges_its_own_figure(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CUDA)
        assert self._charge() == UNIFIED

    def test_the_slot_fit_counts_the_copy_on_a_cpu_host(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)

        def fit(budget: int) -> int:
            return planning_mod._fit_slots(
                4,
                WorkerRole.CHAT,
                model,
                16,
                mmproj_path=None,
                unified=True,
                budget=budget,
                engine=LLAMA,
            )

        # The parser prices every slot count alike, so only the copy moves the fit.
        assert fit(CPU_CHARGE) == 4
        assert fit(CPU_CHARGE - 1) == 1


class TestASharedFile:
    def test_two_roles_on_one_file_charge_it_once(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "embedding_model", "org/shared.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "org/shared.gguf")
        monkeypatch.setattr(
            planning_mod,
            "_estimate_role",
            lambda role, _ref, **_kw: ModelPlacementInput(role, 9000),
        )
        inputs, _refs, _reservation, _skipped = planning_mod._server_model_inputs(
            unified_budget=10**9, engines=engines_for_all_roles(LLAMA)
        )
        by_role = {item.role: item.est_vram_bytes for item in inputs}
        assert by_role == {WorkerRole.EMBED: 9000, WorkerRole.RERANK: 9000 - 4}

    def test_two_cpu_roles_on_one_file_charge_it_once_to_system_memory(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "embedding_model", "org/shared.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "org/shared.gguf")
        monkeypatch.setattr(
            planning_mod,
            "_estimate_cpu_role",
            lambda role, _ref, **_kw: ModelPlacementInput(role, 0, est_ram_bytes=9000),
        )
        inputs, _refs, _reservation, _skipped = planning_mod._server_model_inputs(
            cpu_roles=frozenset({WorkerRole.EMBED, WorkerRole.RERANK}),
            engines=engines_for_all_roles(LLAMA),
        )
        assert {item.role: item.est_ram_bytes for item in inputs} == {
            WorkerRole.EMBED: 9000,
            WorkerRole.RERANK: 9000 - 4,
        }

    def test_a_file_that_cannot_be_resolved_is_charged_as_estimated(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(_ref: str) -> Path:
            raise ProviderError("not installed")

        monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _missing)
        estimate = ModelPlacementInput(WorkerRole.EMBED, 9000)
        assert planning_mod._without_shared_file(estimate, "ref", set()) is estimate

    def test_a_card_charges_each_role_its_file(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_model", "")
        monkeypatch.setattr(cfg, "vision_model", "")
        monkeypatch.setattr(cfg, "embedding_model", "org/shared.gguf")
        monkeypatch.setattr(cfg, "reranker_model", "org/shared.gguf")
        monkeypatch.setattr(
            planning_mod,
            "_estimate_role",
            lambda role, _ref, **_kw: ModelPlacementInput(role, 9000),
        )
        inputs, _refs, _reservation, _skipped = planning_mod._server_model_inputs(
            unified_budget=None, engines=engines_for_all_roles(LLAMA)
        )
        assert [item.est_vram_bytes for item in inputs] == [9000, 9000]

    @pytest.mark.parametrize(
        ("estimate", "field"),
        [
            (ModelPlacementInput(WorkerRole.EMBED, 3), "est_vram_bytes"),
            (ModelPlacementInput(WorkerRole.EMBED, 0, est_ram_bytes=3), "est_ram_bytes"),
        ],
    )
    def test_the_second_role_never_charges_below_zero(
        self, model: Path, estimate: ModelPlacementInput, field: str
    ) -> None:
        charged: set[Path] = {model}
        # The 4-byte file is larger than the 3-byte estimate.
        assert getattr(planning_mod._without_shared_file(estimate, "ref", charged), field) == 0

    @pytest.mark.parametrize("field", ["est_vram_bytes", "est_ram_bytes"])
    def test_an_unknown_size_never_drops_below_the_reserve(self, model: Path, field: str) -> None:
        reserve = model_memory_mod.UNKNOWN_SIZE_RESERVE_BYTES
        estimate = ModelPlacementInput(
            WorkerRole.EMBED,
            reserve if field == "est_vram_bytes" else 0,
            est_ram_bytes=reserve if field == "est_ram_bytes" else 0,
            size_known=False,
        )
        # The 4-byte file is a sliver of the reserve, which is a stated charge, not the file.
        shared = planning_mod._without_shared_file(estimate, "ref", {model})
        assert getattr(shared, field) == reserve

    def test_an_unknown_size_over_the_reserve_loses_only_what_is_above_it(
        self, model: Path
    ) -> None:
        reserve = model_memory_mod.UNKNOWN_SIZE_RESERVE_BYTES
        estimate = ModelPlacementInput(WorkerRole.EMBED, reserve + 10, size_known=False)
        shared = planning_mod._without_shared_file(estimate, "ref", {model})
        assert shared.est_vram_bytes == reserve + 6


class _RecordingEngine:
    """The llama engine, noting every memory request the planner sends it."""

    def __init__(self) -> None:
        self.requests: list[MemoryRequest] = []

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        self.requests.append(request)
        return LLAMA.estimate_memory(request)

    def __getattr__(self, name: str) -> object:
        return getattr(LLAMA, name)


class TestTheRepackFlagReachesTheEstimate:
    @pytest.fixture(autouse=True)
    def _setup(self, model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(planning_mod, "plan_sizing_is_unified", lambda _e: True)
        monkeypatch.setattr(planning_mod, "_role_gpu_layers", lambda _r: 0)
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.METAL)
        self.model = model
        self.engine = _RecordingEngine()

    def _fit_chat(self) -> None:
        planning_mod.fit_chat_ctx(
            self.model,
            {},
            available_bytes=10**9,
            ctx_ceiling=64,
            engine=self.engine,
        )

    def _fit_slots(self, *, unified: bool) -> None:
        planning_mod._fit_slots(
            2,
            WorkerRole.CHAT,
            self.model,
            16,
            mmproj_path=None,
            unified=unified,
            budget=10**9,
            engine=self.engine,
        )

    def test_the_chat_ctx_fit_flags_no_layer_on_a_card_as_cpu(self) -> None:
        self._fit_chat()
        assert self.engine.requests
        assert all(request.on_cpu is True for request in self.engine.requests)

    def test_the_chat_ctx_fit_flags_no_cpu_when_the_pool_is_not_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "plan_sizing_is_unified", lambda _e: False)
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)
        self._fit_chat()
        assert self.engine.requests
        assert all(request.on_cpu is False for request in self.engine.requests)

    def test_the_slot_fit_flags_no_layer_on_a_card_as_cpu(self) -> None:
        self._fit_slots(unified=True)
        assert self.engine.requests
        assert all(request.on_cpu is True for request in self.engine.requests)

    def test_the_slot_fit_flags_no_cpu_when_the_pool_is_not_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CPU)
        self._fit_slots(unified=False)
        assert self.engine.requests
        assert all(request.on_cpu is False for request in self.engine.requests)

    def test_the_role_estimate_reads_the_role_layers_when_none_is_given(self) -> None:
        planning_mod._estimate_role(
            WorkerRole.EMBED, "ref", slots=1, engine=self.engine, unified_budget=10**9
        )
        assert [request.on_cpu for request in self.engine.requests] == [True]

    def _launch_with_flash_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "flash_attention", None)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        monkeypatch.setattr(planning_mod, "_fleet_backend", lambda _e: "Vulkan")

    def test_a_cpu_placed_chat_role_under_flash_auto_is_sent_as_flash_on_with_an_f16_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._launch_with_flash_auto(monkeypatch)
        self._fit_slots(unified=True)
        self._fit_chat()
        planning_mod._estimate_role(
            WorkerRole.CHAT, "ref", slots=1, engine=self.engine, unified_budget=10**9
        )
        assert len(self.engine.requests) >= 3
        assert all(request.on_cpu for request in self.engine.requests)
        assert all(request.flash_attn for request in self.engine.requests)
        assert {request.kv_cache_type for request in self.engine.requests} == {KvCacheType.F16}

    def test_a_cpu_placed_chat_role_under_flash_on_is_sent_with_the_configured_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "flash_attention", True)
        monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
        monkeypatch.setattr(planning_mod, "_fleet_backend", lambda _e: "CUDA")
        self._fit_slots(unified=True)
        self._fit_chat()
        planning_mod._estimate_role(
            WorkerRole.CHAT, "ref", slots=1, engine=self.engine, unified_budget=10**9
        )
        assert len(self.engine.requests) >= 3
        assert all(request.flash_attn for request in self.engine.requests)
        assert {request.kv_cache_type for request in self.engine.requests} == {KvCacheType.Q8_0}
        assert {request.gpu_layers for request in self.engine.requests} == {0}

    def test_a_chat_role_on_a_card_under_flash_auto_is_still_sent_as_flash_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._launch_with_flash_auto(monkeypatch)
        monkeypatch.setattr(planning_mod, "plan_sizing_is_unified", lambda _e: False)
        monkeypatch.setattr(planning_mod, "engine_backend", lambda _e: EngineBackend.CUDA)
        monkeypatch.setattr(planning_mod, "_role_gpu_layers", lambda _r: -1)
        self._fit_slots(unified=False)
        assert self.engine.requests
        assert not any(request.on_cpu for request in self.engine.requests)
        assert not any(request.flash_attn for request in self.engine.requests)

    def _only_for(self, role: WorkerRole, monkeypatch: pytest.MonkeyPatch) -> None:
        """Offload no layer for *role* and every layer for any other role."""
        monkeypatch.setattr(
            planning_mod, "_role_gpu_layers", lambda asked: 0 if asked is role else -1
        )

    def test_the_slot_fit_asks_for_the_layers_of_its_own_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._only_for(WorkerRole.CHAT, monkeypatch)
        self._fit_slots(unified=True)
        assert self.engine.requests
        assert all(request.gpu_layers == 0 for request in self.engine.requests)
        assert all(request.on_cpu is True for request in self.engine.requests)

    def test_the_chat_ctx_fit_asks_for_the_chat_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._only_for(WorkerRole.CHAT, monkeypatch)
        self._fit_chat()
        assert self.engine.requests
        assert all(request.on_cpu is True for request in self.engine.requests)

    def test_the_role_estimate_asks_for_the_layers_of_its_own_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._only_for(WorkerRole.EMBED, monkeypatch)
        planning_mod._estimate_role(
            WorkerRole.EMBED, "ref", slots=1, engine=self.engine, unified_budget=10**9
        )
        assert [request.on_cpu for request in self.engine.requests] == [True]


class TestTheRepackedBytesMemo:
    def test_the_parser_is_asked_about_the_model_file(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argvs: list[list[str]] = []

        def _parser(argv: list[str], _path: str) -> str:
            argvs.append(argv)
            return json.dumps(RAW)

        monkeypatch.setattr(vram_mod, "_run_parser", _parser)
        vram_mod.repacked_weight_bytes(model)
        assert argvs == [[str(Path("/fake/gguf-parser")), "--path", str(model), "--raw"]]

    def test_a_rewritten_file_is_sized_again(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert vram_mod.repacked_weight_bytes(model) == REPACKED
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps({"tensorInfos": []}))
        model.write_bytes(b"GGUF2")
        assert vram_mod.repacked_weight_bytes(model) == 0

    def test_another_engine_build_is_sized_again(
        self, model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "build-a")
        assert vram_mod.repacked_weight_bytes(model) == REPACKED
        monkeypatch.setattr(vram_mod, "_run_parser", lambda _a, _p: json.dumps({"tensorInfos": []}))
        assert vram_mod.repacked_weight_bytes(model) == REPACKED
        monkeypatch.setattr(vram_mod, "engine_build_identity", lambda: "build-b")
        assert vram_mod.repacked_weight_bytes(model) == 0
