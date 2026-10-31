"""Every engine call site uses the engine it is given and keeps what the engine returns.

A site that calls the llama adapter instead of its engine argument fails here. The spy engine
differs from llama.cpp on every member, so each test asserts a value only the spy can produce.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from lilbee.app.engine_choice import BookReading, EngineBook, resolve_engines
from lilbee.catalog.compat import classify
from lilbee.catalog.types import ModelCompat
from lilbee.core.config import cfg
from lilbee.core.health_warnings import WarningCode
from lilbee.data.ingest import adaptive
from lilbee.providers import engine_params
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import ctx as fleet_ctx
from lilbee.providers.fleet import planning, replicas
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineName, RequestProfile
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import EngineBackend, WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from tests._fleet_engines import engines_for_all_roles
from tests.conformance.fake_adapter import FakeAdapter

SPY_META = {"architecture": "spyarch", "context_length": "4096", "expert_count": "8"}
SPY_ENV = {"SPY_RUNTIME": "1"}
SPY_EXTRAS = {"checkEndpoint": "/spy-live"}
SPY_DEVICE = FleetDevice("CUDA", 3, "spy-gpu", 8 << 30, 8 << 30)
BYTES_PER_CTX = 1024


def _mixed_book(spy: Any) -> dict[WorkerRole, Any]:
    """Chat on the spy engine and every other role on llama.cpp: two roles on two adapters."""
    return {**engines_for_all_roles(LLAMA), WorkerRole.CHAT: spy}


class SpyEngine(FakeAdapter):
    """An engine that answers every member with a value llama.cpp never returns."""

    name = "spy-engine"
    architectures = frozenset({"spyarch"})
    memory_kind = MemoryKind.FIXED_RESERVE

    def __init__(self, binary: Path, *, devices: tuple[FleetDevice, ...] = (SPY_DEVICE,)) -> None:
        self._binary = binary
        self._devices = list(devices)
        self.probed: list[Path] = []
        self.sized: list[MemoryRequest] = []
        self.meta_paths: list[Path] = []
        self.readback_asked = 0

    def binary(self) -> Path:
        return self._binary

    def discover_devices(self, binary: Path) -> DeviceProbe:
        self.probed.append(binary)
        return DeviceProbe(
            devices=list(self._devices),
            output="",
            spoke_protocol=False,
            backend=EngineBackend.CUDA,
        )

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        self.sized.append(request)
        size = request.ctx * request.slots * BYTES_PER_CTX
        split = tuple(size // 2 for _ in request.tensor_split) or (size,)
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=size,
            host_bytes=size + 7,
            shared_bytes=size,
            per_device=split,
        )

    def model_meta(self, path: Path) -> dict[str, str] | None:
        self.meta_paths.append(path)
        return dict(SPY_META)

    def runtime_env(self) -> Mapping[str, str]:
        return SPY_ENV

    def swap_extras(self, launch: Any) -> Mapping[str, Any]:
        return SPY_EXTRAS

    def check_readback(self, binary: Path) -> bool:
        return True

    def supports_fit_target(self, binary: Path) -> bool:
        return True

    def supports_shared_kv(self, binary: Path) -> bool:
        return True

    def launch_argv(self, spec: Any, request: Any) -> list[str]:
        return [
            str(request.binary),
            "--spy-weights",
            str(request.model_path),
            f"--spy-window={request.ctx_per_slot * request.slots}",
            f"--spy-readback={request.memory_endpoint}",
            f"--spy-fit={request.fit_target_mib}",
            f"--spy-kv={request.kv_unified_per_slot}",
            f"--spy-moe={request.cpu_moe}",
            *spec.extra_args,
        ]


class BinarylessSpy(SpyEngine):
    """A spy whose binary is missing."""

    def binary(self) -> Path:
        raise ProviderError("spy binary missing", provider="spy", kind=ProviderErrorKind.NOT_FOUND)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"x" * 4096)
    return path


@pytest.fixture
def spy(tmp_path: Path, model_file: Path, monkeypatch: pytest.MonkeyPatch) -> SpyEngine:
    binary = tmp_path / "spy-engine"
    binary.write_bytes(b"spy")
    planning.clear_read_device_cache()
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model_file)
    monkeypatch.setattr(cfg, "cpu_moe", True)
    monkeypatch.setattr(cfg, "n_cpu_moe", None)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    yield SpyEngine(binary)
    planning.clear_read_device_cache()


class FlakySpy(SpyEngine):
    """A spy whose first probe lists no device, as a transient init error does."""

    def discover_devices(self, binary: Path) -> DeviceProbe:
        probe = super().discover_devices(binary)
        if len(self.probed) == 1:
            return DeviceProbe(devices=[], output="", spoke_protocol=True)
        return probe


def _binaryless(spy: SpyEngine) -> BinarylessSpy:
    return BinarylessSpy(spy._binary)


class TestCatalogCompat:
    @pytest.mark.parametrize(
        ("architecture", "verdict"),
        [("spyarch", ModelCompat.SUPPORTED), ("llama", ModelCompat.UNSUPPORTED)],
    )
    def test_the_engine_decides_the_verdict(
        self, spy: SpyEngine, architecture: str, verdict: ModelCompat
    ) -> None:
        assert classify(architecture, engine=spy) is verdict


class TestDeviceReads:
    def test_enumerate_fleet_devices_reads_the_engine_devices(self, spy: SpyEngine) -> None:
        assert adaptive.enumerate_fleet_devices(spy) == [SPY_DEVICE]
        assert spy.probed == [spy.binary()]

    def test_gpu_device_count_counts_the_engine_devices(self, spy: SpyEngine) -> None:
        replicas.gpu_device_count.cache_clear()
        two = SpyEngine(
            spy.binary(), devices=(SPY_DEVICE, dataclasses.replace(SPY_DEVICE, index=4))
        )
        assert replicas.gpu_device_count(two) == 2

    def test_gpu_device_count_is_one_without_an_engine_binary(self, spy: SpyEngine) -> None:
        replicas.gpu_device_count.cache_clear()
        assert replicas.gpu_device_count(_binaryless(spy)) == 1

    def test_probed_devices_and_backend_come_from_the_engine(self, spy: SpyEngine) -> None:
        assert planning.probed_devices(spy) == (SPY_DEVICE,)
        assert planning.engine_backend(spy) is EngineBackend.CUDA
        assert spy.probed == [spy.binary()]

    def test_reads_degrade_when_the_engine_has_no_binary(self, spy: SpyEngine) -> None:
        gone = _binaryless(spy)
        assert planning.probed_devices(gone) == ()
        assert planning.engine_backend(gone) is EngineBackend.UNKNOWN
        assert planning.placeable_total_vram(gone) == 0
        assert planning._live_sizing_devices(gone) == []
        assert planning._engine_binary(gone) is None
        assert planning._engine_identity(gone) == planning._NO_ENGINE_BINARY

    def test_sizing_reads_use_the_engine_devices(self, spy: SpyEngine) -> None:
        assert planning.placeable_total_vram(spy) == SPY_DEVICE.total_bytes
        assert planning._live_sizing_devices(spy) == [SPY_DEVICE]
        assert planning._engine_binary(spy) == spy.binary()

    def test_the_identity_is_the_engine_binary(self, spy: SpyEngine) -> None:
        assert planning._engine_identity(spy).startswith(f"{spy.binary()}@")

    def test_probe_engine_devices_reads_the_engine(self, spy: SpyEngine) -> None:
        reading = planning._probe_engine_devices(spy)
        assert reading.devices == [SPY_DEVICE]
        assert reading.backend is EngineBackend.CUDA
        assert spy.probed[0] == spy.binary()

    def test_an_empty_answer_is_asked_again_of_the_same_engine(
        self, spy: SpyEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lilbee.providers.fleet.gpu_hardware as gpu_hardware

        monkeypatch.setattr(gpu_hardware, "installed_gpu_vendor_ids", lambda: {"10de"})
        monkeypatch.setattr(planning.time, "sleep", lambda seconds: None)
        flaky = FlakySpy(spy._binary)
        assert planning.resolve_devices(flaky.binary(), flaky) == [SPY_DEVICE]
        assert flaky.probed == [flaky.binary(), flaky.binary()]

    def test_resolve_devices_probes_the_engine(self, spy: SpyEngine) -> None:
        assert planning.resolve_devices(spy.binary(), spy) == [SPY_DEVICE]

    def test_resolve_placement_plan_reports_the_engine_devices(self, spy: SpyEngine) -> None:
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(spy))
        assert resolved.engine_backend is EngineBackend.CUDA
        assert [d.name for d in resolved.devices] == ["spy-gpu"]


class TestPlansAskTheEngine:
    def test_plan_all_launches_probes_the_engine(self, spy: SpyEngine) -> None:
        planning.plan_all_launches(engines=engines_for_all_roles(spy))
        assert spy.probed
        assert set(spy.probed) == {spy.binary()}

    def test_plan_all_launches_needs_the_engine_binary(self, spy: SpyEngine) -> None:
        with pytest.raises(ProviderError, match="spy binary missing"):
            planning.plan_all_launches(engines=engines_for_all_roles(_binaryless(spy)))

    def test_plan_launches_builds_the_engine_argv(
        self, spy: SpyEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "chat_model", "org/repo/spy.gguf")
        plan = planning.plan_launches(
            (WorkerRole.CHAT,),
            spy.binary(),
            {SPY_DEVICE.index: SPY_DEVICE},
            [SPY_DEVICE],
            engines=engines_for_all_roles(spy),
        )
        assert plan.launches
        assert all(launch.argv[0] == str(spy.binary()) for launch in plan.launches)
        assert any("--spy-weights" in launch.argv for launch in plan.launches)


class TestMemoryEstimates:
    def test_fit_single_ctx_takes_the_engine_footprint(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        ctx = fleet_ctx.fit_single_ctx(
            model_file,
            engine=spy,
            meta=dict(SPY_META),
            slots=1,
            available_bytes=2048 * BYTES_PER_CTX,
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=cfg.kv_cache_type,
            unified=False,
            ctx_ceiling=4096,
            expert_offload=(),
        )
        assert spy.sized
        assert ctx * BYTES_PER_CTX <= 2048 * BYTES_PER_CTX < (ctx + 2048) * BYTES_PER_CTX

    def test_fit_split_ctx_takes_the_engine_per_device_shares(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        free = 1024 * BYTES_PER_CTX
        ctx = fleet_ctx.fit_split_ctx(
            model_file,
            engine=spy,
            meta=dict(SPY_META),
            slots=1,
            ratio=(1, 1),
            per_device_free_bytes=[free, free],
            gpu_layers=-1,
            flash_attn=False,
            kv_cache_type=cfg.kv_cache_type,
            ctx_ceiling=4096,
            expert_offload=(),
        )
        assert spy.sized
        assert all(r.tensor_split == (1, 1) for r in spy.sized)
        assert ctx * BYTES_PER_CTX // 2 <= free

    def test_fit_slots_takes_the_engine_footprint(self, spy: SpyEngine, model_file: Path) -> None:
        slots = planning._fit_slots(
            8,
            WorkerRole.EMBED,
            model_file,
            1000,
            mmproj_path=None,
            unified=False,
            budget=3 * 1000 * BYTES_PER_CTX,
            engine=spy,
        )
        assert slots == 3

    def test_chat_host_estimate_is_the_engine_host_bytes(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        assert planning._chat_host_estimate(model_file, 100, 2, spy) == 100 * 2 * BYTES_PER_CTX + 7

    def test_declared_reserve_is_the_engine_figure(self, spy: SpyEngine, model_file: Path) -> None:
        request = MemoryRequest(
            model_path=model_file,
            ctx=10,
            slots=1,
            gpu_layers=0,
            flash_attn=False,
            kv_cache_type=cfg.kv_cache_type,
        )
        assert planning._declared_reserve(spy, request, unified=False) == 10 * BYTES_PER_CTX

    @pytest.mark.parametrize(
        "failure", [ProviderError("no estimate", provider="spy"), OSError("gone")]
    )
    def test_declared_reserve_is_zero_when_the_engine_cannot_estimate(
        self,
        spy: SpyEngine,
        model_file: Path,
        failure: Exception,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fail(request: MemoryRequest) -> MemoryEstimate:
            raise failure

        monkeypatch.setattr(spy, "estimate_memory", _fail)
        request = MemoryRequest(
            model_path=model_file,
            ctx=10,
            slots=1,
            gpu_layers=0,
            flash_attn=False,
            kv_cache_type=cfg.kv_cache_type,
        )
        assert planning._declared_reserve(spy, request, unified=False) == 0

    def test_estimate_role_charges_the_engine_footprint_at_the_engine_window(
        self, spy: SpyEngine
    ) -> None:
        placed = planning._estimate_role(WorkerRole.EMBED, "embed-ref", engine=spy, slots=2)
        assert spy.meta_paths
        last = spy.sized[-1]
        assert last.ctx == engine_params.resolve_embed_ctx(SPY_META, last.model_path)
        assert placed.est_vram_bytes == last.ctx * 2 * BYTES_PER_CTX

    def test_peak_estimator_returns_the_engine_per_device_shares(self, spy: SpyEngine) -> None:
        estimate = planning._peak_estimator(
            {WorkerRole.EMBED: "embed-ref"}, engines_for_all_roles(spy)
        )
        shares = estimate(WorkerRole.EMBED, (1, 1))
        assert spy.sized[-1].tensor_split == (1, 1)
        assert shares == spy.estimate_memory(spy.sized[-1]).per_device
        assert spy.sized[-1].expert_offload  # the engine's metadata names experts

    def test_peak_estimator_sizes_the_window_from_the_engine_metadata(self, spy: SpyEngine) -> None:
        planning._peak_estimator({WorkerRole.EMBED: "embed-ref"}, engines_for_all_roles(spy))(
            WorkerRole.EMBED, (1, 1)
        )
        model_path = spy.sized[-1].model_path
        from_header = engine_params.resolve_embed_ctx(SPY_META, model_path)
        assert spy.sized[-1].ctx == planning.apply_ctx_downshift(WorkerRole.EMBED, from_header)
        assert from_header != engine_params.resolve_embed_ctx(None, model_path)

    def test_split_ctx_objective_targets_the_engine_window(self, spy: SpyEngine) -> None:
        fitter, target = planning._chat_split_ctx_objective(
            {WorkerRole.CHAT: "chat-ref"}, engines_for_all_roles(spy)
        )
        assert fitter is not None
        assert spy.meta_paths
        assert target == planning._placement_estimate_ctx(
            WorkerRole.CHAT, spy.meta_paths[0], dict(SPY_META)
        )
        assert target == 4096


class TestModelMeta:
    def test_expert_offload_follows_the_engine_metadata(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        assert planning._role_expert_offload(model_file, engine=spy)
        assert spy.meta_paths == [model_file]

    def test_ref_is_moe_follows_the_engine_metadata(self, spy: SpyEngine) -> None:
        assert planning._ref_is_moe("any-ref", engine=spy) is True

    def test_embed_token_cap_follows_the_engine_window(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        cap = planning.planned_embed_token_cap("embed-ref", engine=spy)
        unread = engine_params.embed_token_cap(
            planning.apply_ctx_downshift(
                WorkerRole.EMBED, planning._aux_role_ctx(WorkerRole.EMBED, model_file, None)
            )
        )
        expected = engine_params.embed_token_cap(
            planning.apply_ctx_downshift(
                WorkerRole.EMBED,
                planning._aux_role_ctx(WorkerRole.EMBED, model_file, dict(SPY_META)),
            )
        )
        assert cap == expected
        assert cap != unread

    def test_fallback_floor_reads_the_engine_metadata(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        floor = planning._fallback_floor_for(WorkerRole.CHAT, "chat-ref", 1 << 30, engine=spy)
        by_hand = planning._analytic_footprint_floor(
            1 << 30,
            role=WorkerRole.CHAT,
            meta=dict(SPY_META),
            ctx=planning._placement_estimate_ctx(WorkerRole.CHAT, model_file, dict(SPY_META)),
            slots=planning._placement_estimate_slots(WorkerRole.CHAT, dict(SPY_META)),
            engine=spy,
        )
        assert spy.meta_paths == [model_file]
        assert floor == by_hand


class TestEngineCapabilities:
    def test_fit_target_follows_the_engine(self, spy: SpyEngine) -> None:
        card = (SPY_DEVICE,)
        assert planning._chat_fit_target_mib(spy.binary(), card, {}, engine=spy)
        assert not planning._chat_fit_target_mib(
            spy.binary(), card, {}, engine=_NoFitSpy(spy._binary)
        )

    def test_chat_offload_fit_follows_the_engine(
        self, spy: SpyEngine, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "cpu_moe", False)
        monkeypatch.setattr(
            planning, "_role_gpu_layers", lambda role: engine_params.N_GPU_LAYERS_AUTO
        )
        assert planning._engine_fits_chat_offload(model_file, spy.binary(), spy) is True
        assert (
            planning._engine_fits_chat_offload(model_file, spy.binary(), _NoFitSpy(spy._binary))
            is False
        )

    def test_shared_kv_follows_the_engine(self, spy: SpyEngine) -> None:
        assert planning._engine_shares_kv(spy.binary(), engine=spy) is True
        assert planning._engine_shares_kv(spy.binary(), engine=_NoFitSpy(spy._binary)) is False


class _NoFitSpy(SpyEngine):
    """A spy that reports none of the optional capabilities."""

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def check_readback(self, binary: Path) -> bool:
        return False


class TestLaunchBuild:
    @pytest.mark.parametrize("role", [WorkerRole.CHAT, WorkerRole.EMBED])
    def test_launch_is_built_from_the_engine_answers(
        self, spy: SpyEngine, model_file: Path, role: WorkerRole, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "cpu_moe", True)
        launch = planning.build_single_role_launch(role, model_file, engine=spy)
        argv = launch.argv
        assert argv[0] == str(spy.binary())  # binary()
        assert "--chat-mode" in argv  # role_spec(): the fake spec adds its flag
        assert "--spy-readback=True" in argv  # check_readback()
        assert "--spy-moe=True" in argv  # model_meta(): the experts in its header
        assert launch.env_overrides["SPY_RUNTIME"] == "1"  # runtime_env()
        assert launch.swap_extras == SPY_EXTRAS  # swap_extras()
        assert spy.meta_paths and set(spy.meta_paths) == {model_file}

    def test_chat_launch_asks_the_engine_for_fit_and_shared_kv(
        self, spy: SpyEngine, model_file: Path
    ) -> None:
        launch = planning.build_single_role_launch(WorkerRole.CHAT, model_file, engine=spy)
        assert any(a.startswith("--spy-fit=(") and a != "--spy-fit=()" for a in launch.argv)
        assert f"--spy-kv={launch.ctx}" in launch.argv

    def test_the_launch_needs_the_engine_binary(self, spy: SpyEngine, model_file: Path) -> None:
        with pytest.raises(ProviderError, match="spy binary missing"):
            planning.build_single_role_launch(WorkerRole.CHAT, model_file, engine=_binaryless(spy))


class TestProvider:
    def test_unusable_reason_names_the_engine_binary_problem(
        self, spy: SpyEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            EngineBook, "read", lambda self: BookReading(engines_for_all_roles(spy), {})
        )
        assert FleetProvider()._unusable_engine_reason() is None
        monkeypatch.setattr(
            EngineBook,
            "read",
            lambda self: BookReading(engines_for_all_roles(_binaryless(spy)), {}),
        )
        reason = FleetProvider()._unusable_engine_reason()
        assert reason is not None
        assert "spy binary missing" in reason

    def test_clients_carry_the_engine_request_profile(
        self, spy: SpyEngine, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        launch = planning.build_single_role_launch(WorkerRole.EMBED, model_file, engine=spy)
        monkeypatch.setattr(
            EngineBook, "read", lambda self: BookReading(engines_for_all_roles(spy), {})
        )
        provider = FleetProvider()
        swap = SimpleNamespace(endpoint=lambda: "http://127.0.0.1:1", log_path=tmp_path / "log")
        with provider._lock:
            provider._adopt_group("embed", swap, [launch])  # type: ignore[arg-type]
        client = provider._clients[WorkerRole.EMBED][0]
        assert client._profile.label == "fake-engine"
        assert client._profile.reasoning_field == "thoughts"

    def _record_engine_use(
        self, monkeypatch: pytest.MonkeyPatch, spy: SpyEngine, model_file: Path, tmp_path: Path
    ) -> dict[str, list[Any]]:
        """Stand in for the planner and the swap manager; record the engine each one gets."""
        import lilbee.providers.fleet.provider as prov_mod

        seen: dict[str, list[Any]] = {"probe": [], "plan": [], "refresh": [], "swap": []}
        launch = planning.build_single_role_launch(WorkerRole.EMBED, model_file, engine=spy)
        monkeypatch.setattr(EngineBook, "read", lambda self: BookReading(_mixed_book(spy), {}))
        monkeypatch.setattr(planning, "snapshot_engine", lambda engines: spy)

        class _Swap:
            bound = False
            log_path = tmp_path / "log"

            def __init__(self, work_dir: Path, group: object) -> None:
                pass

            def start(self, launches: list[Any], *, engines: Any, **kwargs: object) -> None:
                seen["swap"].append(engines)

            def endpoint(self) -> str:
                return "http://127.0.0.1:1"

            def shutdown(self) -> None:
                return None

            def health_warnings(self) -> list[Any]:
                return []

        monkeypatch.setattr(prov_mod, "SwapManager", _Swap)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda *args, **kwargs: None)
        monkeypatch.setattr(prov_mod, "_engine_lifetime", lambda data_dir: None)
        monkeypatch.setattr(prov_mod, "_warm_ttl_seconds", lambda *args, **kwargs: 0)
        monkeypatch.setattr(
            planning, "capture_plan_probe", lambda engine: seen["probe"].append(engine)
        )
        monkeypatch.setattr(
            planning, "refresh_plan_devices", lambda engine: seen["refresh"].append(engine)
        )

        def _plan(*, engines: Any) -> planning.FleetPlan:
            seen["plan"].append(engines)
            return planning.FleetPlan((launch,))

        monkeypatch.setattr(planning, "plan_all_launches", _plan)
        return seen

    def test_build_plans_and_spawns_with_the_provider_engine(
        self, spy: SpyEngine, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._record_engine_use(monkeypatch, spy, model_file, tmp_path)
        provider = FleetProvider()
        assert provider._plan_and_spawn(tmp_path)
        assert seen["probe"] == [spy]
        assert seen["plan"] == [_mixed_book(spy)]
        assert seen["swap"] == [_mixed_book(spy)]

    def test_reload_replans_and_respawns_with_the_provider_engine(
        self, spy: SpyEngine, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = self._record_engine_use(monkeypatch, spy, model_file, tmp_path)
        provider = FleetProvider()
        monkeypatch.setattr(provider, "_reload_dir", lambda: tmp_path)
        monkeypatch.setattr(provider, "_preload_restarted", lambda restarted: None)
        provider._reload_pass()
        assert seen["refresh"] == [spy]
        assert seen["probe"] == [spy]  # nothing was running, so the reload re-captures the box
        assert seen["plan"] == [_mixed_book(spy)]
        assert seen["swap"] == [_mixed_book(spy)]

    def _no_engine_provider(
        self, monkeypatch: pytest.MonkeyPatch, spy: SpyEngine, model_file: Path
    ) -> RoutingProvider:
        """The routing provider over a fleet whose chat model no installed engine loads."""
        monkeypatch.setattr(cfg, "chat_model", "org/repo#mlx")
        notice = resolve_engines(cfg).choices[WorkerRole.CHAT].notice or ""
        launch = planning.build_single_role_launch(WorkerRole.EMBED, model_file, engine=spy)
        monkeypatch.setattr(
            EngineBook,
            "read",
            lambda self: BookReading({WorkerRole.EMBED: spy}, {WorkerRole.CHAT: notice}),
        )
        monkeypatch.setattr(
            planning,
            "plan_all_launches",
            lambda *, engines: planning.FleetPlan(
                (launch,), skipped_no_engine={WorkerRole.CHAT: "org/repo#mlx"}
            ),
        )
        routing = RoutingProvider()
        routing._local = FleetProvider()
        return routing

    def _no_engine_warnings(self, routing: RoutingProvider) -> list[tuple[WarningCode, str]]:
        """Every warning, whatever its code, that speaks of the chat model with no engine."""
        return [
            (w.code, w.message) for w in routing.health_warnings() if "org/repo#mlx" in w.message
        ]

    def test_build_gives_one_warning_for_a_role_with_no_engine(
        self, spy: SpyEngine, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._record_engine_use(monkeypatch, spy, model_file, tmp_path)
        routing = self._no_engine_provider(monkeypatch, spy, model_file)
        assert isinstance(routing._local, FleetProvider)
        assert routing._local._plan_and_spawn(tmp_path)
        warnings = self._no_engine_warnings(routing)
        assert len(warnings) == 1
        code, message = warnings[0]
        assert code is WarningCode.ENGINE_FALLBACK
        assert message.startswith("For chat: No installed engine loads the mlx format")
        assert "Run lilbee engine list" in message

    def test_reload_gives_one_warning_for_a_role_with_no_engine(
        self, spy: SpyEngine, model_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._record_engine_use(monkeypatch, spy, model_file, tmp_path)
        routing = self._no_engine_provider(monkeypatch, spy, model_file)
        provider = routing._local
        assert isinstance(provider, FleetProvider)
        monkeypatch.setattr(provider, "_reload_dir", lambda: tmp_path)
        monkeypatch.setattr(provider, "_preload_restarted", lambda restarted: None)
        provider._reload_pass()
        warnings = self._no_engine_warnings(routing)
        assert [code for code, _ in warnings] == [WarningCode.ENGINE_FALLBACK]


class TestSelfCheck:
    def test_the_self_check_client_carries_the_engine_request_profile(
        self, spy: SpyEngine, model_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import lilbee.providers.fleet.engines.choice as choice
        import lilbee.providers.fleet.swap_manager as swap_manager
        from lilbee.cli.commands import setup as setup_cmd

        class _Swap:
            def __init__(self, work_dir: Path, group: object) -> None:
                self.engines: Any = None

            def start(self, launches: list[Any], *, engines: Any) -> None:
                self.engines = engines

            def endpoint(self) -> str:
                return "http://127.0.0.1:1"

        monkeypatch.setattr(spy, "formats", frozenset({"gguf"}))
        monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, spy)  # type: ignore[arg-type]
        monkeypatch.setattr(swap_manager, "SwapManager", _Swap)
        swap, client, work_dir = setup_cmd._self_check_server(WorkerRole.CHAT, model_file)
        try:
            assert swap.engines[WorkerRole.CHAT] is spy
            assert client._profile.label == "fake-engine"
            assert client._profile.reasoning_field == "thoughts"
        finally:
            work_dir.rmdir()


class TestClientWording:
    @pytest.fixture
    def client(self, spy: SpyEngine) -> LlamaServerClient:
        profile = dataclasses.replace(
            spy.request_profile(WorkerRole.CHAT),
            label="spy-server",
            context_overflow=re.compile("SPY_CTX_FULL"),
            batch_overflow=re.compile("SPY_BATCH_FULL"),
        )
        return LlamaServerClient("http://127.0.0.1:1", "m", profile=profile)

    @staticmethod
    def _raise(client: LlamaServerClient, code: int, body: str) -> ProviderError:
        response = httpx.Response(code, text=body, request=httpx.Request("POST", "http://x/"))
        with pytest.raises(ProviderError) as caught:
            client._raise_for_status(response)
        return caught.value

    def test_overflow_is_recognised_by_the_engine_pattern(self, client: LlamaServerClient) -> None:
        error = self._raise(client, 400, "SPY_CTX_FULL")
        assert error.provider == "spy-server"

    def test_the_llama_overflow_body_is_not_an_overflow(self, client: LlamaServerClient) -> None:
        overflow = self._raise(client, 400, "SPY_CTX_FULL")
        llama_body = self._raise(client, 400, "exceed_context_size")
        assert llama_body.kind is not overflow.kind

    def test_the_message_names_the_engine_server(self, client: LlamaServerClient) -> None:
        error = self._raise(client, 500, "boom")
        assert str(error).startswith("spy-server returned HTTP 500")
        assert "llama-server" not in str(error)


def test_the_spy_differs_from_llama_on_every_member(spy: SpyEngine) -> None:
    profile: RequestProfile = spy.request_profile(WorkerRole.CHAT)
    assert profile.label != LLAMA.request_profile(WorkerRole.CHAT).label
    assert spy.runtime_env() != LLAMA.runtime_env()
    assert spy.swap_extras(None) != LLAMA.swap_extras(None)  # type: ignore[arg-type]
    assert not LLAMA.supports_architecture("spyarch")
