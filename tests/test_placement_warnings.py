"""Placement divergence reaches /api/health warnings when the engine lands off plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests._engine_logs import enrol_launch


class TestPlacementWarnings:
    def test_health_carries_divergence_from_a_swap(self, monkeypatch, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(planning_mod, "planned_embed_token_cap", lambda _r, *, engine: 2048)
        placed = HealthWarning(
            code=WarningCode.PLACEMENT_DIVERGED,
            message="The chat model did not land where it was planned.",
            remedy="Free up GPU memory or use a smaller model.",
        )
        swap = MagicMock()
        swap.health_warnings.return_value = [placed]
        provider = FleetProvider()
        provider._swaps = {SwapGroup.CHAT: swap}
        codes = [w.code for w in provider.health_warnings()]
        assert codes == [WarningCode.PLACEMENT_DIVERGED]

    def test_swap_records_divergence_from_the_engine_log(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet import swap_manager as sm
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.readback import MIB, engine_log_path

        engine_log_path(tmp_path / "logs", "chat-0").parent.mkdir(parents=True, exist_ok=True)
        engine_log_path(tmp_path / "logs", "chat-0").write_text(
            "load_tensors:        CUDA0 model buffer size =  6000.00 MiB\n"
            "load_model: initializing slots\n",
            encoding="utf-8",
        )
        manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
        manager._log_path = tmp_path / "logs" / "llama-swap-chat.log"
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server"],
            env_overrides={},
            model="chat-model",
            est_vram_bytes=4000 * MIB,
        )
        enrol_launch(manager, launch)
        manager._check_estimates({"chat-0"})
        codes = [w.code for w in manager.health_warnings()]
        assert codes == [WarningCode.PLACEMENT_DIVERGED]

    def test_swap_records_divergence_from_the_memory_endpoint(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet import swap_manager as sm
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        GIB = 1024**3

        class _Resp:
            status_code = 200

            def json(self) -> dict:
                return {"data": [{"name": "CUDA0", "model": 8 * GIB}]}

        class _Client:
            def get(self, url: str, timeout: float | None = None) -> _Resp:
                return _Resp()

        monkeypatch.setattr(sm, "_probe_client", lambda: _Client())
        manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server", MEMORY_FLAG],
            env_overrides={},
            model="chat-model",
            est_vram_bytes=4 * GIB,
            est_vram_by_device={"CUDA0": 4 * GIB},
        )
        enrol_launch(manager, launch)
        manager._member_port_by_model = {launch.model_id: 5901}
        manager._check_estimates({launch.model_id})
        codes = [w.code for w in manager.health_warnings()]
        assert codes == [WarningCode.PLACEMENT_DIVERGED]

    def test_routing_reports_nothing_with_no_local_engine(self, monkeypatch) -> None:
        from lilbee.providers.routing_provider import RoutingProvider

        monkeypatch.setattr(cfg, "embedding_model", "openai/text-embedding-3-small")
        routing = RoutingProvider()
        assert routing._local is None
        assert routing.health_warnings() == []


class TestHostSpillInHealth:
    """Chat weights left in system memory reach the health warnings, and clear with the engine."""

    _LOG = (
        "load_tensors:   CPU_Mapped model buffer size =  6000.00 MiB\n"
        "load_tensors:        CUDA0 model buffer size =  4000.00 MiB\n"
        "load_model: initializing slots\n"
    )

    def _manager(self, tmp_path: Path, log_text: str):
        from lilbee.providers.fleet import swap_manager as sm
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.readback import MIB, engine_log_path

        engine_log_path(tmp_path / "logs", "chat-0").parent.mkdir(parents=True, exist_ok=True)
        engine_log_path(tmp_path / "logs", "chat-0").write_text(log_text, encoding="utf-8")
        manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
        manager._log_path = tmp_path / "logs" / "llama-swap-chat.log"
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server"],
            env_overrides={},
            model="chat-model",
            est_vram_bytes=4000 * MIB,
            est_host_bytes=100 * MIB,
        )
        enrol_launch(manager, launch)
        return manager

    def test_a_spill_in_the_engine_log_is_a_health_warning(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import WarningCode

        manager = self._manager(tmp_path, self._LOG)
        manager._check_estimates({"chat-0"})
        spills = [w for w in manager.health_warnings() if w.code is WarningCode.HOST_SPILL]
        assert len(spills) == 1
        assert "does not fit in GPU memory" in spills[0].message
        assert spills[0].remedy == "Free GPU memory or use a smaller model."

    def test_a_reload_without_a_spill_clears_it(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet.readback import engine_log_path

        manager = self._manager(tmp_path, self._LOG)
        manager._check_estimates({"chat-0"})
        engine_log_path(tmp_path / "logs", "chat-0").write_text(
            "load_tensors:        CUDA0 model buffer size =  4000.00 MiB\n"
            "load_model: initializing slots\n",
            encoding="utf-8",
        )
        manager._check_estimates(set())
        manager._check_estimates({"chat-0"})
        assert WarningCode.HOST_SPILL not in [w.code for w in manager.health_warnings()]

    def test_a_reload_with_a_spill_reads_the_new_load(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet.readback import engine_log_path

        clean = (
            "load_tensors:        CUDA0 model buffer size =  4000.00 MiB\n"
            "load_model: initializing slots\n"
        )
        manager = self._manager(tmp_path, clean)
        manager._check_estimates({"chat-0"})
        assert manager.health_warnings() == []
        manager._check_estimates(set())
        engine_log_path(tmp_path / "logs", "chat-0").write_text(self._LOG, encoding="utf-8")
        manager._check_estimates({"chat-0"})
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]

    def test_an_unload_clears_the_placement_divergence(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        manager = self._manager(tmp_path, self._LOG)
        divergence = HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off plan")
        manager._estimate_checked.add("chat-0")
        manager._record_findings("chat-0", divergence, [])
        assert manager.health_warnings() == [divergence]
        manager._check_estimates(set())
        assert manager.health_warnings() == []

    def test_an_unload_clears_the_spill(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import WarningCode

        manager = self._manager(tmp_path, self._LOG)
        manager._check_estimates({"chat-0"})
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]
        manager._check_estimates(set())
        assert manager.health_warnings() == []

    def test_a_poll_of_a_loaded_engine_keeps_the_spill(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import WarningCode

        manager = self._manager(tmp_path, self._LOG)
        manager._check_estimates({"chat-0"})
        manager._check_estimates({"chat-0"})
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]

    def test_shutdown_clears_it(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet import swap_manager as sm

        monkeypatch.setattr(sm, "_stop_own_fleet", lambda *a, **k: [])
        manager = self._manager(tmp_path, self._LOG)
        manager._check_estimates({"chat-0"})
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]
        manager.shutdown()
        assert manager.health_warnings() == []

    def test_a_spill_in_the_memory_endpoint_is_a_health_warning(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet import swap_manager as sm
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        GIB = 1024**3

        class _Resp:
            status_code = 200

            def json(self) -> dict:
                return {
                    "data": [
                        {"name": "CUDA0", "model": 4 * GIB},
                        {"name": "Host", "model": 6 * GIB},
                    ]
                }

        class _Client:
            def get(self, url: str, timeout: float | None = None) -> _Resp:
                return _Resp()

        monkeypatch.setattr(sm, "_probe_client", lambda: _Client())
        manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server", MEMORY_FLAG],
            env_overrides={},
            model="chat-model",
            est_vram_bytes=4 * GIB,
            est_vram_by_device={"CUDA0": 4 * GIB},
            est_host_bytes=GIB,
        )
        enrol_launch(manager, launch)
        manager._member_port_by_model = {launch.model_id: 5901}
        manager._check_estimates({launch.model_id})
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]


class _Answer:
    status_code = 200

    def __init__(self, body: dict) -> None:
        self._body = body

    def json(self) -> dict:
        return self._body


class _FlakyEngine:
    """A probe client whose /running and /memory reads can each fail or answer."""

    def __init__(self) -> None:
        self.running: list[str] | None = ["chat-0"]
        self.memory: dict | None = None

    def get(self, url: str, timeout: float | None = None) -> _Answer:
        import httpx

        if url.endswith("/running"):
            if self.running is None:
                raise httpx.ConnectError("refused")
            return _Answer({"running": [{"model": m, "state": "ready"} for m in self.running]})
        if self.memory is None:
            raise httpx.ConnectError("refused")
        return _Answer(self.memory)


class TestFailedReadyProbe:
    """A failed read of /running says nothing about the engines, so no finding changes."""

    def _engine(self, monkeypatch) -> _FlakyEngine:
        from lilbee.providers.fleet import swap_manager as sm

        engine = _FlakyEngine()
        monkeypatch.setattr(sm, "_probe_client", lambda: engine)
        return engine

    def _manager(self, tmp_path: Path, argv: list[str]):
        from lilbee.providers.fleet import swap_manager as sm
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.readback import MIB, engine_log_path

        engine_log_path(tmp_path / "logs", "chat-0").parent.mkdir(parents=True, exist_ok=True)
        engine_log_path(tmp_path / "logs", "chat-0").write_text(
            "load_tensors:   CPU_Mapped model buffer size =  6000.00 MiB\n"
            "load_tensors:        CUDA0 model buffer size =  4000.00 MiB\n"
            "load_model: initializing slots\n",
            encoding="utf-8",
        )
        manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
        manager._log_path = tmp_path / "logs" / "llama-swap-chat.log"
        manager._port = 5900
        manager._ready_at = 1.0
        launch = InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=argv,
            env_overrides={},
            model="chat-model",
            est_vram_bytes=4000 * MIB,
            est_host_bytes=100 * MIB,
        )
        enrol_launch(manager, launch)
        manager._member_port_by_model = {launch.model_id: 5901}
        return manager

    def test_a_failed_probe_keeps_a_spill_and_a_divergence(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        engine = self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        assert manager.role_ready(WorkerRole.CHAT)
        divergence = HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off plan")
        manager._record_findings("chat-0", divergence, [])
        held = manager.health_warnings()
        assert {w.code for w in held} == {WarningCode.HOST_SPILL, WarningCode.PLACEMENT_DIVERGED}
        engine.running = None
        assert not manager.role_ready(WorkerRole.CHAT)
        assert manager.health_warnings() == held

    def test_health_warnings_read_the_report_with_no_readiness_poll_before_them(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        GIB = 1024**3
        engine = self._engine(monkeypatch)
        engine.memory = {
            "data": [{"name": "CUDA0", "model": 4 * GIB}, {"name": "Host", "model": 6 * GIB}]
        }
        reads: list[str] = []
        probe = engine.get

        def recording_get(url: str, timeout: float | None = None) -> _Answer:
            reads.append(url)
            return probe(url)

        monkeypatch.setattr(engine, "get", recording_get)
        manager = self._manager(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        assert reads == []
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]
        assert [url.rsplit("/", 1)[1] for url in reads] == ["running", "memory"]

    def test_a_log_engine_spill_shows_without_a_readiness_poll(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode

        self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]

    def test_an_empty_running_list_still_clears_both(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        engine = self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        manager.role_ready(WorkerRole.CHAT)
        manager._record_findings(
            "chat-0", HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off"), []
        )
        engine.running = []
        assert not manager.role_ready(WorkerRole.CHAT)
        assert manager.health_warnings() == []

    def test_a_memory_engine_keeps_its_spill_through_a_failed_probe_and_read(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        GIB = 1024**3
        engine = self._engine(monkeypatch)
        engine.memory = {
            "data": [{"name": "CUDA0", "model": 4 * GIB}, {"name": "Host", "model": 6 * GIB}]
        }
        manager = self._manager(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        manager.role_ready(WorkerRole.CHAT)
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]
        engine.running = None
        engine.memory = None
        assert not manager.role_ready(WorkerRole.CHAT)
        engine.running = ["chat-0"]
        assert manager.role_ready(WorkerRole.CHAT)
        assert [w.code for w in manager.health_warnings()] == [WarningCode.HOST_SPILL]
