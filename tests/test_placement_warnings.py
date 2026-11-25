"""Placement divergence reaches /api/health warnings when the engine lands off plan."""

from __future__ import annotations

import threading
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
        manager._record_findings("chat-0", 0, divergence, [])
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


class _ProbedManager:
    """A chat swap manager with one planned replica, chat-0, and a fake engine to probe."""

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


class TestFailedReadyProbe(_ProbedManager):
    """A failed read of /running says nothing about the engines, so no finding changes."""

    def test_a_failed_probe_keeps_a_spill_and_a_divergence(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        engine = self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        assert manager.role_ready(WorkerRole.CHAT)
        divergence = HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off plan")
        manager._record_findings("chat-0", 0, divergence, manager.health_warnings())
        held = manager.health_warnings()
        assert {w.code for w in held} == {WarningCode.HOST_SPILL, WarningCode.PLACEMENT_DIVERGED}
        engine.running = None
        assert not manager.role_ready(WorkerRole.CHAT)
        assert manager.health_warnings() == held

    def test_an_empty_running_list_still_clears_both(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        engine = self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        manager.role_ready(WorkerRole.CHAT)
        manager._record_findings(
            "chat-0", 0, HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off"), []
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


_GIB = 1024**3
_SPILLED = {"data": [{"name": "CUDA0", "model": 4 * _GIB}, {"name": "Host", "model": 6 * _GIB}]}
_ON_CARD = {"data": [{"name": "CUDA0", "model": 4 * _GIB}]}


def _read_load(manager, model_ids: list[str]) -> None:
    """Claim the load of *model_ids* and run its read on this thread."""
    manager.claim_loaded(model_ids)()


class TestALoadTheCallerSaw(_ProbedManager):
    """A load that a warm or a lazy owner saw is read from the ids it names, with no poll."""

    def _memory_manager(self, tmp_path: Path, monkeypatch, report: dict | None):
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        engine = self._engine(monkeypatch)
        engine.memory = report
        return engine, self._manager(tmp_path, ["/bin/llama-server", MEMORY_FLAG])

    def _codes(self, manager) -> list:
        return [w.code for w in manager.health_warnings()]

    def test_a_load_is_read_with_an_empty_running_listing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.core.health_warnings import WarningCode

        engine, manager = self._memory_manager(tmp_path, monkeypatch, _SPILLED)
        engine.running = []
        _read_load(manager, ["chat-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]

    def test_a_log_engine_is_read_when_it_loads(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode

        self._engine(monkeypatch).running = []
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        _read_load(manager, ["chat-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]

    def test_a_rewarm_with_no_poll_between_is_read_again(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode

        engine, manager = self._memory_manager(tmp_path, monkeypatch, _SPILLED)
        _read_load(manager, ["chat-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]
        # The model reloaded with every layer on the card: a clean report clears the spill.
        engine.memory = _ON_CARD
        _read_load(manager, ["chat-0"])
        assert manager.health_warnings() == []
        engine.memory = _SPILLED
        _read_load(manager, ["chat-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]

    def test_a_failed_read_keeps_the_old_finding(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode

        engine, manager = self._memory_manager(tmp_path, monkeypatch, _SPILLED)
        _read_load(manager, ["chat-0"])
        engine.memory = None  # /memory does not answer
        _read_load(manager, ["chat-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]

    def test_a_poll_after_the_load_does_not_read_it_again(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        engine, manager = self._memory_manager(tmp_path, monkeypatch, _ON_CARD)
        reads: list[str] = []
        probe = engine.get

        def recording_get(url: str, timeout: float | None = None) -> _Answer:
            reads.append(url.rsplit("/", 1)[1])
            return probe(url)

        monkeypatch.setattr(engine, "get", recording_get)
        _read_load(manager, ["chat-0"])
        assert manager.role_ready(WorkerRole.CHAT)
        assert reads.count("memory") == 1

    def test_an_unknown_id_is_no_result(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode

        _engine, manager = self._memory_manager(tmp_path, monkeypatch, _SPILLED)
        _read_load(manager, ["chat-0"])
        _read_load(manager, ["ghost-0"])
        assert self._codes(manager) == [WarningCode.HOST_SPILL]

    def test_a_restarted_swap_reads_its_engines_again(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.core.health_warnings import WarningCode

        self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        assert manager.role_ready(WorkerRole.CHAT)
        manager._clear_placement_warnings()  # what a stop and a start do
        assert manager.role_ready(WorkerRole.CHAT)
        assert self._codes(manager) == [WarningCode.HOST_SPILL]


class TestALateRead(_ProbedManager):
    """A read that a poll's unload, a stop or a newer load passed records nothing."""

    def test_an_unload_during_a_log_read_leaves_no_finding(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.providers.fleet import swap_manager as sm

        engine = self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        real = sm.check_launch

        def unloading(*args, **kwargs):
            result = real(*args, **kwargs)
            engine.running = []
            assert not manager.role_ready(WorkerRole.CHAT)
            return result

        monkeypatch.setattr(sm, "check_launch", unloading)
        _read_load(manager, ["chat-0"])
        assert manager.health_warnings() == []

    def test_an_unload_during_a_memory_read_leaves_no_finding(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        engine = self._engine(monkeypatch)
        engine.memory = _SPILLED
        manager = self._manager(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        real = engine.get

        def unloading(url: str, timeout: float | None = None) -> _Answer:
            answer = real(url)
            if url.endswith("/memory"):
                engine.running = []
                assert not manager.role_ready(WorkerRole.CHAT)
            return answer

        monkeypatch.setattr(engine, "get", unloading)
        _read_load(manager, ["chat-0"])
        assert manager.health_warnings() == []

    def test_a_stop_during_a_read_leaves_no_finding(self, tmp_path: Path, monkeypatch) -> None:
        from lilbee.providers.fleet import swap_manager as sm

        self._engine(monkeypatch)
        manager = self._manager(tmp_path, ["/bin/llama-server"])
        real = sm.check_launch

        def stopping(*args, **kwargs):
            result = real(*args, **kwargs)
            manager._clear_placement_warnings()
            return result

        monkeypatch.setattr(sm, "check_launch", stopping)
        _read_load(manager, ["chat-0"])
        assert manager.health_warnings() == []

    def test_a_poll_drops_a_finding_that_is_not_in_its_read_set(self, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        manager = self._manager(tmp_path, ["/bin/llama-server"])
        divergence = HealthWarning(code=WarningCode.PLACEMENT_DIVERGED, message="off plan")
        manager._record_findings("chat-0", 0, divergence, [divergence])
        manager._check_estimates(set())
        assert manager.health_warnings() == []

    def test_an_older_read_that_ends_last_does_not_replace_a_newer_one(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from lilbee.providers.fleet.readback import MEMORY_FLAG

        engine = self._engine(monkeypatch)
        engine.memory = _SPILLED
        manager = self._manager(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        real = engine.get
        reloaded = threading.Event()

        def reloading(url: str, timeout: float | None = None) -> _Answer:
            answer = real(url)
            if url.endswith("/memory") and not reloaded.is_set():
                reloaded.set()
                engine.memory = _ON_CARD  # the engine reloads on the card, read B
                _read_load(manager, ["chat-0"])
            return answer

        monkeypatch.setattr(engine, "get", reloading)
        _read_load(manager, ["chat-0"])  # read A, which got the spilled report
        assert manager.health_warnings() == []
        for _ in range(3):
            assert manager.role_ready(WorkerRole.CHAT)
        assert manager.health_warnings() == []
