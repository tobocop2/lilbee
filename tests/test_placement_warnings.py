"""Placement divergence reaches /api/health warnings when the engine lands off plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole


class TestPlacementWarnings:
    def test_health_carries_divergence_from_a_swap(self, monkeypatch, tmp_path: Path) -> None:
        from lilbee.core.health_warnings import HealthWarning, WarningCode

        monkeypatch.setattr(cfg, "chunk_size", 512)
        monkeypatch.setattr(planning_mod, "planned_embed_token_cap", lambda _r: 2048)
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
        manager._launch_by_model = {"chat-0": launch}
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
        manager._launch_by_model = {launch.model_id: launch}
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
