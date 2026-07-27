"""The self-check has to launch what the fleet would launch."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole


class TestTheSelfCheckAsksThePlanner:
    """Two InstanceLaunch construction sites that disagree make a green check
    meaningless: it can pass on a configuration serving never uses, and fail on
    one it would never have built.

    The disagreements were slots (one versus up to four), the context that
    follows from them, device pinning, and the tensor split.
    """

    @staticmethod
    def _capture(monkeypatch, tmp_path, role: WorkerRole):
        from lilbee.cli.commands import setup
        from lilbee.providers.fleet.devices import FleetDevice

        model = tmp_path / "m.gguf"
        model.write_bytes(b"x" * 1024)
        monkeypatch.setattr("tempfile.mkdtemp", lambda *a, **k: str(tmp_path / "wd"))
        (tmp_path / "wd").mkdir(exist_ok=True)
        monkeypatch.setattr(
            "lilbee.providers.fleet.binary.resolve_llama_server", lambda: Path("/bin/llama-server")
        )
        monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
        monkeypatch.setattr(
            "lilbee.providers.fleet.planning._plan_devices",
            lambda _b: [FleetDevice("CUDA", 0, "gpu", 24 * 1024**3, 24 * 1024**3)],
        )
        captured: dict[str, object] = {}

        class _Swap:
            def start(self, launches):
                captured["launch"] = launches[0]
                raise RuntimeError("stop here")

            def shutdown(self, *_a, **_k):
                pass

        monkeypatch.setattr(
            "lilbee.providers.fleet.swap_manager.SwapManager", lambda *a, **k: _Swap()
        )
        with pytest.raises(RuntimeError):
            setup._self_check_server(role, model)
        return captured["launch"]

    def test_chat_is_checked_at_the_slot_count_the_fleet_serves(
        self, monkeypatch, tmp_path
    ) -> None:
        # The check ran one slot while the fleet runs up to four, so --ctx-size
        # differed by that factor and the memory the check proved was not the
        # memory serving needs.
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        launch = self._capture(monkeypatch, tmp_path, WorkerRole.CHAT)
        argv = launch.argv
        slots = int(argv[argv.index("--parallel") + 1])
        assert slots == launch.slots
        assert argv[argv.index("--ctx-size") + 1] == str(launch.ctx * slots)

    def test_the_launch_carries_the_device_pin_the_fleet_would_use(
        self, monkeypatch, tmp_path
    ) -> None:
        # The check pinned nothing, so it could pass against a card the fleet
        # would never place this role on.
        monkeypatch.setattr(cfg, "num_ctx", 4096)
        launch = self._capture(monkeypatch, tmp_path, WorkerRole.CHAT)
        assert launch.env_overrides.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_an_embed_check_is_still_shaped_like_an_embed_launch(
        self, monkeypatch, tmp_path
    ) -> None:
        launch = self._capture(monkeypatch, tmp_path, WorkerRole.EMBED)
        assert "--embeddings" in launch.argv
        assert launch.role is WorkerRole.EMBED
