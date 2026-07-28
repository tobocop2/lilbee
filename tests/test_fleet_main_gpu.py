"""The main_gpu setting has to reach the engine or stop being a setting."""

from __future__ import annotations

import logging
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.roles import WorkerRole


def _argv(**overrides) -> list[str]:
    kwargs = {
        "binary": Path("/bin/llama-server"),
        "spec": ROLE_SPECS[WorkerRole.CHAT],
        "model_path": Path("/m/c.gguf"),
        "devices": (0, 1),
        "n_gpu_layers": -1,
        "slots": 1,
        "ctx_per_slot": 4096,
    }
    kwargs.update(overrides)
    return build_server_argv(**kwargs)


class TestMainGpuReachesTheEngine:
    """It was surfaced in settings and over MCP and never emitted, so a user
    could set it, see it saved, and have nothing happen."""

    def test_it_is_emitted_for_a_multi_device_instance(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "main_gpu", 1)
        argv = _argv()
        assert argv[argv.index("--main-gpu") + 1] == "1"

    def test_it_is_omitted_when_unset(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "main_gpu", None)
        assert "--main-gpu" not in _argv()

    def test_it_is_omitted_for_a_single_device_instance(self, monkeypatch) -> None:
        # llama.cpp ignores it with one device; emitting it only invites confusion
        # about which index space the number lives in.
        monkeypatch.setattr(cfg, "main_gpu", 0)
        assert "--main-gpu" not in _argv(devices=(0,))

    def test_an_out_of_range_index_is_refused_loudly(self, monkeypatch, caplog) -> None:
        # It indexes this instance's own device list, not the host's cards. Silently
        # passing a number the engine will reject is how the knob stayed broken.
        monkeypatch.setattr(cfg, "main_gpu", 5)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.adapters"):
            argv = _argv(devices=(0, 1))
        assert "--main-gpu" not in argv
        assert "main_gpu" in caplog.text
