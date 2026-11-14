"""The launch argv of every shipped adapter for one fixed request, as a JSON-ready mapping."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.fleet import adapters
from lilbee.providers.fleet.engines import choice, lemonade, openvino
from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.host_caps import collect

BINARY = PurePosixPath("/engines/bin/server")
MODEL = PurePosixPath("/models/model.gguf")
THREADS = 8
HELP = "  --device <dev>  devices\n  -ts, --tensor-split N0,N1  split\n"


def launch_argvs(
    monkeypatch: pytest.MonkeyPatch, models_dir: Path
) -> dict[str, dict[str, list[str]]]:
    """Engine name -> role value -> argv, with the host inputs of the argv pinned."""
    monkeypatch.setattr(adapters, "engine_thread_count", lambda: THREADS)
    monkeypatch.setattr(adapters, "engine_help", lambda _binary: HELP)
    both = staticmethod(lambda _binary: frozenset({"CPU", "GPU"}))
    monkeypatch.setattr(openvino.OpenvinoAdapter, "_listed", both)
    monkeypatch.setattr(cfg, "models_dir", models_dir)
    monkeypatch.setattr(
        runnable,
        "session_host",
        lambda: collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None),
    )
    monkeypatch.setattr(lemonade, "resolve_llama_server", lambda: Path("/engines/llama-server"))
    request = LaunchRequest(
        binary=Path(BINARY),
        model_path=Path(MODEL),
        devices=(0, 1),
        n_gpu_layers=99,
        slots=2,
        ctx_per_slot=4096,
        device_names=("CUDA0", "CUDA1"),
        tensor_split=(1, 1),
    )
    found: dict[str, dict[str, list[str]]] = {}
    for name, adapter in choice.ADAPTERS.items():
        roles = sorted(adapter.served_roles, key=lambda role: role.value)
        found[name.value] = {
            role.value: adapter.launch_argv(adapter.role_spec(role, None, None), request)
            for role in roles
        }
    return found
