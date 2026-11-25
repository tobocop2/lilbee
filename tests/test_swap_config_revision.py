"""The llama-swap config changes only together with SWAP_CONFIG_REVISION.

A peer binds an engine only when its pin, which holds the revision, matches. A change to what
the writer renders without a new revision lets a peer adopt an engine launched under the old
config. The snapshot holds the full config for a fixed launch set, with the
revision it was recorded at. To update it, raise SWAP_CONFIG_REVISION in swap_config.py, then run

    UPDATE_SWAP_CONFIG_SNAPSHOT=1 uv run pytest tests/test_swap_config_revision.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from lilbee.providers.fleet.engines.choice import discover_adapters
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import SWAP_CONFIG_REVISION, build_swap_config
from lilbee.providers.roles import WorkerRole

SNAPSHOT = Path(__file__).parent / "swap_config_revision_snapshot.json"
_UPDATE_VAR = "UPDATE_SWAP_CONFIG_SNAPSHOT"
_BASE_PORT = 5900
_LOG_DIR = PurePosixPath("/engine/logs")


def _launches() -> list[InstanceLaunch]:
    """Two launches with env and residency classes, then one per engine adapter.

    Each adapter launch carries that adapter's swap_extras, so a key or value an engine
    adds to its model entry is in the snapshot.
    """
    chat = InstanceLaunch(
        role=WorkerRole.CHAT,
        replica=0,
        argv=["/engine/llama-server", "--model", "/m/chat.gguf"],
        env_overrides={"CUDA_VISIBLE_DEVICES": "0,1"},
        model="chat",
        swap_class="chat",
    )
    embed = InstanceLaunch(
        role=WorkerRole.EMBED,
        replica=0,
        argv=["/engine/llama-server", "--embeddings", "--memory"],
        env_overrides={},
        model="embed",
        swap_class="embed",
    )
    launches = [chat, embed]
    for name, adapter in discover_adapters().items():
        launches.append(
            InstanceLaunch(
                role=WorkerRole.RERANK,
                replica=len(launches),
                argv=[f"/engine/{name}-server"],
                env_overrides={},
                model=f"{name}-model",
                swap_extras=dict(adapter.swap_extras(chat)),
            )
        )
    return launches


def _rendered() -> dict[str, Any]:
    """The parsed config for each way the writer is driven."""
    launches = _launches()
    ports = {launch.model_id: _BASE_PORT + i for i, launch in enumerate(launches)}
    all_logs: dict[str, Any] = {
        launch.model_id: _LOG_DIR / f"{launch.model_id}.log" for launch in launches
    }
    some_logs = {launches[0].model_id: all_logs[launches[0].model_id]}
    runs: dict[str, dict[str, Any]] = {
        "resident-no-logs": {"engine_logs": {}},
        "resident-logs": {"engine_logs": all_logs},
        "swap-ttl-some-logs": {"engine_logs": some_logs, "swap": True, "ttl_seconds": 300},
        "older-router": {"engine_logs": {}, "router_reads_streams": False},
    }
    return {
        name: json.loads(build_swap_config(launches, ports, **kwargs))
        for name, kwargs in runs.items()
    }


def _recorded() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_swap_config_changes_only_with_a_revision_bump() -> None:
    rendered = _rendered()
    if os.environ.get(_UPDATE_VAR):
        record = {"revision": SWAP_CONFIG_REVISION, "configs": rendered}
        SNAPSHOT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return
    recorded = _recorded()
    same_text = rendered == recorded["configs"]
    same_revision = recorded["revision"] == SWAP_CONFIG_REVISION
    assert same_text or not same_revision, (
        "The config build_swap_config writes changed, and SWAP_CONFIG_REVISION is still "
        f"{SWAP_CONFIG_REVISION}. Raise it in swap_config.py, then run "
        f"{_UPDATE_VAR}=1 uv run pytest tests/test_swap_config_revision.py"
    )
    assert same_revision or not same_text, (
        f"SWAP_CONFIG_REVISION is {SWAP_CONFIG_REVISION} and the snapshot was recorded at "
        f"{recorded['revision']}, but the config is the same. Undo the bump, or change "
        "what the writer renders."
    )
    assert same_text, (
        "The snapshot is out of date. Run "
        f"{_UPDATE_VAR}=1 uv run pytest tests/test_swap_config_revision.py"
    )
