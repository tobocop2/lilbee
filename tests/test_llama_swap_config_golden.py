"""The llama-swap config of the llama.cpp adapter equals the one origin/main writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.fleet import adapters
from lilbee.providers.fleet.engines.base import LaunchRequest
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from tests._llama_swap_scenario import BASE_PORT, BINARY, CONFIGS, THREADS, members

# The text origin/main writes for the same model set, one string per config.
GOLDEN = Path(__file__).parent / "llama_swap_config_main_golden.json"


def _launches() -> list[InstanceLaunch]:
    """The launches the adapter builds for the model set, each argv from the adapter."""
    launches = []
    for member in members():
        spec = LLAMA.role_spec(member.role, member.rerank_mode, member.meta)
        argv = LLAMA.launch_argv(spec, LaunchRequest(binary=BINARY, **member.argv))
        launches.append(
            InstanceLaunch(
                role=member.role,
                argv=argv,
                env_overrides=dict(member.env),
                model=member.argv["model_path"].name,
                weights_bytes=member.weights_bytes,
                replica=member.replica,
                rerank_mode=member.rerank_mode,
            )
        )
    return launches


@pytest.fixture(autouse=True)
def _fixed_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapters, "engine_thread_count", lambda: THREADS)


def test_the_model_set_covers_every_role_and_a_replica() -> None:
    launches = _launches()
    assert {launch.role.value for launch in launches} == {"chat", "embed", "rerank", "vision"}
    assert {launch.model_id for launch in launches} >= {"embed-0", "embed-1"}


def test_every_config_has_a_golden() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert CONFIGS
    assert set(golden) == set(CONFIGS)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_config_is_byte_identical_to_origin_main(name: str) -> None:
    launches = _launches()
    ports = {launch.model_id: BASE_PORT + i for i, launch in enumerate(launches)}
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert build_swap_config(launches, ports, **CONFIGS[name]) == golden[name]
