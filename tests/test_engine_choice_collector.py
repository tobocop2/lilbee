"""The collector that gives each role its engine, and the installed mapping behind it."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app.engine_choice import host_facts, resolve_engines
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import Config
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    EngineAssignment,
    HostFacts,
    NoEngineError,
    choose_engine,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole
from tests.conformance.fake_adapter import FakeAdapter


def test_installed_mapping_holds_the_formats_each_adapter_loads() -> None:
    installed = installed_engines()
    assert installed
    assert set(installed) == set(ADAPTERS)
    for name, adapter in ADAPTERS.items():
        assert installed[name] == adapter.formats
    assert installed[EngineName.LLAMA] == frozenset({ModelFormat.GGUF})


def test_every_role_runs_on_the_default_engine_with_no_setting() -> None:
    assignment = resolve_engines(Config())
    assert set(assignment.choices) == set(ROLE_REGISTRY)
    for role in ROLE_REGISTRY:
        engine = assignment.for_role(role)
        assert engine.name is EngineName.LLAMA
        assert engine.adapter is LLAMA
        assert engine.notice is None


def test_launch_through_the_collector_equals_the_adapter_launch() -> None:
    request = LaunchRequest(
        binary=Path("/b/llama-server"),
        model_path=Path("/m/x.gguf"),
        devices=(0,),
        n_gpu_layers=99,
        slots=2,
        ctx_per_slot=4096,
    )
    chosen = resolve_engines(Config()).for_role(WorkerRole.CHAT).adapter
    spec = chosen.role_spec(WorkerRole.CHAT, None, None)
    assert chosen.launch_argv(spec, request) == LLAMA.launch_argv(
        LLAMA.role_spec(WorkerRole.CHAT, None, None), request
    )


def test_two_roles_hold_two_engines() -> None:
    assignment = EngineAssignment.for_test({"chat": "llama.cpp", "embed": "mlx"})
    assert assignment.for_role(WorkerRole.CHAT).name is EngineName.LLAMA
    assert assignment.for_role(WorkerRole.EMBED).name is EngineName.MLX


def test_a_role_with_no_engine_raises_and_the_others_still_answer() -> None:
    config = Config()
    config.embedding_model = "org/embedder#mlx"
    assignment = resolve_engines(config)
    with pytest.raises(NoEngineError, match="For embed: No installed engine loads the mlx format"):
        assignment.for_role(WorkerRole.EMBED)
    assert assignment.for_role(WorkerRole.CHAT).name is EngineName.LLAMA


def test_the_notice_names_the_role_as_an_input_of_the_text() -> None:
    kwargs = dict(
        fmt=ModelFormat.GGUF,
        host=HostFacts(system="linux", machine="x86_64"),
        installed={EngineName.LLAMA: frozenset({ModelFormat.GGUF})},
        preferred="mlx",
        override=None,
    )
    with_role = choose_engine(**kwargs, role=WorkerRole.CHAT)
    assert with_role.notice == (
        "Engine mlx for chat is not installed. Install mlx to use it. Using llama.cpp."
    )
    assert choose_engine(**kwargs).notice == (
        "Engine mlx is not installed. Install mlx to use it. Using llama.cpp."
    )
    none = choose_engine(**{**kwargs, "fmt": ModelFormat.MLX, "preferred": "auto"})
    assert none.notice == (
        "No installed engine loads the mlx format. Install an engine that loads mlx."
    )
    named = choose_engine(
        **{**kwargs, "fmt": ModelFormat.MLX, "preferred": "auto"}, role=WorkerRole.EMBED
    )
    assert named.notice == f"For embed: {none.notice}"


def test_host_facts_are_lower_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "ARM64")
    facts = host_facts()
    assert (facts.system, facts.machine) == ("darwin", "arm64")


def test_placement_plans_with_the_adapter_the_chat_role_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.app import placement
    from lilbee.providers.fleet.engines import choice
    from lilbee.providers.fleet.placement_spec import PlacementError

    class _Marked(FakeAdapter):
        name = "llama.cpp"
        formats = frozenset({"gguf"})

    marked = _Marked()
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, marked)  # type: ignore[arg-type]
    seen: list[object] = []

    def _resolve(spec: object, *, engine: object, **kwargs: object) -> None:
        seen.append(engine)
        raise PlacementError("stop")

    monkeypatch.setattr(placement, "resolve_placement_plan", _resolve)
    with pytest.raises(PlacementError, match="stop"):
        placement.preview_placement()
    assert seen == [marked]
